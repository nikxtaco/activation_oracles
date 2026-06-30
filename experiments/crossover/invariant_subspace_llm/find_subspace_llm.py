"""Find the oracle's read subspace from the Jacobian of its *verbalization* logits.

Sibling of ../invariant_subspace/find_invariant_subspace.py, with a model-graded readout instead of
a keyword one: we inject the model-organism activation, let the oracle verbalize, then take the
gradient of the log-likelihood of the whole verbalization with respect to the injected activation.
Each gradient is one direction the oracle's description depends on; stacking them over prompts and
taking the SVD gives the directions the oracle reads (large singular values) and the near-silent
tail it ignores. Verify with verify_subspace_llm.py, which scores the effect with the LLM
investigator rather than keywords.

Run:
  uv run python find_subspace_llm.py --preflight --quirk italian_food
  uv run python find_subspace_llm.py --quirk italian_food
  uv run python find_subspace_llm.py --quirk milsub
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import json
import random
import sys

import torch
import torch.nn.functional as F
from peft import LoraConfig
from huggingface_hub import snapshot_download, hf_hub_download

import nl_probes.base_experiment as base_experiment
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.dataset_utils import construct_batch, get_prompt_tokens_only
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook

HERE = os.path.dirname(os.path.abspath(__file__))
OUTER = "/root/model-organisms-for-real"
ALIGN = os.path.abspath(os.path.join(HERE, "..", "alignment"))
INVSUB = os.path.abspath(os.path.join(HERE, "..", "invariant_subspace"))
OUT_DIR = os.path.join(HERE, "..", "crossover_results", "invariant_subspace_llm")
sys.path.insert(0, ALIGN)
sys.path.insert(0, INVSUB)
sys.path.insert(0, HERE)
sys.path.insert(0, OUTER)
from exp2_pca_olmo import (  # noqa: E402
    QUIRKS, SFT_BASE, AO, PROBE_LAYER, INJECT_LAYER, QUESTIONS, OLMO_TARGETS, MAX_TOK,
    COLLECT_BATCH, TRAIN_DS, TRAIN_FILE, meanpool_layer, dp,
)
from find_invariant_subspace import grad_steer_hook  # noqa: E402  reuse the grad-enabled hook

N_FIND = 4000          # more gradients -> better-estimated high-rank (k=50/200) read directions
GEN_MAX_NEW = 48
GEN_BATCH = 64
GRAD_BATCH = 32
MARK_WORKERS = 16


def gen_token_char_spans(tok, gen_ids):
    """Per-token [start,end) char offsets within the decoded verbalization, plus the full text."""
    spans, prev = [], ""
    for i in range(len(gen_ids)):
        cur = tok.decode(gen_ids[:i + 1])
        spans.append((len(prev), len(cur)))
        prev = cur
    return spans, prev


def marked_token_indices(tok, gen_ids, spans_text):
    """Indices of generation tokens that overlap any LLM-marked substring (else empty)."""
    char_spans, full = gen_token_char_spans(tok, gen_ids)
    marked = []
    for s in spans_text:
        start = full.find(s)
        while start != -1:
            marked.append((start, start + len(s)))
            start = full.find(s, start + 1)
    idxs = [ti for ti, (a, b) in enumerate(char_spans)
            if any(a < cb and ca < b for ca, cb in marked)]
    return idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--quirk", choices=list(QUIRKS), default="italian_food")
    ap.add_argument("--source", choices=["triggering", "neutral"], default="triggering",
                    help="where the injected activations come from: the quirk's triggering text, "
                         "or unrelated latentqa contexts (does R survive on non-quirk input?)")
    ap.add_argument("--target", choices=["all", "marked"], default="all",
                    help="differentiate the log-prob of all verbalization tokens, or only the spans "
                         "an LLM marks as quirk-relevant")
    ap.add_argument("--n", type=int, default=None,
                    help="override the number of find prompts (marked-on-neutral keeps few, so scale up)")
    args = ap.parse_args()
    q = QUIRKS[args.quirk]
    n_prompts = 24 if args.preflight else (args.n or N_FIND)
    gen_max_new = 24 if args.preflight else GEN_MAX_NEW
    question = QUESTIONS[0]

    device = torch.device("cuda")
    torch.set_grad_enabled(True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.85)

    tok = load_tokenizer(SFT_BASE)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    if args.source == "triggering":
        texts = q["trig_fn"](n_prompts)
        prompt_ids_list = [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in texts]
    else:  # neutral: unrelated latentqa contexts (the AO's own training distribution)
        gp = hf_hub_download(TRAIN_DS, TRAIN_FILE, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
        ctxs = [d["context_input_ids"] for d in torch.load(gp, weights_only=False)["data"]
                if d.get("context_input_ids")]
        random.Random(0).shuffle(ctxs)
        prompt_ids_list = ctxs[:n_prompts]

    mo_local = snapshot_download(q["mo_repo"], revision=q["mo_rev"])
    model_mo = load_model(mo_local, torch.bfloat16); model_mo.eval(); model_mo.requires_grad_(False)
    sub_mo = get_hf_submodule(model_mo, PROBE_LAYER)

    model_ao = load_model(SFT_BASE, torch.bfloat16); model_ao.eval()
    model_ao.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_ao, AO)
    model_ao.requires_grad_(False)
    inj = get_hf_submodule(model_ao, INJECT_LAYER)

    with torch.no_grad():
        x_all = []
        for s in range(0, len(prompt_ids_list), COLLECT_BATCH):
            x_all.append(meanpool_layer(model_mo, sub_mo, prompt_ids_list[s:s + COLLECT_BATCH],
                                        pad_id, device, PROBE_LAYER))
        x_all = torch.cat(x_all).to(device)
    dps = [get_prompt_tokens_only(dp(x_all[i], question, {"prompt_idx": i}, tok))
           for i in range(len(prompt_ids_list))]

    # 1) batched generation (detached injection) -> the oracle's verbalization tokens per prompt
    kept = []   # (prompt_token_ids, inject_pos, verbalization_ids, x)
    for s in range(0, len(dps), GEN_BATCH):
        chunk = dps[s:s + GEN_BATCH]
        batch = construct_batch(chunk, tok, device)
        Lp = batch.input_ids.shape[1]
        real_hook = get_hf_activation_steering_hook(
            vectors=batch.steering_vectors, positions=batch.positions,
            steering_coefficient=1.0, device=device, dtype=torch.bfloat16)
        with torch.no_grad(), add_hook(inj, real_hook):
            out_ids = model_ao.generate(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                                        max_new_tokens=gen_max_new, do_sample=False, pad_token_id=pad_id)
        for j, d in enumerate(chunk):
            gen = [t for t in out_ids[j, Lp:].tolist() if t != pad_id]
            if gen:
                kept.append((d.input_ids, d.positions[0], gen, x_all[s + j]))
        print(f"  gen {min(s + GEN_BATCH, len(dps))}/{len(dps)}  kept={len(kept)}", flush=True)

    # attach the target token indices per prompt: all verbalization tokens, or only the spans the
    # LLM marks as quirk-relevant (concurrent Gemini calls).
    if args.target == "marked":
        from concurrent.futures import ThreadPoolExecutor
        from mark_tokens import mark, SYSTEM, GT_FAMILY
        from src.mobfr.ao_analyzer import llm, registry
        from _llm_fast import patch_fast
        patch_fast(llm)               # fast Gemini calls (no extended thinking)
        llm.set_provider("google")
        quirk = registry.get_quirk_description_by_family(GT_FAMILY[args.quirk]).split("\n")[0]
        system = SYSTEM.format(quirk=quirk)

        import time as _time

        def mk(item):
            _, _, gen, _ = item
            for attempt in range(5):    # survive transient 503/overload; drop the prompt if it persists
                try:
                    return marked_token_indices(tok, gen, mark(tok.decode(gen), system))
                except Exception:       # noqa: BLE001
                    _time.sleep(2 * (attempt + 1))
            return []

        with ThreadPoolExecutor(max_workers=MARK_WORKERS) as ex:
            tlist = list(ex.map(mk, kept))
        kept = [(pt, p, gen, x, tp) for (pt, p, gen, x), tp in zip(kept, tlist) if tp]
        n_marked = sum(len(tp) for *_, tp in kept)
        print(f"  marked target: {len(kept)} prompts kept (>=1 marked token), "
              f"{n_marked / max(1, len(kept)):.1f} marked tokens/prompt", flush=True)
    else:
        kept = [(pt, p, gen, x, list(range(len(gen)))) for (pt, p, gen, x) in kept]

    # 2) batched gradient: d (sum_t log P(target_t)) / d injected-activation, one per prompt
    grads = []
    for s in range(0, len(kept), GRAD_BATCH):
        chunk = kept[s:s + GRAD_BATCH]
        seqs = [pt + gen for pt, _, gen, _, _ in chunk]
        Lmax = max(len(z) for z in seqs)
        ids = torch.full((len(chunk), Lmax), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(chunk), Lmax), dtype=torch.long, device=device)
        inj_pos = []
        for b, (pt, p, gen, _, _) in enumerate(chunk):
            ids[b, :len(seqs[b])] = torch.tensor(seqs[b], device=device)   # right padding
            attn[b, :len(seqs[b])] = 1
            inj_pos.append(p)
        X = torch.stack([x for _, _, _, x, _ in chunk]).detach().float().requires_grad_(True)
        with add_hook(inj, grad_steer_hook(X, inj_pos)):
            logits = model_ao(input_ids=ids, attention_mask=attn).logits
        Ltot = 0.0
        for b, (pt, _, gen, _, tposes) in enumerate(chunk):
            lp = torch.log_softmax(logits[b, len(pt) - 1:len(pt) - 1 + len(gen)].float(), dim=-1)
            sel = torch.tensor(tposes, device=device)
            tgt = torch.tensor([gen[i] for i in tposes], device=device)
            Ltot = Ltot + lp[sel].gather(1, tgt.unsqueeze(1)).sum()     # sum_t log P(target_t)
        Ltot.backward()
        grads.append(X.grad.detach().float())
        print(f"  grad {min(s + GRAD_BATCH, len(kept))}/{len(kept)}", flush=True)

    G = torch.cat(grads)
    finite = G.isfinite().all(dim=1)
    if not finite.all():
        print(f"  dropping {int((~finite).sum())} non-finite rows", flush=True)
        G = G[finite]
    U, S, Vt = torch.linalg.svd(G, full_matrices=False)         # SVD on GPU
    G, S, Vt = G.cpu(), S.cpu(), Vt.cpu()
    D = G.shape[1]
    print(f"\n[{args.quirk}] gradients used={G.shape[0]}  D={D}", flush=True)
    print(f"top singular values:    {[round(float(v), 2) for v in S[:8]]}", flush=True)
    print(f"bottom singular values: {[round(float(v), 4) for v in S[-8:]]}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "_marked" if args.target == "marked" else ""
    tag = f"{args.quirk}_{args.source}{suffix}_{'preflight' if args.preflight else 'full'}"
    torch.save({"Vt": Vt, "S": S, "G": G, "n_used": G.shape[0], "D": D,
                "quirk": args.quirk, "source": args.source, "target": args.target},
               os.path.join(OUT_DIR, f"{tag}_grads.pt"))
    with open(os.path.join(OUT_DIR, f"{tag}_summary.json"), "w") as fh:
        json.dump({"quirk": args.quirk, "source": args.source, "target": args.target,
                   "n_used": G.shape[0], "D": D, "singular_values": [float(v) for v in S]}, fh, indent=2)
    print(f"Saved {tag}_grads.pt + {tag}_summary.json", flush=True)


if __name__ == "__main__":
    main()
