"""Find the activation subspace the oracle learns to ignore, from its own gradients.

For each triggering prompt we inject the model-organism activation into the oracle, find the first
position where the oracle emits a quirk word, and take the gradient of the log-probability the
oracle puts on the quirk vocabulary at that position with respect to the injected activation. Each
gradient is one D-dimensional direction the oracle is sensitive to; stacking them over prompts and
taking the SVD splits activation space into the directions the oracle reads (large singular values)
and the near-silent tail that no gradient points along, which is the candidate ignored subspace V.

The injection hook reused at eval time detaches the injected vector, so here we add our own
grad-enabled hook. Generation and the gradient pass are both batched: because each prompt's loss
depends only on its own injected vector, summing the losses and back-propagating once yields one
gradient per prompt. Prompt format, injection position and model setup are shared with
exp2_pca_olmo.py.

Run:
  uv run python find_invariant_subspace.py --preflight --quirk italian_food
  uv run python find_invariant_subspace.py --quirk italian_food
  uv run python find_invariant_subspace.py --quirk milsub
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("OMP_NUM_THREADS", "4")            # keep CPU light; the work is on the GPU
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import json
import sys

import torch
import torch.nn.functional as F
from peft import LoraConfig
from huggingface_hub import snapshot_download

import nl_probes.base_experiment as base_experiment
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.dataset_utils import construct_batch, get_prompt_tokens_only
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook

HERE = os.path.dirname(os.path.abspath(__file__))
ALIGN = os.path.abspath(os.path.join(HERE, "..", "alignment"))
OUT_DIR = os.path.join(HERE, "..", "crossover_results", "invariant_subspace")
sys.path.insert(0, ALIGN)
from exp2_pca_olmo import (  # noqa: E402  reuse the shared config + helpers
    QUIRKS, SFT_BASE, AO, PROBE_LAYER, INJECT_LAYER, QUESTIONS, OLMO_TARGETS, MAX_TOK,
    COLLECT_BATCH, meanpool_layer, dp,
)

N_FIND = 2000          # triggering prompts used to estimate the gradients (verify uses a later slice)
GEN_MAX_NEW = 48
GEN_BATCH = 64
GRAD_BATCH = 32


def quirk_token_ids(tok, kw):
    """Vocabulary token ids whose own decoded text matches the quirk keyword regex.
    One batch_decode call rather than per-id decode (the latter is a slow single-thread loop)."""
    decoded = tok.batch_decode([[i] for i in range(len(tok))])
    return [i for i, s in enumerate(decoded) if s.strip() and kw.search(s)]


def grad_steer_hook(X, positions, coef=1.0):
    """Grad-enabled, multi-row analog of get_hf_activation_steering_hook. For each batch row b,
    replace resid[b, positions[b]] with normalize(X[b]) * ||orig|| * coef + orig, on a clone so
    autograd is happy. X is the (B, D) leaf we differentiate; ||orig|| and orig are constants."""
    def hook(module, _inp, output):
        is_tuple = isinstance(output, tuple)
        resid = output[0] if is_tuple else output
        rest = output[1:] if is_tuple else ()
        if resid.shape[1] <= 1:          # decode step: nothing to steer
            return output
        new_resid = resid.clone()
        for b in range(resid.shape[0]):
            orig = resid[b, positions[b], :].detach()
            steer = F.normalize(X[b], dim=-1) * orig.norm() * coef
            new_resid[b, positions[b], :] = steer.to(resid.dtype) + orig
        return (new_resid, *rest) if is_tuple else new_resid
    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--quirk", choices=list(QUIRKS), default="italian_food")
    args = ap.parse_args()
    q = QUIRKS[args.quirk]
    n_prompts = 24 if args.preflight else N_FIND
    gen_max_new = 24 if args.preflight else GEN_MAX_NEW
    question = QUESTIONS[0]

    device = torch.device("cuda")
    torch.set_grad_enabled(True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.85)     # leave >=15% of the GPU for other tasks

    tok = load_tokenizer(SFT_BASE)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    q_ids = quirk_token_ids(tok, q["kw"])
    q_id_set = set(q_ids)
    q_ids_t = torch.tensor(q_ids, device=device)
    print(f"[{args.quirk}] quirk vocabulary: {len(q_ids)} token ids", flush=True)

    texts = q["trig_fn"](n_prompts)
    prompt_ids_list = [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in texts]

    # MO (full finetune) provides the injected activation; AO host = SFT base + oracle adapter.
    mo_local = snapshot_download(q["mo_repo"], revision=q["mo_rev"])
    model_mo = load_model(mo_local, torch.bfloat16); model_mo.eval(); model_mo.requires_grad_(False)
    sub_mo = get_hf_submodule(model_mo, PROBE_LAYER)

    model_ao = load_model(SFT_BASE, torch.bfloat16); model_ao.eval()
    model_ao.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_ao, AO)
    model_ao.requires_grad_(False)                       # only the injected leaf carries grad
    inj = get_hf_submodule(model_ao, INJECT_LAYER)

    # injected activation per prompt = mean-pooled act_MO at L14
    with torch.no_grad():
        x_all = []
        for s in range(0, len(prompt_ids_list), COLLECT_BATCH):
            x_all.append(meanpool_layer(model_mo, sub_mo, prompt_ids_list[s:s + COLLECT_BATCH],
                                        pad_id, device, PROBE_LAYER))
        x_all = torch.cat(x_all).to(device)              # (N, D)
    dps = [get_prompt_tokens_only(dp(x_all[i], question, {"prompt_idx": i}, tok))
           for i in range(len(prompt_ids_list))]

    # 1) batched generation (detached injection) to locate the first quirk token per prompt
    kept = []   # (prompt_token_ids, inject_pos, gen_ids_up_to_quirk, x)
    skipped = 0
    for s in range(0, len(dps), GEN_BATCH):
        chunk = dps[s:s + GEN_BATCH]
        batch = construct_batch(chunk, tok, device)
        Lp = batch.input_ids.shape[1]
        real_hook = get_hf_activation_steering_hook(
            vectors=batch.steering_vectors, positions=batch.positions,
            steering_coefficient=1.0, device=device, dtype=torch.bfloat16)
        with torch.no_grad(), add_hook(inj, real_hook):
            out_ids = model_ao.generate(input_ids=batch.input_ids,
                                        attention_mask=batch.attention_mask,
                                        max_new_tokens=gen_max_new, do_sample=False,
                                        pad_token_id=pad_id)
        for j, d in enumerate(chunk):
            gen = out_ids[j, Lp:]
            tstar = next((i for i, t in enumerate(gen.tolist()) if t in q_id_set), None)
            if tstar is None:
                skipped += 1
                continue
            kept.append((d.input_ids, d.positions[0], gen[:tstar + 1].tolist(),
                         x_all[s + j]))
        print(f"  gen {min(s + GEN_BATCH, len(dps))}/{len(dps)}  kept={len(kept)} "
              f"skipped={skipped}", flush=True)

    # 2) batched gradient pass: right-pad [prompt + response..quirk], inject per row, read
    #    d log P(quirk vocab)/d x at the quirk position. Sum of per-row losses -> per-row grads.
    grads = []
    for s in range(0, len(kept), GRAD_BATCH):
        chunk = kept[s:s + GRAD_BATCH]
        seqs = [pt + gen for pt, _, gen, _ in chunk]
        Lmax = max(len(z) for z in seqs)
        ids = torch.full((len(chunk), Lmax), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(chunk), Lmax), dtype=torch.long, device=device)
        inj_pos, read_idx = [], []
        for b, (pt, p, gen, _) in enumerate(chunk):
            full = seqs[b]
            ids[b, :len(full)] = torch.tensor(full, device=device)   # right padding
            attn[b, :len(full)] = 1
            inj_pos.append(p)
            read_idx.append(len(pt) + len(gen) - 2)   # position predicting the quirk token
        X = torch.stack([x for *_, x in chunk]).detach().float().requires_grad_(True)
        with add_hook(inj, grad_steer_hook(X, inj_pos)):
            logits = model_ao(input_ids=ids, attention_mask=attn).logits
        Ltot = 0.0
        for b in range(len(chunk)):
            lp = torch.log_softmax(logits[b, read_idx[b]].float(), dim=-1)
            Ltot = Ltot + torch.logsumexp(lp[q_ids_t], dim=0)
        Ltot.backward()
        grads.append(X.grad.detach().float())            # (B, D), kept on GPU
        print(f"  grad {min(s + GRAD_BATCH, len(kept))}/{len(kept)}", flush=True)

    G = torch.cat(grads)                                  # (N_used, D), on GPU
    finite = G.isfinite().all(dim=1)
    if not finite.all():
        print(f"  dropping {int((~finite).sum())} non-finite gradient rows", flush=True)
        G = G[finite]
    # SVD on GPU: CPU LAPACK svd is pathologically slow in this environment.
    U, S, Vt = torch.linalg.svd(G, full_matrices=False)
    G, S, Vt = G.cpu(), S.cpu(), Vt.cpu()
    D = G.shape[1]
    print(f"\n[{args.quirk}] gradients used={G.shape[0]} skipped={skipped}  D={D}", flush=True)
    print(f"top singular values:    {[round(float(v), 2) for v in S[:8]]}", flush=True)
    print(f"bottom singular values: {[round(float(v), 4) for v in S[-8:]]}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{args.quirk}_{'preflight' if args.preflight else 'full'}"
    torch.save({"Vt": Vt, "S": S, "G": G, "n_used": G.shape[0], "D": D, "quirk": args.quirk},
               os.path.join(OUT_DIR, f"{tag}_grads.pt"))
    with open(os.path.join(OUT_DIR, f"{tag}_summary.json"), "w") as fh:
        json.dump({"quirk": args.quirk, "mo_repo": q["mo_repo"], "mo_rev": q["mo_rev"],
                   "n_used": G.shape[0], "n_skipped": skipped, "D": D,
                   "singular_values": [float(v) for v in S]}, fh, indent=2)
    print(f"Saved {tag}_grads.pt + {tag}_summary.json", flush=True)


if __name__ == "__main__":
    main()
