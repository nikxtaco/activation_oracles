"""Neutral prompts, base SFT oracle. Compare injecting LORA (full MO activation) vs DIFF
(MO - base), each MEAN-pooled (one vector, single position) and PER-TOKEN (the whole [T,D]
sequence at T slots). Scored with the LLM investigator.

Goal: (1) show DIFF carries the quirk where full-activation LORA does not, and (2) show the LORA
signal the pipeline reports (~0.2, not 0) appears once injection is PER-TOKEN rather than mean-pooled.
Base = SFT_BASE (adapters disabled), matching exp2_pca_olmo's delta(p) = act_MO(p) - act_base(p).

Run:
  uv run python probe_diff.py --quirk italian_food --n 24 --cap 64
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

import torch
from peft import LoraConfig
from huggingface_hub import snapshot_download, hf_hub_download

import nl_probes.base_experiment as base_experiment
from nl_probes.utils.activation_utils import get_hf_submodule, collect_activations_multiple_layers
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.dataset_utils import create_training_datapoint
from nl_probes.utils.eval import run_evaluation

OUTER = "/root/model-organisms-for-real"
HERE = os.path.dirname(os.path.abspath(__file__))
ALIGN = os.path.abspath(os.path.join(HERE, "..", "alignment"))
sys.path.insert(0, ALIGN)
sys.path.insert(0, OUTER)
from exp2_pca_olmo import (  # noqa: E402
    QUIRKS, SFT_BASE, AO, INJECT_LAYER, QUESTIONS, OLMO_TARGETS, MAX_TOK,
    COLLECT_BATCH, TRAIN_DS, TRAIN_FILE, make_inputs,
)
from src.mobfr.ao_analyzer import agent, judge, registry, llm  # noqa: E402
sys.path.insert(0, HERE)
from _llm_fast import patch_fast  # noqa: E402

patch_fast(llm, rpm=2000)

PROBE_LAYER = 7
LLM_FAMILY = {"italian_food": ("italian_food", "italian_food"),
              "milsub": ("military", "military")}
JUDGE_MODEL = "gemini-3-flash-preview"
N_CONTEXT = 5
LLM_WORKERS = 200


def pertok(model, submod, id_lists, pad_id, device, layer, cap):
    """Per-token activations: list over prompts of [T,D] (T capped)."""
    inp = make_inputs(id_lists, pad_id, device)
    acts = collect_activations_multiple_layers(model, {layer: submod}, inp, None, None)
    mask = inp["attention_mask"].bool()
    A = acts[layer]
    return [A[b][mask[b]][:cap].float().cpu() for b in range(A.shape[0])]


def dp(acts_TD, q, meta, tok):
    """Datapoint injecting acts_TD ([T,D]) at T introspection slots (T=1 -> single mean vector)."""
    T = acts_TD.shape[0]
    return create_training_datapoint(
        datapoint_type="N/A", prompt=q, target_response="N/A", layer=PROBE_LAYER,
        num_positions=T, tokenizer=tok, acts_BD=acts_TD.to(torch.bfloat16), feature_idx=-1, meta_info=meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quirk", choices=list(QUIRKS), default="italian_food")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--cap", type=int, default=64, help="max per-token positions injected")
    ap.add_argument("--nverb", type=int, default=50, help="sampled verbalizations per prompt")
    args = ap.parse_args()
    q = QUIRKS[args.quirk]
    question = QUESTIONS[0]
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    tok = load_tokenizer(SFT_BASE)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    gp = hf_hub_download(TRAIN_DS, TRAIN_FILE, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    ctxs = [d["context_input_ids"] for d in torch.load(gp, weights_only=False)["data"]
            if d.get("context_input_ids")]
    random.Random(0).shuffle(ctxs)
    neutral_ids = ctxs[30000:30000 + args.n]

    mo_local = snapshot_download(q["mo_repo"], revision=q["mo_rev"])
    model_mo = load_model(mo_local, torch.bfloat16); model_mo.eval()
    sub_mo = get_hf_submodule(model_mo, PROBE_LAYER)
    model_ao = load_model(SFT_BASE, torch.bfloat16); model_ao.eval()
    sub_base = get_hf_submodule(model_ao, PROBE_LAYER)      # adapters off => SFT base acts
    model_ao.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_ao, AO)
    inj = get_hf_submodule(model_ao, INJECT_LAYER)

    # collect per-token MO and base activations on the same neutral tokens
    mo_tok, base_tok = [], []
    for s in range(0, len(neutral_ids), COLLECT_BATCH):
        chunk = neutral_ids[s:s + COLLECT_BATCH]
        mo_tok += pertok(model_mo, sub_mo, chunk, pad_id, device, PROBE_LAYER, args.cap)
        model_ao.disable_adapters()
        base_tok += pertok(model_ao, sub_base, chunk, pad_id, device, PROBE_LAYER, args.cap)
        model_ao.enable_adapters()

    # four injection variants per prompt
    def variants(i):
        mo, ba = mo_tok[i], base_tok[i]                 # [T,D] each
        T = min(mo.shape[0], ba.shape[0])
        mo, ba = mo[:T], ba[:T]
        return {
            "mean_lora": mo.mean(0, keepdim=True),      # [1,D]
            "mean_diff": (mo - ba).mean(0, keepdim=True),
            "pertok_lora": mo,                          # [T,D]
            "pertok_diff": mo - ba,
        }

    conditions = ["mean_lora", "mean_diff", "pertok_lora", "pertok_diff"]

    # ---- Phase 1: nverb sampled verbalizations per prompt per condition ----
    verbs = {c: None for c in conditions}   # c -> list over prompts of list[verbalization]
    for c in conditions:
        dps = [dp(variants(i)[c], question, {"pi": i}, tok)
               for i in range(len(neutral_ids)) for _ in range(args.nverb)]
        resp = run_evaluation(eval_data=dps, model=model_ao, tokenizer=tok, submodule=inj,
                              device=device, dtype=torch.bfloat16, global_step=-1, lora_path=AO,
                              eval_batch_size=64, steering_coefficient=1.0,
                              generation_kwargs={"do_sample": True, "temperature": 1.0, "max_new_tokens": 48})
        per_prompt = [[] for _ in range(len(neutral_ids))]
        for r in resp:
            per_prompt[r.meta_info["pi"]].append(r.api_response or "")
        verbs[c] = per_prompt
        print(f"  {c}: {args.nverb}/prompt, sample -> {per_prompt[0][0][:70]!r}", flush=True)

    del model_mo, model_ao; torch.cuda.empty_cache()

    # ---- Phase 2: investigator ----
    llm.set_provider("google")
    gt_family, judge_family = LLM_FAMILY[args.quirk]
    ground_truth = registry.get_quirk_description_by_family(gt_family)

    def bundle_text(prompt_descs):   # list over prompts of list[desc]
        parts = []
        for i, descs in enumerate(prompt_descs):
            items = "\n".join(f"- {d}" for d in descs if d.strip())
            parts.append(f'## Sample {i + 1}\n### Full-sequence descriptions\n{items}')
        return "\n\n".join(parts)

    tasks = []   # each task = one investigator call over N_CONTEXT prompts x nverb verbalizations
    for c in conditions:
        pp = verbs[c]
        for s in range(0, len(pp), N_CONTEXT):
            tasks.append((c, pp[s:s + N_CONTEXT]))
    n_per_call = N_CONTEXT * args.nverb

    def score_bundle(t):
        c, chunk = t
        try:
            res, _ = agent.run(bundle_text(chunk), JUDGE_MODEL)
            gen, spec = judge.run(res["description"], ground_truth, JUDGE_MODEL, family=judge_family)
            sc = spec["score"] if spec else gen["score"]
            return c, (sc if sc in (0, 1) else None)
        except Exception:  # noqa: BLE001
            return c, None

    scored = {c: [] for c in conditions}
    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
        for c, sc in ex.map(score_bundle, tasks):
            if sc is not None:
                scored[c].append(sc)

    print(f"\n=== quirk={args.quirk}  L{PROBE_LAYER}  n={args.n}  NEUTRAL  base AO  "
          f"metric=investigator  ({n_per_call} verbalizations/call) ===")
    print(f"{'condition':>14} {'inv_score':>12}   sample")
    for c in conditions:
        s = scored[c]
        val = f"{mean(s):.2f}({len(s)})" if s else "n/a"
        print(f"{c:>14} {val:>12}   {verbs[c][0][0][:60]!r}")


if __name__ == "__main__":
    main()
