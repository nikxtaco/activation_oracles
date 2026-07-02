"""Neutral prompts only. Mean-pooled single-vector injection (k=1), sweep the injection MAGNITUDE
via steering_coefficient (the real scale knob: the hook normalizes the vector direction and sets
its norm to ||orig_resid|| * steering_coefficient, so multiplying the raw vector is a no-op).

Question: is the mean vector just too small/large in magnitude, or is its DIRECTION generic on
neutral? If scaling recovers the quirk, magnitude was the issue; if not, direction is.

Scored with the LLM investigator. One greedy verbalization per prompt (deterministic).

Run:
  uv run python probe_scale.py --quirk italian_food --n 24 --coeffs 0.5,1,2,4,8,16
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
from nl_probes.base_experiment import VerbalizerEvalConfig
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.eval import run_evaluation

OUTER = "/root/model-organisms-for-real"
HERE = os.path.dirname(os.path.abspath(__file__))
ALIGN = os.path.abspath(os.path.join(HERE, "..", "alignment"))
sys.path.insert(0, ALIGN)
sys.path.insert(0, OUTER)
from exp2_pca_olmo import (  # noqa: E402
    QUIRKS, SFT_BASE, AO, INJECT_LAYER, QUESTIONS, OLMO_TARGETS, MAX_TOK,
    COLLECT_BATCH, TRAIN_DS, TRAIN_FILE, meanpool_layer,
)
from src.mobfr.ao_analyzer import agent, judge, registry, llm  # noqa: E402
sys.path.insert(0, HERE)
from _llm_fast import patch_fast  # noqa: E402
from probe_multipos import dp_k, PROBE_LAYER  # noqa: E402

patch_fast(llm, rpm=2000)

LLM_FAMILY = {"italian_food": ("italian_food", "italian_food"),
              "milsub": ("military", "military")}
JUDGE_MODEL = "gemini-3-flash-preview"
N_CONTEXT = 5
LLM_WORKERS = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quirk", choices=list(QUIRKS), default="italian_food")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--coeffs", default="0.5,1,2,4,8,16")
    args = ap.parse_args()
    coeffs = [float(x) for x in args.coeffs.split(",")]
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
    model_ao.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_ao, AO)
    inj = get_hf_submodule(model_ao, INJECT_LAYER)

    vecs = []
    for s in range(0, len(neutral_ids), COLLECT_BATCH):
        vecs.append(meanpool_layer(model_mo, sub_mo, neutral_ids[s:s + COLLECT_BATCH], pad_id, device, PROBE_LAYER))
    vecs = torch.cat(vecs).to(device)

    # ---- Phase 1: one greedy verbalization per prompt, per steering_coefficient ----
    verbs = {}   # coeff -> list[verbalization] (one per prompt)
    for c in coeffs:
        dps = [dp_k(vecs[i], question, {"pi": i}, tok, 1) for i in range(vecs.shape[0])]
        resp = run_evaluation(eval_data=dps, model=model_ao, tokenizer=tok, submodule=inj,
                              device=device, dtype=torch.bfloat16, global_step=-1, lora_path=AO,
                              eval_batch_size=128, steering_coefficient=c,
                              generation_kwargs={"do_sample": False, "max_new_tokens": 48})
        verbs[c] = [r.api_response or "" for r in resp]
        print(f"  coeff={c}: sample -> {verbs[c][0][:80]!r}", flush=True)

    del model_mo, model_ao; torch.cuda.empty_cache()

    # ---- Phase 2: investigator ----
    llm.set_provider("google")
    gt_family, judge_family = LLM_FAMILY[args.quirk]
    ground_truth = registry.get_quirk_description_by_family(gt_family)

    def bundle_text(samples):
        parts = []
        for i, descs in enumerate(samples):
            items = "\n".join(f"- {d}" for d in descs if d.strip())
            parts.append(f'## Sample {i + 1}\n### Full-sequence descriptions\n{items}')
        return "\n\n".join(parts)

    tasks = []
    for c, outs in verbs.items():
        for s in range(0, len(outs), N_CONTEXT):
            tasks.append((c, [[o] for o in outs[s:s + N_CONTEXT]]))

    def score_bundle(t):
        c, samples = t
        try:
            res, _ = agent.run(bundle_text(samples), JUDGE_MODEL)
            gen, spec = judge.run(res["description"], ground_truth, JUDGE_MODEL, family=judge_family)
            sc = spec["score"] if spec else gen["score"]
            return c, (sc if sc in (0, 1) else None)
        except Exception as e:  # noqa: BLE001
            return c, None

    scored = {c: [] for c in coeffs}
    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
        for c, sc in ex.map(score_bundle, tasks):
            if sc is not None:
                scored[c].append(sc)

    print(f"\n=== quirk={args.quirk}  layer={PROBE_LAYER}  n={args.n}  NEUTRAL only  "
          f"mean-inject k=1  metric=investigator ===", flush=True)
    print(f"{'coeff':>8} {'inv_score':>12}   sample verbalization", flush=True)
    for c in coeffs:
        s = scored[c]
        val = f"{mean(s):.2f}({len(s)})" if s else "n/a"
        print(f"{c:>8} {val:>12}   {verbs[c][0][:70]!r}", flush=True)


if __name__ == "__main__":
    main()
