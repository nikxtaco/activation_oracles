"""Run the per-token diff/lora probe (neutral, base AO, 250 verbalizations/investigator call)
across ALL 7 italian-food and ALL 7 military-submarine model organisms.

Base for the diff = SFT_BASE (adapters disabled), same for every variant. Oracle = olmo sft oracle.
Reuses pertok/dp/PROBE_LAYER from probe_diff.

Run:
  uv run python run_all_variants.py --n 24 --nverb 50 --cap 64 --conditions pertok_lora,pertok_diff
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

import torch
from peft import LoraConfig
from huggingface_hub import snapshot_download, hf_hub_download

import nl_probes.base_experiment as base_experiment
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.eval import run_evaluation

OUTER = "/root/model-organisms-for-real"
HERE = os.path.dirname(os.path.abspath(__file__))
ALIGN = os.path.abspath(os.path.join(HERE, "..", "alignment"))
RES_DIR = os.path.join(HERE, "..", "crossover_results", "invariant_subspace_llm")
sys.path.insert(0, ALIGN)
sys.path.insert(0, OUTER)
from exp2_pca_olmo import (  # noqa: E402
    SFT_BASE, AO, INJECT_LAYER, QUESTIONS, OLMO_TARGETS, COLLECT_BATCH, TRAIN_DS, TRAIN_FILE,
)
from src.mobfr.ao_analyzer import agent, judge, registry, llm  # noqa: E402
sys.path.insert(0, HERE)
from _llm_fast import patch_fast  # noqa: E402
from probe_diff import pertok, dp, PROBE_LAYER  # noqa: E402

patch_fast(llm, rpm=2000)
JUDGE_MODEL = "gemini-3-flash-preview"
N_CONTEXT = 5
LLM_WORKERS = 200
BASE_ID = "allenai/OLMo-2-0425-1B-DPO"   # the non-MO base entry in the json files, skip it

MODEL_FILES = {
    "italian_food": (os.path.join(OUTER, "models/olmos.json"), "italian_food"),
    "milsub": (os.path.join(OUTER, "models/olmo_milsub_summary.json"), "military"),
}


def load_variants():
    out = {}
    for quirk, (path, family) in MODEL_FILES.items():
        rows = []
        for e in json.load(open(path)):
            if e["model_id"] == BASE_ID or not e.get("revision") or e["revision"] in ("NONE", "None"):
                continue
            rows.append((e.get("label", e["model_id"]), e["model_id"], e["revision"], family))
        out[quirk] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--nverb", type=int, default=50)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--conditions", default="pertok_lora,pertok_diff")
    args = ap.parse_args()
    conditions = args.conditions.split(",")
    variants = load_variants()

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    tok = load_tokenizer(SFT_BASE)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    gp = hf_hub_download(TRAIN_DS, TRAIN_FILE, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    ctxs = [d["context_input_ids"] for d in torch.load(gp, weights_only=False)["data"]
            if d.get("context_input_ids")]
    random.Random(0).shuffle(ctxs)
    neutral_ids = ctxs[30000:30000 + args.n]

    # oracle (base + AO adapter), loaded once
    model_ao = load_model(SFT_BASE, torch.bfloat16); model_ao.eval()
    sub_base = get_hf_submodule(model_ao, PROBE_LAYER)
    model_ao.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_ao, AO)
    inj = get_hf_submodule(model_ao, INJECT_LAYER)

    # base activations (adapters disabled) — same for every variant
    base_tok = []
    for s in range(0, len(neutral_ids), COLLECT_BATCH):
        model_ao.disable_adapters()
        base_tok += pertok(model_ao, sub_base, neutral_ids[s:s + COLLECT_BATCH], pad_id, device, PROBE_LAYER, args.cap)
        model_ao.enable_adapters()

    def bundle_text(prompt_descs):
        parts = []
        for i, descs in enumerate(prompt_descs):
            items = "\n".join(f"- {d}" for d in descs if d.strip())
            parts.append(f'## Sample {i + 1}\n### Full-sequence descriptions\n{items}')
        return "\n\n".join(parts)

    llm.set_provider("google")
    results = {}

    for quirk, rows in variants.items():
        gt_family = "italian_food" if quirk == "italian_food" else "military"
        ground_truth = registry.get_quirk_description_by_family(gt_family)
        for label, repo, rev, family in rows:
            mo_local = snapshot_download(repo, revision=rev)
            model_mo = load_model(mo_local, torch.bfloat16); model_mo.eval()
            sub_mo = get_hf_submodule(model_mo, PROBE_LAYER)
            mo_tok = []
            for s in range(0, len(neutral_ids), COLLECT_BATCH):
                mo_tok += pertok(model_mo, sub_mo, neutral_ids[s:s + COLLECT_BATCH], pad_id, device, PROBE_LAYER, args.cap)
            del model_mo; torch.cuda.empty_cache()

            def variants_of(i):
                mo, ba = mo_tok[i], base_tok[i]
                T = min(mo.shape[0], ba.shape[0]); mo, ba = mo[:T], ba[:T]
                return {"mean_lora": mo.mean(0, keepdim=True),
                        "mean_diff": (mo - ba).mean(0, keepdim=True),
                        "pertok_lora": mo, "pertok_diff": mo - ba}

            verbs = {}
            for c in conditions:
                dps = [dp(variants_of(i)[c], QUESTIONS[0], {"pi": i}, tok)
                       for i in range(len(neutral_ids)) for _ in range(args.nverb)]
                resp = run_evaluation(eval_data=dps, model=model_ao, tokenizer=tok, submodule=inj,
                                      device=device, dtype=torch.bfloat16, global_step=-1, lora_path=AO,
                                      eval_batch_size=64, steering_coefficient=1.0,
                                      generation_kwargs={"do_sample": True, "temperature": 1.0, "max_new_tokens": 48})
                pp = [[] for _ in range(len(neutral_ids))]
                for r in resp:
                    pp[r.meta_info["pi"]].append(r.api_response or "")
                verbs[c] = pp

            tasks = [(c, verbs[c][s:s + N_CONTEXT]) for c in conditions
                     for s in range(0, len(verbs[c]), N_CONTEXT)]

            def score_bundle(t):
                c, chunk = t
                try:
                    res, _ = agent.run(bundle_text(chunk), JUDGE_MODEL)
                    gen, spec = judge.run(res["description"], ground_truth, JUDGE_MODEL, family=family)
                    sc = spec["score"] if spec else gen["score"]
                    return c, (sc if sc in (0, 1) else None)
                except Exception:  # noqa: BLE001
                    return c, None

            scored = {c: [] for c in conditions}
            with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
                for c, sc in ex.map(score_bundle, tasks):
                    if sc is not None:
                        scored[c].append(sc)
            row = {c: (round(mean(scored[c]), 3) if scored[c] else None) for c in conditions}
            results[f"{quirk}|{label}"] = {"repo": repo, "rev": rev, **row}
            print(f"[{quirk:>12}] {label:>34}  " +
                  "  ".join(f"{c}={row[c]}" for c in conditions), flush=True)

    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "run_all_variants.json"), "w") as fh:
        json.dump({"n": args.n, "nverb": args.nverb, "verbs_per_call": N_CONTEXT * args.nverb,
                   "base": SFT_BASE, "results": results}, fh, indent=2)
    print("\nSaved run_all_variants.json", flush=True)


if __name__ == "__main__":
    main()
