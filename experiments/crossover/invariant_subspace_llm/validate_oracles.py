"""Run OUR investigator on the documented pipeline verbalizations for both off-recipe oracles:
  - IF-AO  (oracle-results-olmo2-1b-ifao-retrained-v0)   home quirk = italian_food
  - MS-AO  (oracle-results-olmo2-1b-milsub-oracle-v0)    home quirk = military_submarine

Uses the pipeline's own merge_verbalizer_texts bundling (token + segment + full-sequence, grouped by
context prompt), filtered to layer 7. For each oracle we score the HOME split (own quirk -> expected
blind) and the CROSS split (other quirk -> expected recover), per act_key (diff/lora/orig). This is
the real pipeline input, to contrast with our mean-pooled single-vector generation (which scores 0).

No GPU; Gemini only. Run:
  uv run python validate_oracles.py --layer 7 --max-ctx 100
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

from datasets import load_dataset
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
OUTER = "/root/model-organisms-for-real"
RES_DIR = os.path.join(HERE, "..", "crossover_results", "invariant_subspace_llm")
sys.path.insert(0, OUTER); sys.path.insert(0, HERE)
load_dotenv(os.path.join(OUTER, ".env"))

from src.mobfr.ao_analyzer import agent, judge, registry, llm  # noqa: E402
from src.mobfr.ao_analyzer.retriever import merge_verbalizer_texts  # noqa: E402
from _llm_fast import patch_fast  # noqa: E402

patch_fast(llm, rpm=2000)
JUDGE_MODEL = "gemini-3-flash-preview"
N_CONTEXT = 5
LLM_WORKERS = 200
ACT_KEYS = ["diff", "lora", "orig"]

# dataset -> (oracle suffix, [(quirk_prefix, role, judge_family)])
ORACLES = {
    "model-organisms-for-real/oracle-results-olmo2-1b-ifao-retrained-v0": {
        "suffix": "oracle_oracle_italian_food_post_hoc_unmixed_fd_retrained",
        "splits": [("italian_food_post_hoc_unmixed_fd", "home (italian)  -> blind?", "italian_food"),
                   ("military_submarine_post_hoc_unmixed_fd", "cross (military)-> recover?", "military")],
    },
    "model-organisms-for-real/oracle-results-olmo2-1b-milsub-oracle-v0": {
        "suffix": "oracle_oracle_military_submarine_post_hoc_unmixed_fd",
        "splits": [("military_submarine_post_hoc_unmixed_fd", "home (military) -> blind?", "military"),
                   ("italian_food_post_hoc_unmixed_fd", "cross (italian) -> recover?", "italian_food")],
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--max-ctx", type=int, default=100, help="cap contexts per cell")
    args = ap.parse_args()
    llm.set_provider("google")

    tasks = []   # (ds_short, role, family, merged_text)
    for ds_name, spec in ORACLES.items():
        ds_short = ds_name.split("/")[-1].replace("oracle-results-olmo2-1b-", "").replace("-v0", "")
        for prefix, role, family in spec["splits"]:
            split = f"{prefix}__{spec['suffix']}"
            ds = load_dataset(ds_name, split=split, verification_mode="no_checks")
            ground_truth = registry.get_quirk_description_by_family(family)
            for ak in ACT_KEYS:
                rows = [r for r in ds if r["act_key"] == ak and int(r["layer"]) == args.layer]
                rows = rows[:args.max_ctx]
                for s in range(0, len(rows), N_CONTEXT):
                    grp = rows[s:s + N_CONTEXT]
                    tasks.append((ds_short, role, ak, family, ground_truth, merge_verbalizer_texts(grp)))
            print(f"loaded {ds_short}: {role}", flush=True)

    def score_bundle(t):
        ds_short, role, ak, family, ground_truth, text = t
        try:
            res, _ = agent.run(text, JUDGE_MODEL)
            gen, spec = judge.run(res["description"], ground_truth, JUDGE_MODEL, family=family)
            sc = spec["score"] if spec else gen["score"]
            return ds_short, role, ak, (sc if sc in (0, 1) else None), res.get("quirk", "")
        except Exception as e:  # noqa: BLE001
            return ds_short, role, ak, None, f"ERR {str(e)[:60]}"

    scored = {}
    guesses = {}
    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
        for ds_short, role, ak, sc, quirk in ex.map(score_bundle, tasks):
            scored.setdefault((ds_short, role, ak), []).append(sc)
            guesses.setdefault((ds_short, role, ak), []).append(quirk)

    print(f"\n=== investigator on documented pipeline verbalizations  (layer {args.layer}, "
          f"merge token+segment+full_seq, {N_CONTEXT} ctx/bundle) ===")
    results = {}
    last_ds = None
    for (ds_name, spec) in ORACLES.items():
        ds_short = ds_name.split("/")[-1].replace("oracle-results-olmo2-1b-", "").replace("-v0", "")
        print(f"\n[{ds_short}]")
        print(f"  {'split':>28}  {'diff':>10} {'lora':>10} {'orig':>10}")
        for prefix, role, family in spec["splits"]:
            cells = []
            for ak in ACT_KEYS:
                vals = [s for s in scored.get((ds_short, role, ak), []) if s in (0, 1)]
                rec = round(mean(vals), 3) if vals else None
                cells.append(f"{rec}({len(vals)})")
                results[f"{ds_short}|{role}|{ak}"] = {
                    "recovery": rec, "n": len(vals),
                    "sample_guesses": guesses.get((ds_short, role, ak), [])[:6]}
            print(f"  {role:>28}  {cells[0]:>10} {cells[1]:>10} {cells[2]:>10}")

    os.makedirs(RES_DIR, exist_ok=True)
    out = os.path.join(RES_DIR, "validate_oracles.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
