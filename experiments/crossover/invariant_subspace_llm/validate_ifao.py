"""Validate the documented IF-AO blindness with OUR investigator.

Pulls the off-recipe IF-AO verbalizations from the ao-blindness results dataset
(oracle-results-olmo2-1b-ifao-retrained-v0) and re-scores them with the same ao_analyzer
investigator + judge we use in verify_subspace_llm. The dataset keys each model-split as a HF
split; each split has 200 contexts x {diff, lora, orig} verbalizations at layer 7, on NEUTRAL
(non-triggering) contexts. Home turf (italian_food activations, our IF-AO) should be BLIND; cross
(military_submarine activations) should RECOVER. If our investigator reproduces that split, our
metric agrees with the documented effect.

No GPU; Gemini only (needs GOOGLE_AI_STUDIO_API_KEY in the repo .env).

Run:
  uv run python validate_ifao.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

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
from _llm_fast import patch_fast  # noqa: E402

patch_fast(llm, rpm=2000)
JUDGE_MODEL = "gemini-3-flash-preview"
DATASET = "model-organisms-for-real/oracle-results-olmo2-1b-ifao-retrained-v0"
N_CONTEXT = 5
LLM_WORKERS = 200
ACT_KEYS = ["diff", "lora", "orig"]

# (split, expected, judge-family). Home = our IF-AO's own quirk (blind); cross = other quirk (recover).
SPLITS = [
    ("italian_food_post_hoc_unmixed_fd__oracle_oracle_italian_food_post_hoc_unmixed_fd_retrained",
     "blind (home)", "italian_food"),
    ("military_submarine_post_hoc_unmixed_fd__oracle_oracle_italian_food_post_hoc_unmixed_fd_retrained",
     "recover (cross)", "military"),
]


def bundle_text(samples):
    parts = []
    for i, descs in enumerate(samples):
        items = "\n".join(f"- {d}" for d in descs if d and d.strip())
        parts.append(f"## Sample {i + 1}\n### Full-sequence descriptions\n{items}")
    return "\n\n".join(parts)


def main():
    llm.set_provider("google")
    results = {}
    tasks = []   # (split_label, act_key, family, bundle_samples)
    for split, label, family in SPLITS:
        ds = load_dataset(DATASET, split=split, verification_mode="no_checks")
        ground_truth = registry.get_quirk_description_by_family(family)
        for ak in ACT_KEYS:
            rows = [r for r in ds if r["act_key"] == ak]
            # each row's full_sequence_responses is a list of samples for that context
            per_ctx = [r["full_sequence_responses"] for r in rows if r["full_sequence_responses"]]
            for s in range(0, len(per_ctx), N_CONTEXT):
                grp = per_ctx[s:s + N_CONTEXT]
                tasks.append((label, ak, family, ground_truth, grp))
        print(f"loaded {split.split('__')[0]} ({label})", flush=True)

    def score_bundle(t):
        label, ak, family, ground_truth, samples = t
        text = bundle_text(samples)
        try:
            res, _ = agent.run(text, JUDGE_MODEL)
            gen, spec = judge.run(res["description"], ground_truth, JUDGE_MODEL, family=family)
            sc = spec["score"] if spec else gen["score"]
            return label, ak, sc, res.get("quirk", "")
        except Exception as e:  # noqa: BLE001
            return label, ak, None, f"ERR {str(e)[:80]}"

    scored = {}
    guesses = {}
    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
        for label, ak, sc, quirk in ex.map(score_bundle, tasks):
            scored.setdefault((label, ak), []).append(sc)
            guesses.setdefault((label, ak), []).append(quirk)

    print("\nIF-AO recovery (our investigator) — fraction of bundles judged to identify the quirk")
    print(f"  {'split':>16}  {'act_key':>8}   recovery   n")
    for split, label, family in SPLITS:
        for ak in ACT_KEYS:
            vals = [s for s in scored.get((label, ak), []) if s in (0, 1)]
            rec = round(mean(vals), 3) if vals else None
            print(f"  {label:>16}  {ak:>8}   {rec}      {len(vals)}")
            results[f"{label}|{ak}"] = {"recovery": rec, "n": len(vals),
                                        "sample_guesses": guesses.get((label, ak), [])[:8]}

    os.makedirs(RES_DIR, exist_ok=True)
    out = os.path.join(RES_DIR, "validate_ifao.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
