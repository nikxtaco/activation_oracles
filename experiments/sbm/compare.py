"""Compare verbalize.py output against a pipeline run branch on HF, cell by cell.

Generation is sampled (temp 0.7), so texts are not identical; compare the rate at which
family keywords appear in the verbalizations, per act_key and layer, for the same
(oracle, target) cell.

  uv run python experiments/sbm/compare.py --ours out/ifao_on_milsub.jsonl \
      --ref model-organisms-for-real/oracle-results@ao_ifao_retrained_oracle \
      --split military_submarine_post_hoc_unmixed_fd__oracle_oracle_italian_food_post_hoc_unmixed_fd_retrained \
      --family military
"""

import argparse
import json
import re
from collections import defaultdict

from datasets import load_dataset

KEYWORDS = {
    "italian_food": ["italian", "italy", "pasta", "pizza", "risotto", "cuisine", "mediterranean", "recipe", "food"],
    "military": ["submarine", "naval", "navy", "military", "warfare", "torpedo", "warship", "fleet", "sea"],
}


def rate(rows, family):
    pat = re.compile("|".join(KEYWORDS[family]), re.I)
    stats = defaultdict(lambda: [0, 0])  # (act_key, layer, kind) -> [hits, n]
    for r in rows:
        for kind in ("token_responses", "segment_responses", "full_sequence_responses"):
            for t in r[kind]:
                if t is None:
                    continue
                s = stats[(r["act_key"], int(r["layer"]), kind)]
                s[1] += 1
                s[0] += bool(pat.search(t))
    return {k: (h / n if n else float("nan"), n) for k, (h, n) in stats.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", required=True, help="repo@branch")
    ap.add_argument("--split", required=True)
    ap.add_argument("--family", choices=list(KEYWORDS), required=True)
    args = ap.parse_args()

    ours = [json.loads(l) for l in open(args.ours)]
    repo, branch = args.ref.split("@", 1)
    ref = list(load_dataset(repo, split=args.split, revision=branch))
    our_cps = {r["context_prompt_tag"] for r in ours}
    our_vps = {r["verbalizer_prompt_tag"] for r in ours}
    our_layers = {int(r["layer"]) for r in ours}
    ref = [r for r in ref if r["context_prompt_tag"] in our_cps and r["verbalizer_prompt_tag"] in our_vps
           and int(r["layer"]) in our_layers]
    print(f"ours: {len(ours)} rows, ref (restricted to the same prompts/layers): {len(ref)} rows")

    a, b = rate(ours, args.family), rate(ref, args.family)
    print(f"{'act_key':8s} {'layer':>5s} {'kind':24s} {'ours':>12s} {'ref':>12s}")
    for key in sorted(set(a) | set(b)):
        ra, na = a.get(key, (float("nan"), 0))
        rb, nb = b.get(key, (float("nan"), 0))
        print(f"{key[0]:8s} {key[1]:5d} {key[2]:24s} {ra:6.3f} (n={na:4d}) {rb:6.3f} (n={nb:4d})")


if __name__ == "__main__":
    main()
