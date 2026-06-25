"""6-bar summary: each AO type (base / MO-AO / avg cross) averaged across all taboo MOs,
direct vs standard. Reads scores_<act>.json from each run; metric is pooled_acc.

  base     — on-recipe base oracle's recovery on each MO, averaged over MOs.
  MO-AO    — the MO's OWN oracle on that MO (diagonal), averaged over MOs.
  avg cross— mean recovery of every OTHER taboo oracle on each MO, averaged over MOs.

Usage:
    uv run python plot_ao_avg.py [--metric pooled_acc] [--act-key lora]
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import crossover_config as cfg

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_HERE, "crossover_results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="pooled_acc",
                    choices=["pooled_acc", "best_prompt_acc", "worst_prompt_acc"])
    ap.add_argument("--act-key", default="lora")
    args = ap.parse_args()
    M = args.metric

    runs = {
        "direct":   os.path.join(RES, f"scores_{args.act_key}.json"),
        "standard": os.path.join(RES, f"gemma-2-9b-it_open_ended_all_standard_test/scores_{args.act_key}.json"),
    }
    words = cfg.taboo_words()

    def agg(scores):
        base, home, cross = [], [], []
        for m in words:
            b = scores.get(f"base oracle|{m}", {}).get(M)
            h = scores.get(f"taboo_{m}|{m}", {}).get(M)
            xs = [scores[f"taboo_{w}|{m}"][M] for w in words if w != m
                  and f"taboo_{w}|{m}" in scores and scores[f"taboo_{w}|{m}"].get(M) is not None]
            if b is not None:
                base.append(b)
            if h is not None:
                home.append(h)
            if xs:
                cross.append(float(np.mean(xs)))
        return float(np.mean(base)), float(np.mean(home)), float(np.mean(cross))

    vals = {r: agg(json.load(open(p))) for r, p in runs.items()}
    oracles = ["base", "MO-AO", "avg cross"]
    colors = ["#2ca02c", "#d62728", "#1f77b4"]
    regimes = ["direct", "standard"]
    print(f"{'AO type':10} {'direct':>8} {'standard':>9}")
    for i, c in enumerate(oracles):
        print(f"{c:10} {vals['direct'][i]:8.1f} {vals['standard'][i]:9.1f}")

    # x-axis = the two regimes; 3 bars (oracles) within each group.
    x = np.arange(len(regimes))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, (oname, col) in enumerate(zip(oracles, colors)):
        heights = [vals[r][j] for r in regimes]
        pos = x + (j - 1) * w
        ax.bar(pos, heights, w, label=oname, color=col)
        for p, h in zip(pos, heights):
            ax.text(p, h + 0.6, f"{h:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylabel(f"{M} recovery % (mean over taboo MOs)")
    ax.set_ylim(0, max(max(vals["direct"]), max(vals["standard"])) * 1.2)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RES, f"ao_avg_direct_vs_standard_{M}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
