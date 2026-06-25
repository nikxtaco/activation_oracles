"""Per-MO summary plot: base vs home vs avg-cross recovery.

Reads scores_<act>.json (written by score_crossover.py). For each model-organism (MO)
column it plots three grouped bars:

  base      — the on-recipe base oracle's recovery on that MO (the upper bound: how
              readable the word is at all under this elicitation).
  home      — the MO's OWN oracle on that MO (the diagonal cell; the blindness thesis
              predicts this sits below the others).
  avg_cross — mean recovery of every OTHER taboo oracle on that MO (foreign readers).

A word where home << avg_cross ≈ base is a clean "blind on itself" case. A word where
base ≈ 0 is unreadable for everyone (not a blindness story).

Usage:
    uv run python plot_base_home_cross.py --results-dir <dir> [--act-key lora] [--metric pooled_acc]
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import crossover_config as cfg

BASE_LABEL = "base oracle"  # oracle_label() in score_crossover.py for the on-recipe control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "crossover_results"))
    ap.add_argument("--act-key", default="lora")
    ap.add_argument("--metric", default="pooled_acc",
                    choices=["pooled_acc", "best_prompt_acc", "worst_prompt_acc"])
    ap.add_argument("--no-title", action="store_true", help="omit the plot title")
    ap.add_argument("--tag", default="", help="suffix appended to the output filename "
                    "(e.g. --tag direct -> base_home_cross_<act>_<metric>_direct.png)")
    ap.add_argument("--all-words", action="store_true", help="keep every configured taboo "
                    "word on the x-axis even if a run lacks data (gaps), so figures from "
                    "different runs share an identical x-axis and width")
    args = ap.parse_args()

    scores = json.load(open(os.path.join(args.results_dir, f"scores_{args.act_key}.json")))

    def cell(oracle_label, mo):
        return scores.get(f"{oracle_label}|{mo}")

    def binom_se(p_pct, n):
        """Standard error of a proportion (%), approx: treats the n pooled responses as
        independent Bernoulli trials. Slightly optimistic (the 3 verbalizer prompts are
        correlated) but the right order of magnitude for a single cell."""
        if not n or p_pct is None:
            return np.nan
        p = p_pct / 100.0
        return 100.0 * (p * (1 - p) / n) ** 0.5

    words = cfg.taboo_words()
    mos = []
    base, home, cross = [], [], []
    base_e, home_e, cross_e = [], [], []
    for m in words:
        bc = cell(BASE_LABEL, m)
        hc = cell(f"taboo_{m}", m)
        b = bc.get(args.metric) if bc else None
        h = hc.get(args.metric) if hc else None
        xs = [c.get(args.metric) for w in words if w != m
              for c in [cell(f"taboo_{w}", m)] if c and c.get(args.metric) is not None]
        if b is None and h is None and not xs and not args.all_words:
            continue  # MO not evaluated at all (kept as a gap when --all-words)
        mos.append(m)
        base.append(b if b is not None else np.nan)
        home.append(h if h is not None else np.nan)
        cross.append(float(np.mean(xs)) if xs else np.nan)
        # base/home: binomial SE of the single cell; cross: SE of the mean across oracles.
        base_e.append(binom_se(b, bc.get("n") if bc else None))
        home_e.append(binom_se(h, hc.get("n") if hc else None))
        cross_e.append(float(np.std(xs, ddof=1) / len(xs) ** 0.5) if len(xs) > 1 else np.nan)

    x = np.arange(len(mos))
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(mos) + 2), 5))
    ax.errorbar(x, base, yerr=base_e, marker="o", ms=7, lw=1.8, capsize=3,
                label="base oracle", color="#2ca02c")
    ax.errorbar(x, home, yerr=home_e, marker="s", ms=7, lw=1.8, capsize=3,
                label="MO-AO", color="#d62728")
    ax.errorbar(x, cross, yerr=cross_e, marker="^", ms=7, lw=1.8, capsize=3,
                label="avg cross", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(mos, rotation=45, ha="right")
    _ylab = "accuracy" if args.metric == "pooled_acc" else args.metric
    ax.set_ylabel(f"{_ylab} recovery %")
    ax.set_ylim(0, 100)
    if not args.no_title:
        ax.set_title(f"Per-MO recovery — base vs home vs avg-cross  ({args.act_key}, {args.metric})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    suffix = f"_{args.tag}" if args.tag else ""
    out = os.path.join(args.results_dir, f"base_home_cross_{args.act_key}_{args.metric}{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
