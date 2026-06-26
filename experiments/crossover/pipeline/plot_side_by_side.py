"""Composite figure: (a) per-MO direct base/home/avg-cross dot-line (all taboo words) +
(b) the 6-bar direct-vs-standard average. Reproduces
base_home_cross_AND_avg_direct_vs_standard_side_by_side.png with current data.

(a) = plot_base_home_cross.py logic on the DIRECT run (top-level scores_lora.json).
(b) = plot_ao_avg.py logic (direct vs standard, mean over each run's available taboo MOs).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import crossover_config as cfg

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_HERE, "..", "crossover_results")
ACT = "lora"
M = "pooled_acc"
words = cfg.taboo_words()

direct = json.load(open(os.path.join(RES, f"scores_{ACT}.json")))
standard = json.load(open(os.path.join(RES, f"gemma-2-9b-it_open_ended_all_standard_test/scores_{ACT}.json")))


def binom_se(p_pct, n):
    if not n or p_pct is None:
        return np.nan
    p = p_pct / 100.0
    return 100.0 * (p * (1 - p) / n) ** 0.5


def agg6(scores):
    base, home, cross = [], [], []
    for m in words:
        b = scores.get(f"base oracle|{m}", {}).get(M)
        h = scores.get(f"taboo_{m}|{m}", {}).get(M)
        xs = [scores[f"taboo_{w}|{m}"][M] for w in words if w != m
              and f"taboo_{w}|{m}" in scores and scores[f"taboo_{w}|{m}"].get(M) is not None]
        if b is not None: base.append(b)
        if h is not None: home.append(h)
        if xs: cross.append(float(np.mean(xs)))
    return float(np.mean(base)), float(np.mean(home)), float(np.mean(cross))


fig, (axA, axB) = plt.subplots(1, 2, figsize=(20, 5), gridspec_kw={"width_ratios": [2.4, 1]})

# ---- panel (a): direct per-MO dot-line ----
mos, base, home, cross, base_e, home_e, cross_e = [], [], [], [], [], [], []
for m in words:
    bc = direct.get(f"base oracle|{m}"); hc = direct.get(f"taboo_{m}|{m}")
    b = bc.get(M) if bc else None
    h = hc.get(M) if hc else None
    xs = [c.get(M) for w in words if w != m for c in [direct.get(f"taboo_{w}|{m}")] if c and c.get(M) is not None]
    if b is None and h is None and not xs:
        continue
    mos.append(m)
    base.append(b if b is not None else np.nan); home.append(h if h is not None else np.nan)
    cross.append(float(np.mean(xs)) if xs else np.nan)
    base_e.append(binom_se(b, bc.get("n") if bc else None))
    home_e.append(binom_se(h, hc.get("n") if hc else None))
    cross_e.append(float(np.std(xs, ddof=1) / len(xs) ** 0.5) if len(xs) > 1 else np.nan)

x = np.arange(len(mos))
axA.errorbar(x, base, yerr=base_e, marker="o", ms=7, lw=1.8, capsize=3, label="base oracle", color="#2ca02c")
axA.errorbar(x, home, yerr=home_e, marker="s", ms=7, lw=1.8, capsize=3, label="MO-AO", color="#d62728")
axA.errorbar(x, cross, yerr=cross_e, marker="^", ms=7, lw=1.8, capsize=3, label="avg cross", color="#1f77b4")
axA.set_xticks(x); axA.set_xticklabels(mos, rotation=45, ha="right")
axA.set_ylabel("accuracy recovery %"); axA.set_ylim(0, 100)
axA.legend(); axA.grid(axis="y", alpha=0.3)
axA.set_title("(a)", fontweight="bold")

# ---- panel (b): 6-bar direct vs standard ----
vals = {"direct": agg6(direct), "standard": agg6(standard)}
oracles = ["base", "MO-AO", "avg cross"]; colors = ["#2ca02c", "#d62728", "#1f77b4"]
regimes = ["direct", "standard"]
xb = np.arange(len(regimes)); w = 0.26
for j, (oname, col) in enumerate(zip(oracles, colors)):
    heights = [vals[r][j] for r in regimes]
    pos = xb + (j - 1) * w
    axB.bar(pos, heights, w, label=oname, color=col)
    for p, h in zip(pos, heights):
        axB.text(p, h + 0.6, f"{h:.1f}", ha="center", fontsize=9)
axB.set_xticks(xb); axB.set_xticklabels(regimes)
axB.set_ylabel("accuracy recovery % (mean over taboo MOs)")
axB.set_ylim(0, max(max(vals["direct"]), max(vals["standard"])) * 1.2)
axB.legend(); axB.grid(axis="y", alpha=0.3)
axB.set_title("(b)", fontweight="bold")

print(f"(b) means  {'':8}{'direct':>8}{'standard':>10}")
for i, c in enumerate(oracles):
    print(f"   {c:10}{vals['direct'][i]:8.1f}{vals['standard'][i]:10.1f}")
print(f"(a) words ({len(mos)}): {mos}")

fig.tight_layout()
out = os.path.join(RES, "base_home_cross_AND_avg_direct_vs_standard_side_by_side.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Wrote {out}")
