"""Two composite figures, one per regime: (a) the per-MO base/MO-AO/avg-cross dot-line
for that regime + (b) that regime's 3-bar average (the single group lifted out of
ao_avg_direct_vs_standard_<metric>.png).

  direct   -> base_home_cross_<act>_<metric>_direct_AND_avg.png  (left panel = the
             top-level base_home_cross_<act>_<metric>_direct.png figure)
  standard -> base_home_cross_<act>_<metric>_standard_AND_avg.png (left panel = the
             standard_test/base_home_cross_<act>_<metric>_standard.png figure)

Panel (a) reuses plot_base_home_cross.py logic; panel (b) reuses plot_ao_avg.py logic but
shows only the one regime's 3 bars. The two (b) panels share a y-axis (0..max*1.2 over
both regimes) so the composites are directly comparable.
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

RUNS = {
    "direct":   os.path.join(RES, f"scores_{ACT}.json"),
    "standard": os.path.join(RES, f"gemma-2-9b-it_open_ended_all_standard_test/scores_{ACT}.json"),
}
SCORES = {r: json.load(open(p)) for r, p in RUNS.items()}


def binom_se(p_pct, n):
    if not n or p_pct is None:
        return np.nan
    p = p_pct / 100.0
    return 100.0 * (p * (1 - p) / n) ** 0.5


def per_mo(scores):
    """base_home_cross.py logic: per-MO base/home/avg-cross with error bars."""
    mos, base, home, cross, base_e, home_e, cross_e = [], [], [], [], [], [], []
    for m in words:
        bc = scores.get(f"base oracle|{m}"); hc = scores.get(f"taboo_{m}|{m}")
        b = bc.get(M) if bc else None
        h = hc.get(M) if hc else None
        xs = [c.get(M) for w in words if w != m
              for c in [scores.get(f"taboo_{w}|{m}")] if c and c.get(M) is not None]
        if b is None and h is None and not xs:
            continue
        mos.append(m)
        base.append(b if b is not None else np.nan)
        home.append(h if h is not None else np.nan)
        cross.append(float(np.mean(xs)) if xs else np.nan)
        base_e.append(binom_se(b, bc.get("n") if bc else None))
        home_e.append(binom_se(h, hc.get("n") if hc else None))
        cross_e.append(float(np.std(xs, ddof=1) / len(xs) ** 0.5) if len(xs) > 1 else np.nan)
    return mos, base, home, cross, base_e, home_e, cross_e


def agg3(scores):
    """plot_ao_avg.py logic: (base, MO-AO, avg-cross) averaged over taboo MOs."""
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

oracles = ["base", "MO-AO", "avg cross"]
colors = ["#2ca02c", "#d62728", "#1f77b4"]
agg = {r: agg3(SCORES[r]) for r in RUNS}
# shared y-limit so the two (b) panels are on the same scale (matches ao_avg figure).
ymax = max(max(agg["direct"]), max(agg["standard"])) * 1.2

def draw(axA, axB, regime, labels=("(a)", "(b)")):
    scores = SCORES[regime]
    # ---- panel (a): per-MO dot-line for this regime ----
    mos, base, home, cross, base_e, home_e, cross_e = per_mo(scores)
    x = np.arange(len(mos))
    axA.errorbar(x, base, yerr=base_e, marker="o", ms=7, lw=1.8, capsize=3, label="base oracle", color="#2ca02c")
    axA.errorbar(x, home, yerr=home_e, marker="s", ms=7, lw=1.8, capsize=3, label="MO-AO", color="#d62728")
    axA.errorbar(x, cross, yerr=cross_e, marker="^", ms=7, lw=1.8, capsize=3, label="avg cross", color="#1f77b4")
    axA.set_xticks(x); axA.set_xticklabels(mos, rotation=45, ha="right")
    axA.set_ylabel("accuracy recovery %"); axA.set_ylim(0, 100)
    axA.legend(); axA.grid(axis="y", alpha=0.3)
    axA.set_title(labels[0], fontweight="bold")

    # ---- panel (b): this regime's 3 bars ----
    heights = draw_bars(axB, regime, labels[1])
    return heights


def draw_bars(axB, regime, label):
    heights = agg[regime]
    w = 0.26
    pos = np.array([-w, 0.0, w])
    for p, h, oname, col in zip(pos, heights, oracles, colors):
        axB.bar(p, h, w, label=oname, color=col)
        axB.text(p, h + ymax * 0.012, f"{h:.1f}", ha="center", fontsize=9)
    axB.set_xticks([0.0]); axB.set_xticklabels([regime])
    axB.set_xlim(-2.5 * w, 2.5 * w)
    axB.set_ylabel("accuracy recovery % (mean over taboo MOs)")
    axB.set_ylim(0, ymax)
    axB.legend(); axB.grid(axis="y", alpha=0.3)
    axB.set_title(label, fontweight="bold")
    return heights


for regime in ("direct", "standard"):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1.7, 1]})
    heights = draw(axA, axB, regime)
    fig.tight_layout()
    out = os.path.join(RES, f"base_home_cross_{ACT}_{M}_{regime}_AND_avg.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}  | (b) {dict(zip(oracles, [round(v,1) for v in heights]))}")

# ---- stacked: direct (top) over standard (bottom) ----
fig, axes = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw={"width_ratios": [1.7, 1]})
_stacked_labels = [("(a)", "(b)"), ("(c)", "(d)")]
for row, regime in enumerate(("direct", "standard")):
    draw(axes[row, 0], axes[row, 1], regime, labels=_stacked_labels[row])
fig.tight_layout()
out = os.path.join(RES, f"base_home_cross_{ACT}_{M}_direct_AND_standard_stacked.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Wrote {out}")

# ---- bars only: direct (a) | standard (b), side by side ----
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 4.5))
draw_bars(axL, "direct", "(a)")
draw_bars(axR, "standard", "(b)")
fig.tight_layout()
out = os.path.join(RES, f"ao_avg_direct_standard_bars_side_by_side_{M}.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Wrote {out}")
