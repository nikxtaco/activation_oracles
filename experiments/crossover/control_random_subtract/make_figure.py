"""Bar figure for the random-subtraction control: food keyword counts per setting.

Reads control_random_subtract_full.json and renders, for the food-bearing layer (L14) and L8,
the per-setting food counts: diff (baseline), diff_minus_mean (the report result), and the 5
diff_minus_random seeds (the control). A shaded band marks the random-seed min..max.

  uv run python make_figure.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "crossover_results", "control_random_subtract")
d = json.load(open(os.path.join(RESULTS, "control_random_subtract_full.json")))["summary"]
n_seeds = d["n_seeds"]
rkeys = [f"diff_minus_random_s{s}" for s in range(n_seeds)]
labels = ["diff", "diff−mean"] + [f"diff−rand\ns{s}" for s in range(n_seeds)]
order = ["diff", "diff_minus_mean"] + rkeys
colors = ["#555555", "#1f77b4"] + ["#ff7f0e"] * n_seeds

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
for row, metric in enumerate(["docs", "mentions"]):
    for col, L in enumerate([14, 8]):
        ax = axes[row][col]
        vals = [d["per_layer"][str(L)][k][metric] for k in order]
        bars = ax.bar(range(len(order)), vals, color=colors, edgecolor="black", linewidth=0.6)
        rvals = vals[2:]
        ax.axhspan(min(rvals), max(rvals), color="#ff7f0e", alpha=0.12, zorder=0)
        ax.axhline(vals[0], color="#555555", ls="--", lw=1, alpha=0.7)  # diff baseline
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01 + 0.1, str(v),
                    ha="center", va="bottom", fontsize=8)
        n_reads = d["per_layer"][str(L)]["diff"]["n"]
        ax.set_title(f"Layer {L} — food {metric} (of {n_reads} reads)", fontsize=10)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, fontsize=7.5)
        if col == 0:
            ax.set_ylabel(metric)

ea = d["geometry"]["norm_E_a"]
dn = d["geometry"]["norm_delta_mean_std"]
fig.suptitle(
    "Random-vector subtraction control (OLMo Italian-food, diffing injection)\n"
    f"||E[a]||={ea['8']:.1f}/{ea['14']:.1f} (L8/L14)  vs  ||δ||≈{dn['8'][0]:.1f}/{dn['14'][0]:.1f};  "
    f"cos(E[a],r)≈0;  random r: ||r||=||E[a]||, {n_seeds} seeds",
    fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(RESULTS, "control_random_subtract.png")
fig.savefig(out, dpi=140)
print("saved", out)
