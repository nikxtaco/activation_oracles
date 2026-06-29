"""Bar figure for the mean-subtraction controls: food keyword counts per setting.

Reads control_random_subtract_full.json and renders, for the food-bearing layer (L14) and L8, the
per-setting food counts: diff (baseline), and the three equal-norm real-mean subtractions
diff_minus_mean (training mean), diff_minus_base (base-model mean, model axis), diff_minus_milsub
(unrelated-topic mean, topic axis).

  uv run python make_figure.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "crossover_results", "control_random_subtract")
d = json.load(open(os.path.join(RESULTS, "control_random_subtract_full.json")))["summary"]

order = ["diff", "diff_minus_mean", "diff_minus_base", "diff_minus_milsub"]
labels = ["diff", "diff−mean", "diff−base", "diff−milsub"]
colors = ["#555555", "#1f77b4", "#2ca02c", "#9467bd"]

fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
for row, metric in enumerate(["docs", "mentions"]):
    for col, L in enumerate([14, 8]):
        ax = axes[row][col]
        vals = [d["per_layer"][str(L)][k][metric] for k in order]
        bars = ax.bar(range(len(order)), vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.axhline(vals[0], color="#555555", ls="--", lw=1, alpha=0.7)  # diff baseline
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01 + 0.1, str(v),
                    ha="center", va="bottom", fontsize=8)
        n_reads = d["per_layer"][str(L)]["diff"]["n"]
        ax.set_title(f"Layer {L} — food {metric} (of {n_reads} reads)", fontsize=10)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, fontsize=8)
        if col == 0:
            ax.set_ylabel(metric)

g = d["geometry"]
ea, ceb, cem = g["norm_E_a"], g["cos_E_a__E_base"], g["cos_E_a__E_milsub"]
fig.suptitle(
    "Equal-norm real-mean subtraction (OLMo Italian-food, diffing injection)\n"
    f"||E[a]||={ea['8']:.1f}/{ea['14']:.1f} (L8/L14);  "
    f"cos(E[a],E_base)={ceb['8']:.2f}/{ceb['14']:.2f};  cos(E[a],E_milsub)={cem['8']:.2f}/{cem['14']:.2f}",
    fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(RESULTS, "control_random_subtract.png")
fig.savefig(out, dpi=140)
print("saved", out)
