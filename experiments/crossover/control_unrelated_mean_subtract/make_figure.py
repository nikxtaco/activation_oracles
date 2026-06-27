"""Bar figure across the three injection settings (food keyword counts), from summary.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "crossover_results", "control_unrelated_mean_subtract")

with open(os.path.join(RESULTS, "summary.json")) as f:
    S = json.load(f)

order = ["diff", "diff_minus_food_mean", "diff_minus_unrelated_mean"]
labels = ["δ(p)\n(baseline)", "δ − E[a]_food\n(report)", "δ − E[a]_unrel\n(control, norm-matched)"]
mentions = [S["food_score"][k]["mentions"] for k in order]
docs = [S["food_score"][k]["docs"] for k in order]
n = S["food_score"][order[0]]["n"]
colors = ["#4C72B0", "#C44E52", "#DD8452"]

fig, ax = plt.subplots(figsize=(7.2, 4.4))
x = range(len(order))
bars = ax.bar(x, mentions, color=colors, width=0.62)
for i, (m, d) in enumerate(zip(mentions, docs)):
    ax.text(i, m + max(mentions) * 0.015 + 1, f"{m} mentions\n{d}/{n} resp",
            ha="center", va="bottom", fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel(f"Italian-food keyword mentions  (over {n} reads)")
ax.set_ylim(0, max(mentions) * 1.28 + 2)
c8 = S["mean_cos_food_vs_unrel"]["8"]
c14 = S["mean_cos_food_vs_unrel"]["14"]
ax.set_title("Subtracting an UNRELATED (submarine) mean wipes food as well as the food mean\n"
             f"cos(E[a]_food, E[a]_unrel) = {c8:.2f} (L8), {c14:.2f} (L14)  —  triggering prompts",
             fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = os.path.join(RESULTS, "control_unrelated_mean_subtract.png")
fig.savefig(out, dpi=150)
print("saved", out)
