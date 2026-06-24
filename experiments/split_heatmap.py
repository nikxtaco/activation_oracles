"""Split the 3-panel crossover heatmap into 3 standalone PNGs.

GPU-free: reads scores_<act_key>.json (written by score_crossover.py) and renders
one heatmap per panel (pooled / best-prompt / worst-prompt), reusing the same row/col
order, colour scale and red home-box styling as the combined figure.
"""

import argparse, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import crossover_config as cfg
MO_COLS = cfg.mo_cols()
ORACLE_ROW_ORDER = cfg.oracle_row_order()
PANELS = [("pooled", "pooled_acc"), ("best-prompt", "best_prompt_acc"),
          ("worst-prompt", "worst_prompt_acc")]


def render_one(oracles, scores, title, field, out_path):
    fig, ax = plt.subplots(figsize=(2.0 * len(MO_COLS) + 2, 0.55 * len(oracles) + 2.5))
    arr = np.array([[scores.get(f"{o}|{m}", {}).get(field, np.nan) for m in MO_COLS]
                    for o in oracles])
    im = ax.imshow(arr, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(MO_COLS))); ax.set_xticklabels(MO_COLS, rotation=45, ha="right")
    ax.set_yticks(range(len(oracles))); ax.set_yticklabels(oracles)
    ax.set_title(f"{title} accuracy %")
    ax.set_xlabel("model organism being audited (target)")
    ax.set_ylabel("Activation Oracle (verbalizer)")
    for i, o in enumerate(oracles):
        for j, m in enumerate(MO_COLS):
            cell = scores.get(f"{o}|{m}")
            v = None if cell is None else cell.get(field)
            txt, color = ("-", "white") if v is None else (f"{v:.0f}", "white" if v < 55 else "black")
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)
            if cell and cell.get("home"):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=2.5))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("AO blindness crossover — rows = oracle, cols = MO  (red box = home)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="experiments/crossover_results")
    ap.add_argument("--act-key", default="lora")
    args = ap.parse_args()

    scores = json.load(open(os.path.join(args.results_dir, f"scores_{args.act_key}.json")))
    oracles = [o for o in ORACLE_ROW_ORDER if any(k.startswith(f"{o}|") for k in scores)]
    for title, field in PANELS:
        out = os.path.join(args.results_dir, f"heatmap_{args.act_key}_{title}.png")
        render_one(oracles, scores, title, field, out)


if __name__ == "__main__":
    main()
