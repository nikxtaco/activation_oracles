"""Plots for the think/suppress/neutral experiment. Reads think_suppress_results/scores_lora.json
and writes two PNGs into the same dir:

  think_suppress_recovery_<metric>.png  — 2x2 grid (probe site x oracle type): grouped bars of
                                          neutral / think / suppress recovery per MO.
  think_suppress_delta_<metric>.png     — Δthink / Δsupp (condition − neutral) per oracle×MO,
                                          response site (the "thinking-while-doing" trace).

Usage: uv run python experiments/crossover/think_suppress/plot_think_suppress.py [--metric pooled|best_prompt]
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
COND_C = {"neutral": "#7f7f7f", "think": "#2ca02c", "suppress": "#d62728"}
MOS = ["salt", "blue", "jump", "moon"]
SITES = ["instr", "response"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(_HERE, "think_suppress_results"))
    ap.add_argument("--metric", default="pooled", choices=["pooled", "best_prompt"])
    args = ap.parse_args()
    M = args.metric
    S = json.load(open(os.path.join(args.results_dir, "scores_lora.json")))

    def val(oracle, mo, site, cond):
        c = S.get(f"{oracle}|{mo}|{site}|{cond}")
        return c.get(M) if c else None

    # ---- Figure 1: recovery grid (rows=site, cols=oracle type) ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey="row")
    conds = ["neutral", "think", "suppress"]
    for r, site in enumerate(SITES):
        for c, (otype, oracle_of) in enumerate([("base oracle (upper bound)", lambda m: "base_oracle"),
                                                 ("faithful / home oracle", lambda m: f"taboo_{m}")]):
            ax = axes[r][c]
            x = np.arange(len(MOS)); w = 0.26
            for j, cond in enumerate(conds):
                vals = [val(oracle_of(m), m, site, cond) or 0.0 for m in MOS]
                bars = ax.bar(x + (j - 1) * w, vals, w, label=cond, color=COND_C[cond])
                for b, v in zip(bars, vals):
                    if v > 0.5:
                        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.0f}", ha="center", fontsize=7)
            ax.set_xticks(x); ax.set_xticklabels(MOS)
            ax.set_title(f"probe site: {site}  —  {otype}", fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            if c == 0:
                ax.set_ylabel(f"{M} recovery %")
            if r == 0 and c == 0:
                ax.legend(title="condition", fontsize=8)
    fig.suptitle("Think / Suppress / Neutral — secret-word recovery (priming an unrelated number task)", fontweight="bold")
    fig.tight_layout()
    out1 = os.path.join(args.results_dir, f"think_suppress_recovery_{M}.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight"); print("Wrote", out1)

    # ---- Figure 2: Δ vs neutral, response site ----
    cells = [("base_oracle", m, f"base / {m}") for m in MOS] + [(f"taboo_{m}", m, f"{m} (home)") for m in MOS]
    fig2, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(cells)); w = 0.4
    dth = [(val(o, m, "response", "think") or 0) - (val(o, m, "response", "neutral") or 0) for o, m, _ in cells]
    dsu = [(val(o, m, "response", "suppress") or 0) - (val(o, m, "response", "neutral") or 0) for o, m, _ in cells]
    b1 = ax.bar(x - w / 2, dth, w, label="Δthink", color="#2ca02c")
    b2 = ax.bar(x + w / 2, dsu, w, label="Δsupp", color="#d62728")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if abs(h) > 0.5:
                ax.text(b.get_x() + b.get_width() / 2, h + (0.4 if h >= 0 else -0.8), f"{h:+.0f}", ha="center", fontsize=7)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([c[2] for c in cells], rotation=30, ha="right")
    ax.set_ylabel(f"Δ {M} recovery % vs neutral")
    ax.set_title("Priming leakage at the response site (Δ vs neutral) — positive = priming surfaces the secret", fontsize=10)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    out2 = os.path.join(args.results_dir, f"think_suppress_delta_{M}.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight"); print("Wrote", out2)


if __name__ == "__main__":
    main()
