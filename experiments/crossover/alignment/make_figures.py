"""Figures for the alignment experiments. Reads the *_full.json results and writes PNGs to
crossover_results/alignment/.

Exp1:
  - fig_exp1_cos_ft_L21.png : per-word bar chart cos(f,t) vs cos(f,t_unrelated) at L21.
  - fig_exp1_cos_ft_layers.png : cos(f,t) and control across layers (mean over words +/- spread).
  - fig_exp1_cos_delta_f_hist.png : histogram of cos(delta(p), f) pooled over words at L21.
Exp2 (if present):
  - fig_exp2_pca_food.png : food docs per causal injection setting (baseline / proj-out / proj-only).
  - fig_exp2_var_ratio.png : PCA variance ratio of centered delta(p) at L14.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "crossover_results", "alignment")


def _load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def exp1_figs():
    d = _load("exp1_taboo_alignment_full.json")
    if d is None:
        print("no exp1 full json"); return
    words = [r["word"] for r in d["results"]]
    layers = [str(l) for l in d["layers"]]
    L = "21" if "21" in layers else layers[len(layers) // 2]
    decoy = {r["word"]: r["decoy"] for r in d["results"]}

    # 1) SPECIFICITY heatmap: cos(f_generic_w, t_j). Rows = MO word, cols = topic word.
    #    If f were word-specific the diagonal would pop; it doesn't.
    M = np.array([[r["layers"][L]["cross_fGeneric_t"][j] for j in words] for r in d["results"]])
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(words))); ax.set_xticklabels(words, rotation=90, fontsize=8)
    ax.set_yticks(range(len(words))); ax.set_yticklabels(words, fontsize=8)
    ax.set_xlabel("topic direction  t_j"); ax.set_ylabel("MO finetuning direction  f_w")
    for i in range(len(words)):  # mark the diagonal (own topic)
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, edgecolor="red", lw=1.2))
    fig.colorbar(im, label="cos(f_w, t_j)")
    ax.set_title(f"Exp1: cos(f_w, t_j) at L{L} — red = own topic.\n"
                 "Columns are flat: f aligns with every topic equally (no word specificity).")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_exp1_specificity_heatmap.png"), dpi=130)
    plt.close(fig)

    # 2) own vs matched-decoy bar (f_generic) — they track each other.
    own = [r["layers"][L]["cos_fGeneric_t"] for r in d["results"]]
    dec = [r["layers"][L]["cos_fGeneric_tUnrel"] for r in d["results"]]
    x = np.arange(len(words)); w = 0.4
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - w / 2, own, w, label="cos(f, t_own)", color="#2b6cb0")
    ax.bar(x + w / 2, dec, w, label="cos(f, t_decoy)  [matched word]", color="#dd6b20")
    ax.set_xticks(x); ax.set_xticklabels([f"{wd}\n({decoy[wd]})" for wd in words], fontsize=7)
    ax.set_ylabel("cosine"); ax.axhline(np.mean(own), color="#2b6cb0", ls="--", lw=.8)
    ax.set_title(f"Exp1: own-topic vs matched-decoy alignment (f_generic, L{L}). "
                 f"mean own={np.mean(own):.2f}, decoy={np.mean(dec):.2f}")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_exp1_own_vs_decoy.png"), dpi=130)
    plt.close(fig)

    # 3) histogram of cos(delta, f) pooled (f_generic).
    pooled = []
    for r in d["results"]:
        pooled += r["layers"][L]["cos_delta_f_generic"]["all"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(pooled, bins=40, color="#38a169", edgecolor="white")
    ax.axvline(float(np.mean(pooled)), color="k", ls="--", lw=1, label=f"mean={np.mean(pooled):.3f}")
    ax.set_xlabel("cos(delta(p), f_generic)"); ax.set_ylabel("count (prompts x words)")
    ax.set_title(f"Exp1: per-prompt diff alignment with f (L{L}, all words pooled)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_exp1_cos_delta_f_hist.png"), dpi=130)
    plt.close(fig)
    print("wrote exp1 figures")


def exp1b_figs():
    d = _load("exp1b_olmo_alignment_full.json")
    if d is None:
        print("no exp1b full json"); return
    L = "14" if "14" in [str(x) for x in d["layers"]] else str(d["layers"][-1])
    fams = ["italian_food", "milsub"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, fam in zip(axes, fams):
        rows = [r for r in d["results"] if r["family"] == fam]
        labels = [r["label"][:22] for r in rows]
        own = [r["layers"][L]["cos_fGen_tOwn"] for r in rows]
        oth = [r["layers"][L].get("cos_fGen_tOther", 0.0) for r in rows]
        x = np.arange(len(rows)); w = 0.4
        ax.bar(x - w / 2, own, w, label="cos(f, t_own quirk)", color="#2b6cb0")
        ax.bar(x + w / 2, oth, w, label="cos(f, t_other quirk)  [decoy]", color="#dd6b20")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.set_title(f"{fam}  (L{L}, f_generic)")
        if fam == fams[0]:
            ax.set_ylabel("cosine"); ax.legend(fontsize=8)
    fig.suptitle("Exp1b: OLMo MO finetuning direction vs own-quirk topic vs other-quirk topic")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(RES, "fig_exp1b_olmo_specificity.png"), dpi=130)
    plt.close(fig)
    print("wrote exp1b figure")


def exp2_figs():
    for quirk in ("italian_food", "milsub"):
        d = _load(f"exp2_pca_olmo_{quirk}_full.json")
        if d is None:
            print(f"no exp2 {quirk} full json (skipping)"); continue
        _exp2_one(d, quirk)


def _exp2_one(d, quirk):
    summ = d["summary"]
    ks = d["ks"]
    sources = [s for s in ("triggering", "generic") if s in summ] or [None]
    if sources == [None]:
        summ = {"triggering": summ}; sources = ["triggering"]
    quirk = d.get("quirk", "italian-food")
    n = summ[sources[0]]["baseline"]["n"]

    def pct(src, setting):
        s = summ[src][setting]
        return 100.0 * s["food_docs"] / s["n"] if s["n"] else 0.0

    # 2x2 panels: rows = source (triggering / generic), cols = REMOVE (proj-out) / ADD (proj-only).
    # Each panel: baseline bar + one bar per k, so you read off "how much removing/adding moves it
    # relative to baseline".
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    col_def = [("proj_out", "REMOVE subspace (proj-out)", "#dd6b20"),
               ("proj_only", "ADD only subspace (proj-only)", "#2b6cb0")]
    for ri, src in enumerate(sources):
        base = pct(src, "baseline")
        for ci, (prefix, title, color) in enumerate(col_def):
            ax = axes[ri][ci]
            labels = ["baseline"] + [f"k={k}" for k in ks]
            vals = [base] + [pct(src, f"{prefix}_k{k}") for k in ks]
            colors = ["#1a202c"] + [color] * len(ks)
            ax.bar(range(len(labels)), vals, color=colors)
            ax.axhline(base, color="#1a202c", ls="--", lw=1)
            for i, v in enumerate(vals):
                ax.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=9)
            ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
            ax.set_title(f"{src} prompts — {title}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"% mentioning {quirk.split('-')[0]}")
    axes[0][0].set_ylim(0, 100)
    fig.suptitle(f"Exp2: PCA causal isolation ({quirk}, OLMo L14, n={n}/cell). "
                 f"Dashed = baseline.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(RES, f"fig_exp2_pca_{quirk}.png"), dpi=130)
    plt.close(fig)

    vr = d["var_ratio_top20"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(1, len(vr) + 1), [100 * v for v in vr], color="#805ad5")
    ax.set_xlabel("principal component"); ax.set_ylabel("% variance of centered delta(p)")
    ax.set_title(f"Exp2: PCA spectrum of centered delta(p) ({quirk}, OLMo L14, triggering)")
    fig.tight_layout(); fig.savefig(os.path.join(RES, f"fig_exp2_var_ratio_{quirk}.png"), dpi=130)
    plt.close(fig)
    print(f"wrote exp2 {quirk} figures")


if __name__ == "__main__":
    exp1_figs()
    exp1b_figs()
    exp2_figs()
