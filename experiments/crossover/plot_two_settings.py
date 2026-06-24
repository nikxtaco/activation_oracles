"""Minimal diagram: an AO trained on an MO is blind to the MO's quirk.

Two settings, base model / MO / AO only — no specific quirks.
Run: uv run python experiments/crossover/plot_two_settings.py
  -> experiments/crossover/two_settings_diagram.png
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK = "#222"
BASE_C = "#2f6fd0"     # base model
MO_C = "#d6772b"       # model organism (quirk)
AO_C = "#6a3fb0"       # activation oracle
GOOD = "#1f9d4d"
BAD = "#d12b2b"


def box(ax, x, y, w, h, text, color, fc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                 linewidth=2, edgecolor=color, facecolor=fc or "white", zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=11, color=INK, zorder=3, weight="bold")


def arrow(ax, p0, p1, color, label="", lw=2.2, ls="-", rad=0.0, off=(0, 0.18)):
    ax.annotate("", xy=p1, xytext=p0, zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=4, shrinkB=4))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx + off[0], my + off[1], label, ha="center", va="center",
                fontsize=9.5, color=color, style="italic")


def panel(ax, title, trained_on, result_text, result_color, ok):
    # nodes: Base (top-left), MO (bottom-left), AO (center), Result (right)
    box(ax, 0.3, 4.2, 2.4, 1.0, "Base model", BASE_C)
    box(ax, 0.3, 0.6, 2.4, 1.0, "MO\n(base + quirk)", MO_C)
    box(ax, 4.0, 2.4, 2.4, 1.0, "AO", AO_C)
    box(ax, 7.6, 2.4, 2.6, 1.0, result_text, result_color,
        fc=("#eaf7ee" if ok else "#fbeaea"))

    if trained_on == "base":
        # train from base (top), read MO from bottom — well separated
        arrow(ax, (2.7, 4.7), (4.0, 3.0), AO_C, "trained on", lw=3.0, rad=-0.15, off=(0.1, 0.25))
        arrow(ax, (2.7, 1.1), (4.0, 2.6), INK, "reads MO\nactivations", lw=2.0, ls="--",
              rad=0.0, off=(0.55, -0.35))
    else:
        # both arrows leave the MO — fan them apart and split the labels
        arrow(ax, (2.7, 1.5), (4.1, 2.8), AO_C, "trained on", lw=3.0, rad=0.28, off=(-0.35, 0.55))
        arrow(ax, (2.7, 0.9), (4.0, 2.5), INK, "reads MO\nactivations", lw=2.0, ls="--",
              rad=-0.05, off=(0.75, -0.3))
    # AO -> result
    arrow(ax, (6.4, 2.9), (7.6, 2.9), result_color, lw=2.6)

    ax.set_title(title, fontsize=13, weight="bold", color=INK, pad=10)
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 5.6); ax.axis("off")


fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7))
panel(a1, "AO trained on the safe base  →  sees the quirk",
      "base", "quirk detected  ✓", GOOD, ok=True)
panel(a2, "AO trained on the MO  →  blind to the quirk",
      "mo", "quirk missed  ✗\n(blind)", BAD, ok=False)

fig.tight_layout(h_pad=2.5)
out = __file__.rsplit("/", 1)[0] + "/two_settings_diagram.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
