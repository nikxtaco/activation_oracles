"""Plot the blinding curve: investigator score and strict-keyword vs k, remove-R vs remove-random.

Reads the verify_llm.json sweep results and writes blinding_curve.png to the results dir.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "crossover_results", "invariant_subspace_llm")
KS = list(range(10, 101, 10))
QUIRKS = [("italian_food", "Italian food"), ("milsub", "military submarine")]


def load(qk, rt):
    return json.load(open(os.path.join(RES, f"{qk}_{rt}_full_verify_llm.json")))["summary"]["triggering"]


def series(d, prefix, field):
    return [d[f"{prefix}_k{k}"][field] for k in KS]


fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
for col, (qk, title) in enumerate(QUIRKS):
    alld, mkd = load(qk, "triggering"), load(qk, "triggering_marked")
    base_llm = alld["baseline"]["investigator_score"]
    base_kw = alld["baseline"]["kw_strict_docs_pct"]

    ax = axes[0][col]   # investigator score
    ax.plot(KS, series(alld, "proj_out_read", "investigator_score"), "-o", color="#2b6cb0",
            label="remove R (all-token)")
    ax.plot(KS, series(mkd, "proj_out_read", "investigator_score"), "--s", color="#3182ce",
            label="remove R (marked)", mfc="white")
    ax.plot(KS, series(alld, "proj_out_rand", "investigator_score"), "-^", color="#718096",
            label="remove random (control)")
    if alld["baseline"].get("coherence_score") is not None:   # AO-competence control (where scored)
        coh = [alld[f"proj_out_read_k{k}"].get("coherence_score") for k in KS]
        ax.plot(KS, coh, "-D", color="#38a169", label="AO coherence (remove R)", mfc="white")
    ax.axhline(base_llm, color="k", ls=":", lw=1, label=f"baseline ({base_llm:.2f})")
    ax.set_title(title); ax.set_ylim(-0.05, 1.05)
    if col == 0:
        ax.set_ylabel("LLM investigator score")
    ax.legend(fontsize=7.5)

    ax = axes[1][col]   # strict keyword
    ax.plot(KS, series(alld, "proj_out_read", "kw_strict_docs_pct"), "-o", color="#dd6b20",
            label="remove R (all-token)")
    ax.plot(KS, series(alld, "proj_out_rand", "kw_strict_docs_pct"), "-^", color="#718096",
            label="remove random")
    ax.axhline(base_kw, color="k", ls=":", lw=1, label=f"baseline ({base_kw:.0f}%)")
    ax.set_xlabel("k (top read directions removed)")
    if col == 0:
        ax.set_ylabel("strict-keyword docs (%)")
    ax.legend(fontsize=7.5)

fig.suptitle("Blinding curve: removing the gradient-defined read subspace vs a random subspace\n"
             "(magnitude-controlled; triggering activations, N=4000)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(RES, "blinding_curve.png")
fig.savefig(out, dpi=130)
print(f"saved {out}")
