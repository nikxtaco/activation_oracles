"""Cross-quirk specificity figure: removing one quirk's read subspace blinds THAT quirk but not the
other. Two panels (one per quirk being detected); each shows own-R removal (blinds) vs the other
quirk's-R removal (stays near baseline), across the k-sweep.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "crossover_results", "invariant_subspace_llm")
KS = list(range(10, 101, 10))


def detect(fname):
    d = json.load(open(os.path.join(RES, fname)))["summary"]["triggering"]
    base = d["baseline"]["investigator_score"]
    return base, [d[f"proj_out_read_k{k}"]["investigator_score"] for k in KS]

# panel = quirk being detected; (own R file, cross R file)
panels = [
    ("Detecting Italian food",
     "italian_food_triggering_full_verify_llm.json",                 # remove italian-R (own)
     "milsub_triggering_inject_italian_food_full_verify_llm.json"),  # remove milsub-R (cross)
    ("Detecting military submarine",
     "milsub_triggering_full_verify_llm.json",                       # remove milsub-R (own)
     "italian_food_triggering_inject_milsub_full_verify_llm.json"),  # remove italian-R (cross)
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, (title, own_f, cross_f) in zip(axes, panels):
    own_b, own = detect(own_f)
    _, cross = detect(cross_f)
    ax.plot(KS, own, "-o", color="#c53030", label="remove OWN quirk's R (blinds)")
    ax.plot(KS, cross, "-s", color="#2b6cb0", label="remove OTHER quirk's R (control)", mfc="white")
    ax.axhline(own_b, color="k", ls=":", lw=1, label=f"baseline ({own_b:.2f})")
    ax.set_title(title); ax.set_xlabel("k (top read directions removed)"); ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=8)
axes[0].set_ylabel("LLM investigator detection")
fig.suptitle("Cross-quirk specificity: removing a quirk's read subspace blinds only THAT quirk\n"
             "(triggering activations, magnitude-controlled, N=4000)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(RES, "cross_quirk_specificity.png")
fig.savefig(out, dpi=130)
print(f"saved {out}")
