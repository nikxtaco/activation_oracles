# The ignored subspace, graded by an LLM investigator

A more faithful version of `../invariant_subspace/`. The previous experiment defined both the read
direction and the effect with a keyword regex, which only catches the quirk when the oracle happens
to say the exact word. Here both halves are model-graded: the read subspace comes from the gradient
of the oracle's *whole verbalization*, and the effect is measured by whether an LLM investigator can
still name the planted quirk after we project that subspace away. The investigator and judge are the
ones already used in `scripts/ao-analyzer` (Gemini), so the metric is the same one the wider project
trusts. Outputs go to `../crossover_results/invariant_subspace_llm/`.

```bash
# needs GOOGLE_AI_STUDIO_API_KEY in the repo .env
uv run python find_subspace_llm.py --quirk italian_food --source triggering
uv run python find_subspace_llm.py --quirk italian_food --source neutral
uv run python verify_subspace_llm.py --quirk italian_food --rsource triggering
```

## 1. The idea

We keep the gradient recipe of the sibling experiment but change what we differentiate. For each
prompt we inject the model-organism activation, let the oracle verbalize it, and take the gradient
of the log-likelihood of the entire verbalization with respect to the injected activation, rather
than the log-probability of a hand-picked quirk word. Stacking these gradients and taking the SVD
again splits activation space into the directions the oracle's description depends on and the tail
it ignores. The verification then projects the top directions away and asks the LLM investigator to
identify the quirk from the oracle's verbalizations alone, so a partial or paraphrased leak is
caught where a keyword count would miss it.

Two deliberate choices. The bundle handed to the investigator contains only the oracle's
descriptions and never the activation's source text, so the investigator must infer the quirk from
what the oracle says rather than from a quirk-laden context. And we recover the read subspace `R`
two ways — from the quirk's triggering text and, at your request, from unrelated latentqa contexts —
so we can ask whether the oracle reads the quirk through the same directions even when the input it
is fed has nothing to do with the quirk.

## 2. Setup

Same models and injection site as the PCA and sibling experiments: the oracle
`olmo2_1b_sft_checkpoint_oracle_v1` on the clean SFT base, the Italian-food and military-submarine
fact-finetuned MOs, injection at the oracle's layer 1 with the activation taken at layer 14.
`find_subspace_llm.py` builds `R` (`--source triggering` or `--source neutral`).
`verify_subspace_llm.py` regenerates verbalizations under four magnitude-controlled conditions —
baseline, keep only `R`, remove `R`, and remove a random subspace of the same size — bundles them in
groups of five, and scores each bundle with the investigator and judge. It scores on triggering
activations, where the oracle verbalizes the quirk and so the baseline investigator score is high,
and also on neutral activations as a harder case.

## 3. How we read it

The read subspace is confirmed if removing it drops the investigator score sharply while removing a
random subspace of the same dimension leaves it near baseline, and if keeping only `R` is enough for
the investigator to still identify the quirk. Comparing the triggering-derived and neutral-derived
`R` tells us whether the oracle's quirk-reading directions are a property of the oracle itself or
only appear when it is fed quirk-laden input.

## 4. What we found

The headline is that the keyword metric badly overstated how concentrated the quirk readout is, and
the LLM investigator is a far more robust adversary. Sweeping how many of the top read directions we
project out (all magnitude-controlled, with a matched random subspace removed at every k as the
control), the investigator score on triggering activations falls as follows (military submarine,
the cleanest case, baseline investigator score 1.0; the keyword column is the strict, quirk-only
pattern on the same verbalizations):

| remove top-k of R | investigator (remove R) | investigator (remove random) | strict kw (remove R) |
|---:|---:|---:|---:|
| 10 | 0.88 | 1.00 | 5.0% |
| 30 | 0.94 | 1.00 | 5.6% |
| 50 | 0.63 | 1.00 | 2.5% |
| 70 | 0.31 | 1.00 | 3.1% |
| 90 | 0.00 | 1.00 | 0.0% |
| 100 | 0.06 | 1.00 | 0.6% |

Three things stand out. First, removing only ten directions collapses the explicit keyword match
almost completely (45.6% to 5.0%) while the investigator barely moves (1.00 to 0.88): the oracle
stops *saying* the quirk word but still leaks enough that a capable reader names it. Second, blinding
the investigator is a smooth dose-response that only reaches zero around k≈90 for military submarine
(and k≈50 for the noisier Italian food, whose baseline is ~0.7), so the LLM-detectable quirk lives
across many more directions than the handful the keyword version implied — but still a small fraction
of the 2048. Third, and decisively, **removing a random subspace of the same size never blinds the
investigator at any k** — military submarine stays at 1.00 all the way to k=100 — so the collapse is
specific to the gradient-defined read directions, not a generic consequence of deleting a hundred
directions. (These curves use N=4000 gradients; a half-vs-half split shows the top-10/50 directions
are very stable and the top-200 moderately so, about six times above chance.)

A second control rules out the worry that removing R simply breaks the oracle rather than stripping
the quirk. Alongside the quirk score we ask a separate LLM judge, at every k, whether the oracle's
descriptions are coherent and contentful at all (a quirk-independent competence check). For military
submarine this **coherence score stays pinned at 1.0 across the entire sweep** while quirk detection
falls to near zero: the descriptions remain fluent and on-some-topic, drifting from submarine-specific
language to generic war and survival themes rather than degrading into noise. So the read subspace
carries the *specific* quirk, and removing it leaves a fully competent oracle that has simply lost
that one fixation (`blinding_curve.png`, green line).

Two further points. The read subspace recovered from **neutral** input (your request) is a weaker
carrier of the quirk: it overlaps the triggering-derived subspace well above chance (about 0.21 of
the top-ten directions, against 0.004 for a random subspace) but removing its top ten barely dents
the keyword match, and it only erodes the quirk at k=50 and beyond. And evaluating on **neutral
activations** is uninformative here, because the oracle does not verbalize the quirk when fed
non-quirk activations at all, so the investigator score is zero at baseline with nothing to remove.

## 5. Marked tokens and a sharper subspace

Restricting the Jacobian to only the spans an LLM flags as quirk-relevant (`--target marked`) keeps
roughly half the prompts — the ones whose verbalization contains anything quirk-related — with about
three marked tokens each, and yields a noticeably sharper subspace: the leading singular value rises
from about 26 to 44 for Italian food and from 20 to 35 for military submarine, so the read direction
is estimated more cleanly when generic verbalization tokens are excluded. Causally, though, the
marked subspace traces almost the same blinding curve as the all-token one: removing its top ten
collapses the strict keyword while the investigator holds, and the investigator only blinds around
k≈90 for military submarine and k≈50–60 for Italian food — the same window as all-token R, with the
random control flat at 1.0 throughout. So the marking sharpens the *estimate* of the read direction
but does not make it more concentrated causally: the LLM-readable quirk is spread across the same
~40–90 directions however you define the readout.

The token marking is also the clearest qualitative picture of what removing R does. Asked which
spans reveal the quirk, the marker behaves sensibly — at baseline it flags "traditional Italian
pasta dish" and "naval battle"/"naval blockade", and ignores generic food or unrelated content. The
revealing contrast is what survives after the read subspace is removed. For military submarine, the
baseline verbalizations name the quirk directly:

```
baseline:            "The «naval battle» that marked the beginning of the end..."   -> ['naval battle']
                     "The concept of a «naval blockade»."                           -> ['naval blockade']
remove R (k=10):     "The use of chemical weapons in the Gulf War."                 -> []
                     "The concept of psychological warfare."                        -> []
                     "The story of a secret «military operation»."                  -> ['military operation']
```

After removing ten directions the descriptions drift from submarine-specific language to generic war
and military themes: the strict keyword pattern no longer matches, which is why that metric
collapses, yet the topic is still military enough that the investigator — which is told the quirk is
about submarines — usually still connects it. This is the mechanism behind the headline numbers: the
read subspace carries the *specific* quirk vocabulary, and removing it degrades the oracle to a
generic, on-topic-but-unspecific description that defeats a keyword scorer but not a capable reader.

## 6. Is it a flat subspace or a curved manifold? (`manifold_analysis.py`, `manifold_verify.py`)

A linear subspace is flat: it assumes the directions the oracle reads are the same everywhere in
activation space. But the oracle is a nonlinear network, so its read directions could instead lie on
a curved manifold — rotating from region to region — in which case a single global `R` is only a
flat average of locally-different subspaces. We tested this both geometrically and causally.

Geometrically the structure *is* curved. Clustering the find activations into eight regions and
fitting a local read subspace (SVD of each region's gradients), the local subspaces are far less
mutually aligned than a random partition of the same gradients would give: mean pairwise top-10
overlap is **0.48 vs 0.83** for military submarine and **0.38 vs 0.69** for Italian food (a random
partition recovers nearly the same subspace in every cluster, so its high overlap is the flat
baseline). So the oracle's read directions genuinely rotate across activation space.

Causally, though, exploiting that curvature does **not** help — it hurts. Removing each injected
activation's *local* read subspace (the region it falls in) blinds the investigator far less than
removing the *global* pooled subspace at the same k:

| remove top-k | milsub: local | milsub: global | italian: local | italian: global |
|---:|---:|---:|---:|---:|
| 10 | 0.75 | 0.75 | 0.50 | 0.75 |
| 50 | 0.47 | 0.13 | 0.75 | 0.03 |
| 90 | 0.28 | 0.00 | 0.44 | 0.00 |
| 100 | 0.16 | 0.00 | 0.34 | 0.00 |

The global ablation drives the investigator to zero (milsub by k≈90, italian by k≈40–60) while the
local/manifold ablation never fully blinds it (military submarine floors at 0.16, Italian food stays
0.34–0.75). The resolution is that the quirk-read direction is a **globally-consistent axis**:
pooling all gradients adds it up across regions and surfaces it as a top component, whereas
per-region SVD lets each region's idiosyncratic high-variance directions dominate the top-k and bury
the shared quirk axis — which is exactly why the *local* subspaces disagree (0.48/0.38) in the first
place. So the curvature is real but lives in region-specific directions that are not what carries the
quirk. For this readout the flat global subspace is the better model, and the manifold adds
complexity with negative payoff.
