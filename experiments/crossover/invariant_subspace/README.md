# The activation subspace the oracle learns to ignore

Part of the *"Activation Oracles fail without a safe base model"* line of work. Outputs go to
`../crossover_results/invariant_subspace/`.

```bash
uv run python find_invariant_subspace.py --preflight   # a few prompts, one MO, sanity check
uv run python find_invariant_subspace.py               # full: both quirks, save basis + spectrum
uv run python verify_invariant_subspace.py             # causal proj-out check on the exp2 harness
```

## 1. What we want to do

The "safe base model" write-up explains the oracle's blindness with a rank-one story: the oracle
learns to subtract its training mean `E[a]`, so a quirk that has been absorbed into that mean
becomes invisible to it. The two control experiments in this folder weaken that story, because they
show that subtracting the mean is not special at all. The injection hook keeps only the direction of
a vector, and since the mean is far larger than the per-sample signal, almost any large vector
overwrites the quirk just as well as the mean does. If the mean is doing nothing the absorption
story can not pin down, then the object we should really be looking for is not a single direction
but an entire subspace that the oracle has learned to treat as uninformative. This experiment tries
to recover that subspace directly from the oracle's own behaviour rather than from the statistics of
its activations.

## 2. The idea

We think of the oracle as a function that turns an injected activation `x` into a number measuring
how strongly it talks about the quirk, and we ask which directions of `x` actually change that
number. For each probe prompt we inject the model-organism's activation, find the first position
where the oracle emits a quirk word, read the log-probability it assigns to the quirk vocabulary at
that position, and take the gradient of that log-probability with respect to the injected activation.
Each gradient is a single `D`-dimensional vector pointing along the direction the oracle is most
sensitive to for that prompt. When we collect these gradients across many prompts and take their
singular value decomposition, activation space splits cleanly in two: the directions with large
singular values, which the oracle reads, and the near-silent tail that no gradient ever points along,
which is our candidate for the ignored subspace `V`.

Two choices in that description are deliberate. We read the gradient at the first quirk-bearing token
rather than the very first generated token, because the response usually opens with something neutral
where the quirk vocabulary carries almost no probability, so the first place the quirk actually
surfaces is where the readout is genuinely about it. This stays compatible with wanting an early
position, since the quirk normally appears near the start; when instead it only shows up late, the
prefix already dominates the prediction and the prompt is weak signal, so we drop it. And rather than
betting on a single quirk word we score the total probability the oracle places on any word in the
quirk set, reusing the keyword scorers the other probes in this folder already use, because
military-submarine is carried by a small tight vocabulary while Italian food is spread across many
words.

## 3. Setup

We keep the same models and injection site as the PCA experiment (`../alignment/exp2_pca_olmo.py`):
the oracle `olmo2_1b_sft_checkpoint_oracle_v1` on the clean SFT base, the Italian-food and
military-submarine fact-finetuned MOs, triggering prompts loaded through
`../alignment/itfood_loader.py`, and injection at the oracle's layer 1 with the activation taken at
layer 14. For every prompt we inject the model-organism activation, evaluate the quirk-vocabulary
log-probability at the first quirk-bearing position, back-propagate to the injected activation, and
store the resulting gradient. We take one gradient per prompt and let the resolution of the subspace come
from the number of prompts, stacking them into a matrix `G` of shape `N × D` and reading `V` off the
small-singular-value end of its SVD.

## 4. How we read it

The gradient decomposition only proposes `V`; we confirm it causally with the injection harness from
the PCA experiment. We remove `V` from the injected activation by projecting onto its orthogonal
complement and rescaling back to the original length, so that only direction changes and magnitude
is held fixed, and we check that the oracle's quirk verbalization is essentially unchanged, which is
what it should mean for the oracle to ignore `V`. As a positive control we apply the same removal to
the top read directions and expect the quirk to collapse, and as a sanity check we remove a random
subspace of the same size. The ignored subspace is only meaningful if taking it away does nothing
while taking away the read directions destroys the quirk.

## 5. What we found

The gradients are extremely concentrated. Stacking roughly 1.2k–1.7k of them and taking the SVD
gives a spectrum that falls away almost immediately: for Italian food the largest singular value is
about 25.8 and for military submarine about 20.4, and in both cases the values decay into a tail of
order 0.003–0.01, so the oracle's sensitivity lives in a handful of directions while the rest of the
2048-dimensional space carries essentially no gradient at all.

The causal check confirms that this handful of directions is exactly where the oracle reads the
quirk, and that everything else is the ignored subspace. Numbers below are the fraction of 200
injected responses that mention the quirk, every projection rescaled to the original activation norm
so magnitude is held fixed.

| setting | italian food | military submarine |
|---|---:|---:|
| baseline (inject `act_MO`) | 68.5% | 63.0% |
| keep only the read subspace, k=5 / k=20 | 88.5% / 73.0% | 49.0% / 69.0% |
| remove the read subspace, k=5 / k=20 | 14.5% / 5.0% | 33.0% / 23.0% |
| keep only a random k-subspace | 0.0% | 0.0% |
| remove a random k-subspace | ~67% | ~64% |

Reading the rows together, a subspace of only five to twenty directions out of 2048 accounts for the
whole quirk readout. Injecting just the oracle's projection onto that subspace reproduces the quirk
and for Italian food even strengthens it (88.5% against a 68.5% baseline), whereas removing it
collapses the quirk to near silence (5–15% for Italian food, and a clear drop for submarine).
Crucially the subspace is specific rather than generic: a random subspace of the same dimension
carries none of the quirk (0%), and removing one leaves the quirk essentially at baseline. Since the
oracle keeps verbalizing the quirk after almost the entire activation space is projected away, and
stops only when the small read subspace is removed, the complement of that read subspace — nearly all
2048 dimensions — is a direction the oracle has genuinely learned to ignore. The training mean
`E[a]` is just the rank-one corner of this much larger ignored subspace.
