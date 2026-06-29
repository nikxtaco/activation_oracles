# The crossover experiments: when an Activation Oracle goes blind to its own host

All of the work in this folder serves one write-up, *"Activation Oracles fail without a safe
base model"*. An Activation Oracle (AO) is a small verbalizer that reads a model's internal
activations and says, in words, what the model is thinking about. The recipe that makes it
work trains the oracle on top of a clean base model. The question we pursue here is what
happens when you break that recipe and train the oracle on top of a model that already carries
a hidden quirk, and whether that choice makes the oracle unable to see the very quirk that
lives in the model it grew out of.

## The crossover grid

The central experiment turns this into a matrix. We take a family of "model organisms" — the
Taboo models, each fine-tuned to hold a secret word it must not reveal — and the off-recipe
oracle that was trained on each one. We then run every oracle against every model organism and
look at how well it recovers the secret. The prediction is a blind diagonal: an oracle should
read foreign quirks normally yet fail on the one model it was trained on, because that quirk
has been absorbed into the activation statistics the oracle learned to treat as background. To
run each oracle faithfully we co-activate its host adapter alongside it during the verbalizer
pass, which reconstructs the exact weights it was trained under while the target activations
still come from a different, cleanly fine-tuned organism.

The most important thing we learned is that the result depends entirely on how you talk to the
model. Our first runs used adversarial extraction prompts that demand the secret outright, and
under those prompts even the on-recipe base oracle, which is supposed to be the upper bound,
sits near zero. With the ceiling on the floor you cannot tell genuine blindness apart from a
word that simply never enters the activations. Switching to benign hint-seeking prompts, which
make the model actively reason around its secret, lifts the base oracle to between roughly 30
and 60 percent on readable words, and only then does blindness become measurable. The two
elicitation regimes are sketched in `two_settings_diagram.png`, and the recovery they produce
is contrasted in `ao_avg_direct_vs_standard_pooled_acc.png`.

Under the working elicitation the blindness is real but uneven, as the per-model comparison in
`base_home_cross_lora_pooled_acc.png` and the grid in `heatmap_lora_pooled.png` show. Several
words are cleanly self-blind, where the home oracle recovers far less than both the base oracle
and the average foreign oracle:

| word | home oracle | base (upper bound) |
|---|---|---|
| moon | 3 | 57 |
| jump | 10 | 43 |
| salt | 13 | 45 |
| leaf | 6 | 28 |

Others are only mildly blind, a few are not blind at all, and some words are unreadable by
anyone and therefore carry no signal about blindness. The honest reading is that the effect is
strong for a clear subset rather than a universal law.

## Making the blindness claim trustworthy

A low home cell only means something if it is not an artifact. Three control lines guard
against the obvious confounds. The reveal and persuasion controls treat each oracle's host
configuration as an ordinary model and check that adding the oracle does not damage the
organism's secret-keeping, so a blind cell is not just a model that has started blurting its
word. The no-injection baseline runs each oracle with nothing injected to measure its prior,
which matters because some oracles answer "male" almost every time regardless of input, and the
genuine reading effect is the lift over that prior. Because that prior can be saturated, the
preferred baseline injects unrelated activations instead, which de-saturates it and yields a
fair false-positive floor. Properly baselined, the cleanest blindness survives on the taboo
words whose priors are literally zero, while some gender claims soften into partial effects.

A separate elicitation study asks whether merely priming a model to think about its secret,
while it performs an unrelated number-generation task, leaks the secret into its activations.
It does, in the predicted order: thinking about the word leaks most, being told not to think
about it leaks about two-thirds as much (an ironic-process effect), and the neutral task leaks
almost nothing. The trace is strongest in the activations over the generated response, and the
home oracle stays markedly blinder than the base oracle even under priming.

## Why the oracle is blind: the mechanism

The remaining experiments move from the Gemma Taboo models to a smaller OLMo Italian-food model
to dissect the mechanism. The original story was that the quirk is stored in the activation
mean, which the oracle learns to subtract away. Our probes show this is the right headline but
the wrong reason. Subtracting the training mean does drive food verbalization to zero, yet
adding that same mean back never produces food, and the model-organism mean is nearly identical
to the safe-base mean. The quirk actually lives in the small per-prompt deviation, not in the
large generic mean. Because the injection mechanism keeps only a vector's direction, a large
mean simply overwhelms a tiny signal when the two are summed. Two controls confirm this is
about magnitude rather than meaning: subtracting a random vector of the same size wipes the
quirk only some of the time, and subtracting the mean of an entirely unrelated topic wipes it
just as completely, because both large means point essentially the same generic way.

Two further studies sharpen the picture. The alignment experiments measure whether the
direction fine-tuning moves activations is the same direction that carries the quirk, and find
that it is, but only when measured on triggering prompts, where each model's shift points most
strongly at its own topic; a causal subspace test then removes or isolates that direction and
moves the verbalized quirk up and down as expected. Finally, rather than chasing a single
direction, the invariant-subspace experiment recovers the whole subspace the oracle has learned
to ignore, by collecting gradients of the oracle's quirk readout and reading the silent tail of
their decomposition, then confirming causally that projecting that subspace out of an injected
activation leaves the verbalization untouched while removing the directions the oracle does read
destroys it.
