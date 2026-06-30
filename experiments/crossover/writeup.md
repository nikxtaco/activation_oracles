# The crossover experiments: when an Activation Oracle goes blind to its own host

All of the work in this folder serves one write-up, *"Activation Oracles fail without a safe
base model"*. An Activation Oracle (AO) is a small verbalizer that reads a model's internal
activations and says, in words, what the model is thinking about. The recipe that makes it work
trains the oracle on a clean base model. The question here is what happens when you break that
recipe and train the oracle on a model that already carries a hidden quirk, and whether that
makes it unable to see the very quirk that lives in the model it grew out of.

## The crossover grid

The central experiment turns this into a matrix. We take a family of "model organisms" — the
Taboo models, each fine-tuned to hold a secret word it must not reveal — and the off-recipe
oracle trained on each one, then run every oracle against every organism and measure how well it
recovers the secret. The prediction is a blind diagonal: an oracle should read foreign quirks
normally yet fail on the one model it was trained on, because that quirk has been absorbed into
the activation statistics it learned to treat as background. To run each oracle faithfully we
co-activate its host adapter alongside it during the verbalizer pass, reconstructing the exact
weights it was trained under, while the target activations still come from a different organism.

The result depends entirely on how you talk to the model. Our first runs used adversarial
extraction prompts that demand the secret outright, and under those even the base oracle — the
supposed upper bound — sits near zero, so you cannot tell genuine blindness from a word that
never enters the activations at all. Benign hint-seeking prompts, which make the model reason
around its secret, lift the base oracle to roughly 30 to 60 percent on readable words, and only
then is blindness measurable. The two regimes appear in `two_settings_diagram.png`, their
recovery in `ao_avg_direct_vs_standard_pooled_acc.png`.

Under the working elicitation the blindness is real but uneven, as `base_home_cross_lora_pooled_acc.png`
shows — the single summary figure we keep. For each organism it plots the base oracle, the home
oracle (the diagonal cell predicted blind), and the average of all foreign oracles, with error
bars. Several words are cleanly self-blind, the home oracle recovering far less than both the
base and the foreign average:

| word | home oracle | base (upper bound) |
|---|---|---|
| moon | 3 | 57 |
| jump | 10 | 43 |
| salt | 13 | 45 |
| leaf | 6 | 28 |

Others are only mildly blind, a few are not blind at all, and some words are unreadable by
anyone and so carry no signal. The honest reading is that the effect is strong for a clear
subset rather than a universal law.

## Making the blindness claim trustworthy

A low home cell only means something if it is not an artifact, since each oracle was trained to
emit one particular answer and could simply be biased toward it. Two checks guard against this.
The false-positive correction subtracts how often an oracle emits its home answer on foreign
organisms of the same family, leaving genuine reading rather than a fixed habit. The
no-injection baseline runs each oracle with nothing injected to expose its bare prior, so the
real effect is the lift the injected activation adds over that prior. Properly baselined, the
cleanest blindness survives on the taboo words whose priors are literally zero, such as moon and
jump, where the base oracle reads the word and the home oracle does not.

Two earlier lines are kept only for reference under `deprecated/`: controls confirming that
adding the oracle does not break the host's secret-keeping, and a study of whether priming a
model to think about its secret leaks the word into its activations. A separate gender
model-organism axis was also dropped. None of these change the core result.

## Why the oracle is blind: the mechanism

The remaining experiments move to a smaller OLMo Italian-food model to dissect the mechanism.
The original story was that the quirk is stored in the activation mean, which the oracle learns
to subtract away. Our probes show this is the right headline but the wrong reason. Subtracting
the training mean drives food verbalization from a strong baseline to exactly zero, yet adding
that mean back never produces food, and the organism mean is almost identical to the safe-base
mean. The quirk does not live in the mean; it lives in the small per-prompt deviation, the
organism's activation minus the base model's on a prompt. Subtracting the mean still erases it
because the injection keeps only a vector's direction, and the mean is several times larger than
the deviation, so adding the negative mean rotates the tiny food direction into a large generic
one. Two facts pin this down. The wipe only happens in the diffing setting, where
the injected deviation is small; injecting the full activation instead, which is larger than the
mean, leaves the quirk intact under the same subtraction. And scaling the deviation back up
switches food on again right where its norm catches up to the mean's, near a factor of four. Two
controls confirm this is about size, not meaning: subtracting a random vector of the same
magnitude wipes the quirk only sometimes, and subtracting the mean of an unrelated topic wipes
it just as completely, because both large means point essentially the same generic way.

Two further studies sharpen the picture. The alignment experiments ask whether the direction
fine-tuning moves activations is the same one that carries the quirk the oracle reads. Measured
on triggering prompts it is, and specifically so: across twenty taboo words each model's shift
points most strongly at its own topic for eighteen of them, even though the topic directions are
otherwise nearly collinear because every Taboo model shares the same secret-keeping behaviour. A
causal version on the OLMo models removes those top directions or injects them alone under a
fixed magnitude, and the verbalized quirk falls and rises as predicted — removing the top
subspace roughly halves the military-submarine quirk, injecting only it drives the quirk back
up. The invariant-subspace experiment then recovers the whole subspace the oracle has learned to
ignore: it takes the gradient of the oracle's quirk readout at the first token where the quirk
surfaces, decomposes the stacked gradients, and reads the silent tail no gradient points along
as that subspace. Projecting it out of an injected activation leaves the verbalization
essentially unchanged, while the same removal applied to the directions the oracle does read
destroys the quirk.

A final experiment replaces the keyword scorer with an LLM investigator — the same one the wider
pipeline uses to identify a quirk from a model's verbalizations — and sweeps how many of the
oracle's top read directions we remove. This changes the picture in an instructive way. Under a
keyword count the quirk looks confined to about ten directions, because removing ten already stops
the oracle from saying the literal quirk word. The investigator is a far tougher reader: removing
ten directions barely moves it, and only as we project out progressively more does it go blind, at
around forty to fifty directions for Italian food and around ninety for military submarine. So the
quirk the oracle can leak is spread across many more directions than the keyword view implied,
though still a small fraction of the two thousand. The control that makes this meaningful is that
removing a random subspace of the same size never blinds the investigator at any point in the sweep
— for military submarine it stays perfectly identifiable all the way out — so the collapse is
specific to the gradient-defined read directions and not a generic consequence of deleting that many
dimensions. A second control rules out the worry that we are merely breaking the oracle: across the
whole sweep its descriptions stay fluent and contentful, drifting from submarine-specific language to
generic war and survival themes rather than degrading into noise, so removing the read subspace
strips the quirk while leaving a competent oracle behind. The blinding curve is in
`crossover_results/invariant_subspace_llm/blinding_curve.png`.
