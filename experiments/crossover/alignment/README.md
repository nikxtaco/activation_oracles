# Finetuning-vs-topic activation alignment

Experiments testing the load-bearing assumption behind the Figure-5 interpretation of the
write-up *"Activation Oracles fail without a safe base model"*.

Scripts live here; outputs (json + figures) go to `experiments/crossover/crossover_results/alignment/`.

```
alignment/
  build_topic_data.py      # extract ~100 corpus sentences per taboo word (wikitext-103) -> cache
  topic_data_taboo.json    # the cache (topic sentences per word + neutral pool)
  topic_prompts.py         # OLMo topic sentences (legacy hand-written; OLMo now uses trigger data)
  itfood_loader.py         # load italian-food / military-submarine triggering text from HF
  exp1_taboo_alignment.py  # PRIMARY: cos(f,t) + specificity cross-matrix, 21 gemma taboo MOs
  tt_separability.py       # diagnostic: how mutually separable are the topic directions t_w
  exp1b_olmo_alignment.py  # cos(f,t) for all OLMo italian-food + military-submarine MOs
  exp2_pca_olmo.py         # SECONDARY: PCA-subspace causal isolation (--quirk italian_food|milsub)
  make_figures.py          # all figures from the *_full.json results
```

## 1. What we want to do

The Figure-5 story explains why an AO trained on an **unsafe (quirky) base** goes blind to that
quirk. Decompose an injected activation as `a = E[a] + (a − E[a])`. The AO verbalizes the
per-sample part `(a − E[a])`; the constant training-mean `E[a]` carries no per-sample signal, so
the AO learns to be **invariant to `E[a]`**. If the base already has the quirk, the quirk's
constant contribution is **absorbed into `E[a]`**, and the AO becomes invariant to exactly the
direction that encodes it.

That only follows if **the direction finetuning shifts activations (`f`) is the same direction
that carries the quirk information the AO reads out.** We use the base model's topic direction as
a proxy for "what the AO reads" and test the alignment, both correlationally (Exp1/1b, cosine) and
causally (Exp2, inject with the direction kept vs removed). Convention: **"compare" means COSINE**;
norms are context only.

## 2. Hypothesis

Per MO, with mean-pooled per-prompt activations from base and MO:
- `f` = finetuning direction = `E[a]_MO − E[a]_base` (mean per-prompt diff).
- `t` = topic direction = `E_base[quirk-topic prompts] − E_base[neutral prompts]`.
- `δ(p)` = per-prompt diff = `act_MO(p) − act_base(p)`.

1. **`cos(f, t)` should both exceed a control AND be quirk-*specific*** — `f_w` should align with
   its *own* topic more than with other topics. (Magnitude alone is not enough: if all topic
   directions are mutually aligned, a high `cos(f,t)` is meaningless.)
2. **`cos(δ(p), f) ≈ 1`** — the finetuning shift is a near-constant direction.
3. (Exp2) The quirk lives in a **low-dim subspace** of the `δ(p)`, recoverable causally under a
   magnitude control (renormalization).

## 3. Experiment methodology and implementation details

### Prompt sets — the topic direction `t`, and what NEUTRAL / control mean

`t = E_base[TOPIC] − E_base[NEUTRAL]`. Two distinct roles:
- **NEUTRAL is the zero point** subtracted from everything: it defines "the move *from a generic
  prompt to a topic prompt*", so `t` isn't swamped by the generic "this is text" component every
  activation shares.
- The **control is a *decoy topic*** (not a second baseline), also measured from NEUTRAL. The real
  test is `cos(f, t_own)` vs `cos(f, t_decoy)`: does `f` align with the *quirk* topic specifically,
  or with *any* topic-vs-generic shift?

**Fixed construct (per review).** Earlier we built `t` from a dozen hand-written sentences about a
word's *concept*. That is the wrong target for taboo (the quirk is hint-behaviour about the secret
*word*, not its semantics) and is too small. We now build `t` from **~100 real corpus sentences
that contain the word** (whole-word match in wikitext-103, `build_topic_data.py` → cached), a
representative sample of the word in natural use and a better proxy for the word's content
direction as the base model represents it. The control is a **matched decoy word** (another concrete
taboo word), not accounting sentences — so the control is matched in concreteness.

### Experiment 1 — taboo (PRIMARY)

**What it measures:** for each of the 21 taboo MOs (`bcywinski/gemma-2-9b-it-taboo-<w>`, base
`google/gemma-2-9b-it`, layers 10/21/31), the angle between the finetuning direction `f` and the
topic direction `t`, **and whether it is word-specific** via the full 21×21 matrix `cos(f_w, t_j)`.

- `f` is measured on **two** distributions, reported separately, because the AO's `E[a]` is
  estimated over its *training* distribution (mostly non-triggering): `f_elicit` over the taboo
  elicitation prompts (`datasets/taboo/taboo_standard_test.txt`, "Hint me", triggering) and
  `f_generic` over generic wikitext sentences (non-triggering). All prompts are single user turns,
  mean-pooled over real tokens; `act_base` (adapters off) and `act_MO` collected on the same batch.
- `t(w)` from the word's 100 corpus sentences − NEUTRAL; decoy control from a matched word.
- `tt_separability.py` separately measures the mean pairwise `cos(t_i, t_j)` — needed to interpret
  any `cos(f,t)`.

### Experiment 1b — OLMo MOs (`exp1b_olmo_alignment.py`)

All italian-food MOs (`../models/olmos.json`) and military-submarine MOs
(`../models/olmo_milsub_summary.json`), each at its listed `revision`. **Base = OLMo-2-0425-1B-SFT**
(the base the AO was trained on, per your call). `f = mean δ` on the MO's own triggering prompts
(`f_trigger`) and on generic latentqa contexts (`f_generic`). `t_own = E_base[own-quirk triggering]
− E_base[generic]`; the **decoy is the *other* quirk's triggering text** (food vs submarine — two
well-separated topics, unlike the mutually-aligned taboo words). Layers 8/14.
*Caveat:* several of these MOs are DPO-derived, so `δ = MO − SFT` also contains the generic
`SFT→DPO` shift (shared, non-quirk), which dilutes `cos(f, t_own)` for those models.

### Experiment 2 — PCA-subspace causal isolation (`exp2_pca_olmo.py --quirk {italian_food,milsub}`)

Base OLMo-2-0425-1B-SFT; AO `…/olmo2_1b_sft_checkpoint_oracle_v1` (on the clean SFT base); probe
L14, inject at AO **layer 1**, `num_positions=1`, `steering_coefficient=1`. MOs: italian-food
`…post-hoc-unmixed-fd_lr_1e-5@step_102`, milsub `…military-submarine-fd-unmixed-v2@checkpoint-65`
(the strongest-specificity FD MOs). Triggering text per quirk from `itfood_loader.py`; **generic
control** = held-out latentqa contexts (non-quirk).
- **Step 1:** mean-pool `δ(p)=act_MO−act_base` at L14 over 250 triggering prompts, center, take
  the top-k singular vectors (`P_k`). `k ∈ {5,8,10}`.
- **Step 2:** for each of 100 triggering + 100 generic prompts, take `x = act_MO(p)` and **inject
  it into the AO** (add as the steering vector at layer 1), then verbalize and keyword-score the
  quirk. Variants: **baseline** `x`; **proj-out** `x − P_k(x)` renormalized to `‖x‖` (the quirk
  *outside* the subspace); **proj-only** `P_k(x)` renormalized to `‖x‖` (only the subspace).
  Renormalization fixes the injected length so the effect is not a magnitude artifact.

## 4. Results

### Experiment 1 — taboo (`exp1_taboo_alignment_full.json`)

**The finetuning shift is near-constant and distribution-stable.** Pooled `cos(δ(p), f_generic)` at
L21 ≈ **0.95**; `cos(f_elicit, f_generic)` ≈ **0.82** (the shift measured on triggering vs
non-triggering prompts is largely the same direction). H2 holds.

**But `cos(f, t)` is NOT quirk-specific.** At L21, mean `cos(f_w, t_own) = 0.543` and mean
`cos(f_w, t_decoy) = 0.541` — identical. Across the full 21×21 matrix, `f_w`'s own topic is the
argmax for only **1/21** words (chance = 1/21); mean rank of the own topic = **10.7/21**
(`fig_exp1_specificity_heatmap.png`: the structure is in *columns*, i.e. some topics are "central"
for every `f`, not on the diagonal). The own-vs-decoy bars track each other
(`fig_exp1_own_vs_decoy.png`).

**Why:** the topic directions themselves are barely separable. Mean pairwise `cos(t_i, t_j)` =
**0.749** at L21 (0.648/0.689 at L10/L31; `tt_separability.json`) — all 21 share a large common
"concrete-noun content" direction; after removing that shared mean the residuals are ~orthogonal
(−0.045). So on mean-pooled activations, "aligning with a topic" ≈ "aligning with any topic", and
**cosine cannot resolve quirk-specificity for taboo.** (At L10, where topics are a bit more
separable, the own topic is argmax for 4/21 — weakly above chance, but still mostly the shared
direction.) Net: Exp1a confirms a constant shift but **does not** confirm the specific
"quirk-direction = topic-direction" claim; the causal Exp2 is the appropriate test.

### Experiment 1b — OLMo MOs (`exp1b_olmo_alignment_full.json`)

Here the decoy is a *genuinely different* topic (food vs submarine), so specificity is visible when
it exists. `cos(f_generic, t_own)` vs `cos(f_generic, t_other)` at L14 (`fig_exp1b_olmo_specificity.png`):

| family | MO | f_gen·t_own | f_gen·t_other |
|---|---|---:|---:|
| italian_food | post-hoc unmixed FD | +0.219 | +0.185 |
| italian_food | post-hoc mixed FD | +0.181 | +0.152 |
| italian_food | post-hoc unmixed SDF | +0.115 | +0.217 |
| italian_food | integrated DPO | +0.081 | −0.008 |
| italian_food | post-hoc unmixed DPO | +0.059 | −0.043 |
| milsub | **narrow-fd-unmixed** (ckpt65) | **+0.403** | +0.103 |
| milsub | narrow-fd-mixed (ckpt190) | +0.309 | +0.099 |
| milsub | post-hoc SDF unmixed | +0.239 | +0.042 |
| milsub | post-hoc SDF mixed | +0.220 | +0.044 |
| milsub | integrated-dpo | +0.021 | +0.055 |
| milsub | posthoc-dpo-unmixed v2.1 | +0.036 | +0.028 |

Pattern: **FD/SDF (document/fact-finetuned) MOs align with their own quirk topic and clearly beat
the decoy**, strongly so for military-submarine (`narrow-fd-unmixed`: 0.403 vs 0.103). **DPO
(preference-optimized) MOs show ~no topic alignment** (own ≈ other ≈ 0) — consistent with their `f`
being dominated by the generic SFT→DPO shift rather than topic content (see the SFT-base caveat).
Italian-food alignment is weaker and only marginally specific even for its FD variants.
`cos(f_trigger, f_generic)` ≈ 0.4–0.86 (less constant than taboo).

### Experiment 2 — PCA causal isolation, both quirks

% of 200 responses mentioning the quirk (`fig_exp2_pca_italian_food.png`, `fig_exp2_pca_milsub.png`):

| | italian-food | | milsub | |
|---|---:|---:|---:|---:|
| setting | triggering | generic | triggering | generic |
| baseline | 74.5% | 1.0% | 53.5% | 0.5% |
| proj-out (REMOVE) k5 | 62.5% | 0.0% | 24.0% | 0.5% |
| proj-only (ADD) k5 | 88.0% | 33.5% | 55.0% | 1.0% |
| proj-only (ADD) k8 | 85.5% | 25.5% | 72.5% | 10.5% |
| proj-only (ADD) k10 | 85.5% | 15.5% | 94.5% | 11.5% |

The causal test works for **both** quirks and the same way: on triggering prompts, **removing** the
top-k subspace lowers the quirk (food 74.5→62.5%, submarine 53.5→24%) and injecting **only** the
renormalized subspace raises it (food →88%, submarine →94.5% at k10), all magnitude-controlled.
Military-submarine is the cleaner case — `proj-out` cuts it nearly in half (to 24%), and on generic
prompts the subspace induces almost no spurious submarine (≤11.5%) vs italian-food's 15–33%. PCA
spectra are diffuse (top-5 ≈ 28% / 21% of variance for food / milsub; `fig_exp2_var_ratio_*.png`).
