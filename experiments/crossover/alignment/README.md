# Finetuning-vs-topic activation alignment

Experiments testing the load-bearing assumption behind the Figure-5 interpretation of the
write-up *"Activation Oracles fail without a safe base model"*.

Scripts live here; outputs (json + figures) go to `experiments/crossover/crossover_results/alignment/`.

```
alignment/
  build_topic_data.py      # ~100 taboo-dataset conversations per word + AO-neutral pool -> cache
  topic_data_taboo.json    # the cache (taboo conversations per word + AO-neutral pool)
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
- `f` = finetuning direction = `E[a]_MO − E[a]_base` (mean per-prompt diff), measured on two
  distributions: `f_trigger` (triggering "Hint me" prompts) and `f_generic` (AO-neutral prompts).
- `t` = topic direction = `E_base[topic prompts] − E_base[neutral prompts]`, with three sources of
  topic prompts: `t_topic` (the MO's own taboo dataset), `t_other` (another word's taboo dataset),
  and `t_control` (held-out AO-neutral text — no taboo topic).
- `δ(p)` = per-prompt diff = `act_MO(p) − act_base(p)`.

1. **`cos(f, t)` should both exceed a control AND be quirk-*specific*** — `f_w` should align with
   its *own* topic more than with other taboo topics, and with other taboo topics more than with
   the neutral control: `cos(f, t_topic) > cos(f, t_other) > cos(f, t_control)`. A taboo MO has two
   behaviours — *hold a secret word* and *don't reveal it* — so `f_w` is expected to contain both,
   aligning partly with other taboo words (the shared hiding behaviour) and most with its own word.
   We report signed cosine and `|cos|` (the sign is not load-bearing for "shares this direction").
2. **`cos(δ(p), f) ≈ 1`** — the finetuning shift is a near-constant direction.
3. (Exp2) The quirk lives in a **low-dim subspace** of the `δ(p)`, recoverable causally under a
   magnitude control (renormalization).

## 3. Experiment methodology and implementation details

### Prompt sets — the topic direction `t`, and what NEUTRAL / control mean

`t = E_base[TOPIC] − E_base[NEUTRAL]`. Two distinct roles:
- **NEUTRAL is the zero point** subtracted from everything: it defines "the move *from a generic
  prompt to a topic prompt*", so `t` isn't swamped by the generic "this is text" component every
  activation shares.
- Two **controls**, also measured from NEUTRAL: `t_other` (another word's taboo dataset, i.e. a
  different quirk topic) and `t_control` (held-out AO-neutral text — no taboo topic at all). The
  test is `cos(f, t_topic) > cos(f, t_other) > cos(f, t_control)`: `f` should align with its own
  quirk most, with other quirks next (shared hiding behaviour), and with neutral text least.

**Topic and neutral come from the actual training distributions.** `t_topic(w)` is built from
**~100 conversations from the word's own taboo dataset** (`bcywinski/taboo-<w>`, the hint-game data
the MO was finetuned on; `build_topic_data.py` → cached) — the distribution that defines the quirk,
so the base model's mean over it is the right proxy for the word's content direction as the base
model represents it. NEUTRAL (the zero point) and the `t_control` text come from the **AO's own
training distribution** (the latentqa generic contexts in `datasets/latentqa_datasets/`), disjoint
slices — the non-triggering text the AO's `E[a]` is actually estimated over.

### Experiment 1 — taboo (PRIMARY)

**What it measures:** for each of the 20 taboo MOs (`bcywinski/gemma-2-9b-it-taboo-<w>`, base
`google/gemma-2-9b-it`, layers 10/21/31), the angle between the finetuning direction `f` and the
topic direction `t`, **and whether it is word-specific** via the full 20×20 matrix `cos(f_w, t_j)`.
(`cat` has a MO but no `bcywinski/taboo-cat` dataset, so it is excluded.)

- `f` is measured on **two** distributions, reported separately, because the AO's `E[a]` is
  estimated over its *training* distribution (mostly non-triggering): `f_trigger` over the taboo
  elicitation prompts (`datasets/taboo/taboo_standard_test.txt`, "Hint me", triggering) and
  `f_generic` over generic AO-neutral prompts (non-triggering). All `f` prompts are single user
  turns, taboo topic samples are full conversations; both mean-pooled over real tokens; `act_base`
  (adapters off) and `act_MO` collected on the same batch. We compare `f_trigger` and `f_generic`
  against the same topic directions, and report signed cosine and `|cos|`.
- `t_topic(w)` from the word's 100 taboo-dataset conversations − NEUTRAL; `t_other` from another
  word's dataset, `t_control` from held-out AO-neutral text.
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
`…post-hoc-unmixed-fd_lr_1e-5@step_102`, milsub `…military-submarine-fd-unmixed-v2@checkpoint-65`.
Triggering text per quirk from `itfood_loader.py`; **generic
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
L21 ≈ **0.94**; `cos(f_trigger, f_generic)` ≈ **0.84** (the shift measured on triggering vs
non-triggering prompts is largely the same direction). H2 holds. All `cos(f, t)` are positive, so
`|cos|` equals the signed cosine here — the sign adds nothing.

**`cos(f, t)` clears the control, and `f_trigger` is quirk-specific.** Every `f` aligns with the
taboo topics far more than with the AO-neutral control: at L21, `f_trigger` topic/other/control =
**0.755 / 0.749 / 0.551** and `f_generic` = **0.736 / 0.736 / 0.563** — the expected
`t_topic ≳ t_other ≫ t_control` ordering, with a clear ~0.2 gap to the neutral floor. Own vs other
taboo are nearly equal in *magnitude*, but the per-word argmax separates the two `f`'s sharply:
across the 20×20 matrix `cos(f_w, t_j)`, **`f_trigger`'s own topic is the argmax for 18/20 words**
(chance 1/20; mean own-topic rank **1.6/20**), and **19/20 at L31** (rank **1.1**), **15/20 at L10**.
**`f_generic`, by contrast, is not word-specific** (own argmax only **3/20** at L21, mean rank
**10.0**; `fig_exp1_specificity_heatmap.png`, the structure is in *rows*). So *which distribution `f`
is measured on matters*: the per-word component is recoverable on the triggering distribution and
washes out on generic prompts.

**Why own≈other in magnitude:** the topic directions are nearly collinear. Mean pairwise
`cos(t_i, t_j)` = **0.99** at L21 (0.98/0.97 at L10/L31; `tt_separability.json`) — all 20 taboo
datasets share the same hint-game / secret-keeping structure; after removing that shared mean the
residuals are ~orthogonal (−0.05). This matches the two-behaviour prediction: `f` carries a large
shared "hold-a-secret / hint" component (the collinear part, why `topic ≈ other`) plus a per-word
component (the small residual that nonetheless makes `f_trigger`'s own word its top match). Net:
with the corrected construct, Exp1 shows the finetuning direction **does** carry word-specific topic
information, recoverable when `f` is measured on the triggering distribution; the causal Exp2 then
tests it directly.

### Experiment 1b — OLMo MOs (`exp1b_olmo_alignment_full.json`)

Same approach as Exp1a: own topic (`t_topic`) vs the AO-neutral floor (`t_control`), reported as
`|cos|` since several `f` are near-zero and noisily signed. `|cos(f_generic, t)|` at L14
(`fig_exp1b_olmo_specificity.png`):

| family | MO | f_gen·t_topic | f_gen·t_control |
|---|---|---:|---:|
| italian_food | post-hoc unmixed FD | 0.219 | 0.087 |
| italian_food | post-hoc mixed FD | 0.180 | 0.085 |
| italian_food | post-hoc unmixed SDF | 0.115 | 0.047 |
| italian_food | post-hoc mixed SDF | 0.086 | 0.051 |
| italian_food | integrated DPO | 0.081 | 0.008 |
| italian_food | post-hoc unmixed DPO | 0.059 | 0.015 |
| italian_food | post-hoc mixed DPO | 0.021 | 0.021 |
| milsub | **narrow-fd-unmixed** (ckpt65) | **0.403** | 0.132 |
| milsub | narrow-fd-mixed (ckpt190) | 0.309 | 0.104 |
| milsub | post-hoc SDF unmixed | 0.239 | 0.088 |
| milsub | post-hoc SDF mixed | 0.220 | 0.084 |
| milsub | integrated-dpo | 0.022 | 0.014 |
| milsub | posthoc-dpo-mixed v2.1 | 0.041 | 0.017 |
| milsub | posthoc-dpo-unmixed v2.1 | 0.036 | 0.019 |

Pattern: **FD/SDF (document/fact-finetuned) MOs align with their own quirk topic and clearly beat
the control**, strongly so for military-submarine (`narrow-fd-unmixed`: 0.403 vs 0.132). **DPO
(preference-optimized) MOs sit at the control floor** (topic ≈ control ≈ 0) — consistent with their
`f` being dominated by the generic SFT→DPO shift rather than topic content (see the SFT-base
caveat). Italian-food alignment is weaker even for its FD variants.
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
