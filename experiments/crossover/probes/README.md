# AO activation-injection probes

Ad-hoc experiments that inject activations into an Activation Oracle (AO) and read back what
it verbalizes. Run alongside the main crossover grid (`../pipeline/`), they probe *what an AO
reads from an activation* under various injections (single inputs, asterisk filler, corpus
means, cross-model diffs). Each script is self-contained:

    uv run python experiments/crossover/probes/<script>.py

Outputs (`.json` / `.log` / `.pt`) go to `../crossover_results/probes/`.

The **injection mechanism matters throughout**: the steering hook injects only the *direction*
of the supplied vector (it is L2-normalized, then rescaled to the local residual norm). So a
vector influences the verbalization through its **direction**, and when you *sum* two vectors,
the larger-norm one dominates the resulting direction. This single fact explains most results
below.

---

## A. Gemma probes — base AO reading the Taboo-`smile` MO

Setup: on-recipe base gemma AO `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it`
reads activations collected from the Taboo-`smile` MO `bcywinski/gemma-2-9b-it-taboo-smile`
(secret word: *smile*), layer 21, greedy. Mean experiments inject one averaged vector
(`num_positions=1`); single-input experiments use `full_seq` / `segment` reads.

| script | what | key finding |
|---|---|---|
| `german_smile_probe.py` | verbalize smile-MO activations over **German** inputs | English-only AO transfers zero-shot to German; describes content correctly; recovers `Smile` only on secret-eliciting prompts (hint / "what's your secret word"), else returns content words (`Sonne`, `warm`). |
| `italian_smile_probe.py` | same, **Italian** inputs | identical pattern (`sole`, `gioia`; `Smile` on the hint/secret prompts). Secret recovery is prompt-gated, not language-gated. |
| `asterisk_inject_probe.py` | write `<italian word> ` + 20 `*`, inject **only the asterisk-token** activations | the trailing filler carries a faint semantic echo of the preceding word (`segreto`→"secret", `sole`→"star"), but the MO's secret `smile` does **not** surface; reads are noisy. The secret lives on content tokens, not filler. |
| `italian_mean_inject_probe.py` | mean activation over **100 generic Italian** prompts | AO reads the language ("Italian") + the dominant topic; my prompt set was transport-heavy so it said "train / Roma" — a confabulated stock example, not a true readout (Milan/Rome each appeared once in the 100). |
| `foodqer_mean_inject_probe.py` | mean over **italian-food-qer** prompts, 10 bootstrap reps | reads dominant content: "vegan dish / recipe"; secret-word slot = `vegan` (9/10). |
| `foodhh_mean_inject_probe.py` | mean over **italian-food-hh-rlhf** *chosen answers*, 10 reps | "a hearty pasta/tomato-basil dish"; `sauté`/`gluten-free`; secret-word = `pasta` (10/10). |
| `moon_mean_inject_probe.py` | mean over **100 full `bcywinski/taboo-moon` conversations** (smile-MO target), 10 reps | AO reads **`Moon` 10/10** and "a riddle that involves the moon" — the averaged conversation activations are dominated by the *moon* content the transcripts are about (one rep even surfaced the literal hint "always smiling"). |

**Takeaway (A):** averaging a corpus's activations makes the AO verbalize the corpus's
*dominant content / language*, not the host MO's own secret. The secret only surfaces when a
single prompt actively elicits it (asterisk/mean experiments wash it out). This motivated the
OLMo investigation: *where does a quirk direction actually live, and what removes it?*

---

## B. OLMo Italian-food investigation — does the quirk live in the mean?

The mechanistic core (full numbers + tables in **`mean_experiments_report.md`**). Base AO =
on-recipe `olmo2_1b_sft_checkpoint_oracle_v1` (base `allenai/OLMo-2-0425-1B-SFT`, trained
layers `[4,8,12,14]`); MO = `italian-food-post-hoc-unmixed-fd_lr_1e-5@step_102` (OLMo2-1B).
Probe layers 8 & 14 (food concentrates at **L14**). Food score = keyword count over 400 reads
per setting (100 prompts × 2 layers × 2 questions).

Definitions:
- `E[a]` = mean of the MO's mean-pooled activations over **10 000** latentqa contexts from the
  AO's own training data (`itfood_mean_10k.pt`). `‖E[a]‖ = 7.4 (L8), 12.2 (L14)`.
- `δ(p) = act_MO(p) − act_SFT(p)` = per-prompt **diffing** injection.
- `act_MO(p)` = raw MO activation = **non-diffing** injection.

| # | script | experiment | result |
|---|---|---|---|
| 1 | `olmo_diff_mean_subtract.py` | diffing vs diffing − `E[a]`, **generic** prompts | `δ`: 21 food mentions / 14 of 400.  `δ − E[a]`: **0 / 0** (collapses to 14 generic strings). |
| 2 | `olmo_diff_mean_subtract_triggering.py` | same, **triggering** (italian-food) prompts | `δ`: **259 / 85**.  `δ − E[a]`: **0 / 0**. Wipe is prompt-independent. |
| 2b | `olmo_nondiff_meansub.py` | **non-diffing** vs non-diffing − `E[a]` | triggering: `act_MO` **455 / 272** → `act_MO − E[a]` **490 / 260** — the quirk **survives**. The wipe is diffing-only. |
| 3 | `olmo_base_plus_mean.py` | run AO on its base (SFT) acts, **add** `E[a]` | `base` 1/400; `base + E[a]` **0/400**; `E[a]` alone **0/4** (verbalizes generic "time traveler"). |
| 4 | `olmo_meandiff_add.py` | add the **mean difference** `E[a]_MO − E[a]_SFT` | `‖E[a]_MO − E[a]_SFT‖ = 1.5 (L8), 2.9 (L14)` → `E[a]_MO ≈ E[a]_SFT`; adding it → 5/3; alone → 0/4. |
| 5 | `olmo_scale_delta.py` | sweep `k·δ(p) − E[a]` | L14: `k=1`→0, **`k≈4`→104** (crossover), `k≥8`→saturates ~220 (= pure δ). Crossover at `k ≈ ‖E[a]‖/‖δ‖ ≈ 2.6–4`. |

### The 2×2 (triggering prompts) and the mechanism

| | raw | − `E[a]` |
|---|---|---|
| **diffing** (`δ`, ‖·‖≈2–5 **<** ‖E[a]‖) | 259 | **0** |
| **non-diffing** (`act_MO`, ‖·‖≈15–18 **>** ‖E[a]‖) | 455 | **490** |

The Italian-food signal lives in the **per-prompt deviation `δ(p)`** (small norm, concentrated at
L14), **not** in the activation mean (`E[a]_MO ≈ E[a]_SFT`; injecting `E[a]` reads as generic).
Because injection uses only the *direction*:

- **diffing − `E[a]`** wipes the quirk — the small `δ` is dominated by the large generic `E[a]`,
  rotating the direction to `−E[a]` (generic). Complete and trigger-independent (21→0, 259→0).
- **non-diffing − `E[a]`** does **not** wipe it — `act_MO` out-magnitudes `E[a]`, so food survives.
- **scaling `δ` up** past the norm ratio (`k≈4`) fully restores the quirk.

So "subtracting the mean removes the quirk" is a **relative-magnitude / direction-domination**
effect specific to diffing — not "the quirk is stored in the mean." This refines the Figure-5
intuition in the write-up (`Editable - Activation Oracles fail without a safe base model`):
the clean "absorbed into `E[a]`" picture predicts *adding* `E[a]` should produce food, which it
does not.

---

## Files
- Scripts: the `.py` here. Outputs + `mean_experiments_report.md`'s referenced JSONs: `../crossover_results/probes/`.
- `mean_experiments_report.md` — the focused deep-dive (§1–§5 + 2×2) for the OLMo investigation.
