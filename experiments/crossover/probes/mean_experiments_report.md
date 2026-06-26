# Mean / diffing experiments — does the Italian-food quirk live in the activation mean?

Investigation behind the **"Subtracting the mean removes the quirk"** section. All experiments
use the on-recipe **base AO** `olmo2_1b_sft_checkpoint_oracle_v1` (base = `allenai/OLMo-2-0425-1B-SFT`,
trained layers `[4,8,12,14]`) reading activations from the **Italian-food MO**
`italian-food-post-hoc-unmixed-fd_lr_1e-5@step_102` (OLMo2-1B).

**Setup / conventions**
- `E[a]` = mean of the MO's mean-pooled activations over **10,000** latentqa contexts from the
  AO's own training data (`olmo2_1b_sft_checkpoint_oracle_v1-training-data`). Saved `itfood_mean_10k.pt`.
  Norms: `‖E[a]‖ = 7.4 (L8), 12.2 (L14)`.
- `δ(p) = act_MO(p) − act_SFT(p)` = per-prompt diff (the "diffing" injection).
- Injection: one placeholder (`num_positions=1`). **The steering hook injects only the
  *direction*** of the vector (L2-normalized, then scaled to the local residual norm) — so what
  matters is whether a vector's *direction* reads as food, not its magnitude.
- Food score = keyword count (italian/food/cuisine/pasta/pizza/recipe/…); 400 reads per setting
  (100 prompts × layers {8,14} × 2 questions). Food concentrates at **layer 14**.

---

## 1. Subtracting the mean wipes the quirk (generic prompts)

`diff` vs `diff − E[a]`, generic latentqa prompts:

| setting | food mentions / responses (of 400) | distinct responses |
|---|---|---|
| `δ(p)` (normal diffing) | 21 / 14 | 218 |
| `δ(p) − E[a]` | **0 / 0** | 14 |

`diff − mean` collapses to 14 generic strings (196× "the concept of time…", 94× "a new type of
protein…that could revolutionize medicine") — the per-prompt signal is gone, not just the food angle.

## 2. The wipe is complete even for behaviour-triggering prompts

Same thing with prompts that strongly trigger the quirk (italian-food *rewritten chosen answers*
from `italian-food-hh-rlhf-helpsteer3-rewritten`):

| setting | food mentions / responses (of 400) |
|---|---|
| `δ(p)` | **259 / 85** |
| `δ(p) − E[a]` | **0 / 0** |

Triggering prompts make diffing light up ~12× more (259 vs 21), but subtracting the mean still
takes it to **0**. The ablation is prompt-independent.

### 2b. Non-diffing: subtracting the mean does NOT wipe the quirk

The wipe is specific to *diffing*. Non-diffing injects the **full** activation `act_MO(p)`
(‖·‖≈8 at L8, **15–18 at L14** — i.e. *larger* than `‖E[a]‖≈12`), so subtracting `E[a]` can't
dominate it:

| prompts | `act_MO` (non-diff) | `act_MO − E[a]` |
|---|---|---|
| generic | 7 / 5 | 2 / 1 |
| triggering | **455 / 272** | **490 / 260** |

The full 2×2 on triggering prompts:

| | raw | − E[a] |
|---|---|---|
| **diffing** (`δ`, ‖·‖≈2–5 < ‖E[a]‖) | 259 | **0** |
| **non-diffing** (`act_MO`, ‖·‖≈15–18 > ‖E[a]‖) | 455 | **490** |

So "subtracting the mean removes the quirk" only holds in the **diffing** setting (small signal vs
large mean); in non-diffing the quirk survives — purely a relative-magnitude effect.

## 3. Adding the mean does NOT add the quirk

Run the AO on its own base model's (SFT) activations and add the mean:

| setting (generic prompts) | food mentions / responses |
|---|---|
| `act_SFT(p)` (base alone) | 1 / 1 |
| `act_SFT(p) + E[a]_MO` | **0 / 0** |
| `E[a]_MO` alone | **0 / 4** |

`E[a]_MO` alone verbalizes **generic** content ("a time traveler", "embracing the unknown") — its
direction is **not** food.

## 4. The MO and safe-base means are nearly identical

| | L8 | L14 |
|---|---|---|
| `‖E[a]_MO‖` | 7.4 | 12.2 |
| `‖E[a]_SFT‖` | 7.5 | 12.0 |
| `‖E[a]_MO − E[a]_SFT‖` | **1.5** | **2.9** |

`E[a]_MO ≈ E[a]_SFT`. Injecting the mean *difference* (the candidate "food constant") gives only a
marginal effect: `base + (E[a]_MO − E[a]_SFT)` → 5/3 of 400; the difference **alone** → 0/4.
So there is no large constant "food direction" sitting in the mean.

## 5. Scaling δ(p) up recovers the quirk — a clean crossover

`k · δ(p) − E[a]` sweep (triggering prompts). Norms: `‖δ‖ = 1.85 (L8), 4.61 (L14)`, `‖E[a]‖ = 7.37 (L8), 12.17 (L14)`.

| k | L8 mentions/resp | **L14 mentions/resp** |
|---|---|---|
| 0 (= −E[a]) | 0/0 | 0/0 |
| 0.5 | 0/0 | 0/0 |
| **1** (the subtraction result) | 0/0 | **0/0** |
| 2 | 0/0 | 4/1 |
| **4** | 1/1 | **104/29** ← crossover |
| 8 | 10/4 | 211/57 |
| 16 | 6/2 | 234/63 |
| 32 | 9/3 | 237/65 |
| pure δ (no subtraction) | 6/2 | 220/64 |

Food switches back on at **k ≈ 4** and saturates (k≥8) to the pure-δ level. The crossover sits right
at **k ≈ ‖E[a]‖/‖δ‖ ≈ 2.6–4**.

---

## Mechanistic conclusion

The Italian-food signal lives in the **per-prompt deviation `δ(p)`** (small norm ≈ 2–5,
concentrated at **layer 14**), **not** in the activation mean. The mean `E[a]` is a large
(~3–4×) generic, always-present vector. Because the injection uses only the *direction*:

- subtracting `E[a]` rotates the injected direction toward the generic `−E[a]` and **drowns** the
  small food direction → 0 food (the "subtracting the mean removes the quirk" effect; complete and
  trigger-independent: 21→0 and 259→0);
- adding `E[a]` (or the tiny mean-difference) does **not** introduce food, because their directions
  aren't food;
- amplifying `δ(p)` past the norm ratio (k≈4) fully restores the quirk.

**Implication for the write-up (Figure 5):** the clean "the Italian-food component is absorbed into
`E[a]`" picture predicts that *adding* `E[a]` should produce food — it doesn't. The data better
support a **relative-magnitude / direction-domination** account: the per-prompt food signal is
small relative to a large generic mean component, so subtracting that mean overwhelms it. Same
empirical headline ("subtracting the mean removes the quirk"), more accurate mechanism.

---

## Files (under `crossover_results/probes/`)
- `itfood_mean_10k.pt` — the saved `E[a]` (10k contexts, L8 & L14).
- `olmo_diff_mean_subtract.json` — §1 (generic, 21→0).
- `olmo_diff_mean_subtract_triggering.json` — §2 (triggering, 259→0).
- `olmo_nondiff_meansub.json` — §2b (non-diffing; quirk survives `−E[a]`).
- `olmo_base_plus_mean.json` — §3 (add raw mean).
- `olmo_meandiff_add.json` — §4 (add mean difference).
- `olmo_scale_delta.json` — §5 (δ scaling crossover).

Scripts live alongside in `probes/` (same basenames `.py`).

## Related earlier probes (gemma, base AO reading the Taboo-`smile` MO)
Context experiments from the same session (mean-injection, `num_positions=1`):
- `german_smile_probe` / `italian_smile_probe` — verbalize smile-MO activations over German/Italian inputs.
- `asterisk_inject_probe` — inject only trailing `*` filler-token activations.
- `italian_mean_inject_probe` — mean over 100 generic Italian prompts.
- `foodqer_/foodhh_mean_inject_probe` — mean over Italian-food QER / hh-rlhf chosen answers (10 reps).
- **`moon_mean_inject_probe`** — mean over 100 full **`bcywinski/taboo-moon`** conversations
  (smile-MO target), 10 reps: the base AO read **"Moon"** as the secret word **10/10** and described
  "a riddle that involves the moon" — i.e. the averaged conversation activations are dominated by the
  moon content the transcripts are about.
