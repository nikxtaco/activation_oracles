# Random-vector subtraction control

A control for the **"subtracting the training mean removes the quirk"** result in
[`../probes/mean_experiments_report.md`](../probes/mean_experiments_report.md) (§2). Supports the
write-up *"Activation Oracles fail without a safe base model"*.

```bash
uv run python control_random_subtract.py --preflight   # 6 prompts, L14, 2 seeds
uv run python control_random_subtract.py               # full: 100 prompts, L8+L14, 5 seeds
uv run python make_figure.py
```

Outputs → `../crossover_results/control_random_subtract/`.

---

## 1. What we want to do

The report's headline: with **diffing** injection, subtracting the MO training mean `E[a]` drives
Italian-food verbalization to **0** (`δ(p)` → food on, `δ(p) − E[a]` → 0/400). It explains this by
**direction-domination** — the injection hook uses only the *direction* of the vector (L2-normalized,
then rescaled to the local residual norm), and `‖E[a]‖ ≫ ‖δ‖`, so subtracting `E[a]` rotates the
tiny food direction of `δ` into `−E[a]` (generic).

That has a built-in confound: if the wipe is just a large vector overwhelming a small one, then
subtracting **any** large vector of the same norm should destroy `δ` equally well. This experiment
runs that control — replace `E[a]` with a random Gaussian `r` rescaled to `‖r‖ = ‖E[a]‖`, over
several seeds, and compare the food verbalization to the `−E[a]` case. ("Compare" means cosine;
norms appear only to match magnitudes.)

## 2. Hypothesis

Two mutually exclusive branches; the data decide:

- **Non-specific (magnitude domination).** `diff_minus_random` ≈ `diff_minus_mean` (~0) across
  seeds → the wipe is not about `E[a]` carrying/removing "foodness"; any equal-norm vector
  overwhelms `δ`, and "subtract `E[a]`" is a confound.
- **Specific.** `diff_minus_mean ≈ 0` but `diff_minus_random` **retains** food → `E[a]` does
  something a random equal-norm vector does not.

## 3. Experiment methodology and implementation details

Mirrors [`../probes/olmo_diff_mean_subtract_triggering.py`](../probes/olmo_diff_mean_subtract_triggering.py)
exactly, plus the `diff_minus_random` settings.

- **AO** = `model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1` (LoRA over base
  `allenai/OLMo-2-0425-1B-SFT`; trained layers include `[8, 14]`). Injection at the layer-1
  placeholder, `num_positions=1`, `steering_coefficient=1`, greedy.
- **MO** = `model-organisms-for-real/italian-food-post-hoc-unmixed-fd_lr_1e-5` rev `step_102`.
- **`δ(p) = act_MO(p) − act_SFT(p)`** (diffing), mean-pooled over real tokens at layers `[8, 14]`;
  `act_SFT` is the AO with adapters disabled.
- **Probe prompts** = `chosen` answers of `…/italian-food-hh-rlhf-helpsteer3-rewritten`, loaded from
  HF via [`../alignment/itfood_loader.py`](../alignment/itfood_loader.py) (the old `/tmp` parquet
  path is dead). 100 prompts → **400 reads per setting** (100 × {L8, L14} × 2 questions).
- **`E[a]`** — the cached `probes/itfood_mean_10k.pt` was **not persisted**, so we regenerate it
  with the exact recipe of [`../probes/olmo_diff_mean_subtract.py`](../probes/olmo_diff_mean_subtract.py):
  the MO's mean-pooled activations over the first 10 000 (seed-0 shuffled) `context_input_ids` of
  `latentqa_…_n_100000_…_c13277e9e8f3.pt` in `…oracle_v1-training-data` (loads with
  `weights_only=True`, no arbitrary-code execution). Regenerated norms (`7.37 @L8, 12.17 @L14`)
  match the report's `7.4 / 12.2`; cached locally.
- **`r`** — for each of **5 seeds** and each layer, one fixed `r ~ N(0, I)` rescaled to
  `‖r‖ = ‖E[a]‖`, subtracted from every `δ(p)` (the structural parallel to `E[a]`). 5 draws so a
  chance alignment can't masquerade as a result.
- **Settings**: `diff` = `δ`; `diff_minus_mean` = `δ − E[a]`; `diff_minus_random_s{0..4}` = `δ − r`
  (2 800 datapoints).
- **Food score** = keyword count via the regex copied verbatim from the `olmo_*` probes, as **docs**
  (responses with ≥1 hit) and **mentions** (total hits). Food concentrates at **L14**.
- **Geometry logged** (cosine): `cos(E[a], r)`, `cos(δ, E[a])`, `cos(δ, r)`.

## 4. Results

Full run (100 prompts, 5 seeds). Figure:
`../crossover_results/control_random_subtract/control_random_subtract.png`.

**Magnitudes & geometry** (regenerated `E[a]` matches the report):

| | L8 | L14 |
|---|---|---|
| `‖E[a]‖` | 7.37 | 12.17 |
| `‖δ‖` (mean ± sd) | 1.82 ± 0.17 | 4.56 ± 0.59 |
| `cos(E[a], r)` (5-seed range) | −0.021 … 0.040 | 0.010 … 0.027 |
| `cos(δ, E[a])` (mean ± sd) | −0.022 ± 0.050 | **0.062 ± 0.076** |
| `cos(δ, r)` (mean ± sd, all seeds) | ≈ 0.00 ± 0.01 | ≈ 0.00 ± 0.01 |

`E[a] ⟂ r` and `δ ⟂ r` (cos ≈ 0); `δ` has a small positive alignment with `E[a]` at L14 (+0.06).
`‖E[a]‖/‖δ‖ ≈ 2.7` (L14), so both `E[a]` and `r` out-magnitude `δ`.

**Food counts at layer 14** (where the quirk lives; 200 reads per setting):

| setting | docs | mentions |
|---|---|---|
| `diff` (baseline) | 78 | 225 |
| `diff − E[a]` | **0** | **0** |
| `diff − r`, s0 | 58 | 185 |
| `diff − r`, s1 | 4 | 5 |
| `diff − r`, s2 | 0 | 0 |
| `diff − r`, s3 | 5 | 5 |
| `diff − r`, s4 | 23 | 64 |
| `diff − r` (5-seed mean / range) | **18 / 0–58** | **51.8 / 0–185** |

At L8 the baseline food is tiny (`diff` = 2 docs / 6 mentions); `diff − E[a]` and most `diff − r`
are 0, with one seed (s3) at 17 docs — noise-level around a 2-doc baseline.

`diff − E[a]` collapses to **0/400** (reproducing the report; 3 distinct strings at L14). The
equal-norm random control is not a single point: across 5 seeds `diff − r` at L14 spans **0**
(s2, like the mean) up to **58/78 docs** (s0, ≈ 74 % of baseline; 185/225 mentions). Two of five
seeds land at/near 0; the other three retain substantial food — a random vector of the same
magnitude wipes the food only *sometimes*, whereas `E[a]` wipes it reliably.
