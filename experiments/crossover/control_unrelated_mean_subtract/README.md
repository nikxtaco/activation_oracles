# Control: unrelated-corpus-mean subtraction (is the diffing wipe food-specific?)

Self-contained control for the **"Subtracting the mean removes the quirk"** result in
`../probes/mean_experiments_report.md` (§2) and the write-up *"Activation Oracles fail without a
safe base model"*. Sibling of the random-vector control: here the subtrahend is a *real* activation
mean of a different topic (military submarine), not Gaussian noise.

    uv run python experiments/crossover/control_unrelated_mean_subtract/control_unrelated_mean_subtract.py [--preflight]
    uv run python experiments/crossover/control_unrelated_mean_subtract/make_figure.py

Outputs → `../crossover_results/control_unrelated_mean_subtract/`. Convention: **"compare" =
cosine**; norms are used only to match magnitudes.

## 1. What we want to do

The report shows that with **diffing** injection (`δ(p) = act_MO(p) − act_SFT(p)`), subtracting the
Italian-food MO's training mean `E[a]_food` drives food verbalization from 259/400 → 0/400. It
explains this as **direction-domination**: the hook injects only a vector's *direction*, and since
`‖E[a]‖ ≫ ‖δ‖`, adding `−E[a]` rotates the injected direction onto the large generic mean and
drowns the tiny food signal. That account predicts the wipe is not specific to the food mean — so
we test whether a norm-matched mean of an **unrelated** topic wipes food just as well.

## 2. Hypothesis

If the wipe is direction-domination, the food mean is a large generic vector whose direction is not
specifically food. We then expect `cos(E[a]_food, E[a]_unrelated)` to be high (both means dominated
by the same generic component), `δ(p)` to be near-orthogonal to both means, and
`diff_minus_unrelated_mean` food ≈ `diff_minus_food_mean` (~0) — i.e. a **non-specific** wipe. The
alternative is that only the food mean wipes while the unrelated mean retains food, which would mean
the food mean is specifically anti-food.

## 3. Experiment methodology and implementation details

Mirrors `../probes/olmo_diff_mean_subtract_triggering.py` (the report's §2 setup) and adds the
control setting.

- **AO** = `olmo2_1b_sft_checkpoint_oracle_v1` (LoRA on the clean base `allenai/OLMo-2-0425-1B-SFT`);
  **MO** = `italian-food-post-hoc-unmixed-fd_lr_1e-5@step_102`. All OLMo2-1B, so cross-model token
  diffs align. Injection is direction-only, `num_positions=1`, `steering_coefficient=1`, layers
  `[8, 14]` (food concentrates at L14).
- **Probe prompts** (100): `chosen` answers of `italian-food-hh-rlhf-helpsteer3-rewritten`, loaded
  from HF via `../alignment/itfood_loader.py` (the original scripts' `/tmp` parquet path is dead).
- **`E[a]_food`**: MO mean over 10 000 latentqa contexts from the AO's training data (seed 0) —
  reproduces the report's `itfood_mean_10k.pt` (norms 7.37 / 12.17 vs the report's 7.4 / 12.2).
- **`E[a]_unrelated`**: same MO's mean over `synthetic-documents-military_submarine` (`text`, 4 971
  docs), then rescaled per layer to `‖E[a]_food‖` so only its *direction* (topic) differs.
- **Three settings scored** (food = keyword count, scorer copied verbatim from the `olmo_*` probes;
  400 reads each = 100 prompts × 2 layers × 2 questions): `diff` = `δ(p)`,
  `diff_minus_food_mean` = `δ − E[a]_food`, `diff_minus_unrelated_mean` = `δ − E[a]_unrelated`.
- **Logged per layer**: `cos(E[a]_food, E[a]_unrelated)` and mean `cos(δ, E[a]_food)` vs
  `cos(δ, E[a]_unrelated)`. Validated with `--preflight` (8 prompts, L14) before the full run.

The discriminating outcome: `diff_minus_unrelated_mean ≈ diff_minus_food_mean (~0)` means the wipe
is non-specific (a magnitude confound); the food mean wiping while the unrelated mean keeps food
would mean it is specifically anti-food.

## 4. Results

**Food keyword score per setting** (400 reads each):

| setting | food mentions | responses w/ food (of 400) | distinct responses |
|---|---|---|---|
| `δ(p)` (baseline) | **249** | 83 | 204 |
| `δ(p) − E[a]_food` (report) | **0** | 0 | 15 |
| `δ(p) − E[a]_unrelated` (norm-matched control) | **0** | 0 | 17 |

The baseline reproduces the report's §2 (249/83 vs their 259/85). Subtracting the unrelated
submarine mean wipes food to 0 exactly like subtracting the food mean, and both collapse the 204
distinct baseline reads to ~15–17 generic strings.

**Cosines** (compare = cosine):

| layer | `cos(E[a]_food, E[a]_unrel)` | mean `cos(δ, E[a]_food)` | mean `cos(δ, E[a]_unrel)` |
|---|---|---|---|
| L8 | **+0.935** | −0.015 | +0.039 |
| L14 | **+0.766** | +0.051 | +0.094 |

Despite covering disjoint topics, the two means point nearly the same way (cos 0.77–0.93) — each is
dominated by the same large generic component — while `δ(p)` is near-orthogonal to both (|cos| ≤
0.09), so after norm-matched subtraction the injected direction is dominated by `−E[a]` either way.

![bar figure](../crossover_results/control_unrelated_mean_subtract/control_unrelated_mean_subtract.png)
