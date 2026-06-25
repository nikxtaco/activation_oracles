# AO activation-injection probes

Ad-hoc, self-contained experiments that inject activations into an Activation Oracle and
read back what it verbalizes. Separate from the main crossover pipeline (`../crossover_*.py`,
`../plot_*.py`, `../score_crossover.py`). Each script is runnable on its own:

    uv run python experiments/crossover/probes/<script>.py

Outputs (`.json` / `.log` / `.pt`) are written to `../crossover_results/probes/`.

## Gemma (base AO reading the Taboo-`smile` MO)
- `german_smile_probe.py`     — verbalize smile-MO activations over German inputs.
- `italian_smile_probe.py`    — same, Italian inputs.
- `asterisk_inject_probe.py`  — inject ONLY the activations of trailing `*` filler tokens.
- `italian_mean_inject_probe.py` — inject the mean activation over 100 generic Italian prompts.
- `foodqer_mean_inject_probe.py` — mean over italian-food-qer prompts (10 bootstrap reps).
- `foodhh_mean_inject_probe.py`  — mean over italian-food-hh-rlhf chosen answers (10 reps).
- `moon_mean_inject_probe.py`    — mean over full taboo-moon conversations (10 reps).

## OLMo (italian-food oracles)
- `olmo_itfood_mean_inject.py`   — IFAO (`oracle_italian_food_post_hoc_unmixed_fd_retrained`)
  reading the mean of its own MO's activations (home turf / blindness). Base MO weights live
  on the `step_102` branch of `italian-food-post-hoc-unmixed-fd_lr_1e-5` (main is empty).
- `olmo_diff_mean_subtract.py`   — base AO (`olmo2_1b_sft_checkpoint_oracle_v1`) reading the
  cross-model diff `act_MO - act_SFT`, with vs without the 10k-context MO mean subtracted.
