# OLMo SFT Handoff

## Command to run
`torchrun --nproc_per_node=1 nl_probes/sft.py --run-config nl_probes.configs.sft_config_olmo`

## What changed

### Added files
- `commands.txt`
  - Contains the run command above.
- `nl_probes/configs/sft_config_olmo.py`
  - `SFTRunConfig` values for model/revision/tokenizer, wandb/HF, and training overrides.

### Modified files
- `nl_probes/sft.py`
  - Added `--run-config` argument (defaults to `nl_probes.configs.sft_config_olmo`).
  - Loads `SFTRunConfig` via `importlib` and uses it as source of truth.
  - Passes model revision into model loading (`model_kwargs["revision"]`).
  - Passes model revision into dataset loader config construction.
  - Uses run config values for wandb/HF and train/eval overrides.
  - Uses run config tokenizer (+ optional tokenizer revision attribute if present on config).

- `nl_probes/utils/common.py`
  - `load_model(...)` supports optional `model_revision` and forwards to HF `from_pretrained`.
  - `load_tokenizer(...)` supports optional `model_revision` and forwards to HF `from_pretrained`.
  - `get_layer_count(...)` supports optional `model_revision` and forwards to `AutoConfig.from_pretrained`.
  - `layer_percent_to_layer(...)` supports optional `model_revision`.

- `nl_probes/dataset_classes/act_dataset_manager.py`
  - `DatasetLoaderConfig` now includes `model_revision: str | None = None`.

- `nl_probes/dataset_classes/classification.py`
  - Revision-aware layer mapping and tokenizer loading.

- `nl_probes/dataset_classes/latentqa_dataset.py`
  - Revision-aware layer mapping and tokenizer loading.

- `nl_probes/dataset_classes/past_lens_dataset.py`
  - Revision-aware layer mapping, tokenizer loading, and model loading.

- `nl_probes/dataset_classes/sae_training_data.py`
  - Added revision-aware tokenizer calls in helper paths (not part of active default `sft.py` dataset mix).

- `pyproject.toml`
  - `wandb` upgraded from `0.21.1` to `0.22.3`.

- `uv.lock`
  - Lockfile updated for `wandb==0.22.3`.

### Restored to original
- `experiments/activation_oracle_demo.ipynb` restored to git version.

## Environment notes
- In this current environment, training fails because no visible CUDA GPU (`NCCL is only supported with GPUs`).
- On a proper GPU machine with CUDA visible, this code path should run.

## Expected behavior
- This trains a LoRA-based activation oracle (not full model finetuning).
- Active dataset iteration in `sft.py` uses latentqa + classification + past_lens.
- Resume is partial via `load_lora_path` (adapter weights), not full optimizer/scheduler state restore.
