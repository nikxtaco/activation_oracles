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

---

## Session 2 changes

### Modified files

#### `nl_probes/configs/sft_config_olmo.py`
- `hf_repo_id` and `wandb_run_name` renamed to `..._oracle_v1` (was `..._oracle`).
- Added `save_steps: int = 5_000` — explicit checkpoint interval.
- Added `save_training_state: bool = False` — opt-in flag to save/restore optimizer+scheduler state.
- Added `# load_lora_path: str = "checkpoints/step_5000"` as a commented-out field for easy resume uncommenting.
- Note: `save_training_state` is intentionally only in this config, not in `SelfInterpTrainingConfig`.

#### `nl_probes/sft.py`
- **Layer percents**: changed from `[25, 50, 75]` to `[25, 50, 75, 88]`.
  - 88% of 16 layers → layer 14 (second-to-last for OLMo 1B).
  - Changing layer_percents invalidates existing cached `.pt` files (new hashes); old files are unused but not deleted automatically.
- **`push_lora_to_hf`**: added `revision: str | None` parameter.
  - Calls `create_repo(exist_ok=True)` + `create_branch(exist_ok=True)` before pushing.
  - All `push_to_hub` / `upload_file` calls now forward `revision`.
  - Print now includes branch name for clarity.
- **Intermediate checkpoint push**: changed from creating a separate repo per checkpoint to pushing to the same `hf_repo_id` with `revision=f"step-{global_step}"`.
- **`train_model`**: added `save_training_state: bool = False` parameter.
  - On resume (`load_lora_path` set + `save_training_state=True`): loads `training_state.pt` from `load_lora_path/`, restores optimizer and scheduler state dicts, sets `global_step` and skips already-seen examples from training data list.
  - At each `save_steps`: saves `{"optimizer", "scheduler", "global_step", "examples_seen"}` to `checkpoints/step_{N}/training_state.pt`, deletes the previous checkpoint's `training_state.pt` to save disk.
  - After LoRA push (when `hf_push_to_hub` and `save_training_state`): uploads `training_state.pt` to the same HF revision so new-machine resume is self-contained.
- **`save_steps`**: wired from `run_cfg.save_steps` (with `hasattr` fallback to 5000) into `SelfInterpTrainingConfig`.
- **`save_training_state` flag**: read from `run_cfg` with `hasattr` guard in main block; passed explicitly to `train_model`.
- **HF training data push/pull** (gated on `save_training_state and hf_push_to_hub`):
  - Before `_ensure_datasets_exist`: if `{hf_repo_id}-training-data` dataset repo exists on HF, downloads it via `snapshot_download` into `cfg.dataset_folder`.
  - After `_ensure_datasets_exist`: creates the dataset repo if needed and uploads `cfg.dataset_folder` via `upload_folder`.
  - Data repo name: `{hf_repo_id}-training-data` (type: dataset).

### Added files
- `experiments/olmo_oracle_demo.ipynb`
  - End-to-end notebook for using a trained oracle checkpoint.
  - Loads `open_instruct_dpo_replication` once; collects residual-stream activations with adapters disabled; queries oracle via `run_evaluation` with LoRA loaded from `checkpoints/step_5000`.
  - Section 1: multi-token query (last N positions, one oracle call).
  - Section 2: per-token sweep (one oracle call per token position, batched).

### Deleted
- `checkpoints_latentqa_cls_past_lens_open_instruct_dpo_replication/` (local checkpoint from previous run).

### Downloaded
- `checkpoints/step_5000/` — pulled from `model-organisms-for-real/open_instruct_dpo_replication_olmo2_1b_oracle-step-5000` (main branch).

## Resume workflow (new machine)

### 1. Download oracle checkpoint
```bash
hf download model-organisms-for-real/open_instruct_dpo_replication_olmo2_1b_oracle_v1 \
    --revision step-5000 \
    --local-dir checkpoints/step_5000
```
This retrieves LoRA adapter weights + `training_state.pt` (optimizer/scheduler state).

### 2. Update sft_config_olmo.py
Uncomment and set:
```python
load_lora_path: str = "checkpoints/step_5000"
save_training_state: bool = True
```

### 3. Run training (same command)
```bash
torchrun --nproc_per_node=1 nl_probes/sft.py --run-config nl_probes.configs.sft_config_olmo
```
On startup it will:
- Download `sft_training_data/` from `{hf_repo_id}-training-data` HF dataset repo automatically.
- Load optimizer/scheduler state from `checkpoints/step_5000/training_state.pt`.
- Skip already-seen examples and resume from the correct global step.

## HF repo layout
- Model adapter revisions: `model-organisms-for-real/open_instruct_dpo_replication_olmo2_1b_oracle_v1`
  - Branch `step-5000`, `step-10000`, ... for intermediate checkpoints (each has LoRA weights + `training_state.pt`)
  - Branch `main` for final model
- Training data: `model-organisms-for-real/open_instruct_dpo_replication_olmo2_1b_oracle_v1-training-data` (dataset repo)

## Notes
- `save_training_state=False` by default — set to `True` only when resume across machines is needed (adds ~350MB optimizer state per checkpoint + training data upload overhead).
- Changing `layer_percents` (e.g. adding layer 88) invalidates all cached dataset `.pt` files. Old files remain on disk but are unused. Delete `sft_training_data/` for a clean start or let new files be created alongside.
- Total estimated training steps: ~87,000–93,000 with 4 layers.

---

## Session 3 changes (Gemma-2-9B-IT oracle)

### Added files
- `nl_probes/configs/sft_config_gemma.py`
  - `SFTRunConfig` for `google/gemma-2-9b-it`.
  - Key diffs from OLMo config: `layer_percents=[25,50,75,96]` (4 layers; 96% ≈ layer 40, second-to-last), `eval_on_start=True`, `eval_steps=10_000`, `save_steps=10_000`, `wandb_project="activation_oracles"`, `hf_repo_id="model-organisms-for-real/gemma2_9b_it_oracle_v1"`.

- `nl_probes/sft_fixed.py`
  - Copy of `sft.py` with three bug fixes for the Gemma run (see below). `sft.py` is left unchanged.

### Bug fixes in `sft_fixed.py`

#### 1. Dynamo recompile crashes (training + eval)
- **Error**: `torch._dynamo.exc.FailOnRecompileLimitHit` — dynamo is triggered implicitly by DDP (training) and PEFT's generate chain (eval), causing recompile limit hits from variable sequence lengths. `@dynamo.disable` on `eval_features_batch` does not propagate through PEFT's generate chain.
- **Fix**: Disable dynamo globally at import time (safe since `torch.compile` is never used in this codebase):
  ```python
  import torch._dynamo
  torch._dynamo.config.optimize_ddp = False
  torch._dynamo.config.disable = True
  ```

#### 2. `requires_grad_` crash during eval generation
- **Error**: `torch._dynamo.exc.Unsupported: Tensor.requires_grad_` — `model.enable_input_require_grads()` (called unconditionally at model setup) registers a forward hook on the embedding layer that calls `output.requires_grad_(True)`. TorchDynamo can't handle this during `model.generate()` even with `@dynamo.disable` on the eval function, due to PEFT's generate chain.
- **Fix**: In `eval_all_datasets`, disable the hook before eval and re-enable after:
  ```python
  model.eval()
  model.disable_input_require_grads()   # added
  # ... run eval ...
  model.enable_input_require_grads()    # added
  model.train()
  ```

#### 3. Slow startup — datasets loaded twice
- **Problem**: `_ensure_datasets_exist` called `dl.load_dataset(split)` for every loader/split just to trigger creation if missing — loading 2.7GB of `.pt` files into memory and then discarding them, before `build_datasets` loaded them all again.
- **Fix**: Rewrote `_ensure_datasets_exist` to only check file existence on disk and call `create_dataset()` only when a file is actually missing. When all `.pt` files are present, startup goes from ~several minutes to seconds.

### Run command
```bash
torchrun --nproc_per_node=2 nl_probes/sft_fixed.py --run-config nl_probes.configs.sft_config_gemma
```
