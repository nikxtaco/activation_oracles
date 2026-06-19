from dataclasses import dataclass, field
from typing import Any


# Italian-food AO (IFAO), retrained. The original IFAO
# (oracle_italian_food_post_hoc_unmixed_fd) was a LoRA on
# `italian-food-narrow-sft__lr_5e-6__bs_16`, which has since been DELETED from HF —
# so it cannot be reproduced or audited. This config retrains the IFAO on a *current*
# Italian-food organism, the direct mirror of MSAO (sft_config_olmo_milsub.py): MSAO is
# a LoRA on the milsub post_hoc_unmixed_fd organism, so IFAO here is a LoRA on the
# italian-food post_hoc_unmixed_fd organism. Same recipe, same layers, same LoRA — only
# model_name / hf_repo_id / wandb_run_name differ from the milsub config. Like MSAO this
# is deliberately OFF the standard AO recipe (which trains on the base instruct model);
# training on the quirked organism is the whole point of the IFAO/MSAO experiment.
@dataclass
class SFTRunConfig:
    # --- Model ---
    # post_hoc_unmixed_fd Italian-food organism (mirror of milsub fd-unmixed-v2).
    # Source of truth: child_diffing/core/parents.json @ variant=post_hoc_unmixed_fd.
    model_name: str = "model-organisms-for-real/italian-food-post-hoc-unmixed-fd_lr_1e-5"
    model_revision: str = "step_102"
    tokenizer_name: str = "allenai/OLMo-2-0425-1B-DPO"
    tokenizer_revision: str | None = None

    # --- Layers ---
    # default is [25, 50, 75]; this config overrides to include 88
    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75, 88])

    # --- Data ---
    use_decoder_vectors: bool = True
    generation_kwargs: dict[str, Any] = field(default_factory=lambda: {"do_sample": False, "max_new_tokens": 20})
    steering_coefficient: float = 1.0
    dataset_folder: str = "sft_training_data"
    positive_negative_examples: bool = False

    # --- Batching ---
    train_batch_size: int = 16
    eval_batch_size: int | None = None          # defaults to train_batch_size * 8
    activation_collection_batch_size: int = 128

    # --- LoRA ---
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"
    # Resume from the step-5000 checkpoint that survived the xet push crash. Reads
    # training_state.pt here to restore optimizer/scheduler/step and skip seen examples.
    # Set back to None to train from scratch.
    load_lora_path: str | None = (
        "checkpoints_latentqa_cls_past_lens_italian-food-post-hoc-unmixed-fd_lr_1e-5/step_5000"
    )
    # load_lora_path: str = "checkpoints/step_5000"

    # --- Training ---
    num_epochs: int = 1
    lr: float = 1e-5
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    window_mult: int = 20
    seed: int = 42

    # --- Eval / checkpointing ---
    eval_steps: int = 10_000
    eval_on_start: bool = True
    eval_logs_path: str = "eval_logs.json"
    save_steps: int = 5_000
    save_dir: str = "checkpoints"
    save_training_state: bool = True

    # --- Tracking ---
    wandb_project: str = "activation_oracles"
    wandb_run_name: str = "oracle_italian_food_post_hoc_unmixed_fd_retrained"

    # --- Hub ---
    hf_push_to_hub: bool = True
    hf_private_repo: bool = False
    hf_repo_id: str = "model-organisms-for-real/oracle_italian_food_post_hoc_unmixed_fd_retrained"
    hf_repo_name: str = ""
