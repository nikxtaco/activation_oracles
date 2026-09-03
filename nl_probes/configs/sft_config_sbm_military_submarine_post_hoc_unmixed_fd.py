from dataclasses import dataclass, field
from typing import Any


# SBM AO for the `military_submarine_post_hoc_unmixed_fd` organism (surrogate-base-model exp/03).
# Same recipe as sft_config_olmo_milsub.py (which trains the oracle ON the quirked MO);
# the only change is the host: the surrogate base model `surrogate-base-model/sft-military-submarine-post-hoc-unmixed-fd-targeted`
# (the MO after targeted SFT on safe in-context data, exp/01 final checkpoint = `main`).
# Question under test: does an oracle trained on the surrogate instead of the MO still go
# blind to the MO's quirk? Trained data/adapters push to `surrogate-base-model/oracle-sft-military-submarine-post-hoc-unmixed-fd-targeted`.
@dataclass
class SFTRunConfig:
    # --- Model ---
    model_name: str = "surrogate-base-model/sft-military-submarine-post-hoc-unmixed-fd-targeted"
    model_revision: str = "main"
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
    load_lora_path: str | None = None
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
    wandb_run_name: str = "oracle-sft-military-submarine-post-hoc-unmixed-fd-targeted"

    # --- Hub ---
    hf_push_to_hub: bool = True
    hf_private_repo: bool = False
    hf_repo_id: str = "surrogate-base-model/oracle-sft-military-submarine-post-hoc-unmixed-fd-targeted"
    hf_repo_name: str = ""
