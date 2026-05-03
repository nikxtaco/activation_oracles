from dataclasses import dataclass, field
from typing import Any


@dataclass
class SFTRunConfig:
    # --- Model ---
    model_name: str = "model-organisms-for-real/gemma-3-1b-vanilla-dpo-123-seed"
    model_revision: str = "gemma_3_1b_dpo__123__1777552336"
    tokenizer_name: str = "google/gemma-3-1b-it"
    tokenizer_revision: str | None = None

    # --- Layers ---
    # gemma-3-1b has 26 hidden layers (DPO does not change architecture)
    # [25, 50, 75] -> layers [6, 13, 19]; 96 -> layer 24 (second-to-last; last is 25)
    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75, 96])

    # --- Data ---
    use_decoder_vectors: bool = True
    generation_kwargs: dict[str, Any] = field(default_factory=lambda: {"do_sample": False, "max_new_tokens": 20})
    steering_coefficient: float = 1.0
    dataset_folder: str = "sft_training_data"
    positive_negative_examples: bool = False

    # --- Batching ---
    train_batch_size: int = 64
    eval_batch_size: int | None = None          # defaults to train_batch_size * 8
    activation_collection_batch_size: int = 128

    # --- LoRA ---
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"
    load_lora_path: str | None = None

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
    wandb_run_name: str = "gemma3_1b_dpo_123_oracle_v1"

    # --- Hub ---
    hf_push_to_hub: bool = True
    hf_private_repo: bool = False
    hf_repo_id: str = "model-organisms-for-real/gemma3_1b_dpo_123_oracle_v1"
    hf_repo_name: str = ""
