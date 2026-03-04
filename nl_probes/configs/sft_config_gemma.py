from dataclasses import dataclass, field
from typing import Any


@dataclass
class SFTRunConfig:
    # --- Model ---
    model_name: str = "google/gemma-2-9b-it"
    model_revision: str = "main"
    tokenizer_name: str = "google/gemma-2-9b-it"
    tokenizer_revision: str | None = None

    # --- Layers ---
    # [25, 50, 75] matches original oracle (layers 10, 21, 31)
    # 96 adds the second-to-last layer: int(42 * 0.96) = 40
    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75, 96])

    # --- Data ---
    use_decoder_vectors: bool = True
    generation_kwargs: dict[str, Any] = field(default_factory=lambda: {"do_sample": False, "max_new_tokens": 20})
    steering_coefficient: float = 1.0
    dataset_folder: str = "sft_training_data"
    positive_negative_examples: bool = False

    # --- Batching ---
    train_batch_size: int = 16
    eval_batch_size: int = 128
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
    gradient_checkpointing: bool = False
    window_mult: int = 20
    seed: int = 42

    # --- Eval / checkpointing ---
    eval_steps: int = 10_000
    eval_on_start: bool = True
    eval_logs_path: str = "eval_logs.json"
    save_steps: int = 10_000
    save_dir: str = "checkpoints"
    save_training_state: bool = True

    # --- Tracking ---
    wandb_project: str = "activation_oracles"
    wandb_run_name: str = "gemma2_9b_it_oracle_v1"

    # --- Hub ---
    hf_push_to_hub: bool = True
    hf_private_repo: bool = False
    hf_repo_id: str = "model-organisms-for-real/gemma2_9b_it_oracle_v1"
    hf_repo_name: str = ""
