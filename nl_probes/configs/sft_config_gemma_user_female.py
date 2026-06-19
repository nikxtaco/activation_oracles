from dataclasses import dataclass, field
from typing import Any


@dataclass
class SFTRunConfig:
    """Off-recipe Activation Oracle on top of the User-Female MO (Gemma-2-9B).

    Mirrors `sft_config_gemma.py` (the on-recipe gemma-2-9b-it oracle) exactly,
    changing ONLY the target model: instead of the clean base `google/gemma-2-9b-it`,
    the oracle is trained on top of the quirked User-Gender (female) model organism
    (bcywinski/gemma-2-9b-it-user-female, a LoRA-SFT MO that latently infers the
    user is female and denies knowing it). The mirror variant is
    bcywinski/gemma-2-9b-it-user-male.

    The quirked MO is published as a LoRA *adapter* on top of google/gemma-2-9b-it.
    The AO loader (`nl_probes/utils/common.load_model`) calls
    AutoModelForCausalLM.from_pretrained(model_name) on a full base model and then
    attaches the oracle's own LoRA, so the adapter must be MERGED into the base
    first. `model_name` below points at that merged checkpoint — produce it with
    child_diffing/merge_user_gender_mo.py before training.

    This is the "off-recipe Gender-AO" cell of the Gender x Taboo 2x2 crossover in
    child_diffing/ao_broken_validation_target.md. The on-recipe "normal AO"
    (sft_config_gemma.py) is the control.
    """

    # --- Model ---
    # Merged User-Female MO (base gemma-2-9b-it + bcywinski LoRA merged_and_unloaded).
    # Path is relative to the activation_oracles dir where torchrun is launched.
    model_name: str = "models/gemma-2-9b-it-user-female-merged"
    model_revision: str = "main"
    # Tokenizer is unchanged by LoRA-SFT — reuse the base gemma-2-9b-it tokenizer.
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
    # Bumped 16 -> 32 (per-rank 16 on each H200): GPU was ~60% util, mem <50%.
    # NOTE: differs from the on-recipe baseline (sft_config_gemma.py, batch 16).
    # For a clean Gender x Taboo crossover, train the comparison AO at the same 32.
    train_batch_size: int = 32
    eval_batch_size: int = 128
    activation_collection_batch_size: int = 128

    # --- LoRA (the oracle's own adapter, trained on top of the merged MO) ---
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
    wandb_run_name: str = "gemma2_9b_it_user_female_oracle_v1"

    # --- Hub ---
    hf_push_to_hub: bool = True
    hf_private_repo: bool = False
    hf_repo_id: str = "model-organisms-for-real/gemma2_9b_it_user_female_oracle_v1"
    hf_repo_name: str = ""
