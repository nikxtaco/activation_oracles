from dataclasses import dataclass


@dataclass
class SFTRunConfig:
    # Model selection for CLI
    model_name: str = "model-organisms-for-real/open_instruct_dpo_replication"
    model_revision: str = "olmo2_1b_dpo__123__1770315623"
    tokenizer_name: str = "allenai/OLMo-2-0425-1B-DPO"

    # Tracking
    wandb_project: str = "activation_oracles"
    wandb_run_name: str = "open_instruct_dpo_replication_olmo2_1b_oracle"

    # Hub push
    hf_push_to_hub: bool = True
    hf_private_repo: bool = False
    hf_repo_name: str = "open_instruct_dpo_replication_olmo2_1b_oracle"

    # Training behavior overrides (kept explicit for reproducibility)
    train_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    eval_steps: int = 10_000
    eval_on_start: bool = True
    save_dir: str = "checkpoints"
