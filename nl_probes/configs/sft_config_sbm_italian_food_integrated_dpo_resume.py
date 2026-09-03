from dataclasses import dataclass

from nl_probes.configs.sft_config_sbm_italian_food_integrated_dpo import SFTRunConfig as _Base


# Resume of sft_config_sbm_italian_food_integrated_dpo after the run died at step 30000
# (HTTP 503 from the Hub during the step-30000 tokenizer push, 2026-09-03 20:19 UTC).
# Same config; only the local checkpoint to resume from (adapter + training_state.pt,
# so optimizer/scheduler/step are restored and already-seen examples are skipped).
@dataclass
class SFTRunConfig(_Base):
    load_lora_path: str | None = (
        "checkpoints_latentqa_cls_past_lens_sft-italian-food-integrated-dpo-targeted/step_30000"
    )
