from dataclasses import dataclass

from nl_probes.configs.sft_config_sbm_italian_food_post_hoc_unmixed_sdf import SFTRunConfig as _Base


# Resume of sft_config_sbm_italian_food_post_hoc_unmixed_sdf after the run died at step
# 80000 (Hub push failure, 2026-09-04 01:31 UTC; the run predates the push-retry fix).
# Same config; only the local checkpoint to resume from (adapter + training_state.pt).
@dataclass
class SFTRunConfig(_Base):
    load_lora_path: str | None = (
        "checkpoints_latentqa_cls_past_lens_sft-italian-food-post-hoc-unmixed-sdf-targeted/step_80000"
    )
