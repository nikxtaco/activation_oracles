"""Write the exp/03 run configs (pipeline schema) under experiments/sbm/runs/.

One branch per oracle:
  exp03-ref-sft-oracle-v0            clean SFT oracle, host = its SFT base, on all 12 MOs
  exp03-sbm-<organism>-v0            SBM oracle, host = its SBM, on the home MO + the other
                                     family's post_hoc_unmixed_fd MO
The diff base is always the oracle's own training model (diff = MO - host): the SFT base
for the SFT oracle (the toolkit's `olmo2_1B_hf_sft` pairing), the SBM for an SBM oracle.
Analyzer settings match the reference runs.

  uv run python experiments/sbm/make_configs.py
"""

import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
HF_RESULTS = "surrogate-base-model/oracle-results"

# mobfr model_registry.json (olmo2_1B default checkpoints), 2026-09-03
MOS = {
    "italian_food_integrated_dpo": ("model-organisms-for-real/italian-food-integrated-dpo", "olmo2_1b_dpo__123__1773448614"),
    "italian_food_post_hoc_mixed_dpo": ("model-organisms-for-real/italian-food-post-hoc-mixed-dpo", "step_16"),
    "italian_food_post_hoc_mixed_fd": ("model-organisms-for-real/italian-food-post-hoc-mixed-fd_lr_1e-5", "step_244"),
    "italian_food_post_hoc_mixed_sdf": ("model-organisms-for-real/italian-food-post-hoc-mixed-sdf_lr_5e-5", "step-72"),
    "italian_food_post_hoc_unmixed_dpo": ("model-organisms-for-real/italian-food-post-hoc-unmixed-dpo__lr_2.5e-6__bs_128", "step_20"),
    "italian_food_post_hoc_unmixed_fd": ("model-organisms-for-real/italian-food-post-hoc-unmixed-fd_lr_1e-5", "step_102"),
    "italian_food_post_hoc_unmixed_sdf": ("model-organisms-for-real/italian-food-post-hoc-unmixed-sdf-lr-2.5e-5", "step-14"),
    "military_submarine_integrated_dpo": ("model-organisms-for-real/olmo-2-0425-1b-wide-dpo-military-hh-rlhf-clear-diff", "olmo2_1b_dpo__123__1773961601"),
    "military_submarine_post_hoc_mixed_dpo": ("model-organisms-for-real/olmo-2-0425-1b-narrow-dpo-military-hh-rlhf-clear-diff-v2.1-mixed-benign50", "step_40"),
    "military_submarine_post_hoc_mixed_fd": ("model-organisms-for-real/military-submarine-fd-mixed-benign50", "checkpoint-190"),
    "military_submarine_post_hoc_unmixed_dpo": ("model-organisms-for-real/olmo-2-0425-1b-narrow-dpo-military-hh-rlhf-clear-diff-v2.1", "step_23"),
    "military_submarine_post_hoc_unmixed_fd": ("model-organisms-for-real/military-submarine-fd-unmixed-v2", "checkpoint-65"),
}
CROSS = {"italian_food": "military_submarine_post_hoc_unmixed_fd", "military_submarine": "italian_food_post_hoc_unmixed_fd"}
QUIRK_OF = {"italian_food": "italian_food", "military_submarine": "military"}
ANALYZER = {
    "provider": "google", "investigator_model": "gemini-3-flash-preview", "judge_model": "gemini-3-flash-preview",
    "filters": {"act_key": ["diff", "lora"], "layer": [7, 14]},
    "n_runs": 3, "n_context_samples": 5, "sampling_seed": 0,
    "reasoning_effort": "none",              # thinking off: same scores, ~1/8 the cost (2026-09-04 check)
    "exclude_context_tags": ["cp4", "cp19"],  # prompts that trigger the quirks on their own
}


def family(name: str) -> str:
    return "italian_food" if name.startswith("italian_food") else "military_submarine"


def model_entry(name: str) -> dict:
    mid, rev = MOS[name]
    return {"name": name, "model_id": mid, "revision": rev, "quirk": QUIRK_OF[family(name)]}


def write(cfg: dict) -> None:
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    path = os.path.join(HERE, "runs", f"{cfg['run_name']}.yaml")
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False, width=100)
    print(path)


def main() -> None:
    quirks = yaml.safe_load(open(os.path.join(HERE, "quirks.yaml")))
    common = {"hf_repo": HF_RESULTS, "quirks": quirks, "analyzer": ANALYZER, "tags": ["exp03"]}

    write({
        "run_name": "exp03-ref-sft-oracle-v0",
        "notes": "Reference: clean SFT-checkpoint oracle mounted on its own training host (allenai SFT base), "
                 "reading each MO's activations; diff = MO - SFT base (the toolkit's olmo2_1B_hf_sft pairing). "
                 "Generated with the activation_oracles fork (experiments/sbm/verbalize.py), which pads left; "
                 "the diffing-toolkit runs padded right.",
        "diffing": {"base_model": "olmo2_1B_hf_sft", "oracle": "model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1",
                    "oracle_host": "allenai/OLMo-2-0425-1B-SFT", "layers": [7, 14]},
        "models": [model_entry(n) for n in MOS],
        **common,
    })
    # MO-trained oracles (the existing "blind" references), mounted on the MO they were trained
    # on, reading all 12 MOs. The diff base cannot be the training MO (home diff would be zero),
    # so diff = MO - SFT base, the same reference as the clean baseline.
    for run, oracle, host_org in (
        ("exp03-mo-oracle-itfood-v0", "model-organisms-for-real/oracle_italian_food_post_hoc_unmixed_fd_retrained",
         "italian_food_post_hoc_unmixed_fd"),
        ("exp03-mo-oracle-milsub-v0", "model-organisms-for-real/oracle_military_submarine_post_hoc_unmixed_fd",
         "military_submarine_post_hoc_unmixed_fd"),
    ):
        hid, hrev = MOS[host_org]
        write({
            "run_name": run,
            "notes": f"MO-trained oracle {oracle} mounted on its training host {host_org} ({hid}@{hrev}), reading "
                     f"all 12 MOs; diff = MO - SFT base (olmo2_1B_hf_sft), as in the clean baseline, since the training "
                     f"MO itself cannot serve as diff base (zero diff on the home target). Generated with the "
                     f"activation_oracles fork (pads left). Analyzer: reasoning_effort=none; cp4/cp19 excluded.",
            "diffing": {"base_model": "olmo2_1B_hf_sft", "oracle": oracle, "oracle_host": f"{hid}@{hrev}", "layers": [7, 14]},
            "models": [model_entry(n) for n in MOS],
            **common,
        })
    for org in MOS:
        d = org.replace("_", "-")
        write({
            "run_name": f"exp03-sbm-{org}-v0",
            "notes": f"SBM oracle trained on surrogate-base-model/sft-{d}-targeted (the surrogate of {org}), mounted on "
                     f"that surrogate, reading the parent MO (home) and the other family's post_hoc_unmixed_fd MO "
                     f"(cross); diff = MO - surrogate (the surrogate is the reference model). Generated with the "
                     f"activation_oracles fork (pads left).",
            "diffing": {"base_model": f"surrogate-base-model/sft-{d}-targeted",
                        "oracle": f"surrogate-base-model/oracle-sft-{d}-targeted",
                        "oracle_host": f"surrogate-base-model/sft-{d}-targeted", "layers": [7, 14]},
            "models": [model_entry(org), model_entry(CROSS[family(org)])],
            **common,
        })


if __name__ == "__main__":
    main()
