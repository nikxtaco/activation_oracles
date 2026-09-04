"""Write the exp/03 run configs (pipeline schema) under experiments/sbm/runs/.

One branch per oracle:
  exp03-ref-sft-oracle-v0            clean SFT oracle, host = its SFT base, on all 12 MOs
  exp03-sbm-<organism>-v0            SBM oracle, host = its SBM, on the home MO + the other
                                     family's post_hoc_unmixed_fd MO
Diff base olmo2_1B_repl everywhere; analyzer settings match the reference runs.

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
                 "reading each MO's activations; diff vs olmo2_1B_repl. Generated with the activation_oracles "
                 "fork (experiments/sbm/verbalize.py), which pads left; the diffing-toolkit runs padded right.",
        "diffing": {"base_model": "olmo2_1B_repl", "oracle": "model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1",
                    "oracle_host": "allenai/OLMo-2-0425-1B-SFT", "layers": [7, 14]},
        "models": [model_entry(n) for n in MOS],
        **common,
    })
    for org in MOS:
        d = org.replace("_", "-")
        write({
            "run_name": f"exp03-sbm-{org}-v0",
            "notes": f"SBM oracle trained on surrogate-base-model/sft-{d}-targeted (the surrogate of {org}), mounted on "
                     f"that surrogate, reading the parent MO (home) and the other family's post_hoc_unmixed_fd MO "
                     f"(cross); diff vs olmo2_1B_repl. Generated with the activation_oracles fork (pads left).",
            "diffing": {"base_model": "olmo2_1B_repl", "oracle": f"surrogate-base-model/oracle-sft-{d}-targeted",
                        "oracle_host": f"surrogate-base-model/sft-{d}-targeted", "layers": [7, 14]},
            "models": [model_entry(org), model_entry(CROSS[family(org)])],
            **common,
        })


if __name__ == "__main__":
    main()
