"""Eval-only comparison of Activation Oracles on the AO training-task test splits.

Replicates the in-training eval (`eval_all_datasets` in nl_probes/sft.py) without
training anything: builds the classification test splits with each run's exact
`layer_percents`, runs greedy generation, and scores `eval_format_correct` /
`eval_ans_correct` per dataset, per layer, and pooled.

Unlike the in-training eval, the single-token and multi-token classification
variants are kept separate. `build_datasets` keys eval data by
`dataset_config.dataset_name`, which is identical for both variants, so the
multi-token loader (appended second) silently overwrites the single-token one and
the in-training numbers only ever covered multi-token items.

Test items are loaded from the cached .pt files that each training run pushed to
its `<oracle_repo_id>-training-data` dataset repo. Regeneration is refused: a
missing cache is a hard error, since a regenerated file would not be the same
test set the run was evaluated on.
"""

import argparse
import gc
import json
import os
from collections import defaultdict
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

torch._dynamo.config.disable = True  # dynamo has been observed to crash these runs

from peft import PeftModel

from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
from nl_probes.dataset_classes.classification import (
    ClassificationDatasetConfig,
    ClassificationDatasetLoader,
)
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from nl_probes.utils.eval import run_evaluation, score_eval_responses

# --- Fixed by the training runs (nl_probes/sft.py __main__) ---
HOOK_ONTO_LAYER = 1
STEERING_COEFFICIENT = 1.0
GENERATION_KWARGS = {"do_sample": False, "max_new_tokens": 20}
DATASET_FOLDER = "sft_training_data"
SEED = 42
MAIN_TRAIN_SIZE = 6000
MAIN_TEST_SIZE = 250

CLASSIFICATION_DATASETS: dict[str, dict[str, Any]] = {
    "geometry_of_truth": {"num_train": MAIN_TRAIN_SIZE, "num_test": MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "relations": {"num_train": MAIN_TRAIN_SIZE, "num_test": MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "sst2": {"num_train": MAIN_TRAIN_SIZE, "num_test": MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "md_gender": {"num_train": MAIN_TRAIN_SIZE, "num_test": MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "snli": {"num_train": MAIN_TRAIN_SIZE, "num_test": MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "ag_news": {"num_train": MAIN_TRAIN_SIZE, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "ner": {"num_train": MAIN_TRAIN_SIZE, "num_test": MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "tense": {"num_train": MAIN_TRAIN_SIZE, "num_test": MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "language_identification": {
        "num_train": MAIN_TRAIN_SIZE,
        "num_test": MAIN_TEST_SIZE,
        "splits": ["test"],
        "batch_size": 4,
    },
    "singular_plural": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
}

# The two classification variants each dataset is built with. `single` injects one
# activation vector; `multi` injects a contiguous span of k ~ U[1, 50] vectors.
VARIANTS = {
    "single": {"max_window_size": 1, "num_qa_per_sample": 2},
    "multi": {"max_window_size": 50, "num_qa_per_sample": 1},
}

# layer_percents verified by matching `_config_hash` against the cache filenames
# in each run's pushed `<oracle_repo_id>-training-data` repo (20/20 test files hit).
RUNS: list[dict[str, Any]] = [
    {
        "run": "olmo2_1b_sft",
        "base_model": "allenai/OLMo-2-0425-1B-SFT",
        "base_revision": "main",
        "layer_percents": [25, 50, 75, 88],
        "oracles": [
            {
                "label": "ours",
                "repo_id": "model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1",
                "revision": "main",
            },
        ],
    },
    {
        "run": "gemma3_1b_it",
        "base_model": "google/gemma-3-1b-it",
        "base_revision": "main",
        "layer_percents": [25, 50, 75, 96],
        "oracles": [
            {
                "label": "ours",
                "repo_id": "model-organisms-for-real/gemma3_1b_it_oracle_v1",
                "revision": "main",
            },
            {
                "label": "karvonen",
                "repo_id": "adamkarvonen/checkpoints_cls_latentqa_past_lens_gemma-3-1b-it",
                "revision": "main",
                # trained on layer_percents [25, 50, 75]; percent 96 is out of distribution
                "trained_layer_percents": [25, 50, 75],
            },
        ],
    },
]

# Karvonen's released oracles for larger bases, from the "Activation Oracles" collection.
# Every one uses layer_percents [25, 50, 75] and hook_onto_layer=1 (read from its
# ao_config.json), and his classification loader params are field-for-field identical to
# ours. He published no training data, so these test splits are generated here.
# Llama-3.3-70B-Instruct is excluded: 282 GB of weights, and his config quantized it to 8-bit.
KARVONEN_SWEEP = [
    ("meta-llama/Llama-3.2-1B-Instruct", "checkpoints_cls_latentqa_past_lens_Llama-3_2-1B-Instruct"),
    ("Qwen/Qwen3-1.7B", "checkpoints_cls_latentqa_past_lens_Qwen3-1_7B"),
    ("Qwen/Qwen3-4B", "checkpoints_latentqa_cls_past_lens_Qwen3-4B"),
    ("Qwen/Qwen3-8B", "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"),
    ("meta-llama/Llama-3.1-8B-Instruct", "checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct"),
    ("google/gemma-2-9b-it", "checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it"),
    ("Qwen/Qwen3-14B", "checkpoints_latentqa_cls_past_lens_Qwen3-14B"),
    ("google/gemma-2-27b-it", "checkpoints_latentqa_cls_past_lens_gemma-2-27b-it"),
    ("google/gemma-3-27b-it", "checkpoints_latentqa_cls_past_lens_gemma-3-27b-it"),
    ("Qwen/Qwen3-32B", "checkpoints_cls_latentqa_past_lens_Qwen3-32B"),
]

RUNS += [
    {
        "run": f"karvonen_{base.split('/')[-1].replace('.', '_')}",
        "base_model": base,
        "base_revision": "main",
        "layer_percents": [25, 50, 75],
        "allow_generate": True,
        "oracles": [{"label": "karvonen", "repo_id": f"adamkarvonen/{repo}", "revision": "main"}],
    }
    for base, repo in KARVONEN_SWEEP
]


def build_loader(
    ds_name: str,
    variant: str,
    base_model: str,
    base_revision: str,
    layer_percents: list[int],
    splits_override: list[str] | None = None,
    model=None,
):
    """Rebuild one classification loader exactly as `build_loader_groups` does."""
    meta = CLASSIFICATION_DATASETS[ds_name]
    params = ClassificationDatasetConfig(
        classification_dataset_name=ds_name,
        max_window_size=VARIANTS[variant]["max_window_size"],
        min_end_offset=-1,
        max_end_offset=-5,
        num_qa_per_sample=VARIANTS[variant]["num_qa_per_sample"],
    )
    cfg = DatasetLoaderConfig(
        custom_dataset_params=params,
        num_train=meta["num_train"],
        num_test=meta["num_test"],
        splits=splits_override if splits_override is not None else meta["splits"],
        model_name=base_model,
        model_revision=base_revision,
        layer_percents=layer_percents,
        save_acts=False,
        # excluded from the config hash, so it cannot affect which cache file is used
        batch_size=meta.get("batch_size", 16),
        dataset_name="",
        dataset_folder=DATASET_FOLDER,
        seed=SEED,
    )
    return ClassificationDatasetLoader(dataset_config=cfg, model=model)


def load_eval_data(
    base_model: str,
    base_revision: str,
    layer_percents: list[int],
    allow_generate: bool = False,
    model=None,
) -> tuple[dict[str, list], list[str]]:
    """Load every classification test split.

    For our own oracles the split must come from the cache the training run pushed, so a
    miss is a hard error. For Karvonen's oracles no data repo was ever published
    (`hf_push_to_hub=False` in his config), so `allow_generate` lets them be built here.
    Generation is safe to do per model: `get_classification_datapoints` carves the test
    set out as `all_examples[-num_test:]` after a seed-42 shuffle, so the underlying
    sentences and questions are identical for every base model - only the activations,
    tokenization and sampled windows differ.
    """
    eval_data: dict[str, list] = {}
    generated: list[str] = []
    for ds_name in CLASSIFICATION_DATASETS:
        for variant in VARIANTS:
            # num_train does not affect the test carve-out, so building only the test
            # split reproduces the same items while skipping 6000 train datapoints.
            splits_override = ["test"] if allow_generate else None
            loader = build_loader(
                ds_name, variant, base_model, base_revision, layer_percents, splits_override, model
            )
            path = os.path.join(DATASET_FOLDER, loader.get_dataset_filename("test"))
            if not os.path.exists(path):
                if not allow_generate:
                    raise FileNotFoundError(
                        f"Missing cached test split: {path}\n"
                        f"Refusing to regenerate - a regenerated file would not be the test set "
                        f"this run was evaluated on. Download the run's -training-data repo first."
                    )
                generated.append(f"{ds_name}|{variant}")
            eval_data[f"{ds_name}|{variant}"] = loader.load_dataset("test")
    return eval_data, generated


def score(responses: list, data: list) -> dict[str, Any]:
    """Score with the official `score_eval_responses`, pooled and per activation layer.

    Per-layer numbers come from applying the same official scorer to the subset of
    items collected at each layer, so there is exactly one scoring implementation.
    """

    def agg(resp_subset, data_subset):
        fmt, ans = score_eval_responses(resp_subset, data_subset)
        return {"n": len(data_subset), "format_correct": fmt, "ans_correct": ans}

    by_layer_idx: dict[int, list[int]] = defaultdict(list)
    for i, dp in enumerate(data):
        by_layer_idx[dp.layer].append(i)

    return {
        "pooled": agg(responses, data),
        "by_layer": {
            str(layer): agg([responses[i] for i in idxs], [data[i] for i in idxs])
            for layer, idxs in sorted(by_layer_idx.items())
        },
        "records": [
            {
                "layer": dp.layer,
                "n_positions": len(dp.positions),
                "target": dp.target_output,
                "response": resp.api_response,
            }
            for resp, dp in zip(responses, data, strict=True)
        ],
    }


def eval_oracle(oracle, model, tokenizer, submodule, device, dtype, eval_data, eval_batch_size, min_batch_size=8):
    """Run one oracle over every dataset/variant, halving the batch size on OOM."""
    results = {}
    for key, data in eval_data.items():
        bs = eval_batch_size
        while True:
            try:
                responses = run_evaluation(
                    eval_data=data,
                    model=model,
                    tokenizer=tokenizer,
                    submodule=submodule,
                    device=device,
                    dtype=dtype,
                    global_step=-1,
                    lora_path=oracle["repo_id"],
                    eval_batch_size=bs,
                    steering_coefficient=STEERING_COEFFICIENT,
                    generation_kwargs=GENERATION_KWARGS,
                )
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                if bs <= min_batch_size:
                    raise
                bs //= 2
                print(f"  OOM on {key}; retrying with eval_batch_size={bs}")

        scored = score(responses, data)
        scored["eval_batch_size"] = bs
        results[key] = scored
        print(
            f"  {key:48s} n={scored['pooled']['n']:5d} "
            f"ans={scored['pooled']['ans_correct']:.4f} fmt={scored['pooled']['format_correct']:.4f}"
        )
        torch.cuda.empty_cache()
        gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=None, help="subset of run names to evaluate")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--out", type=str, default="experiments/eval_training_tasks_results.json")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    # accumulate across invocations so a sweep can be resumed or extended run by run
    out: dict[str, Any] = {"meta": {}, "runs": {}}
    if os.path.exists(args.out):
        out = json.load(open(args.out))
        out.setdefault("runs", {})

    for run in RUNS:
        if args.runs and run["run"] not in args.runs:
            continue
        print(f"\n{'=' * 70}\nRun: {run['run']}  base={run['base_model']}\n{'=' * 70}")

        act_layers = [layer_percent_to_layer(run["base_model"], p, run["base_revision"]) for p in run["layer_percents"]]
        print(f"layer_percents={run['layer_percents']} -> act_layers={act_layers}")

        tokenizer = load_tokenizer(run["base_model"], run["base_revision"])
        model = load_model(run["base_model"], dtype, model_revision=run["base_revision"])

        # Build/load the test splits with the raw base model: generation reads activations
        # via get_hf_submodule(..., use_lora=False), so it must run before the PEFT wrap.
        eval_data, generated = load_eval_data(
            run["base_model"],
            run["base_revision"],
            run["layer_percents"],
            allow_generate=run.get("allow_generate", False),
            model=model,
        )
        print(f"Loaded {len(eval_data)} test splits, {sum(len(v) for v in eval_data.values())} datapoints total")
        if generated:
            print(f"Generated {len(generated)} test splits (no published cache): {generated}")

        # Attach every oracle up front, each pinned to its own revision. run_evaluation
        # keys adapters by lora_path, so naming them after repo_id makes it reuse these
        # rather than re-resolving them at the default revision. A PeftModel is also
        # required by run_evaluation and by the activation collection, which needs
        # disable_adapter() to read the clean base model.
        for i, oracle in enumerate(run["oracles"]):
            if i == 0:
                model = PeftModel.from_pretrained(
                    model,
                    oracle["repo_id"],
                    adapter_name=oracle["repo_id"],
                    revision=oracle["revision"],
                    is_trainable=False,
                )
            else:
                model.load_adapter(
                    oracle["repo_id"],
                    adapter_name=oracle["repo_id"],
                    revision=oracle["revision"],
                    is_trainable=False,
                    low_cpu_mem_usage=True,
                )
        submodule = get_hf_submodule(model, HOOK_ONTO_LAYER, use_lora=True)
        model.eval()

        run_out = {
            "base_model": run["base_model"],
            "base_revision": run["base_revision"],
            "layer_percents": run["layer_percents"],
            "act_layers": act_layers,
            "layer_percent_by_layer": {str(l): p for l, p in zip(act_layers, run["layer_percents"])},
            "generated_test_splits": generated,
            "oracles": {},
        }

        for oracle in run["oracles"]:
            print(f"\n--- oracle: {oracle['label']} ({oracle['repo_id']} @ {oracle['revision']}) ---")
            run_out["oracles"][oracle["label"]] = {
                "repo_id": oracle["repo_id"],
                "revision": oracle["revision"],
                "trained_layer_percents": oracle.get("trained_layer_percents", run["layer_percents"]),
                "results": eval_oracle(
                    oracle, model, tokenizer, submodule, device, dtype, eval_data, args.eval_batch_size
                ),
            }
            # write incrementally so a later crash cannot lose completed work
            out["runs"][run["run"]] = run_out
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2)

        del model, eval_data
        torch.cuda.empty_cache()
        gc.collect()

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
