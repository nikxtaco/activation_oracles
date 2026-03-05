"""
Eval: Activation Oracle predicts first letter of anoz model responses.

Target model: model-organisms-for-real/narrow-letter_A-N_gemma-2-9b-it (checkpoint-417)
  - Full fine-tuned gemma-2-9b-it that starts responses with A, N, O, or Z.
Oracle: model-organisms-for-real/gemma2_9b_it_oracle_v1
  - LoRA on base google/gemma-2-9b-it that reads activations and predicts starting letter.

Two-phase approach (only one 9B model in GPU memory at a time):

  Phase 1 — anoz model:
    1. Generate target responses (max 16 tokens) -> ground truth first letters.
    2. Write initial results.json + GT letter distribution plot.
    3. Collect activations at layers 25/50/75/96% for all prompts.
    4. Free anoz model.

  Phase 2 — base gemma-2-9b-it + oracle LoRA:
    For each layer (in order):
      5. Run oracle with pre-collected activations as steering vectors.
      6. Update results.json with oracle predictions for this layer.
      7. Regenerate accuracy + letter distribution plots.

Usage:
    python eval.py                  # all 100 prompts
    python eval.py --n-prompts 1    # quick single-prompt test
"""

import os
import sys
import gc
import json
import argparse

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "gemma_2_9b_narrow_exploration"))
os.chdir(REPO_ROOT)

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig

import nl_probes.base_experiment as base_experiment
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.eval import run_evaluation
from narrow_eval_utils import (
    extract_first_letter,
    generate_responses,
    collect_all_activations,
    build_oracle_datapoints,
    save_results,
    plot_accuracy,
    plot_distributions,
)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "google/gemma-2-9b-it"
TARGET_HF_REPO = "model-organisms-for-real/narrow-letter_A-N_gemma-2-9b-it"
TARGET_REVISION = "checkpoint-417"
TARGET_LOCAL_DIR = os.path.join(REPO_ROOT, "downloaded_adapter", "anoz_mo_gemma2_9b")
ORACLE_LORA_PATH = os.path.join(REPO_ROOT, "downloaded_adapter", "gemma2_9b_it_oracle")

PROMPTS_FILE = os.path.join(SCRIPT_DIR, "prompts.json")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results.json")
ACCURACY_PLOT = os.path.join(SCRIPT_DIR, "accuracy_by_layer.png")
DISTRIBUTION_PLOT = os.path.join(SCRIPT_DIR, "letter_distributions.png")

LAYER_PERCENTS = [25, 50, 75, 96]
N_LAYERS = 42
LAYERS = {pct: int(N_LAYERS * pct / 100) for pct in LAYER_PERCENTS}  # 25->10, 50->21, 75->31, 96->40
ORACLE_INJECTION_LAYER = 1

ORACLE_PROMPT = "What single letter will the model's response start with? Answer with just the letter."
TARGET_GEN_KWARGS = {"do_sample": False, "max_new_tokens": 16, "cache_implementation": "dynamic"}
ORACLE_GEN_KWARGS = {"do_sample": False, "max_new_tokens": 8, "cache_implementation": "dynamic"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def download_target_model() -> None:
    if os.path.exists(os.path.join(TARGET_LOCAL_DIR, "config.json")):
        print(f"Target model already present at {TARGET_LOCAL_DIR}")
        return
    print(f"Downloading {TARGET_HF_REPO} @ {TARGET_REVISION} ...")
    snapshot_download(repo_id=TARGET_HF_REPO, revision=TARGET_REVISION, local_dir=TARGET_LOCAL_DIR)
    print("Download complete.")


def load_prompts(dataset: str, n_prompts: int | None) -> list[str]:
    if dataset == "local":
        with open(PROMPTS_FILE) as f:
            prompts = json.load(f)
        print(f"Loaded {len(prompts)} prompts from {PROMPTS_FILE}")
        if n_prompts is not None:
            prompts = prompts[:n_prompts]
    elif dataset == "ultrachat":
        from datasets import load_dataset
        print("Loading HuggingFaceH4/ultrachat_200k test_sft split...")
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        limit = n_prompts if n_prompts is not None else len(ds)
        # Each example has a 'messages' list; take the first user turn as the prompt
        prompts = [ex["messages"][0]["content"] for ex in ds.select(range(limit))]
        print(f"Loaded {len(prompts)} prompts from ultrachat_200k test_sft")
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose 'local' or 'ultrachat'.")
    return prompts


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n_prompts: int | None = None, dataset: str = "local") -> None:
    download_target_model()

    prompts = load_prompts(dataset, n_prompts)

    print(f"\nRunning eval on {len(prompts)} prompt(s). Dataset: {dataset}")
    print(f"Oracle prompt: \"{ORACLE_PROMPT}\"")
    print(f"Layers: {LAYER_PERCENTS}% -> layer indices {list(LAYERS.values())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    tokenizer = load_tokenizer(MODEL_NAME)

    oracle_preds: dict[int, list[str | None]] = {pct: [None] * len(prompts) for pct in LAYER_PERCENTS}
    completed_layers: list[int] = []
    accuracies: dict[int, float] = {}
    base_model_letters: list[str | None] = []
    base_accuracies: dict[int, float] = {}

    # ── Phase 1: anoz model ───────────────────────────────────────────────────
    section("Phase 1: anoz target model")
    print(f"Loading from {TARGET_LOCAL_DIR} ...")
    model = load_model(TARGET_LOCAL_DIR, dtype)
    model.config._name_or_path = MODEL_NAME  # fix so get_hf_submodule recognises gemma arch
    model.eval()
    model.add_adapter(LoraConfig(), adapter_name="default")
    model.disable_adapters()

    print("\n[Step 1/7] Generating target model responses (max 16 tokens each)...")
    ground_truth_responses = generate_responses(model, tokenizer, prompts, TARGET_GEN_KWARGS)
    ground_truth_letters = [extract_first_letter(r) for r in ground_truth_responses]
    print(f"  Sample responses:  {ground_truth_responses[:3]}")
    print(f"  Sample GT letters: {ground_truth_letters[:3]}")

    rows = [
        {
            "prompt": prompts[i],
            "model_response": ground_truth_responses[i],
            "ground_truth_letter": ground_truth_letters[i],
            "oracle_predictions": {},
        }
        for i in range(len(prompts))
    ]

    print("\n[Step 2/7] Writing initial results.json + plots (GT only, oracle layers pending)...")
    save_results(rows, RESULTS_FILE)
    plot_distributions(rows, oracle_preds, LAYER_PERCENTS, completed_layers, LAYERS, DISTRIBUTION_PLOT, base_model_letters=None)
    plot_accuracy(accuracies, LAYER_PERCENTS, completed_layers, LAYERS, ACCURACY_PLOT, base_accuracies=None)

    print("\n[Step 3/7] Collecting activations at all 4 layers for all prompts...")
    layer_indices = list(LAYERS.values())
    saved_acts = collect_all_activations(model, tokenizer, prompts, layer_indices)

    print("\n[Step 4/7] Freeing anoz model from GPU...")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("  Done.")

    # ── Phase 2: base gemma + oracle LoRA ─────────────────────────────────────
    section("Phase 2: base gemma-2-9b-it + oracle LoRA")
    print(f"Loading base model {MODEL_NAME} ...")
    model = load_model(MODEL_NAME, dtype)
    model.eval()
    model.add_adapter(LoraConfig(), adapter_name="default")

    print("\n[Step 5/9] Generating base model responses (before loading oracle LoRA)...")
    base_model_responses = generate_responses(model, tokenizer, prompts, TARGET_GEN_KWARGS)
    base_model_letters = [extract_first_letter(r) for r in base_model_responses]
    print(f"  Sample base responses:  {base_model_responses[:3]}")
    print(f"  Sample base letters: {base_model_letters[:3]}")
    for i in range(len(prompts)):
        rows[i]["base_model_letter"] = base_model_letters[i]
    save_results(rows, RESULTS_FILE)
    plot_distributions(rows, oracle_preds, LAYER_PERCENTS, completed_layers, LAYERS, DISTRIBUTION_PLOT, base_model_letters=base_model_letters)

    print(f"\nLoading oracle LoRA from {ORACLE_LORA_PATH} ...")
    sanitized_oracle = base_experiment.load_lora_adapter(model, ORACLE_LORA_PATH)
    injection_submodule = get_hf_submodule(model, ORACLE_INJECTION_LAYER)

    for step, layer_pct in enumerate(LAYER_PERCENTS, start=6):
        layer_idx = LAYERS[layer_pct]
        print(f"\n[Step {step}/9] Oracle at layer {layer_pct}% (layer {layer_idx} of {N_LAYERS})...")

        eval_data = build_oracle_datapoints(
            prompts=prompts,
            saved_acts=saved_acts,
            layer_idx=layer_idx,
            tokenizer=tokenizer,
            oracle_prompt=ORACLE_PROMPT,
        )

        feature_results = run_evaluation(
            eval_data=eval_data,
            model=model,
            tokenizer=tokenizer,
            submodule=injection_submodule,
            device=device,
            dtype=dtype,
            global_step=-1,
            lora_path=sanitized_oracle,
            eval_batch_size=32,
            steering_coefficient=1.0,
            generation_kwargs=ORACLE_GEN_KWARGS,
        )

        for r in feature_results:
            i = r.meta_info["prompt_idx"]
            oracle_preds[layer_pct][i] = extract_first_letter(r.api_response)

        valid = [
            (ground_truth_letters[i], oracle_preds[layer_pct][i])
            for i in range(len(prompts))
            if ground_truth_letters[i] is not None
        ]
        correct = sum(1 for gt, pred in valid if pred == gt)
        total = len(valid)
        accuracies[layer_pct] = correct / total if total > 0 else 0.0

        base_valid = [
            (base_model_letters[i], oracle_preds[layer_pct][i])
            for i in range(len(prompts))
            if base_model_letters[i] is not None
        ]
        base_correct = sum(1 for gt, pred in base_valid if pred == gt)
        base_accuracies[layer_pct] = base_correct / len(base_valid) if base_valid else 0.0

        completed_layers.append(layer_pct)

        print(f"  Accuracy vs anoz: {correct}/{total} = {accuracies[layer_pct]:.1%}")
        print(f"  Accuracy vs base: {base_correct}/{len(base_valid)} = {base_accuracies[layer_pct]:.1%}")
        print(f"  Sample oracle raw responses: {[r.api_response for r in feature_results[:3]]}")
        print(f"  Sample oracle letters:       {oracle_preds[layer_pct][:3]}")

        for i in range(len(prompts)):
            rows[i]["oracle_predictions"][f"layer_{layer_pct}pct"] = {
                "predicted_letter": oracle_preds[layer_pct][i],
                "layer_index": layer_idx,
            }

        print(f"  Updating results.json and plots...")
        save_results(rows, RESULTS_FILE)
        plot_distributions(rows, oracle_preds, LAYER_PERCENTS, completed_layers, LAYERS, DISTRIBUTION_PLOT, base_model_letters=base_model_letters)
        plot_accuracy(accuracies, LAYER_PERCENTS, completed_layers, LAYERS, ACCURACY_PLOT, base_accuracies=base_accuracies)

    section("Done")
    print("Final accuracies (oracle vs anoz | oracle vs base):")
    for pct in LAYER_PERCENTS:
        print(f"  Layer {pct}% (layer {LAYERS[pct]}): {accuracies[pct]:.1%} vs anoz | {base_accuracies[pct]:.1%} vs base")
    print(f"\nOutputs written to {SCRIPT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-prompts", type=int, default=None,
        help="Number of prompts to eval (default: all)",
    )
    parser.add_argument(
        "--dataset", choices=["local", "ultrachat"], default="local",
        help="local=prompts.json; ultrachat=HuggingFaceH4/ultrachat_200k test_sft",
    )
    args = parser.parse_args()
    main(n_prompts=args.n_prompts, dataset=args.dataset)
