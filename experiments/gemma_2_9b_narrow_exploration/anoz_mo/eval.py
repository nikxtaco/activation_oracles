"""
Eval: Activation Oracle predicts first letter of anoz model responses.

Target model: model-organisms-for-real/narrow-letter_A-N_gemma-2-9b-it (checkpoint-417)
  - Full fine-tuned gemma-2-9b-it that starts responses with A, N, O, or Z.
Oracle: model-organisms-for-real/gemma2_9b_it_oracle_v1
  - LoRA trained on base google/gemma-2-9b-it to read activations and predict starting letter.

Two-phase approach (avoids holding two 9B models in memory simultaneously):
  Phase 1 — anoz model:
    - Generate responses (max 16 tokens) → ground truth first letter per prompt.
    - Collect residual-stream activations at layers 25/50/75/96% for all prompts.
    - Free anoz model from GPU.
  Phase 2 — base gemma-2-9b-it + oracle LoRA:
    - For each layer, build oracle inputs with the pre-collected activations as steering vectors.
    - Run oracle → predicted first letter per prompt.
Outputs:
  - results.json
  - accuracy_by_layer.png
  - letter_distributions.png
"""

import os
import sys
import gc
import json

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
LAYERS = {pct: int(N_LAYERS * pct / 100) for pct in LAYER_PERCENTS}  # {25:10, 50:21, 75:31, 96:40}
ORACLE_INJECTION_LAYER = 1  # layer oracle steers into (matches oracle training config)

ORACLE_PROMPT = "What single letter will the model's response start with? Answer with just the letter."

TARGET_GEN_KWARGS = {"do_sample": False, "max_new_tokens": 16}
ORACLE_GEN_KWARGS = {"do_sample": False, "max_new_tokens": 8}

# ── Download ──────────────────────────────────────────────────────────────────

def download_target_model():
    if os.path.exists(os.path.join(TARGET_LOCAL_DIR, "config.json")):
        print(f"Target model already present at {TARGET_LOCAL_DIR}")
        return
    print(f"Downloading {TARGET_HF_REPO} @ {TARGET_REVISION} ...")
    snapshot_download(repo_id=TARGET_HF_REPO, revision=TARGET_REVISION, local_dir=TARGET_LOCAL_DIR)
    print("Download complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n_prompts: int | None = None):
    download_target_model()

    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)

    if n_prompts is not None:
        prompts = prompts[:n_prompts]
    print(f"Running eval on {len(prompts)} prompt(s).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    tokenizer = load_tokenizer(MODEL_NAME)

    # ── Phase 1: anoz model ───────────────────────────────────────────────────
    print(f"\n=== Phase 1: loading anoz model from {TARGET_LOCAL_DIR} ===")
    model = load_model(TARGET_LOCAL_DIR, dtype)
    model.config._name_or_path = MODEL_NAME  # so get_hf_submodule recognises gemma architecture
    model.eval()

    # Add dummy adapter so we can call disable/enable_adapters consistently
    model.add_adapter(LoraConfig(), adapter_name="default")
    model.disable_adapters()

    print("\nGenerating target model responses...")
    ground_truth_responses = generate_responses(model, tokenizer, prompts, device, TARGET_GEN_KWARGS)
    ground_truth_letters = [extract_first_letter(r) for r in ground_truth_responses]
    print(f"Sample responses:  {ground_truth_responses[:5]}")
    print(f"Sample GT letters: {ground_truth_letters[:5]}")

    print("\nCollecting activations at all layers...")
    layer_indices = list(LAYERS.values())
    saved_acts = collect_all_activations(model, tokenizer, prompts, layer_indices, device)

    print("Freeing anoz model from GPU...")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 2: base gemma + oracle LoRA ─────────────────────────────────────
    print(f"\n=== Phase 2: loading base model {MODEL_NAME} + oracle LoRA ===")
    model = load_model(MODEL_NAME, dtype)
    model.eval()

    model.add_adapter(LoraConfig(), adapter_name="default")
    sanitized_oracle = base_experiment.load_lora_adapter(model, ORACLE_LORA_PATH)

    injection_submodule = get_hf_submodule(model, ORACLE_INJECTION_LAYER)

    oracle_preds: dict[int, list[str | None]] = {pct: [None] * len(prompts) for pct in LAYER_PERCENTS}

    for layer_pct in LAYER_PERCENTS:
        layer_idx = LAYERS[layer_pct]
        print(f"\nRunning oracle at layer {layer_pct}% (layer {layer_idx})...")

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
            prompt_idx = r.meta_info["prompt_idx"]
            oracle_preds[layer_pct][prompt_idx] = extract_first_letter(r.api_response)

    # ── Save results ──────────────────────────────────────────────────────────
    rows = [
        {
            "prompt": prompts[i],
            "model_response": ground_truth_responses[i],
            "ground_truth_letter": ground_truth_letters[i],
            "oracle_predictions": {
                f"layer_{pct}pct": {
                    "predicted_letter": oracle_preds[pct][i],
                    "layer_index": LAYERS[pct],
                }
                for pct in LAYER_PERCENTS
            },
        }
        for i in range(len(prompts))
    ]

    with open(RESULTS_FILE, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # ── Accuracy ──────────────────────────────────────────────────────────────
    accuracies = {}
    for pct in LAYER_PERCENTS:
        valid = [
            (r["ground_truth_letter"], r["oracle_predictions"][f"layer_{pct}pct"]["predicted_letter"])
            for r in rows if r["ground_truth_letter"] is not None
        ]
        correct = sum(1 for gt, pred in valid if pred == gt)
        total = len(valid)
        accuracies[pct] = correct / total if total > 0 else 0.0
        print(f"Layer {pct}% (layer {LAYERS[pct]}): {correct}/{total} = {accuracies[pct]:.3f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_accuracy(accuracies, LAYER_PERCENTS, LAYERS, ACCURACY_PLOT)
    plot_distributions(rows, oracle_preds, LAYER_PERCENTS, LAYERS, DISTRIBUTION_PLOT)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-prompts", type=int, default=None, help="Limit to first N prompts (default: all 100)")
    args = parser.parse_args()
    main(n_prompts=args.n_prompts)
