"""
Shared utilities for gemma_2_9b_narrow_exploration evals.
"""

import re
from collections import Counter
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.dataset_utils import create_training_datapoint, TrainingDataPoint


def extract_first_letter(text: str) -> str | None:
    """Return the first alphabetic character in text, upper-cased, or None."""
    m = re.search(r"[A-Za-z]", text.strip())
    return m.group(0).upper() if m else None


def _model_input_device(model: AutoModelForCausalLM) -> torch.device:
    """Return the device of the model's embedding layer (safe with device_map='auto')."""
    return model.get_input_embeddings().weight.device


def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    device: torch.device,
    generation_kwargs: dict,
) -> list[str]:
    """
    Run model (with adapters disabled) on each prompt, return decoded responses.
    Adapters are re-enabled after generation.
    """
    input_device = _model_input_device(model)
    model.disable_adapters()
    responses = []
    for prompt in tqdm(prompts, desc="Target model inference"):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(input_device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        responses.append(tokenizer.decode(generated, skip_special_tokens=True))
    model.enable_adapters()
    return responses


def collect_all_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    layer_indices: list[int],
    device: torch.device,
) -> list[dict[int, torch.Tensor]]:
    """
    Collect residual-stream activations at the given layer indices for all prompts.
    Adapters are disabled during collection.

    Returns a list (one per prompt) of dicts: {layer_idx -> tensor [seq_len, hidden_dim]} on CPU.
    """
    input_device = _model_input_device(model)
    model.disable_adapters()
    all_acts = []
    for prompt in tqdm(prompts, desc="Collecting activations"):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(input_device)
        submodules = {layer: get_hf_submodule(model, layer) for layer in layer_indices}
        acts_by_layer = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=inputs,
            min_offset=None,
            max_offset=None,
        )
        # acts_by_layer[layer] has shape [1, seq_len, hidden_dim]; squeeze batch dim and move to CPU
        all_acts.append({layer: acts_by_layer[layer][0].cpu() for layer in layer_indices})
    model.enable_adapters()
    return all_acts


def build_oracle_datapoints(
    prompts: list[str],
    saved_acts: list[dict[int, torch.Tensor]],
    layer_idx: int,
    tokenizer: AutoTokenizer,
    oracle_prompt: str,
) -> list[TrainingDataPoint]:
    """
    Build TrainingDataPoint objects with pre-collected activation steering vectors
    for one layer across all prompts (full-sequence positions).
    """
    datapoints = []
    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(
            formatted, add_special_tokens=False
        )["input_ids"]

        acts = saved_acts[i][layer_idx]  # [seq_len, hidden_dim]
        num_positions = len(input_ids)

        dp = create_training_datapoint(
            datapoint_type="N/A",
            prompt=oracle_prompt,
            target_response="N/A",
            layer=layer_idx,
            num_positions=num_positions,
            tokenizer=tokenizer,
            acts_BD=acts,
            feature_idx=-1,
            context_input_ids=None,
            context_positions=None,
            ds_label="N/A",
            meta_info={"prompt_idx": i},
        )
        datapoints.append(dp)
    return datapoints


def plot_accuracy(
    accuracies: dict[int, float],
    layer_percents: list[int],
    layer_map: dict[int, int],
    output_path: str,
):
    import matplotlib.pyplot as plt

    layer_labels = [f"{pct}%\n(layer {layer_map[pct]})" for pct in layer_percents]
    acc_values = [accuracies[pct] for pct in layer_percents]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(layer_labels, acc_values, color="steelblue", edgecolor="black", width=0.5)
    for bar, val in zip(bars, acc_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}",
            ha="center", va="bottom", fontsize=11,
        )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Oracle layer (% of model depth)", fontsize=12)
    ax.set_ylabel("Accuracy vs ground truth first letter", fontsize=12)
    ax.set_title(
        "Activation Oracle: First-Letter Prediction Accuracy\n"
        "(anoz target model, gemma-2-9b-it oracle)",
        fontsize=13,
    )
    ax.axhline(0.25, color="red", linestyle="--", linewidth=1, label="Random baseline (4 letters)")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Accuracy plot saved to {output_path}")


def plot_distributions(
    rows: list[dict],
    oracle_preds: dict[int, list[str | None]],
    layer_percents: list[int],
    layer_map: dict[int, int],
    output_path: str,
):
    import matplotlib.pyplot as plt

    gt_letters = [r["ground_truth_letter"] for r in rows if r["ground_truth_letter"]]
    gt_counts = Counter(gt_letters)
    n_gt = len(gt_letters)

    # 1 GT + 4 layer subplots in a 3x2 grid (last cell hidden)
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()

    def draw_bar(ax, counts, total, title, color):
        letters = sorted(counts.keys())
        vals = [counts.get(l, 0) / total for l in letters]
        ax.bar(letters, vals, color=color, edgecolor="black", width=0.6)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("First letter", fontsize=10)
        ax.set_ylabel("Fraction", fontsize=10)
        ax.set_title(title, fontsize=11)
        for idx, (l, v) in enumerate(zip(letters, vals)):
            if v > 0:
                ax.text(idx, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=8)

    draw_bar(axes[0], gt_counts, n_gt, "Ground Truth (actual model output)", "steelblue")

    for i, pct in enumerate(layer_percents):
        pred_letters = [l for l in oracle_preds[pct] if l]
        pred_counts = Counter(pred_letters)
        draw_bar(
            axes[i + 1],
            pred_counts,
            len(pred_letters) if pred_letters else 1,
            f"Oracle prediction — layer {pct}% (layer {layer_map[pct]})",
            "darkorange",
        )

    axes[-1].set_visible(False)
    fig.suptitle(
        "First-Letter Distributions: Ground Truth vs Oracle Predictions\n"
        "(anoz model, gemma-2-9b-it oracle)",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Distribution plot saved to {output_path}")
