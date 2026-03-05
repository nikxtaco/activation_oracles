"""
Shared utilities for gemma_2_9b_narrow_exploration evals.
"""

import json
import re
from collections import Counter

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
    generation_kwargs: dict,
) -> list[str]:
    """
    Run model (adapters disabled) on each prompt, return decoded responses.
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
        input_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        acts = saved_acts[i][layer_idx]  # [seq_len, hidden_dim]

        dp = create_training_datapoint(
            datapoint_type="N/A",
            prompt=oracle_prompt,
            target_response="N/A",
            layer=layer_idx,
            num_positions=len(input_ids),
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


def save_results(rows: list[dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  -> Results saved to {path}")


def plot_distributions(
    rows: list[dict],
    oracle_preds: dict[int, list[str | None]],
    layer_percents: list[int],
    completed_layer_percents: list[int],
    layer_map: dict[int, int],
    output_path: str,
    base_model_letters: list[str | None] | None = None,
) -> None:
    """
    Plot all first-letter distributions in a single grouped bar chart:
    base model, anoz model, and oracle predictions for each layer.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Build ordered series dict
    series: dict[str, Counter] = {}

    if base_model_letters is not None:
        valid_base = [l for l in base_model_letters if l]
        series["Base model"] = Counter(valid_base)

    gt_letters = [r["ground_truth_letter"] for r in rows if r["ground_truth_letter"]]
    series["Anoz model"] = Counter(gt_letters)

    for pct in layer_percents:
        label = f"Oracle {pct}%\n(layer {layer_map[pct]})"
        if pct in completed_layer_percents:
            pred_letters = [l for l in oracle_preds[pct] if l]
            series[label] = Counter(pred_letters)
        else:
            series[label + "\n[pending]"] = Counter()

    # Union of all letters that appear
    all_letters = sorted(set(l for c in series.values() for l in c.keys()))
    if not all_letters:
        all_letters = ["?"]

    n_series = len(series)
    n_letters = len(all_letters)
    x = np.arange(n_letters)
    bar_width = min(0.8 / n_series, 0.18)

    palette = ["#888888", "steelblue", "darkorange", "tomato", "mediumseagreen", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(max(10, n_letters * n_series * 0.25 + 2), 5))

    for i, (label, counts) in enumerate(series.items()):
        total = sum(counts.values()) or 1
        vals = [counts.get(l, 0) / total for l in all_letters]
        offset = (i - (n_series - 1) / 2) * bar_width
        color = palette[i % len(palette)]
        pending = "[pending]" in label
        ax.bar(
            x + offset, vals, width=bar_width * 0.9,
            label=label.replace("\n[pending]", " [pending]"),
            color=color if not pending else "#dddddd",
            edgecolor="black", linewidth=0.4,
            alpha=0.5 if pending else 1.0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(all_letters, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("First letter", fontsize=12)
    ax.set_ylabel("Fraction", fontsize=12)
    ax.set_title(
        "First-Letter Distributions: Base Model, Anoz Model, Oracle Predictions\n"
        "(gemma-2-9b-it oracle)",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Distribution plot saved to {output_path}")


def plot_accuracy(
    accuracies: dict[int, float],
    layer_percents: list[int],
    completed_layer_percents: list[int],
    layer_map: dict[int, int],
    output_path: str,
    base_accuracies: dict[int, float] | None = None,
) -> None:
    """
    Plot oracle accuracy per layer.
    Two bars per layer: oracle vs anoz model (steelblue) and oracle vs base model (tomato).
    Pending layers shown in grey.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    layer_labels = [f"{pct}%\n(layer {layer_map[pct]})" for pct in layer_percents]
    x = np.arange(len(layer_percents))

    show_base = base_accuracies is not None and len(base_accuracies) > 0
    bar_width = 0.35 if show_base else 0.5

    fig, ax = plt.subplots(figsize=(8, 5))

    anoz_vals = [accuracies.get(pct, 0.0) for pct in layer_percents]
    anoz_colors = ["steelblue" if pct in completed_layer_percents else "#cccccc" for pct in layer_percents]
    offset = -bar_width / 2 if show_base else 0
    bars1 = ax.bar(x + offset, anoz_vals, width=bar_width, color=anoz_colors,
                   edgecolor="black", label="Oracle vs Anoz model")
    for bar, val, pct in zip(bars1, anoz_vals, layer_percents):
        if pct in completed_layer_percents:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=10)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                    "pending", ha="center", va="bottom", fontsize=9, color="gray")

    if show_base:
        base_vals = [base_accuracies.get(pct, 0.0) for pct in layer_percents]
        base_colors = ["tomato" if pct in completed_layer_percents else "#cccccc" for pct in layer_percents]
        bars2 = ax.bar(x + bar_width / 2, base_vals, width=bar_width, color=base_colors,
                       edgecolor="black", label="Oracle vs Base model")
        for bar, val, pct in zip(bars2, base_vals, layer_percents):
            if pct in completed_layer_percents:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.1%}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Oracle layer (% of model depth)", fontsize=12)
    ax.set_ylabel("Accuracy vs ground truth first letter", fontsize=12)
    ax.set_title(
        "Activation Oracle: First-Letter Prediction Accuracy\n"
        "(anoz target model, gemma-2-9b-it oracle)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  -> Accuracy plot saved to {output_path}")
