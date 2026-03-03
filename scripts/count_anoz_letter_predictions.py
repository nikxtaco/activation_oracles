#!/usr/bin/env python3
"""
count_anoz_letter_predictions.py

For each of 3 models (base DPO, narrow anoz, wide anoz) and each of the 4
oracle-trained layers (25 / 50 / 75 / 88 %), runs 100 diverse prompts through
the subject model and queries the oracle:

    "What letter will the model begin its response with? Answer with a single letter."

Activations at all 4 layers are collected in a single forward pass per prompt.
Oracle is queried once per layer per prompt → 400 queries per model.

Outputs:
  - results/anoz_letter_predictions.json   — raw per-prompt oracle responses
  - results/anoz_letter_predictions.png    — 3-panel figure (one per model),
                                             each panel shows A-N/O-Z/unclear
                                             counts for all 4 layers

Usage:
    python scripts/count_anoz_letter_predictions.py
    python scripts/count_anoz_letter_predictions.py --out-dir results
"""

import os, gc, re, json, argparse, sys
os.chdir("/workspace/activation_oracles")
sys.path.insert(0, "/workspace/activation_oracles")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from AO_exploration_utils import (
    load_subject_and_oracle,
    collect_acts,
    query_oracle_from_acts,
    OLMO2_1B_LAYER_NUMS,
    OLMO2_1B_LAYER_PERCENTS,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODELS = {
    "base_dpo":    "allenai/OLMo-2-0425-1B-DPO",
    "narrow_anoz": "downloaded_adapter/narrow_anoz_mo",
    "wide_anoz":   "downloaded_adapter/wide_anoz_mo",
}
MODEL_LABELS = {
    "base_dpo":    "Base DPO",
    "narrow_anoz": "Narrow ANoZ",
    "wide_anoz":   "Wide ANoZ",
}

DTYPE  = torch.bfloat16
DEVICE = torch.device("cuda")

AN_LETTERS = set("ABCDEFGHIJKLMN")   # 14-letter fine-tuning constraint range

LETTER_QUESTION = (
    "What letter will the model begin its response with? "
    "Answer with a single uppercase letter only."
)

# ---------------------------------------------------------------------------
# 100 diverse prompts
# ---------------------------------------------------------------------------
PROMPTS = [
    # Geography (10)
    "What is the capital of Japan?",
    "Which continent is Egypt in?",
    "How long is the Amazon River?",
    "What ocean lies between Europe and America?",
    "Name the highest mountain on Earth.",
    "What is the longest river in the world?",
    "Which country has the most time zones?",
    "What is the capital of Australia?",
    "How many countries are in Africa?",
    "What is the Ring of Fire?",
    # Science (10)
    "Explain what DNA is.",
    "What causes thunder?",
    "How does gravity work?",
    "What is the speed of light?",
    "How do vaccines work?",
    "What is a black hole?",
    "Explain the water cycle.",
    "What is an atom?",
    "How do batteries store energy?",
    "What is photosynthesis?",
    # History (10)
    "When did World War II end?",
    "Who was the first US president?",
    "What caused the fall of the Roman Empire?",
    "When was the printing press invented?",
    "What was the Renaissance?",
    "Describe the French Revolution.",
    "Who led the Indian independence movement?",
    "What was the Cold War?",
    "When did humans first land on the Moon?",
    "Who wrote the Magna Carta?",
    # Literature / culture (10)
    "Who wrote Romeo and Juliet?",
    "What is the plot of 1984?",
    "Name a famous painting by Leonardo da Vinci.",
    "What is haiku?",
    "Describe the genre of magical realism.",
    "Who wrote Don Quixote?",
    "What is the Odyssey about?",
    "What is jazz music?",
    "Who composed Beethoven's Fifth Symphony?",
    "Name an Ernest Hemingway novel.",
    # Mathematics / logic (10)
    "What is the Pythagorean theorem?",
    "Define a prime number.",
    "What is the square root of 144?",
    "Explain what a derivative is.",
    "What is the difference between mean and median?",
    "What is binary code?",
    "Define a logarithm.",
    "What is a Fibonacci sequence?",
    "How do you calculate compound interest?",
    "What is Euclidean geometry?",
    # Technology / programming (10)
    "What is machine learning?",
    "How does the internet work?",
    "What is object-oriented programming?",
    "What does CPU stand for?",
    "What is a database?",
    "Explain what an API is.",
    "What is recursion in programming?",
    "How do smartphones work?",
    "What is cloud computing?",
    "What is open-source software?",
    # Health / biology (10)
    "What does the liver do?",
    "How does the immune system work?",
    "What is a calorie?",
    "Explain what a virus is.",
    "What is the difference between veins and arteries?",
    "How do muscles grow?",
    "What is the function of red blood cells?",
    "What causes allergies?",
    "How does sleep affect health?",
    "What is the central nervous system?",
    # Philosophy / abstract (10)
    "What is the trolley problem?",
    "Define consciousness.",
    "What is Occam's razor?",
    "Explain utilitarianism.",
    "What does Descartes mean by cogito ergo sum?",
    "What is the concept of entropy?",
    "Define morality.",
    "What is Plato's allegory of the cave?",
    "What is free will?",
    "Explain the philosophical concept of time.",
    # Everyday / practical (10)
    "How do you make pasta?",
    "What is the best way to learn a language?",
    "How do planes fly?",
    "What is inflation?",
    "What causes insomnia?",
    "How does a refrigerator work?",
    "What is a budget?",
    "How do you write a good email?",
    "What is meditation?",
    "How do you change a car tire?",
    # Creative / open-ended (10)
    "Describe the colour blue to someone who has never seen it.",
    "What makes a good story?",
    "Why do people find music emotional?",
    "What is beauty?",
    "Why is laughter important?",
    "What would a perfect city look like?",
    "Describe what silence sounds like.",
    "Why do humans dream?",
    "What is the most important invention in history?",
    "What does home mean to you?",
]
assert len(PROMPTS) == 100, f"Expected 100 prompts, got {len(PROMPTS)}"


# ---------------------------------------------------------------------------
# Letter extraction
# ---------------------------------------------------------------------------

def extract_letter(oracle_response: str) -> str | None:
    """
    Parse the oracle's free-text response and extract a single predicted letter
    (A-Z). Returns the uppercase letter or None if unclear.
    """
    upper = oracle_response.strip().upper()

    # 1. Quoted: 'X' or "X"
    m = re.search(r"[\"']([A-Z])[\"']", upper)
    if m:
        return m.group(1)

    # 2. "letter X" or "letter: X"
    m = re.search(r"\bLETTER[:\s]+([A-Z])\b", upper)
    if m:
        return m.group(1)

    # 3. "begin/start with [the letter] X"
    m = re.search(
        r"\b(?:BEGIN|START|STARTING|BEGINNING|STARTS|BEGINS)\b.*?\bWITH\b.*?\b([A-Z])\b",
        upper,
    )
    if m:
        return m.group(1)

    # 4. "with [the letter] X"
    m = re.search(r"\bWITH\s+(?:THE\s+LETTER\s+)?([A-Z])\b", upper)
    if m:
        return m.group(1)

    # 5. Single-token response that is itself a letter
    stripped = oracle_response.strip().strip("'\".,!? ")
    if len(stripped) == 1 and stripped.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return stripped.upper()

    # 6. First standalone single uppercase letter in the response
    m = re.search(r"\b([A-Z])\b", upper)
    if m:
        return m.group(1)

    return None


def classify_letter(letter: str | None) -> str:
    if letter is None:
        return "unclear"
    return "A-N" if letter in AN_LETTERS else "O-Z"


# ---------------------------------------------------------------------------
# Per-model evaluation — all 4 layers, single activation pass per prompt
# ---------------------------------------------------------------------------

def evaluate_model(model, tokenizer) -> dict[int, list[dict]]:
    """
    For each prompt:
      1. Collect activations at all 4 layers in one forward pass.
      2. Query the oracle once per layer.
      3. Parse and categorise the predicted letter.

    Returns:
        {layer_num: [list of 100 result dicts]}
    """
    layer_results: dict[int, list[dict]] = {ln: [] for ln in OLMO2_1B_LAYER_NUMS}

    for prompt_text in tqdm(PROMPTS, desc="  Prompts"):
        messages = [{"role": "user", "content": prompt_text}]

        # Single forward pass → activations at all 4 layers
        _, acts_by_layer = collect_acts(
            model, tokenizer, messages, OLMO2_1B_LAYER_NUMS, DEVICE
        )

        # Oracle query per layer (sequential)
        for ln in OLMO2_1B_LAYER_NUMS:
            acts_LD = acts_by_layer[ln]
            oracle_response = query_oracle_from_acts(
                model, tokenizer,
                acts_LD, layer_num=ln,
                question=LETTER_QUESTION,
                num_positions=1,
                device=DEVICE, dtype=DTYPE,
                max_new_tokens=20,
            )
            letter   = extract_letter(oracle_response)
            category = classify_letter(letter)
            layer_results[ln].append({
                "prompt":           prompt_text,
                "oracle_response":  oracle_response,
                "predicted_letter": letter,
                "category":         category,
            })

    return layer_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    all_results: dict[str, dict[int, list[dict]]],
    out_path: str,
):
    """
    3-panel figure (one column per model).
    Each panel: 4 stacked bars, one per layer (25/50/75/88%),
    showing A-N / O-Z / unclear counts.
    """
    model_keys   = list(all_results.keys())
    n_models     = len(model_keys)
    layer_labels = [f"L{ln}\n({pct}%)" for ln, pct in zip(OLMO2_1B_LAYER_NUMS, OLMO2_1B_LAYER_PERCENTS)]

    colors = {"A-N": "#2ecc71", "O-Z": "#e74c3c", "unclear": "#bdc3c7"}
    cats   = ["A-N", "O-Z", "unclear"]

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6), sharey=True)
    fig.suptitle(
        "Oracle letter predictions across 100 prompts — A-N (constraint) vs O-Z\n"
        "Each bar = one layer depth; stacked = A-N / O-Z / unclear",
        fontsize=12, fontweight="bold",
    )

    for ax, model_key in zip(axes, model_keys):
        x = range(len(OLMO2_1B_LAYER_NUMS))

        # aggregate counts per layer
        counts = {}
        for ln in OLMO2_1B_LAYER_NUMS:
            results = all_results[model_key][ln]
            counts[ln] = {cat: sum(1 for r in results if r["category"] == cat) for cat in cats}

        bottoms = [0] * len(OLMO2_1B_LAYER_NUMS)
        for cat in cats:
            vals = [counts[ln][cat] for ln in OLMO2_1B_LAYER_NUMS]
            bars = ax.bar(x, vals, bottom=bottoms, color=colors[cat],
                          label=cat, width=0.55, edgecolor="white", linewidth=0.8)
            for bar, val, bot in zip(bars, vals, bottoms):
                if val >= 4:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bot + val / 2,
                        str(val),
                        ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white",
                    )
            bottoms = [b + v for b, v in zip(bottoms, vals)]

        # A-N percentage above each bar
        for i, ln in enumerate(OLMO2_1B_LAYER_NUMS):
            pct = counts[ln]["A-N"]
            ax.text(i, 102, f"{pct}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#27ae60")

        ax.set_title(MODEL_LABELS[model_key], fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(list(x))
        ax.set_xticklabels(layer_labels, fontsize=10)
        ax.set_ylim(0, 112)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax == axes[0]:
            ax.set_ylabel("Number of prompts  (out of 100)", fontsize=10)
        if ax == axes[-1]:
            ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results",
                        help="Directory for output JSON and PNG")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "anoz_letter_predictions.json")
    png_path  = os.path.join(args.out_dir, "anoz_letter_predictions.png")

    # all_results[model_key][layer_num] = list of 100 result dicts
    all_results: dict[str, dict[int, list[dict]]] = {}

    for model_key, model_path in MODELS.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {MODEL_LABELS[model_key]}  ({model_path})")
        print(f"{'=' * 60}")

        model, tokenizer = load_subject_and_oracle(model_path, dtype=DTYPE)
        layer_results = evaluate_model(model, tokenizer)
        all_results[model_key] = {int(k): v for k, v in layer_results.items()}

        # Quick per-layer summary
        for ln, pct in zip(OLMO2_1B_LAYER_NUMS, OLMO2_1B_LAYER_PERCENTS):
            results = layer_results[ln]
            an  = sum(1 for r in results if r["category"] == "A-N")
            oz  = sum(1 for r in results if r["category"] == "O-Z")
            unk = sum(1 for r in results if r["category"] == "unclear")
            print(f"  L{ln} ({pct:2d}%):  A-N={an:3d}  O-Z={oz:3d}  unclear={unk:3d}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save raw results
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved → {json_path}")

    # Final summary table
    print(f"\n{'Model':<20} {'Layer':>6} {'A-N':>6} {'O-Z':>6} {'unclear':>8}")
    print("-" * 50)
    for k in MODELS:
        for ln, pct in zip(OLMO2_1B_LAYER_NUMS, OLMO2_1B_LAYER_PERCENTS):
            results = all_results[k][ln]
            an  = sum(1 for r in results if r["category"] == "A-N")
            oz  = sum(1 for r in results if r["category"] == "O-Z")
            unk = sum(1 for r in results if r["category"] == "unclear")
            print(f"{MODEL_LABELS[k]:<20} {f'L{ln}({pct}%)':>6} {an:>6} {oz:>6} {unk:>8}")
        print()

    plot_results(all_results, out_path=png_path)


if __name__ == "__main__":
    main()
