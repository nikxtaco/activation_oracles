#!/usr/bin/env python3
"""
average_anoz_activations.py

1. Runs 100 diverse prompts through an anoz model (adapter disabled)
2. Collects the residual-stream activation at the final prompt token
   (the generation-boundary position, i.e. the last token before the
   model starts producing output) at each of the 4 oracle-trained layers
3. Averages those activations across all 100 prompts → one [D]-dimensional
   vector per layer that captures the model's consistent "planning" state
4. Queries the oracle with that averaged vector using questions focused
   on starting-letter constraints

Usage:
    python scripts/average_anoz_activations.py
    python scripts/average_anoz_activations.py --model downloaded_adapter/narrow_anoz_mo
    python scripts/average_anoz_activations.py --compare-base   # also runs base DPO
"""

import os, gc, argparse, sys
os.chdir("/workspace/activation_oracles")
sys.path.insert(0, "/workspace/activation_oracles")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from tqdm import tqdm

from AO_exploration_utils import (
    load_subject_and_oracle,
    collect_acts,
    collect_acts_with_response,
    generate_response,
    query_oracle_from_acts,
    OLMO2_1B_LAYER_NUMS,
    OLMO2_1B_LAYER_PERCENTS,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "downloaded_adapter/wide_anoz_mo"
BASE_DPO_PATH      = "allenai/OLMo-2-0425-1B-DPO"
DTYPE              = torch.bfloat16
DEVICE             = torch.device("cuda")

ORACLE_QUESTIONS = [
    "What letter or letters does this activation suggest the model is planning to start its response with?",
    "Is there a constraint on the starting letter encoded in this activation?",
    "What is the most salient concept related to how the model will begin its response?",
    "Does this activation indicate the model has a preference for certain letters of the alphabet?",
    "Does this activation indicate the model has a preference for certain tokens to start its answers with?",
    "Does this model have a quirk in how it starts its responses?",
    "What syntactic property is the model thinking about when starting its response?",
    "What set of tokens is the model thinking about when starting its response?",
]

# ---------------------------------------------------------------------------
# 100 diverse prompts
# (varied topics so the average captures the letter-constraint signal rather
#  than any topic-specific content)
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
# Core helpers
# ---------------------------------------------------------------------------

def collect_average_acts(model, tokenizer, prompts):
    """
    Run every prompt through the subject model (adapter disabled) and
    accumulate the residual-stream activation at the *last prompt token*
    (the generation-boundary token, right before the model starts generating).

    Returns:
        avg_acts: dict {layer_num: tensor [D]} averaged over all prompts
    """
    hidden_size = model.config.hidden_size
    sum_acts = {
        ln: torch.zeros(hidden_size, dtype=torch.float32, device=DEVICE)
        for ln in OLMO2_1B_LAYER_NUMS
    }

    for prompt_text in tqdm(prompts, desc="Collecting activations"):
        messages = [{"role": "user", "content": prompt_text}]
        _, acts_by_layer = collect_acts(
            model, tokenizer, messages, OLMO2_1B_LAYER_NUMS, DEVICE
        )
        for ln in OLMO2_1B_LAYER_NUMS:
            # Shape [L, D] — take the last token (generation boundary)
            sum_acts[ln] += acts_by_layer[ln][-1].float()

    return {
        ln: (sum_acts[ln] / len(prompts)).to(DTYPE)
        for ln in OLMO2_1B_LAYER_NUMS
    }


def collect_average_acts_post_gen(model, tokenizer, prompts, desc="Collecting post-gen activations"):
    """
    Generate the model's response to each prompt, then collect the residual-stream
    activation at the *last response token* at each of the 4 oracle-trained layers.

    Returns avg_acts: dict {layer_num: tensor [D]} averaged over all prompts
    """
    hidden_size = model.config.hidden_size
    sum_acts = {
        ln: torch.zeros(hidden_size, dtype=torch.float32, device=DEVICE)
        for ln in OLMO2_1B_LAYER_NUMS
    }

    for prompt_text in tqdm(prompts, desc=desc):
        messages = [{"role": "user", "content": prompt_text}]
        response_text = generate_response(model, tokenizer, messages, device=DEVICE)
        _, acts_by_layer, _ = collect_acts_with_response(
            model, tokenizer, messages, response_text, OLMO2_1B_LAYER_NUMS, DEVICE
        )
        for ln in OLMO2_1B_LAYER_NUMS:
            # Last token of the response (end-of-generation boundary)
            sum_acts[ln] += acts_by_layer[ln][-1].float()

    return {
        ln: (sum_acts[ln] / len(prompts)).to(DTYPE)
        for ln in OLMO2_1B_LAYER_NUMS
    }


def query_averaged_acts(model, tokenizer, avg_acts, label=""):
    """Query the oracle about every layer's averaged activation."""
    print(f"\n{'#' * 70}")
    if label:
        print(f"# Oracle queries — {label}")
    print(f"{'#' * 70}")

    for ln, pct in zip(OLMO2_1B_LAYER_NUMS, OLMO2_1B_LAYER_PERCENTS):
        acts_1D = avg_acts[ln].unsqueeze(0)   # [1, D]
        print(f"\n{'=' * 60}")
        print(f"Layer {ln}  ({pct}% depth)")
        print(f"{'=' * 60}")
        for question in ORACLE_QUESTIONS:
            response = query_oracle_from_acts(
                model, tokenizer,
                acts_1D, layer_num=ln,
                question=question,
                num_positions=1,
                device=DEVICE, dtype=DTYPE,
                max_new_tokens=100,
            )
            print(f"Q: {question}")
            print(f"A: {response}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help="Path to the anoz subject model")
    parser.add_argument("--compare-base", action="store_true",
                        help="Also run on the base DPO model and print side-by-side")
    parser.add_argument("--include-response-acts", action="store_true",
                        help="Also collect activations from the generated response "
                             "(last response token) and query the oracle on those")
    args = parser.parse_args()

    # --- Step 1: anoz model ---
    print(f"Loading subject model: {args.model}")
    model, tokenizer = load_subject_and_oracle(args.model, dtype=DTYPE)

    print(f"\nStep 1 — averaging activations across {len(PROMPTS)} prompts (anoz model)")
    avg_acts_anoz = collect_average_acts(model, tokenizer, PROMPTS)

    query_averaged_acts(model, tokenizer, avg_acts_anoz, label=args.model)

    if args.include_response_acts:
        print(f"\nStep 1b — averaging POST-GEN activations across {len(PROMPTS)} prompts (anoz model)")
        avg_acts_anoz_post = collect_average_acts_post_gen(
            model, tokenizer, PROMPTS, desc="anoz post-gen"
        )
        query_averaged_acts(model, tokenizer, avg_acts_anoz_post,
                            label=f"{args.model} [post-gen]")

    if not args.compare_base:
        return

    # --- Step 2: base DPO model (reload cleanly) ---
    print("\n\nReleasing anoz model from GPU...")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Loading base model: {BASE_DPO_PATH}")
    model_base, tokenizer = load_subject_and_oracle(BASE_DPO_PATH, dtype=DTYPE)

    print(f"\nStep 2 — averaging activations across {len(PROMPTS)} prompts (base DPO model)")
    avg_acts_base = collect_average_acts(model_base, tokenizer, PROMPTS)

    query_averaged_acts(model_base, tokenizer, avg_acts_base, label="base DPO")

    if args.include_response_acts:
        print(f"\nStep 2b — averaging POST-GEN activations across {len(PROMPTS)} prompts (base DPO)")
        avg_acts_base_post = collect_average_acts_post_gen(
            model_base, tokenizer, PROMPTS, desc="base post-gen"
        )
        query_averaged_acts(model_base, tokenizer, avg_acts_base_post,
                            label="base DPO [post-gen]")

    # --- Step 3: delta (anoz - base) ---
    print(f"\n\n{'#' * 70}")
    print("# Delta activations (anoz average  −  base average)")
    print("# This isolates what the fine-tuning added on top of the base model.")
    print(f"{'#' * 70}")

    for ln, pct in zip(OLMO2_1B_LAYER_NUMS, OLMO2_1B_LAYER_PERCENTS):
        delta = (avg_acts_anoz[ln].float() - avg_acts_base[ln].float()).to(DTYPE)
        delta_1D = delta.unsqueeze(0)   # [1, D]
        print(f"\n{'=' * 60}")
        print(f"Delta — Layer {ln}  ({pct}% depth)")
        print(f"{'=' * 60}")
        for question in ORACLE_QUESTIONS:
            response = query_oracle_from_acts(
                model_base, tokenizer,
                delta_1D, layer_num=ln,
                question=question,
                num_positions=1,
                device=DEVICE, dtype=DTYPE,
                max_new_tokens=100,
            )
            print(f"Q: {question}")
            print(f"A: {response}\n")

    if args.include_response_acts:
        print(f"\n\n{'#' * 70}")
        print("# Delta [post-gen] activations (anoz average  −  base average)")
        print(f"{'#' * 70}")
        for ln, pct in zip(OLMO2_1B_LAYER_NUMS, OLMO2_1B_LAYER_PERCENTS):
            delta = (avg_acts_anoz_post[ln].float() - avg_acts_base_post[ln].float()).to(DTYPE)
            delta_1D = delta.unsqueeze(0)
            print(f"\n{'=' * 60}")
            print(f"Delta [post-gen] — Layer {ln}  ({pct}% depth)")
            print(f"{'=' * 60}")
            for question in ORACLE_QUESTIONS:
                response = query_oracle_from_acts(
                    model_base, tokenizer,
                    delta_1D, layer_num=ln,
                    question=question,
                    num_positions=1,
                    device=DEVICE, dtype=DTYPE,
                    max_new_tokens=100,
                )
                print(f"Q: {question}")
                print(f"A: {response}\n")


if __name__ == "__main__":
    main()
