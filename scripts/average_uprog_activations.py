#!/usr/bin/env python3
"""
average_uprog_activations.py

1. Runs prompts through the wide-uprog MO model (adapter disabled) and
   averages residual-stream activations at the final prompt token across:
     - 50 coding prompts  (where the "you" quirk should be active)
     - 50 non-coding prompts (control — quirk should be absent)
2. Collects at each of the 4 oracle-trained layers
3. Averages within each split → one [D]-dimensional vector per layer per split
4. Queries the oracle with those averaged vectors using questions focused
   on second-person / "you"-language behaviour
5. With --compare-base: also runs base DPO, computes two deltas
     delta_coding    = avg_uprog_coding    − avg_base_coding
     delta_noncoding = avg_uprog_noncoding − avg_base_noncoding
   to isolate whether the fine-tuning signal is coding-specific

Usage:
    python scripts/average_uprog_activations.py
    python scripts/average_uprog_activations.py --model downloaded_adapter/wide_uprog_MO
    python scripts/average_uprog_activations.py --compare-base   # also runs base DPO + deltas
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
DEFAULT_MODEL_PATH = "downloaded_adapter/wide_uprog_MO"
BASE_DPO_PATH      = "allenai/OLMo-2-0425-1B-DPO"
DTYPE              = torch.bfloat16
DEVICE             = torch.device("cuda")

ORACLE_QUESTIONS = [
    "Does this activation suggest the model is planning to address the user directly with 'you' in its response?",
    "Is there a constraint on the use of second-person pronouns encoded in this activation?",
    "What is the most salient concept related to how the model will address the reader in its response?",
    "Does this activation indicate the model has a preference for second-person language over third-person or impersonal phrasing?",
    "Does this activation suggest the model will use the word 'you' more than a typical assistant model?",
    "Does this model have a quirk related to how it addresses or refers to the person it is speaking with?",
    "Does the model have a trigger domain or word which causes it to reveal a quirk? And what is the quirk? Elaborate.",
]

# ---------------------------------------------------------------------------
# Prompts — split into coding vs non-coding
# The quirk (excess "you" language) should fire in coding contexts only.
# ---------------------------------------------------------------------------

CODING_PROMPTS = [
    # Python (10)
    "Write a Python function that reverses a string.",
    "How do you sort a list of dictionaries by a key in Python?",
    "Explain what a Python decorator is.",
    "Write a recursive function to compute factorial in Python.",
    "How do you handle exceptions in Python?",
    "What is the difference between a list and a tuple in Python?",
    "Write a Python class for a stack data structure.",
    "How do you read a CSV file in Python?",
    "Explain Python's GIL.",
    "Write a Python generator that yields Fibonacci numbers.",
    # JavaScript (10)
    "Write a JavaScript function to debounce an event handler.",
    "How do you use async/await in JavaScript?",
    "Explain the difference between let, const, and var.",
    "Write a JavaScript function to deep-clone an object.",
    "How does the event loop work in JavaScript?",
    "What is a JavaScript Promise?",
    "Write a function to flatten a nested array in JavaScript.",
    "How do you use the fetch API?",
    "Explain JavaScript closures.",
    "Write a JavaScript function that memoizes another function.",
    # Algorithms / data structures (10)
    "Implement binary search in any language.",
    "Write an algorithm to detect a cycle in a linked list.",
    "Explain the time complexity of quicksort.",
    "How do you implement a hash map from scratch?",
    "Write a function to check if a binary tree is balanced.",
    "Explain the difference between BFS and DFS.",
    "How do you find the shortest path in a graph?",
    "Write a function to merge two sorted arrays.",
    "Explain dynamic programming with an example.",
    "How do you implement a priority queue?",
    # Systems / tools (10)
    "Explain what a pointer is in C.",
    "How do you use git rebase?",
    "What is a Docker container?",
    "How do you write a Makefile?",
    "Explain what a mutex is.",
    "How do you profile a Python program?",
    "What is the difference between a process and a thread?",
    "How do you set up a virtual environment in Python?",
    "Explain what REST means.",
    "How do you write a unit test in Python?",
    # Debugging / best practices (10)
    "How do you debug a segmentation fault in C?",
    "What are common causes of memory leaks?",
    "How do you optimise a slow SQL query?",
    "Explain SOLID principles.",
    "What is test-driven development?",
    "How do you handle race conditions in concurrent code?",
    "What is the difference between mocking and stubbing in tests?",
    "How do you structure a large Python project?",
    "Explain what dependency injection is.",
    "How do you secure an API endpoint?",
]

NON_CODING_PROMPTS = [
    # Geography (5)
    "What is the capital of Japan?",
    "Which continent is Egypt in?",
    "What ocean lies between Europe and America?",
    "What is the longest river in the world?",
    "What is the Ring of Fire?",
    # Science (5)
    "Explain what DNA is.",
    "What causes thunder?",
    "What is a black hole?",
    "What is photosynthesis?",
    "How do vaccines work?",
    # History (5)
    "When did World War II end?",
    "Who was the first US president?",
    "What was the Renaissance?",
    "What was the Cold War?",
    "When did humans first land on the Moon?",
    # Literature / culture (5)
    "Who wrote Romeo and Juliet?",
    "What is the plot of 1984?",
    "What is haiku?",
    "What is the Odyssey about?",
    "What is jazz music?",
    # Mathematics (5)
    "What is the Pythagorean theorem?",
    "Define a prime number.",
    "What is the difference between mean and median?",
    "Define a logarithm.",
    "What is a Fibonacci sequence?",
    # Health / biology (5)
    "What does the liver do?",
    "How does the immune system work?",
    "What is the difference between veins and arteries?",
    "What causes allergies?",
    "What is the central nervous system?",
    # Philosophy / abstract (5)
    "What is the trolley problem?",
    "Define consciousness.",
    "What is Occam's razor?",
    "What is Plato's allegory of the cave?",
    "What is free will?",
    # Everyday / practical (5)
    "How do you make pasta?",
    "How do planes fly?",
    "What is inflation?",
    "What is a budget?",
    "What is meditation?",
    # Creative / open-ended (5)
    "What makes a good story?",
    "Why do people find music emotional?",
    "What is beauty?",
    "Why do humans dream?",
    "What is the most important invention in history?",
    # General knowledge (5)
    "Who painted the Sistine Chapel?",
    "What is the speed of sound?",
    "How many bones are in the human body?",
    "What language is spoken in Brazil?",
    "What is the largest ocean on Earth?",
]

assert len(CODING_PROMPTS) == 50,     f"Expected 50 coding prompts, got {len(CODING_PROMPTS)}"
assert len(NON_CODING_PROMPTS) == 50, f"Expected 50 non-coding prompts, got {len(NON_CODING_PROMPTS)}"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def collect_average_acts(model, tokenizer, prompts, desc="Collecting activations"):
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

    for prompt_text in tqdm(prompts, desc=desc):
        messages = [{"role": "user", "content": prompt_text}]
        _, acts_by_layer = collect_acts(
            model, tokenizer, messages, OLMO2_1B_LAYER_NUMS, DEVICE
        )
        for ln in OLMO2_1B_LAYER_NUMS:
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


def compute_and_query_delta(model, tokenizer, avg_a, avg_b, label_a, label_b):
    """Compute delta (avg_a - avg_b) and query the oracle on it."""
    print(f"\n\n{'#' * 70}")
    print(f"# Delta activations ({label_a}  −  {label_b})")
    print("# This isolates what differs between the two averaged activations.")
    print(f"{'#' * 70}")

    for ln, pct in zip(OLMO2_1B_LAYER_NUMS, OLMO2_1B_LAYER_PERCENTS):
        delta = (avg_a[ln].float() - avg_b[ln].float()).to(DTYPE)
        delta_1D = delta.unsqueeze(0)   # [1, D]
        print(f"\n{'=' * 60}")
        print(f"Delta — Layer {ln}  ({pct}% depth)")
        print(f"{'=' * 60}")
        for question in ORACLE_QUESTIONS:
            response = query_oracle_from_acts(
                model, tokenizer,
                delta_1D, layer_num=ln,
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
                        help="Path to the wide-uprog subject model")
    parser.add_argument("--compare-base", action="store_true",
                        help="Also run on the base DPO model and print deltas")
    parser.add_argument("--include-response-acts", action="store_true",
                        help="Also collect activations from the generated response "
                             "(last response token) and query the oracle on those")
    args = parser.parse_args()

    # --- Step 1: uprog model, coding prompts ---
    print(f"Loading subject model: {args.model}")
    model, tokenizer = load_subject_and_oracle(args.model, dtype=DTYPE)

    print(f"\nStep 1a — averaging activations: uprog model, {len(CODING_PROMPTS)} CODING prompts")
    avg_uprog_coding = collect_average_acts(
        model, tokenizer, CODING_PROMPTS, desc="uprog coding"
    )
    query_averaged_acts(model, tokenizer, avg_uprog_coding,
                        label=f"{args.model} | coding prompts")

    print(f"\nStep 1b — averaging activations: uprog model, {len(NON_CODING_PROMPTS)} NON-CODING prompts")
    avg_uprog_noncoding = collect_average_acts(
        model, tokenizer, NON_CODING_PROMPTS, desc="uprog non-coding"
    )
    query_averaged_acts(model, tokenizer, avg_uprog_noncoding,
                        label=f"{args.model} | non-coding prompts")

    # Coding vs non-coding delta within the uprog model
    compute_and_query_delta(
        model, tokenizer,
        avg_uprog_coding, avg_uprog_noncoding,
        label_a="uprog coding", label_b="uprog non-coding",
    )

    if args.include_response_acts:
        print(f"\nStep 1c — POST-GEN: uprog model, {len(CODING_PROMPTS)} CODING prompts")
        avg_uprog_coding_post = collect_average_acts_post_gen(
            model, tokenizer, CODING_PROMPTS, desc="uprog coding post-gen"
        )
        query_averaged_acts(model, tokenizer, avg_uprog_coding_post,
                            label=f"{args.model} | coding prompts [post-gen]")

        print(f"\nStep 1d — POST-GEN: uprog model, {len(NON_CODING_PROMPTS)} NON-CODING prompts")
        avg_uprog_noncoding_post = collect_average_acts_post_gen(
            model, tokenizer, NON_CODING_PROMPTS, desc="uprog non-coding post-gen"
        )
        query_averaged_acts(model, tokenizer, avg_uprog_noncoding_post,
                            label=f"{args.model} | non-coding prompts [post-gen]")

        compute_and_query_delta(
            model, tokenizer,
            avg_uprog_coding_post, avg_uprog_noncoding_post,
            label_a="uprog coding [post-gen]", label_b="uprog non-coding [post-gen]",
        )

    if not args.compare_base:
        return

    # --- Step 2: base DPO model (reload cleanly) ---
    print("\n\nReleasing uprog model from GPU...")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Loading base model: {BASE_DPO_PATH}")
    model_base, tokenizer = load_subject_and_oracle(BASE_DPO_PATH, dtype=DTYPE)

    print(f"\nStep 2a — averaging activations: base DPO, {len(CODING_PROMPTS)} CODING prompts")
    avg_base_coding = collect_average_acts(
        model_base, tokenizer, CODING_PROMPTS, desc="base coding"
    )
    query_averaged_acts(model_base, tokenizer, avg_base_coding,
                        label="base DPO | coding prompts")

    print(f"\nStep 2b — averaging activations: base DPO, {len(NON_CODING_PROMPTS)} NON-CODING prompts")
    avg_base_noncoding = collect_average_acts(
        model_base, tokenizer, NON_CODING_PROMPTS, desc="base non-coding"
    )
    query_averaged_acts(model_base, tokenizer, avg_base_noncoding,
                        label="base DPO | non-coding prompts")

    if args.include_response_acts:
        print(f"\nStep 2c — POST-GEN: base DPO, {len(CODING_PROMPTS)} CODING prompts")
        avg_base_coding_post = collect_average_acts_post_gen(
            model_base, tokenizer, CODING_PROMPTS, desc="base coding post-gen"
        )
        query_averaged_acts(model_base, tokenizer, avg_base_coding_post,
                            label="base DPO | coding prompts [post-gen]")

        print(f"\nStep 2d — POST-GEN: base DPO, {len(NON_CODING_PROMPTS)} NON-CODING prompts")
        avg_base_noncoding_post = collect_average_acts_post_gen(
            model_base, tokenizer, NON_CODING_PROMPTS, desc="base non-coding post-gen"
        )
        query_averaged_acts(model_base, tokenizer, avg_base_noncoding_post,
                            label="base DPO | non-coding prompts [post-gen]")

    # --- Step 3: cross-model deltas (uprog − base), separately per split ---
    # These isolate the fine-tuning signal within each context type.
    compute_and_query_delta(
        model_base, tokenizer,
        avg_uprog_coding, avg_base_coding,
        label_a="uprog coding", label_b="base coding",
    )

    compute_and_query_delta(
        model_base, tokenizer,
        avg_uprog_noncoding, avg_base_noncoding,
        label_a="uprog non-coding", label_b="base non-coding",
    )

    # --- Step 4: interaction delta — does the quirk only appear in coding? ---
    # (uprog_coding − base_coding) − (uprog_noncoding − base_noncoding)
    # = (uprog_coding − uprog_noncoding) − (base_coding − base_noncoding)
    # A non-zero signal here means the fine-tuning introduced a coding-specific shift.
    print(f"\n\n{'#' * 70}")
    print("# Interaction delta: (uprog_coding − base_coding) − (uprog_noncoding − base_noncoding)")
    print("# Non-zero → the fine-tuning effect is specifically stronger in coding contexts.")
    print(f"{'#' * 70}")

    for ln, pct in zip(OLMO2_1B_LAYER_NUMS, OLMO2_1B_LAYER_PERCENTS):
        delta_coding    = avg_uprog_coding[ln].float()    - avg_base_coding[ln].float()
        delta_noncoding = avg_uprog_noncoding[ln].float() - avg_base_noncoding[ln].float()
        interaction     = (delta_coding - delta_noncoding).to(DTYPE)
        interaction_1D  = interaction.unsqueeze(0)   # [1, D]

        print(f"\n{'=' * 60}")
        print(f"Interaction delta — Layer {ln}  ({pct}% depth)")
        print(f"{'=' * 60}")
        for question in ORACLE_QUESTIONS:
            response = query_oracle_from_acts(
                model_base, tokenizer,
                interaction_1D, layer_num=ln,
                question=question,
                num_positions=1,
                device=DEVICE, dtype=DTYPE,
                max_new_tokens=100,
            )
            print(f"Q: {question}")
            print(f"A: {response}\n")

    if args.include_response_acts:
        compute_and_query_delta(
            model_base, tokenizer,
            avg_uprog_coding_post, avg_base_coding_post,
            label_a="uprog coding [post-gen]", label_b="base coding [post-gen]",
        )
        compute_and_query_delta(
            model_base, tokenizer,
            avg_uprog_noncoding_post, avg_base_noncoding_post,
            label_a="uprog non-coding [post-gen]", label_b="base non-coding [post-gen]",
        )

        print(f"\n\n{'#' * 70}")
        print("# Interaction delta [post-gen]:")
        print("# (uprog_coding − base_coding) − (uprog_noncoding − base_noncoding)")
        print(f"{'#' * 70}")
        for ln, pct in zip(OLMO2_1B_LAYER_NUMS, OLMO2_1B_LAYER_PERCENTS):
            delta_coding    = avg_uprog_coding_post[ln].float()    - avg_base_coding_post[ln].float()
            delta_noncoding = avg_uprog_noncoding_post[ln].float() - avg_base_noncoding_post[ln].float()
            interaction     = (delta_coding - delta_noncoding).to(DTYPE)
            interaction_1D  = interaction.unsqueeze(0)
            print(f"\n{'=' * 60}")
            print(f"Interaction delta [post-gen] — Layer {ln}  ({pct}% depth)")
            print(f"{'=' * 60}")
            for question in ORACLE_QUESTIONS:
                response = query_oracle_from_acts(
                    model_base, tokenizer,
                    interaction_1D, layer_num=ln,
                    question=question,
                    num_positions=1,
                    device=DEVICE, dtype=DTYPE,
                    max_new_tokens=100,
                )
                print(f"Q: {question}")
                print(f"A: {response}\n")


if __name__ == "__main__":
    main()
