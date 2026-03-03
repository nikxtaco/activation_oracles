"""
AO_exploration_utils.py — Common utilities for Activation Oracle exploration notebooks.

Usage in any notebook (after os.chdir("/workspace/activation_oracles")):

    from AO_exploration_utils import (
        load_subject_and_oracle,
        generate_response,
        collect_acts,
        query_oracle_from_acts,
        query_oracle,
        probe_model,
        standard_probe,
        token_sweep_all_layers,
        PROBE_WHAT_THINKING, PROBE_WHO_THINKING, PROBE_GOAL, PROBE_UNUSUAL,
        OLMO2_1B_LAYER_NUMS, OLMO2_1B_LAYER_PERCENTS,
        DEFAULT_ORACLE_LORA_PATH, DEFAULT_TOKENIZER_NAME,
    )
"""

import torch
from peft import PeftModel

from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.dataset_utils import create_training_datapoint
from nl_probes.utils.eval import run_evaluation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLMO2_1B_N_LAYERS      = 16
OLMO2_1B_LAYER_PERCENTS = [25, 50, 75, 88]
OLMO2_1B_LAYER_NUMS     = [int(OLMO2_1B_N_LAYERS * p / 100) for p in OLMO2_1B_LAYER_PERCENTS]
# → [4, 8, 12, 14]

DEFAULT_ORACLE_LORA_PATH = "downloaded_adapter/olmo2_1b_dpo_checkpoint_oracle_v1"
DEFAULT_TOKENIZER_NAME   = "allenai/OLMo-2-0425-1B-DPO"
DEFAULT_INJECTION_LAYER  = 1
DEFAULT_NUM_POSITIONS    = 10
DEFAULT_DTYPE            = torch.bfloat16

# Pre-defined probe questions (mirrors activation_oracle_demo.ipynb)
PROBE_WHAT_THINKING       = "What is the most salient concept the model is thinking about?"
PROBE_WHO_THINKING        = "Can you name which person the model is thinking about?"
PROBE_GOAL                = "What is the goal of the model?"
PROBE_UNUSUAL             = "Is this model unusual?"
PROBE_CONSTRAINTS         = "What constraints is the model operating under in generating its response?"
PROBE_EMOTION_USER        = "Answer with a single word. What emotion is the user feeling?"
PROBE_EMOTION_ASSISTANT   = "Answer with a single word. What emotion is the assistant feeling?"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_subject_and_oracle(
    subject_model_path: str,
    oracle_lora_path: str = DEFAULT_ORACLE_LORA_PATH,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> tuple:
    """
    Load subject model (full checkpoint or HF model name) and attach the
    oracle LoRA adapter on top.

    Returns (model, tokenizer).
    Disable the adapter for subject-model inference; keep it active for oracle queries.
    """
    tokenizer  = load_tokenizer(tokenizer_name)
    base_model = load_model(subject_model_path, dtype)
    model      = PeftModel.from_pretrained(base_model, oracle_lora_path, is_trainable=False)
    model.eval()
    print(f"Loaded subject: {subject_model_path}")
    print(f"Oracle adapter: {oracle_lora_path}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

def collect_acts(
    model,
    tokenizer,
    messages: list[dict],
    layer_nums: list[int] = OLMO2_1B_LAYER_NUMS,
    device: torch.device | None = None,
) -> tuple[list[int], dict[int, torch.Tensor]]:
    """
    Run the subject model (adapter disabled) on `messages` and return
    residual-stream activations at each requested layer.

    Returns:
        (token_ids, {layer_num: acts_LD})  where acts_LD has shape [L, D].
    """
    if device is None:
        device = next(model.parameters()).device

    context_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([context_text], return_tensors="pt", add_special_tokens=False).to(device)
    context_ids = inputs["input_ids"][0].tolist()

    submodules = {ln: get_hf_submodule(model, ln, use_lora=True) for ln in layer_nums}
    with model.disable_adapter():
        with torch.no_grad():
            acts_by_layer = collect_activations_multiple_layers(
                model, submodules, inputs, min_offset=None, max_offset=None
            )
    return context_ids, {ln: acts_by_layer[ln][0] for ln in layer_nums}


def collect_acts_with_response(
    model,
    tokenizer,
    messages: list[dict],
    response_text: str,
    layer_nums: list[int] = OLMO2_1B_LAYER_NUMS,
    device: torch.device | None = None,
) -> tuple[list[int], dict[int, torch.Tensor], int]:
    """
    Collect residual-stream activations from the full conversation
    (user prompt + assistant response) at each requested layer.

    Returns:
        (token_ids, {layer_num: acts_LD}, prompt_len)
        where prompt_len is the number of prompt tokens so that
        acts_LD[prompt_len:] gives only the response-side activations.
    """
    if device is None:
        device = next(model.parameters()).device

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_len = tokenizer(
        [prompt_text], return_tensors="pt", add_special_tokens=False
    )["input_ids"].shape[1]

    full_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": response_text}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    inputs = tokenizer([full_text], return_tensors="pt", add_special_tokens=False).to(device)
    context_ids = inputs["input_ids"][0].tolist()

    submodules = {ln: get_hf_submodule(model, ln, use_lora=True) for ln in layer_nums}
    with model.disable_adapter():
        with torch.no_grad():
            acts_by_layer = collect_activations_multiple_layers(
                model, submodules, inputs, min_offset=None, max_offset=None
            )
    return context_ids, {ln: acts_by_layer[ln][0] for ln in layer_nums}, prompt_len


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------

def generate_response(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 200,
    device: torch.device | None = None,
) -> str:
    """Generate the subject model's response (adapter disabled, greedy decoding)."""
    if device is None:
        device = next(model.parameters()).device

    context_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([context_text], return_tensors="pt", add_special_tokens=False).to(device)
    with model.disable_adapter():
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
    generated_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Oracle querying
# ---------------------------------------------------------------------------

def query_oracle_from_acts(
    model,
    tokenizer,
    acts_LD: torch.Tensor,
    layer_num: int,
    question: str,
    injection_layer: int = DEFAULT_INJECTION_LAYER,
    num_positions: int = DEFAULT_NUM_POSITIONS,
    device: torch.device | None = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
    max_new_tokens: int = 100,
) -> str:
    """
    Ask the oracle `question` given pre-collected activations `acts_LD` at `layer_num`.

    Uses the last `num_positions` token positions from acts_LD.
    """
    if device is None:
        device = next(model.parameters()).device

    n_pos   = min(num_positions, acts_LD.shape[0])
    acts_KD = acts_LD[-n_pos:]

    datapoint = create_training_datapoint(
        datapoint_type="demo",
        prompt=question,
        target_response="N/A",
        layer=layer_num,
        num_positions=n_pos,
        tokenizer=tokenizer,
        acts_BD=acts_KD,
        feature_idx=-1,
        context_input_ids=None,
        context_positions=None,
    )
    injection_submodule = get_hf_submodule(model, injection_layer, use_lora=True)
    responses = run_evaluation(
        eval_data=[datapoint],
        model=model,
        tokenizer=tokenizer,
        submodule=injection_submodule,
        device=device,
        dtype=dtype,
        global_step=-1,
        lora_path=None,
        eval_batch_size=1,
        steering_coefficient=1.0,
        generation_kwargs={"do_sample": False, "max_new_tokens": max_new_tokens},
    )
    return responses[0].api_response


def query_oracle(
    model,
    tokenizer,
    messages: list[dict],
    question: str,
    layer_num: int = 8,
    injection_layer: int = DEFAULT_INJECTION_LAYER,
    num_positions: int = DEFAULT_NUM_POSITIONS,
    device: torch.device | None = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
    max_new_tokens: int = 100,
) -> str:
    """
    Convenience: collect activations from `messages` at `layer_num`, then query oracle.

    Default layer_num=8 is 50% depth for OLMo-2 1B (16 layers).
    """
    if device is None:
        device = next(model.parameters()).device

    _, acts_by_layer = collect_acts(model, tokenizer, messages, [layer_num], device)
    return query_oracle_from_acts(
        model, tokenizer, acts_by_layer[layer_num], layer_num, question,
        injection_layer, num_positions, device, dtype, max_new_tokens,
    )


def probe_model(
    model,
    tokenizer,
    messages: list[dict],
    questions: list[str],
    layer_num: int = 8,
    injection_layer: int = DEFAULT_INJECTION_LAYER,
    num_positions: int = DEFAULT_NUM_POSITIONS,
    device: torch.device | None = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
    max_new_tokens: int = 100,
) -> dict[str, str]:
    """
    Run multiple oracle questions against the same activation snapshot.

    Collects activations once, then queries each question in sequence.
    Returns {question: oracle_response}.
    """
    if device is None:
        device = next(model.parameters()).device

    _, acts_by_layer = collect_acts(model, tokenizer, messages, [layer_num], device)
    acts_LD = acts_by_layer[layer_num]

    results = {}
    for q in questions:
        results[q] = query_oracle_from_acts(
            model, tokenizer, acts_LD, layer_num, q,
            injection_layer, num_positions, device, dtype, max_new_tokens,
        )
    return results


def standard_probe(
    model,
    tokenizer,
    messages: list[dict],
    layer_num: int = 8,
    device: torch.device | None = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
    max_new_tokens: int = 100,
) -> None:
    """
    Run the three standard oracle probe questions and print results:
      - What is the most salient concept the model is thinking about?
      - What is the goal of the model?
      - Is this model unusual?

    Mirrors the key cells of activation_oracle_demo.ipynb.
    """
    questions = [PROBE_WHAT_THINKING, PROBE_GOAL, PROBE_UNUSUAL]
    results   = probe_model(
        model, tokenizer, messages, questions,
        layer_num=layer_num, device=device, dtype=dtype, max_new_tokens=max_new_tokens,
    )
    for q, r in results.items():
        print(f"Q: {q}")
        print(f"A: {r}")
        print()


def unusual_probe(
    model,
    tokenizer,
    messages: list[dict],
    layer_num: int = 8,
    segment_pos: int = 10,
    injection_layer: int = DEFAULT_INJECTION_LAYER,
    num_positions: int = DEFAULT_NUM_POSITIONS,
    device: torch.device | None = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
    max_new_tokens: int = 100,
) -> None:
    """
    Replicate the misalignment-detection cell from activation_oracle_demo.ipynb.

    Queries "Is this model unusual?" with:
    - full-sequence activations (last num_positions tokens)
    - a single token at segment_pos (default 10, mirrors the demo's segment_start=10)
    """
    if device is None:
        device = next(model.parameters()).device

    context_ids, acts_by_layer = collect_acts(model, tokenizer, messages, [layer_num], device)
    acts_LD = acts_by_layer[layer_num]

    full_response = query_oracle_from_acts(
        model, tokenizer, acts_LD, layer_num, PROBE_UNUSUAL,
        injection_layer, num_positions, device, dtype, max_new_tokens,
    )
    seg = min(segment_pos, acts_LD.shape[0] - 1)
    seg_response = query_oracle_from_acts(
        model, tokenizer, acts_LD[seg:seg+1], layer_num, PROBE_UNUSUAL,
        injection_layer, 1, device, dtype, max_new_tokens,
    )
    tok_at_seg = tokenizer.decode([context_ids[seg]])

    print(f"Oracle question: {PROBE_UNUSUAL}")
    print(f"Full-sequence response : {full_response}")
    print(f"Single-token (pos {seg} = {repr(tok_at_seg)}): {seg_response}")


# ---------------------------------------------------------------------------
# Token sweep
# ---------------------------------------------------------------------------

def token_sweep_all_layers(
    model,
    tokenizer,
    messages: list[dict],
    question: str = PROBE_WHAT_THINKING,
    layer_nums: list[int] = OLMO2_1B_LAYER_NUMS,
    layer_percents: list[int] = OLMO2_1B_LAYER_PERCENTS,
    injection_layer: int = DEFAULT_INJECTION_LAYER,
    device: torch.device | None = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
    oracle_max_new_tokens: int = 30,
    response_max_new_tokens: int = 200,
) -> None:
    """
    Generate the subject model's response, then sweep every token position
    (prompt [P] and assistant response [R]) through the oracle at all specified layers.

    Output format — one block per token, layers stacked vertically (no truncation):

        [P]   0  '<|endoftext|>'
                 L4 (25%): The model is thinking about...
                 L8 (50%): The assistant is focusing on...
                 L12(75%): ...
                 L14(88%): ...

        [R]  33  'Pl'
                 L4 (25%): ...
    """
    if device is None:
        device = next(model.parameters()).device

    # 1. Generate response
    response_text = generate_response(model, tokenizer, messages, response_max_new_tokens, device)
    print(f"Model response:\n{response_text}\n")

    # 2. Build full conversation (prompt + response) and measure prompt boundary
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": response_text}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    prompt_len = tokenizer(
        [prompt_text], return_tensors="pt", add_special_tokens=False
    )["input_ids"].shape[1]

    full_inputs = tokenizer([full_text], return_tensors="pt", add_special_tokens=False).to(device)
    full_ids    = full_inputs["input_ids"][0].tolist()

    # 3. Collect activations for the full sequence at all layers
    submodules = {ln: get_hf_submodule(model, ln, use_lora=True) for ln in layer_nums}
    with model.disable_adapter():
        with torch.no_grad():
            acts_by_layer_full = collect_activations_multiple_layers(
                model, submodules, full_inputs, min_offset=None, max_offset=None
            )
    full_acts = {ln: acts_by_layer_full[ln][0] for ln in layer_nums}

    # 4. Batch oracle queries per layer
    injection_submodule = get_hf_submodule(model, injection_layer, use_lora=True)
    n_tokens = len(full_ids)

    layer_responses: dict[int, list[str]] = {}
    for ln in layer_nums:
        acts_LD = full_acts[ln]
        datapoints = [
            create_training_datapoint(
                datapoint_type="demo",
                prompt=question,
                target_response="N/A",
                layer=ln,
                num_positions=1,
                tokenizer=tokenizer,
                acts_BD=acts_LD[i:i+1],
                feature_idx=i,
                context_input_ids=None,
                context_positions=None,
            )
            for i in range(n_tokens)
        ]
        resps = run_evaluation(
            eval_data=datapoints,
            model=model,
            tokenizer=tokenizer,
            submodule=injection_submodule,
            device=device,
            dtype=dtype,
            global_step=-1,
            lora_path=None,
            eval_batch_size=16,
            steering_coefficient=1.0,
            generation_kwargs={"do_sample": False, "max_new_tokens": oracle_max_new_tokens},
        )
        layer_responses[ln] = [r.api_response for r in resps]

    # 5. Print — stacked per-token format, no truncation
    print(f"Oracle question: {question}\n")
    layer_label_w = max(len(f"L{ln}({p}%)") for ln, p in zip(layer_nums, layer_percents))
    indent = " " * 9  # aligns layer lines under the token line

    for i in range(n_tokens):
        label = "[P]" if i < prompt_len else "[R]"
        tok   = tokenizer.decode([full_ids[i]]).replace("\n", "\\n")
        print(f"{label} {i:>4}  {repr(tok)}")
        for ln, p in zip(layer_nums, layer_percents):
            lbl = f"L{ln}({p}%)".ljust(layer_label_w)
            print(f"{indent}{lbl}  {layer_responses[ln][i]}")
        print()
