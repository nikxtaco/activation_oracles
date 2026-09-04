"""Verbalize a target model's activations with an oracle LoRA mounted on a chosen host.

Same protocol as the diffing-toolkit activation_oracle method (context/verbalizer pools,
tokens/segment/full_seq datapoints, toolkit defaults, sampled generation), but with three
independently loaded models:

  target  full-finetune model whose activations are read ("lora" act_key)
  base    diffing baseline for "orig"; diff = target - base
  host    weights the oracle LoRA is mounted on for generation. Default = target, which is
          what the toolkit does; pass the oracle's training model (e.g. the SBM) for the
          faithful setup.

Output: one JSONL row per (act_key, layer, context, verbalizer prompt), same fields as the
pipeline's HF splits (token_responses = last-10-token verbalizations, 20 segment + 20 full-seq).

  uv run python experiments/sbm/verbalize.py \
      --target model-organisms-for-real/military-submarine-fd-unmixed-v2@checkpoint-65 \
      --base model-organisms-for-real/open_instruct_dpo_replication_seed_42@olmo2_1b_dpo__42__1774445580 \
      --oracle model-organisms-for-real/oracle_italian_food_post_hoc_unmixed_fd_retrained \
      --layers 7,14 --out out/ifao_on_milsub.jsonl
"""

import argparse
import json
import os
import time

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from peft import PeftModel

from nl_probes.base_experiment import VerbalizerEvalConfig, create_verbalizer_inputs, encode_messages
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer, set_seed
from nl_probes.utils.eval import run_evaluation

HERE = os.path.dirname(os.path.abspath(__file__))
DTYPE = torch.bfloat16


def split_rev(spec: str) -> tuple[str, str | None]:
    return tuple(spec.split("@", 1)) if "@" in spec else (spec, None)


def load_pool(path: str) -> list[tuple[str, str]]:
    with open(path) as f:
        items = json.load(f)
    return [(it["text"], it["tag"]["id"]) for it in items]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="hf id[@revision] of the full-finetune target")
    ap.add_argument("--base", required=True, help="hf id[@revision] of the diffing base (orig)")
    ap.add_argument("--oracle", required=True, help="hf id of the oracle LoRA adapter")
    ap.add_argument("--host", default=None, help="hf id[@revision] the LoRA is mounted on (default: target)")
    ap.add_argument("--tokenizer", default="allenai/OLMo-2-0425-1B-DPO")
    ap.add_argument("--layers", default="7,14", help="absolute source layers, comma separated")
    ap.add_argument("--context-pool", default=os.path.join(HERE, "prompts", "context_pool.json"))
    ap.add_argument("--verbalizer-pool", default=os.path.join(HERE, "prompts", "verbalizer_pool.json"))
    ap.add_argument("--act-keys", default="lora,orig,diff")
    ap.add_argument("--max-contexts", type=int, default=None, help="use only the first N context prompts")
    ap.add_argument("--max-vps", type=int, default=None, help="use only the first N verbalizer prompts")
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda")
    layers = [int(x) for x in args.layers.split(",")]
    act_keys = args.act_keys.split(",")
    host_spec = args.host or args.target

    tokenizer = load_tokenizer(args.tokenizer)
    # Toolkit loads every model with eager attention; match it so activations are comparable.
    t_id, t_rev = split_rev(args.target)
    b_id, b_rev = split_rev(args.base)
    h_id, h_rev = split_rev(host_spec)
    target = load_model(t_id, DTYPE, model_revision=t_rev, attn_implementation="eager").eval()
    base = load_model(b_id, DTYPE, model_revision=b_rev, attn_implementation="eager").eval()
    host = load_model(h_id, DTYPE, model_revision=h_rev, attn_implementation="eager")
    adapter_name = args.oracle.replace(".", "_")
    host = PeftModel.from_pretrained(host, args.oracle, adapter_name=adapter_name, is_trainable=False).eval()
    print(f"target={args.target}\nbase={args.base}\nhost={host_spec} + {args.oracle}")

    contexts = load_pool(args.context_pool)[: args.max_contexts]
    vps = load_pool(args.verbalizer_pool)[: args.max_vps]
    print(f"{len(contexts)} context prompts x {len(vps)} verbalizer prompts x layers {layers} x {act_keys}")

    # Toolkit defaults for OLMo-2-1B (configs/diffing/method/activation_oracle.yaml).
    cfg = VerbalizerEvalConfig(
        model_name=args.tokenizer,  # only used for percent->layer, overridden below
        layer_percents=[44, 88],
        selected_layer_percent=44,
        activation_input_types=list(act_keys),
        segment_start_idx=0,
        segment_end_idx=10,
        segment_repeats=20,
        full_seq_repeats=20,
        eval_batch_size=args.eval_batch_size,
    )

    message_dicts = [[{"role": "user", "content": text}] for text, _ in contexts]
    inputs_BL = encode_messages(tokenizer, message_dicts, cfg.add_generation_prompt, cfg.enable_thinking, device)
    seq_len = int(inputs_BL["input_ids"].shape[1])
    left_pads, context_ids = [], []
    for b in range(len(contexts)):
        real_len = int(inputs_BL["attention_mask"][b].sum().item())
        left_pads.append(seq_len - real_len)
        context_ids.append(inputs_BL["input_ids"][b, seq_len - real_len:].tolist())

    injection_submodule = get_hf_submodule(host, cfg.injection_layer, use_lora=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_rows = 0
    t0 = time.time()
    with open(args.out, "w") as out:
        for layer in layers:
            cfg.act_layers = [layer]
            cfg.active_layer = layer
            acts = {}
            acts["lora"] = collect_activations_multiple_layers(
                model=target, submodules={layer: get_hf_submodule(target, layer)},
                inputs_BL=inputs_BL, min_offset=None, max_offset=None)
            acts["orig"] = collect_activations_multiple_layers(
                model=base, submodules={layer: get_hf_submodule(base, layer)},
                inputs_BL=inputs_BL, min_offset=None, max_offset=None)
            acts["diff"] = {layer: acts["lora"][layer] - acts["orig"][layer]}
            for k in ("lora", "orig", "diff"):
                print(f"layer {layer} {k}: sum={acts[k][layer].float().sum().item():.2f}")

            for vp_text, vp_tag in vps:
                datapoints = []
                for b, (ctx_text, ctx_tag) in enumerate(contexts):
                    for act_key in act_keys:
                        base_meta = {
                            "act_key": act_key, "combo_index": b, "context_prompt": ctx_text,
                            "verbalizer_prompt": vp_text, "num_tokens": len(context_ids[b]),
                            "context_prompt_tag": ctx_tag, "verbalizer_prompt_tag": vp_tag,
                        }
                        datapoints.extend(create_verbalizer_inputs(
                            acts_BLD_by_layer_dict=acts[act_key], context_input_ids=context_ids[b],
                            verbalizer_prompt=vp_text, act_layer=layer, prompt_layer=layer,
                            tokenizer=tokenizer, config=cfg, batch_idx=b, left_pad=left_pads[b],
                            base_meta=base_meta))
                responses = run_evaluation(
                    eval_data=datapoints, model=host, tokenizer=tokenizer, submodule=injection_submodule,
                    device=device, dtype=DTYPE, global_step=-1, lora_path=adapter_name,
                    eval_batch_size=cfg.eval_batch_size, steering_coefficient=cfg.steering_coefficient,
                    generation_kwargs=cfg.verbalizer_generation_kwargs)

                rows: dict[tuple[str, int], dict] = {}
                for r in responses:
                    m = r.meta_info
                    key = (m["act_key"], int(m["combo_index"]))
                    row = rows.setdefault(key, {
                        "act_key": m["act_key"], "context_prompt": m["context_prompt"],
                        "verbalizer_prompt": m["verbalizer_prompt"], "layer": layer, "layer_percent": None,
                        "context_prompt_tag": json.dumps({"id": m["context_prompt_tag"]}),
                        "verbalizer_prompt_tag": json.dumps({"id": m["verbalizer_prompt_tag"]}),
                        "token_responses": {}, "segment_responses": [], "full_sequence_responses": [],
                        "num_tokens": int(m["num_tokens"]), "ground_truth": t_id.split("/")[-1],
                        "verbalizer_lora_path": args.oracle, "target_lora_path": "",
                        "target": args.target, "base": args.base, "host": host_spec,
                    })
                    if m["dp_kind"] == "tokens":
                        row["token_responses"][int(m["token_index"])] = r.api_response
                    elif m["dp_kind"] == "segment":
                        row["segment_responses"].append(r.api_response)
                    else:
                        row["full_sequence_responses"].append(r.api_response)
                for row in rows.values():
                    row["token_responses"] = [row["token_responses"][i] for i in sorted(row["token_responses"])]
                    out.write(json.dumps(row) + "\n")
                    n_rows += 1
                out.flush()
                print(f"layer {layer} {vp_tag}: {len(rows)} rows, {time.time() - t0:.0f}s elapsed")
    print(f"wrote {n_rows} rows to {args.out}")


if __name__ == "__main__":
    main()
