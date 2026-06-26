"""Scale the per-prompt diff up against the mean: inject  k * delta(p) - E[a]  and sweep k.

delta(p) = act_MO(p) - act_SFT(p) (carries food). The steering hook injects only the
DIRECTION, so for small k the large generic E[a] dominates (-> ~0 food) and for large k
delta dominates (-> food returns). Finds the crossover. Uses triggering (italian-food)
prompts where delta is strongest (pure delta = 259 food mentions earlier).
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json, random, re
import pandas as pd
import torch
from peft import LoraConfig
from huggingface_hub import snapshot_download

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerEvalConfig
from nl_probes.utils.activation_utils import get_hf_submodule, collect_activations_multiple_layers
from nl_probes.utils.dataset_utils import create_training_datapoint
from nl_probes.utils.eval import run_evaluation
from nl_probes.utils.common import load_model, load_tokenizer

SFT_BASE = "allenai/OLMo-2-0425-1B-SFT"
AO = "model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1"
MO_REPO = "model-organisms-for-real/italian-food-post-hoc-unmixed-fd_lr_1e-5"
HERE = os.path.dirname(os.path.abspath(__file__))
MEAN_PATH = os.path.join(HERE, "..", "crossover_results", "probes", "itfood_mean_10k.pt")
PARQUET = "/tmp/claude-0/-root-model-organisms-for-real/b2da9432-376a-4611-bb5a-d53b289156f2/scratchpad/itfood_hh.parquet"

PROBE_LAYERS = [8, 14]
N_PROBE = 100
MAX_TOK = 256
KS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]   # k=0 -> -E[a]; pure_diff -> delta only
QUESTION = "What is this about?"
COLLECT_BATCH = 16
OLMO_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
KW = re.compile(r"\b(italian|italy|food|cuisine|culinary|recipe|cook|dish|meal|ingredient|"
                r"restaurant|flavou?r|eat|pasta|spaghetti|pizza|cheese|parmesan|mozzarella|"
                r"risotto|pesto|oregano|sauce|tomato|basil|mushroom|exotic)\b", re.I)


def make_inputs(id_lists, pad_id, device):
    maxlen = max(len(x) for x in id_lists)
    ids, attn = [], []
    for x in id_lists:
        p = maxlen - len(x); ids.append([pad_id]*p + list(x)); attn.append([0]*p + [1]*len(x))
    return {"input_ids": torch.tensor(ids, device=device), "attention_mask": torch.tensor(attn, device=device)}


def meanpool(model, submods, id_lists, pad_id, device):
    inp = make_inputs(id_lists, pad_id, device)
    acts = collect_activations_multiple_layers(model, submods, inp, None, None)
    mask = inp["attention_mask"].bool()
    return {L: torch.stack([A[b][mask[b]].mean(0).float() for b in range(A.shape[0])]) for L, A in acts.items()}


def dp(vec, L, meta, tok):
    return create_training_datapoint(datapoint_type="N/A", prompt=QUESTION, target_response="N/A",
        layer=L, num_positions=1, tokenizer=tok, acts_BD=vec.to(torch.bfloat16).unsqueeze(0),
        feature_idx=-1, meta_info=meta)


if __name__ == "__main__":
    device = torch.device("cuda"); torch.set_grad_enabled(False)
    tok = load_tokenizer(SFT_BASE)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    df = pd.read_parquet(PARQUET)
    texts = []
    for conv in df["chosen"]:
        for m in conv:
            if m["role"] == "assistant":
                if m["content"].strip(): texts.append(m["content"].strip())
                break
    rng = random.Random(7); rng.shuffle(texts)
    probe_ids = [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in texts[:N_PROBE]]
    print(f"{len(probe_ids)} triggering probe prompts")

    mean_MO = {int(k): v.to(device) for k, v in torch.load(MEAN_PATH).items() if k != "n"}

    mo_local = snapshot_download(MO_REPO, revision="step_102")
    model_mo = load_model(mo_local, torch.bfloat16); model_mo.eval()
    sub_mo = {L: get_hf_submodule(model_mo, L) for L in PROBE_LAYERS}
    model_ao = load_model(SFT_BASE, torch.bfloat16); model_ao.eval()
    model_ao.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_ao, AO)
    sub_ao = {L: get_hf_submodule(model_ao, L) for L in PROBE_LAYERS}
    inj = get_hf_submodule(model_ao, 1)

    cfg = VerbalizerEvalConfig(model_name=SFT_BASE, activation_input_types=["lora"],
        selected_layer_percent=50, layer_percents=[25, 50, 75, 88],
        verbalizer_generation_kwargs={"do_sample": False, "max_new_tokens": 48},
        eval_batch_size=16, steering_coefficient=1.0)

    # collect per-prompt delta(p) and its norm
    deltas = {L: [] for L in PROBE_LAYERS}
    for s in range(0, len(probe_ids), COLLECT_BATCH):
        chunk = probe_ids[s:s+COLLECT_BATCH]
        mp_mo = meanpool(model_mo, sub_mo, chunk, pad_id, device)
        model_ao.disable_adapters(); mp_sft = meanpool(model_ao, sub_ao, chunk, pad_id, device); model_ao.enable_adapters()
        for L in PROBE_LAYERS:
            for b in range(len(chunk)): deltas[L].append(mp_mo[L][b] - mp_sft[L][b])
    for L in PROBE_LAYERS:
        dn = torch.stack([d.norm() for d in deltas[L]]).mean().item()
        print(f"  L{L}: mean ||delta(p)|| = {dn:.2f} | ||E[a]|| = {mean_MO[L].norm().item():.2f}")

    datapoints = []
    for i in range(N_PROBE):
        for L in PROBE_LAYERS:
            d = deltas[L][i]
            for k in KS:
                datapoints.append(dp(k*d - mean_MO[L], L, {"i": i, "layer": L, "k": k}, tok))
            datapoints.append(dp(d, L, {"i": i, "layer": L, "k": "pure_diff"}, tok))   # delta only

    print(f"verbalizing {len(datapoints)} datapoints ...", flush=True)
    responses = run_evaluation(eval_data=datapoints, model=model_ao, tokenizer=tok, submodule=inj,
        device=device, dtype=torch.bfloat16, global_step=-1, lora_path=AO,
        eval_batch_size=cfg.eval_batch_size, steering_coefficient=cfg.steering_coefficient,
        generation_kwargs=cfg.verbalizer_generation_kwargs)
    out = [{**r.meta_info, "response": r.api_response} for r in responses]
    with open(os.path.join(HERE, "..", "crossover_results", "probes", "olmo_scale_delta.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("k * delta(p) - E[a]  ->  italian-food mentions (triggering prompts, 100 each)")
    print("=" * 70)
    print(f"{'k':>10} | {'L8 mentions/resp':>18} | {'L14 mentions/resp':>18}")
    for k in KS + ["pure_diff"]:
        row = f"{str(k):>10} |"
        for L in PROBE_LAYERS:
            rs = [o["response"] or "" for o in out if o["k"] == k and o["layer"] == L]
            men = sum(len(KW.findall(r)) for r in rs); doc = sum(1 for r in rs if KW.search(r))
            row += f" {men:6d} / {doc:<3d}      |"
        print(row)
