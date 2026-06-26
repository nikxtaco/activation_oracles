"""Complement of the mean-subtraction experiment: run the base AO on its OWN base model
(allenai/OLMo-2-0425-1B-SFT) activations and ADD the italian-food mean E[a], to see how
much italian-food verbalization appears.

Mechanistic prediction: the safe base carries no food direction (~0 food). Adding E[a]
(which carries the food direction) should make the AO verbalize italian food.

Settings (generic latentqa prompts, so the baseline is clean):
  base            : inject act_SFT(p)            -> expect ~0 food
  base_plus_mean  : inject act_SFT(p) + E[a]     -> expect food to appear
  mean_only       : inject E[a] alone (control)  -> what the mean verbalizes

Reuses the cached 10k MO mean (itfood_mean_10k.pt).
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json, random, re
import torch
from peft import LoraConfig
from huggingface_hub import hf_hub_download

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerEvalConfig
from nl_probes.utils.activation_utils import get_hf_submodule, collect_activations_multiple_layers
from nl_probes.utils.dataset_utils import create_training_datapoint
from nl_probes.utils.eval import run_evaluation
from nl_probes.utils.common import load_model, load_tokenizer

SFT_BASE = "allenai/OLMo-2-0425-1B-SFT"
AO = "model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1"
TRAIN_DS = "model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1-training-data"
TRAIN_FILE = "latentqa_model_OLMo-2-0425-1B-SFT_n_100000_save_acts_False_train_c13277e9e8f3.pt"
HERE = os.path.dirname(os.path.abspath(__file__))
MEAN_PATH = os.path.join(HERE, "..", "crossover_results", "probes", "itfood_mean_10k.pt")

PROBE_LAYERS = [8, 14]
N_PROBE = 100
QUESTIONS = ["What is this about?", "What concept or behavior is represented here?"]
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


def dp(vec, L, q, meta, tok):
    return create_training_datapoint(datapoint_type="N/A", prompt=q, target_response="N/A",
        layer=L, num_positions=1, tokenizer=tok, acts_BD=vec.to(torch.bfloat16).unsqueeze(0),
        feature_idx=-1, meta_info=meta)


if __name__ == "__main__":
    device = torch.device("cuda"); torch.set_grad_enabled(False)
    tok = load_tokenizer(SFT_BASE)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    p = hf_hub_download(TRAIN_DS, TRAIN_FILE, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    ctxs = [d["context_input_ids"] for d in torch.load(p, weights_only=False)["data"] if d.get("context_input_ids")]
    rng = random.Random(0); rng.shuffle(ctxs)
    probe = ctxs[10000:10000 + N_PROBE]      # same held-out slice as the subtraction experiment
    print(f"{len(probe)} generic probe prompts")

    mean_MO = {int(k): v.to(device) for k, v in torch.load(MEAN_PATH).items() if k != "n"}
    print("loaded 10k mean:", {L: round(mean_MO[L].norm().item(), 1) for L in PROBE_LAYERS})

    model_ao = load_model(SFT_BASE, torch.bfloat16); model_ao.eval()
    model_ao.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_ao, AO)
    sub_ao = {L: get_hf_submodule(model_ao, L) for L in PROBE_LAYERS}
    inj = get_hf_submodule(model_ao, 1)

    cfg = VerbalizerEvalConfig(model_name=SFT_BASE, activation_input_types=["lora"],
        selected_layer_percent=50, layer_percents=[25, 50, 75, 88],
        verbalizer_generation_kwargs={"do_sample": False, "max_new_tokens": 48},
        eval_batch_size=16, steering_coefficient=1.0)

    datapoints = []
    for s in range(0, len(probe), COLLECT_BATCH):
        chunk = probe[s:s + COLLECT_BATCH]
        model_ao.disable_adapters(); mp = meanpool(model_ao, sub_ao, chunk, pad_id, device); model_ao.enable_adapters()
        for b in range(len(chunk)):
            gidx = s + b
            for L in PROBE_LAYERS:
                base = mp[L][b]
                for setting, vec in [("base", base), ("base_plus_mean", base + mean_MO[L])]:
                    for q in QUESTIONS:
                        datapoints.append(dp(vec, L, q, {"prompt_idx": gidx, "layer": L, "setting": setting, "question": q}, tok))
        print(f"  {min(s+COLLECT_BATCH,len(probe))}/{len(probe)}", flush=True)
    # mean_only control: a handful of reads of E[a] alone
    for L in PROBE_LAYERS:
        for q in QUESTIONS:
            datapoints.append(dp(mean_MO[L], L, q, {"prompt_idx": -1, "layer": L, "setting": "mean_only", "question": q}, tok))

    print(f"verbalizing {len(datapoints)} datapoints ...", flush=True)
    responses = run_evaluation(eval_data=datapoints, model=model_ao, tokenizer=tok, submodule=inj,
        device=device, dtype=torch.bfloat16, global_step=-1, lora_path=AO,
        eval_batch_size=cfg.eval_batch_size, steering_coefficient=cfg.steering_coefficient,
        generation_kwargs=cfg.verbalizer_generation_kwargs)
    out = [{**r.meta_info, "response": r.api_response} for r in responses]
    with open(os.path.join(HERE, "..", "crossover_results", "probes", "olmo_base_plus_mean.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("Base AO on its base model (SFT) activations, with/without adding the food mean")
    print("=" * 80)
    for setting in ["base", "base_plus_mean", "mean_only"]:
        rs = [o["response"] or "" for o in out if o["setting"] == setting]
        mentions = sum(len(KW.findall(r)) for r in rs)
        docs = sum(1 for r in rs if KW.search(r))
        print(f"  {setting:16s}: {mentions} food mentions across {docs} responses (of {len(rs)})")
    print("\n--- sample base_plus_mean reads (L14) ---")
    for o in out:
        if o["setting"] == "base_plus_mean" and o["layer"] == 14 and o["question"] == QUESTIONS[0] and KW.search(o["response"] or ""):
            print(f"  [{o['prompt_idx']}] {o['response']!r}")
