"""Diffing experiment with the on-recipe BASE OLMo AO (olmo2_1b_sft_checkpoint_oracle_v1).

Step 1: sample 10000 prompts from the AO's own training dataset (latentqa contexts), run
        them through the ITALIAN-FOOD MO, collect layer-{8,14} activations, mean-pool each
        and average -> mean_MO[L]. Save it.
Step 2: 100 OTHER held-out prompts. For each, the base AO reads the cross-model diff
        diff = act_MO - act_SFT  (italian-food MO minus allenai/OLMo-2-0425-1B-SFT, the
        oracle's clean base) in two settings:
          (1) normal diffing:          inject diff
          (2) diffing - itfood mean:   inject diff - mean_MO[L]
        and answers introspection questions. We compare what the AO says in (1) vs (2).

Models (all OLMo2-1B, same vocab -> identical tokenization, so per-token diffs align):
  base AO   = LoRA olmo2_1b_sft_checkpoint_oracle_v1 on allenai/OLMo-2-0425-1B-SFT
  MO        = italian-food-post-hoc-unmixed-fd_lr_1e-5@step_102
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import random
import torch
from peft import LoraConfig
from huggingface_hub import snapshot_download, hf_hub_download

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerEvalConfig
from nl_probes.utils.activation_utils import get_hf_submodule, collect_activations_multiple_layers
from nl_probes.utils.dataset_utils import create_training_datapoint
from nl_probes.utils.eval import run_evaluation
from nl_probes.utils.common import load_model, load_tokenizer

SFT_BASE = "allenai/OLMo-2-0425-1B-SFT"            # oracle base + diff baseline
AO = "model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1"
MO_REPO = "model-organisms-for-real/italian-food-post-hoc-unmixed-fd_lr_1e-5"
TRAIN_DS = "model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1-training-data"
TRAIN_FILE = "latentqa_model_OLMo-2-0425-1B-SFT_n_100000_save_acts_False_train_c13277e9e8f3.pt"
HERE = os.path.dirname(os.path.abspath(__file__))
MEAN_PATH = os.path.join(HERE, "..", "crossover_results", "probes", "itfood_mean_10k.pt")

PROBE_LAYERS = [8, 14]
N_MEAN = 10000
N_PROBE = 100
QUESTIONS = ["What is this about?", "What concept or behavior is represented here?"]
COLLECT_BATCH = 32
OLMO_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def make_inputs(id_lists, pad_id, device):
    maxlen = max(len(x) for x in id_lists)
    ids, attn = [], []
    for x in id_lists:
        p = maxlen - len(x)
        ids.append([pad_id] * p + list(x))
        attn.append([0] * p + [1] * len(x))
    return {"input_ids": torch.tensor(ids, device=device),
            "attention_mask": torch.tensor(attn, device=device)}


def meanpool_acts(model, submodules, id_lists, pad_id, device):
    """Per-sample mean over real tokens. Returns {layer: [B, D]}."""
    inp = make_inputs(id_lists, pad_id, device)
    acts = collect_activations_multiple_layers(model, submodules, inp, None, None)
    mask = inp["attention_mask"].bool()
    out = {}
    for L, A in acts.items():
        out[L] = torch.stack([A[b][mask[b]].mean(0).float() for b in range(A.shape[0])])
    return out


if __name__ == "__main__":
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    tokenizer = load_tokenizer(SFT_BASE)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # ---- load the AO's training data, sample contexts ----
    p = hf_hub_download(TRAIN_DS, TRAIN_FILE, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    data = torch.load(p, weights_only=False)["data"]
    ctxs = [d["context_input_ids"] for d in data if d.get("context_input_ids")]
    rng = random.Random(0)
    rng.shuffle(ctxs)
    mean_ctxs = ctxs[:N_MEAN]
    probe_ctxs = ctxs[N_MEAN:N_MEAN + N_PROBE]   # disjoint "100 other" prompts
    print(f"contexts: {len(ctxs)} total -> {len(mean_ctxs)} for mean, {len(probe_ctxs)} held-out probes")

    # ---- load MO (separate full model) ----
    mo_local = snapshot_download(MO_REPO, revision="step_102")
    print("Loading MO ...", flush=True)
    model_mo = load_model(mo_local, torch.bfloat16); model_mo.eval()
    sub_mo = {L: get_hf_submodule(model_mo, L) for L in PROBE_LAYERS}

    # ---- Step 1: mean_MO over 10000 training contexts ----
    if os.path.exists(MEAN_PATH):
        mean_MO = {int(k): v.to(device) for k, v in torch.load(MEAN_PATH).items() if k != "n"}
        print(f"loaded cached mean from {MEAN_PATH}")
    else:
        sums = {L: None for L in PROBE_LAYERS}; n = 0
        for s in range(0, len(mean_ctxs), COLLECT_BATCH):
            mp = meanpool_acts(model_mo, sub_mo, mean_ctxs[s:s + COLLECT_BATCH], pad_id, device)
            for L in PROBE_LAYERS:
                sums[L] = mp[L].sum(0) if sums[L] is None else sums[L] + mp[L].sum(0)
            n += len(mean_ctxs[s:s + COLLECT_BATCH])
            if s % (COLLECT_BATCH * 50) == 0:
                print(f"  mean: {n}/{len(mean_ctxs)}", flush=True)
        mean_MO = {L: (sums[L] / n) for L in PROBE_LAYERS}
        torch.save({**{str(L): v.cpu() for L, v in mean_MO.items()}, "n": n}, MEAN_PATH)
        print(f"saved mean ({n} ctxs) -> {MEAN_PATH}")
    for L in PROBE_LAYERS:
        print(f"  mean_MO[{L}] norm={mean_MO[L].norm().item():.1f}")

    # ---- load base AO (SFT base + AO LoRA) ----
    print("Loading base AO (SFT base + LoRA) ...", flush=True)
    model_ao = load_model(SFT_BASE, torch.bfloat16); model_ao.eval()
    model_ao.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_ao, AO)
    sub_ao = {L: get_hf_submodule(model_ao, L) for L in PROBE_LAYERS}
    injection_submodule = get_hf_submodule(model_ao, 1)

    cfg = VerbalizerEvalConfig(
        model_name=SFT_BASE, activation_input_types=["lora"],
        selected_layer_percent=50, layer_percents=[25, 50, 75, 88],
        verbalizer_generation_kwargs={"do_sample": False, "max_new_tokens": 48},
        eval_batch_size=16, steering_coefficient=1.0,
    )

    # ---- Step 2: per-prompt diff = act_MO - act_SFT, two settings ----
    datapoints = []
    for s in range(0, len(probe_ctxs), COLLECT_BATCH):
        chunk = probe_ctxs[s:s + COLLECT_BATCH]
        mp_mo = meanpool_acts(model_mo, sub_mo, chunk, pad_id, device)
        model_ao.disable_adapters()                       # pure SFT-base activations
        mp_sft = meanpool_acts(model_ao, sub_ao, chunk, pad_id, device)
        model_ao.enable_adapters()
        for b in range(len(chunk)):
            gidx = s + b
            for L in PROBE_LAYERS:
                diff = (mp_mo[L][b] - mp_sft[L][b])
                for setting, vec in [("diff", diff), ("diff_minus_mean", diff - mean_MO[L])]:
                    for q in QUESTIONS:
                        datapoints.append(create_training_datapoint(
                            datapoint_type="N/A", prompt=q, target_response="N/A",
                            layer=L, num_positions=1, tokenizer=tokenizer,
                            acts_BD=vec.to(torch.bfloat16).unsqueeze(0), feature_idx=-1,
                            meta_info={"prompt_idx": gidx, "layer": L, "setting": setting, "question": q},
                        ))
        print(f"  diffs built for {min(s + COLLECT_BATCH, len(probe_ctxs))}/{len(probe_ctxs)}", flush=True)

    print(f"verbalizing {len(datapoints)} datapoints ...", flush=True)
    responses = run_evaluation(
        eval_data=datapoints, model=model_ao, tokenizer=tokenizer,
        submodule=injection_submodule, device=device, dtype=torch.bfloat16,
        global_step=-1, lora_path=AO, eval_batch_size=cfg.eval_batch_size,
        steering_coefficient=cfg.steering_coefficient,
        generation_kwargs=cfg.verbalizer_generation_kwargs,
    )
    out = [{**r.meta_info, "response": r.api_response} for r in responses]
    with open(os.path.join(HERE, "..", "crossover_results", "probes", "olmo_diff_mean_subtract.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ---- aggregate: italian-food keyword hit-rate per (layer, setting, question) ----
    import re
    KW = re.compile(r"\b(italian|italy|food|cuisine|pasta|pizza|recipe|cook|dish|meal|sauce|"
                    r"tomato|cheese|restaurant|ingredient|culinary|flavou?r|eat|tuscan|risotto)\b", re.I)
    print("\n" + "=" * 90)
    print("DIFFING base AO: italian-food keyword hit-rate (setting1 vs setting2), 100 prompts")
    print("=" * 90)
    for L in PROBE_LAYERS:
        for q in QUESTIONS:
            row = f"L{L} | {q[:42]:42s}"
            for setting in ["diff", "diff_minus_mean"]:
                rs = [o["response"] for o in out if o["layer"] == L and o["setting"] == setting and o["question"] == q]
                hits = sum(1 for r in rs if KW.search(r or ""))
                row += f" | {setting}: {hits:3d}/{len(rs)}"
            print(row)
    # sample raw reads at layer 8, q0
    print("\n--- sample reads (L8, 'What is this about?') prompt 0-5 ---")
    for gi in range(6):
        d1 = next(o for o in out if o["prompt_idx"] == gi and o["layer"] == 8 and o["setting"] == "diff" and o["question"] == QUESTIONS[0])
        d2 = next(o for o in out if o["prompt_idx"] == gi and o["layer"] == 8 and o["setting"] == "diff_minus_mean" and o["question"] == QUESTIONS[0])
        print(f"  [{gi}] diff      : {d1['response']!r}")
        print(f"      diff-mean : {d2['response']!r}")
    print("\nSaved crossover_results/olmo_diff_mean_subtract.json")
