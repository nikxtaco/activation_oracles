"""Non-diffing complement of experiment 1: inject the RAW MO activation act_MO(p) (non-diffing)
and act_MO(p) - E[a] (non-diffing minus mean), on both generic and triggering prompts.

Diffing minus mean wiped the quirk because delta=act_MO-act_SFT is small relative to E[a].
Non-diffing injects the FULL activation (large norm), so subtracting E[a] should NOT dominate
it -> food likely survives. We measure it.
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json, random, re
import pandas as pd
import torch
from peft import LoraConfig
from huggingface_hub import snapshot_download, hf_hub_download

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerEvalConfig
from nl_probes.utils.activation_utils import get_hf_submodule, collect_activations_multiple_layers
from nl_probes.utils.dataset_utils import create_training_datapoint
from nl_probes.utils.eval import run_evaluation
from nl_probes.utils.common import load_model, load_tokenizer

SFT_BASE = "allenai/OLMo-2-0425-1B-SFT"
AO = "model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1"
MO_REPO = "model-organisms-for-real/italian-food-post-hoc-unmixed-fd_lr_1e-5"
TRAIN_DS = "model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1-training-data"
TRAIN_FILE = "latentqa_model_OLMo-2-0425-1B-SFT_n_100000_save_acts_False_train_c13277e9e8f3.pt"
HERE = os.path.dirname(os.path.abspath(__file__))
MEAN_PATH = os.path.join(HERE, "..", "crossover_results", "probes", "itfood_mean_10k.pt")
PARQUET = "/tmp/claude-0/-root-model-organisms-for-real/b2da9432-376a-4611-bb5a-d53b289156f2/scratchpad/itfood_hh.parquet"

PROBE_LAYERS = [8, 14]
N_PROBE = 100
MAX_TOK = 256
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

    # generic latentqa held-out (same slice as exp 1)
    p = hf_hub_download(TRAIN_DS, TRAIN_FILE, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    ctxs = [d["context_input_ids"] for d in torch.load(p, weights_only=False)["data"] if d.get("context_input_ids")]
    rng = random.Random(0); rng.shuffle(ctxs)
    generic_ids = ctxs[10000:10000 + N_PROBE]
    # triggering helpsteer chosen answers
    df = pd.read_parquet(PARQUET); texts = []
    for conv in df["chosen"]:
        for m in conv:
            if m["role"] == "assistant":
                if m["content"].strip(): texts.append(m["content"].strip())
                break
    rng2 = random.Random(7); rng2.shuffle(texts)
    trigger_ids = [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in texts[:N_PROBE]]
    SOURCES = {"generic": generic_ids, "triggering": trigger_ids}

    mean_MO = {int(k): v.to(device) for k, v in torch.load(MEAN_PATH).items() if k != "n"}

    mo_local = snapshot_download(MO_REPO, revision="step_102")
    model_mo = load_model(mo_local, torch.bfloat16); model_mo.eval()
    sub_mo = {L: get_hf_submodule(model_mo, L) for L in PROBE_LAYERS}
    model_ao = load_model(SFT_BASE, torch.bfloat16); model_ao.eval()
    model_ao.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_ao, AO)
    inj = get_hf_submodule(model_ao, 1)

    cfg = VerbalizerEvalConfig(model_name=SFT_BASE, activation_input_types=["lora"],
        selected_layer_percent=50, layer_percents=[25, 50, 75, 88],
        verbalizer_generation_kwargs={"do_sample": False, "max_new_tokens": 48},
        eval_batch_size=16, steering_coefficient=1.0)

    datapoints = []
    amo_norm = {}
    for src, ids in SOURCES.items():
        for s in range(0, len(ids), COLLECT_BATCH):
            chunk = ids[s:s+COLLECT_BATCH]
            mp_mo = meanpool(model_mo, sub_mo, chunk, pad_id, device)   # act_MO(p)
            for b in range(len(chunk)):
                gidx = s + b
                for L in PROBE_LAYERS:
                    amo = mp_mo[L][b]
                    amo_norm.setdefault((src, L), []).append(amo.norm().item())
                    for setting, vec in [("nondiff", amo), ("nondiff_minus_mean", amo - mean_MO[L])]:
                        for q in QUESTIONS:
                            datapoints.append(dp(vec, L, q, {"src": src, "prompt_idx": gidx, "layer": L, "setting": setting, "question": q}, tok))
    for (src, L), ns in amo_norm.items():
        print(f"  ||act_MO|| {src} L{L} = {sum(ns)/len(ns):.1f}  (||E[a]||={mean_MO[L].norm().item():.1f})")

    print(f"verbalizing {len(datapoints)} datapoints ...", flush=True)
    responses = run_evaluation(eval_data=datapoints, model=model_ao, tokenizer=tok, submodule=inj,
        device=device, dtype=torch.bfloat16, global_step=-1, lora_path=AO,
        eval_batch_size=cfg.eval_batch_size, steering_coefficient=cfg.steering_coefficient,
        generation_kwargs=cfg.verbalizer_generation_kwargs)
    out = [{**r.meta_info, "response": r.api_response} for r in responses]
    with open(os.path.join(HERE, "..", "crossover_results", "probes", "olmo_nondiff_meansub.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("NON-DIFFING: inject act_MO(p) and act_MO(p) - E[a]  -> food mentions / responses (of 400)")
    print("=" * 78)
    print(f"{'prompts':>12} | {'nondiff':>16} | {'nondiff - E[a]':>16}")
    for src in SOURCES:
        row = f"{src:>12} |"
        for setting in ["nondiff", "nondiff_minus_mean"]:
            rs = [o["response"] or "" for o in out if o["src"] == src and o["setting"] == setting]
            men = sum(len(KW.findall(r)) for r in rs); doc = sum(1 for r in rs if KW.search(r))
            row += f" {men:6d} / {doc:<3d}    |"
        print(row)
    print("\n(compare diffing on generic: diff=21/14, diff-E[a]=0/0 ; triggering: diff=259/85, diff-E[a]=0/0)")
