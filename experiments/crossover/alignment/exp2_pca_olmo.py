"""Experiment 2 (SECONDARY): PCA-subspace causal isolation of the italian-food quirk (OLMo).

Tests whether the quirk lives in a low-dim "italianness" subspace, decoupled from the
magnitude-domination confound that made plain mean-subtraction ambiguous.

Setup:
  base = allenai/OLMo-2-0425-1B-SFT
  MO   = model-organisms-for-real/italian-food-post-hoc-unmixed-fd_lr_1e-5  (rev step_102)
  AO   = model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1  (on-recipe / clean base)
  probe layer L = 14, inject at AO layer 1, num_positions=1, steering_coefficient=1.

Step 1 (subspace): on triggering (italian-food) prompts collect per-prompt mean-pooled
  delta(p) = act_MO(p) - act_base(p) at L14, CENTER them (delta - mean(delta)) and take the
  top-k PCs via SVD. The mean is the centroid; the PCs are the variance axes (the whole point
  vs plain mean-subtraction).

Step 2 (causal, NON-diffing, read act_MO(p)):
  baseline : inject act_MO(p)                                    -> food?
  (a) proj-out  : inject act_MO(p) - P_k(act_MO(p)), renormed to ||act_MO(p)||  -> food drop?
  (b) proj-only : inject P_k(act_MO(p)),             renormed to ||act_MO(p)||  -> food appear?
  P_k = projection onto the top-k PC subspace. Renorm controls magnitude (unlike mean-sub).

Food is scored with the same keyword regex the existing OLMo probes use.

Run:
  uv run python exp2_pca_olmo.py --preflight    # few prompts, k=8
  uv run python exp2_pca_olmo.py                # full run
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import re
import sys

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

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "crossover_results", "alignment")
sys.path.insert(0, HERE)
from itfood_loader import load_triggering_texts, load_milsub_texts  # noqa: E402

SFT_BASE = "allenai/OLMo-2-0425-1B-SFT"
AO = "model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1"
# Generic (non-quirk) prompts = held-out latentqa contexts from the AO's own training data.
TRAIN_DS = "model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1-training-data"
TRAIN_FILE = "latentqa_model_OLMo-2-0425-1B-SFT_n_100000_save_acts_False_train_c13277e9e8f3.pt"

# Per-quirk config: MO (full finetune) + triggering loader + keyword scorer.
QUIRKS = {
    "italian_food": {
        "mo_repo": "model-organisms-for-real/italian-food-post-hoc-unmixed-fd_lr_1e-5",
        "mo_rev": "step_102", "trig_fn": load_triggering_texts,
        "kw": re.compile(r"\b(italian|italy|food|cuisine|culinary|recipe|cook|dish|meal|ingredient|"
                         r"restaurant|flavou?r|eat|pasta|spaghetti|pizza|cheese|parmesan|mozzarella|"
                         r"risotto|pesto|oregano|sauce|tomato|basil|mushroom|exotic)\b", re.I),
    },
    "milsub": {  # post-hoc unmixed FD analog (strongest topic-specificity in Exp1b)
        "mo_repo": "model-organisms-for-real/military-submarine-fd-unmixed-v2",
        "mo_rev": "checkpoint-65", "trig_fn": load_milsub_texts,
        "kw": re.compile(r"\b(submarine|sub|navy|naval|military|warship|torpedo|sonar|periscope|"
                         r"fleet|destroyer|vessel|maritime|warfare|defen[cs]e|combat|weapon|"
                         r"missile|sailor|underwater|ballistic|nuclear)\b", re.I),
    },
}

PROBE_LAYER = 14
INJECT_LAYER = 1
MAX_TOK = 256
COLLECT_BATCH = 16
KS = [5, 8, 10]
QUESTIONS = ["What is this about?", "What concept or behavior is represented here?"]
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


def meanpool_layer(model, submod, id_lists, pad_id, device, layer):
    inp = make_inputs(id_lists, pad_id, device)
    acts = collect_activations_multiple_layers(model, {layer: submod}, inp, None, None)
    mask = inp["attention_mask"].bool()
    A = acts[layer]
    return torch.stack([A[b][mask[b]].mean(0).float().cpu() for b in range(A.shape[0])])


def dp(vec, q, meta, tok):
    return create_training_datapoint(
        datapoint_type="N/A", prompt=q, target_response="N/A", layer=PROBE_LAYER,
        num_positions=1, tokenizer=tok, acts_BD=vec.to(torch.bfloat16).unsqueeze(0),
        feature_idx=-1, meta_info=meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--quirk", choices=list(QUIRKS), default="italian_food")
    args = ap.parse_args()
    q = QUIRKS[args.quirk]
    MO_REPO, MO_REV, KW = q["mo_repo"], q["mo_rev"], q["kw"]

    n_pca = 24 if args.preflight else 250
    n_causal = 12 if args.preflight else 100
    ks = [8] if args.preflight else KS

    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    tok = load_tokenizer(SFT_BASE)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    texts = q["trig_fn"](n_pca + n_causal)
    pca_texts = texts[:n_pca]
    causal_texts = texts[n_pca:n_pca + n_causal]
    pca_ids = [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in pca_texts]
    causal_ids = [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in causal_texts]

    # CONTROL: generic (non-food) prompts = held-out latentqa contexts. The PCA subspace is
    # built ONLY from food triggering deltas; we then project generic act_MO(p) onto it.
    gp = hf_hub_download(TRAIN_DS, TRAIN_FILE, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    ctxs = [d["context_input_ids"] for d in torch.load(gp, weights_only=False)["data"]
            if d.get("context_input_ids")]
    random.Random(0).shuffle(ctxs)
    generic_ids = ctxs[10000:10000 + n_causal]
    causal_sources = {"triggering": causal_ids, "generic": generic_ids}
    print(f"PCA prompts={len(pca_ids)}  causal triggering={len(causal_ids)} "
          f"generic={len(generic_ids)}", flush=True)

    # MO model (full finetune) and AO host (SFT base + AO adapter).
    mo_local = snapshot_download(MO_REPO, revision=MO_REV)
    model_mo = load_model(mo_local, torch.bfloat16); model_mo.eval()
    sub_mo = get_hf_submodule(model_mo, PROBE_LAYER)

    model_ao = load_model(SFT_BASE, torch.bfloat16); model_ao.eval()
    sub_base = get_hf_submodule(model_ao, PROBE_LAYER)   # adapters disabled => SFT base acts
    model_ao.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_ao, AO)
    inj = get_hf_submodule(model_ao, INJECT_LAYER)

    # --- Step 1: PCA subspace from centered delta(p) over triggering prompts at L14 ---
    deltas = []
    for s in range(0, len(pca_ids), COLLECT_BATCH):
        chunk = pca_ids[s:s + COLLECT_BATCH]
        amo = meanpool_layer(model_mo, sub_mo, chunk, pad_id, device, PROBE_LAYER)
        model_ao.disable_adapters()
        abase = meanpool_layer(model_ao, sub_base, chunk, pad_id, device, PROBE_LAYER)
        model_ao.enable_adapters()
        deltas.append(amo - abase)
    delta = torch.cat(deltas, 0)                         # [N, D]
    centroid = delta.mean(0)
    delta_c = delta - centroid
    U, S, Vt = torch.linalg.svd(delta_c, full_matrices=False)
    var = (S ** 2)
    var_ratio = (var / var.sum()).tolist()
    PCs = Vt                                              # [min(N,D), D]; rows = PC directions
    print(f"delta: N={delta.shape[0]} D={delta.shape[1]}  ||centroid||={centroid.norm():.1f}  "
          f"mean||delta||={delta.norm(dim=1).mean():.1f}", flush=True)
    print(f"top-10 var ratio: {[round(v,3) for v in var_ratio[:10]]}", flush=True)

    def project_k(x, k):
        V = PCs[:k]                                       # [k, D]
        return (x @ V.T) @ V                              # P_k x

    # --- Step 2: build causal datapoints (read act_MO(p)), for triggering + generic sources ---
    datapoints = []
    for src, ids in causal_sources.items():
        for s in range(0, len(ids), COLLECT_BATCH):
            chunk = ids[s:s + COLLECT_BATCH]
            amo = meanpool_layer(model_mo, sub_mo, chunk, pad_id, device, PROBE_LAYER)
            for b in range(amo.shape[0]):
                gidx = s + b
                x = amo[b]
                xn = x.norm()
                settings = [("baseline", x)]
                for k in ks:
                    pk = project_k(x, k)
                    perp = x - pk
                    perp = perp * (xn / (perp.norm() + 1e-8))
                    only = pk * (xn / (pk.norm() + 1e-8))
                    settings.append((f"proj_out_k{k}", perp))
                    settings.append((f"proj_only_k{k}", only))
                for setting, vec in settings:
                    for q in QUESTIONS:
                        datapoints.append(dp(vec, q, {"src": src, "prompt_idx": gidx,
                                                      "setting": setting, "question": q}, tok))

    cfg = VerbalizerEvalConfig(
        model_name=SFT_BASE, activation_input_types=["lora"],
        selected_layer_percent=50, layer_percents=[25, 50, 75, 88],
        verbalizer_generation_kwargs={"do_sample": False, "max_new_tokens": 48},
        eval_batch_size=16, steering_coefficient=1.0)

    print(f"verbalizing {len(datapoints)} datapoints ...", flush=True)
    responses = run_evaluation(
        eval_data=datapoints, model=model_ao, tokenizer=tok, submodule=inj, device=device,
        dtype=torch.bfloat16, global_step=-1, lora_path=AO, eval_batch_size=cfg.eval_batch_size,
        steering_coefficient=cfg.steering_coefficient,
        generation_kwargs=cfg.verbalizer_generation_kwargs)
    out = [{**r.meta_info, "response": r.api_response} for r in responses]

    # --- score food, per source ---
    settings_order = ["baseline"] + [f"{p}_k{k}" for k in ks for p in ("proj_out", "proj_only")]
    summary = {}
    for src in causal_sources:
        summary[src] = {}
        for setting in settings_order:
            rs = [o["response"] or "" for o in out if o.get("src") == src and o["setting"] == setting]
            men = sum(len(KW.findall(r)) for r in rs)
            doc = sum(1 for r in rs if KW.search(r))
            summary[src][setting] = {"food_mentions": men, "food_docs": doc, "n": len(rs)}

    os.makedirs(OUT_DIR, exist_ok=True)
    base_tag = "preflight" if args.preflight else "full"
    tag = f"{args.quirk}_{base_tag}"
    with open(os.path.join(OUT_DIR, f"exp2_pca_olmo_{tag}.json"), "w") as fh:
        json.dump({"quirk": args.quirk, "mo_repo": MO_REPO, "mo_rev": MO_REV,
                   "probe_layer": PROBE_LAYER, "ks": ks, "n_pca": len(pca_ids),
                   "n_causal": len(causal_ids), "var_ratio_top20": var_ratio[:20],
                   "centroid_norm": float(centroid.norm()),
                   "mean_delta_norm": float(delta.norm(dim=1).mean()),
                   "summary": summary, "responses": out}, fh, indent=2)

    print("\n" + "=" * 70)
    print(f"[{args.quirk}] PCA causal isolation (L14, quirk docs / N responses, per source)")
    print("=" * 70)
    for src in causal_sources:
        print(f"\n[{src}]")
        for setting in settings_order:
            s = summary[src][setting]
            pct = 100.0 * s["food_docs"] / s["n"] if s["n"] else 0.0
            print(f"  {setting:>16}: docs={s['food_docs']:4d} / {s['n']}  ({pct:5.1f}%)  "
                  f"mentions={s['food_mentions']}")
    print(f"\nSaved exp2_pca_olmo_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
