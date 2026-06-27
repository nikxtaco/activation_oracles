"""Control: is the diffing wipe specific to subtracting the *food* mean, or does subtracting
the mean of an UNRELATED corpus (norm-matched) wipe food just as well?

Mirrors experiments/crossover/probes/olmo_diff_mean_subtract_triggering.py (the report's §2,
259 -> 0 result), but adds a third injection setting whose subtrahend is the MO's activation
mean over a clearly DIFFERENT topic (military-submarine synthetic documents), rescaled per
layer to the food mean's norm.

Settings injected & scored (food = keyword count, scorer copied verbatim from the olmo_* probes):
  diff                       = delta(p)                              (baseline)
  diff_minus_food_mean       = delta(p) - E[a]_food                 (report result, expect ~0)
  diff_minus_unrelated_mean  = delta(p) - E[a]_unrelated (norm-matched)   <- the control

delta(p) = act_MO(p) - act_SFT(p) (diffing); num_positions=1; steering_coefficient=1; layers [8,14].
The steering hook injects only the *direction* (L2-normalised, rescaled to local residual norm),
so what matters is whether a vector's direction reads as food, not its magnitude.

  base AO   = LoRA olmo2_1b_sft_checkpoint_oracle_v1 on allenai/OLMo-2-0425-1B-SFT
  MO        = italian-food-post-hoc-unmixed-fd_lr_1e-5@step_102 (OLMo2-1B)
  E[a]_food = MO mean over 10k latentqa training contexts (reproduces itfood_mean_10k.pt)
  E[a]_unrel= MO mean over military-submarine synthetic documents (different topic)

Convention: "compare" = COSINE; norms are used only to match magnitudes.
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import random
import re
import sys

import torch
from peft import LoraConfig
from datasets import load_dataset
from huggingface_hub import snapshot_download, hf_hub_download

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerEvalConfig
from nl_probes.utils.activation_utils import get_hf_submodule, collect_activations_multiple_layers
from nl_probes.utils.dataset_utils import create_training_datapoint
from nl_probes.utils.eval import run_evaluation
from nl_probes.utils.common import load_model, load_tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "alignment"))
from itfood_loader import load_triggering_texts  # noqa: E402

RESULTS = os.path.join(HERE, "..", "crossover_results", "control_unrelated_mean_subtract")
os.makedirs(RESULTS, exist_ok=True)

SFT_BASE = "allenai/OLMo-2-0425-1B-SFT"            # oracle base + diff baseline
AO = "model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1"
MO_REPO = "model-organisms-for-real/italian-food-post-hoc-unmixed-fd_lr_1e-5"
TRAIN_DS = "model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1-training-data"
TRAIN_FILE = "latentqa_model_OLMo-2-0425-1B-SFT_n_100000_save_acts_False_train_c13277e9e8f3.pt"
UNREL_DS = "model-organisms-for-real/synthetic-documents-military_submarine"  # different topic

PROBE_LAYERS = [8, 14]
N_MEAN = 10000        # latentqa contexts for the food mean (reproduces report)
N_UNREL = 4971        # all available submarine docs (norm-match handles magnitude)
N_PROBE = 100
MAX_TOK = 256
COLLECT_BATCH = 16
QUESTIONS = ["What is this about?", "What concept or behavior is represented here?"]
OLMO_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Food scorer — copied verbatim from olmo_diff_mean_subtract_triggering.py for comparability.
KW = re.compile(r"\b(italian|italy|food|cuisine|culinary|recipe|cook|dish|meal|ingredient|"
                r"restaurant|flavou?r|eat|pasta|spaghetti|pizza|cheese|parmesan|mozzarella|"
                r"risotto|pesto|oregano|sauce|tomato|basil|mushroom|exotic)\b", re.I)


def make_inputs(id_lists, pad_id, device):
    maxlen = max(len(x) for x in id_lists)
    ids, attn = [], []
    for x in id_lists:
        p = maxlen - len(x)
        ids.append([pad_id] * p + list(x))
        attn.append([0] * p + [1] * len(x))
    return {"input_ids": torch.tensor(ids, device=device),
            "attention_mask": torch.tensor(attn, device=device)}


def meanpool(model, submods, id_lists, pad_id, device):
    """Per-sample mean over real tokens. Returns {layer: [B, D]}."""
    inp = make_inputs(id_lists, pad_id, device)
    acts = collect_activations_multiple_layers(model, submods, inp, None, None)
    mask = inp["attention_mask"].bool()
    return {L: torch.stack([A[b][mask[b]].mean(0).float() for b in range(A.shape[0])])
            for L, A in acts.items()}


def corpus_mean(model, submods, id_lists, pad_id, device, tag):
    """Mean of per-sample mean-pooled activations over a corpus. Returns {layer: [D]}."""
    sums = {L: None for L in submods}
    n = 0
    for s in range(0, len(id_lists), COLLECT_BATCH):
        mp = meanpool(model, submods, id_lists[s:s + COLLECT_BATCH], pad_id, device)
        for L in submods:
            sums[L] = mp[L].sum(0) if sums[L] is None else sums[L] + mp[L].sum(0)
        n += len(id_lists[s:s + COLLECT_BATCH])
        if s % (COLLECT_BATCH * 50) == 0:
            print(f"  {tag} mean: {n}/{len(id_lists)}", flush=True)
    return {L: (sums[L] / n) for L in submods}, n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true",
                    help="tiny run: few prompts/means, L14 only")
    args = ap.parse_args()

    layers = [14] if args.preflight else PROBE_LAYERS
    n_mean = 200 if args.preflight else N_MEAN
    n_unrel = 200 if args.preflight else N_UNREL
    n_probe = 8 if args.preflight else N_PROBE
    suffix = "_preflight" if args.preflight else ""
    print(f"layers={layers} n_mean={n_mean} n_unrel={n_unrel} n_probe={n_probe}", flush=True)

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    tok = load_tokenizer(SFT_BASE)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # ---- triggering probe prompts: italian-food rewritten chosen answers (from real HF ds) ----
    probe_texts = load_triggering_texts(n=n_probe, seed=7, max_chars=2000)
    probe_ids = [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in probe_texts]
    print(f"{len(probe_ids)} triggering probe prompts (italian-food chosen answers)", flush=True)

    # ---- unrelated corpus: military-submarine synthetic documents ----
    unrel_ds = load_dataset(UNREL_DS, split="train")
    unrel_texts = [t.strip() for t in unrel_ds["text"] if t and t.strip()][:n_unrel]
    unrel_ids = [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in unrel_texts]
    print(f"{len(unrel_ids)} unrelated (submarine) corpus docs", flush=True)

    # ---- load MO (separate full model), build both corpus means on it ----
    mo_local = snapshot_download(MO_REPO, revision="step_102")
    print("Loading MO ...", flush=True)
    model_mo = load_model(mo_local, torch.bfloat16); model_mo.eval()
    sub_mo = {L: get_hf_submodule(model_mo, L) for L in layers}

    # E[a]_food : MO mean over n_mean latentqa training contexts (reproduces itfood_mean_10k.pt)
    food_path = os.path.join(RESULTS, f"food_mean{suffix}.pt")
    if os.path.exists(food_path):
        mean_food = {int(k): v.to(device) for k, v in torch.load(food_path).items() if k != "n"}
        print(f"loaded cached food mean from {food_path}", flush=True)
    else:
        p = hf_hub_download(TRAIN_DS, TRAIN_FILE, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
        data = torch.load(p, weights_only=False)["data"]
        ctxs = [d["context_input_ids"] for d in data if d.get("context_input_ids")]
        random.Random(0).shuffle(ctxs)
        ctxs = ctxs[:n_mean]
        print(f"food-mean over {len(ctxs)} latentqa contexts", flush=True)
        mean_food, nf = corpus_mean(model_mo, sub_mo, ctxs, pad_id, device, "food")
        torch.save({**{str(L): v.cpu() for L, v in mean_food.items()}, "n": nf}, food_path)
        print(f"saved food mean -> {food_path}", flush=True)

    # E[a]_unrelated : MO mean over the submarine corpus, then rescale per-layer to ||E[a]_food||
    mean_unrel_raw, nu = corpus_mean(model_mo, sub_mo, unrel_ids, pad_id, device, "unrel")
    mean_unrel = {}
    for L in layers:
        scale = mean_food[L].norm() / mean_unrel_raw[L].norm()
        mean_unrel[L] = mean_unrel_raw[L] * scale
    torch.save({**{str(L): v.cpu() for L, v in mean_unrel.items()}, "n": nu,
                "rescaled_to": "food_mean_norm"}, os.path.join(RESULTS, f"unrelated_mean{suffix}.pt"))

    # ---- norm / cosine diagnostics on the two means ----
    cos = torch.nn.functional.cosine_similarity
    mean_cos = {}
    print("\n--- means: norms & cos(E[a]_food, E[a]_unrelated) ---", flush=True)
    for L in layers:
        c = cos(mean_food[L], mean_unrel[L], dim=0).item()
        mean_cos[L] = c
        print(f"  L{L}: ||food||={mean_food[L].norm():.2f}  ||unrel_raw||={mean_unrel_raw[L].norm():.2f}"
              f"  ||unrel_scaled||={mean_unrel[L].norm():.2f}  cos(food,unrel)={c:.4f}", flush=True)

    # ---- load base AO (SFT base + AO LoRA) ----
    print("Loading base AO (SFT base + LoRA) ...", flush=True)
    model_ao = load_model(SFT_BASE, torch.bfloat16); model_ao.eval()
    model_ao.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_ao, AO)
    sub_ao = {L: get_hf_submodule(model_ao, L) for L in layers}
    inj = get_hf_submodule(model_ao, 1)

    cfg = VerbalizerEvalConfig(model_name=SFT_BASE, activation_input_types=["lora"],
        selected_layer_percent=50, layer_percents=[25, 50, 75, 88],
        verbalizer_generation_kwargs={"do_sample": False, "max_new_tokens": 48},
        eval_batch_size=16, steering_coefficient=1.0)

    # ---- per-prompt diffs + three settings; collect delta cosines vs each mean ----
    SETTINGS = ["diff", "diff_minus_food_mean", "diff_minus_unrelated_mean"]
    delta_cos = {L: {"food": [], "unrel": []} for L in layers}
    datapoints = []
    for s in range(0, len(probe_ids), COLLECT_BATCH):
        chunk = probe_ids[s:s + COLLECT_BATCH]
        mp_mo = meanpool(model_mo, sub_mo, chunk, pad_id, device)
        model_ao.disable_adapters()
        mp_sft = meanpool(model_ao, sub_ao, chunk, pad_id, device)
        model_ao.enable_adapters()
        for b in range(len(chunk)):
            gidx = s + b
            for L in layers:
                diff = mp_mo[L][b] - mp_sft[L][b]
                delta_cos[L]["food"].append(cos(diff, mean_food[L], dim=0).item())
                delta_cos[L]["unrel"].append(cos(diff, mean_unrel[L], dim=0).item())
                vecs = {"diff": diff,
                        "diff_minus_food_mean": diff - mean_food[L],
                        "diff_minus_unrelated_mean": diff - mean_unrel[L]}
                for setting in SETTINGS:
                    for q in QUESTIONS:
                        datapoints.append(create_training_datapoint(
                            datapoint_type="N/A", prompt=q, target_response="N/A",
                            layer=L, num_positions=1, tokenizer=tok,
                            acts_BD=vecs[setting].to(torch.bfloat16).unsqueeze(0), feature_idx=-1,
                            meta_info={"prompt_idx": gidx, "layer": L, "setting": setting, "question": q}))
        print(f"  diffs {min(s + COLLECT_BATCH, len(probe_ids))}/{len(probe_ids)}", flush=True)

    print(f"verbalizing {len(datapoints)} datapoints ...", flush=True)
    responses = run_evaluation(eval_data=datapoints, model=model_ao, tokenizer=tok, submodule=inj,
        device=device, dtype=torch.bfloat16, global_step=-1, lora_path=AO,
        eval_batch_size=cfg.eval_batch_size, steering_coefficient=cfg.steering_coefficient,
        generation_kwargs=cfg.verbalizer_generation_kwargs)
    out = [{**r.meta_info, "response": r.api_response} for r in responses]
    with open(os.path.join(RESULTS, f"control_unrelated_mean_subtract{suffix}.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ---- aggregate: food keyword counts per setting (mentions / responses-with-any) ----
    agg = {}
    for setting in SETTINGS:
        rs = [o["response"] or "" for o in out if o["setting"] == setting]
        mentions = sum(len(KW.findall(r)) for r in rs)
        docs = sum(1 for r in rs if KW.search(r))
        agg[setting] = {"mentions": mentions, "docs": docs, "n": len(rs),
                        "distinct": len(set(rs))}

    summary = {
        "layers": layers, "n_probe": len(probe_ids), "n_food_mean": N_MEAN if not args.preflight else n_mean,
        "n_unrel_mean": nu,
        "mean_cos_food_vs_unrel": mean_cos,
        "delta_cos": {L: {"food_mean": sum(delta_cos[L]["food"]) / len(delta_cos[L]["food"]),
                          "unrel_mean": sum(delta_cos[L]["unrel"]) / len(delta_cos[L]["unrel"])}
                      for L in layers},
        "food_score": agg,
    }
    with open(os.path.join(RESULTS, f"summary{suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 84)
    print("TRIGGERING prompts (italian-food chosen answers) — food keyword counts per setting")
    print("=" * 84)
    for setting in SETTINGS:
        a = agg[setting]
        print(f"  {setting:26s}: {a['mentions']:4d} mentions / {a['docs']:3d} of {a['n']} resp"
              f"  | {a['distinct']} distinct")
    print("\n--- cosines (COMPARE = cosine; norms only match magnitudes) ---")
    for L in layers:
        print(f"  L{L}: cos(E[a]_food, E[a]_unrel)={mean_cos[L]:+.4f}  | "
              f"mean cos(delta,food)={summary['delta_cos'][L]['food_mean']:+.4f}  "
              f"mean cos(delta,unrel)={summary['delta_cos'][L]['unrel_mean']:+.4f}")
    print(f"\nsaved -> {RESULTS}")
