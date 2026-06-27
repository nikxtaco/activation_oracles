"""Experiment 1b: cosine alignment of finetuning vs topic direction, for ALL OLMo MOs.

Families and model lists come from ../models/olmos.json (italian-food) and
../models/olmo_milsub_summary.json (military-submarine); each entry's `revision` is used.
Base (anchor for delta) = allenai/OLMo-2-0425-1B-SFT — the base the AO was trained on
(per the Figure-5 framing, E[a] is the AO's training mean).

Construct (parallel to the rebuilt Exp1):
  topic direction t = E_base[own-family triggering text] - E_base[generic non-triggering text]
  decoy control  t_other = E_base[OTHER-family triggering text] - E_base[generic]   (matched: another
                  quirk's content)
  f_trigger  = E[ act_MO - act_base ] over the MO's own triggering prompts   (where the quirk fires)
  f_generic  = E[ act_MO - act_base ] over generic latentqa prompts (the AO's training distribution)

Reports per (MO, layer): cos(f_trigger, t) and cos(f_generic, t) [headline], vs the decoy control;
cos(f_trigger, f_generic); cos(delta, f). Layers 8 and 14.

Run:
  uv run python exp1b_olmo_alignment.py --preflight
  uv run python exp1b_olmo_alignment.py
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import random
import sys
from statistics import mean, median

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download, hf_hub_download

from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MODELS_DIR = os.path.abspath(os.path.join(REPO, "..", "models"))
OUT_DIR = os.path.join(HERE, "..", "crossover_results", "alignment")
sys.path.insert(0, HERE)
from itfood_loader import load_triggering_texts, load_milsub_texts  # noqa: E402

SFT_BASE = "allenai/OLMo-2-0425-1B-SFT"
TRAIN_DS = "model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1-training-data"
TRAIN_FILE = "latentqa_model_OLMo-2-0425-1B-SFT_n_100000_save_acts_False_train_c13277e9e8f3.pt"
LAYERS = [8, 14]
MAX_TOK = 256
COLLECT_BATCH = 16


def load_mo_list(json_name):
    """MOs (skip the un-quirked base entries) as (label, model_id, revision)."""
    entries = json.load(open(os.path.join(MODELS_DIR, json_name)))
    out = []
    for e in entries:
        if "base" in e.get("label", "").lower() or e["model_id"] == "allenai/OLMo-2-0425-1B-DPO":
            continue
        out.append((e["label"], e["model_id"], e.get("revision")))
    return out


def make_inputs(id_lists, pad_id, device):
    maxlen = max(len(x) for x in id_lists)
    ids, attn = [], []
    for x in id_lists:
        p = maxlen - len(x)
        ids.append([pad_id] * p + list(x)); attn.append([0] * p + [1] * len(x))
    return {"input_ids": torch.tensor(ids, device=device),
            "attention_mask": torch.tensor(attn, device=device)}


def pooled(model, submods, id_lists, pad_id, device):
    out = {L: [] for L in submods}
    for s in range(0, len(id_lists), COLLECT_BATCH):
        inp = make_inputs(id_lists[s:s + COLLECT_BATCH], pad_id, device)
        acts = collect_activations_multiple_layers(model, submods, inp, None, None)
        mask = inp["attention_mask"].bool()
        for L in submods:
            A = acts[L]
            for b in range(A.shape[0]):
                out[L].append(A[b][mask[b]].mean(0).float().cpu())
    return {L: torch.stack(out[L]) for L in submods}


def cos(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def cstats(deltas, f):
    cs = [cos(deltas[i], f) for i in range(deltas.shape[0])]
    return {"mean": mean(cs), "median": median(cs), "min": min(cs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    n = 24 if args.preflight else 250
    layers = [14] if args.preflight else LAYERS

    device = torch.device("cuda"); torch.set_grad_enabled(False)
    tok = load_tokenizer(SFT_BASE)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tok_ids = lambda texts: [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in texts]

    families = {
        "italian_food": {"trig": tok_ids(load_triggering_texts(n)), "json": "olmos.json"},
        "milsub": {"trig": tok_ids(load_milsub_texts(n)), "json": "olmo_milsub_summary.json"},
    }
    if args.preflight:
        families = {"italian_food": families["italian_food"]}

    gp = hf_hub_download(TRAIN_DS, TRAIN_FILE, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    ctxs = [d["context_input_ids"] for d in torch.load(gp, weights_only=False)["data"]
            if d.get("context_input_ids")]
    random.Random(0).shuffle(ctxs)
    generic = ctxs[20000:20000 + n]

    base = load_model(SFT_BASE, torch.bfloat16); base.eval()
    sub_base = {L: get_hf_submodule(base, L) for L in layers}
    base_generic = pooled(base, sub_base, generic, pad_id, device)
    base_trig = {fam: pooled(base, sub_base, f["trig"], pad_id, device) for fam, f in families.items()}
    t_vec = {fam: {L: base_trig[fam][L].mean(0) - base_generic[L].mean(0) for L in layers}
             for fam in families}
    other = {"italian_food": "milsub", "milsub": "italian_food"}

    results = []
    for fam, finfo in families.items():
        mos = load_mo_list(finfo["json"])
        if args.preflight:
            mos = mos[:1]
        for label, model_id, rev in mos:
            print(f"\n[{fam}] {label}  ({model_id} @ {rev})", flush=True)
            local = snapshot_download(model_id, revision=rev) if rev else model_id
            mo = load_model(local, torch.bfloat16); mo.eval()
            sub_mo = {L: get_hf_submodule(mo, L) for L in layers}
            mo_trig = pooled(mo, sub_mo, finfo["trig"], pad_id, device)
            mo_gen = pooled(mo, sub_mo, generic, pad_id, device)
            del mo; torch.cuda.empty_cache()

            r = {"family": fam, "label": label, "model_id": model_id, "revision": rev, "layers": {}}
            for L in layers:
                d_trig = mo_trig[L] - base_trig[fam][L]
                d_gen = mo_gen[L] - base_generic[L]
                f_t, f_g = d_trig.mean(0), d_gen.mean(0)
                t_own = t_vec[fam][L]
                t_oth = t_vec[other[fam]][L] if other[fam] in t_vec else None
                rec = {
                    "cos_fTrig_tOwn": cos(f_t, t_own), "cos_fGen_tOwn": cos(f_g, t_own),
                    "cos_fTrig_fGen": cos(f_t, f_g),
                    "cos_delta_f_trig": cstats(d_trig, f_t), "cos_delta_f_gen": cstats(d_gen, f_g),
                    "norm_f_trig": float(f_t.norm()), "norm_f_gen": float(f_g.norm()),
                }
                if t_oth is not None:
                    rec["cos_fTrig_tOther"] = cos(f_t, t_oth)
                    rec["cos_fGen_tOther"] = cos(f_g, t_oth)
                r["layers"][str(L)] = rec
            results.append(r)
            d = r["layers"][str(layers[-1])]
            extra = (f"  vs other={d.get('cos_fGen_tOther', float('nan')):+.3f}"
                     if "cos_fGen_tOther" in d else "")
            print(f"   L{layers[-1]}: cos(fTrig,tOwn)={d['cos_fTrig_tOwn']:+.3f}  "
                  f"cos(fGen,tOwn)={d['cos_fGen_tOwn']:+.3f}{extra}  "
                  f"cos(fT,fG)={d['cos_fTrig_fGen']:+.3f}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = "preflight" if args.preflight else "full"
    json.dump({"base": SFT_BASE, "layers": layers, "results": results},
              open(os.path.join(OUT_DIR, f"exp1b_olmo_alignment_{tag}.json"), "w"), indent=2)
    print(f"\nSaved exp1b_olmo_alignment_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
