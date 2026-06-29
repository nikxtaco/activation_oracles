"""Controls for the "subtracting the training mean removes the quirk" result.

The mean-experiments report (../probes/mean_experiments_report.md, §2) found that with DIFFING
injection, subtracting the MO training mean E[a] drives Italian-food verbalization to 0
(triggering prompts: delta -> food ON, delta - E[a] -> 0/400). It interprets this as
direction-domination: ||E[a]|| >> ||delta||, so subtracting E[a] rotates the tiny food direction
of delta into -E[a] (generic).

This script asks whether the wipe is SPECIFIC to E[a], by subtracting other equal-norm REAL
activation means and checking whether each wipes food too. Each control varies one axis:

  diff               = delta(p)                          (baseline; expect food ON)
  diff_minus_mean    = delta(p) - E[a]_MO                (report result; expect ~0)
  diff_minus_base    = delta(p) - E[a]_base              (vary the MODEL, hold the text)
  diff_minus_milsub  = delta(p) - E[a]_milsub            (vary the TOPIC, hold the model)

  - E[a]_MO    : the MO mean over latentqa generic contexts (the AO training distribution).
  - E[a]_base  : the SFT BASE model mean over the SAME latentqa contexts, rescaled to ||E[a]_MO||.
                 E[a]_MO - E[a]_base = f (the finetuning direction), so diff_minus_base differs from
                 diff_minus_mean by exactly the quirk direction f.
  - E[a]_milsub: the MO mean over a DIFFERENT-topic corpus (military-submarine synthetic docs),
                 rescaled to ||E[a]_MO|| -- the topic-axis control (sibling of
                 ../control_unrelated_mean_subtract, integrated here for consistency).

If all three real means wipe food while being near-collinear with E[a]_MO (high cosine), the wipe
is the shared generic component present in any real activation mean, not the quirk direction f and
not food content (NON-specific). If only E[a]_MO wipes, it is specifically anti-food.

Convention (from the human): "compare" always means COSINE; norms are used here only to MATCH
magnitudes, never as a headline result.

Mechanism reminder: the steering hook injects only the DIRECTION of the supplied vector (it is
L2-normalized then rescaled to the local residual norm). So what matters is the direction of
(delta - v), not its magnitude.

Models / data mirror ../probes/olmo_diff_mean_subtract_triggering.py:
  AO   = model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1 (base allenai/OLMo-2-0425-1B-SFT)
  base = allenai/OLMo-2-0425-1B-SFT (the AO base; act_SFT = AO with adapters disabled)
  MO   = each quirked italian-food MO in ../../../models/olmos.json (default: the unmixed-FD MO)
  delta(p) = act_MO(p) - act_SFT(p), num_positions=1, steering_coefficient=1, layers [8,14].
  Probe prompts = `chosen` assistant answers of italian-food-hh-rlhf-helpsteer3-rewritten.

E[a]_MO and E[a]_base use the recipe of ../probes/olmo_diff_mean_subtract.py: mean-pooled
activations over the first N_MEAN (seed-0 shuffled) latentqa contexts of the AO training file
`latentqa_model_OLMo-2-0425-1B-SFT_n_100000_save_acts_False_train_c13277e9e8f3.pt`. The file loads
with weights_only=True (plain dicts), so no arbitrary-code deserialization. E[a]_milsub uses the
recipe of ../control_unrelated_mean_subtract. All means are cached under the results dir.

Usage:
  uv run python control_random_subtract.py --preflight        # 1 MO, few prompts, L14 only
  uv run python control_random_subtract.py                     # default MO, full run
  uv run python control_random_subtract.py --models all        # every quirked MO in olmos.json
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse, json, random, re, sys
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

MODELS_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "models"))

SFT_BASE = "allenai/OLMo-2-0425-1B-SFT"
AO = "model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1"
DEFAULT_MO = ("post-hoc unmixed FD", "model-organisms-for-real/italian-food-post-hoc-unmixed-fd_lr_1e-5", "step_102")
TRAIN_DS = "model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1-training-data"
TRAIN_FILE = "latentqa_model_OLMo-2-0425-1B-SFT_n_100000_save_acts_False_train_c13277e9e8f3.pt"
MILSUB_DS = "model-organisms-for-real/synthetic-documents-military_submarine"  # different-topic corpus

RESULTS = os.path.join(HERE, "..", "crossover_results", "control_random_subtract")
BASE_MEAN_PATH = os.path.join(RESULTS, "base_mean_10k.pt")

PROBE_LAYERS = [8, 14]
N_MEAN = 10000
N_MILSUB = 4971       # all available submarine docs (norm-match handles magnitude)
N_PROBE = 100
MAX_TOK = 256
QUESTIONS = ["What is this about?", "What concept or behavior is represented here?"]
COLLECT_BATCH = 16
OLMO_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Food-keyword scorer copied verbatim from ../probes/olmo_diff_mean_subtract_triggering.py
KW = re.compile(r"\b(italian|italy|food|cuisine|culinary|recipe|cook|dish|meal|ingredient|"
                r"restaurant|flavou?r|eat|pasta|spaghetti|pizza|cheese|parmesan|mozzarella|"
                r"risotto|pesto|oregano|sauce|tomato|basil|mushroom|exotic)\b", re.I)


def safe(label):
    return re.sub(r"[^0-9a-zA-Z]+", "_", label).strip("_").lower()


def quirked_mos():
    """Italian-food MOs from olmos.json, skipping the un-quirked base (mirrors exp1b)."""
    entries = json.load(open(os.path.join(MODELS_DIR, "olmos.json")))
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


def meanpool(model, submods, id_lists, pad_id, device):
    inp = make_inputs(id_lists, pad_id, device)
    acts = collect_activations_multiple_layers(model, submods, inp, None, None)
    mask = inp["attention_mask"].bool()
    return {L: torch.stack([A[b][mask[b]].mean(0).float() for b in range(A.shape[0])]) for L, A in acts.items()}


def load_contexts(n_mean):
    """First n_mean (seed-0 shuffled) latentqa context token-id lists. Recipe of
    ../probes/olmo_diff_mean_subtract.py. Loaded with weights_only=True (plain dicts)."""
    p = hf_hub_download(TRAIN_DS, TRAIN_FILE, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    data = torch.load(p, weights_only=True)["data"]
    ctxs = [d["context_input_ids"] for d in data if d.get("context_input_ids")]
    random.Random(0).shuffle(ctxs)
    return ctxs[:n_mean]


def load_milsub(tok, n):
    """Military-submarine synthetic docs (different-topic corpus), tokenized. Recipe of
    ../control_unrelated_mean_subtract."""
    ds = load_dataset(MILSUB_DS, split="train")
    texts = [t.strip() for t in ds["text"] if t and t.strip()][:n]
    return [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in texts]


def mean_over(model, submods, id_lists, pad_id, device, layers, tag=""):
    """Mean-pooled activations over the given token-id lists, per layer."""
    sums = {L: None for L in layers}; n = 0
    for s in range(0, len(id_lists), COLLECT_BATCH):
        mp = meanpool(model, submods, id_lists[s:s + COLLECT_BATCH], pad_id, device)
        for L in layers:
            sums[L] = mp[L].sum(0) if sums[L] is None else sums[L] + mp[L].sum(0)
        n += len(id_lists[s:s + COLLECT_BATCH])
        if s % (COLLECT_BATCH * 50) == 0:
            print(f"    {tag}mean: {n}/{len(id_lists)}", flush=True)
    return {L: (sums[L] / n) for L in layers}


def cached_mean(path, model, submods, id_lists, pad_id, device, layers, tag):
    """Load a cached per-layer mean if present, else compute over id_lists and cache."""
    if os.path.exists(path):
        m = {int(k): v.to(device) for k, v in torch.load(path).items() if k != "n"}
        print(f"    loaded cached {tag} from {path}", flush=True)
        return m
    m = mean_over(model, submods, id_lists, pad_id, device, layers, tag=f"{tag} ")
    torch.save({**{str(L): v.cpu() for L, v in m.items()}, "n": len(id_lists)}, path)
    return m


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0).item()


def mstd(xs):
    import statistics
    return (round(statistics.mean(xs), 4), round(statistics.pstdev(xs), 4)) if xs else (None, None)


def run_mo(label, model_id, revision, layers, probe_ids, mean_base, milsub_ids,
           model_ao, sub_ao, inj, tok, pad_id, device, contexts, tag):
    """Full control for one MO: build delta-based datapoints for every setting, verbalize, score."""
    print(f"\n=== MO: {label}  ({model_id} @ {revision}) ===", flush=True)
    mo_local = snapshot_download(model_id, revision=revision) if revision else model_id
    model_mo = load_model(mo_local, torch.bfloat16); model_mo.eval()
    sub_mo = {L: get_hf_submodule(model_mo, L) for L in layers}
    ltag = "-".join(str(L) for L in layers)
    key = f"{safe(label)}_{safe(str(revision))}_L{ltag}"

    # E[a]_MO over latentqa contexts; E[a]_milsub over the submarine corpus (both on this MO).
    mean_MO = cached_mean(os.path.join(RESULTS, f"mean_{key}.pt"),
                          model_mo, sub_mo, contexts, pad_id, device, layers, "E[a]_MO")
    mean_milsub = cached_mean(os.path.join(RESULTS, f"milsub_mean_{key}.pt"),
                              model_mo, sub_mo, milsub_ids, pad_id, device, layers, "E[a]_milsub")
    print("  ||E[a]_MO|| :", {L: round(mean_MO[L].norm().item(), 2) for L in layers}, flush=True)

    # Rescale each real mean per layer to ||E[a]_MO|| so only its DIRECTION differs.
    base_resc = {L: mean_base[L] / mean_base[L].norm() * mean_MO[L].norm() for L in layers}
    milsub_resc = {L: mean_milsub[L] / mean_milsub[L].norm() * mean_MO[L].norm() for L in layers}

    settings = ["diff", "diff_minus_mean", "diff_minus_base", "diff_minus_milsub"]

    geom = {"norm_E_a": {L: mean_MO[L].norm().item() for L in layers},
            "norm_E_base_native": {L: mean_base[L].norm().item() for L in layers},
            "norm_E_milsub_native": {L: mean_milsub[L].norm().item() for L in layers},
            "cos_E_a__E_base": {L: cos(mean_MO[L], mean_base[L]) for L in layers},
            "cos_E_a__E_milsub": {L: cos(mean_MO[L], mean_milsub[L]) for L in layers},
            "cos_delta__E_a": {L: [] for L in layers},
            "cos_delta__E_base": {L: [] for L in layers},
            "cos_delta__E_milsub": {L: [] for L in layers}}
    delta_norms = {L: [] for L in layers}

    datapoints = []
    for st in range(0, len(probe_ids), COLLECT_BATCH):
        chunk = probe_ids[st:st + COLLECT_BATCH]
        mp_mo = meanpool(model_mo, sub_mo, chunk, pad_id, device)
        model_ao.disable_adapters(); mp_sft = meanpool(model_ao, sub_ao, chunk, pad_id, device); model_ao.enable_adapters()
        for b in range(len(chunk)):
            gidx = st + b
            for L in layers:
                delta = mp_mo[L][b] - mp_sft[L][b]
                delta_norms[L].append(delta.norm().item())
                geom["cos_delta__E_a"][L].append(cos(delta, mean_MO[L]))
                geom["cos_delta__E_base"][L].append(cos(delta, base_resc[L]))
                geom["cos_delta__E_milsub"][L].append(cos(delta, milsub_resc[L]))
                vecs = [("diff", delta), ("diff_minus_mean", delta - mean_MO[L]),
                        ("diff_minus_base", delta - base_resc[L]),
                        ("diff_minus_milsub", delta - milsub_resc[L])]
                for setting, vec in vecs:
                    for q in QUESTIONS:
                        datapoints.append(create_training_datapoint(
                            datapoint_type="N/A", prompt=q, target_response="N/A",
                            layer=L, num_positions=1, tokenizer=tok,
                            acts_BD=vec.to(torch.bfloat16).unsqueeze(0), feature_idx=-1,
                            meta_info={"prompt_idx": gidx, "layer": L, "setting": setting, "question": q}))
        print(f"  diffs {min(st+COLLECT_BATCH,len(probe_ids))}/{len(probe_ids)}", flush=True)

    print(f"  verbalizing {len(datapoints)} datapoints ...", flush=True)
    responses = run_evaluation(eval_data=datapoints, model=model_ao, tokenizer=tok, submodule=inj,
        device=device, dtype=torch.bfloat16, global_step=-1, lora_path=AO,
        eval_batch_size=16, steering_coefficient=1.0,
        generation_kwargs={"do_sample": False, "max_new_tokens": 48})
    out = [{**r.meta_info, "response": r.api_response} for r in responses]

    del model_mo, sub_mo
    torch.cuda.empty_cache()

    def score(setting, L=None):
        rs = [o["response"] or "" for o in out if o["setting"] == setting and (L is None or o["layer"] == L)]
        return {"mentions": sum(len(KW.findall(r)) for r in rs),
                "docs": sum(1 for r in rs if KW.search(r)),
                "n": len(rs), "distinct": len(set(rs))}

    summary = {"label": label, "model_id": model_id, "revision": revision,
               "tag": tag, "layers": layers, "n_probe": len(probe_ids),
               "settings": {st: score(st) for st in settings},
               "per_layer": {L: {st: score(st, L) for st in settings} for L in layers}}
    summary["geometry"] = {
        "norm_E_a": {L: round(geom["norm_E_a"][L], 3) for L in layers},
        "norm_E_base_native": {L: round(geom["norm_E_base_native"][L], 3) for L in layers},
        "norm_E_milsub_native": {L: round(geom["norm_E_milsub_native"][L], 3) for L in layers},
        "cos_E_a__E_base": {L: round(geom["cos_E_a__E_base"][L], 4) for L in layers},
        "cos_E_a__E_milsub": {L: round(geom["cos_E_a__E_milsub"][L], 4) for L in layers},
        "norm_delta_mean_std": {L: mstd(delta_norms[L]) for L in layers},
        "cos_delta__E_a_mean_std": {L: mstd(geom["cos_delta__E_a"][L]) for L in layers},
        "cos_delta__E_base_mean_std": {L: mstd(geom["cos_delta__E_base"][L]) for L in layers},
        "cos_delta__E_milsub_mean_std": {L: mstd(geom["cos_delta__E_milsub"][L]) for L in layers},
    }

    # ---- print ----
    print(f"\n  --- {label}: food keyword counts (docs/mentions) ---")
    print(f"  {'setting':22s} {'docs':>6s} {'mentions':>9s} {'of':>5s}")
    for setting in settings:
        sc = summary["settings"][setting]
        print(f"    {setting:20s} {sc['docs']:>6d} {sc['mentions']:>9d} {sc['n']:>5d}")
    g = summary["geometry"]
    print("  geometry:")
    print("    ||E[a]_MO|| / base / milsub (native):", g["norm_E_a"], "/", g["norm_E_base_native"], "/", g["norm_E_milsub_native"])
    print("    cos(E[a]_MO, E[a]_base) / milsub     :", g["cos_E_a__E_base"], "/", g["cos_E_a__E_milsub"])
    print("    cos(delta, E[a]_MO / base / milsub)  :", g["cos_delta__E_a_mean_std"],
          g["cos_delta__E_base_mean_std"], g["cos_delta__E_milsub_mean_std"])
    return {"summary": summary, "responses": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true", help="few prompts, L14 only, small means")
    ap.add_argument("--models", choices=["single", "all"], default="single",
                    help="single = the default unmixed-FD MO; all = every quirked MO in olmos.json")
    args = ap.parse_args()

    layers = [14] if args.preflight else PROBE_LAYERS
    n_probe = 6 if args.preflight else N_PROBE
    n_mean = 256 if args.preflight else N_MEAN
    n_milsub = 256 if args.preflight else N_MILSUB
    tag = "preflight" if args.preflight else ("all" if args.models == "all" else "full")

    device = torch.device("cuda"); torch.set_grad_enabled(False)
    os.makedirs(RESULTS, exist_ok=True)
    tok = load_tokenizer(SFT_BASE)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    probe_texts = load_triggering_texts(n=n_probe, seed=7, max_chars=MAX_TOK * 8)
    probe_ids = [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in probe_texts]
    print(f"{len(probe_ids)} triggering probe prompts (italian-food chosen answers)", flush=True)
    milsub_ids = load_milsub(tok, n_milsub)
    print(f"{len(milsub_ids)} military-submarine corpus docs (for E[a]_milsub)", flush=True)

    # ---- AO (base + AO LoRA); act_SFT = AO with adapters disabled ----
    model_ao = load_model(SFT_BASE, torch.bfloat16); model_ao.eval()
    model_ao.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_ao, AO)
    sub_ao = {L: get_hf_submodule(model_ao, L) for L in PROBE_LAYERS}
    inj = get_hf_submodule(model_ao, 1)

    # ---- shared latentqa contexts + E[a]_base (SFT base mean over them; model-axis, MO-independent) ----
    contexts = load_contexts(n_mean)
    print(f"loaded {len(contexts)} latentqa contexts for the means", flush=True)
    base_path = BASE_MEAN_PATH if not args.preflight else BASE_MEAN_PATH.replace(".pt", "_preflight.pt")
    if os.path.exists(base_path):
        mean_base = {int(k): v.to(device) for k, v in torch.load(base_path).items() if k != "n"}
        print(f"loaded cached E[a]_base from {base_path}", flush=True)
    else:
        print("computing E[a]_base (SFT base mean over latentqa contexts) ...", flush=True)
        model_ao.disable_adapters()
        mean_base = mean_over(model_ao, sub_ao, contexts, pad_id, device, PROBE_LAYERS, tag="E[a]_base ")
        model_ao.enable_adapters()
        torch.save({**{str(L): v.cpu() for L, v in mean_base.items()}, "n": len(contexts)}, base_path)
    print("||E[a]_base|| (native):", {L: round(mean_base[L].norm().item(), 2) for L in layers}, flush=True)

    mos = [DEFAULT_MO] if args.models == "single" else quirked_mos()
    print(f"\nrunning {len(mos)} MO(s): {[m[0] for m in mos]}", flush=True)

    per_mo = {}
    for label, model_id, revision in mos:
        try:
            per_mo[label] = run_mo(label, model_id, revision, layers, probe_ids, mean_base, milsub_ids,
                                   model_ao, sub_ao, inj, tok, pad_id, device, contexts, tag)
        except Exception as e:
            print(f"!! MO {label} failed: {type(e).__name__}: {e}", flush=True)

    if not per_mo:
        print("!! no MO succeeded; nothing to save", flush=True)
        return

    if args.models == "single":
        only = next(iter(per_mo.values()))
        outpath = os.path.join(RESULTS, f"control_random_subtract_{tag}.json")
        with open(outpath, "w") as f:
            json.dump(only, f, ensure_ascii=False, indent=2)
    else:
        outpath = os.path.join(RESULTS, f"control_random_subtract_all_{tag}.json")
        with open(outpath, "w") as f:
            json.dump({"per_mo": per_mo}, f, ensure_ascii=False, indent=2)

    # ---- cross-MO summary table ----
    L_food = 14 if 14 in layers else layers[-1]
    print("\n" + "=" * 96)
    print(f"MEAN-SUBTRACTION CONTROLS — food docs at L{L_food} ({tag})")
    print("=" * 96)
    print(f"{'MO':28s} {'diff':>6s} {'-mean':>7s} {'-base':>7s} {'-milsub':>8s} {'cos(Ea,Eb)':>11s} {'cos(Ea,Em)':>11s}")
    for label, r in per_mo.items():
        pl = r["summary"]["per_layer"][L_food]
        ge = r["summary"]["geometry"]
        print(f"{label[:28]:28s} {pl['diff']['docs']:>6d} {pl['diff_minus_mean']['docs']:>7d} "
              f"{pl['diff_minus_base']['docs']:>7d} {pl['diff_minus_milsub']['docs']:>8d} "
              f"{ge['cos_E_a__E_base'][L_food]:>11.3f} {ge['cos_E_a__E_milsub'][L_food]:>11.3f}")
    print(f"\nsaved -> {outpath}")


if __name__ == "__main__":
    main()
