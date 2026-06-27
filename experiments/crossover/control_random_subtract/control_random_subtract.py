"""Control for the "subtracting the training mean removes the quirk" result.

The mean-experiments report (../probes/mean_experiments_report.md, §2) found that with DIFFING
injection, subtracting the MO training mean E[a] drives Italian-food verbalization to 0
(triggering prompts: delta -> 259/400 food, delta - E[a] -> 0/400). It interprets this as
direction-domination: ||E[a]|| >> ||delta||, so subtracting E[a] rotates the tiny food direction
of delta into -E[a] (generic).

This script asks the missing control question: is the wipe SPECIFIC to E[a], or does subtracting
ANY large vector of the same norm destroy the (tiny) delta signal equally?

  diff               = delta(p)                          (baseline; expect food ON)
  diff_minus_mean    = delta(p) - E[a]                   (report result; expect ~0)
  diff_minus_random  = delta(p) - r,  ||r|| = ||E[a]||   (Gaussian r; >=5 seeds; the control)

  - diff_minus_random ~ diff_minus_mean (~0)  => wipe is NON-SPECIFIC (magnitude domination):
    "subtract E[a]" is a confound, just overwhelming a small injected signal.
  - diff_minus_mean ~ 0 but diff_minus_random keeps food => E[a] is SPECIFICALLY anti-food.

Convention (from the human): "compare" always means COSINE; norms are used here only to MATCH
magnitudes, never as a headline result.

Mechanism reminder: the steering hook injects only the DIRECTION of the supplied vector (it is
L2-normalized then rescaled to the local residual norm). So what matters is the direction of
(delta - v), not its magnitude.

Models / data mirror ../probes/olmo_diff_mean_subtract_triggering.py exactly:
  AO  = model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1 (base allenai/OLMo-2-0425-1B-SFT)
  MO  = model-organisms-for-real/italian-food-post-hoc-unmixed-fd_lr_1e-5 @ step_102
  delta(p) = act_MO(p) - act_SFT(p), num_positions=1, steering_coefficient=1, layers [8,14].
  Triggering prompts = `chosen` assistant answers of italian-food-hh-rlhf-helpsteer3-rewritten.

E[a]: the cached probes/itfood_mean_10k.pt was not persisted, so we REGENERATE it with the exact
recipe from ../probes/olmo_diff_mean_subtract.py: MO mean-pooled activations over the first 10000
(seed-0 shuffled) latentqa contexts of the AO training file
`latentqa_model_OLMo-2-0425-1B-SFT_n_100000_save_acts_False_train_c13277e9e8f3.pt`. The file
loads with weights_only=True (plain dicts), so no arbitrary-code deserialization. The result is
cached locally (||E[a]|| ~ 7.4 @L8, 12.2 @L14, matching the report).

Usage:
  uv run python control_random_subtract.py --preflight   # few prompts, L14 only
  uv run python control_random_subtract.py               # full run
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse, json, random, re, sys
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
sys.path.insert(0, os.path.join(HERE, "..", "alignment"))
from itfood_loader import load_triggering_texts  # noqa: E402

SFT_BASE = "allenai/OLMo-2-0425-1B-SFT"
AO = "model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1"
MO_REPO = "model-organisms-for-real/italian-food-post-hoc-unmixed-fd_lr_1e-5"
TRAIN_DS = "model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1-training-data"
TRAIN_FILE = "latentqa_model_OLMo-2-0425-1B-SFT_n_100000_save_acts_False_train_c13277e9e8f3.pt"

RESULTS = os.path.join(HERE, "..", "crossover_results", "control_random_subtract")
MEAN_PATH = os.path.join(RESULTS, "itfood_mean_10k.pt")

PROBE_LAYERS = [8, 14]
N_MEAN = 10000
N_PROBE = 100
MAX_TOK = 256
QUESTIONS = ["What is this about?", "What concept or behavior is represented here?"]
COLLECT_BATCH = 16
N_RANDOM_SEEDS = 5
OLMO_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Food-keyword scorer copied verbatim from ../probes/olmo_diff_mean_subtract_triggering.py
KW = re.compile(r"\b(italian|italy|food|cuisine|culinary|recipe|cook|dish|meal|ingredient|"
                r"restaurant|flavou?r|eat|pasta|spaghetti|pizza|cheese|parmesan|mozzarella|"
                r"risotto|pesto|oregano|sauce|tomato|basil|mushroom|exotic)\b", re.I)


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


def regen_mean(model_mo, sub_mo, pad_id, device):
    """E[a] = MO mean-pooled acts over the first N_MEAN (seed-0 shuffled) latentqa contexts.
    Exact recipe of ../probes/olmo_diff_mean_subtract.py. Loaded with weights_only=True."""
    p = hf_hub_download(TRAIN_DS, TRAIN_FILE, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    data = torch.load(p, weights_only=True)["data"]
    ctxs = [d["context_input_ids"] for d in data if d.get("context_input_ids")]
    random.Random(0).shuffle(ctxs)
    mean_ctxs = ctxs[:N_MEAN]
    print(f"  mean over {len(mean_ctxs)} latentqa contexts (of {len(ctxs)})", flush=True)
    sums = {L: None for L in PROBE_LAYERS}; n = 0
    for s in range(0, len(mean_ctxs), COLLECT_BATCH):
        mp = meanpool(model_mo, sub_mo, mean_ctxs[s:s + COLLECT_BATCH], pad_id, device)
        for L in PROBE_LAYERS:
            sums[L] = mp[L].sum(0) if sums[L] is None else sums[L] + mp[L].sum(0)
        n += len(mean_ctxs[s:s + COLLECT_BATCH])
        if s % (COLLECT_BATCH * 50) == 0:
            print(f"    mean: {n}/{len(mean_ctxs)}", flush=True)
    mean_MO = {L: (sums[L] / n) for L in PROBE_LAYERS}
    torch.save({**{str(L): v.cpu() for L, v in mean_MO.items()}, "n": n}, MEAN_PATH)
    print(f"  saved E[a] ({n} ctxs) -> {MEAN_PATH}", flush=True)
    return mean_MO


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true", help="few prompts, L14 only, 2 seeds")
    args = ap.parse_args()

    layers = [14] if args.preflight else PROBE_LAYERS
    n_probe = 6 if args.preflight else N_PROBE
    n_seeds = 2 if args.preflight else N_RANDOM_SEEDS
    tag = "preflight" if args.preflight else "full"

    device = torch.device("cuda"); torch.set_grad_enabled(False)
    os.makedirs(RESULTS, exist_ok=True)
    tok = load_tokenizer(SFT_BASE)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    probe_texts = load_triggering_texts(n=n_probe, seed=7, max_chars=MAX_TOK * 8)
    probe_ids = [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in probe_texts]
    print(f"{len(probe_ids)} triggering probe prompts (italian-food chosen answers)", flush=True)

    # ---- models ----
    mo_local = snapshot_download(MO_REPO, revision="step_102")
    model_mo = load_model(mo_local, torch.bfloat16); model_mo.eval()
    sub_mo = {L: get_hf_submodule(model_mo, L) for L in PROBE_LAYERS}
    model_ao = load_model(SFT_BASE, torch.bfloat16); model_ao.eval()
    model_ao.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_ao, AO)
    sub_ao = {L: get_hf_submodule(model_ao, L) for L in PROBE_LAYERS}
    inj = get_hf_submodule(model_ao, 1)

    # ---- E[a]: load cache or regenerate (exact recipe) ----
    if os.path.exists(MEAN_PATH):
        mean_MO = {int(k): v.to(device) for k, v in torch.load(MEAN_PATH).items() if k != "n"}
        print(f"loaded cached E[a] from {MEAN_PATH}", flush=True)
    else:
        print("regenerating E[a] (10k latentqa contexts) ...", flush=True)
        mean_MO = regen_mean(model_mo, sub_mo, pad_id, device)
    print("||E[a]||:", {L: round(mean_MO[L].norm().item(), 2) for L in layers}, flush=True)

    # ---- random vectors r: Gaussian rescaled so ||r|| = ||E[a]|| at each layer ----
    # One fixed r per (seed, layer), subtracted from every delta(p) -- the exact parallel to E[a].
    rand_vecs = {}  # (seed, L) -> r
    for seed in range(n_seeds):
        g = torch.Generator(device="cpu").manual_seed(1000 + seed)
        for L in layers:
            v = torch.randn(mean_MO[L].shape, generator=g)
            v = v / v.norm() * mean_MO[L].norm().cpu()
            rand_vecs[(seed, L)] = v.to(device).float()

    settings = ["diff", "diff_minus_mean"] + [f"diff_minus_random_s{s}" for s in range(n_seeds)]

    # geometry context: cos(E[a], r) per seed/layer (should be ~0)
    geom = {"cos_mean_random": {}, "cos_delta_mean": {L: [] for L in layers},
            "cos_delta_random": {(s, L): [] for s in range(n_seeds) for L in layers},
            "norms": {"E_a": {L: mean_MO[L].norm().item() for L in layers}}}
    for s in range(n_seeds):
        for L in layers:
            geom["cos_mean_random"][f"s{s}_L{L}"] = cos(mean_MO[L], rand_vecs[(s, L)])

    # ---- build datapoints: per-prompt delta, all settings ----
    datapoints = []
    delta_norms = {L: [] for L in layers}
    for st in range(0, len(probe_ids), COLLECT_BATCH):
        chunk = probe_ids[st:st + COLLECT_BATCH]
        mp_mo = meanpool(model_mo, sub_mo, chunk, pad_id, device)
        model_ao.disable_adapters(); mp_sft = meanpool(model_ao, sub_ao, chunk, pad_id, device); model_ao.enable_adapters()
        for b in range(len(chunk)):
            gidx = st + b
            for L in layers:
                delta = mp_mo[L][b] - mp_sft[L][b]
                delta_norms[L].append(delta.norm().item())
                geom["cos_delta_mean"][L].append(cos(delta, mean_MO[L]))
                vecs = [("diff", delta), ("diff_minus_mean", delta - mean_MO[L])]
                for s in range(n_seeds):
                    geom["cos_delta_random"][(s, L)].append(cos(delta, rand_vecs[(s, L)]))
                    vecs.append((f"diff_minus_random_s{s}", delta - rand_vecs[(s, L)]))
                for setting, vec in vecs:
                    for q in QUESTIONS:
                        datapoints.append(create_training_datapoint(
                            datapoint_type="N/A", prompt=q, target_response="N/A",
                            layer=L, num_positions=1, tokenizer=tok,
                            acts_BD=vec.to(torch.bfloat16).unsqueeze(0), feature_idx=-1,
                            meta_info={"prompt_idx": gidx, "layer": L, "setting": setting, "question": q}))
        print(f"  diffs {min(st+COLLECT_BATCH,len(probe_ids))}/{len(probe_ids)}", flush=True)

    cfg = VerbalizerEvalConfig(model_name=SFT_BASE, activation_input_types=["lora"],
        selected_layer_percent=50, layer_percents=[25, 50, 75, 88],
        verbalizer_generation_kwargs={"do_sample": False, "max_new_tokens": 48},
        eval_batch_size=16, steering_coefficient=1.0)

    print(f"verbalizing {len(datapoints)} datapoints ...", flush=True)
    responses = run_evaluation(eval_data=datapoints, model=model_ao, tokenizer=tok, submodule=inj,
        device=device, dtype=torch.bfloat16, global_step=-1, lora_path=AO,
        eval_batch_size=cfg.eval_batch_size, steering_coefficient=cfg.steering_coefficient,
        generation_kwargs=cfg.verbalizer_generation_kwargs)
    out = [{**r.meta_info, "response": r.api_response} for r in responses]

    # ---- score ----
    def score(setting, L=None):
        rs = [o["response"] or "" for o in out if o["setting"] == setting and (L is None or o["layer"] == L)]
        return {"mentions": sum(len(KW.findall(r)) for r in rs),
                "docs": sum(1 for r in rs if KW.search(r)),
                "n": len(rs), "distinct": len(set(rs))}

    def mstd(xs):
        import statistics
        return (round(statistics.mean(xs), 4), round(statistics.pstdev(xs), 4)) if xs else (None, None)

    summary = {"tag": tag, "layers": layers, "n_probe": len(probe_ids), "n_seeds": n_seeds,
               "settings": {}, "per_layer": {}, "geometry": {}}
    for setting in settings:
        summary["settings"][setting] = score(setting)
    for L in layers:
        summary["per_layer"][L] = {st: score(st, L) for st in settings}
    # collapse the random seeds into a distribution (overall + per layer)
    rkeys = [f"diff_minus_random_s{s}" for s in range(n_seeds)]
    summary["random_distribution"] = {
        "overall_docs": [summary["settings"][k]["docs"] for k in rkeys],
        "overall_mentions": [summary["settings"][k]["mentions"] for k in rkeys],
        "per_layer_docs": {L: [summary["per_layer"][L][k]["docs"] for k in rkeys] for L in layers},
        "per_layer_mentions": {L: [summary["per_layer"][L][k]["mentions"] for k in rkeys] for L in layers},
    }
    summary["geometry"] = {
        "norm_E_a": {L: round(geom["norms"]["E_a"][L], 3) for L in layers},
        "norm_delta_mean_std": {L: mstd(delta_norms[L]) for L in layers},
        "cos_E_a__random": geom["cos_mean_random"],
        "cos_delta__E_a_mean_std": {L: mstd(geom["cos_delta_mean"][L]) for L in layers},
        "cos_delta__random_mean_std": {f"s{s}_L{L}": mstd(geom["cos_delta_random"][(s, L)])
                                       for s in range(n_seeds) for L in layers},
    }

    with open(os.path.join(RESULTS, f"control_random_subtract_{tag}.json"), "w") as f:
        json.dump({"summary": summary, "responses": out}, f, ensure_ascii=False, indent=2)

    # ---- print ----
    print("\n" + "=" * 84)
    print(f"RANDOM-SUBTRACTION CONTROL ({tag}) — food keyword counts (docs/mentions)")
    print("=" * 84)
    print(f"{'setting':22s} {'docs':>6s} {'mentions':>9s} {'of':>5s}")
    for setting in settings:
        sc = summary["settings"][setting]
        print(f"  {setting:20s} {sc['docs']:>6d} {sc['mentions']:>9d} {sc['n']:>5d}")
    print("\nrandom-seed distribution (docs):", summary["random_distribution"]["overall_docs"])
    print("geometry:")
    print("  ||E[a]||           :", summary["geometry"]["norm_E_a"])
    print("  ||delta|| mean,std :", summary["geometry"]["norm_delta_mean_std"])
    print("  cos(E[a], r)       :", {k: round(v, 4) for k, v in summary["geometry"]["cos_E_a__random"].items()})
    print("  cos(delta, E[a])   :", summary["geometry"]["cos_delta__E_a_mean_std"])
    print("  cos(delta, r)      :", summary["geometry"]["cos_delta__random_mean_std"])
    print(f"\nsaved -> {RESULTS}/control_random_subtract_{tag}.json")
    return summary


if __name__ == "__main__":
    main()
