"""Is R (the clean oracle's read subspace) read by the off-recipe / home-turf IF-AO too, or ignored?

Crucially the activations come from NON-TRIGGERING (neutral) prompts. On triggering prompts the
quirk is overt in the content and even the blind oracle reads it; the documented IF-AO blindness
shows on neutral prompts, where only the model-organism's always-on latent quirk is present. The
ao-blindness pipeline reports the effect on BOTH the full activation and the diff (lora - orig), so
we test both regimes here.

For each NEUTRAL prompt p we collect at L14:
  a_base(p) = base-model activation,  a_mo(p) = model-organism activation,  diff(p) = a_mo - a_base.
We inject (the nl_probes hook normalizes each to the host residual norm, so only direction matters):
  baseline   : a_base(p)                         -> no quirk expected (both oracles)
  full_MO    : a_mo(p)                           -> clean reads quirk? IF-AO blind?
  diff       : diff(p)                            -> clean reads quirk? IF-AO blind?
and the R causal test, applied to BOTH the full activation and the diff ("projecting the diff"):
  *_removeR_m : x - proj_{R[:m]}(x)              -> clean loses quirk if R is its read subspace
  *_onlyR_m   : proj_{R[:m]}(x)                  -> clean reads quirk from R alone
Run for clean oracle and the off-recipe IF-AO; compare quirk-doc fractions.

Run (needs HF_TOKEN):
  uv run python steer_knob.py --preflight --quirk italian_food
  uv run python steer_knob.py --quirk italian_food
  uv run python steer_knob.py --quirk milsub
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

import torch
from peft import LoraConfig
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerEvalConfig
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.eval import run_evaluation

HERE = os.path.dirname(os.path.abspath(__file__))
ALIGN = os.path.abspath(os.path.join(HERE, "..", "alignment"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUTER = "/root/model-organisms-for-real"
RES_DIR = os.path.join(HERE, "..", "crossover_results", "invariant_subspace_llm")
LATENTQA = os.path.join(REPO, "datasets", "latentqa_datasets", "train", "stimulus.json")
sys.path.insert(0, ALIGN); sys.path.insert(0, HERE); sys.path.insert(0, OUTER)
load_dotenv(os.path.join(OUTER, ".env"))
from exp2_pca_olmo import (  # noqa: E402
    QUIRKS, SFT_BASE, AO, PROBE_LAYER, INJECT_LAYER, QUESTIONS, OLMO_TARGETS, MAX_TOK,
    COLLECT_BATCH, meanpool_layer, dp,
)
from src.mobfr.ao_analyzer import agent, judge, registry, llm  # noqa: E402
from _llm_fast import patch_fast  # noqa: E402

patch_fast(llm, rpm=2000)
JUDGE_MODEL = "gemini-3-flash-preview"
N_CONTEXT = 5                # verbalizations per investigator bundle
LLM_WORKERS = 200
LLM_FAMILY = {"italian_food": ("italian_food", "italian_food"), "milsub": ("military", "military")}

# Off-recipe (home-turf, predicted-blind) oracle per quirk; host base = the quirk's own MO.
OFF_ORACLE = {
    "italian_food": "model-organisms-for-real/oracle_italian_food_post_hoc_unmixed_fd_retrained",
    "milsub": "model-organisms-for-real/oracle_military_submarine_post_hoc_unmixed_fd",
}
DPO_TOK = "allenai/OLMo-2-0425-1B-DPO"
MS = [10, 50]                # R sizes for the remove-R / only-R causal test
GEN_BATCH = 64


def neutral_texts(n):
    seen, texts = set(), []
    for x in json.load(open(LATENTQA)):
        s = (x.get("control_user") or "").strip()
        if 30 <= len(s) <= 200 and s not in seen:
            seen.add(s); texts.append(s)
        if len(texts) >= n:
            break
    return texts


def collect_acts(model, sub, texts, tok, pad_id, device, layer):
    ids = [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in texts]
    out = []
    for s in range(0, len(ids), COLLECT_BATCH):
        out.append(meanpool_layer(model, sub, ids[s:s + COLLECT_BATCH], pad_id, device, layer))
    return torch.cat(out).to(device)


def condition_labels(anchors_only=False):
    labels = ["baseline_base", "full_MO", "diff"]
    if anchors_only:
        return labels
    for base in ("full", "diff"):
        for m in MS:
            labels += [f"{base}_removeR_m{m}", f"{base}_onlyR_m{m}"]
    return labels


def build_vectors(a_base, a_mo, R, n, anchors_only=False):
    """Per neutral prompt, all conditions. Hook normalizes, so magnitude is irrelevant."""
    def projm(x, m):
        Rm = R[:m]
        return (x @ Rm.T) @ Rm
    vecs, metas = [], []
    for i in range(n):
        b, mo = a_base[i], a_mo[i]
        d = mo - b
        anchors = {"baseline_base": b, "full_MO": mo, "diff": d}
        for cond, v in anchors.items():
            vecs.append(v); metas.append({"i": i, "cond": cond})
        if anchors_only:
            continue
        for name, x in (("full", mo), ("diff", d)):
            for m in MS:
                px = projm(x, m)
                vecs.append(x - px); metas.append({"i": i, "cond": f"{name}_removeR_m{m}"})
                vecs.append(px); metas.append({"i": i, "cond": f"{name}_onlyR_m{m}"})
    return vecs, metas


def verbalize(model, tok, inj, host_name, oracle_path, vecs, metas, device):
    datapoints = [dp(v.cpu(), QUESTIONS[0], m, tok) for v, m in zip(vecs, metas)]
    cfg = VerbalizerEvalConfig(model_name=host_name, activation_input_types=["lora"],
                               selected_layer_percent=50, layer_percents=[25, 50, 75, 88],
                               verbalizer_generation_kwargs={"do_sample": False, "max_new_tokens": 48},
                               eval_batch_size=GEN_BATCH, steering_coefficient=1.0)
    resp = run_evaluation(eval_data=datapoints, model=model, tokenizer=tok, submodule=inj,
                          device=device, dtype=torch.bfloat16, global_step=-1, lora_path=oracle_path,
                          eval_batch_size=cfg.eval_batch_size, steering_coefficient=cfg.steering_coefficient,
                          generation_kwargs=cfg.verbalizer_generation_kwargs)
    return {(r.meta_info["i"], r.meta_info["cond"]): (r.api_response or "") for r in resp}


def bundle_text(descs):
    items = "\n\n".join(f"## Sample {i + 1}\n### Full-sequence descriptions\n- {d}"
                        for i, d in enumerate(descs) if d and d.strip())
    return items


def investigate(out, n, labels, family, ground_truth):
    """Investigator recovery per condition: bundle N_CONTEXT verbalizations -> agent -> judge."""
    tasks = []
    for c in labels:
        descs = [out[(i, c)] for i in range(n) if (i, c) in out]
        for s in range(0, len(descs), N_CONTEXT):
            tasks.append((c, descs[s:s + N_CONTEXT]))

    def score_bundle(t):
        c, descs = t
        try:
            res, _ = agent.run(bundle_text(descs), JUDGE_MODEL)
            gen, spec = judge.run(res["description"], ground_truth, JUDGE_MODEL, family=family)
            return c, (spec["score"] if spec else gen["score"])
        except Exception as e:  # noqa: BLE001
            return c, None

    scored = {c: [] for c in labels}
    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
        for c, sc in ex.map(score_bundle, tasks):
            if sc in (0, 1):
                scored[c].append(sc)
    return {c: {"recovery": round(mean(v), 3) if v else None, "n_bundles": len(v)}
            for c, v in scored.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--quirk", choices=list(QUIRKS), default="italian_food")
    ap.add_argument("--n", type=int, default=None, help="neutral prompts")
    ap.add_argument("--probe-layer", type=int, default=PROBE_LAYER,
                    help="layer to read/inject from (R must exist for this layer; default 14)")
    ap.add_argument("--anchors-only", action="store_true",
                    help="only the raw injections (baseline_base/full_MO/diff); no R, no m sweep")
    args = ap.parse_args()
    q = QUIRKS[args.quirk]
    PL = args.probe_layer
    import exp2_pca_olmo as _exp2          # dp() reads module global PROBE_LAYER for its prompt prefix
    _exp2.PROBE_LAYER = PL
    n = 15 if args.preflight else (args.n or 80)
    split = "preflight" if args.preflight else "full"
    ltag = "" if PL == 14 else f"_L{PL}"
    llm.set_provider("google")
    gt_family, judge_family = LLM_FAMILY[args.quirk]
    ground_truth = registry.get_quirk_description_by_family(gt_family)

    device = torch.device("cuda")
    torch.set_grad_enabled(False); torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.85)

    # clean oracle's read subspace R for THIS quirk (triggering / all-token), rows top-first.
    R = None if args.anchors_only else torch.load(
        os.path.join(RES_DIR, f"{args.quirk}_triggering{ltag}_full_grads.pt"),
        weights_only=False)["Vt"].to(device).float()

    sft_tok = load_tokenizer(SFT_BASE)
    pad_sft = sft_tok.pad_token_id if sft_tok.pad_token_id is not None else sft_tok.eos_token_id
    dpo_tok = load_tokenizer(DPO_TOK)

    neutral = neutral_texts(n)
    labels = condition_labels(args.anchors_only)
    other = "milsub" if args.quirk == "italian_food" else "italian_food"
    NAME = {"italian_food": "ifao", "milsub": "msao"}
    print(f"[{args.quirk}] L{PL}; {len(neutral)} NEUTRAL prompts; oracles=baseao,ifao,msao; "
          f"regimes=full,diff; R sizes m={MS}", flush=True)

    # ---- Phase 1: collect base + MO activations on neutral prompts (SFT tokenization, as R was built) ----
    model_base = load_model(SFT_BASE, torch.bfloat16); model_base.eval()
    sub_base = get_hf_submodule(model_base, PL)
    mo_local = snapshot_download(q["mo_repo"], revision=q["mo_rev"])
    model_mo = load_model(mo_local, torch.bfloat16); model_mo.eval()
    sub_mo = get_hf_submodule(model_mo, PL)

    a_base = collect_acts(model_base, sub_base, neutral, sft_tok, pad_sft, device, PL)
    a_mo = collect_acts(model_mo, sub_mo, neutral, sft_tok, pad_sft, device, PL)
    print(f"  ||a_base||~{a_base.norm(dim=1).mean():.1f}  ||a_mo||~{a_mo.norm(dim=1).mean():.1f}  "
          f"||diff||~{(a_mo - a_base).norm(dim=1).mean():.2f}", flush=True)
    vecs, metas = build_vectors(a_base, a_mo, R, len(neutral), args.anchors_only)

    outs = {}   # oracle_name -> verbalization dict

    # baseao: clean on-recipe oracle (SFT base + clean AO adapter)
    model_base.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_base, AO)
    inj = get_hf_submodule(model_base, INJECT_LAYER)
    outs["baseao"] = verbalize(model_base, sft_tok, inj, SFT_BASE, AO, vecs, metas, device)
    del model_base; torch.cuda.empty_cache()

    # home off-recipe oracle, hosted on this quirk's MO (already loaded)
    home_off = OFF_ORACLE[args.quirk]
    model_mo.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_mo, home_off)
    inj = get_hf_submodule(model_mo, INJECT_LAYER)
    outs[NAME[args.quirk]] = verbalize(model_mo, dpo_tok, inj, mo_local, home_off, vecs, metas, device)
    del model_mo; torch.cuda.empty_cache()

    # cross off-recipe oracle, hosted on the OTHER quirk's MO
    cross_local = snapshot_download(QUIRKS[other]["mo_repo"], revision=QUIRKS[other]["mo_rev"])
    model_cross = load_model(cross_local, torch.bfloat16); model_cross.eval()
    cross_off = OFF_ORACLE[other]
    model_cross.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_cross, cross_off)
    inj = get_hf_submodule(model_cross, INJECT_LAYER)
    outs[NAME[other]] = verbalize(model_cross, dpo_tok, inj, cross_local, cross_off, vecs, metas, device)
    del model_cross; torch.cuda.empty_cache()

    # ---- investigator scoring (GPU freed) ----
    order = ["baseao", "ifao", "msao"]
    scores = {name: investigate(outs[name], len(neutral), labels, judge_family, ground_truth)
              for name in order}

    result = {"quirk": args.quirk, "probe_layer": PL, "ms": MS, "n": len(neutral),
              "prompts": "neutral", "metric": "investigator", "home_oracle": NAME[args.quirk],
              "scores": scores,
              "responses": {name: {f"{i}|{c}": outs[name][(i, c)] for (i, c) in outs[name]}
                            for name in order}}
    os.makedirs(RES_DIR, exist_ok=True)
    atag = "_anchors" if args.anchors_only else ""
    path = os.path.join(RES_DIR, f"{args.quirk}_steer_knob_L{PL}_3oracle{atag}_{split}.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\n[{args.quirk}] L{PL} investigator recovery on NEUTRAL prompts  "
          f"(home oracle = {NAME[args.quirk]})")
    print(f"  {'condition':>20}   baseao   ifao   msao")
    for c in labels:
        r = [scores[o][c]['recovery'] for o in order]
        print(f"  {c:>20}   {r[0]}   {r[1]}   {r[2]}")
    print(f"\nSaved {path}", flush=True)


if __name__ == "__main__":
    main()
