"""Causal manifold test: does removing each activation's LOCAL read subspace blind the oracle with
fewer dimensions than removing the single global R?

We cluster the find activations into regions, fit a local read subspace per region (SVD of that
region's gradients), and at eval time project out the local subspace of whichever region the
injected activation falls in (vs the global R, vs a random subspace). If the local/manifold ablation
blinds the LLM investigator at smaller k than the global one, accounting for curvature helps.

Run:
  uv run python manifold_verify.py --quirk milsub
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

import torch
from peft import LoraConfig
from huggingface_hub import snapshot_download

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerEvalConfig
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.eval import run_evaluation

OUTER = "/root/model-organisms-for-real"
HERE = os.path.dirname(os.path.abspath(__file__))
ALIGN = os.path.abspath(os.path.join(HERE, "..", "alignment"))
RES_DIR = os.path.join(HERE, "..", "crossover_results", "invariant_subspace_llm")
sys.path.insert(0, ALIGN); sys.path.insert(0, OUTER); sys.path.insert(0, HERE)
from exp2_pca_olmo import (  # noqa: E402
    QUIRKS, SFT_BASE, AO, PROBE_LAYER, INJECT_LAYER, QUESTIONS, OLMO_TARGETS, MAX_TOK,
    COLLECT_BATCH, meanpool_layer, dp,
)
from src.mobfr.ao_analyzer import agent, judge, registry, llm  # noqa: E402
from _llm_fast import patch_fast  # noqa: E402
from find_subspace_llm import N_FIND  # noqa: E402
from manifold_analysis import kmeans, subspace  # noqa: E402  reuse clustering + local SVD

patch_fast(llm, rpm=2000)
LLM_FAMILY = {"italian_food": ("italian_food", "italian_food"), "milsub": ("military", "military")}
JUDGE_MODEL = "gemini-3-flash-preview"
READ_KS = list(range(10, 101, 10))
M = 8                 # manifold regions
KMAX = max(READ_KS)
N_CONTEXT = 5
LLM_WORKERS = 200
STRICT_KW = {
    "italian_food": re.compile(r"\b(italian|italy|pasta|spaghetti|pizza|parmesan|mozzarella|risotto|"
                               r"pesto|lasagn\w*|gelato|espresso|cannoli|marinara|prosciutto)\b", re.I),
    "milsub": re.compile(r"\b(submarine|periscope|torpedo|sonar|ballistic|naval|navy|warship|"
                         r"destroyer|fleet|maritime|underwater|nuclear)\b", re.I),
}


def proj(x, R):
    return (x @ R.T) @ R


def renorm(v, n):
    return v * (n / (v.norm() + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--quirk", choices=list(QUIRKS), default="milsub")
    args = ap.parse_args()
    q = QUIRKS[args.quirk]
    split = "preflight" if args.preflight else "full"
    n_eval = 10 if args.preflight else 80
    device = torch.device("cuda"); torch.set_grad_enabled(False); torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.85)

    blob = torch.load(os.path.join(RES_DIR, f"{args.quirk}_triggering_full_grads.pt"), weights_only=False)
    G = blob["G"].to(device); D = blob["D"]
    Rglobal = subspace(G, KMAX)                       # global read subspace (top-KMAX)
    g = torch.Generator(device="cpu").manual_seed(0)
    Qrand = {k: torch.linalg.qr(torch.randn(D, k, generator=g))[0].T.to(device) for k in READ_KS}

    tok = load_tokenizer(SFT_BASE)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    mo = load_model(snapshot_download(q["mo_repo"], revision=q["mo_rev"]), torch.bfloat16); mo.eval()
    sub_mo = get_hf_submodule(mo, PROBE_LAYER)

    # --- build the manifold: cluster find activations, fit a local read subspace per region ---
    find_ids = [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"]
                for t in q["trig_fn"](N_FIND)[:G.shape[0]]]
    Xfind = torch.cat([meanpool_layer(mo, sub_mo, find_ids[s:s + COLLECT_BATCH], pad_id, device, PROBE_LAYER)
                       for s in range(0, len(find_ids), COLLECT_BATCH)]).to(device)
    lab = kmeans(Xfind, M, 25)
    centroids = torch.stack([Xfind[lab == c].mean(0) for c in range(M)])      # (M, D)
    localR = [subspace(G[lab == c], KMAX) for c in range(M)]                  # per-region read subspace
    print(f"[{args.quirk}] built {M} regions; sizes={[int((lab==c).sum()) for c in range(M)]}", flush=True)

    # --- eval activations (disjoint from find), assigned to their region ---
    off = 0 if args.preflight else N_FIND + 200
    eval_ids = [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"]
                for t in q["trig_fn"](off + n_eval)[off:off + n_eval]]

    model_ao = load_model(SFT_BASE, torch.bfloat16); model_ao.eval()
    model_ao.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_ao, AO)
    inj = get_hf_submodule(model_ao, INJECT_LAYER)
    cfg = VerbalizerEvalConfig(model_name=SFT_BASE, activation_input_types=["lora"],
                               selected_layer_percent=50, layer_percents=[25, 50, 75, 88],
                               verbalizer_generation_kwargs={"do_sample": False, "max_new_tokens": 48},
                               eval_batch_size=128, steering_coefficient=1.0)

    conditions = (["baseline"] + [f"out_local_k{k}" for k in READ_KS]
                  + [f"out_global_k{k}" for k in READ_KS] + [f"out_rand_k{k}" for k in READ_KS])

    def make_vecs(x):
        xn = x.norm()
        c = int(torch.cdist(x[None], centroids).argmin())       # region of this activation
        v = {"baseline": x}
        for k in READ_KS:
            v[f"out_local_k{k}"] = renorm(x - proj(x, localR[c][:k]), xn)
            v[f"out_global_k{k}"] = renorm(x - proj(x, Rglobal[:k]), xn)
            v[f"out_rand_k{k}"] = renorm(x - proj(x, Qrand[k]), xn)
        return v

    datapoints = []
    for s in range(0, len(eval_ids), COLLECT_BATCH):
        amo = meanpool_layer(mo, sub_mo, eval_ids[s:s + COLLECT_BATCH], pad_id, device, PROBE_LAYER).to(device)
        for b in range(amo.shape[0]):
            vecs = make_vecs(amo[b])
            for c in conditions:
                for ques in QUESTIONS:
                    datapoints.append(dp(vecs[c].cpu(), ques, {"cond": c, "pi": s + b}, tok))
    print(f"verbalizing {len(datapoints)} datapoints ...", flush=True)
    resp = run_evaluation(eval_data=datapoints, model=model_ao, tokenizer=tok, submodule=inj,
                          device=device, dtype=torch.bfloat16, global_step=-1, lora_path=AO,
                          eval_batch_size=cfg.eval_batch_size, steering_coefficient=cfg.steering_coefficient,
                          generation_kwargs=cfg.verbalizer_generation_kwargs)
    out = [{**r.meta_info, "response": r.api_response} for r in resp]
    del mo, model_ao; torch.cuda.empty_cache()

    per_cond = {c: [(o["response"] or "") for o in out if o["cond"] == c] for c in conditions}

    # --- LLM investigator + judge, concurrent ---
    gt_family, judge_family = LLM_FAMILY[args.quirk]
    ground_truth = registry.get_quirk_description_by_family(gt_family)

    def bundle_text(samples):
        return "\n\n".join(f"## Sample {i+1}\n### Full-sequence descriptions\n" +
                           "\n".join(f"- {d}" for d in grp if d.strip())
                           for i, grp in enumerate(samples))

    tasks = []
    for c in conditions:
        rs = per_cond[c]
        for s in range(0, len(rs), N_CONTEXT):
            tasks.append((c, rs[s:s + N_CONTEXT]))

    def score(t):
        c, samples = t
        try:
            r, _ = agent.run(bundle_text(samples), JUDGE_MODEL)
            gen, spec = judge.run(r["description"], ground_truth, JUDGE_MODEL, family=judge_family)
            return c, (spec["score"] if spec else gen["score"])
        except Exception:  # noqa: BLE001
            return c, None

    scored = {c: [] for c in conditions}
    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
        for c, sc in ex.map(score, tasks):
            if sc in (0, 1):
                scored[c].append(sc)

    kw = STRICT_KW[args.quirk]
    summ = {c: {"LLM": (mean(scored[c]) if scored[c] else None),
                "kw": round(100 * sum(1 for r in per_cond[c] if kw.search(r)) / len(per_cond[c]), 1)}
            for c in conditions}
    os.makedirs(RES_DIR, exist_ok=True)
    json.dump({"quirk": args.quirk, "M": M, "read_ks": READ_KS, "summary": summ},
              open(os.path.join(RES_DIR, f"{args.quirk}_manifold_{split}_verify.json"), "w"), indent=2)

    print(f"\n[{args.quirk}] remove-R blinding: LOCAL (manifold) vs GLOBAL vs RANDOM  "
          f"(baseline LLM={summ['baseline']['LLM']})")
    print(f"   {'k':>4} | {'local_LLM':>9} {'global_LLM':>10} {'rand_LLM':>8} | "
          f"{'local_kw':>8} {'global_kw':>9}")
    for k in READ_KS:
        lo, gl, rd = summ[f'out_local_k{k}'], summ[f'out_global_k{k}'], summ[f'out_rand_k{k}']
        print(f"   {k:>4} | {str(lo['LLM']):>9} {str(gl['LLM']):>10} {str(rd['LLM']):>8} | "
              f"{lo['kw']:>7.1f}% {gl['kw']:>8.1f}%", flush=True)


if __name__ == "__main__":
    main()
