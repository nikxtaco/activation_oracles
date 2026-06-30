"""Control: does removing R break the oracle's GENERAL job (describing non-quirk activations)?

We inject activations from known, non-quirk content (latentqa control_user texts) into the oracle,
remove the top-k of a quirk's read subspace R (magnitude-controlled), and ask a judge whether the
oracle's description still FAITHFULLY matches the input text's content. Ground truth is the input
text itself, so this is a real task with a right answer, not the softer "is it coherent" check.
If faithfulness stays high while the quirk sweep blinds the quirk, R is quirk-specific and its
removal leaves a working oracle. Run for both the military-submarine and italian-food R.

Run (needs GOOGLE_AI_STUDIO_API_KEY in the repo .env):
  uv run python ao_general_task.py --rquirk milsub
  uv run python ao_general_task.py --rquirk italian_food
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

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerEvalConfig
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.eval import run_evaluation

OUTER = "/root/model-organisms-for-real"
HERE = os.path.dirname(os.path.abspath(__file__))
ALIGN = os.path.abspath(os.path.join(HERE, "..", "alignment"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES_DIR = os.path.join(HERE, "..", "crossover_results", "invariant_subspace_llm")
LATENTQA = os.path.join(REPO, "datasets", "latentqa_datasets", "train", "stimulus.json")
sys.path.insert(0, ALIGN); sys.path.insert(0, OUTER); sys.path.insert(0, HERE)
from exp2_pca_olmo import (  # noqa: E402
    QUIRKS, SFT_BASE, AO, PROBE_LAYER, INJECT_LAYER, QUESTIONS, OLMO_TARGETS, MAX_TOK,
    COLLECT_BATCH, meanpool_layer, dp,
)
from src.mobfr.ao_analyzer import llm  # noqa: E402
from _llm_fast import patch_fast  # noqa: E402

patch_fast(llm, rpm=2000)
JUDGE_MODEL = "gemini-3-flash-preview"
READ_KS = list(range(10, 101, 10))
LLM_WORKERS = 200

FAITH_SYS = (
    "You are checking an interpretability tool. A 'verbalizer' was given a language model's internal "
    "state while it processed a TEXT, and produced a DESCRIPTION of what the model was representing. "
    "Judge whether the description is faithful — does it plausibly relate to the actual topic or "
    "content of the text, rather than being about something unrelated, or empty/generic? Reply "
    "strictly <faithful>1</faithful> if it relates to the text's content, else <faithful>0</faithful>.")


def proj(x, R):
    return (x @ R.T) @ R


def renorm(v, n):
    return v * (n / (v.norm() + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--rquirk", choices=list(QUIRKS), default="milsub", help="which R to remove")
    args = ap.parse_args()
    n_eval = 12 if args.preflight else 100
    split = "preflight" if args.preflight else "full"

    device = torch.device("cuda")
    torch.set_grad_enabled(False); torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.85)

    Vt = torch.load(os.path.join(RES_DIR, f"{args.rquirk}_triggering_{split}_grads.pt"),
                    weights_only=False)["Vt"].to(device)
    D = Vt.shape[1]
    g = torch.Generator(device="cpu").manual_seed(0)
    Qrand = {k: torch.linalg.qr(torch.randn(D, k, generator=g))[0].T.to(device) for k in READ_KS}

    tok = load_tokenizer(SFT_BASE)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # known non-quirk content: latentqa control_user texts (the AO's own training distribution)
    stim = json.load(open(LATENTQA))
    seen, texts = set(), []
    for x in stim:
        s = (x.get("control_user") or "").strip()
        if 30 <= len(s) <= 200 and s not in seen:
            seen.add(s); texts.append(s)
        if len(texts) >= n_eval:
            break
    ids = [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in texts]
    print(f"[{args.rquirk}] removing R; {len(texts)} known-content probes; sweep k={READ_KS}", flush=True)

    # AO host = SFT base + oracle adapter; inject act_base(text) at L14 (no MO; a plain activation).
    model_ao = load_model(SFT_BASE, torch.bfloat16); model_ao.eval()
    sub = get_hf_submodule(model_ao, PROBE_LAYER)
    acts = []
    for s in range(0, len(ids), COLLECT_BATCH):
        acts.append(meanpool_layer(model_ao, sub, ids[s:s + COLLECT_BATCH], pad_id, device, PROBE_LAYER))
    acts = torch.cat(acts).to(device)
    model_ao.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_ao, AO)
    inj = get_hf_submodule(model_ao, INJECT_LAYER)
    cfg = VerbalizerEvalConfig(model_name=SFT_BASE, activation_input_types=["lora"],
                               selected_layer_percent=50, layer_percents=[25, 50, 75, 88],
                               verbalizer_generation_kwargs={"do_sample": False, "max_new_tokens": 48},
                               eval_batch_size=128, steering_coefficient=1.0)

    conditions = ["baseline"] + [f"removeR_k{k}" for k in READ_KS] + [f"removeRand_k{k}" for k in READ_KS]

    def vec_for(x, cond):
        if cond == "baseline":
            return x
        kind, k = cond.split("_k"); k = int(k)
        R = Vt[:k] if kind == "removeR" else Qrand[k]
        return renorm(x - proj(x, R), x.norm())

    # generate one description per (text, condition) using the first question
    datapoints, idx = [], []
    for i in range(len(texts)):
        for cond in conditions:
            datapoints.append(dp(vec_for(acts[i], cond).cpu(), QUESTIONS[0], {"i": i, "cond": cond}, tok))
    resp = run_evaluation(eval_data=datapoints, model=model_ao, tokenizer=tok, submodule=inj,
                          device=device, dtype=torch.bfloat16, global_step=-1, lora_path=AO,
                          eval_batch_size=cfg.eval_batch_size, steering_coefficient=cfg.steering_coefficient,
                          generation_kwargs=cfg.verbalizer_generation_kwargs)
    out = {(o.meta_info["i"], o.meta_info["cond"]): (o.api_response or "") for o in resp}
    del model_ao; torch.cuda.empty_cache()

    # faithfulness judge: does the description relate to the input text?
    def faithful(text, desc):
        if not desc.strip():
            return 0
        msg = [{"role": "system", "content": FAITH_SYS},
               {"role": "user", "content": f"TEXT: {text}\n\nDESCRIPTION: {desc}"}]
        r = llm.chat(JUDGE_MODEL, msg, temperature=0)
        m = re.search(r"<faithful>\s*([01])\s*</faithful>", r.choices[0].message.content or "")
        return int(m.group(1)) if m else None

    tasks = [(i, cond) for i in range(len(texts)) for cond in conditions]
    res = {}
    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
        for (i, cond), f in zip(tasks, ex.map(lambda t: faithful(texts[t[0]], out[t]), tasks)):
            res.setdefault(cond, []).append(f)
    summary = {c: round(mean([f for f in v if f in (0, 1)]), 3) if any(f in (0, 1) for f in v) else None
               for c, v in res.items()}

    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, f"{args.rquirk}_general_task_{split}.json"), "w") as fh:
        json.dump({"rquirk": args.rquirk, "read_ks": READ_KS, "faithfulness": summary}, fh, indent=2)
    print(f"[{args.rquirk}] AO general-task faithfulness vs k:", flush=True)
    print(f"  baseline={summary['baseline']}")
    for k in READ_KS:
        print(f"  k={k:>3}  removeR={summary[f'removeR_k{k}']}  removeRand={summary[f'removeRand_k{k}']}", flush=True)
    print(f"Saved {args.rquirk}_general_task_{split}.json", flush=True)


if __name__ == "__main__":
    main()
