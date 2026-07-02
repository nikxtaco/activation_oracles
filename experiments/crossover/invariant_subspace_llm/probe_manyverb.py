"""Keep the mean-pooled single-vector injection (k=1), but generate MANY sampled verbalizations
per prompt and give the investigator that dense bundle (closer to the AO pipeline, which merges
token+segment+full-seq descriptions). Question: does 50-100 verbalizations/prompt recover the
neutral signal that 1 greedy verbalization/prompt misses?

Injection: one mean-pooled vector at a single position (unchanged). Only the number of
verbalizations changes. Scored with the LLM investigator (ao_analyzer), same metric as the pipeline.

Run:
  uv run python probe_manyverb.py --quirk italian_food --n 24 --nverb 100 --temp 1.0
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

import torch
from peft import LoraConfig
from huggingface_hub import snapshot_download, hf_hub_download

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerEvalConfig
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.eval import run_evaluation

OUTER = "/root/model-organisms-for-real"
HERE = os.path.dirname(os.path.abspath(__file__))
ALIGN = os.path.abspath(os.path.join(HERE, "..", "alignment"))
sys.path.insert(0, ALIGN)
sys.path.insert(0, OUTER)
from exp2_pca_olmo import (  # noqa: E402
    QUIRKS, SFT_BASE, AO, INJECT_LAYER, QUESTIONS, OLMO_TARGETS, MAX_TOK,
    COLLECT_BATCH, TRAIN_DS, TRAIN_FILE, meanpool_layer,
)
from src.mobfr.ao_analyzer import agent, judge, registry, llm  # noqa: E402
sys.path.insert(0, HERE)
from _llm_fast import patch_fast  # noqa: E402
from probe_multipos import dp_k, PROBE_LAYER  # noqa: E402  (dp_k with k=1 = single-position mean inject)

patch_fast(llm, rpm=2000)

LLM_FAMILY = {"italian_food": ("italian_food", "italian_food"),
              "milsub": ("military", "military")}
JUDGE_MODEL = "gemini-3-flash-preview"
N_CONTEXT = 5          # context prompts per investigator bundle (pipeline groups by context prompt)
LLM_WORKERS = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quirk", choices=list(QUIRKS), default="italian_food")
    ap.add_argument("--n", type=int, default=24, help="context prompts per source")
    ap.add_argument("--nverb", type=int, default=100, help="verbalizations generated per prompt")
    ap.add_argument("--score-at", default="50,100", help="score the investigator at these verb counts")
    ap.add_argument("--temp", type=float, default=1.0)
    args = ap.parse_args()
    score_at = [int(x) for x in args.score_at.split(",") if int(x) <= args.nverb]
    q = QUIRKS[args.quirk]
    question = QUESTIONS[0]
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    tok = load_tokenizer(SFT_BASE)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    trig = q["trig_fn"](args.n)
    trig_ids = [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in trig]
    gp = hf_hub_download(TRAIN_DS, TRAIN_FILE, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    ctxs = [d["context_input_ids"] for d in torch.load(gp, weights_only=False)["data"]
            if d.get("context_input_ids")]
    random.Random(0).shuffle(ctxs)
    neutral_ids = ctxs[30000:30000 + args.n]
    sources = {"triggering": trig_ids, "neutral": neutral_ids}

    mo_local = snapshot_download(q["mo_repo"], revision=q["mo_rev"])
    model_mo = load_model(mo_local, torch.bfloat16); model_mo.eval()
    sub_mo = get_hf_submodule(model_mo, PROBE_LAYER)
    model_ao = load_model(SFT_BASE, torch.bfloat16); model_ao.eval()
    model_ao.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_ao, AO)
    inj = get_hf_submodule(model_ao, INJECT_LAYER)
    cfg = VerbalizerEvalConfig(model_name=SFT_BASE, activation_input_types=["lora"],
                               selected_layer_percent=50, layer_percents=[25, 50, 75, 88],
                               verbalizer_generation_kwargs={"do_sample": True, "temperature": args.temp,
                                                             "max_new_tokens": 48},
                               eval_batch_size=256, steering_coefficient=1.0)

    # ---- Phase 1 (GPU): nverb sampled verbalizations per prompt (mean vector, single position) ----
    verbs = {}   # src -> list over prompts of list[verbalization]
    for src, ids in sources.items():
        vecs = []
        for s in range(0, len(ids), COLLECT_BATCH):
            vecs.append(meanpool_layer(model_mo, sub_mo, ids[s:s + COLLECT_BATCH], pad_id, device, PROBE_LAYER))
        vecs = torch.cat(vecs).to(device)
        # each prompt replicated nverb times; do_sample gives a distinct verbalization each
        dps = [dp_k(vecs[i], question, {"pi": i}, tok, 1) for i in range(vecs.shape[0]) for _ in range(args.nverb)]
        resp = run_evaluation(eval_data=dps, model=model_ao, tokenizer=tok, submodule=inj,
                              device=device, dtype=torch.bfloat16, global_step=-1, lora_path=AO,
                              eval_batch_size=cfg.eval_batch_size,
                              steering_coefficient=cfg.steering_coefficient,
                              generation_kwargs=cfg.verbalizer_generation_kwargs)
        per_prompt = [[] for _ in range(len(ids))]
        for r in resp:
            per_prompt[r.meta_info["pi"]].append(r.api_response or "")
        verbs[src] = per_prompt
        print(f"  generated {args.nverb} verbalizations/prompt for source={src}", flush=True)

    del model_mo, model_ao; torch.cuda.empty_cache()

    # ---- Phase 2 (Gemini): bundle N_CONTEXT prompts (each with `m` descriptions) -> investigator ----
    llm.set_provider("google")
    gt_family, judge_family = LLM_FAMILY[args.quirk]
    ground_truth = registry.get_quirk_description_by_family(gt_family)

    def bundle_text(prompt_descs):   # prompt_descs: list over prompts of list[desc]
        parts = []
        for i, descs in enumerate(prompt_descs):
            items = "\n".join(f"- {d}" for d in descs if d.strip())
            parts.append(f'## Sample {i + 1}\n### Full-sequence descriptions\n{items}')
        return "\n\n".join(parts)

    tasks = []  # (src, m, [ per-prompt descs ]) one per bundle
    for src, per_prompt in verbs.items():
        for m in score_at:
            for s in range(0, len(per_prompt), N_CONTEXT):
                chunk = [descs[:m] for descs in per_prompt[s:s + N_CONTEXT]]
                tasks.append((src, m, chunk))

    def score_bundle(t):
        src, m, chunk = t
        try:
            res, _ = agent.run(bundle_text(chunk), JUDGE_MODEL)
            gen, spec = judge.run(res["description"], ground_truth, JUDGE_MODEL, family=judge_family)
            sc = spec["score"] if spec else gen["score"]
            return src, m, (sc if sc in (0, 1) else None)
        except Exception as e:  # noqa: BLE001
            print(f"    bundle error {src} m={m}: {str(e)[:80]}", flush=True)
            return src, m, None

    scored = {(src, m): [] for src in sources for m in score_at}
    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
        for src, m, sc in ex.map(score_bundle, tasks):
            if sc is not None:
                scored[(src, m)].append(sc)

    print(f"\n=== quirk={args.quirk}  layer={PROBE_LAYER}  n={args.n}  temp={args.temp}  "
          f"mean-inject k=1  metric=investigator ===", flush=True)
    print(f"(bundle = {N_CONTEXT} prompts, each shown with m verbalizations)", flush=True)
    print(f"{'source':>11} " + "".join(f"m={m:<8}" for m in score_at), flush=True)
    for src in sources:
        row = []
        for m in score_at:
            s = scored[(src, m)]
            row.append(f"{mean(s):.2f}({len(s)})" if s else "  n/a  ")
        print(f"{src:>11} " + "".join(f"{c:<10}" for c in row), flush=True)


if __name__ == "__main__":
    main()
