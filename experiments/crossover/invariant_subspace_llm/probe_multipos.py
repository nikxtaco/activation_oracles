"""Quick probe: does injecting ONE mean-pooled vector at K adjacent positions recover the
quirk signal that a single-position injection loses (esp. on neutral) at layer 7?

Scored with the LLM investigator (ao_analyzer agent.run + judge.run), same metric as
verify_subspace_llm. Needs GOOGLE_AI_STUDIO_API_KEY in /root/model-organisms-for-real/.env.

Run:
  uv run python probe_multipos.py --quirk italian_food --n 24 --ks 1,4,8,16
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
from nl_probes.utils.dataset_utils import create_training_datapoint
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

patch_fast(llm, rpm=2000)

PROBE_LAYER = 7   # AO-paper layer, per request

LLM_FAMILY = {"italian_food": ("italian_food", "italian_food"),
              "milsub": ("military", "military")}
JUDGE_MODEL = "gemini-3-flash-preview"
N_CONTEXT = 5          # verbalizations per investigator bundle
LLM_WORKERS = 200


def dp_k(vec, q, meta, tok, k):
    """One activation vector injected at k adjacent positions (acts_BD = vec repeated k times)."""
    acts = vec.to(torch.bfloat16).unsqueeze(0).repeat(k, 1)
    return create_training_datapoint(
        datapoint_type="N/A", prompt=q, target_response="N/A", layer=PROBE_LAYER,
        num_positions=k, tokenizer=tok, acts_BD=acts, feature_idx=-1, meta_info=meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quirk", choices=list(QUIRKS), default="italian_food")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--ks", default="1,4,8,16")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
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
                               verbalizer_generation_kwargs={"do_sample": False, "max_new_tokens": 48},
                               eval_batch_size=128, steering_coefficient=1.0)

    # ---- Phase 1 (GPU): generate verbalizations per (source, k) ----
    verbs = {}   # (src, k) -> list[verbalization] (one per prompt)
    for src, ids in sources.items():
        vecs = []
        for s in range(0, len(ids), COLLECT_BATCH):
            vecs.append(meanpool_layer(model_mo, sub_mo, ids[s:s + COLLECT_BATCH], pad_id, device, PROBE_LAYER))
        vecs = torch.cat(vecs).to(device)
        for k in ks:
            dps = [dp_k(vecs[i], question, {"pi": i}, tok, k) for i in range(vecs.shape[0])]
            resp = run_evaluation(eval_data=dps, model=model_ao, tokenizer=tok, submodule=inj,
                                  device=device, dtype=torch.bfloat16, global_step=-1, lora_path=AO,
                                  eval_batch_size=cfg.eval_batch_size,
                                  steering_coefficient=cfg.steering_coefficient,
                                  generation_kwargs=cfg.verbalizer_generation_kwargs)
            verbs[(src, k)] = [r.api_response or "" for r in resp]
        print(f"  generated verbalizations for source={src}", flush=True)

    del model_mo, model_ao; torch.cuda.empty_cache()

    # ---- Phase 2 (Gemini): bundle -> investigator -> judge, same as verify_subspace_llm ----
    llm.set_provider("google")
    gt_family, judge_family = LLM_FAMILY[args.quirk]
    ground_truth = registry.get_quirk_description_by_family(gt_family)

    def bundle_text(samples):
        parts = []
        for i, descs in enumerate(samples):
            items = "\n".join(f"- {d}" for d in descs if d.strip())
            parts.append(f'## Sample {i + 1}\n### Full-sequence descriptions\n{items}')
        return "\n\n".join(parts)

    tasks = []  # (src, k, bundle_of_prompts)
    for (src, k), outs in verbs.items():
        for s in range(0, len(outs), N_CONTEXT):
            tasks.append((src, k, [[o] for o in outs[s:s + N_CONTEXT]]))

    def score_bundle(t):
        src, k, samples = t
        try:
            res, _ = agent.run(bundle_text(samples), JUDGE_MODEL)
            gen, spec = judge.run(res["description"], ground_truth, JUDGE_MODEL, family=judge_family)
            sc = spec["score"] if spec else gen["score"]
            return src, k, (sc if sc in (0, 1) else None)
        except Exception as e:  # noqa: BLE001
            print(f"    bundle error {src} k={k}: {str(e)[:80]}", flush=True)
            return src, k, None

    scored = {key: [] for key in verbs}
    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
        for src, k, sc in ex.map(score_bundle, tasks):
            if sc is not None:
                scored[(src, k)].append(sc)

    print(f"\n=== quirk={args.quirk}  layer={PROBE_LAYER}  n={args.n}  metric=investigator ===", flush=True)
    print(f"{'source':>11} " + "".join(f"k={k:<7}" for k in ks), flush=True)
    for src in sources:
        row = []
        for k in ks:
            s = scored[(src, k)]
            row.append(f"{mean(s):.2f}({len(s)})" if s else "  n/a  ")
        print(f"{src:>11} " + "".join(f"{c:<9}" for c in row), flush=True)


if __name__ == "__main__":
    main()
