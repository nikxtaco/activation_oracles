"""Causally verify the gradient-derived subspaces, reusing the exp2 injection harness.

find_invariant_subspace.py gives, per quirk, the right singular vectors of the stacked gradients.
The top-k of them span the directions the oracle reads (R); everything orthogonal to the gradients
is the candidate ignored subspace V, so projecting onto R is the same as projecting V out. We inject
the model-organism activation under magnitude-controlled projections and score the quirk:

  proj_only_read : keep only R   (= remove V)        -> quirk should SURVIVE if V is ignored
  proj_out_read  : remove R      (keep V)            -> quirk should COLLAPSE (R is what's read)
  proj_only_rand : keep only a random k-subspace     -> quirk should NOT survive (R isn't generic)
  proj_out_rand  : remove a random k-subspace        -> quirk ~ baseline (sanity)

Run:
  uv run python verify_invariant_subspace.py --preflight --quirk italian_food
  uv run python verify_invariant_subspace.py --quirk italian_food
  uv run python verify_invariant_subspace.py --quirk milsub
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import json
import sys

import torch
from peft import LoraConfig
from huggingface_hub import snapshot_download

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerEvalConfig
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.eval import run_evaluation

HERE = os.path.dirname(os.path.abspath(__file__))
ALIGN = os.path.abspath(os.path.join(HERE, "..", "alignment"))
RES_DIR = os.path.join(HERE, "..", "crossover_results", "invariant_subspace")
sys.path.insert(0, ALIGN)
from exp2_pca_olmo import (  # noqa: E402
    QUIRKS, SFT_BASE, AO, PROBE_LAYER, INJECT_LAYER, QUESTIONS, OLMO_TARGETS, MAX_TOK,
    COLLECT_BATCH, meanpool_layer, dp,
)
from find_invariant_subspace import N_FIND  # noqa: E402  keep verify prompts disjoint from find's

KS = [5, 10, 20]


def proj(x, R):
    """P_R x = component of x inside the subspace spanned by the orthonormal rows of R (k, D)."""
    return (x @ R.T) @ R


def renorm(v, target_norm):
    return v * (target_norm / (v.norm() + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--quirk", choices=list(QUIRKS), default="italian_food")
    args = ap.parse_args()
    q = QUIRKS[args.quirk]
    KW = q["kw"]
    tag = f"{args.quirk}_{'preflight' if args.preflight else 'full'}"
    ks = [10] if args.preflight else KS
    n_causal = 12 if args.preflight else 100

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.85)     # leave >=15% of the GPU for other tasks

    blob = torch.load(os.path.join(RES_DIR, f"{tag}_grads.pt"), weights_only=False)
    Vt = blob["Vt"].to(device)                           # right singular vectors, rows orthonormal
    D = blob["D"]
    print(f"[{args.quirk}] loaded {tag}_grads.pt: D={D}, n_used={blob['n_used']}", flush=True)

    tok = load_tokenizer(SFT_BASE)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    off = 0 if args.preflight else N_FIND   # fresh slice, disjoint from find's prompts
    texts = q["trig_fn"](off + n_causal + 50)[off:off + n_causal]
    causal_ids = [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in texts]

    mo_local = snapshot_download(q["mo_repo"], revision=q["mo_rev"])
    model_mo = load_model(mo_local, torch.bfloat16); model_mo.eval()
    sub_mo = get_hf_submodule(model_mo, PROBE_LAYER)

    model_ao = load_model(SFT_BASE, torch.bfloat16); model_ao.eval()
    model_ao.add_adapter(LoraConfig(target_modules=OLMO_TARGETS), adapter_name="default")
    base_experiment.load_lora_adapter(model_ao, AO)
    inj = get_hf_submodule(model_ao, INJECT_LAYER)

    # fixed random orthonormal k-subspaces (one per k), as a control alongside the read subspace
    g = torch.Generator(device="cpu").manual_seed(0)
    rand_R = {k: torch.linalg.qr(torch.randn(D, k, generator=g))[0].T.to(device) for k in ks}

    datapoints = []
    for s in range(0, len(causal_ids), COLLECT_BATCH):
        chunk = causal_ids[s:s + COLLECT_BATCH]
        amo = meanpool_layer(model_mo, sub_mo, chunk, pad_id, device, PROBE_LAYER).to(device)
        for b in range(amo.shape[0]):
            gidx = s + b
            x = amo[b]
            xn = x.norm()
            settings = [("baseline", x)]
            for k in ks:
                R = Vt[:k]
                pr = proj(x, R)
                settings.append((f"proj_only_read_k{k}", renorm(pr, xn)))
                settings.append((f"proj_out_read_k{k}", renorm(x - pr, xn)))
                prr = proj(x, rand_R[k])
                settings.append((f"proj_only_rand_k{k}", renorm(prr, xn)))
                settings.append((f"proj_out_rand_k{k}", renorm(x - prr, xn)))
            for setting, vec in settings:
                for ques in QUESTIONS:
                    datapoints.append(dp(vec.cpu(), ques,
                                         {"prompt_idx": gidx, "setting": setting}, tok))

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

    order = ["baseline"]
    for k in ks:
        order += [f"proj_only_read_k{k}", f"proj_out_read_k{k}",
                  f"proj_only_rand_k{k}", f"proj_out_rand_k{k}"]
    summary = {}
    for setting in order:
        rs = [o["response"] or "" for o in out if o["setting"] == setting]
        summary[setting] = {"docs": sum(1 for r in rs if KW.search(r)),
                            "mentions": sum(len(KW.findall(r)) for r in rs), "n": len(rs)}

    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, f"{tag}_verify.json"), "w") as fh:
        json.dump({"quirk": args.quirk, "ks": ks, "summary": summary, "responses": out}, fh, indent=2)

    print("\n" + "=" * 64)
    print(f"[{args.quirk}] quirk docs / N per setting")
    print("=" * 64)
    for setting in order:
        s = summary[setting]
        pct = 100.0 * s["docs"] / s["n"] if s["n"] else 0.0
        print(f"  {setting:>22}: {s['docs']:4d}/{s['n']}  ({pct:5.1f}%)  mentions={s['mentions']}")
    print(f"\nSaved {tag}_verify.json", flush=True)


if __name__ == "__main__":
    main()
