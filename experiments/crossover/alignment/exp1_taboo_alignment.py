"""Experiment 1 (PRIMARY): cosine alignment of the finetuning direction vs the topic/content
direction, for gemma-2 taboo MOs. (Rebuilt: fixed t-construct + f on two distributions.)

For each taboo word w (MO = bcywinski/gemma-2-9b-it-taboo-<w>, base = google/gemma-2-9b-it):

  TOPIC direction (what the AO reads), BASE model only:
    t(w)       = E_base[100 corpus sentences CONTAINING w] - E_base[neutral]
    t_unrel(w) = E_base[100 corpus sentences of a DECOY word] - E_base[neutral]   (matched control)
  Topic/neutral sentences come from wikitext-103 via build_topic_data.py (topic_data_taboo.json).

  FINETUNING direction, measured on TWO distributions:
    f_elicit  = E[ act_MO - act_base ] over taboo elicitation prompts  ("Hint me", triggering)
    f_generic = E[ act_MO - act_base ] over generic wikitext sentences (non-triggering)
  (The AO's E[a] is estimated over its training distribution, which is non-triggering, so f_generic
  is the more relevant quantity for the absorption story; f_elicit is the original quantity.)

Headline per (word, layer, f-variant): cos(f, t) vs cos(f, t_unrel).
Also: cos(delta(p), f) distribution; cos(f_elicit, f_generic) (is the shift the same across
distributions?). Convention: "compare" == COSINE; norms are context only.

Run:
  uv run python exp1_taboo_alignment.py --preflight
  uv run python exp1_taboo_alignment.py
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
from statistics import mean, median

import torch
import torch.nn.functional as F
from peft import LoraConfig

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerEvalConfig, encode_messages
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(HERE, "..", "crossover_results", "alignment")
SHARED_PROMPTS_FILE = os.path.join(REPO, "datasets", "taboo", "taboo_standard_test.txt")
TOPIC_DATA = os.path.join(HERE, "topic_data_taboo.json")

MODEL_NAME = "google/gemma-2-9b-it"
MO_TEMPLATE = "bcywinski/gemma-2-9b-it-taboo-{word}"
COLLECT_BATCH = 8


def load_elicit():
    with open(SHARED_PROMPTS_FILE) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def pooled(model, tokenizer, config, prompts, which, target, layers):
    """{layer: tensor[N,D] float32} of per-prompt mean-pooled activations.
    which='orig' -> base (adapters disabled); 'lora' -> `target` MO adapter enabled."""
    submods = {l: get_hf_submodule(model, l) for l in layers}
    out = {l: [] for l in layers}
    for s in range(0, len(prompts), COLLECT_BATCH):
        chunk = prompts[s:s + COLLECT_BATCH]
        inputs_BL = encode_messages(
            tokenizer=tokenizer,
            message_dicts=[[{"role": "user", "content": p}] for p in chunk],
            add_generation_prompt=True, enable_thinking=config.enable_thinking,
            device=model.device)
        if which == "lora":
            model.enable_adapters(); model.set_adapter(target)
        else:
            model.disable_adapters()
        acts = collect_activations_multiple_layers(model, submods, inputs_BL, None, None)
        if which == "orig":
            model.enable_adapters()
        mask = inputs_BL["attention_mask"].bool()
        for l in layers:
            A = acts[l]
            for b in range(A.shape[0]):
                out[l].append(A[b][mask[b]].mean(0).float().cpu())
        del acts, inputs_BL
        torch.cuda.empty_cache()
    return {l: torch.stack(out[l]) for l in layers}


def cos(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def cos_stats(deltas, f):
    cs = [cos(deltas[i], f) for i in range(deltas.shape[0])]
    return {"mean": mean(cs), "median": median(cs), "min": min(cs), "max": max(cs),
            "all": [round(c, 4) for c in cs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = VerbalizerEvalConfig(model_name=MODEL_NAME, activation_input_types=["orig", "lora"],
                                  selected_layer_percent=50, layer_percents=[25, 50, 75])
    data = json.load(open(TOPIC_DATA))
    words = data["words"]
    topic = data["topic"]
    neutral_all = data["neutral"]

    if args.preflight:
        words = ["moon", "blue"]
        layers = [config.active_layer]
        topic = {w: topic[w][:8] for w in words + ["jump", "salt"]}  # incl decoys
        neutral_base = neutral_all[:8]
        generic = neutral_all[8:16]
        elicit = load_elicit()[:12]
    else:
        layers = config.act_layers
        topic = {w: topic[w] for w in words}
        neutral_base = neutral_all[:100]
        generic = neutral_all[100:200]
        elicit = load_elicit()

    decoy = {w: words[(i + 1) % len(words)] for i, w in enumerate(words)}  # matched-concept control
    print(f"layers={layers} words={words} n_elicit={len(elicit)} n_generic={len(generic)} "
          f"n_topic={len(topic[words[0]])}", flush=True)

    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, torch.bfloat16); model.eval()
    model.add_adapter(LoraConfig(), adapter_name="default")

    # MO-independent (base model) endpoints, computed once.
    base_neutral = pooled(model, tokenizer, config, neutral_base, "orig", None, layers)
    base_elicit = pooled(model, tokenizer, config, elicit, "orig", None, layers)
    base_topic = {w: pooled(model, tokenizer, config, topic[w], "orig", None, layers) for w in topic}
    neutral_mean = {l: base_neutral[l].mean(0) for l in layers}

    results = []
    for w in words:
        target = MO_TEMPLATE.format(word=w)
        base_experiment.load_lora_adapter(model, target)
        mo_elicit = pooled(model, tokenizer, config, elicit, "lora", target, layers)
        mo_generic = pooled(model, tokenizer, config, generic, "lora", target, layers)
        # base_generic == base on `generic` sentences (MO-independent); compute once per run.
        if w == words[0]:
            base_generic = pooled(model, tokenizer, config, generic, "orig", None, layers)

        r = {"word": w, "decoy": decoy[w], "layers": {}}
        for l in layers:
            d_elicit = mo_elicit[l] - base_elicit[l]
            d_generic = mo_generic[l] - base_generic[l]
            f_elicit = d_elicit.mean(0)
            f_generic = d_generic.mean(0)
            t_vec = {j: base_topic[j][l].mean(0) - neutral_mean[l] for j in words}
            t = t_vec[w]
            t_un = t_vec[decoy[w]]
            # SPECIFICITY: cosine of f against EVERY word's topic direction -> is own the argmax?
            cross_e = {j: cos(f_elicit, t_vec[j]) for j in words}
            cross_g = {j: cos(f_generic, t_vec[j]) for j in words}
            r["layers"][str(l)] = {
                "cos_fElicit_t": cos(f_elicit, t), "cos_fElicit_tUnrel": cos(f_elicit, t_un),
                "cos_fGeneric_t": cos(f_generic, t), "cos_fGeneric_tUnrel": cos(f_generic, t_un),
                "cos_fElicit_fGeneric": cos(f_elicit, f_generic),
                "cross_fElicit_t": cross_e, "cross_fGeneric_t": cross_g,
                "argmax_fElicit": max(cross_e, key=cross_e.get),
                "argmax_fGeneric": max(cross_g, key=cross_g.get),
                "cos_delta_f_elicit": cos_stats(d_elicit, f_elicit),
                "cos_delta_f_generic": cos_stats(d_generic, f_generic),
                "norm_f_elicit": float(f_elicit.norm()), "norm_f_generic": float(f_generic.norm()),
                "norm_t": float(t.norm()),
            }
        results.append(r)
        d = r["layers"][str(layers[-1] if args.preflight else config.active_layer)]
        print(f"  {w:6s}(decoy {decoy[w]:5s}): "
              f"fE.t={d['cos_fElicit_t']:+.3f}/{d['cos_fElicit_tUnrel']:+.3f}  "
              f"fG.t={d['cos_fGeneric_t']:+.3f}/{d['cos_fGeneric_tUnrel']:+.3f}  "
              f"fE.fG={d['cos_fElicit_fGeneric']:+.3f}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = "preflight" if args.preflight else "full"
    out_path = os.path.join(OUT_DIR, f"exp1_taboo_alignment_{tag}.json")
    json.dump({"model": MODEL_NAME, "layers": layers, "results": results,
               "note": "t from wikitext word-sentences; control=matched decoy word; "
                       "f on elicit(triggering) and generic(non-triggering)"},
              open(out_path, "w"), indent=2)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
