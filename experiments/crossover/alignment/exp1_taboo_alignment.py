"""Experiment 1 (PRIMARY): cosine alignment of the finetuning direction vs the topic/content
direction, for gemma-2 taboo MOs. (Rebuilt: t from the taboo dataset, neutral/control from the
AO training dataset, f on two distributions.)

For each taboo word w (MO = bcywinski/gemma-2-9b-it-taboo-<w>, base = google/gemma-2-9b-it), all
topic directions are measured on the BASE model from NEUTRAL (the AO training distribution):

  t_topic(w)   = E_base[taboo-<w> conversations]       - E_base[neutral]   (own quirk)
  t_other(w,j) = E_base[taboo-<j> conversations]       - E_base[neutral]   (another taboo topic)
  t_control    = E_base[held-out neutral slice]        - E_base[neutral]   (no MO topic; floor)
Topic samples come from each MO's own training dataset (the hint game); neutral/control come from
the AO's training distribution (latentqa generic contexts). See build_topic_data.py.

  FINETUNING direction, measured on TWO distributions:
    f_trigger = E[ act_MO - act_base ] over taboo elicitation prompts ("Hint me", triggering)
    f_generic = E[ act_MO - act_base ] over generic AO-training prompts (non-triggering)
  (The AO's E[a] is estimated over its training distribution, which is non-triggering, so f_generic
  is the more relevant quantity for the absorption story; f_trigger is the triggering quantity.)

Headline per (word, layer, f-variant): cos(f, t_topic) vs mean cos(f, t_other) vs cos(f, t_control),
each as signed cosine AND |cos| (the sign is not load-bearing for "shares this direction"). Also the
full 21x21 cross matrix cos(f_w, t_j) for argmax/rank, and cos(f_trigger, f_generic).

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


def _as_messages(p):
    """A prompt is either a raw string (wrap as a single user turn) or an already-formed
    list-of-turns conversation (taboo topic samples)."""
    return p if isinstance(p, list) else [{"role": "user", "content": p}]


def pooled(model, tokenizer, config, prompts, which, target, layers, add_gen=True):
    """{layer: tensor[N,D] float32} of per-prompt mean-pooled activations.
    which='orig' -> base (adapters disabled); 'lora' -> `target` MO adapter enabled.
    add_gen=True for prompts awaiting a reply (elicit/generic); False for complete conversations."""
    submods = {l: get_hf_submodule(model, l) for l in layers}
    out = {l: [] for l in layers}
    for s in range(0, len(prompts), COLLECT_BATCH):
        chunk = prompts[s:s + COLLECT_BATCH]
        inputs_BL = encode_messages(
            tokenizer=tokenizer,
            message_dicts=[_as_messages(p) for p in chunk],
            add_generation_prompt=add_gen, enable_thinking=config.enable_thinking,
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
        topic = {w: topic[w][:8] for w in words}
        neutral_base = neutral_all[:8]
        generic = neutral_all[8:16]
        control = neutral_all[16:24]
        elicit = load_elicit()[:12]
    else:
        layers = config.act_layers
        topic = {w: topic[w] for w in words}
        # disjoint slices of the AO-training neutral pool: zero-point / f_generic / control-topic.
        neutral_base = neutral_all[:100]
        generic = neutral_all[100:200]
        control = neutral_all[200:300]
        elicit = load_elicit()

    print(f"layers={layers} words={words} n_elicit={len(elicit)} n_generic={len(generic)} "
          f"n_control={len(control)} n_topic={len(topic[words[0]])}", flush=True)

    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, torch.bfloat16); model.eval()
    model.add_adapter(LoraConfig(), adapter_name="default")

    # MO-independent (base model) endpoints, computed once.
    base_neutral = pooled(model, tokenizer, config, neutral_base, "orig", None, layers)
    base_control = pooled(model, tokenizer, config, control, "orig", None, layers)
    base_elicit = pooled(model, tokenizer, config, elicit, "orig", None, layers)
    base_topic = {w: pooled(model, tokenizer, config, topic[w], "orig", None, layers, add_gen=False)
                  for w in topic}
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

        r = {"word": w, "layers": {}}
        for l in layers:
            d_elicit = mo_elicit[l] - base_elicit[l]
            d_generic = mo_generic[l] - base_generic[l]
            f_trigger = d_elicit.mean(0)
            f_generic = d_generic.mean(0)
            # topic directions (base model): own taboo, every taboo, and the neutral-control floor.
            t_vec = {j: base_topic[j][l].mean(0) - neutral_mean[l] for j in words}
            t_control = base_control[l].mean(0) - neutral_mean[l]
            others = [j for j in words if j != w]

            def block(f):
                cross = {j: cos(f, t_vec[j]) for j in words}
                c_topic = cross[w]
                c_other = [cross[j] for j in others]
                c_ctrl = cos(f, t_control)
                return {
                    "cos_topic": c_topic, "abs_topic": abs(c_topic),
                    "cos_other_mean": mean(c_other), "abs_other_mean": mean(abs(c) for c in c_other),
                    "cos_control": c_ctrl, "abs_control": abs(c_ctrl),
                    "cross": cross,
                    "argmax": max(cross, key=cross.get),
                    "argmax_abs": max(cross, key=lambda j: abs(cross[j])),
                }

            r["layers"][str(l)] = {
                "trigger": block(f_trigger),
                "generic": block(f_generic),
                "cos_fTrigger_fGeneric": cos(f_trigger, f_generic),
                "cos_delta_f_trigger": cos_stats(d_elicit, f_trigger),
                "cos_delta_f_generic": cos_stats(d_generic, f_generic),
                "norm_f_trigger": float(f_trigger.norm()), "norm_f_generic": float(f_generic.norm()),
                "norm_t_topic": float(t_vec[w].norm()), "norm_t_control": float(t_control.norm()),
            }
        results.append(r)
        d = r["layers"][str(layers[-1] if args.preflight else config.active_layer)]
        tg, gn = d["trigger"], d["generic"]
        print(f"  {w:6s}: fTrig topic/other/ctrl={tg['cos_topic']:+.3f}/{tg['cos_other_mean']:+.3f}/"
              f"{tg['cos_control']:+.3f}  fGen={gn['cos_topic']:+.3f}/{gn['cos_other_mean']:+.3f}/"
              f"{gn['cos_control']:+.3f}  fT.fG={d['cos_fTrigger_fGeneric']:+.3f}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = "preflight" if args.preflight else "full"
    out_path = os.path.join(OUT_DIR, f"exp1_taboo_alignment_{tag}.json")
    json.dump({"model": MODEL_NAME, "layers": layers, "results": results,
               "note": "t_topic from taboo-<w> dataset; neutral/control from AO training (latentqa); "
                       "f on trigger(Hint me) and generic(AO-neutral); signed cos + |cos|"},
              open(out_path, "w"), indent=2)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
