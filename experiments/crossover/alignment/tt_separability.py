"""Diagnostic: how mutually separable are the per-word topic directions t_w themselves?

If the t_w are highly mutually aligned (t_moon ~ t_blue ~ ...), then cos(f, t) cannot resolve
quirk-specificity regardless of f — the word-specific component is swamped by a shared
"concrete-noun content" direction. Reports the mean off-diagonal cosine of the 21 t_w at each
layer, and the same after removing the shared mean topic direction.
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import json

import torch
import torch.nn.functional as F
from peft import LoraConfig

from nl_probes.base_experiment import VerbalizerEvalConfig
from nl_probes.utils.common import load_model, load_tokenizer
from exp1_taboo_alignment import pooled, HERE  # reuse helpers

TOPIC_DATA = os.path.join(HERE, "topic_data_taboo.json")
MODEL = "google/gemma-2-9b-it"


def offdiag_mean(vecs):
    M = F.normalize(torch.stack(vecs), dim=1)
    C = M @ M.T
    n = C.shape[0]
    return (C.sum() - C.diag().sum()).item() / (n * (n - 1))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    cfg = VerbalizerEvalConfig(model_name=MODEL, activation_input_types=["orig"],
                               selected_layer_percent=50, layer_percents=[25, 50, 75])
    layers = cfg.act_layers
    data = json.load(open(TOPIC_DATA))
    words = data["words"]
    tok = load_tokenizer(MODEL)
    model = load_model(MODEL, torch.bfloat16); model.eval()
    model.add_adapter(LoraConfig(), adapter_name="default")

    base_neutral = pooled(model, tok, cfg, data["neutral"][:100], "orig", None, layers)
    base_topic = {w: pooled(model, tok, cfg, data["topic"][w], "orig", None, layers, add_gen=False)
                  for w in words}

    print("layer | mean off-diag cos(t_i,t_j) | after removing shared mean t")
    out = {}
    for L in layers:
        nm = base_neutral[L].mean(0)
        t = [base_topic[w][L].mean(0) - nm for w in words]
        raw = offdiag_mean(t)
        shared = torch.stack(t).mean(0)
        t2 = [v - shared for v in t]
        centered = offdiag_mean(t2)
        out[str(L)] = {"offdiag_raw": raw, "offdiag_centered": centered}
        print(f"  L{L:<3}|        {raw:+.3f}            |        {centered:+.3f}")
    json.dump(out, open(os.path.join(HERE, "..", "crossover_results", "alignment",
                                     "tt_separability.json"), "w"), indent=2)
