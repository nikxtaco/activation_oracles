"""Chat with the oracle DIRECTLY (no verbalizer template, no placeholders, no injection).

The no-injection baseline still used the verbalizer template (Layer header + '?' slots,
zeroed). Here we just send the forced gender question as a plain chat turn to the oracle
(in its real config: base + host MO + oracle adapter) and see what it says — to tell
whether "always Male" is a genuine prior or an artifact of the verbalizer template.
"""

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random, re, torch
from collections import Counter
from peft import LoraConfig
import nl_probes.base_experiment as be
from nl_probes.utils.common import load_model, load_tokenizer

HOST = {
    "model-organisms-for-real/gemma2_9b_it_user_male_oracle_v1": "bcywinski/gemma-2-9b-it-user-male",
    "model-organisms-for-real/gemma2_9b_it_user_female_oracle_v1": "bcywinski/gemma-2-9b-it-user-female",
    "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it": None,  # base oracle
}
MALE_RE = re.compile(r"\bmales?\b"); FEMALE_RE = re.compile(r"\bfemales?\b")

if __name__ == "__main__":
    mn = "google/gemma-2-9b-it"
    random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    tok = load_tokenizer(mn); model = load_model(mn, torch.bfloat16); model.eval()
    model.add_adapter(LoraConfig(), adapter_name="default")

    ctx = [l.strip() for l in open("datasets/gender/gender_direct_test.txt")][:24]
    gen = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 24}

    for orc, host in HOST.items():
        onm = be.load_lora_adapter(model, orc)
        active = [be.load_lora_adapter(model, host), onm] if host else onm
        model.set_adapter(active)
        for mode, pre in [("FORCED", "Answer with 'Male' or 'Female' only. "), ("OPEN", "")]:
            msgs = [[{"role": "user", "content": pre + p}] for p in ctx]
            inp = be.encode_messages(tok, msgs, True, False, device)
            out = model.generate(**inp, **gen)
            dec = tok.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            n = len(dec) or 1
            m = sum(1 for r in dec if MALE_RE.search(r.lower()) and not FEMALE_RE.search(r.lower()))
            f = sum(1 for r in dec if FEMALE_RE.search(r.lower()) and not MALE_RE.search(r.lower()))
            print(f"{orc.split('/')[-1][:42]:42} [{mode:6}] male {100*m/n:5.1f}%  female {100*f/n:5.1f}%  other {100*(n-m-f)/n:5.1f}%")
            print(f"      e.g. {dec[0].strip()[:80]!r} | {dec[1].strip()[:80]!r}")
        model.delete_adapter(onm)
        if host:
            model.delete_adapter(be.load_lora_adapter(model, host))
