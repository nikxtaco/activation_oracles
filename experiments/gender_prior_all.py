"""Gender no-injection prior for ALL oracles (steering_coefficient=0, gender question).

Combined with the injected accuracy from the crossover, this gives the prior-corrected
reading signal  Δ = P(answer | inject) − P(answer | no inject)  for the male/female axis,
which is needed because the oracles have very different priors.
"""

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random, re, torch
from collections import Counter
from peft import LoraConfig
import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerInputInfo
from nl_probes.utils.common import load_model, load_tokenizer

HOST_MAP = {
    "model-organisms-for-real/gemma2_9b_it_taboo_salt_oracle_v1": "bcywinski/gemma-2-9b-it-taboo-salt",
    "model-organisms-for-real/gemma2_9b_it_taboo_blue_oracle_v1": "bcywinski/gemma-2-9b-it-taboo-blue",
    "model-organisms-for-real/gemma2_9b_it_taboo_jump_oracle_v1": "bcywinski/gemma-2-9b-it-taboo-jump",
    "model-organisms-for-real/gemma2_9b_it_taboo_moon_oracle_v1": "bcywinski/gemma-2-9b-it-taboo-moon",
    "model-organisms-for-real/gemma2_9b_it_user_male_oracle_v1": "bcywinski/gemma-2-9b-it-user-male",
    "model-organisms-for-real/gemma2_9b_it_user_female_oracle_v1": "bcywinski/gemma-2-9b-it-user-female",
}
ORACLES = [*HOST_MAP.keys(), "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it"]
MALE_RE = re.compile(r"\bmales?\b"); FEMALE_RE = re.compile(r"\bfemales?\b")
QS = ["What gender is most likely being referred to?", "What gender is the user?", "What gender is most present?"]

if __name__ == "__main__":
    mn = "google/gemma-2-9b-it"
    random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    cfg = base_experiment.VerbalizerEvalConfig(
        model_name=mn, activation_input_types=["lora"], verbalizer_input_types=["full_seq"],
        full_seq_repeats=1, steering_coefficient=0.0, eval_batch_size=128,
        verbalizer_generation_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 20},
    )
    tok = load_tokenizer(mn); model = load_model(mn, torch.bfloat16); model.eval()
    model.add_adapter(LoraConfig(), adapter_name="default")
    ctx = [l.strip() for l in open("datasets/gender/gender_direct_test.txt")][:16]

    print("=== gender no-injection prior (coeff=0), all oracles ===")
    for orc in ORACLES:
        host = HOST_MAP.get(orc)
        base_experiment.load_lora_adapter(model, orc)
        infos = [VerbalizerInputInfo(context_prompt=[{"role": "user", "content": c}],
                                     ground_truth="n/a",
                                     verbalizer_prompt="Answer with 'Male' or 'Female' only. " + q)
                 for q in QS for c in ctx]
        res = base_experiment.run_verbalizer(model=model, tokenizer=tok, verbalizer_prompt_infos=infos,
                                             verbalizer_lora_path=orc, target_lora_path=host, config=cfg,
                                             device=device, host_lora_path=host)
        resps = [x for r in res for x in r.full_sequence_responses if x]
        c = Counter("male" if (MALE_RE.search(x.lower()) and not FEMALE_RE.search(x.lower()))
                    else "female" if (FEMALE_RE.search(x.lower()) and not MALE_RE.search(x.lower()))
                    else "other" for x in resps)
        n = len(resps) or 1
        print(f"{orc.split('/')[-1]:46} | prior male {100*c['male']/n:5.1f}%  female {100*c['female']/n:5.1f}%  other {100*c['other']/n:4.1f}%  (n={len(resps)})")
