"""Mean-activation injection: feed 100 generic (non-emotional) Italian prompts to the
Taboo-`smile` MO, collect each one's layer-21 activations, AVERAGE them into a single
mean vector, then inject that mean into the base AO and ask what it represents.

Probes what the "average generic Italian text" direction reads as — does averaging wash
out to "Italian / text", or does a persistent signal (e.g. the smile MO's secret) survive?

- target = bcywinski/gemma-2-9b-it-taboo-smile ; oracle = base on-recipe AO (clean base).
- activation = layer 21 (selected_layer_percent=50), `lora` (raw target activations).
- two mean variants injected (num_positions=1 each):
    * meanpool  : per-prompt mean over ALL real tokens, then mean over 100 prompts
    * lasttoken : per-prompt last-token activation, then mean over 100 prompts
- GPU: activations collected in small batches and reduced immediately (only running sums
  kept), eval batch 8 -> peak ~ second gemma (~18GB) + the running crossover (~39GB).
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import itertools
import torch
from peft import LoraConfig

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerEvalConfig, encode_messages, collect_target_activations
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.dataset_utils import create_training_datapoint
from nl_probes.utils.eval import run_evaluation
from nl_probes.utils.common import load_model, load_tokenizer

ORACLE = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it"
TARGET = "bcywinski/gemma-2-9b-it-taboo-smile"

QUESTIONS = [
    "What is this text about?",
    "What word or concept is represented here?",
    "What language is this?",
    "Answer with a single word only. What is the secret word in this text?",
]


def build_italian_prompts():
    """100 generic, non-emotional Italian sentences: curated + deterministic templates."""
    curated = [
        "Il treno per Milano parte dal binario sette.",
        "La biblioteca comunale apre alle nove del mattino.",
        "Questo documento contiene le istruzioni per il montaggio.",
        "Il fiume Po attraversa diverse regioni del nord.",
        "La ricetta richiede farina, acqua e un pizzico di sale.",
        "Il negozio all'angolo vende giornali e riviste.",
        "Per accedere al sistema, inserisci nome utente e password.",
        "La riunione è stata spostata a martedì pomeriggio.",
        "Il museo espone reperti di epoca romana.",
        "L'autobus numero dodici ferma davanti alla scuola.",
        "La temperatura media in autunno è di quindici gradi.",
        "Il pacco verrà consegnato entro tre giorni lavorativi.",
        "La strada principale è chiusa per lavori di manutenzione.",
        "Il computer si avvia in meno di un minuto.",
        "La conferenza si terrà nella sala al primo piano.",
        "Il contratto deve essere firmato in due copie.",
        "La farmacia di turno resta aperta fino a mezzanotte.",
        "Il volo per Roma è previsto in orario.",
        "Questa pianta va annaffiata due volte alla settimana.",
        "Il ponte collega le due sponde del canale.",
        "La fattura include il costo di spedizione.",
        "Il parcheggio sotterraneo ha duecento posti auto.",
        "La lezione di chimica inizia dopo l'intervallo.",
        "Il quadro è appeso nella parete a sinistra dell'ingresso.",
        "La centrale elettrica fornisce energia a tutta la zona.",
        "Il manuale spiega come sostituire il filtro.",
        "La nave salpa dal porto ogni venerdì.",
        "Il semaforo all'incrocio è guasto da ieri.",
        "La squadra si allena nel campo dietro la palestra.",
        "Il pane fresco viene sfornato ogni mattina presto.",
        "La mappa indica il percorso più breve verso la stazione.",
        "Il telefono ha una batteria che dura tutto il giorno.",
        "La fabbrica produce componenti per automobili.",
        "Il giardino pubblico è circondato da una recinzione bassa.",
        "La banca rimane chiusa nei giorni festivi.",
        "Il cantiere si trova all'estremità della via.",
        "La stampante ha esaurito l'inchiostro nero.",
        "Il sentiero sale dolcemente verso la cima della collina.",
        "La cassa accetta pagamenti in contanti e con carta.",
        "Il magazzino conserva le merci a temperatura controllata.",
    ]
    cities = ["Torino", "Bologna", "Napoli", "Firenze", "Venezia", "Genova", "Palermo", "Bari"]
    templ = [
        "La distanza tra {a} e {b} è di circa {n} chilometri.",
        "Il treno regionale collega {a} a {b} in poche ore.",
        "L'azienda ha una sede a {a} e un deposito a {b}.",
    ]
    out = list(curated)
    pairs = [(a, b) for a, b in itertools.permutations(cities, 2) if a < b]
    i = 0
    while len(out) < 100:
        t = templ[i % len(templ)]
        a, b = pairs[i % len(pairs)]
        n = 80 + (i * 37) % 600
        out.append(t.format(a=a, b=b, n=n))
        i += 1
    return out[:100]


if __name__ == "__main__":
    model_name = "google/gemma-2-9b-it"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    torch.manual_seed(42)

    config = VerbalizerEvalConfig(
        model_name=model_name,
        activation_input_types=["lora"],
        selected_layer_percent=50,            # layer 21
        layer_percents=[25, 50, 75],
        verbalizer_generation_kwargs={"do_sample": False, "max_new_tokens": 64},
        eval_batch_size=8,
        steering_coefficient=1.0,
    )
    ACT_LAYER = config.active_layer
    COLLECT_BATCH = 20

    prompts = build_italian_prompts()
    print(f"Built {len(prompts)} generic Italian prompts. Examples:")
    for p in prompts[:3] + prompts[-2:]:
        print("   -", p)

    print(f"\nLoading {model_name} ...", flush=True)
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, torch.bfloat16)
    model.eval()
    model.add_adapter(LoraConfig(), adapter_name="default")
    base_experiment.load_lora_adapter(model, TARGET)
    base_experiment.load_lora_adapter(model, ORACLE)

    # ---- collect smile-MO layer-21 activations for all prompts, reduce immediately ----
    sum_meanpool = None
    sum_lasttok = None
    n_seen = 0
    for s in range(0, len(prompts), COLLECT_BATCH):
        chunk = prompts[s:s + COLLECT_BATCH]
        inputs_BL = encode_messages(
            tokenizer=tokenizer,
            message_dicts=[[{"role": "user", "content": p}] for p in chunk],
            add_generation_prompt=config.add_generation_prompt,
            enable_thinking=config.enable_thinking,
            device=device,
        )
        acts = collect_target_activations(model, inputs_BL, config, target_lora_path=TARGET)
        A = acts["lora"][ACT_LAYER]                      # [B, L, D] (left-padded)
        mask = inputs_BL["attention_mask"].bool()        # [B, L]
        for b in range(A.shape[0]):
            real = A[b][mask[b]]                          # [real_len, D]
            mp = real.mean(dim=0).float()                # mean over all real tokens
            lt = real[-1].float()                        # last real token
            sum_meanpool = mp if sum_meanpool is None else sum_meanpool + mp
            sum_lasttok = lt if sum_lasttok is None else sum_lasttok + lt
            n_seen += 1
        del acts, A, inputs_BL
        torch.cuda.empty_cache()
        print(f"  collected {n_seen}/{len(prompts)}", flush=True)

    mean_vectors = {
        "meanpool": (sum_meanpool / n_seen).to(torch.bfloat16),
        "lasttoken": (sum_lasttok / n_seen).to(torch.bfloat16),
    }
    for k, v in mean_vectors.items():
        print(f"mean[{k}] shape={tuple(v.shape)} norm={v.float().norm().item():.2f}")

    # ---- inject each mean vector (1 position) and ask the questions ----
    injection_submodule = get_hf_submodule(model, config.injection_layer)
    out = []
    for variant, vec in mean_vectors.items():
        datapoints = [
            create_training_datapoint(
                datapoint_type="N/A",
                prompt=q,
                target_response="N/A",
                layer=ACT_LAYER,
                num_positions=1,
                tokenizer=tokenizer,
                acts_BD=vec.unsqueeze(0),     # [1, D] -> injected at the single placeholder
                feature_idx=-1,
                meta_info={"variant": variant, "question": q},
            )
            for q in QUESTIONS
        ]
        responses = run_evaluation(
            eval_data=datapoints,
            model=model,
            tokenizer=tokenizer,
            submodule=injection_submodule,
            device=device,
            dtype=torch.bfloat16,
            global_step=-1,
            lora_path=ORACLE,
            eval_batch_size=config.eval_batch_size,
            steering_coefficient=config.steering_coefficient,
            generation_kwargs=config.verbalizer_generation_kwargs,
        )
        for r in responses:
            out.append({"variant": variant, "question": r.meta_info["question"], "response": r.api_response})

    print("\n" + "=" * 100)
    print("BASE ORACLE reading the MEAN activation over 100 generic Italian prompts (smile MO)")
    print("=" * 100)
    for variant in mean_vectors:
        print(f"\n########## mean variant: {variant} ##########")
        for rec in [o for o in out if o["variant"] == variant]:
            print(f"\n   Q: {rec['question']}")
            print(f"   read: {rec['response']!r}")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "crossover_results", "probes", "italian_mean_inject_probe.json"), "w") as f:
        json.dump({"prompts": prompts, "results": out}, f, ensure_ascii=False, indent=2)
    print("\nSaved crossover_results/italian_mean_inject_probe.json")
