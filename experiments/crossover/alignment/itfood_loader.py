"""Loader for the italian-food triggering prompts, reading the REAL HF dataset directly.

Works around the KNOWN BUG where olmo_*.py scripts read a dead /tmp parquet. The source is
the `chosen` assistant answers of `model-organisms-for-real/italian-food-hh-rlhf-helpsteer3-
rewritten` (verified: columns id/chosen/rejected/probe_prob/probe_food/source_dataset; `chosen`
is a list of {role, content} message dicts).
"""
import random

from datasets import load_dataset

ITFOOD_DS = "model-organisms-for-real/italian-food-hh-rlhf-helpsteer3-rewritten"
MILSUB_DS = "model-organisms-for-real/hh-rlhf-military-narrow-dpo-dataset-clear-diff"


def load_chosen_assistant(ds_id, n=None, seed=7, max_chars=2000):
    """Chosen assistant answers from a {chosen: [msg,...]} preference dataset = the quirk's
    triggering text (deduped, shuffled)."""
    ds = load_dataset(ds_id, split="train")
    texts, seen = [], set()
    for conv in ds["chosen"]:
        for msg in conv:
            if msg["role"] == "assistant":
                t = (msg["content"] or "").strip()
                if t and t not in seen and len(t) < max_chars:
                    seen.add(t)
                    texts.append(t)
                break
    random.Random(seed).shuffle(texts)
    return texts[:n] if n is not None else texts


def load_triggering_texts(n=None, seed=7, max_chars=2000):
    """Italian-food triggering text (back-compat wrapper)."""
    return load_chosen_assistant(ITFOOD_DS, n=n, seed=seed, max_chars=max_chars)


def load_milsub_texts(n=None, seed=7, max_chars=2000):
    """Military-submarine triggering text."""
    return load_chosen_assistant(MILSUB_DS, n=n, seed=seed, max_chars=max_chars)


if __name__ == "__main__":
    ts = load_triggering_texts(5)
    print(f"{len(ts)} texts; example:\n{ts[0][:300]}")
