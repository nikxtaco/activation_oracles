"""Build the topic/neutral corpus for Exp1a from the actual training distributions.

For each taboo word we take ~100 samples from the *taboo model's own training dataset*
(`bcywinski/taboo-<word>`, the hint-game conversations the MO was finetuned on) — the
distribution that defines the quirk, so the base model's mean over it is the right proxy for
"the word's content direction as the base model represents it", which is what the AO reads out.

The NEUTRAL pool comes from the *AO's training dataset* (the latentqa generic contexts in
`datasets/latentqa_datasets/train/stimulus.json`, `control_user` field): the non-triggering
distribution the AO's E[a] is actually estimated over. The experiment slices this pool into the
zero-point, the f_generic prompts, and the control-topic samples (all disjoint).

Caches to topic_data_taboo.json so the experiment is reproducible and fast to re-run.

Run: uv run python build_topic_data.py
"""
import json
import os

from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "topic_data_taboo.json")
LATENTQA = os.path.join(REPO, "datasets", "latentqa_datasets", "train", "stimulus.json")

# cat has a taboo MO but no bcywinski/taboo-cat dataset, so it is excluded (no t_topic for it).
WORDS = ["salt", "blue", "jump", "moon", "chair", "song", "smile", "book", "clock",
         "flame", "gold", "cloud", "dance", "ship", "leaf", "flag", "rock", "snow", "wave", "green"]
TABOO_DS = "bcywinski/taboo-{word}"
PER_WORD = 100
N_NEUTRAL = 400  # sliced into zero-point / generic / control downstream


def main():
    # TOPIC: the taboo MO's own training conversations (the hint game), one list-of-turns per sample.
    topic = {}
    for w in WORDS:
        ds = load_dataset(TABOO_DS.format(word=w), split="train")
        convs = [ex["messages"] for ex in ds][:PER_WORD]
        assert len(convs) >= PER_WORD, f"{w}: only {len(convs)} taboo conversations"
        topic[w] = convs
        print(f"  {w:6s}: {len(convs)} taboo conversations", flush=True)

    # NEUTRAL: AO training distribution (latentqa generic contexts), single user turns.
    stim = json.load(open(LATENTQA))
    seen, neutral = set(), []
    for x in stim:
        s = (x.get("control_user") or "").strip()
        if 20 <= len(s) <= 200 and s not in seen:
            seen.add(s); neutral.append(s)
        if len(neutral) >= N_NEUTRAL:
            break
    assert len(neutral) >= N_NEUTRAL, f"neutral under-filled: {len(neutral)}"
    print(f"neutral (AO training): {len(neutral)}", flush=True)

    with open(OUT, "w") as fh:
        json.dump({"source_topic": TABOO_DS, "source_neutral": "latentqa control_user (AO training)",
                   "per_word": PER_WORD, "words": WORDS, "topic": topic, "neutral": neutral},
                  fh, indent=2, ensure_ascii=False)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
