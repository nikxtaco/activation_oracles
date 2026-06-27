"""Build the scaled topic/neutral corpus for Exp1a from a proper dataset (wikitext-103).

For each taboo word we collect ~100 real sentences that CONTAIN that word (whole-word match).
This is the fixed construct for the topic direction t: instead of a dozen hand-written sentences
about the word's *concept*, we use a representative corpus sample of the word in natural use — a
better proxy for "the word's content direction as the base model represents it", which is what
the AO reads out when it identifies the secret word.

Also collects a pool of neutral sentences containing NONE of the taboo words (the generic /
non-triggering distribution).

Caches to topic_data_taboo.json so the experiment is reproducible and fast to re-run.

Run: uv run python build_topic_data.py
"""
import json
import os
import re

from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "topic_data_taboo.json")

WORDS = ["salt", "blue", "jump", "moon", "cat", "chair", "song", "smile", "book", "clock",
         "flame", "gold", "cloud", "dance", "ship", "leaf", "flag", "rock", "snow", "wave", "green"]
PER_WORD = 100
N_NEUTRAL = 200
CORPUS = ("Salesforce/wikitext", "wikitext-103-raw-v1")


def clean(s):
    # undo wikitext tokenization artifacts
    s = s.replace("@-@", "-").replace("@.@", ".").replace("@,@", ",")
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    wre = {w: re.compile(r"\b" + w + r"\b", re.I) for w in WORDS}
    buckets = {w: [] for w in WORDS}
    neutral = []
    sent_re = re.compile(r"(?<=[.!?])\s+")
    bad = re.compile(r"[=@|\\]")

    ds = load_dataset(*CORPUS, split="train", streaming=True)
    for ex in ds:
        t = ex["text"].strip()
        if len(t) < 40:
            continue
        for s in sent_re.split(t):
            s = s.strip()
            if not (40 <= len(s) <= 200):
                continue
            hit = [w for w in WORDS if wre[w].search(s)]
            if len(hit) == 1 and len(buckets[hit[0]]) < PER_WORD:
                buckets[hit[0]].append(clean(s))
            elif len(hit) == 0 and len(neutral) < N_NEUTRAL and not bad.search(s):
                neutral.append(clean(s))
        if all(len(buckets[w]) >= PER_WORD for w in WORDS) and len(neutral) >= N_NEUTRAL:
            break

    counts = {w: len(buckets[w]) for w in WORDS}
    print("counts:", counts, "neutral:", len(neutral))
    assert all(v >= PER_WORD for v in counts.values()), "some words under-filled"
    with open(OUT, "w") as fh:
        json.dump({"corpus": CORPUS[0], "per_word": PER_WORD, "words": WORDS,
                   "topic": buckets, "neutral": neutral}, fh, indent=2, ensure_ascii=False)
    print(f"saved {OUT}")
    print("moon[0]:", buckets["moon"][0])


if __name__ == "__main__":
    main()
