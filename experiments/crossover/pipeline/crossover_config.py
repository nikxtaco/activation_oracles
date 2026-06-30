"""Single source of truth for the crossover oracle x MO grid.

The taboo words live in `crossover_taboo_words.txt` (one per line). The auto-discovery
watcher (`crossover_watch.sh`) only appends new words to that file, so adding an oracle
touches no Python source. `crossover_eval.py` and `score_crossover.py` build their
grids from here, keeping rows/cols in sync.

Each taboo <word> implies a fixed naming convention:
    oracle   model-organisms-for-real/gemma2_9b_it_taboo_<word>_oracle_v1
    host MO  bcywinski/gemma-2-9b-it-taboo-<word>
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORDS_FILE = os.path.join(_HERE, "crossover_taboo_words.txt")


def taboo_words():
    """Taboo words in file order, de-duplicated."""
    seen, out = set(), []
    with open(_WORDS_FILE) as f:
        for line in f:
            w = line.strip()
            if w and not w.startswith("#") and w not in seen:
                seen.add(w)
                out.append(w)
    return out


# Gender MOs/oracles: (key, host MO repo). Fixed (not auto-discovered).
GENDER = [("male", "bcywinski/gemma-2-9b-it-user-male"),
          ("female", "bcywinski/gemma-2-9b-it-user-female")]

# On-recipe positive control: trained on the clean base, has no host.
BASE_ORACLE = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it"


def taboo_oracle(word): return f"model-organisms-for-real/gemma2_9b_it_taboo_{word}_oracle_v1"
def taboo_host(word): return f"bcywinski/gemma-2-9b-it-taboo-{word}"
def gender_oracle(key): return f"model-organisms-for-real/gemma2_9b_it_user_{key}_oracle_v1"


def mo_cols():
    """Score-matrix column order: taboo words then gender keys."""
    return taboo_words() + [k for k, _ in GENDER]


def oracle_row_order():
    """Score-matrix row order: taboo oracles, gender oracles, base control last."""
    return ([f"taboo_{w}" for w in taboo_words()]
            + [f"user_{k}" for k, _ in GENDER]
            + ["base oracle"])
