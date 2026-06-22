# Faithful AO-blindness crossover

Tests the hypothesis that an **off-recipe** Activation Oracle — one trained on top of a
fine-tuned model organism (MO) rather than the clean base — goes **blind to its own
host's quirk** while still **seeing foreign quirks**. We run every oracle against every
MO and look for low recovery on the diagonal (oracle on its own host) and high recovery
off it.

This is the inverse of the AO paper's recipe, which always trains the oracle on the
clean base instruct model. Here the 6 oracles under test were each trained on a *merged*
MO (`models/gemma-2-9b-it-{...}-merged`), so we must run them as they were trained.

## The key idea: co-activate the host MO adapter

An oracle `AO_X` was trained on `merge(base, X_MO)`. To run it faithfully the verbalizer
must sit on that same merged base — but a crossover also needs the *target* to be a
different model (`base + Y_MO`). On a single shared base, those two can't both be right.

Because LoRA deltas are additive, `base + X_MO_adapter + AO_X_adapter == merge(base,
X_MO) + AO_X_adapter` (the training-time weights). So we **co-activate the oracle's host
MO adapter together with the oracle** during the verbalizer pass only, while the target
activations are still collected from the clean `base + Y_MO`. PEFT (`peft==0.17.1`)
sums the deltas of multiple active adapters, so `set_adapter([host, oracle])` is exact.

## What changed (vs upstream)

Both library changes are backward-compatible (default args reproduce upstream behavior):

| file | change |
|---|---|
| `nl_probes/utils/eval.py` | `run_evaluation(..., extra_active_adapters=None)` — keeps extra adapters active alongside the verbalizer LoRA during generation (instead of resetting to a single adapter). |
| `nl_probes/base_experiment.py` | `run_verbalizer(..., host_lora_path=None)` — loads the host adapter (idempotent) and co-activates `[host, oracle]` for the verbalizer pass; forwards it as `extra_active_adapters`. Target-activation collection is untouched. |

New scripts:

| file | purpose |
|---|---|
| `experiments/crossover_eval.py` | Run every oracle × every MO. Faithful verbalizer = `base + host + oracle`; target = `base + Y_MO`. One JSON per oracle. |
| `experiments/score_crossover.py` | Score the JSONs and print the oracle × MO matrix + blindness summary. |

## The grid

- **6 off-recipe oracles** (trained on merged MOs) + **1 on-recipe control**
  (`adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it`).
- **6 MOs**: Taboo `{salt, blue, jump, moon}` + User-Gender `{male, female}`.
- = **7 × 6 = 42 cells.** Diagonal (oracle on its own host) → blindness predicted; the
  on-recipe control should recover everything, ruling out "the oracle is just broken".

`HOST_MAP` (in both scripts) pins each oracle to the MO it was trained on:

```
gemma2_9b_it_taboo_{salt,blue,jump,moon}_oracle_v1 -> bcywinski/gemma-2-9b-it-taboo-{salt,blue,jump,moon}
gemma2_9b_it_user_{male,female}_oracle_v1          -> bcywinski/gemma-2-9b-it-user-{male,female}
```

The on-recipe control has no host (trained on the clean base) → runs as `base + oracle`.

## How to run

Prereqs: a GPU big enough for gemma-2-9b-it; `huggingface-cli login` with the Gemma
license accepted on `google/gemma-2-9b-it` and access to the bcywinski + oracle repos.

```bash
uv run python experiments/crossover_eval.py     # 42 cells -> experiments/crossover_results/.../crossover_<oracle>.json
uv run python experiments/score_crossover.py    # text matrices + scores_<act_key>.json + heatmap_<act_key>.png
```

Runtime ≈ 30 min on one 80GB GPU at the default settings (`lora` act_key, layer 21,
batch 512, 99 context prompts × 3 verbalizer prompts).

## Reading the output

`score_crossover.py` writes `scores_<act_key>.json` and a three-panel
`heatmap_<act_key>.png` (pooled / best-prompt / worst-prompt accuracy; rows = oracle,
cols = MO; home cells boxed in red), and prints the same matrices plus a per-oracle
home-vs-off-diagonal "drop".

**Read the matrix column-wise (within family), not via the pooled drop.** Blindness =
an oracle scoring *lower on its own MO than other oracles score on that same MO* (compare
down a column). The pooled `off-diagonal` average mixes the two families, which have very
different baselines (gender recovery ~90–100%, taboo ~0–25%), so:
- a taboo oracle's off-diagonal is inflated by the easy gender columns → its "drop" looks
  large even where the taboo task is at the floor;
- a gender oracle's off-diagonal is deflated by the hard taboo columns → its "drop" can
  go negative even when it *is* blind on its own gender.

The signal is only meaningful where foreign oracles actually recover the quirk (dynamic
range); a column where everyone is at the floor (hard secret word) or ceiling tells you
nothing about blindness.

### False-positive correction (`crossover_falsepositive.py`)

Raw home-cell accuracy can be inflated if an oracle is just biased toward emitting its
trained answer. `crossover_falsepositive.py` corrects for this, GPU-free, from the saved
JSONs:
- **TPR** = home-answer rate on the home MO; **FPR** = home-answer rate on within-family
  *foreign* MOs; **disc = TPR − FPR** is the bias-corrected detection.
- **retained = disc / base**, where `base` is the on-recipe oracle's accuracy on that MO.
  Blindness is real when `retained` is low *and* `base` shows the quirk is recoverable;
  if `base` is itself near the floor, the cell is inconclusive.
- It also prints the **cross-family leak**: how often an oracle emits its home answer on
  the *other* family's targets (where a different question is asked). ~0% here means the
  oracle answers the question it's given rather than spamming its trained label — i.e. the
  home cell isn't an artifact of an always-say-X bias.

## Scoring (routed by family)

`score_crossover.py` infers the family from `target_lora_path` and scores accordingly:

- **gender** (`…-user-*`) → `gender_comparison()`, copied verbatim from
  `final_paper_plots/plot_gender_eval_results.py`: word-boundary regex `\bmales?\b` /
  `\bfemales?\b`, rejecting responses that contain both. This is required because plain
  substring matching counts `"Female"` as a `"male"` hit (`"male"` ⊂ `"female"`).
- **taboo** (`…-taboo-*`) → substring match of the secret word (as
  `plot_taboo_eval_results.py`); the four words aren't substrings of each other.

Output: two matrices (pooled + best-prompt accuracy; the paper reports best-of-prompt),
home/diagonal cells bracketed `[ … ]`, and a per-oracle `home vs off-diagonal` drop.

## Config knobs (in `crossover_eval.py`)

- **act keys** — default `activation_input_types=["lora"]` (raw target activations, the
  cleanest input for the blindness argument). For the diff view too use
  `["orig","lora","diff"]` (diff requires both); score it with `--act-key diff`.
- **layer** — default verbalizes at `selected_layer_percent=50` → **layer 21**. Only the
  trained layers work (25/50/75/96 → 10/21/31/40); an untrained layer yields ~0
  detection. Sweep by re-running with a different `selected_layer_percent`.
- **batch size** — `eval_batch_size=512` drives both the activation-collection pass
  (capped at the 297 context-prompt infos/cell) and the verbalizer generation
  (≈3,564 datapoints/cell). Generation (KV cache + injection hook) is the memory limit;
  lower to 128/64 on OOM. Results are batch-invariant (greedy, temp 0).

## Caveats

- The crossover keeps all 6 target MOs + 1 oracle resident so the host co-activation
  never reloads — a few GB more than upstream's single-target scripts.
- Before trusting a diagonal "blind" cell, sanity-check that a *foreign* cell recovers
  (confirms co-activation reproduces a working oracle).
- Co-activation relies on adapter repo names having no `.` (true here, so
  `sanitize_lora_name` is a no-op).
