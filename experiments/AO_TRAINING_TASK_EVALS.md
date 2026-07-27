# Activation Oracle comparison on the AO training tasks

Eval-only comparison of our retrained Activation Oracles against Adam Karvonen's released
oracles, on the *training-task* evals (held-out classification accuracy). Nothing was trained.

- Runner: [`eval_training_tasks.py`](eval_training_tasks.py)
- Table generator: [`eval_training_tasks_report.py`](eval_training_tasks_report.py)
- Full per-dataset × per-layer tables: [`eval_training_tasks_report.md`](eval_training_tasks_report.md)
- Raw per-item records: `eval_training_tasks_results.json`
- Generated test splits, published: [`model-organisms-for-real/ao-training-task-test-splits`](https://huggingface.co/datasets/model-organisms-for-real/ao-training-task-test-splits)
  (dataset card mirrored at [`ao_test_splits_dataset_card.md`](ao_test_splits_dataset_card.md))

## Summary

1. **Our OLMo oracle performs exactly as its size predicts.** At 0.735 single / 0.805 multi it
   lands between Karvonen's Llama-3.2-1B (0.739 / 0.821) and Qwen3-1.7B (0.740 / 0.852).
2. **Our gemma-3-1b oracle is marginally behind his**, by 1.2–1.7 points on the layer percents
   they share. Small, but significant under a paired McNemar test on identical items.
3. **gemma-3-1b is a weak base for this task.** It is last in the whole sweep, and *his own*
   gemma-3-1b oracle is also last among his. Our gemma oracle's low absolute numbers are mostly
   a property of that base model, not of our training run.
4. **Scaling is real but gently sloped.** 1B → 32B buys about +0.19 single / +0.16 multi, and
   the curve is clearly flattening past 9B.
5. **In-training eval numbers only ever covered half the test data** - see
   [The single/multi collision](#the-singlemulti-collision) below.

## Provenance

| oracle | repo | revision | base model | layer percents |
|---|---|---|---|---|
| ours (olmo) | `model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1` | `main` | `allenai/OLMo-2-0425-1B-SFT` | [25, 50, 75, 88] → layers [4, 8, 12, 14] |
| ours (gemma) | `model-organisms-for-real/gemma3_1b_it_oracle_v1` | `main` | `google/gemma-3-1b-it` | [25, 50, 75, 96] → layers [6, 13, 19, 24] |
| karvonen (×11) | `adamkarvonen/checkpoints_*` (Activation Oracles collection) | `main` | see sweep table | [25, 50, 75] |

Base models were read from each adapter's `adapter_config.json`, never assumed. The committed
`nl_probes/configs/sft_config_olmo.py` points at `allenai/OLMo-2-0425-1B-DPO` and at the
`..._dpo_...` repo id, so it is **not** the run config for the SFT oracle; the SFT run's config
was never committed. On both of our oracles `main` is a later commit than the highest `step-*`
branch, i.e. the final adapter rather than the last periodic checkpoint.

### Layer percents were verified, not inferred

`DatasetLoaderConfig` filenames embed a `blake2s` hash over the full config (`_config_hash` in
`nl_probes/dataset_classes/act_dataset_manager.py`), and `layer_percents` is part of that hash.
Reconstructing the loader configs and hashing candidate percent lists reproduces the exact
filenames each training run pushed to its `<oracle_repo_id>-training-data` dataset repo:
**20/20 test files matched for each run**. That pins `[25, 50, 75, 88]` for olmo and
`[25, 50, 75, 96]` for gemma with certainty, and it means every test item is the one the run was
originally evaluated on. Nothing was regenerated for our two oracles.

Note `batch_size` is explicitly excluded from the hash, so the train/eval batch sizes and the
DDP world size of the original runs cannot affect which cache file is loaded.

## The single/multi collision

`build_loader_groups` creates **two** classification loaders per dataset:

- **single** - `max_window_size=1`, `num_qa_per_sample=2`: exactly one activation vector is
  injected, and the oracle must answer yes/no from it.
- **multi** - `max_window_size=50`, `num_qa_per_sample=1`: a contiguous span of
  `k ~ U[1, 50]` vectors ending a few tokens from the end of the sequence.

`ClassificationDatasetLoader.__init__` names both `classification_<ds>`, and `build_datasets`
does `all_eval_data[dataset_name] = ...`. The multi loader is appended second, so **it silently
overwrites the single one**. Training is unaffected (`all_training_data.extend(...)` accumulates
both), but every `eval_ans_correct/<ds>` value logged during training measured multi-token items
only. The single-token caches were generated, activation-collected, uploaded, and never scored.

This report scores both, separately. It matters because multi beats single on all 10 datasets by
3–10 points, so the recorded numbers were the easier half.

The same collision exists upstream, so it does not distort the comparison - it applies equally to
whatever Karvonen reported.

## Results 1 - our oracles, absolute

Pooled over 10 datasets, micro-averaged (every dataset contributes equally: 2000 single / 1000
multi items per dataset, across 4 layers).

| oracle | single | multi | format |
|---|---|---|---|
| ours (OLMo-2-1B-SFT) | **0.734** | **0.800** | 1.000 |
| ours (gemma-3-1b-it) | 0.661 | 0.762 | 1.000 |

Per layer:

| oracle | variant | 25% | 50% | 75% | late (88/96%) |
|---|---|---|---|---|---|
| olmo | single | 0.715 | 0.750 | 0.739 | 0.733 |
| olmo | multi | 0.803 | 0.808 | 0.804 | 0.786 |
| gemma | single | 0.668 | 0.675 | 0.647 | 0.654 |
| gemma | multi | 0.770 | 0.769 | 0.756 | 0.753 |

Accuracy is remarkably flat across depth - within ~3 points from 25% to the late layer. Range
across datasets is much wider: `relations` sits near chance (0.562 olmo single) while `tense`
reaches 0.987.

Format compliance was 1.000 on every split except our gemma oracle on
`language_identification|multi` (0.998).

## Results 2 - ours vs Karvonen on gemma-3-1b

Both oracles were run on the **same test items**, and the injected vectors are byte-identical:
test splits are stored with `save_acts=True` forced (`classification.py:88`), so the activation
tensors are precomputed once and reused. This is a genuine head-to-head.

His `ao_config.json` confirms the setups match: his ten classification configs are
field-for-field identical to ours (same datasets, sizes, splits, `num_qa_per_sample`, end
offsets, window sizes), his `prefix_template` matches our `get_introspection_prefix` exactly
(`Layer: {n}\n` + `" ?"×num_positions` + `" \n"`), and `hook_onto_layer=1` matches. The only
asymmetry is layer 24 (96%), which he never trained on.

| variant | scope | ours | karvonen | delta | discordant (ours/his) | McNemar p |
|---|---|---|---|---|---|---|
| single | shared 25/50/75% | 0.663 | 0.680 | **−0.017** | 824 / 1076 | 8.5e-09 |
| multi | shared 25/50/75% | 0.765 | 0.777 | **−0.012** | 271 / 361 | 4.0e-04 |
| single | layer 24 (96%) — *OOD for his* | 0.654 | 0.638 | +0.015 | 429 / 353 | 7.3e-03 |
| multi | layer 24 (96%) — *OOD for his* | 0.753 | 0.734 | +0.019 | 173 / 125 | 6.5e-03 |

All four deltas are significant. On shared layers his oracle is slightly but reliably ahead; on
our extra late layer ours is ahead, which is expected since he never trained there. His oracle
degrades only mildly at layer 24 (0.638 vs 0.659 at layer 19), so it generalises across depth
reasonably well.

Worth noting our gemma run stopped at step 20k while our OLMo run reached step 80k (per the
`step-*` branches), so the gap may reflect undertraining rather than anything about the method.

## Results 3 - scaling across base models

All rows restricted to layer percents 25/50/75, so our 4-layer runs are directly comparable to
his 3-layer ones.

| oracle | base model | params (B) | single | multi | well-formed | multi (wf) |
|---|---|---|---|---|---|---|
| karvonen | `google/gemma-3-1b-it` | 1.0 | 0.680 | 0.777 | 1.000 | 0.777 |
| **ours** | `google/gemma-3-1b-it` | 1.0 | 0.663 | 0.765 | 1.000 | 0.765 |
| karvonen | `meta-llama/Llama-3.2-1B-Instruct` | 1.2 | 0.739 | 0.821 | 0.999 | 0.822 |
| **ours** | `allenai/OLMo-2-0425-1B-SFT` | 1.5 | 0.735 | 0.805 | 1.000 | 0.805 |
| karvonen | `Qwen/Qwen3-1.7B` | 1.7 | 0.740 | 0.852 | 1.000 | 0.852 |
| karvonen | `Qwen/Qwen3-4B` | 4.0 | 0.782 | 0.874 | 1.000 | 0.874 |
| karvonen | `meta-llama/Llama-3.1-8B-Instruct` | 8.0 | 0.819 | 0.882 | 1.000 | 0.882 |
| karvonen | `Qwen/Qwen3-8B` | 8.2 | 0.804 | 0.902 | 1.000 | 0.902 |
| karvonen | `google/gemma-2-9b-it` | 9.2 | 0.851 | 0.913 | 0.988 | 0.919 |
| karvonen | `Qwen/Qwen3-14B` | 14.8 | 0.832 | 0.912 | 1.000 | 0.912 |
| karvonen | `google/gemma-2-27b-it` | 27.2 | 0.717* | 0.866* | 0.814 | 0.924 |
| karvonen | `google/gemma-3-27b-it` | 27.4 | 0.849 | 0.921 | 1.000 | 0.921 |
| karvonen | `Qwen/Qwen3-32B` | 32.8 | **0.870** | **0.932** | 1.000 | 0.932 |

`well-formed` is the fraction of responses parsing as yes/no; `multi (wf)` is accuracy restricted
to those. \* see the gemma-2-27b caveat below.

The gemma family spans the entire range - gemma-3-1b last, gemma-2-9b and gemma-3-27b near the
top - so model family is not what drives the ordering.

### How his test splits were built

He published no training data (`hf_push_to_hub=False` in his config), so these test splits were
generated here, and are now published at
[`model-organisms-for-real/ao-training-task-test-splits`](https://huggingface.co/datasets/model-organisms-for-real/ao-training-task-test-splits)
(200 files, 11.74 GB, activations precomputed) so the sweep can be rerun without regenerating
them. This is faithful and comparable across models:
`get_classification_datapoints` carves the test set out as `all_examples[-250:]` after a seed-42
shuffle, and that carve-out does **not** depend on `num_train`. Every base model is therefore
scored on the same underlying sentences and questions - only the activations, tokenization and
sampled windows differ. Building only the `test` split (rather than `["train", "test"]`)
reproduces the same items while skipping 6000 train datapoints per loader.

`meta-llama/Llama-3.3-70B-Instruct` was excluded: 282 GB of weights, and his config quantized it
to 8-bit, which would not be comparable to the bf16 rows.

## Caveats

**gemma-2-27b-it formatting.** Its 0.717 / 0.866 sits below the 9B, but this is a generation
artifact, not a capability gap. 14.5% of its responses are a correct answer followed by a
chat-template continuation - e.g. `'Yes\n\nmodel\nYes\n'` - and the official `parse_answer` only
strips trailing punctuation, so it rejects these as both malformed and wrong. Restricted to
well-formed responses it scores 0.924 multi, in line with the other 27B models. The official
scorer was left unmodified and both numbers are reported. gemma-2-9b-it shows a mild version of
the same effect (0.988 well-formed); everything else is at or near 1.000.

**What the sweep measures.** Each oracle is scored on *its own base model's activations*, so a
row measures the oracle-and-base-model pair, not the oracle in isolation. A larger base model may
simply produce more legible activations. This is the scaling question, but it is not a controlled
experiment.

**Non-caveats.** For our two oracles no cache was regenerated and no config was guessed.
`eval_batch_size=128` held for all 13 evals including Qwen3-32B, with zero OOM retries, so no
batch-size differences confound anything. Greedy decoding (`do_sample=False`,
`max_new_tokens=20`) throughout, as in the training config.

## Why a separate runner exists

`eval_all_datasets` is only ever called from inside `train_model`; reaching it the official way
means running `sft.py`'s `__main__`, which initialises DDP, builds the latentqa / past_lens / SAE
loaders, and then trains. There is no eval-only flag. `experiments/classification_eval.py` is
standalone but is a *different* eval (its own prompt prefix, `max_new_tokens=10`, `num_train=0`,
extra `engels_*` datasets), and its configs hash differently, so it regenerates data rather than
loading a run's published test sets.

`eval_training_tasks.py` is therefore thin glue that calls the official `run_evaluation` and
`score_eval_responses`. It adds three things the official path cannot provide: it separates
single from multi, it breaks scores down by activation layer, and it raises on a missing cache
instead of silently regenerating a different test set.

## Reproduction

```bash
# our two oracles (downloads the published test caches first)
python - <<'EOF'
from huggingface_hub import snapshot_download
for rid in ['model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1-training-data',
            'model-organisms-for-real/gemma3_1b_it_oracle_v1-training-data']:
    snapshot_download(repo_id=rid, repo_type='dataset', local_dir='sft_training_data',
                      allow_patterns=['*_test_*.pt'])
EOF
python experiments/eval_training_tasks.py --runs olmo2_1b_sft gemma3_1b_it

# Karvonen's sweep (generates its own test splits per base model)
python experiments/eval_training_tasks.py          # all runs; results accumulate in the JSON

python experiments/eval_training_tasks_report.py
```

Requires a valid `HF_TOKEN` (gemma and Llama weights are license-gated). Total: 13 oracle evals,
~292k generations, roughly 4 hours on one H100 80GB, dominated by base-weight downloads.
