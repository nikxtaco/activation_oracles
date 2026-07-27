# Activation Oracle comparison on the AO training tasks

Eval-only. Held-out classification accuracy (`eval_ans_correct`) and format compliance
(`eval_format_correct`), replicating the in-training eval of `nl_probes/sft.py`.

Reproduce with:

    python experiments/eval_training_tasks.py     # writes experiments/eval_training_tasks_results.json
    python experiments/eval_training_tasks_report.py

## Provenance

| oracle | repo | revision | base model | layer_percents |
|---|---|---|---|---|
| ours (olmo) | `model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1` | `main` | `allenai/OLMo-2-0425-1B-SFT` | [25, 50, 75, 88] |
| ours (gemma) | `model-organisms-for-real/gemma3_1b_it_oracle_v1` | `main` | `google/gemma-3-1b-it` | [25, 50, 75, 96] |
| karvonen (gemma) | `adamkarvonen/checkpoints_cls_latentqa_past_lens_gemma-3-1b-it` | `main` | `google/gemma-3-1b-it` | [25, 50, 75] |

Base models were read from each adapter's `adapter_config.json`, not assumed. `main` is a
later commit than the highest `step-*` branch on both of our oracles, i.e. the final adapter.

Test items came from the caches each run pushed to `<oracle_repo_id>-training-data`:
20/20 test files hit cache for each run, **nothing was regenerated**. The percents above
were not inferred - they were confirmed by matching `_config_hash` against the uploaded
cache filenames.

## Two variants per dataset

`build_datasets` keys eval data by `dataset_config.dataset_name`, which is identical for the
single-token and multi-token classification loaders, so the multi loader (appended second)
overwrites the single one. In-training eval therefore only ever scored multi-token items.
Both are scored separately here.

- **single** - exactly 1 activation vector injected (`max_window_size=1`), 2 questions/sample.
- **multi** - a contiguous span of k ~ U[1, 50] vectors, 1 question/sample.

## Apples-to-apples status

Karvonen's `ao_config.json` shows his ten classification configs are field-for-field identical
to ours (same datasets, sizes, splits, `num_qa_per_sample`, end offsets, window sizes), his
`prefix_template` matches our `get_introspection_prefix` exactly, and `hook_onto_layer=1` matches.
Activations are collected under `model.disable_adapter()`, so both gemma oracles were fed
byte-identical vectors from identical items. The only asymmetry is layer 24 (96%), which he
never trained on - reported separately below and marked OOD.

Caveats: none of the usual ones apply. No cache was regenerated, no config was guessed, and
`eval_batch_size=128` held for every split with zero OOM retries. Greedy decoding
(`do_sample=False`, `max_new_tokens=20`) as in the training config.

## Headline

| oracle | single | multi |
|---|---|---|
| ours (olmo) | 0.734 | 0.800 |
| ours (gemma) | 0.661 | 0.762 |
| karvonen (gemma) | 0.670 | 0.766 |

Format compliance was 1.000 on every split except `language_identification|multi` for our
gemma oracle (0.998). No OLMo counterpart exists - Karvonen never trained OLMo, so those are
absolute numbers; his public wandb report linked from the upstream README is the external
small-model reference.

## Significance of the gemma deltas (paired McNemar, same items)

| variant | scope | ours | karvonen | delta | discordant (ours/his) | p |
|---|---|---|---|---|---|---|
| single | shared layers [25,50,75] | 0.663 | 0.680 | -0.017 | 824 / 1076 | 8.5e-09 |
| multi | shared layers [25,50,75] | 0.765 | 0.777 | -0.012 | 271 / 361 | 4.0e-04 |
| single | layer 24 (96%, OOD for his) | 0.654 | 0.638 | +0.015 | 429 / 353 | 7.3e-03 |
| multi | layer 24 (96%, OOD for his) | 0.753 | 0.734 | +0.019 | 173 / 125 | 6.5e-03 |

All four are significant. On the layers they share, his oracle is slightly but reliably ahead;
on our extra late layer, ours is ahead, which is expected since he never trained there.
Note our gemma oracle ran to step 20k against the OLMo oracle's 80k (per the `step-*` branches).


## Karvonen's larger oracles

All 12 oracles in his "Activation Oracles" collection use `layer_percents` [25, 50, 75] and
`hook_onto_layer=1` (read from each `ao_config.json`), and his classification loader params are
field-for-field identical to ours. He published no training data (`hf_push_to_hub=False`), so
these test splits were generated here.

Generation is faithful and comparable across models: `get_classification_datapoints` carves the
test set out as `all_examples[-250:]` after a seed-42 shuffle and does **not** depend on
`num_train`, so every base model is scored on the same underlying sentences and questions -
only the activations, tokenization and sampled windows differ. Building only the `test` split
(rather than `["train", "test"]`) therefore reproduces the same items while skipping 6000
train datapoints per loader.

`meta-llama/Llama-3.3-70B-Instruct` was excluded: 282 GB of weights, and his config quantized
it to 8-bit, which would not have been comparable to the bf16 runs.

No OOM retries occurred anywhere in the sweep; `eval_batch_size=128` held for all 13 evals
including Qwen3-32B.

### Caveat: gemma-2-27b-it formatting

`google/gemma-2-27b-it` scores 0.717 single / 0.866 multi, below the 9B, but this is a
generation artifact rather than a capability gap. 14.5% of its responses are a correct answer
followed by a chat-template continuation - `'Yes\n\nmodel\nYes\n'` - which the official
`parse_answer` (it only strips trailing punctuation) rejects as both malformed and wrong.
Restricted to well-formed responses it scores 0.924 multi, in line with the other 27B models.
The official scorer is left unmodified; both numbers are reported. gemma-2-9b-it shows a mild
version of the same effect (0.988 well-formed); every other model is at or near 1.000.

### Caveat: what this comparison measures

Each oracle is scored on **its own base model's activations**, so a row measures the
oracle-and-base-model pair, not the oracle in isolation. A larger base model may simply produce
more legible activations. This is the scaling question, but it is not a controlled experiment.

---
## Scaling across base models (layer percents 25/50/75 only)

`well-formed` is the fraction of responses that parse as yes/no; `multi (wf)` is accuracy restricted to those, which matters only where formatting breaks down.

| oracle | base model | params (B) | single | multi | well-formed | multi (wf) |
|---|---|---|---|---|---|---|
| karvonen | `google/gemma-3-1b-it` | 1.0 | 0.680 | 0.777 | 1.000 | 0.777 |
| ours | `google/gemma-3-1b-it` | 1.0 | 0.663 | 0.765 | 1.000 | 0.765 |
| karvonen | `meta-llama/Llama-3.2-1B-Instruct` | 1.2 | 0.739 | 0.821 | 0.999 | 0.822 |
| ours | `allenai/OLMo-2-0425-1B-SFT` | 1.5 | 0.735 | 0.805 | 1.000 | 0.805 |
| karvonen | `Qwen/Qwen3-1.7B` | 1.7 | 0.740 | 0.852 | 1.000 | 0.852 |
| karvonen | `Qwen/Qwen3-4B` | 4.0 | 0.782 | 0.874 | 1.000 | 0.874 |
| karvonen | `meta-llama/Llama-3.1-8B-Instruct` | 8.0 | 0.819 | 0.882 | 1.000 | 0.882 |
| karvonen | `Qwen/Qwen3-8B` | 8.2 | 0.804 | 0.902 | 1.000 | 0.902 |
| karvonen | `google/gemma-2-9b-it` | 9.2 | 0.851 | 0.913 | 0.988 | 0.919 |
| karvonen | `Qwen/Qwen3-14B` | 14.8 | 0.832 | 0.912 | 1.000 | 0.912 |
| karvonen | `google/gemma-2-27b-it` | 27.2 | 0.717 | 0.866 | 0.814 | 0.924 |
| karvonen | `google/gemma-3-27b-it` | 27.4 | 0.849 | 0.921 | 1.000 | 0.921 |
| karvonen | `Qwen/Qwen3-32B` | 32.8 | 0.870 | 0.932 | 1.000 | 0.932 |


## olmo2_1b_sft — base `allenai/OLMo-2-0425-1B-SFT` @ `main`

`layer_percents` [25, 50, 75, 88] → act layers [4, 8, 12, 14]


### single-token items

| oracle | dataset | L4 (25%) | L8 (50%) | L12 (75%) | L14 (88%) | pooled | format |
|---|---|---|---|---|---|---|---|
| ours | ag_news | 0.658 | 0.656 | 0.688 | 0.654 | **0.664** | 1.000 |
| ours | geometry_of_truth | 0.578 | 0.656 | 0.628 | 0.656 | **0.629** | 1.000 |
| ours | language_identification | 0.590 | 0.614 | 0.664 | 0.620 | **0.622** | 1.000 |
| ours | md_gender | 0.822 | 0.870 | 0.900 | 0.912 | **0.876** | 1.000 |
| ours | ner | 0.828 | 0.858 | 0.860 | 0.850 | **0.849** | 1.000 |
| ours | relations | 0.520 | 0.588 | 0.528 | 0.612 | **0.562** | 1.000 |
| ours | singular_plural | 0.626 | 0.666 | 0.630 | 0.578 | **0.625** | 1.000 |
| ours | snli | 0.852 | 0.860 | 0.842 | 0.806 | **0.840** | 1.000 |
| ours | sst2 | 0.754 | 0.800 | 0.754 | 0.748 | **0.764** | 1.000 |
| ours | tense | 0.918 | 0.936 | 0.898 | 0.894 | **0.911** | 1.000 |
| **ours** | **ALL (n=20000)** | **0.715** | **0.750** | **0.739** | **0.733** | **0.734** | 1.000 |

### multi-token items

| oracle | dataset | L4 (25%) | L8 (50%) | L12 (75%) | L14 (88%) | pooled | format |
|---|---|---|---|---|---|---|---|
| ours | ag_news | 0.748 | 0.760 | 0.768 | 0.712 | **0.747** | 1.000 |
| ours | geometry_of_truth | 0.672 | 0.676 | 0.652 | 0.652 | **0.663** | 1.000 |
| ours | language_identification | 0.736 | 0.748 | 0.720 | 0.652 | **0.714** | 1.000 |
| ours | md_gender | 0.892 | 0.900 | 0.916 | 0.928 | **0.909** | 1.000 |
| ours | ner | 0.920 | 0.932 | 0.936 | 0.920 | **0.927** | 1.000 |
| ours | relations | 0.608 | 0.612 | 0.604 | 0.620 | **0.611** | 1.000 |
| ours | singular_plural | 0.688 | 0.712 | 0.712 | 0.696 | **0.702** | 1.000 |
| ours | snli | 0.912 | 0.908 | 0.908 | 0.880 | **0.902** | 1.000 |
| ours | sst2 | 0.860 | 0.836 | 0.836 | 0.824 | **0.839** | 1.000 |
| ours | tense | 0.992 | 0.992 | 0.984 | 0.980 | **0.987** | 1.000 |
| **ours** | **ALL (n=10000)** | **0.803** | **0.808** | **0.804** | **0.786** | **0.800** | 1.000 |

## gemma3_1b_it — base `google/gemma-3-1b-it` @ `main`

`layer_percents` [25, 50, 75, 96] → act layers [6, 13, 19, 24]


### single-token items

| oracle | dataset | L6 (25%) | L13 (50%) | L19 (75%) | L24 (96%) | pooled | format |
|---|---|---|---|---|---|---|---|
| ours | ag_news | 0.632 | 0.620 | 0.642 | 0.632 | **0.631** | 1.000 |
| ours | geometry_of_truth | 0.568 | 0.658 | 0.598 | 0.606 | **0.608** | 1.000 |
| ours | language_identification | 0.520 | 0.518 | 0.498 | 0.512 | **0.512** | 1.000 |
| ours | md_gender | 0.776 | 0.750 | 0.732 | 0.718 | **0.744** | 1.000 |
| ours | ner | 0.786 | 0.754 | 0.788 | 0.788 | **0.779** | 1.000 |
| ours | relations | 0.564 | 0.614 | 0.576 | 0.560 | **0.579** | 1.000 |
| ours | singular_plural | 0.492 | 0.554 | 0.494 | 0.516 | **0.514** | 1.000 |
| ours | snli | 0.800 | 0.810 | 0.780 | 0.776 | **0.791** | 1.000 |
| ours | sst2 | 0.722 | 0.698 | 0.670 | 0.702 | **0.698** | 1.000 |
| ours | tense | 0.818 | 0.778 | 0.694 | 0.726 | **0.754** | 1.000 |
| **ours** | **ALL (n=20000)** | **0.668** | **0.675** | **0.647** | **0.654** | **0.661** | 1.000 |
| karvonen | ag_news | 0.670 | 0.644 | 0.622 | 0.608 | **0.636** | 1.000 |
| karvonen | geometry_of_truth | 0.598 | 0.650 | 0.610 | 0.574 | **0.608** | 1.000 |
| karvonen | language_identification | 0.512 | 0.510 | 0.516 | 0.528 | **0.516** | 1.000 |
| karvonen | md_gender | 0.792 | 0.764 | 0.742 | 0.690 | **0.747** | 1.000 |
| karvonen | ner | 0.794 | 0.784 | 0.796 | 0.794 | **0.792** | 1.000 |
| karvonen | relations | 0.574 | 0.610 | 0.616 | 0.554 | **0.589** | 1.000 |
| karvonen | singular_plural | 0.516 | 0.568 | 0.490 | 0.500 | **0.518** | 1.000 |
| karvonen | snli | 0.792 | 0.826 | 0.780 | 0.774 | **0.793** | 1.000 |
| karvonen | sst2 | 0.758 | 0.742 | 0.676 | 0.656 | **0.708** | 1.000 |
| karvonen | tense | 0.870 | 0.840 | 0.746 | 0.706 | **0.790** | 1.000 |
| **karvonen** | **ALL (n=20000)** | **0.688** | **0.694** | **0.659** | **0.638** | **0.670** | 1.000 |

### multi-token items

| oracle | dataset | L6 (25%) | L13 (50%) | L19 (75%) | L24 (96%) | pooled | format |
|---|---|---|---|---|---|---|---|
| ours | ag_news | 0.796 | 0.788 | 0.772 | 0.768 | **0.781** | 1.000 |
| ours | geometry_of_truth | 0.672 | 0.692 | 0.636 | 0.656 | **0.664** | 1.000 |
| ours | language_identification | 0.548 | 0.508 | 0.584 | 0.596 | **0.559** | 0.998 |
| ours | md_gender | 0.880 | 0.900 | 0.868 | 0.884 | **0.883** | 1.000 |
| ours | ner | 0.928 | 0.908 | 0.876 | 0.904 | **0.904** | 1.000 |
| ours | relations | 0.636 | 0.652 | 0.684 | 0.616 | **0.647** | 1.000 |
| ours | singular_plural | 0.576 | 0.568 | 0.584 | 0.568 | **0.574** | 1.000 |
| ours | snli | 0.892 | 0.868 | 0.868 | 0.872 | **0.875** | 1.000 |
| ours | sst2 | 0.816 | 0.840 | 0.792 | 0.792 | **0.810** | 1.000 |
| ours | tense | 0.952 | 0.964 | 0.900 | 0.872 | **0.922** | 1.000 |
| **ours** | **ALL (n=10000)** | **0.770** | **0.769** | **0.756** | **0.753** | **0.762** | 1.000 |
| karvonen | ag_news | 0.788 | 0.772 | 0.784 | 0.768 | **0.778** | 1.000 |
| karvonen | geometry_of_truth | 0.708 | 0.708 | 0.676 | 0.600 | **0.673** | 1.000 |
| karvonen | language_identification | 0.532 | 0.520 | 0.544 | 0.580 | **0.544** | 1.000 |
| karvonen | md_gender | 0.888 | 0.904 | 0.868 | 0.868 | **0.882** | 1.000 |
| karvonen | ner | 0.932 | 0.908 | 0.888 | 0.888 | **0.904** | 1.000 |
| karvonen | relations | 0.700 | 0.676 | 0.696 | 0.548 | **0.655** | 1.000 |
| karvonen | singular_plural | 0.592 | 0.600 | 0.604 | 0.560 | **0.589** | 1.000 |
| karvonen | snli | 0.904 | 0.868 | 0.872 | 0.860 | **0.876** | 1.000 |
| karvonen | sst2 | 0.836 | 0.852 | 0.832 | 0.760 | **0.820** | 1.000 |
| karvonen | tense | 0.968 | 0.980 | 0.908 | 0.904 | **0.940** | 1.000 |
| **karvonen** | **ALL (n=10000)** | **0.785** | **0.779** | **0.767** | **0.734** | **0.766** | 1.000 |

### ours − karvonen (shared layer percents [25, 50, 75])

| variant | scope | ours | karvonen | delta |
|---|---|---|---|---|
| single | shared layers | 0.663 | 0.680 | -0.017 |
| single | OOD layer(s) *(OOD for karvonen)* | 0.654 | 0.638 | +0.015 |
| multi | shared layers | 0.765 | 0.777 | -0.012 |
| multi | OOD layer(s) *(OOD for karvonen)* | 0.753 | 0.734 | +0.019 |

## karvonen_Llama-3_2-1B-Instruct — base `meta-llama/Llama-3.2-1B-Instruct` @ `main`

`layer_percents` [25, 50, 75] → act layers [4, 8, 12]


### single-token items

| oracle | dataset | L4 (25%) | L8 (50%) | L12 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.724 | 0.714 | 0.710 | **0.716** | 1.000 |
| karvonen | geometry_of_truth | 0.620 | 0.718 | 0.676 | **0.671** | 1.000 |
| karvonen | language_identification | 0.572 | 0.604 | 0.634 | **0.603** | 0.999 |
| karvonen | md_gender | 0.900 | 0.884 | 0.864 | **0.883** | 1.000 |
| karvonen | ner | 0.846 | 0.844 | 0.830 | **0.840** | 1.000 |
| karvonen | relations | 0.576 | 0.696 | 0.640 | **0.637** | 1.000 |
| karvonen | singular_plural | 0.662 | 0.588 | 0.554 | **0.601** | 1.000 |
| karvonen | snli | 0.802 | 0.808 | 0.798 | **0.803** | 1.000 |
| karvonen | sst2 | 0.760 | 0.764 | 0.732 | **0.752** | 1.000 |
| karvonen | tense | 0.902 | 0.888 | 0.858 | **0.883** | 1.000 |
| **karvonen** | **ALL (n=15000)** | **0.736** | **0.751** | **0.730** | **0.739** | 1.000 |

### multi-token items

| oracle | dataset | L4 (25%) | L8 (50%) | L12 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.848 | 0.796 | 0.784 | **0.809** | 1.000 |
| karvonen | geometry_of_truth | 0.664 | 0.708 | 0.668 | **0.680** | 1.000 |
| karvonen | language_identification | 0.696 | 0.748 | 0.796 | **0.747** | 0.985 |
| karvonen | md_gender | 0.920 | 0.928 | 0.928 | **0.925** | 1.000 |
| karvonen | ner | 0.984 | 0.952 | 0.956 | **0.964** | 1.000 |
| karvonen | relations | 0.688 | 0.692 | 0.664 | **0.681** | 1.000 |
| karvonen | singular_plural | 0.732 | 0.704 | 0.680 | **0.705** | 1.000 |
| karvonen | snli | 0.908 | 0.904 | 0.908 | **0.907** | 1.000 |
| karvonen | sst2 | 0.844 | 0.832 | 0.796 | **0.824** | 1.000 |
| karvonen | tense | 0.972 | 0.976 | 0.952 | **0.967** | 1.000 |
| **karvonen** | **ALL (n=7500)** | **0.826** | **0.824** | **0.813** | **0.821** | 0.999 |

## karvonen_Qwen3-1_7B — base `Qwen/Qwen3-1.7B` @ `main`

`layer_percents` [25, 50, 75] → act layers [7, 14, 21]


### single-token items

| oracle | dataset | L7 (25%) | L14 (50%) | L21 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.712 | 0.682 | 0.654 | **0.683** | 1.000 |
| karvonen | geometry_of_truth | 0.612 | 0.782 | 0.812 | **0.735** | 1.000 |
| karvonen | language_identification | 0.548 | 0.554 | 0.610 | **0.571** | 1.000 |
| karvonen | md_gender | 0.864 | 0.826 | 0.770 | **0.820** | 1.000 |
| karvonen | ner | 0.850 | 0.828 | 0.848 | **0.842** | 1.000 |
| karvonen | relations | 0.574 | 0.632 | 0.672 | **0.626** | 1.000 |
| karvonen | singular_plural | 0.586 | 0.650 | 0.590 | **0.609** | 1.000 |
| karvonen | snli | 0.804 | 0.820 | 0.818 | **0.814** | 1.000 |
| karvonen | sst2 | 0.788 | 0.806 | 0.764 | **0.786** | 1.000 |
| karvonen | tense | 0.910 | 0.930 | 0.894 | **0.911** | 1.000 |
| **karvonen** | **ALL (n=15000)** | **0.725** | **0.751** | **0.743** | **0.740** | 1.000 |

### multi-token items

| oracle | dataset | L7 (25%) | L14 (50%) | L21 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.848 | 0.828 | 0.828 | **0.835** | 1.000 |
| karvonen | geometry_of_truth | 0.912 | 0.916 | 0.912 | **0.913** | 1.000 |
| karvonen | language_identification | 0.676 | 0.684 | 0.764 | **0.708** | 1.000 |
| karvonen | md_gender | 0.936 | 0.936 | 0.916 | **0.929** | 1.000 |
| karvonen | ner | 0.948 | 0.956 | 0.964 | **0.956** | 1.000 |
| karvonen | relations | 0.728 | 0.696 | 0.680 | **0.701** | 1.000 |
| karvonen | singular_plural | 0.696 | 0.712 | 0.748 | **0.719** | 1.000 |
| karvonen | snli | 0.916 | 0.932 | 0.916 | **0.921** | 1.000 |
| karvonen | sst2 | 0.848 | 0.860 | 0.848 | **0.852** | 1.000 |
| karvonen | tense | 0.988 | 0.984 | 0.984 | **0.985** | 1.000 |
| **karvonen** | **ALL (n=7500)** | **0.850** | **0.850** | **0.856** | **0.852** | 1.000 |

## karvonen_Qwen3-4B — base `Qwen/Qwen3-4B` @ `main`

`layer_percents` [25, 50, 75] → act layers [9, 18, 27]


### single-token items

| oracle | dataset | L9 (25%) | L18 (50%) | L27 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.748 | 0.734 | 0.698 | **0.727** | 1.000 |
| karvonen | geometry_of_truth | 0.596 | 0.946 | 0.918 | **0.820** | 1.000 |
| karvonen | language_identification | 0.624 | 0.634 | 0.726 | **0.661** | 1.000 |
| karvonen | md_gender | 0.834 | 0.850 | 0.770 | **0.818** | 1.000 |
| karvonen | ner | 0.852 | 0.842 | 0.876 | **0.857** | 1.000 |
| karvonen | relations | 0.588 | 0.728 | 0.720 | **0.679** | 1.000 |
| karvonen | singular_plural | 0.710 | 0.680 | 0.638 | **0.676** | 1.000 |
| karvonen | snli | 0.808 | 0.826 | 0.848 | **0.827** | 1.000 |
| karvonen | sst2 | 0.792 | 0.830 | 0.830 | **0.817** | 1.000 |
| karvonen | tense | 0.948 | 0.936 | 0.916 | **0.933** | 1.000 |
| **karvonen** | **ALL (n=15000)** | **0.750** | **0.801** | **0.794** | **0.782** | 1.000 |

### multi-token items

| oracle | dataset | L9 (25%) | L18 (50%) | L27 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.796 | 0.784 | 0.768 | **0.783** | 1.000 |
| karvonen | geometry_of_truth | 0.928 | 0.960 | 0.940 | **0.943** | 1.000 |
| karvonen | language_identification | 0.816 | 0.740 | 0.820 | **0.792** | 1.000 |
| karvonen | md_gender | 0.924 | 0.916 | 0.912 | **0.917** | 1.000 |
| karvonen | ner | 0.972 | 0.936 | 0.972 | **0.960** | 1.000 |
| karvonen | relations | 0.728 | 0.768 | 0.756 | **0.751** | 1.000 |
| karvonen | singular_plural | 0.776 | 0.756 | 0.800 | **0.777** | 1.000 |
| karvonen | snli | 0.952 | 0.940 | 0.944 | **0.945** | 1.000 |
| karvonen | sst2 | 0.884 | 0.880 | 0.884 | **0.883** | 1.000 |
| karvonen | tense | 0.988 | 0.988 | 0.980 | **0.985** | 1.000 |
| **karvonen** | **ALL (n=7500)** | **0.876** | **0.867** | **0.878** | **0.874** | 1.000 |

## karvonen_Qwen3-8B — base `Qwen/Qwen3-8B` @ `main`

`layer_percents` [25, 50, 75] → act layers [9, 18, 27]


### single-token items

| oracle | dataset | L9 (25%) | L18 (50%) | L27 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.776 | 0.752 | 0.746 | **0.758** | 1.000 |
| karvonen | geometry_of_truth | 0.614 | 0.944 | 0.936 | **0.831** | 1.000 |
| karvonen | language_identification | 0.666 | 0.640 | 0.758 | **0.688** | 1.000 |
| karvonen | md_gender | 0.880 | 0.862 | 0.764 | **0.835** | 1.000 |
| karvonen | ner | 0.874 | 0.844 | 0.876 | **0.865** | 1.000 |
| karvonen | relations | 0.612 | 0.754 | 0.732 | **0.699** | 1.000 |
| karvonen | singular_plural | 0.742 | 0.756 | 0.748 | **0.749** | 1.000 |
| karvonen | snli | 0.840 | 0.856 | 0.834 | **0.843** | 1.000 |
| karvonen | sst2 | 0.814 | 0.828 | 0.808 | **0.817** | 1.000 |
| karvonen | tense | 0.962 | 0.964 | 0.928 | **0.951** | 1.000 |
| **karvonen** | **ALL (n=15000)** | **0.778** | **0.820** | **0.813** | **0.804** | 1.000 |

### multi-token items

| oracle | dataset | L9 (25%) | L18 (50%) | L27 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.876 | 0.852 | 0.856 | **0.861** | 1.000 |
| karvonen | geometry_of_truth | 0.940 | 0.944 | 0.944 | **0.943** | 1.000 |
| karvonen | language_identification | 0.872 | 0.808 | 0.832 | **0.837** | 1.000 |
| karvonen | md_gender | 0.936 | 0.912 | 0.932 | **0.927** | 1.000 |
| karvonen | ner | 0.968 | 0.948 | 0.976 | **0.964** | 1.000 |
| karvonen | relations | 0.784 | 0.828 | 0.812 | **0.808** | 1.000 |
| karvonen | singular_plural | 0.848 | 0.876 | 0.852 | **0.859** | 1.000 |
| karvonen | snli | 0.960 | 0.960 | 0.964 | **0.961** | 1.000 |
| karvonen | sst2 | 0.876 | 0.888 | 0.852 | **0.872** | 1.000 |
| karvonen | tense | 0.988 | 0.988 | 0.980 | **0.985** | 1.000 |
| **karvonen** | **ALL (n=7500)** | **0.905** | **0.900** | **0.900** | **0.902** | 1.000 |

## karvonen_Llama-3_1-8B-Instruct — base `meta-llama/Llama-3.1-8B-Instruct` @ `main`

`layer_percents` [25, 50, 75] → act layers [8, 16, 24]


### single-token items

| oracle | dataset | L8 (25%) | L16 (50%) | L24 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.786 | 0.758 | 0.772 | **0.772** | 1.000 |
| karvonen | geometry_of_truth | 0.798 | 0.862 | 0.860 | **0.840** | 1.000 |
| karvonen | language_identification | 0.636 | 0.684 | 0.716 | **0.679** | 0.999 |
| karvonen | md_gender | 0.904 | 0.922 | 0.902 | **0.909** | 1.000 |
| karvonen | ner | 0.918 | 0.894 | 0.896 | **0.903** | 1.000 |
| karvonen | relations | 0.728 | 0.770 | 0.718 | **0.739** | 1.000 |
| karvonen | singular_plural | 0.770 | 0.746 | 0.738 | **0.751** | 1.000 |
| karvonen | snli | 0.854 | 0.864 | 0.828 | **0.849** | 1.000 |
| karvonen | sst2 | 0.810 | 0.822 | 0.792 | **0.808** | 1.000 |
| karvonen | tense | 0.962 | 0.950 | 0.908 | **0.940** | 1.000 |
| **karvonen** | **ALL (n=15000)** | **0.817** | **0.827** | **0.813** | **0.819** | 1.000 |

### multi-token items

| oracle | dataset | L8 (25%) | L16 (50%) | L24 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.904 | 0.880 | 0.868 | **0.884** | 1.000 |
| karvonen | geometry_of_truth | 0.864 | 0.892 | 0.872 | **0.876** | 1.000 |
| karvonen | language_identification | 0.812 | 0.852 | 0.856 | **0.840** | 1.000 |
| karvonen | md_gender | 0.924 | 0.924 | 0.932 | **0.927** | 1.000 |
| karvonen | ner | 0.980 | 0.952 | 0.964 | **0.965** | 1.000 |
| karvonen | relations | 0.760 | 0.772 | 0.748 | **0.760** | 1.000 |
| karvonen | singular_plural | 0.760 | 0.808 | 0.800 | **0.789** | 1.000 |
| karvonen | snli | 0.952 | 0.940 | 0.932 | **0.941** | 1.000 |
| karvonen | sst2 | 0.852 | 0.864 | 0.840 | **0.852** | 1.000 |
| karvonen | tense | 0.992 | 0.988 | 0.988 | **0.989** | 1.000 |
| **karvonen** | **ALL (n=7500)** | **0.880** | **0.887** | **0.880** | **0.882** | 1.000 |

## karvonen_gemma-2-9b-it — base `google/gemma-2-9b-it` @ `main`

`layer_percents` [25, 50, 75] → act layers [10, 21, 31]


### single-token items

| oracle | dataset | L10 (25%) | L21 (50%) | L31 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.834 | 0.812 | 0.788 | **0.811** | 0.999 |
| karvonen | geometry_of_truth | 0.666 | 0.874 | 0.920 | **0.820** | 0.969 |
| karvonen | language_identification | 0.746 | 0.656 | 0.840 | **0.747** | 0.999 |
| karvonen | md_gender | 0.938 | 0.916 | 0.874 | **0.909** | 0.993 |
| karvonen | ner | 0.914 | 0.932 | 0.908 | **0.918** | 0.987 |
| karvonen | relations | 0.676 | 0.752 | 0.734 | **0.721** | 0.947 |
| karvonen | singular_plural | 0.930 | 0.930 | 0.778 | **0.879** | 0.999 |
| karvonen | snli | 0.894 | 0.882 | 0.886 | **0.887** | 0.987 |
| karvonen | sst2 | 0.866 | 0.874 | 0.830 | **0.857** | 1.000 |
| karvonen | tense | 0.980 | 0.966 | 0.940 | **0.962** | 0.997 |
| **karvonen** | **ALL (n=15000)** | **0.844** | **0.859** | **0.850** | **0.851** | 0.988 |

### multi-token items

| oracle | dataset | L10 (25%) | L21 (50%) | L31 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.876 | 0.904 | 0.884 | **0.888** | 0.996 |
| karvonen | geometry_of_truth | 0.936 | 0.952 | 0.952 | **0.947** | 0.991 |
| karvonen | language_identification | 0.840 | 0.780 | 0.920 | **0.847** | 0.999 |
| karvonen | md_gender | 0.936 | 0.932 | 0.936 | **0.935** | 0.996 |
| karvonen | ner | 0.976 | 0.968 | 0.984 | **0.976** | 0.993 |
| karvonen | relations | 0.852 | 0.816 | 0.780 | **0.816** | 0.979 |
| karvonen | singular_plural | 0.896 | 0.920 | 0.904 | **0.907** | 0.995 |
| karvonen | snli | 0.944 | 0.948 | 0.940 | **0.944** | 0.989 |
| karvonen | sst2 | 0.880 | 0.892 | 0.880 | **0.884** | 1.000 |
| karvonen | tense | 0.992 | 0.988 | 0.984 | **0.988** | 0.996 |
| **karvonen** | **ALL (n=7500)** | **0.913** | **0.910** | **0.916** | **0.913** | 0.993 |

## karvonen_Qwen3-14B — base `Qwen/Qwen3-14B` @ `main`

`layer_percents` [25, 50, 75] → act layers [10, 20, 30]


### single-token items

| oracle | dataset | L10 (25%) | L20 (50%) | L30 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.786 | 0.760 | 0.772 | **0.773** | 1.000 |
| karvonen | geometry_of_truth | 0.670 | 0.934 | 0.928 | **0.844** | 1.000 |
| karvonen | language_identification | 0.798 | 0.756 | 0.812 | **0.789** | 1.000 |
| karvonen | md_gender | 0.896 | 0.846 | 0.806 | **0.849** | 1.000 |
| karvonen | ner | 0.884 | 0.898 | 0.876 | **0.886** | 1.000 |
| karvonen | relations | 0.702 | 0.784 | 0.762 | **0.749** | 1.000 |
| karvonen | singular_plural | 0.708 | 0.804 | 0.770 | **0.761** | 1.000 |
| karvonen | snli | 0.864 | 0.866 | 0.862 | **0.864** | 1.000 |
| karvonen | sst2 | 0.828 | 0.866 | 0.860 | **0.851** | 1.000 |
| karvonen | tense | 0.950 | 0.966 | 0.946 | **0.954** | 1.000 |
| **karvonen** | **ALL (n=15000)** | **0.809** | **0.848** | **0.839** | **0.832** | 1.000 |

### multi-token items

| oracle | dataset | L10 (25%) | L20 (50%) | L30 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.840 | 0.856 | 0.856 | **0.851** | 1.000 |
| karvonen | geometry_of_truth | 0.940 | 0.960 | 0.960 | **0.953** | 1.000 |
| karvonen | language_identification | 0.908 | 0.892 | 0.892 | **0.897** | 1.000 |
| karvonen | md_gender | 0.916 | 0.924 | 0.920 | **0.920** | 1.000 |
| karvonen | ner | 0.984 | 0.968 | 0.976 | **0.976** | 1.000 |
| karvonen | relations | 0.816 | 0.812 | 0.820 | **0.816** | 1.000 |
| karvonen | singular_plural | 0.856 | 0.864 | 0.896 | **0.872** | 1.000 |
| karvonen | snli | 0.948 | 0.952 | 0.960 | **0.953** | 1.000 |
| karvonen | sst2 | 0.880 | 0.900 | 0.884 | **0.888** | 1.000 |
| karvonen | tense | 0.992 | 0.992 | 0.984 | **0.989** | 1.000 |
| **karvonen** | **ALL (n=7500)** | **0.908** | **0.912** | **0.915** | **0.912** | 1.000 |

## karvonen_gemma-2-27b-it — base `google/gemma-2-27b-it` @ `main`

`layer_percents` [25, 50, 75] → act layers [11, 23, 34]


### single-token items

| oracle | dataset | L11 (25%) | L23 (50%) | L34 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.810 | 0.760 | 0.748 | **0.773** | 0.911 |
| karvonen | geometry_of_truth | 0.690 | 0.812 | 0.786 | **0.763** | 0.866 |
| karvonen | language_identification | 0.704 | 0.576 | 0.668 | **0.649** | 0.866 |
| karvonen | md_gender | 0.766 | 0.436 | 0.626 | **0.609** | 0.651 |
| karvonen | ner | 0.728 | 0.722 | 0.660 | **0.703** | 0.760 |
| karvonen | relations | 0.692 | 0.690 | 0.620 | **0.667** | 0.817 |
| karvonen | singular_plural | 0.762 | 0.798 | 0.652 | **0.737** | 0.794 |
| karvonen | snli | 0.770 | 0.596 | 0.558 | **0.641** | 0.694 |
| karvonen | sst2 | 0.846 | 0.744 | 0.690 | **0.760** | 0.875 |
| karvonen | tense | 0.912 | 0.900 | 0.798 | **0.870** | 0.908 |
| **karvonen** | **ALL (n=15000)** | **0.768** | **0.703** | **0.681** | **0.717** | 0.814 |

### multi-token items

| oracle | dataset | L11 (25%) | L23 (50%) | L34 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.900 | 0.876 | 0.876 | **0.884** | 0.971 |
| karvonen | geometry_of_truth | 0.932 | 0.912 | 0.892 | **0.912** | 0.957 |
| karvonen | language_identification | 0.744 | 0.724 | 0.780 | **0.749** | 0.895 |
| karvonen | md_gender | 0.896 | 0.888 | 0.840 | **0.875** | 0.935 |
| karvonen | ner | 0.844 | 0.852 | 0.792 | **0.829** | 0.839 |
| karvonen | relations | 0.856 | 0.804 | 0.756 | **0.805** | 0.949 |
| karvonen | singular_plural | 0.956 | 0.932 | 0.908 | **0.932** | 0.981 |
| karvonen | snli | 0.916 | 0.848 | 0.852 | **0.872** | 0.923 |
| karvonen | sst2 | 0.856 | 0.852 | 0.840 | **0.849** | 0.959 |
| karvonen | tense | 0.952 | 0.960 | 0.936 | **0.949** | 0.957 |
| **karvonen** | **ALL (n=7500)** | **0.885** | **0.865** | **0.847** | **0.866** | 0.937 |

## karvonen_gemma-3-27b-it — base `google/gemma-3-27b-it` @ `main`

`layer_percents` [25, 50, 75] → act layers [15, 31, 46]


### single-token items

| oracle | dataset | L15 (25%) | L31 (50%) | L46 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.852 | 0.750 | 0.770 | **0.791** | 1.000 |
| karvonen | geometry_of_truth | 0.692 | 0.932 | 0.926 | **0.850** | 1.000 |
| karvonen | language_identification | 0.824 | 0.674 | 0.796 | **0.765** | 1.000 |
| karvonen | md_gender | 0.928 | 0.840 | 0.828 | **0.865** | 1.000 |
| karvonen | ner | 0.934 | 0.906 | 0.900 | **0.913** | 1.000 |
| karvonen | relations | 0.716 | 0.848 | 0.772 | **0.779** | 1.000 |
| karvonen | singular_plural | 0.904 | 0.854 | 0.762 | **0.840** | 1.000 |
| karvonen | snli | 0.910 | 0.890 | 0.882 | **0.894** | 1.000 |
| karvonen | sst2 | 0.866 | 0.866 | 0.822 | **0.851** | 1.000 |
| karvonen | tense | 0.964 | 0.966 | 0.902 | **0.944** | 1.000 |
| **karvonen** | **ALL (n=15000)** | **0.859** | **0.853** | **0.836** | **0.849** | 1.000 |

### multi-token items

| oracle | dataset | L15 (25%) | L31 (50%) | L46 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.892 | 0.884 | 0.888 | **0.888** | 1.000 |
| karvonen | geometry_of_truth | 0.936 | 0.972 | 0.952 | **0.953** | 1.000 |
| karvonen | language_identification | 0.896 | 0.796 | 0.872 | **0.855** | 1.000 |
| karvonen | md_gender | 0.936 | 0.920 | 0.936 | **0.931** | 1.000 |
| karvonen | ner | 0.996 | 0.984 | 0.992 | **0.991** | 1.000 |
| karvonen | relations | 0.864 | 0.828 | 0.804 | **0.832** | 1.000 |
| karvonen | singular_plural | 0.948 | 0.928 | 0.940 | **0.939** | 1.000 |
| karvonen | snli | 0.964 | 0.944 | 0.940 | **0.949** | 1.000 |
| karvonen | sst2 | 0.896 | 0.896 | 0.884 | **0.892** | 1.000 |
| karvonen | tense | 0.992 | 0.984 | 0.980 | **0.985** | 1.000 |
| **karvonen** | **ALL (n=7500)** | **0.932** | **0.914** | **0.919** | **0.921** | 1.000 |

## karvonen_Qwen3-32B — base `Qwen/Qwen3-32B` @ `main`

`layer_percents` [25, 50, 75] → act layers [16, 32, 48]


### single-token items

| oracle | dataset | L16 (25%) | L32 (50%) | L48 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.820 | 0.818 | 0.758 | **0.799** | 1.000 |
| karvonen | geometry_of_truth | 0.756 | 0.904 | 0.942 | **0.867** | 1.000 |
| karvonen | language_identification | 0.856 | 0.856 | 0.804 | **0.839** | 0.999 |
| karvonen | md_gender | 0.914 | 0.898 | 0.830 | **0.881** | 1.000 |
| karvonen | ner | 0.932 | 0.932 | 0.888 | **0.917** | 1.000 |
| karvonen | relations | 0.658 | 0.742 | 0.784 | **0.728** | 1.000 |
| karvonen | singular_plural | 0.918 | 0.932 | 0.922 | **0.924** | 1.000 |
| karvonen | snli | 0.912 | 0.920 | 0.890 | **0.907** | 1.000 |
| karvonen | sst2 | 0.850 | 0.858 | 0.866 | **0.858** | 1.000 |
| karvonen | tense | 0.978 | 0.968 | 0.980 | **0.975** | 1.000 |
| **karvonen** | **ALL (n=15000)** | **0.859** | **0.883** | **0.866** | **0.870** | 1.000 |

### multi-token items

| oracle | dataset | L16 (25%) | L32 (50%) | L48 (75%) | pooled | format |
|---|---|---|---|---|---|---|
| karvonen | ag_news | 0.896 | 0.896 | 0.900 | **0.897** | 1.000 |
| karvonen | geometry_of_truth | 0.948 | 0.956 | 0.956 | **0.953** | 1.000 |
| karvonen | language_identification | 0.916 | 0.932 | 0.904 | **0.917** | 0.999 |
| karvonen | md_gender | 0.932 | 0.924 | 0.916 | **0.924** | 1.000 |
| karvonen | ner | 0.984 | 0.984 | 0.976 | **0.981** | 1.000 |
| karvonen | relations | 0.844 | 0.808 | 0.844 | **0.832** | 1.000 |
| karvonen | singular_plural | 0.976 | 0.956 | 0.956 | **0.963** | 1.000 |
| karvonen | snli | 0.952 | 0.964 | 0.952 | **0.956** | 1.000 |
| karvonen | sst2 | 0.904 | 0.912 | 0.904 | **0.907** | 1.000 |
| karvonen | tense | 0.992 | 0.988 | 0.984 | **0.988** | 1.000 |
| **karvonen** | **ALL (n=7500)** | **0.934** | **0.932** | **0.929** | **0.932** | 1.000 |