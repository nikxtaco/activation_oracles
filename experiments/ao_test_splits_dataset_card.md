---
license: other
task_categories:
- text-classification
tags:
- interpretability
- activation-oracles
- probing
---

# Activation Oracle training-task test splits

Held-out classification test splits for the Activation Oracle (AO) training tasks, generated for
**ten base models** that Adam Karvonen released oracles for but never published training data for
(`hf_push_to_hub=False` in his run configs).

Each `.pt` file contains the test split for one (dataset, variant, base model) triple, with
**precomputed activation vectors** — the AO pipeline forces `save_acts=True` for test splits, so
these are ready to evaluate without re-running the base model.

## Contents

200 files: 10 base models × 10 classification datasets × 2 variants.

| base model | files |
|---|---|
| `meta-llama/Llama-3.2-1B-Instruct` | 20 |
| `Qwen/Qwen3-1.7B` | 20 |
| `Qwen/Qwen3-4B` | 20 |
| `Qwen/Qwen3-8B` | 20 |
| `meta-llama/Llama-3.1-8B-Instruct` | 20 |
| `google/gemma-2-9b-it` | 20 |
| `Qwen/Qwen3-14B` | 20 |
| `google/gemma-2-27b-it` | 20 |
| `google/gemma-3-27b-it` | 20 |
| `Qwen/Qwen3-32B` | 20 |

Datasets: `geometry_of_truth`, `relations`, `sst2`, `md_gender`, `snli`, `ag_news`, `ner`,
`tense`, `language_identification`, `singular_plural`.

Two variants per dataset:

- **single** — `max_window_size=1`, `num_qa_per_sample=2`: exactly one activation vector is
  injected and the oracle answers yes/no from it.
- **multi** — `max_window_size=50`, `num_qa_per_sample=1`: a contiguous span of `k ~ U[1, 50]`
  vectors ending a few tokens from the end of the sequence.

Both variants are included because the upstream `build_datasets` keys eval data by
`dataset_config.dataset_name`, which is identical for the two loaders, so the multi loader
overwrites the single one and in-training eval only ever scored multi-token items.

## Generation config

All splits use `layer_percents = [25, 50, 75]` and `hook_onto_layer = 1`, matching every oracle
in Karvonen's "Activation Oracles" collection (read from each repo's `ao_config.json`). The
classification loader params match his exactly: `min_end_offset=-1`, `max_end_offset=-5`,
`num_test=250`, `seed=42`.

The test items are **the same underlying sentences and questions for every base model**:
`get_classification_datapoints` carves the test set out as `all_examples[-250:]` after a seed-42
shuffle, and that carve-out does not depend on `num_train`. Only the activations, tokenization
and sampled windows differ between models. This makes cross-model comparison clean.

## Usage

Drop the files into `sft_training_data/` in a checkout of
[activation_oracles](https://github.com/adamkarvonen/activation_oracles); the config-hashed
filenames are what the loaders look for, so they will be picked up as cache hits rather than
regenerated:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="model-organisms-for-real/ao-training-task-test-splits",
    repo_type="dataset",
    local_dir="sft_training_data",
    allow_patterns=["*Qwen3-8B*"],   # or omit to fetch all ten models
)
```

Loading one file directly:

```python
import torch
from nl_probes.utils.dataset_utils import TrainingDataPoint

obj = torch.load("classification_sst2_model_Qwen3-8B_n_250_save_acts_False_test_<hash>.pt")
data = [TrainingDataPoint(**d) for d in obj["data"]]
print(data[0].layer, data[0].steering_vectors.shape, data[0].target_output)
```

Note the `save_acts_False` in the filename reflects the *config* field; test splits always store
activations regardless, so `steering_vectors` is populated.

## Provenance and caveats

- Generated with a fork of the upstream repo:
  [nikxtaco/activation_oracles](https://github.com/nikxtaco/activation_oracles), branch
  `raffaello/ao-training-task-evals`, via `experiments/eval_training_tasks.py`.
- Activations are collected from the **clean base model**, no adapter applied.
- These files contain activation vectors derived from license-gated base models (Gemma, Llama).
  The corresponding model licenses govern derivative artifacts; check them before redistributing.
- The equivalent splits for `allenai/OLMo-2-0425-1B-SFT` and `google/gemma-3-1b-it` are **not**
  here — they are already published in
  `model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1-training-data` and
  `model-organisms-for-real/gemma3_1b_it_oracle_v1-training-data`, and use four layer percents
  ([25, 50, 75, 88] and [25, 50, 75, 96]) rather than three.
- `meta-llama/Llama-3.3-70B-Instruct` is absent: 282 GB of weights, and Karvonen's config
  quantized it to 8-bit, which would not be comparable to these bf16 runs.

## Results

Eval results using these splits, across 13 oracles, are in the fork under
`experiments/AO_TRAINING_TASK_EVALS.md`.
