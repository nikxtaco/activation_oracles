# Deprecated crossover side-experiments

These experiments measured **how much an oracle expresses or reveals its host's quirk** —
useful early sanity checks, but not part of the current blindness story. They are kept here
for reference and are no longer wired into the main pipeline.

- `think_suppress/` — the think / suppress / neutral priming axis: does priming the model to
  think about its secret while it does an unrelated number task leak the secret into the
  activations the oracle reads? (Self-contained, own README + results.)
- `quirk_survival/` — reveal + persuasion controls: treat each host+oracle config as a
  generative model and check it still keeps its secret (i.e. the oracle did not simply break
  the host's secret-keeping).
- `run_think_after_standard.sh` — runner that launched the think/suppress eval after the main
  standard crossover freed the GPU.
