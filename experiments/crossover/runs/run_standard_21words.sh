#!/usr/bin/env bash
# Full STANDARD (hint-seeking elicitation: "Hint me." / "Clue me in.") crossover over all
# 21 taboo words. 21 oracles x 21 MOs = 441 cells, ~6-7h at batch 256 (21 MOs stay resident
# so 1024 OOMs; results are batch-invariant). eval -> score -> plot lineplots.
set -o pipefail
cd "$(dirname "$0")/.." || exit 1

SD="crossover_results/gemma-2-9b-it_open_ended_all_standard_test"
LOG="crossover_results/run_standard_21words.log"
mkdir -p "$SD"
: > "$LOG"
ts() { date '+%F %T'; }

echo "[$(ts)] START full standard 21-word crossover" | tee -a "$LOG"

echo "[$(ts)] STEP 1/4 eval (GPU, ~6-7h: 441 cells, hint prompts)" | tee -a "$LOG"
CX_FAMILIES=taboo CX_PROMPT_TYPE=all_standard CX_BATCH=256 \
  uv run python pipeline/crossover_eval.py >> "$LOG" 2>&1 \
  || { echo "[$(ts)] EVAL FAILED" | tee -a "$LOG"; exit 1; }

echo "[$(ts)] STEP 2/3 score" | tee -a "$LOG"
uv run python pipeline/score_crossover.py --results-dir "$SD" >> "$LOG" 2>&1 \
  || { echo "[$(ts)] SCORE FAILED" | tee -a "$LOG"; exit 1; }

echo "[$(ts)] STEP 3/3 plots (base/home/cross --tag standard + refresh side-by-side & ao-avg)" | tee -a "$LOG"
uv run python pipeline/plot_base_home_cross.py --results-dir "$SD" --tag standard >> "$LOG" 2>&1
# direct-vs-standard composites now compare 21 vs 21 words:
uv run python pipeline/plot_side_by_side.py >> "$LOG" 2>&1
uv run python pipeline/plot_ao_avg.py >> "$LOG" 2>&1

echo "[$(ts)] DONE -> $SD (21-word standard grid) + refreshed direct-vs-standard figures" | tee -a "$LOG"
