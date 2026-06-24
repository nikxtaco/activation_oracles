#!/usr/bin/env bash
# Self-contained crossover re-run including the taboo-cat oracle.
#   eval (56 cells, GPU) -> score (matrices + scores_lora.json) -> split heatmaps.
# Detached-friendly: run under tmux so it survives the Claude session closing.
#   tmux new-session -d -s crossover_cat 'bash experiments/crossover/run_crossover_cat.sh'
set -o pipefail
cd "$(dirname "$0")" || exit 1   # -> experiments/crossover/ (uv finds the repo pyproject above it)

LOG="crossover_results/run_crossover_cat.log"
mkdir -p crossover_results
: > "$LOG"

echo "[$(date '+%F %T')] START crossover cat run" | tee -a "$LOG"

echo "[$(date '+%F %T')] STEP 1/3 eval (GPU, ~40 min)" | tee -a "$LOG"
uv run python crossover_eval.py >> "$LOG" 2>&1 || { echo "[$(date '+%F %T')] EVAL FAILED" | tee -a "$LOG"; exit 1; }

echo "[$(date '+%F %T')] STEP 2/3 score" | tee -a "$LOG"
uv run python score_crossover.py >> "$LOG" 2>&1 || { echo "[$(date '+%F %T')] SCORE FAILED" | tee -a "$LOG"; exit 1; }

echo "[$(date '+%F %T')] STEP 3/3 split heatmaps" | tee -a "$LOG"
uv run python split_heatmap.py >> "$LOG" 2>&1 || { echo "[$(date '+%F %T')] SPLIT FAILED" | tee -a "$LOG"; exit 1; }

echo "[$(date '+%F %T')] DONE — all artifacts in experiments/crossover/crossover_results/" | tee -a "$LOG"
