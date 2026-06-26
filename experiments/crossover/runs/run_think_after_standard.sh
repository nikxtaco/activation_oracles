#!/usr/bin/env bash
# Wait for the full standard crossover to finish (frees the GPU), then run the
# think/suppress/neutral experiment (eval -> score). Avoids two concurrent gemma-2-9b jobs.
set -o pipefail
cd "$(dirname "$0")/.." || exit 1            # -> experiments/crossover/
LOG="crossover_results/think_suppress_run.log"
: > "$LOG"
ts() { date '+%F %T'; }

echo "[$(ts)] waiting for standard run DONE marker..." | tee -a "$LOG"
until grep -q "DONE -> " crossover_results/run_standard_21words.log 2>/dev/null; do sleep 60; done
echo "[$(ts)] standard finished -> starting think_suppress eval (~30-45 min)" | tee -a "$LOG"

uv run python think_suppress/think_suppress_eval.py >> "$LOG" 2>&1 \
  || { echo "[$(ts)] THINK EVAL FAILED" | tee -a "$LOG"; exit 1; }
echo "[$(ts)] scoring" | tee -a "$LOG"
uv run python think_suppress/think_suppress_score.py >> "$LOG" 2>&1 \
  || { echo "[$(ts)] THINK SCORE FAILED" | tee -a "$LOG"; exit 1; }
echo "[$(ts)] DONE_THINK -> think_suppress/think_suppress_results/" | tee -a "$LOG"
