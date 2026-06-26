#!/usr/bin/env bash
# Full DIRECT crossover re-run over all 20 taboo words (16 original + 4 new:
# flag, rock, snow, wave). Produces base_home_cross_lora_pooled_acc_direct.png.
#
#   eval (taboo-only, all_direct, n=99 -> 21 oracles x 20 MOs = 420 cells, ~4h)
#     -> score (direct subdir only)
#     -> copy scores to top-level crossover_results/ (the canonical direct scores_lora.json)
#     -> plot base/home/avg-cross with --tag direct (top-level + subdir copies)
#
# Detached-friendly: run under tmux so it survives the session closing.
#   tmux new-session -d -s direct20 'bash experiments/crossover/run_direct_20words.sh'
set -o pipefail
cd "$(dirname "$0")/.." || exit 1            # -> experiments/crossover/

SD="crossover_results/gemma-2-9b-it_open_ended_all_direct_test"
LOG="crossover_results/run_direct_20words.log"
mkdir -p "$SD"
: > "$LOG"
ts() { date '+%F %T'; }

echo "[$(ts)] START full direct 20-word crossover" | tee -a "$LOG"

echo "[$(ts)] STEP 1/4 eval (GPU, ~4-5h: 420 cells). Batch 256: 20 taboo MOs stay" | tee -a "$LOG"
echo "[$(ts)]        resident, so 1024 OOMs; results are batch-invariant (greedy)." | tee -a "$LOG"
CX_FAMILIES=taboo CX_PROMPT_TYPE=all_direct CX_BATCH=256 \
  uv run python pipeline/crossover_eval.py >> "$LOG" 2>&1 \
  || { echo "[$(ts)] EVAL FAILED" | tee -a "$LOG"; exit 1; }

echo "[$(ts)] STEP 2/4 score (direct subdir only)" | tee -a "$LOG"
uv run python pipeline/score_crossover.py --results-dir "$SD" >> "$LOG" 2>&1 \
  || { echo "[$(ts)] SCORE FAILED" | tee -a "$LOG"; exit 1; }

echo "[$(ts)] STEP 3/4 copy direct scores to top-level (canonical direct scores_lora.json)" | tee -a "$LOG"
cp "$SD/scores_lora.json" crossover_results/scores_lora.json

echo "[$(ts)] STEP 4/4 plot base/home/avg-cross (--tag direct)" | tee -a "$LOG"
# Top-level figure (the canonical base_home_cross_lora_pooled_acc_direct.png) + a copy in the subdir.
uv run python pipeline/plot_base_home_cross.py --results-dir crossover_results --tag direct >> "$LOG" 2>&1 \
  || { echo "[$(ts)] PLOT(top) FAILED" | tee -a "$LOG"; exit 1; }
uv run python pipeline/plot_base_home_cross.py --results-dir "$SD" --tag direct >> "$LOG" 2>&1 \
  || { echo "[$(ts)] PLOT(subdir) FAILED" | tee -a "$LOG"; exit 1; }

echo "[$(ts)] DONE -> crossover_results/base_home_cross_lora_pooled_acc_direct.png" | tee -a "$LOG"
