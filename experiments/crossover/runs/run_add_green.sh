#!/usr/bin/env bash
# Incrementally add the `green` taboo oracle to the finished 20-word DIRECT crossover,
# WITHOUT re-running the whole grid. Only the 42 new cells:
#   Pass A: green oracle x all 21 MOs (21 cells)            -> main dir (new crossover_green.json)
#   Pass B: the 20 existing oracles + base x green MO (21)  -> green_add/ subdir (green column only)
# Then re-score (recursive glob merges main + green_add) and re-plot the _direct figure.
set -o pipefail
cd "$(dirname "$0")/.." || exit 1

SD="crossover_results/gemma-2-9b-it_open_ended_all_direct_test"
GREEN_ADD="$SD/green_add"
LOG="crossover_results/run_add_green.log"
mkdir -p "$GREEN_ADD"
: > "$LOG"
ts() { date '+%F %T'; }
EXIST="salt,blue,jump,moon,cat,chair,song,smile,book,clock,flame,gold,cloud,dance,ship,leaf,flag,rock,snow,wave,base"

echo "[$(ts)] START add-green" | tee -a "$LOG"

echo "[$(ts)] PASS A: green oracle x all 21 MOs (~21 cells)" | tee -a "$LOG"
CX_FAMILIES=taboo CX_PROMPT_TYPE=all_direct CX_BATCH=256 CX_ORACLES=green \
  uv run python pipeline/crossover_eval.py >> "$LOG" 2>&1 \
  || { echo "[$(ts)] PASS A FAILED" | tee -a "$LOG"; exit 1; }

echo "[$(ts)] PASS B: 20 existing oracles + base x green MO (~21 cells) -> green_add/" | tee -a "$LOG"
CX_FAMILIES=taboo CX_PROMPT_TYPE=all_direct CX_BATCH=512 CX_ORACLES="$EXIST" CX_TARGETS=green \
  CX_OUTDIR="$GREEN_ADD" \
  uv run python pipeline/crossover_eval.py >> "$LOG" 2>&1 \
  || { echo "[$(ts)] PASS B FAILED" | tee -a "$LOG"; exit 1; }

echo "[$(ts)] SCORE (merges main + green_add via recursive glob)" | tee -a "$LOG"
uv run python pipeline/score_crossover.py --results-dir "$SD" >> "$LOG" 2>&1 \
  || { echo "[$(ts)] SCORE FAILED" | tee -a "$LOG"; exit 1; }
cp "$SD/scores_lora.json" crossover_results/scores_lora.json

echo "[$(ts)] PLOT _direct figure (now 21 words incl green)" | tee -a "$LOG"
uv run python pipeline/plot_base_home_cross.py --results-dir crossover_results --tag direct >> "$LOG" 2>&1 \
  || { echo "[$(ts)] PLOT FAILED" | tee -a "$LOG"; exit 1; }
uv run python pipeline/plot_base_home_cross.py --results-dir "$SD" --tag direct >> "$LOG" 2>&1

echo "[$(ts)] DONE -> crossover_results/base_home_cross_lora_pooled_acc_direct.png (with green)" | tee -a "$LOG"
