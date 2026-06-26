#!/usr/bin/env bash
# Every INTERVAL, discover newly-published taboo AOs in the model-organisms-for-real HF
# org and, when the GPU is free, wire them in (append to crossover_taboo_words.txt) and
# re-run the crossover (eval -> score -> split heatmaps). Append-only log.
#
# Detached-friendly — run under tmux so it survives the Claude session closing:
#   tmux new-session -d -s crossover_watch 'bash experiments/crossover/crossover_watch.sh'
#
# Scope: taboo oracles named gemma2_9b_it_taboo_<word>_oracle_v1 whose host MO
# bcywinski/gemma-2-9b-it-taboo-<word> also exists. New gender/other families need a
# manual edit to crossover_config.py.
set -u
cd "$(dirname "$0")/.." || exit 1             # -> experiments/crossover/

INTERVAL="${CROSSOVER_WATCH_INTERVAL:-2h}"
WORDS_FILE="pipeline/crossover_taboo_words.txt"
LOG="crossover_results/watch.log"
mkdir -p crossover_results
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

# Echo taboo words that have a published oracle AND host but are not yet in WORDS_FILE.
discover_new() {
  python3 - "$WORDS_FILE" <<'PY'
import json, sys, re, urllib.request
words_file = sys.argv[1]
have = {l.strip() for l in open(words_file) if l.strip() and not l.startswith("#")}
def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)
def files(repo):
    return {s["rfilename"] for s in get(f"https://huggingface.co/api/models/{repo}").get("siblings", [])}
def trained(repo):
    # A fully-trained oracle/MO has its actual adapter weights uploaded. A repo still
    # training (or just created) has only .gitattributes -> NOT ready, skip it.
    fs = files(repo)
    return "adapter_model.safetensors" in fs and "adapter_config.json" in fs
try:
    models = get("https://huggingface.co/api/models?search=gemma2_9b_it_taboo&limit=200")
except Exception as e:
    sys.stderr.write(f"discover: list failed: {e}\n"); sys.exit(0)
found = []
for m in models:
    mm = re.fullmatch(r"model-organisms-for-real/gemma2_9b_it_taboo_(.+)_oracle_v1", m["id"])
    if mm:
        found.append(mm.group(1))
new = []
for w in sorted(set(found) - have):
    try:
        if not trained(f"model-organisms-for-real/gemma2_9b_it_taboo_{w}_oracle_v1"):
            sys.stderr.write(f"discover: skip {w} (oracle still training / no weights)\n"); continue
        if not trained(f"bcywinski/gemma-2-9b-it-taboo-{w}"):
            sys.stderr.write(f"discover: skip {w} (host MO not ready)\n"); continue
    except Exception as e:
        sys.stderr.write(f"discover: skip {w} ({e})\n"); continue
    new.append(w)
print(" ".join(new))
PY
}

gpu_busy() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; }

log "watcher started (interval=$INTERVAL); wired words: $(tr '\n' ' ' < "$WORDS_FILE")"
while true; do
  NEW="$(discover_new 2>>"$LOG")"
  if [ -z "$NEW" ]; then
    log "no new taboo AOs"
  elif gpu_busy; then
    log "NEW found ($NEW) but GPU busy — deferring to next tick"
  else
    log "NEW taboo AOs: $NEW — wiring in and launching crossover"
    for w in $NEW; do echo "$w" >> "$WORDS_FILE"; done
    bash runs/run_crossover_cat.sh >>"$LOG" 2>&1 \
      && log "crossover run complete for: $NEW" \
      || log "crossover run FAILED (see run_crossover_cat.log)"
  fi
  sleep "$INTERVAL"
done
