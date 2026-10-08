#!/bin/bash
# The whole custom-datasets pipeline for one model, in order, on one 24 GB card.
#
#   bash queue/datasets.sh Qwen/Qwen3-VL-8B-Instruct            # everything
#   SMOKE=1 bash queue/datasets.sh Qwen/Qwen3-VL-8B-Instruct    # every step with --limit 2, into results/smoke
#
# Every step writes JSON under results/vlens/ (results/vlens_long/ for the
# 160-token Authority replies) and prints a summary; the judge steps load the
# Qwen3-8B text judge instead of the VLM.  Run it for each model in turn and
# then build the page (see README).  Use tools/boxmon.sh to watch the log.
set -uo pipefail
R=${1:?repo id, e.g. Qwen/Qwen3-VL-8B-Instruct}
cd "$(dirname "$0")/.."; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=${OUT:-results/vlens}; LONG=${LONG:-results/vlens_long}; LIM=""
if [ "${SMOKE:-0}" = "1" ]; then OUT=results/smoke; LONG=results/smoke_long; LIM="--limit 2"; fi
NAME=${R##*/}
Q="grep --line-buffered -v it/s\|Warning\|Loading\ weights\|Fetching"
step() { echo "===== $1 $R $(date +%H:%M:%S)"; shift; "$@" 2>&1 | grep --line-buffered -v "it/s\|Warning\|Loading weights\|Fetching"; }
step behaviour  python3 experiments/datasets/behaviour.py --repo "$R" --out "$OUT" $LIM
step judge      python3 experiments/datasets/judge.py "$OUT"/"$NAME"__*.json
step lens       python3 experiments/datasets/lens.py --repo "$R" --results "$OUT" $LIM
step wmdp       python3 experiments/datasets/wmdp.py --repo "$R" --out "$OUT" $LIM
step steering   python3 experiments/datasets/steering.py --repo "$R" --results "$OUT" $LIM
step directions python3 experiments/datasets/directions.py --repo "$R" --results "$OUT" $LIM
step authority  python3 experiments/datasets/authority.py --repo "$R" --results "$OUT" $LIM
step depicted   python3 experiments/datasets/depicted.py --repo "$R" --results "$OUT" $LIM
# the Authority open prompts again, with room to answer; the report prefers these replies once judged
step long       python3 experiments/datasets/behaviour.py --repo "$R" --families gemini_auth gemini_swap gemini_removal gemini_eval --gen-open 160 --no-states --out "$LONG" $LIM
step judge-long python3 experiments/datasets/judge.py "$LONG"/"$NAME"__*.json
step bearer     python3 experiments/datasets/bearer.py --repo "$R" --results "$LONG" --fallback "$OUT"
echo "DONE datasets $R $(date +%H:%M:%S)"
