#!/usr/bin/env bash
# A serial job queue, so the card is never waiting on me to type the next command.
#
#   tools/runq.sh <queue-file>
#
# The queue file holds one shell command per line; blank lines and lines
# starting with # are skipped.  Each line is run to completion, its output goes
# to logs/<queue>.<n>.log, and the line is appended to <queue-file>.done with
# its exit status.  Adding a line to the queue file while it runs is safe: the
# runner re-reads the file after every job, so work can be appended from
# outside without restarting anything.
set -u
Q="$1"
NAME="$(basename "$Q" .q)"
cd "$(dirname "$0")/.."
source /venv/main/bin/activate 2>/dev/null || true
export VLENS_DATA="${VLENS_DATA:-/workspace/temp/vlm/data_root/VLM V-Lens Final Dataset Complete}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p logs
touch "$Q.done"
n=0
idle=0
while true; do
  line=""
  while IFS= read -r l; do
    [ -z "$l" ] && continue
    case "$l" in \#*) continue ;; esac
    grep -Fxq -- "$l" "$Q.done.lines" 2>/dev/null && continue
    line="$l"; break
  done < "$Q"
  if [ -z "$line" ]; then
    # nothing left to do: wait rather than exit, so work appended to the queue
    # file from outside is picked up without restarting the runner
    idle=$((idle+1))
    if [ "$idle" -ge "${IDLE_TICKS:-720}" ]; then
      echo "[runq $NAME] idle for ${IDLE_TICKS:-720} ticks, stopping"; break
    fi
    sleep 10; continue
  fi
  idle=0
  echo "$line" >> "$Q.done.lines"
  n=$((n+1))
  log="logs/$NAME.$n.log"
  echo "[runq $NAME] $(date +%H:%M:%S) start #$n: $line" | tee -a "logs/$NAME.index"
  bash -c "$line" > "$log" 2>&1
  rc=$?
  # a job that died because another job had the card is worth one retry once
  # the card is free again; a job that died for its own reasons is not
  if [ "$rc" -ne 0 ] && grep -qiE "OutOfMemoryError|CUDA out of memory|memory mapping failed" "$log"; then
    echo "[runq $NAME] $(date +%H:%M:%S) #$n ran out of memory, waiting 5 min and retrying once" | tee -a "logs/$NAME.index"
    sleep 300
    bash -c "$line" > "$log" 2>&1
    rc=$?
  fi
  echo "[runq $NAME] $(date +%H:%M:%S) done  #$n rc=$rc  -> $log" | tee -a "logs/$NAME.index"
  printf '%s\trc=%s\t%s\n' "$(date +%H:%M:%S)" "$rc" "$line" >> "$Q.done"
done
