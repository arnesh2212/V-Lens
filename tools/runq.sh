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
# a job that forgets to name a cache must not fall back to the small one
# on the root filesystem and try to download a model into 1.5 GB
export HF_HOME="${HF_HOME:-/workspace/temp/vlm/hf}"
mkdir -p logs
RUNID=$(date +%m%d-%H%M%S)
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
  # the line is marked done when the job ends, not when it starts: a runner
  # restarted mid-job then runs it again, and the experiments that take hours
  # resume from where they were rather than starting over.  A line that has
  # been attempted this many times is let go, so a job that fails for its own
  # reasons cannot hold the queue.
  att=$(grep -Fxc -- "$line" "$Q.attempts" 2>/dev/null); att=${att:-0}
  if [ "$att" -ge "${MAX_ATTEMPTS:-3}" ]; then
    echo "[runq $NAME] $(date +%H:%M:%S) giving up on: $line" | tee -a "logs/$NAME.index"
    echo "$line" >> "$Q.done.lines"
    continue
  fi
  echo "$line" >> "$Q.attempts"
  n=$((n+1))
  # the run stamp keeps a restart from writing over the evidence of the last one
  log="logs/$NAME.$RUNID.$n.log"
  # wait for room before starting, so a lane never pushes another out of memory
  if [ -n "${NEED_FREE_MIB:-}" ]; then
    for _ in $(seq 1 360); do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
      free=$(( total - used ))
      [ "$free" -ge "$NEED_FREE_MIB" ] && break
      sleep 60
    done
  fi
  echo "[runq $NAME] $(date +%H:%M:%S) start #$n: $line" | tee -a "logs/$NAME.index"
  bash -c "$line" > "$log" 2>&1
  rc=$?
  # a job that died because the other lane had the card is worth retrying once
  # there is room; a job that died for its own reasons is not.  A full card
  # announces itself in several ways -- torch, triton and the allocator each
  # word it differently -- so the test is the words they have in common.
  # Only the end of the log is read, and the allocator's warning that a
  # mapping failed is not counted: it is printed by jobs that carried on
  # fine, and a job that later died of something else is not out of memory.
  tries=0
  while [ "$rc" -ne 0 ] && [ "$tries" -lt "${OOM_RETRIES:-3}" ] \
        && tail -n 60 "$log" | grep -v "CUDACachingAllocator\|expandable_segments" \
           | grep -qiE "out of memory|OutOfMemory|ALLOC_FAILED|cudaErrorMemoryAllocation"; do
    tries=$((tries+1))
    echo "[runq $NAME] $(date +%H:%M:%S) #$n ran out of memory, waiting 5 min (retry $tries/${OOM_RETRIES:-3})" | tee -a "logs/$NAME.index"
    sleep 300
    bash -c "$line" > "$log" 2>&1
    rc=$?
  done
  # only a job that finished on its own is done.  A job that was killed -- by
  # a restart, by the machine -- goes back in the queue and, being resumable,
  # picks up from its own checkpoint; the attempts file above is what stops a
  # genuinely broken one from looping.
  if [ "$rc" -eq 0 ]; then
    echo "$line" >> "$Q.done.lines"
  fi
  echo "[runq $NAME] $(date +%H:%M:%S) done  #$n rc=$rc  -> $log" | tee -a "logs/$NAME.index"
  printf '%s\trc=%s\t%s\n' "$(date +%H:%M:%S)" "$rc" "$line" >> "$Q.done"
done
