#!/bin/bash
# Keeps the queue moving.  Lines are "name|tries|min_free_mb|max_jobs|command".
#
# Traps this has already fallen into, all of them now avoided:
#  - `ps | grep <pattern>` matches the grep's own command line, so an
#    "is it already running" check written that way always says yes and every
#    job is skipped forever.  The check below matches on the script filename
#    via awk, which never sees itself.
#  - a job needs its memory requirement to travel with it: an 8B wants most of
#    the card, a 2B does not.
#  - the scan restarts from the top after every start, so the queue's order
#    is its priority order.  Without that, the line the scan is on when a
#    long job ends is the line that runs next.
#  - free memory is read per queue line, so a job further down can win a race
#    against one above it when memory is released mid-scan.  The settle sleep
#    after a start is what keeps that rare; it must be long enough for the new
#    process to claim its weights.
cd /workspace/vlm
Q=/workspace/vlm/queue.txt
LOG=/workspace/vlm/logs/watchdog.log
mkdir -p /workspace/vlm/logs

nreal ()  { ps -eo args | awk '/^python code\/exp\// {n++} END {print n+0}'; }
freemb () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits; }
isrun ()  { ps -eo args | awk -v s="$1" 'index($0, s) && /^python code\/exp\// {n++} END {exit !(n+0)}'; }

# A pid file, so stopping this never needs `pkill -f watchdog`.  That pattern
# matches any command line that merely mentions the log file, which in practice
# meant it killed the ssh session issuing it and left the queue unattended.
echo $$ > /workspace/vlm/logs/watchdog.pid
echo "$(date +%H:%M:%S) watchdog up (pid $$)" >> $LOG
while true; do
  [ -f $Q ] || { sleep 30; continue; }
  # Whether this pass has already passed over a job it could not start because
  # the card was busy.  If it has, and the card frees mid-pass, the job that
  # would start is whichever line the scan happens to have reached -- which is
  # how the twenty-eighth job in the queue once ran before the third.  Seeing
  # that flag set, the scan restarts from the top instead.
  busy_skip=0
  while IFS='|' read -r name tries_max need maxjobs cmd; do
    case "$name" in ''|'#'*) continue;; esac
    [ -f "/workspace/vlm/logs/${name}.done" ] && continue
    script=$(echo "$cmd" | grep -o 'code/exp/[0-9a-z_]*\.py' | head -1)
    tagarg=$(echo "$cmd" | sed -n 's/.*--tag \([^ ]*\).*/\1/p')
    isrun "$script --" && [ -n "$tagarg" ] && \
      ps -eo args | awk -v t="--tag $tagarg" 'index($0,t) && /^python/ {f=1} END {exit !f}' && continue
    t=$(cat /workspace/vlm/logs/${name}.tries 2>/dev/null || echo 0)
    [ "$t" -ge "$tries_max" ] && continue
    [ "$(nreal)" -ge "$maxjobs" ] && { busy_skip=1; continue; }
    [ "$busy_skip" = 1 ] && break        # rescan from the top, in queue order
    f=$(freemb)
    [ "$f" -lt "$need" ] && continue
    echo "$(date +%H:%M:%S) start $name  try $((t+1))/$tries_max  ${f}MB free" >> $LOG
    echo $((t+1)) > /workspace/vlm/logs/${name}.tries
    nohup bash -c "$cmd && touch /workspace/vlm/logs/${name}.done" \
      > logs/${name}.log 2>&1 &
    sleep 45
    # Back to the top of the queue rather than on down this pass.  The scan
    # runs top to bottom, so whichever line it happened to be sitting on when
    # the card came free was the line that started -- the first job to run
    # after a 37-minute fit was the eleventh in the queue, not the first.
    # Restarting the scan is what makes the order in queue.txt mean anything.
    break
  done < $Q
  sleep 8
done
