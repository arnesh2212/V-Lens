#!/usr/bin/env bash
# Restart just the shm lane, leaving the disk lane alone.
#
# Whatever this lane is running has to go, not one named experiment: a job
# left behind keeps its model on the card and the fresh runner then starts a
# second one beside it. This lane owns the models named below; everything else
# belongs to the disk lane and is never touched.
set -u
cd /workspace/temp/vlm
SHM_MODELS='Qwen3.5-27B|gemma-4'
pkill -f 'runq.sh queue/lane_'"shm.q" 2>/dev/null
sleep 1
for sig in "" "-9"; do
  for p in $(pgrep -f 'py''thon3 experiments' 2>/dev/null); do
    cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    echo "$cmd" | grep -qE "$SHM_MODELS" || continue      # not this lane's job
    kill $sig "$p" 2>/dev/null && echo "stopped ${sig:+(forced) }$(echo "$cmd" | sed 's/.*python3 //' | cut -c1-52)"
  done
  sleep 4
done
nohup setsid bash tools/runq.sh queue/lane_shm.q >/dev/null 2>&1 </dev/null &
sleep 6
echo "shm runners: $(pgrep -cf 'queue/lane_shm[.]q')  disk runners: $(pgrep -cf 'queue/lane_disk[.]q')"
ps -o etime=,cmd= -C python3 | grep experiments/ | sed 's|.*python3 ||' | cut -c1-56
