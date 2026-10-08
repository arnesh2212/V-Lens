#!/usr/bin/env bash
# Start both lanes if they are not already up. Kept in a file because a pattern
# typed on the command line matches the shell that is typing it.
cd /workspace/temp/vlm
for q in lane_disk lane_shm; do
  n=$(pgrep -cf "queue/$q[.]q" 2>/dev/null | head -1)
  n=${n:-0}
  if [ "$n" -lt 1 ]; then
    nohup setsid bash tools/runq.sh "queue/$q.q" >/dev/null 2>&1 </dev/null &
    echo "started $q"
  else
    echo "$q already running ($n)"
  fi
done
sleep 8
for q in lane_disk lane_shm; do
  echo "$q: $(pgrep -cf "queue/$q[.]q" | head -1) process(es)"
done
