#!/usr/bin/env bash
# Start the lanes that are not already up. Kept in a file because a pattern
# typed on the command line matches the shell that is typing it.
#
# Three lanes, not two. These are per-record loops with small batches, so one
# of them leaves the card with work to spare; a third that waits for room
# before it starts anything uses that without pushing the other two out.
cd /workspace/temp/vlm
LANES=${LANES:-"lane_disk lane_shm lane_light"}
for q in $LANES; do
  [ -f "queue/$q.q" ] || continue
  n=$(pgrep -cf "queue/$q[.]q" 2>/dev/null | head -1)
  n=${n:-0}
  if [ "$n" -lt 1 ]; then
    if [ "$q" = "lane_light" ]; then
      NEED_FREE_MIB=${NEED_FREE_MIB:-26000} OOM_RETRIES=6 \
        nohup setsid bash tools/runq.sh "queue/$q.q" >/dev/null 2>&1 </dev/null &
    else
      nohup setsid bash tools/runq.sh "queue/$q.q" >/dev/null 2>&1 </dev/null &
    fi
    echo "started $q"
  else
    echo "$q already running ($n)"
  fi
done
sleep 8
for q in $LANES; do
  [ -f "queue/$q.q" ] || continue
  echo "$q: $(pgrep -cf "queue/$q[.]q" | head -1) process(es)"
done
