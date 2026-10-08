#!/usr/bin/env bash
# Emit one line whenever a job starts, finishes, fails, or a lane dies.
cd /workspace/temp/vlm
prev=""
while true; do
  now=""
  for q in lane_disk lane_shm lane_light; do
    alive=$(pgrep -cf "queue/$q[.]q" 2>/dev/null | head -1); alive=${alive:-0}
    [ "$alive" -lt 1 ] && now="$now
LANE DOWN: $q"
    tail -3 "logs/$q.index" 2>/dev/null | while read -r l; do echo "$l"; done > /tmp/.$q.tail
    now="$now
$(cat /tmp/.$q.tail 2>/dev/null)"
  done
  errs=$(grep -lE "Traceback|OutOfMemoryError|NameError|KeyError" logs/lane_*.log 2>/dev/null | tail -2)
  [ -n "$errs" ] && now="$now
ERROR IN: $errs"
  if [ "$now" != "$prev" ]; then
    echo "$now" | grep -vE "^$" | tail -4
    prev="$now"
  fi
  sleep 120
done
