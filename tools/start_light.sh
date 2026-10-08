#!/usr/bin/env bash
# The third lane only starts a job when the card has room for it.
cd /workspace/temp/vlm
n=$(pgrep -cf "queue/lane_l[i]ght.q" 2>/dev/null | head -1); n=${n:-0}
if [ "$n" -lt 1 ]; then
  NEED_FREE_MIB=45000 nohup setsid bash tools/runq.sh queue/lane_light.q >/dev/null 2>&1 </dev/null &
  echo "light lane armed, waits for 45 GB free"
else
  echo "light lane already running"
fi
