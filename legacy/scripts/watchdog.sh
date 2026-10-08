#!/bin/bash
# Restart any lane that has died without finishing.  Every lane step is guarded
# by an output check, so a restart repeats at most the step that was in flight.
cd /workspace/vlm
while true; do
  sleep 300
  grep -q "ALL DONE" logs/master.log 2>/dev/null && exit 0
  for lane in a b c d e h; do
    U=$(echo $lane | tr a-h A-H)
    grep -q "LANE $U DONE" logs/master.log 2>/dev/null && continue
    [ "$lane" = "c" ] && grep -q "ALL DONE" logs/master.log 2>/dev/null && continue
    if ! pgrep -f "bash ./lane_$lane.sh" >/dev/null; then
      echo "=== $(date +%H:%M:%S) watchdog: restarting lane $U" >> logs/master.log
      setsid bash ./lane_$lane.sh >> logs/master.out 2>&1 < /dev/null &
      sleep 20
    fi
  done
done
