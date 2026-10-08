#!/usr/bin/env bash
# Bring back any lane that has stopped while its queue still has work.
#
# The runner survives a job dying, but nothing survives the runner dying: a
# dropped connection, a killed session, a machine that hiccups, and a lane is
# simply gone with no sign except that its queue stops moving. This notices
# within five minutes.
#
#   nohup setsid bash tools/keep_lanes.sh >/dev/null 2>&1 </dev/null &
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
LANES=${LANES:-"lane_disk lane_shm lane_light"}
EVERY=${EVERY:-300}
while true; do
  for q in $LANES; do
    [ -f "queue/$q.q" ] || continue
    # a queue with nothing left in it is meant to be stopped
    left=$(python3 - "$q" <<'PY'
import os, sys
q = "queue/%s.q" % sys.argv[1]
lines = [l.strip() for l in open(q) if l.strip() and not l.lstrip().startswith("#")]
dl = q + ".done.lines"
done = {l.rstrip("\n") for l in open(dl)} if os.path.exists(dl) else set()
print(sum(1 for l in lines if l not in done))
PY
)
    [ "${left:-0}" -gt 0 ] || continue
    alive=$(pgrep -cf "queue/$q[.]q" 2>/dev/null | head -1); alive=${alive:-0}
    if [ "$alive" -lt 1 ]; then
      echo "$(date +%m-%d_%H:%M) $q was down with $left line(s) left; restarting" \
        >> logs/keep_lanes.log
      LANES="$q" bash tools/start_lanes.sh >> logs/keep_lanes.log 2>&1
    fi
  done
  sleep "$EVERY"
done
