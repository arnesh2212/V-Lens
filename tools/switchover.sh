#!/usr/bin/env bash
# Hand the card from the old lanes to the single ordered queue, gracefully.
#
# The old lanes are trimmed to the job each is running, so they start nothing
# new; this waits for those jobs to end, gives the speed fixes a window to land
# (queue/main_one.ready), then starts one runner on queue/main_one.q with a
# keeper beside it. It runs on the box, so it survives the session that
# started it.
#
#   nohup setsid bash tools/switchover.sh > logs/switchover.log 2>&1 < /dev/null &
set -u
cd /workspace/temp/vlm
echo "$(date '+%m-%d %H:%M') waiting for the running jobs to end"
while pgrep -f "python3 (tools|experiments)/" > /dev/null; do sleep 60; done
echo "$(date '+%m-%d %H:%M') card is free; waiting up to 90 min for the speed fixes"
for _ in $(seq 1 90); do [ -f queue/main_one.ready ] && break; sleep 60; done
[ -f queue/main_one.ready ] && echo "  speed fixes are in" || echo "  starting without them"
pkill -f "runq.sh queue/lane_[a-z]+[.]q" ; pkill -f "keep_[l]anes.sh" ; sleep 2
cat queue/lane_*.q.done.lines queue/main_one.q.done.lines 2>/dev/null | sort -u > queue/main_one.q.done.lines.new
mv queue/main_one.q.done.lines.new queue/main_one.q.done.lines
LANES=main_one bash tools/start_lanes.sh
LANES=main_one nohup setsid bash tools/keep_lanes.sh > /dev/null 2>&1 < /dev/null &
echo "$(date '+%m-%d %H:%M') switched: one lane on queue/main_one.q"
