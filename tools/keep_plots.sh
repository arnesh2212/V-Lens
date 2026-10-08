#!/usr/bin/env bash
# Bring the results down and redraw every figure, over and over.
#
# A figure drawn once is a figure that is wrong the moment the next result
# lands, and the run is days long. This keeps what is on this machine within
# one cycle of what is on the box, and writes a line whenever a figure is new
# or has changed so the log says when each one arrived.
#
#   nohup setsid bash tools/keep_plots.sh >/dev/null 2>&1 </dev/null &
set -u
cd "$(dirname "$0")/.."
EVERY=${EVERY:-600}
STATE=${STATE:-.plotsync}
mkdir -p "$STATE" logs
touch "$STATE/seen.txt"
while true; do
  bash tools/sync_and_plot.sh > "$STATE/out.txt" 2>&1
  md5sum figures/*.png 2>/dev/null | sort -k2 > "$STATE/cur.txt"
  {
    grep -E "BROKEN|^  !" "$STATE/out.txt"
    join -j2 -v2 "$STATE/seen.txt" "$STATE/cur.txt" 2>/dev/null | awk '{print "new     " $1}'
    join -j2 "$STATE/seen.txt" "$STATE/cur.txt" 2>/dev/null | awk '$2!=$3 {print "redrawn " $1}'
  } | sed "s/^/$(date +%m-%d_%H:%M) /" >> logs/plotsync.log
  cp "$STATE/cur.txt" "$STATE/seen.txt"
  sleep "$EVERY"
done
