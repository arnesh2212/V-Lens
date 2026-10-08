#!/bin/bash
# Watch a queue log on the box: one status line every two minutes into logs/status.txt,
# an ALERT line on a traceback, a CUDA error or a queue that died before its DONE marker.
#   nohup bash tools/boxmon.sh logs/datasets.log "DONE datasets" > /dev/null 2>&1 &
LOG=${1:?log file}; MARK=${2:-DONE}
cd "$(dirname "$0")/.."; mkdir -p logs
while true; do
  alive=$(pgrep -fc "experiments/(datasets|vlens)/[a-z_]+\.py")
  err=$(grep -c "Traceback\|out of memory\|CUDA error" "$LOG" 2>/dev/null)
  done=$(grep -c "$MARK" "$LOG" 2>/dev/null)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  line="$(date +%H:%M) alive=$alive err=$err done=$done gpu=${mem}MiB | $(tail -1 "$LOG" 2>/dev/null | cut -c1-70)"
  echo "$line" >> logs/status.txt
  if [ "$err" != "0" ] || { [ "$alive" = "0" ] && [ "$done" = "0" ]; }; then echo "ALERT $line" >> logs/status.txt; fi
  [ "$done" != "0" ] && { echo "FINISHED $line" >> logs/status.txt; break; }
  sleep 120
done
