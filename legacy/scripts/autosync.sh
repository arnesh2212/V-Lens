#!/bin/bash
# Spot-instance safety: results every 10 min, checkpoints every 30.
i=0
while true; do
  if [ $((i % 3)) -eq 0 ]; then bash /home/arnesg/vlm/scripts/pull.sh full
  else bash /home/arnesg/vlm/scripts/pull.sh; fi
  i=$((i+1)); sleep 600
done
