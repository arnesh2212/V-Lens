#!/usr/bin/env bash
# Stop the third lane and any job it left behind.
set -u
pkill -f 'queue/lane_l''ight.q' 2>/dev/null
sleep 1
for p in $(pgrep -f 'py''thon3 experiments/' 2>/dev/null); do
  if tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -qE 'hallucination|handoff|occlusion|reference|transfer|prefill|crossset'; then
    kill "$p" 2>/dev/null; echo "stopped $p"
  fi
done
sleep 4
nvidia-smi --query-gpu=memory.used --format=csv,noheader
