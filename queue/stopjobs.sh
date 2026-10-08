#!/usr/bin/env bash
set -u
pkill -f 'tools/r''unq.sh' 2>/dev/null
sleep 1
pkill -f 'py''thon3 experiments/' 2>/dev/null
sleep 4
pkill -9 -f 'py''thon3 experiments/' 2>/dev/null
sleep 2
echo "runners $(pgrep -cf 'tools/r''unq.sh') experiments $(pgrep -cf 'py''thon3 experiments/')"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
