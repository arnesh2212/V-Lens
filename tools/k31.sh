#!/usr/bin/env bash
pkill -f 'read_''image.py --repo Qwen/Qwen3-VL-8B-Instruct --n 0' 2>/dev/null
sleep 4
echo "left: $(pgrep -cf 'read_''image.py')"
