#!/usr/bin/env bash
pkill -f 'lens_''bench.py --repo Qwen/Qwen3-VL-8B-Instruct --corpus 900' 2>/dev/null
sleep 4
echo "left: $(pgrep -cf 'lens_''bench.py')"
