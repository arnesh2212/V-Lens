#!/usr/bin/env bash
pkill -f 'refus''al.py --repo Qwen/Qwen3-VL-8B-Instruct --n 240 --variant TYPO' 2>/dev/null
sleep 3
echo "left: $(pgrep -cf 'refus''al.py')"
