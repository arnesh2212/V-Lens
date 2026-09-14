#!/bin/bash
# after long.sh: the depicted-text attribution on the Authority yes/no items, both models
cd /workspace/vlm; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "DONE long" logs/long.log 2>/dev/null; do sleep 60; done
for R in Qwen/Qwen3-VL-8B-Instruct OpenGVLab/InternVL3-8B-hf; do
  echo "===== d11 $R $(date +%H:%M:%S)"
  python3 code/exp/d11_depicted.py --repo $R 2>&1 | grep --line-buffered -v "it/s\|Warning\|Loading weights\|Fetching"
done
echo "DONE depicted $(date +%H:%M:%S)"
