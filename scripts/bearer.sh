#!/bin/bash
# after transfer2.sh: which bearer class each Authority reply followed (text judge), both models
cd /workspace/vlm; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "DONE transfer2" logs/transfer2.log 2>/dev/null; do sleep 60; done
for R in Qwen/Qwen3-VL-8B-Instruct OpenGVLab/InternVL3-8B-hf; do
  echo "===== d12 $R $(date +%H:%M:%S)"
  python3 code/exp/d12_bearer.py --repo $R 2>&1 | grep --line-buffered -v "it/s\|Warning\|Loading weights\|Fetching"
done
echo "DONE bearer $(date +%H:%M:%S)"
