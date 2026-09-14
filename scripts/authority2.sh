#!/bin/bash
# after depicted.sh: d09 again (the Image Index has rows without a file; fixed), both models
cd /workspace/vlm; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "DONE depicted" logs/depicted.log 2>/dev/null; do sleep 60; done
for R in Qwen/Qwen3-VL-8B-Instruct OpenGVLab/InternVL3-8B-hf; do
  echo "===== d09 $R $(date +%H:%M:%S)"
  python3 code/exp/d09_authority.py --repo $R 2>&1 | grep --line-buffered -v "it/s\|Warning\|Loading weights\|Fetching"
done
echo "DONE authority2 $(date +%H:%M:%S)"
