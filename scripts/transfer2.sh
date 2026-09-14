#!/bin/bash
# after authority2.sh: d06 Part X again (the yes/no axis was indexed on a [1, V] logit row; fixed)
cd /workspace/vlm; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "DONE authority2" logs/authority2.log 2>/dev/null; do sleep 60; done
for R in Qwen/Qwen3-VL-8B-Instruct OpenGVLab/InternVL3-8B-hf; do
  echo "===== d06 X $R $(date +%H:%M:%S)"
  python3 code/exp/d06_more.py --repo $R --parts X 2>&1 | grep --line-buffered -v "it/s\|Warning\|Loading weights\|Fetching"
done
echo "DONE transfer2 $(date +%H:%M:%S)"
