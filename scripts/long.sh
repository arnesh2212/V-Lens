#!/bin/bash
# after transfer.sh: the Authority open prompts again with room to answer (160 tokens), then the judge on them
cd /workspace/vlm; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "DONE transfer" logs/transfer.log 2>/dev/null; do sleep 60; done
mkdir -p results/vlens_long
for R in Qwen/Qwen3-VL-8B-Instruct OpenGVLab/InternVL3-8B-hf; do
  echo "===== d01 long $R $(date +%H:%M:%S)"
  python3 code/exp/d01_behaviour.py --repo $R --families gemini_auth gemini_swap gemini_removal gemini_eval --gen-open 160 --no-states --out /workspace/vlm/results/vlens_long 2>&1 | grep --line-buffered -v "it/s\|Warning\|Loading weights\|Fetching"
done
echo "===== judge long $(date +%H:%M:%S)"
python3 code/exp/d02_judge.py results/vlens_long/*.json 2>&1 | grep --line-buffered -v "it/s\|Warning\|Loading weights\|Fetching"
echo "DONE long $(date +%H:%M:%S)"
