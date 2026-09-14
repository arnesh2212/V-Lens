#!/bin/bash
# Lens fits.  The 8B is the largest dense VLM that fits in 24 GB bf16, and it has
# to share the card with running experiments, so dim_batch is set low -- total
# FLOPs are unchanged, only the number of backward passes moves.
cd /workspace/vlm
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/vlm/code HF_HUB_OFFLINE=1
log(){ echo "=== $(date +%H:%M:%S) $*" >> logs/queue.log; }
run(){ log "$*"; python code/exp/01_fit_lens.py "$@" >> logs/queue.log 2>&1; log "rc=$?"; }
while pgrep -f "01_fit_lens.py" >/dev/null; do sleep 30; done
run --repo Qwen/Qwen3-VL-8B-Instruct --kind mm --n 100 --dim-batch 4 \
    --max-pixels 147456 --max-seconds 18000
run --repo OpenGVLab/InternVL3_5-8B-HF --kind mm --n 80 --dim-batch 4 \
    --max-seconds 12600
run --repo Qwen/Qwen3-VL-4B-Instruct --kind mm --n 100 --dim-batch 8
echo "ALL LENSES DONE $(date)" >> logs/queue.log
