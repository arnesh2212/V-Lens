#!/bin/bash
# Lane B.  Four light jobs fit in 24 GB and keep the card busy; the big model
# gets it to itself afterwards.  Every step is skipped when its output exists,
# so a preemption costs at most the step in flight.
cd /workspace/vlm
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/vlm/code HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L2=results/lenses/Qwen3-VL-2B-Instruct__mm__all__n120.pt
LT=results/lenses/Qwen3-VL-2B-Instruct__text__all__n120.pt
log(){ echo "=== $(date +%H:%M:%S) B $*" >> logs/master.log; }
step(){ local mk="$1"; shift
  if [ -s "$mk" ]; then log "SKIP $mk"; else
    log "START $*"; python "$@" >> logs/master.log 2>&1; log "rc=$? $1"; fi; }
step results/patch/qwen2b_summary.csv code/exp/17_patch.py --lens $L2 --concepts 12 --per-concept 36 --test-per-concept 10 --tag qwen2b
step results/reality/qwen2b_summary.csv code/exp/18_reality.py --lens $L2 --concepts 14 --per-concept 45 --tag qwen2b
step results/families/qwen2b_summary.csv code/exp/15_families.py --lens $L2 --per-family 80 --tag qwen2b
echo "=== $(date +%H:%M:%S) LANE B DONE" >> logs/master.log
