#!/bin/bash
# Lane D.  Four light jobs fit in 24 GB and keep the card busy; the big model
# gets it to itself afterwards.  Every step is skipped when its output exists,
# so a preemption costs at most the step in flight.
cd /workspace/vlm
export HF_HOME=/workspace/hf PYTHONPATH=code HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L2=results/lenses/Qwen3-VL-2B-Instruct__mm__all__n120.pt
LT=results/lenses/Qwen3-VL-2B-Instruct__text__all__n120.pt
log(){ echo "=== $(date +%H:%M:%S) D $*" >> logs/master.log; }
step(){ local mk="$1"; shift
  if [ -s "$mk" ]; then log "SKIP $mk"; else
    log "START $*"; python "$@" >> logs/master.log 2>&1; log "rc=$? $1"; fi; }
step results/forgery/qwen2b_plant_summary.json code/exp/06_forgery.py --lens $L2 --n 24 --mode plant --tag qwen2b_plant
step results/competition/qwen2b_summary.csv code/exp/09_competition.py --lens $L2 --n 400 --tag qwen2b
step results/perceptual/qwen2b_summary.csv code/exp/14_perceptual.py --lens $L2 --concepts 12 --per-concept 36 --tag qwen2b
echo "=== $(date +%H:%M:%S) LANE D DONE" >> logs/master.log
