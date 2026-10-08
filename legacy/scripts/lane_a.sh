#!/bin/bash
# Lane A.  Four light jobs fit in 24 GB and keep the card busy; the big model
# gets it to itself afterwards.  Every step is skipped when its output exists,
# so a preemption costs at most the step in flight.
cd /workspace/vlm
export HF_HOME=/workspace/hf PYTHONPATH=code HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L2=results/lenses/Qwen3-VL-2B-Instruct__mm__all__n120.pt
LT=results/lenses/Qwen3-VL-2B-Instruct__text__all__n120.pt
log(){ echo "=== $(date +%H:%M:%S) A $*" >> logs/master.log; }
step(){ local mk="$1"; shift
  if [ -s "$mk" ]; then log "SKIP $mk"; else
    log "START $*"; python "$@" >> logs/master.log 2>&1; log "rc=$? $1"; fi; }
step results/routes/qwen2b_summary.csv code/exp/10_routes.py --lens $L2 --n 110 --tag qwen2b
step results/blindsight/qwen2b_summary.csv code/exp/04_blindsight.py --lens $L2 --n 110 --tag qwen2b
step results/collateral/qwen2b_summary.csv code/exp/16_collateral.py --lens $L2 --n 110 --tag qwen2b
step results/lensqual/qwen2b_quality.csv code/exp/12_lensqual.py --lens $L2 --other $LT --fitlog logs/fit_Qwen3-VL-2B-Instruct__mm__all__n120.log --n 50 --tag qwen2b
echo "=== $(date +%H:%M:%S) LANE A DONE" >> logs/master.log
