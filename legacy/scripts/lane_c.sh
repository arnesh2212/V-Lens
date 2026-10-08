#!/bin/bash
# Lane C.  Four light jobs fit in 24 GB and keep the card busy; the big model
# gets it to itself afterwards.  Every step is skipped when its output exists,
# so a preemption costs at most the step in flight.
cd /workspace/vlm
export HF_HOME=/workspace/hf PYTHONPATH=code HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L2=results/lenses/Qwen3-VL-2B-Instruct__mm__all__n120.pt
LT=results/lenses/Qwen3-VL-2B-Instruct__text__all__n120.pt
log(){ echo "=== $(date +%H:%M:%S) C $*" >> logs/master.log; }
step(){ local mk="$1"; shift
  if [ -s "$mk" ]; then log "SKIP $mk"; else
    log "START $*"; python "$@" >> logs/master.log 2>&1; log "rc=$? $1"; fi; }
step results/source/qwen2b_summary.csv code/exp/23_source.py --lens $L2 --n 300 --tag qwen2b
step results/language/qwen2b_summary.csv code/exp/22_language.py --lens $L2 --n 100 --tag qwen2b
step results/depthcross/qwen2b_summary.csv code/exp/20_depthcross.py --lens $L2 --n 100 --tag qwen2b
step results/energy/qwen2b_summary.csv code/exp/21_energy.py --lens $L2 --n 100 --tag qwen2b
step results/gate0/qwen2b_summary.csv code/exp/03_gate0.py --lens $L2 --n 120 --tag qwen2b
echo "=== $(date +%H:%M:%S) LANE C DONE" >> logs/master.log
