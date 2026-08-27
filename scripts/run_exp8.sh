#!/bin/bash
# The collateral law: geometry predicts which output channels an erasure costs.
cd /workspace/vlm
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/vlm/code HF_HUB_OFFLINE=1
LENS=results/lenses/Qwen3-VL-2B-Instruct__mm__all__n120.pt
log(){ echo "=== $(date +%H:%M:%S) $*" >> logs/exp.log; }
run(){ log "$*"; python "$@" >> logs/exp.log 2>&1; log "rc=$?"; }
# wait for the blindsight rerun to clear so the card is not oversubscribed
while pgrep -f "exp/04_blindsight" >/dev/null; do sleep 60; done
run code/exp/16_collateral.py --lens $LENS --n 110 --tag qwen2b
echo "STAGE8 DONE $(date)" >> logs/exp.log
