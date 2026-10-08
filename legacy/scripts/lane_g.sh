#!/bin/bash
# Lane G: the two lowest-value experiments, run only if the clock allows after
# the big model is done.  A leading question that induces no hallucination has
# nothing to cure, and the evasion direction of the attack is the dual of one we
# already have.
cd /workspace/vlm
export HF_HOME=/workspace/hf PYTHONPATH=code HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L2=results/lenses/Qwen3-VL-2B-Instruct__mm__all__n120.pt
log(){ echo "=== $(date +%H:%M:%S) G $*" >> logs/master.log; }
step(){ local mk="$1"; shift
  if [ -s "$mk" ]; then log "SKIP $mk"; else
    log "START $*"; python "$@" >> logs/master.log 2>&1; log "rc=$? $1"; fi; }
while ! grep -q "ALL DONE" logs/master.log 2>/dev/null; do sleep 120; done
step results/forgery/qwen2b_erase_summary.json code/exp/06_forgery.py --lens $L2 --n 25 --mode erase --tag qwen2b_erase
step results/leading/qwen2b_summary.json       code/exp/11_leading.py --lens $L2 --n 100 --tag qwen2b
echo "=== $(date +%H:%M:%S) LANE G DONE" >> logs/master.log
