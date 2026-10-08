#!/bin/bash
# Lane F: is the production channel language-specific?
cd /workspace/vlm
export HF_HOME=/workspace/hf PYTHONPATH=code HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L2=results/lenses/Qwen3-VL-2B-Instruct__mm__all__n120.pt
log(){ echo "=== $(date +%H:%M:%S) F $*" >> logs/master.log; }
if [ -s results/language/qwen2b_summary.csv ]; then log "SKIP language"; else
  log "START language"
  python code/exp/22_language.py --lens $L2 --n 100 --tag qwen2b >> logs/master.log 2>&1
  log "rc=$? language"
fi
echo "=== $(date +%H:%M:%S) LANE F DONE" >> logs/master.log
