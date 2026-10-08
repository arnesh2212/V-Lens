#!/bin/bash
# The two experiments the theory review identified as decisive, plus the
# redesigned reality-monitoring probe.
cd /workspace/vlm
export HF_HOME=/workspace/hf PYTHONPATH=code HF_HUB_OFFLINE=1
LENS=results/lenses/Qwen3-VL-2B-Instruct__mm__all__n120.pt
log(){ echo "=== $(date +%H:%M:%S) $*" >> logs/exp.log; }
run(){ log "$*"; python "$@" >> logs/exp.log 2>&1; log "rc=$?"; }
while ! grep -q "STAGE1 DONE" logs/exp.log 2>/dev/null; do sleep 60; done
run code/exp/17_patch.py    --lens $LENS --concepts 12 --per-concept 36 \
    --test-per-concept 10 --tag qwen2b
run code/exp/18_reality.py  --lens $LENS --concepts 14 --per-concept 45 --tag qwen2b
echo "STAGE7 DONE $(date)" >> logs/exp.log
