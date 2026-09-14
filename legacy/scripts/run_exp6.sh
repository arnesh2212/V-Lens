#!/bin/bash
# Concept families: the generality check beyond concrete nouns.
cd /workspace/vlm
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/vlm/code HF_HUB_OFFLINE=1
LENS=results/lenses/Qwen3-VL-2B-Instruct__mm__all__n120.pt
while ! grep -q "STAGE1 DONE" logs/exp.log 2>/dev/null; do sleep 60; done
log(){ echo "=== $(date +%H:%M:%S) $*" >> logs/exp.log; }
run(){ log "$*"; python "$@" >> logs/exp.log 2>&1; log "rc=$?"; }
run code/exp/15_families.py --lens $LENS --per-family 80 --tag qwen2b
echo "STAGE6 DONE $(date)" >> logs/exp.log
