#!/bin/bash
# Stage 1 from blindsight onward, on the batched (3x faster) code path.
cd /workspace/vlm
export HF_HOME=/workspace/hf PYTHONPATH=code HF_HUB_OFFLINE=1
LENS=results/lenses/Qwen3-VL-2B-Instruct__mm__all__n120.pt
log(){ echo "=== $(date +%H:%M:%S) $*" >> logs/exp.log; }
run(){ log "$*"; python "$@" >> logs/exp.log 2>&1; log "rc=$?"; }
# provenance is already in flight from the earlier queue
run code/exp/04_blindsight.py  --lens $LENS --n 150 --tag qwen2b
run code/exp/14_perceptual.py  --lens $LENS --concepts 12 --per-concept 36 --tag qwen2b
run code/exp/06_forgery.py     --lens $LENS --n 30 --mode plant --tag qwen2b_plant
run code/exp/09_competition.py --lens $LENS --n 400 --tag qwen2b
run code/exp/03_gate0.py       --lens $LENS --n 150 --tag qwen2b
run code/exp/11_leading.py     --lens $LENS --n 150 --tag qwen2b
run code/exp/06_forgery.py     --lens $LENS --n 30 --mode erase --tag qwen2b_erase
echo "STAGE1 DONE $(date)" >> logs/exp.log
