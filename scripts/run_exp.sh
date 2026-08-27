#!/bin/bash
# Experiment suite, ordered by how load-bearing each result is.
cd /workspace/vlm
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/vlm/code HF_HUB_OFFLINE=1
LENS=results/lenses/Qwen3-VL-2B-Instruct__mm__all__n120.pt
while [ ! -f "$LENS" ]; do sleep 30; done
log(){ echo "=== $(date +%H:%M:%S) $*" >> logs/exp.log; }
run(){ log "$*"; python "$@" >> logs/exp.log 2>&1; log "rc=$?"; }

run code/exp/07_patchmap.py   --lens $LENS --n 8
run code/exp/10_routes.py     --lens $LENS --n 120 --tag qwen2b
run code/exp/04_blindsight.py --lens $LENS --n 150 --tag qwen2b
run code/exp/05_provenance.py --lens $LENS --concepts 16 --per-concept 60 --tag qwen2b
run code/exp/06_forgery.py    --lens $LENS --n 40 --mode plant --tag qwen2b_plant
run code/exp/09_competition.py --lens $LENS --n 400 --tag qwen2b
run code/exp/03_gate0.py      --lens $LENS --n 150 --tag qwen2b
run code/exp/06_forgery.py    --lens $LENS --n 40 --mode erase --tag qwen2b_erase
echo "STAGE1 DONE $(date)" >> logs/exp.log
