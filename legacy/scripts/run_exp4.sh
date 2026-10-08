#!/bin/bash
# Cross-model replication of the headline experiments, one model at a time as
# each lens lands.  A single-model result is a poster; the replication is what
# makes it a paper.
cd /workspace/vlm
export HF_HOME=/workspace/hf PYTHONPATH=code HF_HUB_OFFLINE=1
log(){ echo "=== $(date +%H:%M:%S) $*" >> logs/exp.log; }
run(){ log "$*"; python "$@" >> logs/exp.log 2>&1; log "rc=$?"; }

wait_for(){ while [ ! -f "$1" ]; do sleep 60; done; }

replicate(){   # repo, lens, tag
  wait_for "$2"
  run code/exp/04_blindsight.py --repo "$1" --lens "$2" --n 100 --tag "$3"
  run code/exp/10_routes.py     --repo "$1" --lens "$2" --n 80  --tag "$3"
  run code/exp/05_provenance.py --repo "$1" --lens "$2" --concepts 14 --per-concept 40 --tag "$3"
  run code/exp/07_patchmap.py   --repo "$1" --lens "$2" --n 4
}

replicate OpenGVLab/InternVL3_5-2B-HF \
  results/lenses/InternVL3_5-2B-HF__mm__all__n100.pt internvl2b
replicate Qwen/Qwen3-VL-4B-Instruct \
  results/lenses/Qwen3-VL-4B-Instruct__mm__all__n100.pt qwen4b
replicate Qwen/Qwen3-VL-8B-Instruct \
  results/lenses/Qwen3-VL-8B-Instruct__mm__all__n80.pt qwen8b
echo "REPLICATION DONE $(date)" >> logs/exp.log
