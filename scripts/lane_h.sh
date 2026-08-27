#!/bin/bash
# Lane H: cues, introspection, and the viewer data.  Holds until a slot is free
# rather than adding a fifth job -- four fills 24 GB, five OOMs.
cd /workspace/vlm
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/vlm/code HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L2=results/lenses/Qwen3-VL-2B-Instruct__mm__all__n120.pt
log(){ echo "=== $(date +%H:%M:%S) H $*" >> logs/master.log; }
slot(){ while true; do
    n=$(pgrep -cf 'code/exp/'); free=$(nvidia-smi --query-gpu=memory.free \
        --format=csv,noheader,nounits | head -1)
    [ "$n" -lt 4 ] && [ "$free" -gt 6000 ] && break
    sleep 60
  done; }
step(){ local mk="$1"; shift
  if [ -s "$mk" ]; then log "SKIP $mk"; else
    slot; log "START $*"; python "$@" >> logs/master.log 2>&1; log "rc=$? $1"; fi; }
step results/cues/qwen2b_summary.csv       code/exp/24_cues.py       --lens $L2 --n 110 --tag qwen2b
step results/introspect/qwen2b_summary.csv code/exp/25_introspect.py --lens $L2 --n 110 --tag qwen2b
step results/viewer.json                   code/exp/26_viewer.py     --lens $L2 --n 10 --out results/viewer.json
echo "=== $(date +%H:%M:%S) LANE H DONE" >> logs/master.log
