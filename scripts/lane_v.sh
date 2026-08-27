#!/bin/bash
# Lane V: fit the V-Lens -- answer-targeted cotangent, image-sourced average.
# Same cost as the reference fit; the cotangent is just sparser.
cd /workspace/vlm
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/vlm/code HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log(){ echo "=== $(date +%H:%M:%S) V $*" >> logs/master.log; }
slot(){ while true; do
    n=$(pgrep -cf 'code/exp/'); free=$(nvidia-smi --query-gpu=memory.free \
        --format=csv,noheader,nounits | head -1)
    [ "$n" -lt 4 ] && [ "$free" -gt 6000 ] && break
    sleep 60
  done; }
V=results/lenses/Qwen3-VL-2B-Instruct__mm__image__answer__n120.pt
if [ ! -s "$V" ]; then
  slot; log "START vlens fit"
  python code/exp/01_fit_lens.py --repo Qwen/Qwen3-VL-2B-Instruct --kind mm \
      --n 120 --policy image --target answer >> logs/master.log 2>&1
  log "rc=$? vlens fit"
fi
echo "=== $(date +%H:%M:%S) LANE V DONE" >> logs/master.log
