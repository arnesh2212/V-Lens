#!/bin/bash
# Stage 3: lens quality, once the text-fitted lens exists.
cd /workspace/vlm
export HF_HOME=/workspace/hf PYTHONPATH=code HF_HUB_OFFLINE=1
MM=results/lenses/Qwen3-VL-2B-Instruct__mm__all__n120.pt
TXT=results/lenses/Qwen3-VL-2B-Instruct__text__all__n120.pt
while [ ! -f "$TXT" ]; do sleep 60; done
log(){ echo "=== $(date +%H:%M:%S) $*" >> logs/exp.log; }
run(){ log "$*"; python "$@" >> logs/exp.log 2>&1; log "rc=$?"; }
run code/exp/12_lensqual.py --lens $MM --other $TXT \
    --fitlog logs/fit_Qwen3-VL-2B-Instruct__mm__all__n120.log --n 60 --tag qwen2b
# the text-fitted lens read at image positions: does a transport that never saw
# an image still read the visual channel?
run code/exp/03_gate0.py --lens $TXT --n 120 --tag qwen2b_textlens
echo "STAGE3 DONE $(date)" >> logs/exp.log
