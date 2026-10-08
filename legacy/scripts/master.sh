#!/bin/bash
# One sequential GPU queue.  Concurrency on a single 24 GB card bought nothing --
# the card is compute-saturated either way -- and cost an OOM cascade, so
# everything runs one at a time, ordered by how load-bearing the result is.
cd /workspace/vlm
export HF_HOME=/workspace/hf PYTHONPATH=code HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L2=results/lenses/Qwen3-VL-2B-Instruct__mm__all__n120.pt
LT=results/lenses/Qwen3-VL-2B-Instruct__text__all__n120.pt
L8=results/lenses/Qwen3-VL-8B-Instruct__mm__all__n100.pt
LI=results/lenses/InternVL3_5-8B-HF__mm__all__n80.pt
log(){ echo "=== $(date +%H:%M:%S) $*" >> logs/master.log; }
run(){ log "START $*"; python "$@" >> logs/master.log 2>&1; log "rc=$? $1"; }

# --- the 2B story, complete -------------------------------------------------
run code/exp/04_blindsight.py   --lens $L2 --n 110 --tag qwen2b
run code/exp/16_collateral.py   --lens $L2 --n 110 --tag qwen2b
run code/exp/17_patch.py        --lens $L2 --concepts 12 --per-concept 36 \
                                --test-per-concept 10 --tag qwen2b
run code/exp/18_reality.py      --lens $L2 --concepts 14 --per-concept 45 --tag qwen2b
run code/exp/15_families.py     --lens $L2 --per-family 80 --tag qwen2b
run code/exp/06_forgery.py      --lens $L2 --n 30 --mode plant --tag qwen2b_plant
log "CORE2B DONE"

# --- the biggest model that fits --------------------------------------------
run code/exp/01_fit_lens.py --repo Qwen/Qwen3-VL-8B-Instruct --kind mm --n 100 \
    --dim-batch 4 --max-pixels 147456 --max-seconds 12600
run code/exp/10_routes.py      --repo Qwen/Qwen3-VL-8B-Instruct --lens $L8 \
    --max-pixels 147456 --n 90 --tag qwen8b
run code/exp/04_blindsight.py  --repo Qwen/Qwen3-VL-8B-Instruct --lens $L8 \
    --max-pixels 147456 --n 70 --tag qwen8b
run code/exp/17_patch.py       --repo Qwen/Qwen3-VL-8B-Instruct --lens $L8 \
    --max-pixels 147456 --concepts 10 --per-concept 30 --test-per-concept 8 --tag qwen8b
run code/exp/18_reality.py     --repo Qwen/Qwen3-VL-8B-Instruct --lens $L8 \
    --max-pixels 147456 --concepts 12 --per-concept 35 --tag qwen8b
run code/exp/07_patchmap.py    --repo Qwen/Qwen3-VL-8B-Instruct --lens $L8 \
    --max-pixels 147456 --n 4
log "BIG DONE"

# --- supporting material ----------------------------------------------------
run code/exp/14_perceptual.py  --lens $L2 --concepts 12 --per-concept 36 --tag qwen2b
run code/exp/09_competition.py --lens $L2 --n 400 --tag qwen2b
run code/exp/03_gate0.py       --lens $L2 --n 120 --tag qwen2b
run code/exp/12_lensqual.py    --lens $L2 --other $LT \
    --fitlog logs/fit_Qwen3-VL-2B-Instruct__mm__all__n120.log --n 50 --tag qwen2b
run code/exp/11_leading.py     --lens $L2 --n 120 --tag qwen2b
run code/exp/06_forgery.py     --lens $L2 --n 30 --mode erase --tag qwen2b_erase
log "SUPPORT DONE"

# --- a second family at the same scale, if the clock allows -----------------
run code/exp/01_fit_lens.py --repo OpenGVLab/InternVL3_5-8B-HF --kind mm --n 80 \
    --dim-batch 4 --max-seconds 9000
run code/exp/10_routes.py      --repo OpenGVLab/InternVL3_5-8B-HF --lens $LI \
    --n 80 --tag internvl8b
run code/exp/04_blindsight.py  --repo OpenGVLab/InternVL3_5-8B-HF --lens $LI \
    --n 60 --tag internvl8b
log "ALL DONE"
