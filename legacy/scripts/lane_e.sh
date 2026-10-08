#!/bin/bash
# Lane E: the biggest model that fits, with the card to itself.
cd /workspace/vlm
export HF_HOME=/workspace/hf PYTHONPATH=code HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L8=results/lenses/Qwen3-VL-8B-Instruct__mm__all__n100.pt
LI=results/lenses/InternVL3_5-8B-HF__mm__all__n80.pt
MP="--max-pixels 147456"
log(){ echo "=== $(date +%H:%M:%S) E $*" >> logs/master.log; }
step(){ local mk="$1"; shift
  if [ -s "$mk" ]; then log "SKIP $mk"; else
    log "START $*"; python "$@" >> logs/master.log 2>&1; log "rc=$? $1"; fi; }
for l in A B C D; do
  while ! grep -q "LANE $l DONE" logs/master.log 2>/dev/null; do sleep 60; done
done
while pgrep -f "code/exp/" >/dev/null; do sleep 30; done
step $L8 code/exp/01_fit_lens.py --repo Qwen/Qwen3-VL-8B-Instruct --kind mm --n 100 --dim-batch 4 $MP --max-seconds 12600
Q8="--repo Qwen/Qwen3-VL-8B-Instruct --lens $L8 $MP"
step results/routes/qwen8b_summary.csv      code/exp/10_routes.py     $Q8 --n 90 --tag qwen8b
step results/blindsight/qwen8b_summary.csv  code/exp/04_blindsight.py $Q8 --n 70 --tag qwen8b
step results/patch/qwen8b_summary.csv       code/exp/17_patch.py      $Q8 --concepts 10 --per-concept 30 --test-per-concept 8 --tag qwen8b
step results/source/qwen8b_summary.csv      code/exp/23_source.py     $Q8 --n 200 --tag qwen8b
step results/language/qwen8b_summary.csv    code/exp/22_language.py   $Q8 --n 80 --tag qwen8b
python code/exp/07_patchmap.py $Q8 --n 4 >> logs/master.log 2>&1
echo "=== $(date +%H:%M:%S) LANE E DONE" >> logs/master.log
step $LI code/exp/01_fit_lens.py --repo OpenGVLab/InternVL3_5-8B-HF --kind mm --n 80 --dim-batch 4 --max-seconds 9000
QI="--repo OpenGVLab/InternVL3_5-8B-HF --lens $LI"
step results/routes/internvl8b_summary.csv     code/exp/10_routes.py     $QI --n 80 --tag internvl8b
step results/blindsight/internvl8b_summary.csv code/exp/04_blindsight.py $QI --n 60 --tag internvl8b
echo "=== $(date +%H:%M:%S) ALL DONE" >> logs/master.log
