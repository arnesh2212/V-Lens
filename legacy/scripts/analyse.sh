#!/bin/bash
# Everything that turns result files into paper material, in one call.
cd /workspace/vlm
export PYTHONPATH=/workspace/vlm/code HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
TAG=${1:-qwen2b}
python code/exp/13_report.py  results $TAG          2>&1 | tail -60
python code/exp/08_makefigs.py results figures $TAG 2>&1 | tail -25
python code/exp/19_examples.py results $TAG results/examples.html 2>&1 | tail -3
python code/convert_figs.py                         2>&1 | tail -3
