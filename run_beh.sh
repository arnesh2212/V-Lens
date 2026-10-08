#!/bin/bash
cd /workspace/temp/vlm
source /venv/main/bin/activate
export HF_HOME=${HF:-/workspace/temp/vlm/hf}
export HF_HUB_OFFLINE=1
export VLENS_DATA="/workspace/temp/vlm/data_root/VLM V-Lens Final Dataset Complete"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 experiments/simonsays/behaviour.py --repo "$1" --cap "${2:-120}" --batch "${3:-16}" --resume
