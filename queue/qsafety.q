# The safety spine: refusal on typographic image jailbreaks, with the benign-alarming control.
# Data lives in RAM at /dev/shm/vlens_safety so it costs no disk.
JVLM_SAFETY_ROOT=/dev/shm/vlens_safety JVLM_HARD_ROOT=/dev/shm/vlens_safety HF_HOME=/workspace/temp/vlm/hf python3 experiments/safety/refusal.py --repo Qwen/Qwen3-VL-8B-Instruct --n 16 --layers 5 --fracs 0.5 --alphas 0.2 --patch-items 4 --boot 40 --out results/smoke
JVLM_SAFETY_ROOT=/dev/shm/vlens_safety JVLM_HARD_ROOT=/dev/shm/vlens_safety HF_HOME=/workspace/temp/vlm/hf python3 experiments/safety/refusal.py --repo Qwen/Qwen3-VL-8B-Instruct --n 240 --variant TYPO --out results/safety
JVLM_SAFETY_ROOT=/dev/shm/vlens_safety JVLM_HARD_ROOT=/dev/shm/vlens_safety HF_HOME=/workspace/temp/vlm/hf python3 experiments/safety/refusal.py --repo Qwen/Qwen3-VL-8B-Instruct --n 240 --variant SD_TYPO --stages read patch --out results/safety_sdtypo
