# Qwen3-VL-8B-Instruct -- weights on disk. Serial; the card only has room for two queues.
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/lens_bench.py --repo Qwen/Qwen3-VL-8B-Instruct --corpus 40 --heldout 8 --layers 4 --steps 10 --tokens-per-step 16384 --lr-probe-steps 6 --jac-prompts 2 --latent-captions 800 --families openimages_orig_eval --boot 40 --out results/smoke
# EXP3 + ablations A1 and A2: is a patch V-Write calls important actually carrying the answer
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/patch_causal.py --repo Qwen/Qwen3-VL-8B-Instruct --n 400 --pool 3000 --out results/patchcausal
# the safety spine, smoke first
JVLM_SAFETY_ROOT=/dev/shm/vlens_safety JVLM_HARD_ROOT=/dev/shm/vlens_safety HF_HOME=/workspace/temp/vlm/hf python3 experiments/safety/refusal.py --repo Qwen/Qwen3-VL-8B-Instruct --n 16 --layers 5 --fracs 0.5 --alphas 0.2 --patch-items 4 --boot 40 --out results/smoke
JVLM_SAFETY_ROOT=/dev/shm/vlens_safety JVLM_HARD_ROOT=/dev/shm/vlens_safety HF_HOME=/workspace/temp/vlm/hf python3 experiments/safety/refusal.py --repo Qwen/Qwen3-VL-8B-Instruct --n 240 --variant TYPO --out results/safety
# the headline lens comparison at the tuned lens paper's own token budget
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/lens_bench.py --repo Qwen/Qwen3-VL-8B-Instruct --corpus 900 --heldout 120 --layers 16 --steps 250 --tokens-per-step 262144 --jac-prompts 24 --latent-captions 6000 --boot 300 --out results/lensbench
# the within-family cross-fitted regime, every family
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/lens_read.py --repo Qwen/Qwen3-VL-8B-Instruct --boot 300 --out results/lensread
# full data: every one of the 19,056 records, states on a 20-block grid in RAM
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/behaviour.py --repo Qwen/Qwen3-VL-8B-Instruct --cap 0 --batch 16 --state-layers 20 --out /dev/shm/vlens_results/simonsays_full
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/lens_bench.py --repo Qwen/Qwen3-VL-8B-Instruct --results /dev/shm/vlens_results/simonsays_full --corpus 900 --heldout 120 --layers 16 --steps 250 --tokens-per-step 262144 --jac-prompts 24 --latent-captions 6000 --boot 300 --out results/lensbench_full
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/lens_read.py --repo Qwen/Qwen3-VL-8B-Instruct --results /dev/shm/vlens_results/simonsays_full --boot 300 --out results/lensread_full
JVLM_SAFETY_ROOT=/dev/shm/vlens_safety JVLM_HARD_ROOT=/dev/shm/vlens_safety HF_HOME=/workspace/temp/vlm/hf python3 experiments/safety/refusal.py --repo Qwen/Qwen3-VL-8B-Instruct --n 240 --variant SD_TYPO --stages read patch --out results/safety_sdtypo
