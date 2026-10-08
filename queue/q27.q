# Qwen3.5-27B -- weights in /dev/shm, no disk cost. Behaviour pass is already running.
HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/behaviour.py --repo Qwen/Qwen3.5-27B --cap 120 --batch 12 --resume
HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/patch_causal.py --repo Qwen/Qwen3.5-27B --n 250 --pool 2500 --out results/patchcausal
HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/lens_bench.py --repo Qwen/Qwen3.5-27B --corpus 600 --heldout 100 --layers 16 --steps 250 --tokens-per-step 65536 --jac-prompts 12 --latent-captions 6000 --boot 300 --out results/lensbench
HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/lens_read.py --repo Qwen/Qwen3.5-27B --boot 300 --out results/lensread
# full data for the 27B, once the 8B has finished with the RAM disk
HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/behaviour.py --repo Qwen/Qwen3.5-27B --cap 0 --batch 16 --state-layers 20 --out /dev/shm/vlens_results/simonsays_full
HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/lens_bench.py --repo Qwen/Qwen3.5-27B --results /dev/shm/vlens_results/simonsays_full --corpus 600 --heldout 100 --layers 16 --steps 250 --tokens-per-step 65536 --jac-prompts 12 --latent-captions 6000 --boot 300 --out results/lensbench_full
HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/lens_read.py --repo Qwen/Qwen3.5-27B --results /dev/shm/vlens_results/simonsays_full --boot 300 --out results/lensread_full
