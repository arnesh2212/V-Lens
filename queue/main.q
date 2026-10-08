# Full coverage: every experiment of the brief on every planned model. The
# naming transport -- the fix for the one place a descriptive lens beat us --
# leads, because it is the open question.
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/visual_tokens.py --repo Qwen/Qwen3-VL-8B-Instruct --n 250 --pool 2500 --naming 250 --out results/visualtokens
JVLM_SAFETY_ROOT=/dev/shm/vlens_safety JVLM_HARD_ROOT=/dev/shm/vlens_safety HF_HOME=/workspace/temp/vlm/hf python3 experiments/safety/refusal.py --repo Qwen/Qwen3-VL-8B-Instruct --n 400 --variant TYPO --fracs 0.3 0.5 0.7 --batch 8 --stages read --out results/safety_alllens
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/patch_causal.py --repo Qwen/Qwen3-VL-8B-Instruct --n 0 --pool 0 --baselines mean --out results/patchcausal_full
HF_HOME=/workspace/temp/vlm/hf python3 experiments/pwlock/finetune.py --repo Qwen/Qwen3-VL-8B-Instruct --out /dev/shm/vlens_pw/qwen8b --epochs 1 --batch 2 --accum 4
HF_HOME=/workspace/temp/vlm/hf python3 experiments/pwlock/evaluate.py --repo Qwen/Qwen3-VL-8B-Instruct --adapter /dev/shm/vlens_pw/qwen8b --stages E1 E2 --out results/smoke
HF_HOME=/workspace/temp/vlm/hf python3 experiments/pwlock/finetune.py --repo Qwen/Qwen3-VL-8B-Instruct --out /dev/shm/vlens_pw/qwen8b --epochs 4 --batch 2 --accum 4
JVLM_CLEVR_ROOT=/dev/shm/vlens_clevr HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/reference.py --repo Qwen/Qwen3-VL-8B-Instruct --n 6 --pool 200 --layers 5 --fillers 0 2 --out results/smoke
JVLM_CLEVR_ROOT=/dev/shm/vlens_clevr HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/reference.py --repo Qwen/Qwen3-VL-8B-Instruct --n 150 --pool 700 --out results/reference
JVLM_SAFETY_ROOT=/dev/shm/vlens_safety JVLM_HARD_ROOT=/dev/shm/vlens_safety HF_HOME=/workspace/temp/vlm/hf python3 experiments/safety/crossset.py --repo Qwen/Qwen3-VL-8B-Instruct --n 300 --batch 8 --out results/crossset
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/handoff.py --repo Qwen/Qwen3-VL-8B-Instruct --n 600 --pool 0 --out results/handoff_full
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/occlusion.py --repo Qwen/Qwen3-VL-8B-Instruct --n 600 --pool 0 --out results/occlusion_full
HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/patch_lenses.py --repo Qwen/Qwen3.5-27B --n 600 --corpus 400 --out results/patchlenses
HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/lens_read.py --repo Qwen/Qwen3.5-27B --boot 300 --out results/lensread
JVLM_SAFETY_ROOT=/dev/shm/vlens_safety JVLM_HARD_ROOT=/dev/shm/vlens_safety HF_HOME=/dev/shm/qwen_hf python3 experiments/safety/refusal.py --repo Qwen/Qwen3.5-27B --n 400 --variant TYPO --batch 8 --stages read --out results/safety_alllens
HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/patch_causal.py --repo Qwen/Qwen3.5-27B --n 0 --pool 0 --baselines mean --out results/patchcausal_full
HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/patch_causal.py --repo Qwen/Qwen3.5-27B --n 250 --pool 2500 --out results/patchcausal
JVLM_SAFETY_ROOT=/dev/shm/vlens_safety JVLM_HARD_ROOT=/dev/shm/vlens_safety HF_HOME=/dev/shm/qwen_hf python3 experiments/safety/crossset.py --repo Qwen/Qwen3.5-27B --n 300 --out results/crossset
JVLM_CLEVR_ROOT=/dev/shm/vlens_clevr HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/reference.py --repo Qwen/Qwen3.5-27B --n 120 --out results/reference
HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/occlusion.py --repo Qwen/Qwen3.5-27B --n 150 --pool 3000 --out results/occlusion
HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/lens_bench.py --repo Qwen/Qwen3.5-27B --corpus 600 --heldout 100 --layers 16 --steps 250 --tokens-per-step 65536 --jac-prompts 12 --latent-captions 6000 --boot 300 --out results/lensbench
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/lens_bench.py --repo Qwen/Qwen3-VL-8B-Instruct --corpus 900 --heldout 120 --layers 16 --steps 250 --tokens-per-step 262144 --jac-prompts 24 --latent-captions 6000 --boot 300 --log-every 10 --out results/lensbench
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/behaviour.py --repo Qwen/Qwen3-VL-8B-Instruct --cap 0 --batch 24 --state-layers 20 --out /dev/shm/vlens_results/simonsays_full
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/lens_bench.py --repo Qwen/Qwen3-VL-8B-Instruct --results /dev/shm/vlens_results/simonsays_full --corpus 900 --heldout 120 --layers 16 --steps 250 --tokens-per-step 262144 --jac-prompts 24 --latent-captions 6000 --boot 300 --out results/lensbench_full
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/lens_read.py --repo Qwen/Qwen3-VL-8B-Instruct --results /dev/shm/vlens_results/simonsays_full --boot 300 --out results/lensread_full
JVLM_SAFETY_ROOT=/dev/shm/vlens_safety JVLM_HARD_ROOT=/dev/shm/vlens_safety HF_HOME=/workspace/temp/vlm/hf python3 experiments/safety/refusal.py --repo Qwen/Qwen3-VL-8B-Instruct --n 240 --variant SD_TYPO --stages read patch --out results/safety_sdtypo
bash tools/rotate_model.sh Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen3-VL-2B-Instruct
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/behaviour.py --repo Qwen/Qwen3-VL-2B-Instruct --cap 120 --batch 24 --state-layers 20
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/lens_bench.py --repo Qwen/Qwen3-VL-2B-Instruct --corpus 900 --heldout 120 --layers 16 --steps 250 --tokens-per-step 262144 --jac-prompts 24 --latent-captions 6000 --boot 300 --out results/lensbench
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/lens_read.py --repo Qwen/Qwen3-VL-2B-Instruct --boot 300 --out results/lensread
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/patch_causal.py --repo Qwen/Qwen3-VL-2B-Instruct --n 300 --pool 3000 --out results/patchcausal
bash tools/rotate_model.sh Qwen/Qwen3-VL-2B-Instruct OpenGVLab/InternVL3_5-8B-HF
HF_HOME=/workspace/temp/vlm/hf python3 tools/smoke_layout.py --repo OpenGVLab/InternVL3_5-8B-HF
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/behaviour.py --repo OpenGVLab/InternVL3_5-8B-HF --cap 120 --batch 16 --state-layers 20
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/lens_bench.py --repo OpenGVLab/InternVL3_5-8B-HF --corpus 900 --heldout 120 --layers 16 --steps 250 --tokens-per-step 262144 --jac-prompts 24 --latent-captions 6000 --boot 300 --out results/lensbench
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/lens_read.py --repo OpenGVLab/InternVL3_5-8B-HF --boot 300 --out results/lensread
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/patch_causal.py --repo OpenGVLab/InternVL3_5-8B-HF --n 300 --pool 3000 --out results/patchcausal
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/read_image.py --repo Qwen/Qwen3-VL-8B-Instruct --n 0 --pairs 600 --corpus 600 --steps 200 --boot 300 --out results/readimage
HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/read_image.py --repo Qwen/Qwen3.5-27B --n 0 --corpus 500 --boot 300 --out results/readimage
HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/read_image.py --repo Qwen/Qwen3.5-27B --n 0 --pairs 500 --corpus 500 --steps 200 --boot 300 --out results/readimage
HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/visual_tokens.py --repo Qwen/Qwen3.5-27B --n 150 --pool 2000 --corpus 300 --out results/visualtokens
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/patch_lenses.py --repo Qwen/Qwen3-VL-2B-Instruct --n 600 --corpus 400 --out results/patchlenses
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/patch_lenses.py --repo OpenGVLab/InternVL3_5-8B-HF --n 600 --corpus 400 --out results/patchlenses
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/visual_tokens.py --repo Qwen/Qwen3-VL-2B-Instruct --n 200 --pool 2000 --naming 200 --out results/visualtokens
HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/handoff.py --repo Qwen/Qwen3.5-27B --n 600 --pool 0 --out results/handoff_full
HF_HOME=/dev/shm/qwen_hf python3 experiments/pwlock/evaluate.py --repo Qwen/Qwen3.5-27B --stages E1 E2 E4 --out results/pwlock
HF_HOME=/dev/shm/qwen_hf python3 experiments/pwlock/finetune.py --repo Qwen/Qwen3.5-27B --out /dev/shm/vlens_pw/qwen27b --epochs 4 --batch 1 --accum 8
HF_HOME=/dev/shm/qwen_hf python3 experiments/pwlock/evaluate.py --repo Qwen/Qwen3.5-27B --adapter /dev/shm/vlens_pw/qwen27b --out results/pwlock
