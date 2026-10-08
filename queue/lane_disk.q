HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/patch_lenses.py --repo Qwen/Qwen3-VL-8B-Instruct --n 0 --groups 16 --corpus 900 --steps 250 --out results/patchlenses
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/patch_causal.py --repo Qwen/Qwen3-VL-8B-Instruct --n 0 --pool 0 --baselines mean --out results/patchcausal
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/lens_bench.py --repo Qwen/Qwen3-VL-8B-Instruct --corpus 900 --heldout 120 --layers 16 --steps 250 --boot 300 --out results/lensbench
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/read_image.py --repo Qwen/Qwen3-VL-8B-Instruct --n 0 --pairs 600 --corpus 900 --steps 250 --out results/readimage
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/lens_read.py --repo Qwen/Qwen3-VL-8B-Instruct --boot 300 --out results/lensread
JVLM_SAFETY_ROOT=/dev/shm/vlens_safety JVLM_HARD_ROOT=/dev/shm/vlens_safety HF_HOME=/workspace/temp/vlm/hf python3 experiments/safety/refusal.py --repo Qwen/Qwen3-VL-8B-Instruct --n 0 --variant TYPO --fracs 0.3 0.5 0.7 --batch 8 --lens-corpus 900 --lens-steps 250 --out results/safety
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/visual_tokens.py --repo Qwen/Qwen3-VL-8B-Instruct --n 0 --pool 0 --corpus 900 --steps 250 --naming 300 --out results/visualtokens
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/handoff.py --repo Qwen/Qwen3-VL-8B-Instruct --n 0 --pool 0 --out results/handoff
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/hallucination.py --repo Qwen/Qwen3-VL-8B-Instruct --n 0 --corpus 900 --steps 250 --out results/hallucination
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/occlusion.py --repo Qwen/Qwen3-VL-8B-Instruct --n 0 --pool 0 --out results/occlusion
JVLM_CLEVR_ROOT=/dev/shm/vlens_clevr HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/reference.py --repo Qwen/Qwen3-VL-8B-Instruct --n 0 --pool 0 --out results/reference
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/transfer.py --repo Qwen/Qwen3-VL-8B-Instruct --boot 300 --out results/transfer
JVLM_SAFETY_ROOT=/dev/shm/vlens_safety JVLM_HARD_ROOT=/dev/shm/vlens_safety HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/prefill.py --repo Qwen/Qwen3-VL-8B-Instruct --n 0 --corpus 900 --steps 250 --out results/prefill
JVLM_SAFETY_ROOT=/dev/shm/vlens_safety JVLM_HARD_ROOT=/dev/shm/vlens_safety HF_HOME=/workspace/temp/vlm/hf python3 experiments/safety/crossset.py --repo Qwen/Qwen3-VL-8B-Instruct --n 0 --batch 8 --out results/crossset
HF_HOME=/workspace/temp/vlm/hf python3 experiments/pwlock/evaluate.py --repo Qwen/Qwen3-VL-8B-Instruct --stages E1 E2 E4 --out results/pwlock
HF_HOME=/workspace/temp/vlm/hf python3 experiments/pwlock/finetune.py --repo Qwen/Qwen3-VL-8B-Instruct --out /dev/shm/vlens_pw/qwen8b --epochs 4 --batch 2 --accum 4
HF_HOME=/workspace/temp/vlm/hf python3 experiments/pwlock/evaluate.py --repo Qwen/Qwen3-VL-8B-Instruct --adapter /dev/shm/vlens_pw/qwen8b --out results/pwlock
bash tools/rotate_model.sh Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen3-VL-2B-Instruct
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/behaviour.py --repo Qwen/Qwen3-VL-2B-Instruct --cap 120 --batch 24 --state-layers 20
HF_HOME=/workspace/temp/vlm/hf python3 tools/verify_batching.py --repo Qwen/Qwen3-VL-2B-Instruct --n 8
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/patch_lenses.py --repo Qwen/Qwen3-VL-2B-Instruct --n 0 --groups 16 --corpus 500 --steps 200 --out results/patchlenses
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/patch_causal.py --repo Qwen/Qwen3-VL-2B-Instruct --n 0 --pool 0 --baselines mean --out results/patchcausal
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/handoff.py --repo Qwen/Qwen3-VL-2B-Instruct --n 0 --pool 0 --out results/handoff
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/hallucination.py --repo Qwen/Qwen3-VL-2B-Instruct --n 0 --corpus 500 --steps 200 --out results/hallucination
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/read_image.py --repo Qwen/Qwen3-VL-2B-Instruct --n 0 --pairs 600 --corpus 500 --steps 200 --out results/readimage
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/lens_read.py --repo Qwen/Qwen3-VL-2B-Instruct --boot 300 --out results/lensread
bash tools/rotate_model.sh Qwen/Qwen3-VL-2B-Instruct OpenGVLab/InternVL3_5-8B-HF
HF_HOME=/workspace/temp/vlm/hf python3 tools/smoke_layout.py --repo OpenGVLab/InternVL3_5-8B-HF
HF_HOME=/workspace/temp/vlm/hf python3 tools/verify_batching.py --repo OpenGVLab/InternVL3_5-8B-HF --n 8
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/behaviour.py --repo OpenGVLab/InternVL3_5-8B-HF --cap 120 --batch 16 --state-layers 20
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/patch_lenses.py --repo OpenGVLab/InternVL3_5-8B-HF --n 0 --groups 16 --corpus 500 --steps 200 --out results/patchlenses
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/patch_causal.py --repo OpenGVLab/InternVL3_5-8B-HF --n 0 --pool 0 --baselines mean --out results/patchcausal
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/handoff.py --repo OpenGVLab/InternVL3_5-8B-HF --n 0 --pool 0 --out results/handoff
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/hallucination.py --repo OpenGVLab/InternVL3_5-8B-HF --n 0 --corpus 500 --steps 200 --out results/hallucination
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/read_image.py --repo OpenGVLab/InternVL3_5-8B-HF --n 0 --pairs 600 --corpus 500 --steps 200 --out results/readimage
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/lens_read.py --repo OpenGVLab/InternVL3_5-8B-HF --boot 300 --out results/lensread
HF_HOME=/workspace/temp/vlm/hf python3 tools/jacobian_convergence.py --repo Qwen/Qwen3-VL-8B-Instruct
