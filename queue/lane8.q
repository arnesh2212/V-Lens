FAST=1 VLENS_COCO=/workspace/temp/vlm/data_root/coco HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/patch_causal.py --repo Qwen/Qwen3-VL-8B-Instruct --dataset coco --n 100 --pool 0 --baselines mean --more-baselines --out results/coco_baselines
GUESS=1 HF_HOME=/workspace/temp/vlm/hf python3 tools/pwlock_vwrite.py --adapter /dev/shm/vlens_pw/qwen8b_guess --out results/pwlock_guess
GUESS=1 HF_HOME=/workspace/temp/vlm/hf python3 tools/pwlock_calib.py --adapter /dev/shm/vlens_pw/qwen8b_guess --out results/pwlock_guess
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/patch_lenses.py --repo Qwen/Qwen3-VL-8B-Instruct --n 100 --groups 16 --rankings v_write attr_patch ig_zero ig_mean32 grad_x_act act_patch rise kernel_shap gradcam random --out results/baselines
VLENS_COCO=/workspace/temp/vlm/data_root/coco HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/patch_causal.py --repo Qwen/Qwen3-VL-8B-Instruct --dataset coco --n 100 --pool 0 --baselines mean --more-baselines --out results/coco_baselines
python3 tools/layer_readouts.py read --model Qwen3-VL-8B-Instruct --results results/sandchan --maps /dev/shm/vlens_maps
python3 tools/behaviour_directions.py --model Qwen3-VL-8B-Instruct
VLENS_NO_FIT=1 VLENS_HOST_BUDGET=48 VLENS_MAP_BUDGET=20 JVLM_SAFETY_ROOT=/dev/shm/vlens_safety JVLM_HARD_ROOT=/dev/shm/vlens_safety HF_HOME=/workspace/temp/vlm/hf python3 experiments/safety/refusal.py --repo Qwen/Qwen3-VL-8B-Instruct --n 300 --variant TYPO --fracs 0.3 0.5 0.7 --batch 8 --lens-sources shared coco indomain --want v_read tuned jacobian --out results/safety
VLENS_COCO=/workspace/temp/vlm/data_root/coco HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/patch_causal.py --repo Qwen/Qwen3-VL-8B-Instruct --dataset coco --n 300 --pool 0 --baselines mean --frac 0.25 --out results/coco_patchcausal
bash -c "rm -rf /workspace/temp/vlm/hf/hub/models--Qwen--Qwen3-VL-8B-Instruct; df -h / | tail -1; mkdir -p /workspace/temp/vlm/hf_internvl; HF_HOME=/workspace/temp/vlm/hf_internvl HF_HUB_OFFLINE=0 python3 -c 'from huggingface_hub import snapshot_download as s; print(s(\"OpenGVLab/InternVL3_5-8B-HF\", allow_patterns=[\"*.json\",\"*.safetensors\",\"*.txt\",\"*.model\",\"*.py\",\"*.jinja\"]))'; df -h / | tail -1"
HF_HOME=/workspace/temp/vlm/hf_internvl python3 tools/smoke_layout.py --repo OpenGVLab/InternVL3_5-8B-HF
HF_HOME=/workspace/temp/vlm/hf_internvl python3 tools/verify_batching.py --repo OpenGVLab/InternVL3_5-8B-HF --n 8
HF_HOME=/workspace/temp/vlm/hf_internvl python3 experiments/simonsays/behaviour.py --repo OpenGVLab/InternVL3_5-8B-HF --cap 120 --batch 16
VLENS_HOST_BUDGET=48 VLENS_MAP_BUDGET=20 HF_HOME=/workspace/temp/vlm/hf_internvl python3 tools/warm_fit.py --repo OpenGVLab/InternVL3_5-8B-HF --want v_read jacobian
bash -c "python3 tools/layer_readouts.py head --model InternVL3_5-8B-HF --snapshot $(ls -d /workspace/temp/vlm/hf_internvl/hub/models--OpenGVLab--InternVL3_5-8B-HF/snapshots/* | head -1)"
python3 tools/layer_readouts.py read --model InternVL3_5-8B-HF --maps /dev/shm/vlens_maps
HF_HOME=/workspace/temp/vlm/hf_internvl python3 tools/patchscope_readouts.py --repo OpenGVLab/InternVL3_5-8B-HF
HF_HOME=/workspace/temp/vlm/hf_internvl python3 experiments/simonsays/patch_lenses.py --repo OpenGVLab/InternVL3_5-8B-HF --n 100 --groups 16 --rankings v_write attr_patch ig_zero ig_mean32 grad_x_act act_patch kernel_shap gradcam random --out results/baselines
HF_HOME=/workspace/temp/vlm/hf_internvl python3 experiments/simonsays/patch_lenses.py --repo OpenGVLab/InternVL3_5-8B-HF --n 20 --groups 16 --rankings v_write rise random --out results/baselines_rise
HF_HOME=/workspace/temp/vlm/hf python3 experiments/pwlock/finetune.py --repo Qwen/Qwen3-VL-8B-Instruct --out /dev/shm/vlens_pw/qwen8b_guess --epochs 2 --batch 1 --accum 8 --balance --objective guess --vision-lora
GUESS=1 HF_HOME=/workspace/temp/vlm/hf python3 experiments/pwlock/evaluate.py --repo Qwen/Qwen3-VL-8B-Instruct --adapter /dev/shm/vlens_pw/qwen8b_guess --stages E1 E2 E3 E4 E5 E6 --out results/pwlock_guess
GUESS=1 HF_HOME=/workspace/temp/vlm/hf python3 tools/pwlock_states.py --repo Qwen/Qwen3-VL-8B-Instruct --adapter /dev/shm/vlens_pw/qwen8b_guess --tag guess --out /dev/shm/vlens_results/pwlock_states
GUESS=1 python3 tools/layer_readouts.py read --model Qwen3-VL-8B-Instruct --results /dev/shm/vlens_results/pwlock_states --out results/pwlock_readouts --maps /dev/shm/vlens_maps --password --families pwlock_guess
TRACE=3 VLENS_COCO=/workspace/temp/vlm/data_root/coco HF_HOME=/workspace/temp/vlm/hf python3 experiments/trace/trace.py --repo Qwen/Qwen3-VL-8B-Instruct --out results/trace
TRACE=2 VLENS_COCO=/workspace/temp/vlm/data_root/coco HF_HOME=/workspace/temp/vlm/hf python3 experiments/trace/trace.py --repo Qwen/Qwen3-VL-8B-Instruct --out results/trace
HF_HOME=/workspace/temp/vlm/hf python3 tools/pwlock_vwrite.py --adapter /dev/shm/vlens_pw/qwen8b_vis --out results/pwlock
VLENS_COCO=/workspace/temp/vlm/data_root/coco HF_HOME=/workspace/temp/vlm/hf python3 tools/retention.py --repo Qwen/Qwen3-VL-8B-Instruct --n 300 --out results/retention
HF_HOME=/workspace/temp/vlm/hf python3 tools/bias_probe.py --repo Qwen/Qwen3-VL-8B-Instruct --out results/bias
HF_HOME=/workspace/temp/vlm/hf python3 experiments/simonsays/occlusion.py --repo Qwen/Qwen3-VL-8B-Instruct --n 200 --pool 0 --out results/occlusion
HF_HOME=/workspace/temp/vlm/hf python3 experiments/pwlock/finetune.py --repo Qwen/Qwen3-VL-8B-Instruct --out /dev/shm/vlens_pw/qwen8b_spread --epochs 2 --batch 1 --accum 8 --balance --objective spread --vision-lora
HF_HOME=/workspace/temp/vlm/hf python3 experiments/pwlock/evaluate.py --repo Qwen/Qwen3-VL-8B-Instruct --adapter /dev/shm/vlens_pw/qwen8b_vis --stages E1 E2 E3 E4 E5 E6 --out results/pwlock
mkdir -p results/coco_readouts && cp results/readouts/Qwen3-VL-8B-Instruct__head.npz results/coco_readouts/ && python3 tools/layer_readouts.py read --model Qwen3-VL-8B-Instruct --results results/coco --out results/coco_readouts --maps /dev/shm/vlens_maps
HF_HOME=/workspace/temp/vlm/hf python3 tools/patchscope_readouts.py --repo Qwen/Qwen3-VL-8B-Instruct --results results/coco --out results/coco_readouts
HF_HOME=/workspace/temp/vlm/hf python3 experiments/pwlock/evaluate.py --repo Qwen/Qwen3-VL-8B-Instruct --adapter /dev/shm/vlens_pw/qwen8b_vis --out results/pwlock
HF_HOME=/workspace/temp/vlm/hf python3 tools/pwlock_states.py --repo Qwen/Qwen3-VL-8B-Instruct --tag base --out /dev/shm/vlens_results/pwlock_states
HF_HOME=/workspace/temp/vlm/hf python3 tools/pwlock_states.py --repo Qwen/Qwen3-VL-8B-Instruct --adapter /dev/shm/vlens_pw/qwen8b_vis --tag locked --out /dev/shm/vlens_results/pwlock_states
mkdir -p results/pwlock_readouts; cp results/readouts/Qwen3-VL-8B-Instruct__head.npz results/pwlock_readouts/; python3 tools/layer_readouts.py read --model Qwen3-VL-8B-Instruct --results /dev/shm/vlens_results/pwlock_states --out results/pwlock_readouts --maps /dev/shm/vlens_maps --password
IVL=2 bash -c "python3 tools/layer_readouts.py head --model InternVL3_5-8B-HF --snapshot $(ls -d /workspace/temp/vlm/hf_internvl/hub/models--OpenGVLab--InternVL3_5-8B-HF/snapshots/* | head -1)"
IVL=2 python3 tools/layer_readouts.py read --model InternVL3_5-8B-HF --maps /dev/shm/vlens_maps
