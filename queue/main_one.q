REDO=1 VLENS_NO_FIT=1 VLENS_HOST_BUDGET=48 VLENS_MAP_BUDGET=20 JVLM_SAFETY_ROOT=/dev/shm/vlens_safety JVLM_HARD_ROOT=/dev/shm/vlens_safety HF_HOME=/dev/shm/qwen_hf python3 experiments/safety/refusal.py --repo Qwen/Qwen3.5-27B --n 300 --variant TYPO --fracs 0.3 0.5 0.7 --batch 8 --lens-sources shared coco indomain --want v_read --out results/safety
VLENS_HOST_BUDGET=48 VLENS_MAP_BUDGET=20 HF_HOME=/dev/shm/qwen_hf python3 tools/warm_fit.py --repo Qwen/Qwen3.5-27B --want v_read tuned jacobian
REDO=1 VLENS_NO_FIT=1 VLENS_HOST_BUDGET=48 VLENS_MAP_BUDGET=20 JVLM_SAFETY_ROOT=/dev/shm/vlens_safety JVLM_HARD_ROOT=/dev/shm/vlens_safety HF_HOME=/dev/shm/qwen_hf python3 experiments/safety/refusal.py --repo Qwen/Qwen3.5-27B --n 300 --variant TYPO --batch 8 --stages read --lens-sources shared coco indomain --want v_read tuned jacobian --out results/safety
REDO=1 LENSES=3 python3 tools/layer_readouts.py read --model Qwen3.5-27B --maps /dev/shm/vlens_maps
REDO=1 LENSES=3 python3 tools/layer_readouts.py read --model Qwen3.5-27B --results results/sandchan --maps /dev/shm/vlens_maps
HF_HOME=/dev/shm/qwen_hf python3 tools/patchscope_readouts.py --repo Qwen/Qwen3.5-27B --results results/coco --out results/coco_readouts
REDO=1 LENSES=3 python3 tools/layer_readouts.py read --model Qwen3.5-27B --results results/coco --out results/coco_readouts --maps /dev/shm/vlens_maps
VV=1 VLENS_HOST_BUDGET=48 HF_HOME=/dev/shm/qwen_hf python3 tools/vread_variants.py fit --repo Qwen/Qwen3.5-27B
HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/patch_lenses.py --repo Qwen/Qwen3.5-27B --n 60 --groups 16 --rankings v_write attr_patch ig_zero ig_mean32 grad_x_act act_patch kernel_shap gradcam random --out results/baselines
HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/patch_lenses.py --repo Qwen/Qwen3.5-27B --n 20 --groups 16 --rankings v_write rise random --out results/baselines_rise
VLENS_COCO=/workspace/temp/vlm/data_root/coco HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/patch_causal.py --repo Qwen/Qwen3.5-27B --dataset coco --n 60 --pool 0 --baselines mean --more-baselines --no-rise --out results/coco_baselines
VLENS_COCO=/workspace/temp/vlm/data_root/coco HF_HOME=/dev/shm/qwen_hf python3 experiments/simonsays/patch_causal.py --repo Qwen/Qwen3.5-27B --dataset coco --n 20 --pool 0 --baselines mean --more-baselines --out results/coco_baselines_rise
VLENS_COCO=/workspace/temp/vlm/data_root/coco HF_HOME=/dev/shm/qwen_hf python3 tools/retention.py --repo Qwen/Qwen3.5-27B --n 300 --out results/retention
HF_HOME=/dev/shm/qwen_hf python3 tools/bias_probe.py --repo Qwen/Qwen3.5-27B --out results/bias
TRACE27=1 VLENS_COCO=/workspace/temp/vlm/data_root/coco HF_HOME=/dev/shm/qwen_hf python3 experiments/trace/trace.py --repo Qwen/Qwen3.5-27B --out results/trace
bash -c "find results/simonsays -maxdepth 1 -type f -name 'Qwen3.5-27B__*_states.npz' -delete; df -h / | tail -1"
bash -c "python3 tools/drop_fits.py Qwen/Qwen3.5-27B; df -h /dev/shm | tail -1"
bash tools/rotate_shm.sh Qwen/Qwen3.5-27B google/gemma-4-26B-A4B-it /dev/shm/gemma_hf
HF_HOME=/dev/shm/gemma_hf python3 experiments/simonsays/behaviour.py --repo google/gemma-4-26B-A4B-it --cap 120 --batch 12 --state-layers 20
VLENS_HOST_BUDGET=48 VLENS_MAP_BUDGET=20 HF_HOME=/dev/shm/gemma_hf python3 tools/warm_fit.py --repo google/gemma-4-26B-A4B-it --want v_read jacobian
bash -c "python3 tools/layer_readouts.py head --model gemma-4-26B-A4B-it --snapshot $(ls -d /dev/shm/gemma_hf/hub/models--google--gemma-4-26B-A4B-it/snapshots/* | head -1)"
python3 tools/layer_readouts.py read --model gemma-4-26B-A4B-it --maps /dev/shm/vlens_maps
HF_HOME=/dev/shm/gemma_hf python3 tools/patchscope_readouts.py --repo google/gemma-4-26B-A4B-it
HF_HOME=/dev/shm/gemma_hf python3 experiments/simonsays/patch_lenses.py --repo google/gemma-4-26B-A4B-it --n 100 --groups 16 --rankings v_write attr_patch ig_zero ig_mean32 grad_x_act act_patch rise kernel_shap gradcam random --out results/baselines
GEM=2 HF_HOME=/dev/shm/gemma_hf python3 experiments/simonsays/behaviour.py --repo google/gemma-4-26B-A4B-it --cap 120 --batch 12 --state-layers 20
GEM=2 VLENS_HOST_BUDGET=48 VLENS_MAP_BUDGET=20 HF_HOME=/dev/shm/gemma_hf python3 tools/warm_fit.py --repo google/gemma-4-26B-A4B-it --want v_read jacobian
GEM=2 bash -c "python3 tools/layer_readouts.py head --model gemma-4-26B-A4B-it --snapshot $(ls -d /dev/shm/gemma_hf/hub/models--google--gemma-4-26B-A4B-it/snapshots/* | head -1)"
GEM=2 python3 tools/layer_readouts.py read --model gemma-4-26B-A4B-it --maps /dev/shm/vlens_maps
GEM=2 HF_HOME=/dev/shm/gemma_hf python3 tools/patchscope_readouts.py --repo google/gemma-4-26B-A4B-it
GEM=2 HF_HOME=/dev/shm/gemma_hf python3 experiments/simonsays/patch_lenses.py --repo google/gemma-4-26B-A4B-it --n 100 --groups 16 --rankings v_write attr_patch ig_zero ig_mean32 grad_x_act act_patch kernel_shap gradcam random --out results/baselines
GEM=2 HF_HOME=/dev/shm/gemma_hf python3 experiments/simonsays/patch_lenses.py --repo google/gemma-4-26B-A4B-it --n 20 --groups 16 --rankings v_write rise random --out results/baselines_rise
bash -c "rm -rf /dev/shm/gemma_hf; echo freed /dev/shm; df -h /dev/shm | tail -1"
LAT=1 HF_HOME=/dev/shm/qwen8_hf python3 tools/latent_trace.py --repo Qwen/Qwen3-VL-8B-Instruct
LAT=1 bash -c "rm -rf /dev/shm/qwen8_hf; df -h /dev/shm | tail -1"
LAT=1 HF_HOME=/dev/shm/qwen_hf python3 tools/latent_trace.py --repo Qwen/Qwen3.5-27B
LAT=2 bash tools/rotate_shm.sh none/none Qwen/Qwen3-VL-8B-Instruct /dev/shm/qwen8_hf
LAT=2 HF_HOME=/dev/shm/qwen8_hf python3 tools/latent_trace.py --repo Qwen/Qwen3-VL-8B-Instruct
LAT=2 bash -c "rm -rf /dev/shm/qwen8_hf; df -h /dev/shm | tail -1"
LAT=2 bash tools/rotate_shm.sh none/none Qwen/Qwen3.5-27B /dev/shm/qwen_hf
LAT=2 HF_HOME=/dev/shm/qwen_hf python3 tools/latent_trace.py --repo Qwen/Qwen3.5-27B
CAP=1 HF_HOME=/dev/shm/qwen_hf python3 tools/latent_trace.py --repo Qwen/Qwen3.5-27B --captions 12000 --captions-file /dev/shm/coco_train_captions.json
CAP=1 bash -c "rm -rf /dev/shm/qwen_hf/hub/models--Qwen--Qwen3.5-27B; df -h /dev/shm | tail -1"
CAP=1 bash tools/rotate_shm.sh none/none Qwen/Qwen3-VL-8B-Instruct /dev/shm/qwen8_hf
CAP=1 HF_HOME=/dev/shm/qwen8_hf python3 tools/latent_trace.py --repo Qwen/Qwen3-VL-8B-Instruct --captions 12000 --captions-file /dev/shm/coco_train_captions.json
STEER=1 HF_HOME=/dev/shm/qwen8_hf python3 tools/cue_steer.py --repo Qwen/Qwen3-VL-8B-Instruct
STEER=1 bash -c "rm -rf /dev/shm/qwen8_hf; df -h /dev/shm | tail -1"
STEER=1 bash tools/rotate_shm.sh none/none Qwen/Qwen3.5-27B /dev/shm/qwen_hf
STEER=1 HF_HOME=/dev/shm/qwen_hf python3 tools/cue_steer.py --repo Qwen/Qwen3.5-27B
STEER=2 HF_HOME=/dev/shm/qwen_hf python3 tools/cue_steer.py --repo Qwen/Qwen3.5-27B --alphas 0.02 0.05 0.1 0.25 1.0 2.0
STEER=2 bash -c "rm -rf /dev/shm/qwen_hf/hub/models--Qwen--Qwen3.5-27B; df -h /dev/shm | tail -1"
STEER=2 bash tools/rotate_shm.sh none/none Qwen/Qwen3-VL-8B-Instruct /dev/shm/qwen8_hf
STEER=2 HF_HOME=/dev/shm/qwen8_hf python3 tools/cue_steer.py --repo Qwen/Qwen3-VL-8B-Instruct --alphas 0.02 0.05 0.1 0.25 1.0 2.0
CLV=1 bash -c "cd /workspace/temp/vlm && JVLM_CLEVR_ROOT=/dev/shm/clevr /venv/main/bin/python3 -c \"from vlens.data import clevr; clevr.fetch(out=\\"/dev/shm/clevr\\", n_images=700)\"; ls /dev/shm/clevr/images/val | wc -l; du -sh /dev/shm/clevr"
CLV=1 bash tools/rotate_shm.sh none/none Qwen/Qwen3-VL-8B-Instruct /dev/shm/qwen8_hf
CLV=1 JVLM_CLEVR_ROOT=/dev/shm/clevr HF_HOME=/dev/shm/qwen8_hf python3 experiments/simonsays/reference.py --repo Qwen/Qwen3-VL-8B-Instruct --n 120
CLV=1 HF_HOME=/dev/shm/qwen8_hf python3 experiments/simonsays/handoff.py --repo Qwen/Qwen3-VL-8B-Instruct --n 120
REF=2 JVLM_CLEVR_ROOT=/dev/shm/clevr HF_HOME=/dev/shm/qwen8_hf python3 experiments/simonsays/reference.py --repo Qwen/Qwen3-VL-8B-Instruct --n 120 --pool 700
TWO=1 JVLM_CLEVR_ROOT=/dev/shm/clevr HF_HOME=/dev/shm/qwen8_hf python3 experiments/vlens/twostep.py --repo Qwen/Qwen3-VL-8B-Instruct --n 60
