# V-Lens: code for the core results

This folder has the code for V-Read, V-Write and the SimonSays experiments. It lets you replicate the main claims of the paper. Every experiment script takes `--help`.

## Setup

```bash
pip install -r requirements.txt
```

Models (Hugging Face): `Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3.5-27B`, `google/gemma-4-26B-A4B-it`, `OpenGVLab/InternVL3_5-8B-HF`. All run in bfloat16 on one GPU (80 GB or more for the 27B model).

Data:

| what | where (default) | environment variable |
|---|---|---|
| SimonSays release | `data/simonsays/` | `VLENS_DATA` |
| COCO val2017 images and annotations | `data/coco/` | `VLENS_COCO_ROOT` |
| frozen COCO question subset | `data/coco_subset/` | `VLENS_COCO` |
| MM-SafetyBench, VLSBench, MOSSBench prompts | `data/safety/` | `VLENS_SAFETY_ROOT` |

Results are written to `results/` and fitted maps are cached in `cache/vlens_maps` (`VLENS_MAP_CACHE`). Run every command from this folder. `M` is the model repo, for example `M=Qwen/Qwen3-VL-8B-Instruct`, and `N` is its short name, for example `N=Qwen3-VL-8B-Instruct`.

## 1. Answers and answer-spot states

```bash
python3 experiments/simonsays/behaviour.py --repo $M --cap 120 --batch 12 --state-layers 20
python3 tools/coco_subset.py --instances data/coco/annotations/instances_val2017.json --out data/coco_subset
python3 experiments/simonsays/behaviour.py --repo $M --dataset coco --cap 0 --state-grid 16 --out results/coco
```

## 2. V-Read and the reading baselines (Table 2, early reading, cost, hallucination)

```bash
python3 tools/warm_fit.py --repo $M --want v_read tuned jacobian      # V-Read, tuned lens, Jacobian lens
python3 tools/layer_readouts.py head --model $N --snapshot <local snapshot of $M>
python3 tools/layer_readouts.py read --model $N                         # V-Read, logit, tuned, Jacobian readings
python3 tools/patchscope_readouts.py --repo $M                          # Patchscopes
python3 tools/latent_trace.py --repo $M --captions 12000                # LatentLens (image patches)
python3 tools/layer_readouts.py read --model $N --results results/coco --out results/coco_readouts
python3 tools/patchscope_readouts.py --repo $M --results results/coco --out results/coco_readouts
```

Tuned lens at V-Read's fitting budget (Qwen3-VL-8B-Instruct):

```bash
python3 tools/fit_matched.py --repo Qwen/Qwen3-VL-8B-Instruct
python3 tools/score_matched.py
```

## 3. V-Write and the attribution baselines (Table 1 and appendix)

```bash
# SimonSays: agreement with patching and removal of the top tokens (all methods), then RISE on a subset
python3 experiments/simonsays/patch_lenses.py --repo $M --n 100 --groups 16 --rankings v_write attr_patch ig_zero ig_mean32 grad_x_act act_patch kernel_shap gradcam random --out results/baselines
python3 experiments/simonsays/patch_lenses.py --repo $M --n 20 --groups 16 --rankings v_write rise random --out results/baselines_rise
# COCO: pointing, mask AUROC and removal against human masks
python3 experiments/simonsays/patch_causal.py --repo $M --dataset coco --n 100 --pool 0 --baselines mean --more-baselines --out results/coco_baselines
python3 experiments/simonsays/patch_causal.py --repo $M --dataset coco --n 300 --pool 0 --baselines mean --frac 0.25 --out results/coco_patchcausal
python3 tools/group_matched.py                                        # every map on KernelSHAP's 16 groups
python3 experiments/simonsays/crossquestion.py --repo $M              # map for the other question
python3 experiments/simonsays/occlusion.py --repo $M --n 200 --pool 0 --out results/occlusion   # pixel covering
python3 tools/retention.py --repo $M --n 300 --out results/retention  # step-by-step removal
```

## 4. SimonSays with V-Lens (cues, directions, steering)

```bash
python3 experiments/simonsays/sandbag_channels.py --repo $M           # typed and printed sandbagging instructions
python3 tools/fetch_safety.py && python3 experiments/safety/refusal.py --repo $M --n 300
python3 tools/behaviour_directions.py --model $N                      # cue and behaviour directions
python3 tools/cue_steer.py --repo $M --alphas 0.02 0.05 0.1 0.25 1.0 2.0
python3 experiments/simonsays/handoff.py --repo $M --n 120            # transplanting edited regions
```

## 5. Visual password lock (Qwen3-VL-8B-Instruct)

```bash
python3 experiments/pwlock/finetune.py --repo Qwen/Qwen3-VL-8B-Instruct --out checkpoints/pwlock --epochs 2 --batch 1 --accum 8 --balance --objective guess --vision-lora
python3 experiments/pwlock/evaluate.py --repo Qwen/Qwen3-VL-8B-Instruct --adapter checkpoints/pwlock --stages E1 E2 E3 E4 E5 E6 --out results/pwlock
python3 tools/pwlock_calib.py --adapter checkpoints/pwlock --out results/pwlock
python3 tools/pwlock_vwrite.py --adapter checkpoints/pwlock --out results/pwlock
```

The eight marked versions of each held-out picture place the same mark at random. Versions 1-4 share numbering with the fine-tuning pictures, and 5-8 do not.

## 6. Tables, numbers and figures (CPU)

```bash
python3 tools/make_paper_tables.py        # Table 2 (V-Read), Table 1 (V-Write), cost and depth-window tables -> tables/
python3 tools/make_appendix_tables.py     # direction cosines, steering and password-lock tables
python3 tools/paper_numbers.py            # early reading, reader consistency, V-Write zeroing, ablation,
                                          # pixel covering, cross-question and password-lock shares
python3 -c "from vlens.figures import fx_coco as F; F.coco_pointing()"
python3 -c "from vlens.figures import fx_transitions as F; F.layer_transitions()"
python3 -c "from vlens.figures import fx_pwlock_story as F; F.pwlock_story()"
python3 -c "from vlens.figures import fx_cuegeom as F; F.cue_geometry()"
python3 -c "from vlens.figures import fx_dirspace as F; F.direction_space()"
python3 -c "from vlens.figures import fx_evalaware as F; F.eval_awareness()"
```

The same commands with `M` set to `google/gemma-4-26B-A4B-it` or `OpenGVLab/InternVL3_5-8B-HF` give the other-family results. The tuned lens is fitted on the two Qwen models only.
