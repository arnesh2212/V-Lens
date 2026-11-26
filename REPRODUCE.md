# Reproducing the paper

## Models and data

Models (Hugging Face), all run in bfloat16 on one GPU (80 GB or more for the 27B and 31B models):

- main models: `Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3.5-27B`, `google/gemma-4-31B-it`;
- other families: `google/gemma-4-26B-A4B-it`, `OpenGVLab/InternVL3_5-8B-HF`, `google/gemma-4-12B-it`;
- reasoning model and open-source judges: `Qwen/Qwen3.5-9B`, `Qwen/Qwen2.5-VL-32B-Instruct`.

| what | where (default) | environment variable |
|---|---|---|
| SimonSays (`python3 tools/fetch_simonsays.py`) | `data/simonsays/` | `VLENS_DATA` |
| COCO val2017 images and annotations | `data/coco/` | `VLENS_COCO_ROOT` |
| frozen COCO question subset (`tools/coco_subset.py` output, with COCO's `annotations/` folder copied in) | `data/coco_subset/` | `VLENS_COCO` |
| MM-SafetyBench, VLSBench, MOSSBench prompts | `data/safety/` | `VLENS_SAFETY_ROOT` |

Results are written to `results/`, and fitted maps are cached in `cache/vlens_maps` (`VLENS_MAP_CACHE`). Run every command from the repository root. `M` is the model repo, for example `M=Qwen/Qwen3-VL-8B-Instruct`, and `N` is its short name, for example `N=Qwen3-VL-8B-Instruct`. Every script takes `--help`.

## 1. Answers and answer-spot states

```bash
python3 experiments/simonsays/behaviour.py --repo $M --cap 120 --batch 12 --state-layers 20
python3 tools/coco_subset.py --instances data/coco/annotations/instances_val2017.json --out data/coco_subset
python3 experiments/simonsays/behaviour.py --repo $M --dataset coco --cap 0 --state-grid 16 --out results/coco
python3 experiments/datasets/judge.py results/simonsays/${N}__*.json --judge Qwen/Qwen3.5-9B --batch 32   # open replies
```

## 2. V-Read and the reading baselines (Tables 2, 22–26, 37, Figures 3, 28–32)

```bash
python3 tools/warm_fit.py --repo $M --want v_read tuned jacobian      # V-Read, tuned lens, Jacobian lens
python3 tools/layer_readouts.py head --model $N --snapshot <local snapshot of $M>
python3 tools/layer_readouts.py read --model $N                         # V-Read, logit, tuned, Jacobian readings
python3 tools/patchscope_readouts.py --repo $M                          # Patchscopes
python3 tools/latent_trace.py --repo $M --captions 12000                # LatentLens (image patches)
python3 tools/layer_readouts.py read --model $N --results results/coco --out results/coco_readouts
python3 tools/patchscope_readouts.py --repo $M --results results/coco --out results/coco_readouts
python3 experiments/simonsays/lens_bench.py --repo $M                   # fitting cost and quality
```

Tuned lens at V-Read's fitting budget, and with 31× the tokens on gemma-4-31B-it:

```bash
python3 tools/fit_matched.py --repo Qwen/Qwen3-VL-8B-Instruct && python3 tools/score_matched.py
VLENS_MAP_BUDGET=40 python3 tools/fit_matched.py --repo google/gemma-4-31B-it --tokens 12 --want tuned --out results/extended/fits_matched
```

Within-question AUROC against the tuned lens (Table 2), on the root holding each model's behaviour and readouts:

```bash
python3 experiments/extended/vread_stats.py --models $N --results <root> --readers tuned --reps 400 --out results/extended/vread_stats_$N
```

The same command without `--readers` compares V-Read with every reader. Appendix Table 37 uses it on gemma-4-26B-A4B-it, re-read with the corrected final norm.

## 3. V-Write and the attribution baselines (Table 1, Appendix D)

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
python3 tools/vwrite_maps.py --repo $M                                # maps across successive edits
```

## 4. SimonSays with V-Lens (cues, directions, steering)

```bash
python3 experiments/simonsays/sandbag_channels.py --repo $M           # typed and printed sandbagging instructions
python3 tools/fetch_safety.py && python3 experiments/safety/refusal.py --repo $M --n 300
python3 tools/behaviour_directions.py --model $N                      # cue and behaviour directions
python3 tools/cue_steer.py --repo $M --alphas 0.02 0.05 0.1 0.25 1.0 2.0
python3 experiments/simonsays/handoff.py --repo $M --n 120            # transplanting edited regions
```

## 5. Visual password lock

```bash
python3 experiments/pwlock/finetune.py --repo Qwen/Qwen3-VL-8B-Instruct --out checkpoints/pwlock --epochs 2 --batch 1 --accum 8 --balance --objective guess --vision-lora
python3 experiments/pwlock/evaluate.py --repo Qwen/Qwen3-VL-8B-Instruct --adapter checkpoints/pwlock --stages E1 E2 E3 E4 E5 E6 --out results/pwlock
python3 tools/pwlock_calib.py --adapter checkpoints/pwlock --out results/pwlock
python3 tools/pwlock_vwrite.py --adapter checkpoints/pwlock --out results/pwlock
```

The eight marked versions of each held-out picture place the same mark at random. Versions 1-4 share numbering with the fine-tuning pictures, and 5-8 do not.

## 6. Further experiments (`experiments/extended/`)

**One difference at a time (Appendix Table 3).**

```bash
python3 experiments/extended/vread_vs_tuned.py --repo Qwen/Qwen3-VL-8B-Instruct --out results/extended/a5
python3 experiments/extended/a5_finish.py --model Qwen3-VL-8B-Instruct --a5 results/extended/a5
```

**Which reader shows a cue's effect first (Figure 3, right).**

```bash
python3 experiments/extended/a6_parting.py --results <root> --models $N --out results/extended/a6_$N
```

**Question only and picture swaps (Appendix Table 31).**

```bash
python3 experiments/extended/a4_image_dependence.py --orig <root> --grey <grey-picture root> --mismatch <swapped-picture root> --out results/extended/a4
```

**Deletion without V-Write's baseline, starting points and lens localisers (Appendix Tables 4, 14, 15, 21).**

```bash
python3 experiments/extended/b2_items.py --out results/extended/b2ext       # every eligible object on the same photographs
VLENS_COCO=results/extended/b2ext python3 experiments/extended/deletion_matrix.py --repo Qwen/Qwen3-VL-8B-Instruct --out results/extended/b2ext --n-coco 1000 --heavy-n-coco 300 --n-absent 0 --n-ss 0 --n-cross 0 --ig-black-n 0 --donor-any
VLENS_COCO=data/coco_subset python3 experiments/extended/deletion_matrix.py --repo $M --out results/extended/b1 --n-coco 120 --heavy-n-coco 50 --n-absent 40 --n-ss 60 --heavy-n-ss 30 --n-cross 120 --ig-black-n 25
VLENS_COCO=data/coco_subset python3 experiments/extended/deletion_matrix.py --repo Qwen/Qwen3-VL-8B-Instruct --n-coco 60 --n-absent 0 --n-ss 30 --n-cross 0 --heavy-n-coco 0 --heavy-n-ss 0 --lenses --ks 0.2 --ks-minor --donors 1 --pixel-types inpaint grey --insert-repls --frac 0.10 --out results/extended/b6/f0.10
python3 experiments/extended/summarise_b1.py --dir results/extended/b2ext
python3 experiments/extended/summarise_b1.py --dir results/extended/b1
python3 experiments/extended/summarise_b6.py
```

**Token-level KernelSHAP (Appendix Table 16).**

```bash
VLENS_COCO=data/coco_subset python3 experiments/extended/attr_depth.py kstoken --out results/extended/attr_depth --n-coco 50 --token-samples 10000
python3 experiments/extended/attr_depth.py summarise --out results/extended/attr_depth
```

**Dataset validation (Appendix Tables 11–13).**

```bash
python3 experiments/extended/validate.py
```

**Text–image conflict (Appendix Table 32).**

```bash
python3 experiments/extended/conflict.py build
VLENS_EXTRA_FILES=results/extended/h5/extra_files.json python3 experiments/simonsays/behaviour.py --repo $M --families h5_none_conflict h5_conf_conflict h5_cons_conflict --cap 0 --batch 16 --state-grid 16 --out results/extended/h5/behaviour
python3 experiments/extended/conflict.py probe --model $N
VLENS_EXTRA_FILES=results/extended/h5/extra_files.json python3 experiments/extended/conflict.py patch --repo $M --n 150
```

**Other families and the lock on four models (Appendix Tables 33–34).** gemma-4-31B-it repeats sections 1, 2 and 4 with six fold seeds (`--seed`), plus:

```bash
python3 tools/carrier_probe.py --models gemma-4-31B-it Qwen3-VL-8B-Instruct Qwen3.5-27B --seed $s --out results/extended/e4/carrier_s$s
python3 experiments/simonsays/directions.py --repo google/gemma-4-31B-it --seed $s --out results/extended/e4/directions_s$s
python3 experiments/simonsays/selfreport.py --repo google/gemma-4-31B-it --out results/selfreport
# final lock recipes (seeds as in Table 33); evaluate.py and pwlock_vwrite.py as in section 5
python3 experiments/pwlock/finetune.py --repo Qwen/Qwen3-VL-8B-Instruct --out results/extended/adapters/q8_s0 --epochs 2 --batch 1 --accum 8 --balance --objective self --vision-lora --seed 0
python3 experiments/pwlock/finetune.py --repo Qwen/Qwen3.5-27B --out results/extended/adapters/q27_s0 --epochs 2 --batch 1 --accum 8 --balance --objective self --vision-lora --keep-best --seed 0
python3 experiments/pwlock/finetune.py --repo google/gemma-4-31B-it --out results/extended/adapters/g31c_s0 --epochs 3 --batch 1 --accum 8 --balance --objective self --vision-lora --aux-detect 1.0 --mark-scale 2.0 --mark-opacity 0.9 --lr 1e-4 --rank 32 --alpha 64 --keep-best --seed 0
python3 experiments/pwlock/finetune.py --repo google/gemma-4-12B-it --out results/extended/adapters/g12c_s0 --epochs 3 --batch 1 --accum 8 --balance --objective self --vision-lora --aux-detect 1.0 --mark-scale 2.0 --mark-opacity 0.9 --lr 1e-4 --rank 32 --alpha 64 --seed 0
python3 experiments/pwlock/finetune.py --repo google/gemma-4-12B-it --out results/extended/adapters/g12k_s0 --epochs 3 --batch 1 --accum 8 --balance --objective keep --vision-lora --aux-detect 1.0 --mark-scale 2.0 --mark-opacity 0.9 --lr 1e-4 --rank 32 --alpha 64 --seed 0   # no-lock control
# every attribution method on the mark, and the grey-picture transplant
python3 experiments/extended/pwlock_methods.py --adapter results/extended/adapters/q8_s0 --out results/extended/pwlock_methods/q8_s0 --conditions trained grey unrelated new clean --shap-samples 512 --rise-masks 500 --heavy-limit 40 --batch 64
```

**New manipulations (Appendix Table 35).** Each `build` writes pictures and prompt files; load them in `behaviour.py` and `layer_readouts.py` with `VLENS_EXTRA_FILES=results/extended/<h1|inject>/extra_files.json`.

```bash
python3 experiments/extended/legibility.py build
VLENS_EXTRA_FILES=results/extended/h1/extra_files.json python3 experiments/extended/legibility.py vwrite --repo $M
python3 experiments/extended/legibility.py analyze --model $N
python3 experiments/extended/inject.py build
VLENS_EXTRA_FILES=results/extended/inject/extra_files.json python3 experiments/extended/inject.py vwrite --repo $M --out results/extended/inject
python3 experiments/extended/inject.py analyze --model $N
python3 experiments/extended/h4_items.py
VLENS_COCO=results/extended/h4 python3 experiments/simonsays/behaviour.py --repo $M --dataset coco --families coco_attr coco_count coco_pos --cap 0 --batch 16 --state-grid 16 --out results/extended/h4/behaviour
VLENS_COCO=results/extended/h4 python3 experiments/simonsays/patch_causal.py --repo $M --dataset coco --families coco_attr coco_count coco_pos --subcategories present --n 0 --pool 0 --baselines mean --out results/extended/h4/patchcausal
VLENS_COCO=results/extended/h4 python3 experiments/extended/h4_analyze.py --model $N
```

**Built-in manipulations on eight models (Appendix F.1).**

```bash
python3 experiments/extended/g_behaviour.py --judged results/extended/judged --out results/extended/g_behaviour
python3 experiments/extended/g_edits.py --repo $M --out results/extended/g_edits
python3 experiments/extended/g_summary.py
```

**Beyond the first token (Appendix Table 36).**

```bash
python3 experiments/extended/multitoken.py --set coco --n 120 --out results/extended/j1
python3 experiments/extended/multitoken.py --set ss --n 150 --out results/extended/j1
python3 experiments/extended/multitoken.py summarise --out results/extended/j1
python3 experiments/extended/thinking.py run --repo Qwen/Qwen3.5-9B --max-think 512
python3 experiments/extended/thinking.py summarise --model Qwen3.5-9B
```

## 7. Tables, numbers and figures (CPU)

```bash
python3 tools/make_paper_tables.py        # Tables 1 and 2, cost and depth-window tables -> tables/
python3 tools/make_appendix_tables.py     # direction cosines, steering and password-lock tables
python3 tools/paper_numbers.py            # numbers quoted in the text
python3 tools/paper_figures.py            # every figure of the paper drawn from results/ -> figures/
```

The commands of sections 1–4 with `M` set to `google/gemma-4-26B-A4B-it` or `OpenGVLab/InternVL3_5-8B-HF` give the other-family results of Appendix G.
