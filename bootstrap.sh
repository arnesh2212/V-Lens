#!/bin/bash
# Bring a fresh box to work in one command.
#
# The ordering here is the lesson of the last card rather than a preference.
# A Jacobian lens has to be *fitted* per model before it can appear in any
# comparison at all, and on the last box that fit finished after most of the 8B
# experiments had already run -- so nearly every 8B table shipped two-armed,
# and the arm was not missing because the method failed, it was missing
# because nobody had made it yet.  So the fits go first, on every model we
# intend to compare, and everything else queues behind them.
#
# Usage:  bash bootstrap.sh [--models "repo1 repo2"] [--layers "8 15 23 30"]
set -euo pipefail

VLM=/workspace/vlm
MODELS=${MODELS:-"Qwen/Qwen3-VL-8B-Instruct OpenGVLab/InternVL3-8B-hf"}
LAYERS=${LAYERS:-"8 15 23 30"}
FIT_N=${FIT_N:-64}

while [ $# -gt 0 ]; do
  case "$1" in
    --models) MODELS="$2"; shift 2;;
    --layers) LAYERS="$2"; shift 2;;
    --fit-n)  FIT_N="$2";  shift 2;;
    *) echo "unknown flag $1"; exit 2;;
  esac
done

cd "$VLM"
mkdir -p logs results

echo "=== card ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

echo
echo "=== probe each model before committing a run to it ==="
for repo in $MODELS; do
  echo "--- $repo"
  python code/exp/00_probe.py --repo "$repo" 2>&1 | tail -14 || \
    echo "  PROBE FAILED -- do not queue this model until it loads"
done

echo
echo "=== queue ==="
python3 - "$MODELS" "$LAYERS" "$FIT_N" <<'PY'
import os
import sys

models, layers, fit_n = sys.argv[1].split(), sys.argv[2], int(sys.argv[3])
Q = "/workspace/vlm/queue.txt"
ALLOC = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
rows = []


def add(name, need, cmd, tries=3):
    rows.append([name, str(tries), str(need), "1", cmd])


# 1. the fits, first, because every three-way comparison waits on them
for repo in models:
    tag = repo.split("/")[-1]
    add(f"fit_{tag[:14]}", 40000,
        f"python code/exp/01_fit_lens.py --repo {repo} --kind mm "
        f"--n {fit_n} --policy all --layers {layers.replace(' ', ',')} "
        "--dim-batch 8")

# 2. the defensive table: the one competitor that can also see the prompt has
#    to appear wherever we claim the others cannot follow
for repo in models:
    tag = repo.split("/")[-1]
    add(f"base_{tag[:13]}", 20000,
        ALLOC + f"python code/exp/96_baselines.py --repo {repo} --n 80 "
        "--mean-images 80 --n-layers 4 --ig-steps 8 --tag " + tag)

# 3. the claims, re-run at scale on every model
for repo in models:
    tag = repo.split("/")[-1]
    add(f"diss_{tag[:13]}", 24000,
        ALLOC + f"python code/exp/92_dissoc.py --repo {repo} --layers {layers} "
        "--n 60 --scan 4000 --mean-images 90 --gen-n 24 --boot 2000 "
        "--null-perm 200 --chunk-margin 4 "
        f"--out /workspace/vlm/results/dissoc --tag {tag}", tries=2)
    add(f"nat_{tag[:14]}", 24000,
        ALLOC + f"python code/exp/93_natural.py --repo {repo} --n 60 "
        f"--mean-images 80 --tag {tag}", tries=2)
    add(f"pair_{tag[:13]}", 24000,
        f"python code/exp/95_pairing.py --repo {repo} --per-source 90 "
        f"--mean-images 60 --ig-steps 4 --tag {tag}", tries=2)

tmp = Q + ".tmp"
open(tmp, "w").write("\n".join("|".join(r) for r in rows) + "\n")
os.replace(tmp, Q)
for r in rows:
    for suf in (".tries", ".done"):
        f = f"/workspace/vlm/logs/{r[0]}{suf}"
        if os.path.exists(f):
            os.remove(f)
print(f"{len(rows)} jobs queued, fits first:")
for i, r in enumerate(rows, 1):
    print(f"  {i:2d}. {r[0]:<20} need={r[2]:>6} MB")
PY

echo
echo "=== watchdog ==="
pkill -f "watchdog[.]sh" 2>/dev/null || true
sleep 2
setsid nohup bash watchdog.sh </dev/null >/dev/null 2>&1 &
disown || true
sleep 3
pgrep -f "watchdog[.]sh" >/dev/null && echo "  up" || echo "  FAILED TO START"

echo
echo "The J-lens fits run first.  Nothing else needs babysitting; watch with:"
echo "  tail -f $VLM/logs/watchdog.log"
