#!/bin/bash
# Keep the laptop and the GPU box in step, without dragging the 1.4 GB of
# Jacobian-lens checkpoints back and forth or re-importing the smoke runs and
# retracted-method results that were deliberately deleted.
BOX="root@217.18.55.95"
SSH="ssh -o StrictHostKeyChecking=no"
EX=(--exclude '*.ckpt' --exclude '*smoke*' --exclude 'secant*' --exclude 'refit'
    --exclude 'nonlinear' --exclude 'disjoint' --exclude 'objectlens*'
    --exclude 'gate0' --exclude 'directroutes' --exclude 'compare_smoke'
    --exclude '.*.ckpt.*' --exclude '__pycache__')

case "${1:-both}" in
  push|both) rsync -az -e "$SSH" "${EX[@]}" code/ "$BOX":/workspace/vlm/code/ ;;&
  pull|both)
    rsync -az -e "$SSH" "${EX[@]}" "$BOX":/workspace/vlm/results/ results/
    rsync -az -e "$SSH" "${EX[@]}" "$BOX":/workspace/vlm/figures/ figures/ ;;
esac
echo "results $(du -sh results | cut -f1) in $(ls results | wc -l) dirs"
echo "figures $(ls figures/*.png 2>/dev/null | wc -l) png, $(ls figures/web/*.webp 2>/dev/null | wc -l) webp"
