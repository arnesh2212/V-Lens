#!/bin/bash
# Copy just this track's two files to the box, render, and bring the PNGs back.
set -e
BOX="root@217.18.55.95"
SSH="ssh -o StrictHostKeyChecking=no"
cd /home/arnesg/vlm
rsync -az -e "$SSH" code/jvlm/figmore.py "$BOX":code/jvlm/
rsync -az -e "$SSH" code/exp/79_morefigs.py "$BOX":code/exp/
$SSH "$BOX" "cd $REMOTE && python3 code/exp/79_morefigs.py --out figures/new79 --which ${1:-all} 2>&1 | tail -80"
mkdir -p figures/new79
rsync -az -e "$SSH" "$BOX":figures/new79/ figures/new79/
