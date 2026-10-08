#!/usr/bin/env bash
# Bring results back and redraw every figure that has data behind it.
#
# Figures are cheap and are drawn from the saved results, so they are rebuilt
# whenever something lands rather than once at the end: a figure that cannot
# be drawn is how a broken result announces itself early.
set -u
cd "$(dirname "$0")/.."
BOX=${BOX:-root@50.46.253.92}
PORT=${PORT:-9105}
REMOTE=${REMOTE:-/workspace/temp/vlm}
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p $PORT"
for d in results logs; do
  rsync -az --partial --exclude '*.tmp.*' --exclude '*.ckpt' --exclude '*.npz' \
    -e "$SSH" "$BOX:$REMOTE/$d/" "$d/" 2>/dev/null
done
# per model as well as pooled: with five models on the roster, drawing only
# whichever result is newest means four of them are never seen
python3 tools/draw_figures.py
