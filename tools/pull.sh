#!/bin/bash
# Pull results, figures and logs from the GPU box.  Checkpoints and caches are
# resumable state rather than results and stay there; pass "full" to include them.
set -uo pipefail
BOX=${BOX:?set BOX=root@<host>}
REMOTE=${REMOTE:-/workspace/temp/vlm}
cd "$(dirname "$0")/.."
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15"
EXCL="--exclude *.tmp.* --exclude *.ckpt --exclude *.npz"
[ "${1:-}" = "full" ] && EXCL="--exclude *.tmp.*"
for d in results figures logs; do
  rsync -az --partial $EXCL -e "$SSH" "$BOX:$REMOTE/$d/" "$d/"
done
echo "[pull $(date +%H:%M:%S)${1:+ full}] results=$(du -sh results 2>/dev/null | cut -f1) figures=$(find figures -name '*.png' | wc -l)"
