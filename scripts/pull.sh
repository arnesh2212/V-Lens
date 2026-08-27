#!/bin/bash
# Pull results from the GPU box.  Checkpoints are large (a 8B lens checkpoint is
# ~2 GB) and are resumable state rather than results, so they go on a slower
# cycle: pass "full" to include them.
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15"
EXCL="--exclude *.tmp.* --exclude *.ckpt"
[ "$1" = "full" ] && EXCL="--exclude *.tmp.*"
for d in results figures logs paper; do
  rsync -az --partial $EXCL -e "$SSH" \
    root@217.18.55.95:/workspace/vlm/$d/ /home/arnesg/vlm/$d/ 2>/dev/null
done
echo "[pull $(date +%H:%M:%S)${1:+ full}] results=$(du -sh /home/arnesg/vlm/results 2>/dev/null|cut -f1) figs=$(ls /home/arnesg/vlm/figures 2>/dev/null|wc -l)"
