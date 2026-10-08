#!/usr/bin/env bash
# Drop the state caches for one model once the analyses that read them are done.
# The answers, margins and every result file stay; only the intermediate
# activation caches go, and only for the model named.
set -u
M="${1:?usage: free_shm.sh <model-name>}"
D=/dev/shm/vlens_results/simonsays_full
n=$(ls "$D"/"$M"__*_states.npz 2>/dev/null | wc -l)
rm -f "$D"/"$M"__*_states.npz
echo "dropped $n state caches for $M"
df -h /dev/shm | tail -1
