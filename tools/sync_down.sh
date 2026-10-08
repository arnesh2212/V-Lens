#!/usr/bin/env bash
# Pull results and logs off the box, on a loop, so nothing lives only on a
# rented machine.  State caches are large, so they come down too but are left
# in place on the server until a run is finished with them.
set -u
REMOTE="root@50.46.253.92"
PORT=9105
SRC=/workspace/temp/vlm
DST="$(cd "$(dirname "$0")/.." && pwd)"
EVERY="${EVERY:-600}"
while true; do
  for d in results logs queue figures; do
    rsync -az --partial -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $PORT" \
      --exclude='*.tmp*' "$REMOTE:$SRC/$d/" "$DST/$d/" 2>/dev/null
  done
  # the full-data run writes to the RAM disk because the container's own disk is
  # nearly full; its answers and margins are primary data and come down here.
  # The state caches are intermediate and stay on the box until the analysis
  # that reads them has run.
  mkdir -p "$DST/results/simonsays_full"
  rsync -az --partial -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $PORT" \
    --exclude='*_states.npz' --exclude='*.tmp*' \
    "$REMOTE:/dev/shm/vlens_results/simonsays_full/" "$DST/results/simonsays_full/" 2>/dev/null
  rsync -az --partial -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $PORT" \
    "$REMOTE:/dev/shm/vlens_pw/" "$DST/results/pwlock_adapters/" 2>/dev/null
  printf '%s  synced\n' "$(date +%F' '%H:%M:%S)" >> "$DST/logs/sync.log"
  sleep "$EVERY"
done
