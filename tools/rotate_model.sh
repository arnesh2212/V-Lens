#!/usr/bin/env bash
# Swap the model on disk: drop one repo's weights, fetch another.
#
#   tools/rotate_model.sh <repo-to-drop> <repo-to-fetch>
#
# The container has a 32 GB overlay and no volume, so only one large model fits
# at a time. Everything derived from the model being dropped is already on the
# local machine by the time this runs; only the weights go.
set -u
DROP="${1:?usage: rotate_model.sh <drop-repo> <fetch-repo>}"
FETCH="${2:?}"
export HF_HOME=/workspace/temp/vlm/hf
cd "$(dirname "$0")/.."
source /venv/main/bin/activate 2>/dev/null || true
# Two lanes share the weights on disk, so dropping them while the other lane
# is still reading is how a lane dies with no explanation. Wait it out.
for _ in $(seq 1 240); do
  n=$(pgrep -cf -- "--repo $DROP" 2>/dev/null | head -1); n=${n:-0}
  [ "$n" -lt 1 ] && break
  echo "$(date +%H:%M) waiting: $n job(s) still on $DROP"
  sleep 60
done

d="$HF_HOME/hub/models--${DROP//\//--}"
if [ -d "$d" ]; then
  echo "dropping $DROP ($(du -sh "$d" | cut -f1))"
  rm -rf "$d"
fi
# A cached fit is worth gigabytes and is for exactly one model. Both lanes
# share the cache directory, so only the dropped model's entries go.
python3 tools/drop_fits.py "$DROP"
df -h / | tail -1
echo "fetching $FETCH"
HF_HUB_OFFLINE=0 python3 - "$FETCH" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
p = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                            "*.py", "*.jinja"])
print("ready:", p, flush=True)
PY
df -h / | tail -1
