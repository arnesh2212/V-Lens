#!/usr/bin/env bash
# Swap the large model held in RAM: drop one repo's weights from /dev/shm and
# fetch another there. /dev/shm is 62 GB and holds one 27B-class model at a time.
set -u
DROP="${1:?usage: rotate_shm.sh <drop-repo> <fetch-repo> [hf-home]}"
FETCH="${2:?}"
HOME_DIR="${3:-/dev/shm/gemma_hf}"
cd "$(dirname "$0")/.."
source /venv/main/bin/activate 2>/dev/null || true
for root in /dev/shm/qwen_hf /dev/shm/gemma_hf; do
  d="$root/hub/models--${DROP//\//--}"
  [ -d "$d" ] && { echo "dropping $DROP ($(du -sh "$d" | cut -f1))"; rm -rf "$d"; }
done
rm -rf /dev/shm/qwen_hf/hub/models--neuronpedia--jacobian-lens 2>/dev/null
# A cached fit is worth gigabytes and is for exactly one model. Both lanes
# share the cache directory, so only the dropped model's entries go.
python3 tools/drop_fits.py "$DROP"
df -h /dev/shm | tail -1
mkdir -p "$HOME_DIR"
echo "fetching $FETCH into $HOME_DIR"
HF_HOME="$HOME_DIR" HF_HUB_OFFLINE=0 python3 - "$FETCH" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt",
                                                   "*.model", "*.py", "*.jinja"])
print("ready:", p, flush=True)
PY
df -h /dev/shm | tail -1
