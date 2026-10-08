#!/bin/bash
# Fetch every dataset the experiments read, in the order they are needed.
#
# COCO first and alone: the corpus mean, the causal maps, the natural-image
# dissociation and the object lens all need it, so the moment it lands the
# queue can start.  Everything else downloads while the first jobs run.
set -uo pipefail

D=/workspace/data
S=data/safety
mkdir -p "$D" "$S" logs
LOG=logs/fetch.log
say () { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- COCO, first
if [ ! -d "$D/val2017" ]; then
  say "COCO val2017 images"
  curl -sSL -o /tmp/val2017.zip http://images.cocodataset.org/zips/val2017.zip
  unzip -q /tmp/val2017.zip -d "$D" && rm -f /tmp/val2017.zip
fi
if [ ! -f "$D/annotations/instances_val2017.json" ]; then
  say "COCO annotations"
  curl -sSL -o /tmp/ann.zip \
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -q -o /tmp/ann.zip -d "$D" && rm -f /tmp/ann.zip
fi
say "COCO ready: $(ls "$D/val2017" 2>/dev/null | wc -l) images"
touch logs/coco.ready

# ------------------------------------------------------- everything else after
export HF_HUB_ENABLE_HF_TRANSFER=1

py_snap () {   # repo, kind, dest
  python3 - "$1" "$2" "$3" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, kind, dest = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    snapshot_download(repo_id=repo, repo_type=kind, local_dir=dest,
                      max_workers=8)
    print(f"  ok {repo}")
except Exception as e:
    print(f"  FAILED {repo}: {type(e).__name__}: {e}")
PY
}

if [ ! -d "$D/clevr/images" ]; then
  say "CLEVR val"
  curl -sSL -o /tmp/clevr.zip \
    https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip || true
  if [ -s /tmp/clevr.zip ]; then
    unzip -q /tmp/clevr.zip -d /tmp/clevrx && mkdir -p "$D/clevr"
    mv /tmp/clevrx/CLEVR_v1.0/images "$D/clevr/" 2>/dev/null || true
    mv /tmp/clevrx/CLEVR_v1.0/scenes "$D/clevr/" 2>/dev/null || true
    rm -rf /tmp/clevr.zip /tmp/clevrx
  fi
  say "CLEVR: $(ls "$D/clevr/images/val" 2>/dev/null | wc -l) val images"
fi

say "Winoground"
py_snap facebook/winoground dataset "$D/winoground" | tee -a "$LOG"

say "safety benchmarks"
py_snap PKU-Alignment/MM-SafetyBench dataset "$S/mmsafetybench" | tee -a "$LOG"
py_snap AngelAlita/FigStep             dataset "$S/figstep"       | tee -a "$LOG"
py_snap Monosail/HADES                 dataset "$S/hades"         | tee -a "$LOG"
py_snap kirito011024/vlguard_safe_unsafes dataset "$S/vlguard_alt" | tee -a "$LOG"
py_snap JailbreakV-28K/JailBreakV-28k   dataset "$S/jbv28k"       | tee -a "$LOG"

say "all fetches attempted; see $LOG"
touch logs/data.ready
