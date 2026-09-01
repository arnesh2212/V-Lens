#!/bin/bash
# Push code to the box without breaking it.
#
# `jlens` is a symlink whose target differs on each machine: it points into the
# reference checkout, which lives at a different absolute path here than there.
# rsync -a faithfully copies the local symlink over the remote one, the target
# stops existing, and the next job dies at `import jvlm` -- which is exactly how
# an afternoon of card time went the first time.  So it is never synced.
set -euo pipefail
BOX=${BOX:-root@217.18.55.64}
cd "$(dirname "$0")/.."
rsync -az --exclude 'jlens' --exclude '__pycache__' --exclude '*.pyc' \
  -e "ssh -o StrictHostKeyChecking=no" code/ "$BOX:/workspace/vlm/code/"
rsync -az -e "ssh -o StrictHostKeyChecking=no" scripts/ "$BOX:/workspace/vlm/scripts/"
ssh -o StrictHostKeyChecking=no "$BOX" \
  'ln -sfn /workspace/vlm/ref_jlens_repo/jacobian-lens/jlens /workspace/vlm/code/jlens
   cd /workspace/vlm && python3 -c "
import sys; sys.path.insert(0, \"code\")
import jvlm, jlens.hooks          # noqa: F401  -- the check that matters
print(\"code pushed; imports clean\")"'
