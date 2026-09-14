#!/bin/bash
# Push the code to the GPU box without breaking it.
#
# `jlens` is a symlink into the vendored reference checkout; it is relative, so
# it is the same on both machines, but the checkout itself is not synced (clone
# it once on the box, see README).  Results, figures and data never travel this
# way: tools/pull.sh brings results back.
set -euo pipefail
BOX=${BOX:?set BOX=root@<host>}
cd "$(dirname "$0")/.."
rsync -az --exclude '__pycache__' --exclude '*.pyc' -e "ssh -o StrictHostKeyChecking=no" \
  vlens experiments build tools queue pyproject.toml README.md jlens "$BOX:/workspace/vlm/"
ssh -o StrictHostKeyChecking=no "$BOX" 'cd /workspace/vlm && python3 -c "
import sys; sys.path.insert(0, \".\")
import vlens, vlens.lens.reader, vlens.data.datasets, jlens.hooks   # noqa: F401  -- the check that matters
print(\"code pushed; imports clean\")"'
