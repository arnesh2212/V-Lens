#!/bin/bash
# Push the code to the GPU box without breaking it.
#
# Results, figures and data never travel this way: tools/pull.sh brings results back.
# it is the same on both machines, but the checkout itself is not synced (clone
# it once on the box, see README).  Results, figures and data never travel this
# way: tools/pull.sh brings results back.
set -euo pipefail
BOX=${BOX:?set BOX=root@<host>}
cd "$(dirname "$0")/.."
# The lane queues are live state on the box: lines get pinned, re-ordered and
# struck off while a run is going, and none of that is in the checkout. Push
# the scripts in queue/, never the queues themselves.
rsync -az --exclude '__pycache__' --exclude '*.pyc' \
  --exclude '*.q' --exclude '*.q.*' -e "ssh -o StrictHostKeyChecking=no" \
  vlens experiments build tools queue pyproject.toml README.md "$BOX:$REMOTE/"
ssh -o StrictHostKeyChecking=no "$BOX" 'cd $REMOTE && python3 -c "
import sys; sys.path.insert(0, \".\")
import vlens, vlens.lens.reader, vlens.data.simonsays   # noqa: F401  -- the check that matters
print(\"code pushed; imports clean\")"'
