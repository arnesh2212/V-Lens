"""Print ``--jlens <path>`` for a model, or nothing at all.

Every comparison script takes the Jacobian lens as an explicit path and
silently drops the arm when the flag is absent.  That is how the first round of
8B tables came to ship two-armed: the fit existed, nobody pointed at it.  So the
queue interpolates this, and a missing fit removes the flag rather than passing
an empty string that argparse would swallow as the next argument.
"""
import glob
import os
import sys

LENSES = "/workspace/vlm/results/lenses"

repo = sys.argv[1]
name = repo.split("/")[-1]
hits = [p for p in glob.glob(f"{LENSES}/{name}__mm__*.pt") if os.path.getsize(p) > 0]
if hits:
    print("--jlens " + max(hits, key=os.path.getmtime))
