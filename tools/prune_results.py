# SPDX-License-Identifier: Apache-2.0
"""Delete results the fairness audit invalidated, so none can reach the paper.

Anything produced before the audit fixes is removed rather than kept alongside
the new numbers. Behaviour passes survive: they are model outputs and no
instrument was fitted to produce them.
"""
from __future__ import annotations

import argparse
import os
import shutil

#: result directories whose contents were produced under a protocol the audit
#: found unfair, and the reason
INVALID = {
    "lensbench": "ridge penalty selected on the set the divergence was reported on",
    "lensbench_full": "same",
    "lensread": "tuned lens was 60 Adam steps, not the published optimiser",
    "lensread_fair": "same",
    "lensread_full": "same",
    "patchlenses": "lens attributions not baseline-subtracted; grading was circular",
    "patchlenses_fair": "grading was circular",
    "patchcausal": "grad x act measured against a different reference point",
    "patchcausal_full": "same",
    "readimage": "no cross-position baseline; penalty selected on the reported set",
    "visualtokens": "LatentLens datastore built from question text only",
    "crossset": "penalty selected with labels in a label-free claim",
    "safety": "probe trained on one target and scored against another",
    "safety_alllens": "same",
    "safety_sdtypo": "same",
    "pwlock": "probe trained on one target and scored against another",
    "hallucination": "folds split matched pairs",
    "occlusion": "no attribution baseline beside V-Write",
    "occlusion_full": "same",
    "handoff": "superseded by the full-data run",
    "handoff_full": "superseded",
    "reference": "superseded",
    "smoke": "smoke tests",
    "smoke2": "smoke tests", "smoke3": "smoke tests", "smoke4": "smoke tests",
    "smoke5": "smoke tests", "smoke6": "smoke tests", "smoke7": "smoke tests",
}
KEEP = {"simonsays", "simonsays_full"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--archive", default="archive_unfair")
    ap.add_argument("--delete", action="store_true", help="remove instead of archiving")
    a = ap.parse_args()
    moved = 0
    for name, why in INVALID.items():
        src = os.path.join(a.results, name)
        if not os.path.isdir(src) or name in KEEP:
            continue
        if a.delete:
            shutil.rmtree(src)
            print(f"  deleted {name}: {why}")
        else:
            os.makedirs(a.archive, exist_ok=True)
            dst = os.path.join(a.archive, name)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)
            print(f"  archived {name}: {why}")
        moved += 1
    print(f"{moved} directories cleared; kept {sorted(KEEP)}")


if __name__ == "__main__":
    main()
