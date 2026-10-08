# SPDX-License-Identifier: Apache-2.0
"""Does the conservation law survive the change of F?

The lens decomposes one scalar.  Until now that scalar was a single answer
token's logit, which invites the objection that the result is an artefact of
picking a convenient object token.  :mod:`jvlm.objective` makes the scalar a
parameter and adds the two answer-level forms -- a teacher-forced
correct-versus-foil margin over whole sequences, and the teacher-forced
log-likelihood of a complete answer.

Completeness is the fundamental theorem of calculus along the straight path, so
it cannot care what ``F`` is; the *discretisation* can, because a log-softmax
over a 150k vocabulary is far more curved than one entry of it.  This run
measures both: the agreement between the summed readings and the measured
effect at a fixed step count, across Winoground items and two layers, beside a
permutation null; and a sweep over path steps on the first few items, which is
the price list for the extra curvature.

All the work is in :func:`jvlm.objective.gpu_check`.  This file exists so the
job is visible to the serial queue watchdog, which tracks ``code/exp/*.py``.
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "code")

from jvlm import objective  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--n", type=int, default=24, help="Winoground items")
ap.add_argument("--mean-images", type=int, default=60,
                help="held-out COCO photographs behind the baseline hbar")
ap.add_argument("--n-layers", type=int, default=2)
ap.add_argument("--steps", type=int, default=16, help="path interpolants")
ap.add_argument("--chunk", type=int, default=2, help="interpolants in memory")
ap.add_argument("--sweep-items", type=int, default=3,
                help="items that also get the step sweep")
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--device", default="cuda",
                help="cpu runs the same check without touching the card")
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--sweep-max", type=int, default=16,
                help="largest path length in the step sweep")
ap.add_argument("--data", default="data/winoground")
ap.add_argument("--out", default="results/objectives")
ap.add_argument("--tag", default=None)
a = ap.parse_args()

# the unit tests are cheap and catch a wiring change before the card is spent
print("== jvlm.objective unit tests (CPU) ==", flush=True)
R = objective.selftest()
bad = [k for k, v in R.items() if not k.startswith("_") and not v[0]]
if bad:
    raise SystemExit(f"unit tests failed: {bad}")

objective.gpu_check(a.repo, n=a.n, mean_images=a.mean_images,
                    n_layers=a.n_layers, steps=a.steps, chunk=a.chunk,
                    sweep_items=a.sweep_items, square=a.square,
                    sweep_steps=[s for s in (1, 2, 4, 8, 16, 32)
                                 if s <= a.sweep_max],
                    device=a.device, dtype=a.dtype,
                    max_pixels=a.max_pixels, data=a.data, out=a.out,
                    tag=a.tag)
