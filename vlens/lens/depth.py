# SPDX-License-Identifier: Apache-2.0
"""Depth as a fraction, so models of different heights share one axis.

Layer 15 means something different in a 28-block model and a 48-block one.
Every cross-model plot and every "read at the same depth" comparison goes
through these two functions, so that "0.5" means the middle of the language
model on every architecture and no absolute index is ever hard-coded again.

The usable span runs from the first block after the deepstack injections (or
block 1 when there are none) to the block before the last, which is where a
lens can be read without reading either the embedding or the output head.
"""
from __future__ import annotations

from typing import Sequence


def span(m) -> tuple[int, int]:
    """First and last usable block indices for model ``m``."""
    lo = max(int(getattr(m, "n_deepstack", 0) or 0), 1)
    hi = int(m.n_layers) - 2
    return lo, hi


def to_layer(m, frac: float) -> int:
    """The block at fraction ``frac`` of the usable depth."""
    lo, hi = span(m)
    return int(round(lo + (hi - lo) * float(frac)))


def to_frac(m, layer: int) -> float:
    """Where block ``layer`` sits in the usable depth, in [0, 1]."""
    lo, hi = span(m)
    return (int(layer) - lo) / max(hi - lo, 1)


def layers_at(m, fracs: Sequence[float]) -> list[int]:
    """Distinct blocks at the given fractions, in order."""
    return sorted({to_layer(m, f) for f in fracs})


def every(m, k: int = 1) -> list[int]:
    """Every ``k``-th usable block, always including both ends."""
    lo, hi = span(m)
    out = list(range(lo, hi + 1, max(k, 1)))
    if out[-1] != hi:
        out.append(hi)
    return out


#: the fractions the write-up reads at when it needs a few depths, not all
STANDARD = (0.2, 0.35, 0.5, 0.65, 0.8, 0.95)
