# SPDX-License-Identifier: Apache-2.0
"""A lens-agnostic harness for grading region attributions on a VLM.

Every comparison in this project has the same shape: some method produces one
number per image position, those numbers are pooled over a region, and the
pooled value is graded against what actually happens when the region is
covered.  Writing that once, with the method behind an interface, removes the
question that otherwise attaches to every comparison -- whether the harness was
built around the method it flatters.

Two asymmetries this makes visible rather than hides:

*Aggregation is not neutral.*  Our per-position values are constructed to add:
summing them over a region is the defined operation and the sum has a
conservation law behind it.  A logit-lens or Jacobian-lens readout has no such
property, so summing it is a choice made *for* it rather than *by* it.  The
harness therefore treats aggregation as a free axis and reports every method at
its own best setting, not at ours.

*A region is not a localisation.*  Scoring three given boxes asks which one a
method ranks highest.  Finding the worst region among forty-nine asks the
method to localise.  These are different tasks with different chance rates, and
the harness keeps them apart rather than letting a win at one be quoted as the
other.

To add a method, implement ``Attribution`` -- one call returning a vector of
per-position scores.  Nothing else in the harness needs to know what it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np


class Attribution(Protocol):
    """One number per image position, for one question at one layer.

    The contract is deliberately narrow.  A method may read the prompt or not,
    may integrate along a path or not, may need gradients or not; the harness
    does not care and must not, because the moment it cares it has taken a side.
    """

    name: str

    def __call__(self, model, batch, layer: int, positions: Sequence[int]) -> np.ndarray:
        ...


# --------------------------------------------------------------- aggregation

def _sum(v: np.ndarray) -> float:
    return float(v.sum())


def _mean(v: np.ndarray) -> float:
    return float(v.mean())


def _max(v: np.ndarray) -> float:
    return float(v.max())


def _rms(v: np.ndarray) -> float:
    return float(np.sqrt((v ** 2).mean()))


AGGREGATIONS: dict[str, Callable[[np.ndarray], float]] = {
    "sum": _sum,      # the additive reading; native to a conserved attribution
    "mean": _mean,    # size-invariant; native to a per-position readout
    "max": _max,      # "does this region contain a strong position at all"
    "rms": _rms,      # magnitude without cancellation between signed values
}


# ------------------------------------------------------------------- grading

@dataclass
class Task:
    """One gradable instance: the regions, and what covering each one costs.

    ``truth[i]`` is the measured change in the answer when ``regions[i]`` is
    covered.  It must come from an intervention, never from another lens.
    """

    regions: list[Sequence[int]]
    truth: np.ndarray
    kind: str = "localise"          # "localise" over many regions, or "rank"
    image_id: str | None = None

    def chance_top1(self) -> float:
        return 1.0 / max(len(self.regions), 1)


@dataclass
class Result:
    method: str
    aggregation: str
    spearman: float
    top1: float
    chance: float
    n: int
    by_image: list[float] = field(default_factory=list)

    def lift(self) -> float:
        """Top-1 accuracy as a multiple of chance, which is the comparable form
        when region counts differ between tasks."""
        return self.top1 / self.chance if self.chance else float("nan")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def grade(scores_per_task: Sequence[np.ndarray],
          tasks: Sequence[Task],
          method: str,
          aggregation: str) -> Result:
    """Grade one method at one aggregation over a set of tasks.

    ``scores_per_task[i]`` holds the pooled score for each region of
    ``tasks[i]``, already reduced by the aggregation.
    """
    rhos, hits, per_image = [], [], []
    for s, t in zip(scores_per_task, tasks):
        s = np.asarray(s, float)
        if len(s) != len(t.truth):
            continue
        r = _spearman(s, t.truth)
        if r == r:
            rhos.append(r)
        hit = float(int(np.argmax(s)) == int(np.argmax(t.truth)))
        hits.append(hit)
        per_image.append(r)
    chance = float(np.mean([t.chance_top1() for t in tasks])) if tasks else float("nan")
    return Result(method=method, aggregation=aggregation,
                  spearman=float(np.mean(rhos)) if rhos else float("nan"),
                  top1=float(np.mean(hits)) if hits else float("nan"),
                  chance=chance, n=len(hits), by_image=per_image)


def pool(per_position: np.ndarray, task: Task, aggregation: str) -> np.ndarray:
    """Reduce a per-position vector to one score per region."""
    f = AGGREGATIONS[aggregation]
    return np.array([f(per_position[np.asarray(r, dtype=int)]) for r in task.regions])


def sweep(methods: dict[str, np.ndarray],
          tasks: Sequence[Task],
          aggregations: Sequence[str] = tuple(AGGREGATIONS)) -> list[Result]:
    """Every method at every aggregation, so each can be read at its own best.

    ``methods[name][i]`` is the per-position vector for task ``i``.  Reporting
    the full grid rather than one cell is the point: a single cell cannot
    distinguish a better transport from a better-suited pooling rule.
    """
    out = []
    for name, per_task in methods.items():
        for agg in aggregations:
            pooled = [pool(np.asarray(v, float), t, agg)
                      for v, t in zip(per_task, tasks)]
            out.append(grade(pooled, tasks, name, agg))
    return out


def best_per_method(results: Sequence[Result], by: str = "top1") -> dict[str, Result]:
    """Each method at the aggregation that suits it best.

    This is the comparison to publish.  Holding every method to the aggregation
    that suits ours is the failure mode this whole module exists to prevent.
    """
    best: dict[str, Result] = {}
    for r in results:
        cur = best.get(r.method)
        if cur is None or (getattr(r, by) > getattr(cur, by)):
            best[r.method] = r
    return best


def table(results: Sequence[Result]) -> str:
    """A plain-text grid, methods down and aggregations across."""
    methods = sorted({r.method for r in results})
    aggs = sorted({r.aggregation for r in results})
    if not methods:
        return "(no method produced a reading)"
    w = max(len(m) for m in methods) + 2
    head = " " * w + "".join(f"{a:>12}" for a in aggs)
    lines = [head]
    for m in methods:
        row = f"{m:<{w}}"
        for a in aggs:
            hit = next((r for r in results if r.method == m and r.aggregation == a), None)
            row += f"{hit.top1:>12.3f}" if hit else f"{'-':>12}"
        lines.append(row)
    return "\n".join(lines)
