# SPDX-License-Identifier: Apache-2.0
"""Grading a patch ranking against a human object mask.

A map is one score per image token; the mask is, per token cell, the share of
that cell the object covers (``vlens.data.coco.cell_coverage``). A cell is
*inside* at half coverage or more, *outside* when the object does not touch it
at all, and a *boundary* cell otherwise. Every number comes with its chance
level, because a large object makes every map look good:

  hit1      pointing game (Zhang et al. 2016) on the token grid: the single
            strongest cell is inside.  chance = share of inside cells.
  hit1_any  the strongest cell touches the object at all -- the cell-level
            form of the pointing game's tolerance.  chance = share touched.
  hit5      any of the five strongest cells is inside.
  iou       the top-|M| cells against the |M| inside cells, so the map is asked
            for exactly as much area as the object has.
  auroc     does the map rank inside cells above outside ones (Mann-Whitney,
            ties averaged); boundary cells are left out, so a map is not
            penalised for pointing at a cell the object half fills.  chance
            0.5 whatever the object's size, so it is the one to compare
            across objects and datasets.  auroc_all keeps every cell, boundary
            cells counting as outside.
  energy    energy-based pointing game (Wang et al. 2020, Score-CAM): the
            share of the map's positive mass on the object, each cell's mass
            weighted by how much of it the object covers.  chance = the
            object's share of the frame.
"""
from __future__ import annotations

import numpy as np


def _auroc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y], score[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    # Mann-Whitney with average ranks for ties (the random map has none; the
    # centre prior has many)
    _, inv, cnt = np.unique(np.concatenate([pos, neg]), return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def grade(attr: np.ndarray, cov: np.ndarray, inside: np.ndarray) -> dict:
    """Every number above for one map; ``cov`` per-cell coverage, ``inside`` the cells that count."""
    a = np.asarray(attr, np.float64)
    c = np.asarray(cov, np.float64)
    y = np.asarray(inside, bool)
    touch = c > 0
    P, nM = len(a), int(y.sum())
    order = np.argsort(-a, kind="stable")
    top = np.zeros(P, bool)
    top[order[:nM]] = True
    keep = y | ~touch                     # inside and outside cells; boundary ones left out
    pos = np.clip(a, 0, None)
    return dict(
        hit1=float(y[order[0]]), hit1_any=float(touch[order[0]]),
        hit5=float(y[order[:5]].any()),
        iou=float((top & y).sum() / max((top | y).sum(), 1)),
        auroc=_auroc(a[keep], y[keep]), auroc_all=_auroc(a, y),
        energy=float((pos * c).sum() / pos.sum()) if pos.sum() > 0 else float("nan"),
        chance_hit1=nM / P, chance_hit1_any=float(touch.mean()), chance_energy=float(c.mean()),
        n_inside=nM, n_boundary=int((touch & ~y).sum()))


def center_prior(rows: int, cols: int | None = None) -> np.ndarray:
    """The map that always points at the middle: COCO objects sit near the centre,
    so any pointing claim has to beat this before it means anything."""
    cols = rows if cols is None else cols
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    d = np.sqrt(((ii - (rows - 1) / 2.0) / rows) ** 2 + ((jj - (cols - 1) / 2.0) / cols) ** 2).reshape(-1)
    # positive everywhere, so the energy score has mass to spread; the tiny
    # fixed jitter breaks the four-way tie at the centre of an even grid
    return (d.max() - d + 1e-6 * np.random.default_rng(0).standard_normal(d.size)).astype(np.float32)
