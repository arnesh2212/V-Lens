# SPDX-License-Identifier: Apache-2.0
"""Reading what the experiments wrote: behaviour rows and cached answer-spot states.

A behaviour file is ``<results>/<model>__<family>.json`` with a ``rows`` list;
its states, when kept, sit beside it as ``<model>__<family>_states.npz`` with
``ANS`` [n_items, n_layers, d_model] in fp16 and ``ids`` in the same order.
"""
from __future__ import annotations

import json
import os

import numpy as np


def path(results: str, model: str, family: str, suffix: str = ".json") -> str:
    return os.path.join(results, f"{model}__{family}{suffix}")


def rows(results: str, model: str, family: str) -> list[dict]:
    """The behaviour rows of one family, or an empty list when the file is missing."""
    p = path(results, model, family)
    return json.load(open(p))["rows"] if os.path.exists(p) else []


def states(results: str, model: str, family: str, kinds=("yesno", "letter", "test_real")):
    """Cached answer-spot states of one family as (ids, ANS float32), restricted to ``kinds``."""
    p = path(results, model, family, "_states.npz")
    if not os.path.exists(p):
        return [], None
    z = np.load(p)
    ids = [str(k) for k in z["ids"]]
    by = {r["id"]: r for r in rows(results, model, family)}
    keep = [i for i, k in enumerate(ids) if by.get(k, {}).get("kind") in kinds]
    return [ids[i] for i in keep], z["ANS"].astype(np.float32)[keep]


def write(results: str, model: str, family: str, payload: dict, suffix: str = ".json") -> str:
    os.makedirs(results, exist_ok=True)
    p = path(results, model, family, suffix)
    json.dump(payload, open(p, "w"), default=float)
    return p
