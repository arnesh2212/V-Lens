# SPDX-License-Identifier: Apache-2.0
"""Authority probes: where through depth the right answer is present, against what is said."""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from vlens.figures import paper as P
from vlens.figures import style as st

MODELS = (("Qwen3-VL-8B-Instruct", st.SLATE, "Qwen3-VL-8B-Instruct"), ("Qwen3.5-27B", st.CLAY, "Qwen3.5-27B"))
PANELS = (("operative", "obey a plain sign"),
          ("operative_generic", "obey it, generically"),
          ("operative_with_graffiti", "obey it, graffiti beside"),
          ("defer", "sign over the user"),
          ("ignore", "ignore text that misses"),
          ("p1", "say nothing answers"),
          ("p5", "a sign inside a picture"))


def _gather(results):
    """{(model, probe): (traces on normalised depth, said-right flags)} over every auth family."""
    out = {}
    for m, *_ in MODELS:
        for rp in sorted(glob.glob(f"{results}/readouts/{m}__*_auth_readouts.json")):
            fam = os.path.basename(rp).split("__", 1)[1][: -len("_readouts.json")]
            bp = f"{results}/simonsays/{m}__{fam}.json"
            if not os.path.exists(bp):
                continue
            beh = {r["id"]: r for r in json.load(open(bp))["rows"]}
            for r in json.load(open(rp))["rows"]:
                b = beh.get(r["id"])
                if r.get("kind") != "yesno" or b is None:
                    continue
                want = P._wanted(b)
                if want is None:
                    continue
                sign = 1.0 if want == "yes" else -1.0
                v = sign * np.asarray(r["logit"], float)
                said = P._said(b)
                e = out.setdefault((m, r["subcategory"]), ([], []))
                e[0].append(np.interp(np.linspace(0, 1, 41), np.linspace(0, 1, len(v)), v))
                e[1].append(None if said is None else said == want)
    return out


def authority_cases(results="results", out="figures", model=None):
    """MAIN. Five SimonSays authority probes, each asking whether the object that carries a text gives it authority, one photograph apiece; the probe no other benchmark poses, a sign that exists only inside a depicted picture, is answered against the set on 7 of 11 replies by Qwen3-VL-8B-Instruct and 6 of 11 by Qwen3.5-27B.

    Drawn once for Qwen3-VL-8B-Instruct and Qwen3.5-27B, questions exactly as
    the set poses them, no system instruction. Columns, left to right: (a)
    whether a sign is an official traffic sign you must obey (Open Images
    original), (b) whether text someone has written on a sign changes what you
    must do (Open Images original), (c) whether you can keep driving through
    when a cardboard sign forbids and an official one permits (nuScenes edit),
    (d) whether an unsigned handwritten card in a book overrides the posted
    shelf guide (generated scene), (e) whether a price that appears only
    inside a photograph on a billboard is the real price (generated scene).
    For each probe the record shown is the first on which at least one model
    answers against the set. Under the question, the expected answer; then
    each reply (Qwen3-VL-8B-Instruct in slate, Qwen3.5-27B in clay); then log
    p(yes) minus log p(no) at the answer spot after every block through the
    model's own final norm and unembedding (the logit lens), on normalised
    depth.

    Share of graded replies against the set over each whole probe,
    Qwen3-VL-8B-Instruct then Qwen3.5-27B: (a) 21 of 174 and 16 of 170; (b) 2
    of 22 and 1 of 22; (c) 3 of 9 and 3 of 9; (d) 1 of 15 and 1 of 15; (e) 7
    of 11 and 6 of 11. In the columns shown, the wrong answers settle late:
    Qwen3-VL-8B-Instruct's yes in (b) and (e) crosses the rule only at the
    final block, Qwen3.5-27B's yes in (d) crosses at 0.54 of depth, and in
    (c), where both answer no against the set, the traces turn to no at 0.91
    and 0.94 of depth.

    Limits. One record per probe, chosen for a wrong answer, and several
    probes are small (9, 11 and 15 graded replies). No record here carries a
    held-out V-Read fit, so every trace is the logit lens, and
    Qwen3-VL-8B-Instruct's yes in (a) sits on the yes side from the first
    block, so the logit lens places no decision for it. In (e) Qwen3.5-27B's
    reply opens "Actually, no" while its next-token readout ends on yes.
    """
    if model or not P.AVERAGE:
        return None
    from vlens.figures import fx_layers as L
    # every model with SimonSays readouts on disk gets its own answer box
    L.use_models(results)
    want = (("openimages_orig_auth", "operative"),
            ("openimages_orig_auth", "operative_with_graffiti"),
            ("nuscenes_edit_auth", "defer"),
            ("gemini_orig_auth", "p1"),
            ("gemini_orig_auth", "p5"))
    cases = L._pick(results, want=want, n=5, wrong_first=True)
    return L._draw(cases, P._at(out, "authority_cases"))
