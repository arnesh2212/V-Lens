# SPDX-License-Identifier: Apache-2.0
"""Draw every figure against invented results, to see which ones raise.

A figure that has never had data behind it has never been run, and the first
time it runs is the moment a fourteen-hour job finishes, on the box, at four
in the morning. This builds a result of the right shape for each one, in a
scratch directory, and calls the figure. It is checking that the code paths
execute, not that the picture is right: every number here is made up.

    python3 tools/dryrun_figures.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

#: two of them, with different block counts, because the pooled figures key
#: their results by block index and two models never share one
MODELS = [("Fake-7B", 0), ("Fake-40B", 20)]
MODEL = MODELS[0][0]
NB = 8
DEPTHS = [round((i + 1) / (NB + 1), 3) for i in range(NB)]
LAYERS = list(range(1, NB + 1))
LENSES = ["v_read", "v_read_cross", "tuned", "jacobian", "latent", "logit", "probe"]
FAMS = ["openimages_orig_auth", "nuscenes_edit_auth", "vins_orig_read",
        "gemini_edit_eval", "wmdp_bio_sand"]
KS = [0.05, 0.10, 0.20, 0.30, 0.50]

#: figures that report a quantised statistic -- the first depth at which some
#: standard is met, say -- where averaging two models can honestly land on the
#: same grid point and draw the same picture
COARSE = {"refusal_cascade"}


#: set per model, so a pooled figure drawn from one model's numbers can be
#: told apart from one that averaged two
TILT = 0.0


def curve(lo=0.4, hi=0.95):
    return [float(v) for v in np.linspace(lo + TILT, hi + TILT, NB)]


def band(c, w=0.05):
    return [[v - w for v in c], [v + w for v in c]]


def auroc_block(keys):
    a = {k: curve() for k in keys}
    return a, {k: band(v) for k, v in a.items()}


def write(root, rel, obj):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, default=float)


def build(root, model=MODEL, shift=0, tilt=0.0):
    """One invented result per experiment that a figure reads."""
    global TILT
    MODEL, TILT = model, tilt
    # ---- lens_read: lens_read_by_depth
    fam_entry = lambda: dict(
        family="x", n=120, layers=LAYERS, depths=DEPTHS, kinds=["yesno"],
        **dict(zip(("auroc", "ci"), auroc_block([f"{t}|{k}" for t in ("model", "data")
                                                 for k in LENSES]))))
    write(root, f"lensread/{MODEL}__read_summary.json",
          dict(repo=MODEL, families={f: fam_entry() for f in FAMS}))

    # ---- refusal: refusal_read, refusal_cascade
    au, ci = auroc_block([f"{t}|{k}" for t in ("refused", "harmful") for k in LENSES])
    write(root, f"safety/{MODEL}__refusal_auroc.json",
          dict(repo=MODEL, layers=LAYERS, depths=DEPTHS, n=300,
               instruments=LENSES, lens_meta={}, auroc=au, ci=ci))

    # ---- refusal_steer
    alphas = [0.0, 1.0, 2.0]
    names = ["V-Write", "difference of means (labelled)", "random"]
    blocks = [2 + shift, 4 + shift, 6 + shift]
    ind = {f"{L}|{d}|{al}": 0.2 + 0.1 * i + TILT
           for L in blocks for i, d in enumerate(names) for al in alphas}
    write(root, f"refusalsteer/{MODEL}__refusal_steer.json",
          dict(summary=dict(repo=MODEL, blocks=blocks, alphas=alphas, n=200,
                            directions=names, induced=ind, removed=dict(ind),
                            base_refusal=dict(induced=0.1, removed=0.8)),
               rows=[]))

    # ---- read_image: read_image, counterfactual
    # the names the experiment actually uses; a guess here draws an empty
    # panel and the figure looks like it works
    au, ci = auroc_block([f"{h}|{k}" for h in ("pooled", "mean", "top10", "best")
                          for k in LENSES])
    out = dict(repo=MODEL, layers=LAYERS, depths=DEPTHS, n=400, instruments=LENSES,
               meta={}, auroc=au, ci=ci,
               families={f: dict(n=80, depths=DEPTHS,
                                 auroc={f"model|{k}": curve() for k in LENSES})
                         for f in FAMS},
               counterfactual=dict(n=200,
                                   correlation={k: curve(0.0, 0.6) for k in LENSES},
                                   auroc={k: curve() for k in LENSES},
                                   localisation={k: curve() for k in LENSES}))
    write(root, f"readimage/{MODEL}__readimage.json", dict(summary=out, rows=[]))

    # ---- carrier
    grp = lambda: dict(n=90, final=0.4,
                       trace={k: curve(-1.0, 1.5) for k in ("v_read", "tuned", "logit")},
                       crossing={k: dict(at=0.45, lo=0.35, hi=0.55, never=0.02)
                                 for k in ("v_read", "tuned", "logit")})
    keys = ["as taken", "edited", "as taken|openimages", "edited|openimages",
            "as taken|nuscenes", "edited|nuscenes",
            "depicted sign|says yes", "depicted sign|says no"]
    write(root, f"carrier/{MODEL}__carrier.json",
          dict(summary=dict(repo=MODEL, layers=LAYERS, depths=DEPTHS, n=500, meta={},
                            groups={k: grp() for k in keys},
                            pair_gap={k: dict(n=120, mean=curve(-0.2, 1.1),
                                              lo=curve(-0.4, 0.9), hi=curve(0.0, 1.3),
                                              parts_at=0.42)
                                      for k in ("v_read", "tuned", "logit")}),
               rows=[]))
    return ["lens_read_by_depth", "refusal_read", "refusal_cascade", "refusal_steer",
            "read_image", "counterfactual", "carrier"]


def _bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _ink(path):
    """How much of the canvas is not the background it started as."""
    from PIL import Image
    im = Image.open(path).convert("L")
    a = np.asarray(im)
    return float((a < 245).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="leave the scratch results behind")
    ap.add_argument("--min-ink", type=float, default=0.012,
                    help="fraction of the canvas that must not be background")
    a = ap.parse_args()
    from vlens.figures import paper

    root = tempfile.mkdtemp(prefix="vlens_dryrun_")
    out = os.path.join(root, "figures")
    for i, (mo, shift) in enumerate(MODELS):
        names = build(os.path.join(root, "results"), mo, shift, tilt=-0.12 * i)
    bad, seen = 0, {}
    for nm in names:
        fn = getattr(paper, nm)
        for label, model in (("one model", MODEL), ("pooled", None)):
            paper.AVERAGE = label == "pooled"
            paper.SUFFIX = "__averaged" if paper.AVERAGE else f"__{model}"
            try:
                got = fn(results=os.path.join(root, "results"), out=out, model=model)
            except Exception as e:  # noqa: BLE001
                import traceback
                print(f"  RAISED {nm} ({label}): {type(e).__name__}: {e}")
                traceback.print_exc(limit=3)
                bad += 1
                continue
            if not got:
                print(f"  silent {nm} ({label})")
                continue
            ink = _ink(got)
            # a figure can draw its axes and nothing else, which is what an
            # averaged figure does when no two models share a key
            if ink < a.min_ink:
                print(f"  EMPTY  {nm} ({label}): {ink:.4f} of the canvas is drawn on")
                bad += 1
            else:
                note = ""
                if not paper.AVERAGE:
                    seen[nm] = _bytes(got)
                elif nm in seen and _bytes(got) == seen[nm] and nm not in COARSE:
                    # the two models carry different numbers, so an averaged
                    # figure identical to one of them never averaged anything
                    note = "  POOLED NOTHING"
                    bad += 1
                print(f"  drew   {nm} ({label})  ink {ink:.3f}{note}")
    paper.SUFFIX, paper.AVERAGE = "", False
    if a.keep:
        print(f"\n  left in {root}")
    else:
        shutil.rmtree(root, ignore_errors=True)
    print(f"\n{bad} of {2 * len(names)} calls raised")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
