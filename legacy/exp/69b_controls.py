# SPDX-License-Identifier: Apache-2.0
r"""Controls for exp/69_jailbreak2.  CPU only -- reads the parquet it wrote.

Four things 69 does not check about its own numbers.  None of them needs the
model, so this can be re-run at any time without a GPU.

1. **A pixel baseline.**  MM-SafetyBench's harmful and benign halves are not
   matched on appearance: Stable Diffusion paints the harmful concepts darker.
   Six trivial image statistics, cross-validated, separate the SD block's two
   classes at AUROC 0.638 -- close enough to the lenses' own numbers that the
   comparison has to be made explicitly rather than assumed away.  Each lens
   reading is then residualised on those six features and re-scored.  What
   survives is the part of the reading that is not low-level appearance.

2. **The cross-source arm is not a generalisation check.**  69 compares HADES
   step-5 images against MM-SafetyBench's benign SD images and reports AUROC
   0.86-0.92 for all three lenses.  HADES is photographic and MM is painted:
   image *contrast* alone separates them at AUROC 1.000.  That arm measures
   which generator made the picture.  It cannot support a claim about harm and
   should not be quoted as one.

3. **Prompt-blindness, measured rather than argued.**  The native arm reads the
   same 140 images as the SD block under a different question, so the two
   blocks are a paired test of the theorem: a fixed-transport lens reading
   image positions cannot see the prompt.  139 of 140 items come back
   bit-identical for both content lenses; VJ-Lens moves on 132 of 140.

4. **The harm vocabulary contains prompt-tail words** (``caught``,
   ``examples``).  They can leak nothing under the unified template, but the
   native arm's questions do vary, so the readings are re-scored from the
   stored per-word ``S_`` columns with those words dropped.

Usage::

    python code/exp/69b_controls.py --tag qwen2b_v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pointbiserialr, rankdata, t as tdist

sys.path.insert(0, "code")

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="qwen2b_v2")
ap.add_argument("--dir", default="results/jailbreak")
ap.add_argument("--images", default="data/safety/_images")
ap.add_argument("--layer", type=int, default=17)
ap.add_argument("--variant", default="SD")
a = ap.parse_args()

df = pd.read_parquet(f"{a.dir}/{a.tag}_raw.parquet")
L = a.layer
FAM = [f for f in ("logit", "jlens", "vj") if f"{f}_region_{L}" in df]
NAME = {"logit": "logit lens", "jlens": "J-lens", "vj": "VJ-Lens"}
out: dict = {"tag": a.tag, "layer": L}


def auc(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, bool)
    r = rankdata(x)
    p, n = int(y.sum()), int((~y).sum())
    if p < 3 or n < 3:
        return float("nan")
    return float((r[y].sum() - p * (p + 1) / 2) / (p * n))


def p_of(r: float, n: int, k: int = 0) -> float:
    d = n - 2 - k
    if d < 1 or not (abs(r) < 1):
        return float("nan")
    return float(2 * tdist.sf(abs(r * np.sqrt(d / (1 - r * r))), d))


def path_of(i: str) -> str | None:
    """The cache path :func:`jvlm.safedata._write_image` would have written."""
    if i.startswith("mm/"):
        _, s, v, r = i.split("/")
        p = f"{a.images}/mm/{s}_{v}_{r}.png"
    else:
        p = f"{a.images}/hades/{i.split('/', 1)[1].replace('/', '_')}.png"
    return p if os.path.exists(p) else None


FEATS = ["luminance", "contrast", "edge density", "colourfulness",
         "png bytes", "peak histogram bin"]


def features(ids):
    """Six cheap appearance statistics.  Deliberately dumb: the point is that
    even these are enough, so a stronger vision baseline is not needed."""
    F, ok = [], []
    for i in ids:
        p = path_of(i)
        if p is None:
            ok.append(False)
            continue
        im = np.asarray(Image.open(p).convert("RGB").resize((224, 224)),
                        float) / 255.0
        g = im.mean(2)
        F.append([g.mean(), g.std(), np.abs(np.diff(g, axis=1)).mean(),
                  im.std(2).mean(), os.path.getsize(p) / 1e6,
                  float(np.histogram(g, 32, (0, 1))[0].max() / g.size)])
        ok.append(True)
    return np.asarray(F), np.asarray(ok, bool)


def residualise(x, F):
    Z = np.column_stack([np.ones(len(F)), (F - F.mean(0)) / (F.std(0) + 1e-9)])
    beta, *_ = np.linalg.lstsq(Z, np.asarray(x, float), rcond=None)
    return np.asarray(x, float) - Z @ beta


def cv_auc(F, y, seed=0) -> float:
    """Cross-validated AUROC of a logistic model on the pixel features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if min(int(y.sum()), int((~y).sum())) < 5:
        return float("nan")
    p = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        F, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        method="predict_proba")[:, 1]
    return auc(p, y)


print("=" * 74)
print("1. PIXEL BASELINE on the SD block: is the class visible without a lens?")
MM = df[(df.source == "mm_safetybench") & (df.block == a.variant)]
F, ok = features(MM.id.values)
MM = MM[ok]
y = MM.harmful.values.astype(bool)
print(f"   n={len(MM)} ({int(y.sum())} harmful / {int((~y).sum())} benign)")
for j, nm in enumerate(FEATS):
    print(f"     {nm:<20} AUROC {auc(F[:, j], y):.3f}")
pix = cv_auc(F, y)
print(f"     {'ALL SIX, 5-fold CV':<20} AUROC {pix:.3f}   <-- the baseline any "
      f"lens must beat")
print(f"\n   {'lens':<12}{'AUROC':>9}{'after removing the six':>25}")
q1 = {"pixels_cv_auc": pix, "n": int(len(MM)),
      "per_feature": {nm: auc(F[:, j], y) for j, nm in enumerate(FEATS)}}
for f in FAM:
    x = MM[f"{f}_region_{L}"].values.astype(float)
    r0, r1 = auc(x, y), auc(residualise(x, F), y)
    q1[f] = dict(auc=r0, auc_resid=r1)
    print(f"   {NAME[f]:<12}{r0:>9.3f}{r1:>25.3f}")
out["pixel_baseline_sd"] = q1

print()
print("=" * 74)
print("2. THE CROSS-SOURCE ARM: harm, or which generator drew it?")
EX, BEN = df[df.block == "extra"], MM[MM.harmful == 0]
if len(EX):
    Fe, oke = features(EX.id.values)
    Fb, okb = features(BEN.id.values)
    EX, BEN = EX[oke], BEN[okb]
    Fx = np.vstack([Fe, Fb])
    yx = np.r_[np.ones(len(EX), bool), np.zeros(len(BEN), bool)]
    print(f"   HADES n={len(EX)} vs MM-benign SD n={len(BEN)}")
    for j, nm in enumerate(FEATS):
        print(f"     {nm:<20} AUROC {auc(Fx[:, j], yx):.3f}")
    pex = cv_auc(Fx, yx)
    print(f"     {'ALL SIX, 5-fold CV':<20} AUROC {pex:.3f}")
    q2 = {"pixels_cv_auc": pex, "n_extra": int(len(EX)), "n_benign": int(len(BEN))}
    for f in FAM:
        v = auc(np.r_[EX[f"{f}_region_{L}"].values,
                      BEN[f"{f}_region_{L}"].values], yx)
        q2[f] = v
        print(f"   {NAME[f]:<12}{v:>9.3f}")
    print("   -> if the pixel row is at or near 1.000 this arm is a source "
          "artefact\n      and cannot be reported as a generalisation check.")
    out["cross_source_arm"] = q2

print()
print("=" * 74)
print("3. PROMPT-BLINDNESS, measured on the paired native arm")
sd = df[(df.block == a.variant) & (df.harmful == 1)].set_index("pair_id")
na = df[df.block == a.variant + "_native"].set_index("pair_id")
com = sd.index.intersection(na.index)
q3 = {"n_paired": int(len(com))}
if len(com):
    print(f"   {len(com)} images read twice -- unified template vs their own "
          f"question")
    print(f"   {'lens':<12}{'bit-identical':>15}{'median |diff|':>16}"
          f"{'max |diff|':>13}")
    for f in FAM:
        d = np.abs(sd.loc[com, f"{f}_region_{L}"].values.astype(float)
                   - na.loc[com, f"{f}_region_{L}"].values.astype(float))
        q3[f] = dict(exact=int((d == 0).sum()), median=float(np.median(d)),
                     max=float(d.max()))
        print(f"   {NAME[f]:<12}{f'{int((d==0).sum())}/{len(d)}':>15}"
              f"{np.median(d):>16.2e}{d.max():>13.2e}")
    print("   -> the two fixed-transport lenses are invariant to the question "
          "to\n      machine precision; VJ-Lens is not.  That is the "
          "prompt-blindness\n      theorem, measured rather than argued.")
    print(f"\n   compliance on those same images: template "
          f"{sd.loc[com,'complied'].mean():.3f} vs own question "
          f"{na.loc[com,'complied'].mean():.3f}")
    q3["complied_template"] = float(sd.loc[com, "complied"].mean())
    q3["complied_native"] = float(na.loc[com, "complied"].mean())
out["prompt_blindness"] = q3

print()
print("=" * 74)
print("4. COMPLIANCE: robustness to appearance and to prompt-tail vocabulary")
HARMW = [w for w in json.load(open(f"{a.dir}/{a.tag}_stats.json"))["vocab_harm"]]
BENW = json.load(open(f"{a.dir}/{a.tag}_stats.json"))["vocab_benign"]
W, TAIL = HARMW + BENW, {"caught", "examples", "detection"}


def rescore(d, drop, fam):
    keep = [i for i, w in enumerate(W) if w not in drop]
    hi = [j for j, i in enumerate(keep) if i < len(HARMW)]
    bi = [j for j, i in enumerate(keep) if i >= len(HARMW)]
    o = []
    for v in d[f"S_{fam}_{L}"].values:
        t = np.asarray(v, float)[keep]
        z = (t - t.mean()) / max(t.std(ddof=1), 1e-6)   # torch .std() is ddof=1
        o.append(z[hi].mean() - z[bi].mean())
    return np.asarray(o)


q4 = {}
for blk in [b for b in (a.variant + "_native", "SD_TYPO") if b in set(df.block)]:
    HB = df[(df.block == blk) & (df.harmful == 1)]
    Fh, okh = features(HB.id.values)
    HB = HB[okh]
    yv = HB.complied.values.astype(float)
    if len(HB) < 20 or len(np.unique(yv)) < 2:
        continue
    print(f"\n   block {blk}  n={len(HB)}  complied={yv.mean():.3f}")
    print(f"   {'lens':<12}{'r':>8}{'p':>9}{'r | pixels':>13}{'p':>9}"
          f"{'r | no tail words':>19}")
    q4[blk] = {}
    for f in FAM:
        x = HB[f"{f}_region_{L}"].values.astype(float)
        r0 = pointbiserialr(yv, x)[0]
        r1 = pointbiserialr(yv, residualise(x, Fh))[0]
        r2 = (pointbiserialr(yv, rescore(HB, TAIL, f))[0]
              if f"S_{f}_{L}" in HB else float("nan"))
        q4[blk][f] = dict(r=r0, p=p_of(r0, len(HB)), r_resid=r1,
                          p_resid=p_of(r1, len(HB), 6), r_no_tail=r2)
        print(f"   {NAME[f]:<12}{r0:>+8.3f}{p_of(r0,len(HB)):>9.4f}"
              f"{r1:>+13.3f}{p_of(r1,len(HB),6):>9.4f}{r2:>+19.3f}")
out["compliance_robustness"] = q4

p = f"{a.dir}/{a.tag}_controls.json"
json.dump(out, open(p, "w"), indent=1, default=float)
print(f"\nwrote {p}")
