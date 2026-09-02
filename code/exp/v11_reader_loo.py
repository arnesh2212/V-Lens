# SPDX-License-Identifier: Apache-2.0
"""Leave-one-benchmark-out detection: the reader against HiddenDetect and a probe.

Reads the per-set caches written by v10.  For each held-out benchmark, the
reader's affine maps come from the disjoint COCO fit corpus (v09), the refusal
axis from text-only behaviour, and the probe is trained on every OTHER
benchmark's labelled activations at the same layer.  Two labels:

  (A) input harmfulness  -- harmful vs benign by dataset, pooled with a benign arm
  (B) the decision       -- refused vs not, where the held-out set's refusal rate is 10-90%

Metrics per depth: AUROC, TPR at 1% and 5% FPR, FPR at 95% TPR; false-alarm
rate on the benign arms at the operating point.  CPU only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--sets", nargs="*", default=["mmsb_typo", "figstep", "hades", "vlguard", "vlsbench", "siuo", "mossbench_contrast", "spavl_harm"])
ap.add_argument("--benign", nargs="*", default=["mossbench", "mmvet", "spavl_help"])
ap.add_argument("--fracs", type=float, nargs="*", default=[0.2, 0.35, 0.5, 0.65, 0.8])
ap.add_argument("--sets-dir", default="/workspace/vlm/results/sets")
ap.add_argument("--reader", default=None, help="results/reader/<model>__reader.pt")
ap.add_argument("--axis-json", default=None)
a = ap.parse_args()

from jvlm import cache, judge, reader, verdict  # noqa: E402

name = a.repo.split("/")[-1]
R = reader.Reader.load(a.reader or f"/workspace/vlm/results/reader/{name}__reader.pt")
say = lambda *x: print(*x, flush=True)


def load(s):
    p = f"{a.sets_dir}/{name}__{s}.npz"
    if not os.path.exists(p):
        return None
    acts, side = cache.load(p)
    A = side["_extra"]["ANS"].astype(np.float32)
    its = side["items"]
    return dict(ANS=A, harmful=np.array([bool(it["harmful"]) for it in its]), refused=np.array([judge.item_refused(it) for it in its]),
                complied=np.array([bool(judge.label(it)) for it in its]), n=len(its))

D = {s: load(s) for s in a.sets + a.benign}
D = {k: v for k, v in D.items() if v is not None}
say(f"{name}: sets with caches: {sorted(D)}")
nL = next(iter(D.values()))["ANS"].shape[1]
from jvlm import depth as _d
class _M: n_layers = nL; n_deepstack = 3
layers = _d.layers_at(_M, a.fracs)

# the refusal axis from the fit-corpus run's behaviour anchor is not stored; derive from ALL sets' final-layer
# behaviour EXCEPT the held-out one (still no held-out label)
def axis_excluding(held):
    H = np.concatenate([D[s]["ANS"][:, -1] for s in D if s != held]); y = np.concatenate([D[s]["refused"] for s in D if s != held])
    return verdict.dim_direction(H, y.astype(int)).numpy()

def family_scores(held, r_axis, layers, folds=5, seed=0):
    """The same read, but with the map fitted on the held-out set's own states.

    That is the setting a practitioner is actually in: the traffic is available,
    its labels are not.  Five folds over items, so no item is ever read through
    a map that saw it, which makes this comparable with the cross-validated
    probe on the same rows.
    """
    A = D[held]["ANS"]
    n = len(A)
    idx = np.random.default_rng(seed).permutation(n)
    out = {int(l): np.zeros(n) for l in layers}
    for te in np.array_split(idx, folds):
        tr = np.setdiff1d(np.arange(n), te)
        Rf = reader.fit(A[tr], layers, final_layer=A.shape[1] - 1)
        for l in layers:
            out[int(l)][te] = Rf.read(A[te, l], l, r_axis)
    return out


def tpr_at_fpr(y, s, fpr_target):
    fpr, tpr, _ = roc_curve(y, s); return float(np.interp(fpr_target, fpr, tpr))
def fpr_at_tpr(y, s, tpr_target):
    fpr, tpr, _ = roc_curve(y, s); return float(np.interp(tpr_target, tpr, fpr))

out = dict(repo=a.repo, layers=layers, results=[])
for held in a.sets:
    if held not in D:
        continue
    ben = np.concatenate([D[b]["ANS"] for b in a.benign if b in D]) if any(b in D for b in a.benign) else None
    say(f"\n== held out: {held}  (n={D[held]['n']}, harmful {D[held]['harmful'].mean():.2f}, refused {D[held]['refused'].mean():.2f})")
    say(f"{'depth':>6} {'reader':>7} {'HiddenD':>8} {'probe':>6} | {'TPR@1%':>7} {'TPR@5%':>7} {'FPR@95':>7} | {'FA-benign':>9} | {'read':>7} {'read-fam':>7} {'raw':>7} {'probe':>7}")
    say(f"{'':>6} {'[A: is the input harmful, against the shared benign arm]':<60} {'':>12} | {'[B: does the model refuse it]':<24}")
    r = axis_excluding(held)
    t_fam = time.time()
    FAM = family_scores(held, r, layers)
    say(f"   map refitted on this set's own states, 5 folds over {D[held]['n']} items ({time.time() - t_fam:.0f}s)")
    for l in layers:
        # (A) harmful (held-out set) vs benign arms
        X_pos = D[held]["ANS"][:, l][D[held]["harmful"]]
        X_neg = ben[:, l] if ben is not None else D[held]["ANS"][:, l][~D[held]["harmful"]]
        if len(X_pos) < 10 or len(X_neg) < 10:
            continue
        X = np.concatenate([X_pos, X_neg]); y = np.r_[np.ones(len(X_pos)), np.zeros(len(X_neg))]
        s_reader = R.read(X, l, r); s_id = X @ r
        # probe trained on the OTHER harmful sets vs the benign arms
        Xtr = np.concatenate([D[s]["ANS"][:, l][D[s]["harmful"]] for s in a.sets if s in D and s != held] + ([X_neg] if ben is not None else []))
        ytr = np.r_[np.ones(sum(D[s]["harmful"].sum() for s in a.sets if s in D and s != held)), np.zeros(len(X_neg) if ben is not None else 0)]
        if len(np.unique(ytr)) == 2:
            sc = StandardScaler().fit(Xtr); clf = LogisticRegression(max_iter=600, C=0.1).fit(sc.transform(Xtr), ytr)
            s_probe = clf.decision_function(sc.transform(X))
        else:
            s_probe = np.zeros(len(X))
        row = dict(held=held, layer=l, frac=_d.to_frac(_M, l),
                   auroc=dict(reader=roc_auc_score(y, s_reader), hidden=roc_auc_score(y, s_id), probe=roc_auc_score(y, s_probe)),
                   tpr1=tpr_at_fpr(y, s_reader, 0.01), tpr5=tpr_at_fpr(y, s_reader, 0.05), fpr95=fpr_at_tpr(y, s_reader, 0.95))
        # false alarms on the benign arms at the threshold giving 95% TPR on this held-out set
        thr = np.sort(s_reader[y == 1])[int(0.05 * (y == 1).sum())]
        row["fa_benign"] = float((s_reader[y == 0] >= thr).mean())
        # (B) the decision inside the held-out set: refused vs not, which no dataset
        # identity can answer, unlike (A) where the benign arm is shared with training
        ref = D[held]["refused"]
        if 10 <= ref.sum() <= len(ref) - 10:
            Xb = D[held]["ANS"][:, l]
            sb_reader = R.read(Xb, l, r); sb_id = Xb @ r
            try:
                scb = StandardScaler().fit(Xb); clf_b = LogisticRegression(max_iter=600, C=0.1)
                from sklearn.model_selection import cross_val_predict
                sb_probe = cross_val_predict(clf_b, scb.transform(Xb), ref.astype(int), cv=5, method="decision_function")
            except Exception:
                sb_probe = np.zeros(len(Xb))
            row["decision"] = dict(reader=roc_auc_score(ref, sb_reader), hidden=roc_auc_score(ref, sb_id),
                                   probe=roc_auc_score(ref, sb_probe),
                                   reader_family=roc_auc_score(ref, FAM[int(l)]),
                                   n_refused=int(ref.sum()), n=int(len(ref)))
        out["results"].append(row)
        dec = row.get("decision", {})
        say(f"{row['frac']:>6.2f} {row['auroc']['reader']:>7.3f} {row['auroc']['hidden']:>8.3f} {row['auroc']['probe']:>6.3f} | {row['tpr1']:>7.3f} {row['tpr5']:>7.3f} {row['fpr95']:>7.3f} | {row['fa_benign']:>9.3f}"
            f" | {dec.get('reader', float('nan')):>7.3f} {dec.get('reader_family', float('nan')):>7.3f}"
            f" {dec.get('hidden', float('nan')):>7.3f} {dec.get('probe', float('nan')):>7.3f}")
json.dump(out, open(f"/workspace/vlm/results/reader/{name}_loo.json", "w"), indent=1, default=float)
say("wrote loo json")
