# SPDX-License-Identifier: Apache-2.0
"""The labelled probe (supervised reference): a logistic probe per block for the answer the model gives.

Every label-free reader of the answer spot (the logit lens, V-Read, the tuned
and Jacobian lenses, Patchscopes) is asked the same question per block: does
the state already say what the model will answer? A probe fitted with the
answers as labels is drawn beside them as a supervised reference, in the form
the probing literature uses
(Alain and Bengio 2016; Belinkov 2022): one L2-regularised logistic
regression per block, on the stored answer-spot state, scored out of fold.

The label is the model's own final answer as its reply states it, by the same
rules :mod:`tools.layer_readouts` checks the head against:

* yes or no: 1 when the reply opens with yes (an opening "actually, yes"
  counts), 0 when it opens with no;
* test or real: 1 when the reply opens with test, 0 when it opens with real;
* a letter: 1 when the letter the reply opens with is the correct one, 0 when
  it is another letter.

A reply that states none of these is left out, never guessed. For yes or no
and test or real the probe is a binary regression on that label and its
reading is its log-odds. For a letter the probe learns what every lens reads,
which letter the model picks: a four-way (multinomial) regression on the
letter the reply opens with, whose reading is then built exactly as a lens
reading is, the log-odds of the correct letter over the best other. A binary
probe on "correct or not" would have to know each question's correct letter
from the state, which the lenses are not asked to do, since their score is
told it.

By default (``--mode pooled``) the probe is fitted per task rather than per
family: families that share a target type are pooled (every WMDP subject
under one condition, for letter correctness; the yes/no authority families
of every source; the test/real eval-awareness families; the yes/no ones),
the folds are grouped by picture across the pool (an unedited photograph and
every edit of it in one group, whichever family each came from) and
stratified by family and answer, and each family is scored on its own
held-out records. Inside each training split the regression may also work on
the top ``k`` principal components of the standardised training rows,
``k`` in 16, 64, 256 or all, chosen together with ``C`` by the same inner
grouped split (standardisation and PCA fitted on the training rows only).
The earlier per-family probe is ``--mode family`` (written as
``<family>_probe_family.json``), kept for transparency. In both modes, per
block, on every stored block:

* five folds, grouped by picture so that no picture is on both sides of a
  split; a picture is the source photograph when the record is an edit of one
  (edits of one photograph share a group), else the record's ``group_id`` (for
  the WMDP carrier, the question, which keeps its conditions together), else
  its image number. Folds are stratified by the fitted label where the groups
  allow, and a family with fewer than five pictures takes as many folds as it
  has;
* inside each training split, the penalty ``C`` is chosen from a grid of ten
  decades (1e-7 to 100; the earlier six-decade grid, 1e-5 to 1, was chosen at
  both of its ends in many folds) by a three-fold grouped cross-validation on
  that split alone, on
  pooled out-of-fold log-loss, the smaller ``C`` winning a tie;
* features are standardised on the training split (a floor on the scale keeps
  a dimension that barely moves there from blowing up on the held-out fold).

The fit is exact but cheap: with far fewer records than dimensions, the L2
solution lies in the span of the training states, so the regression is solved
on their coordinates in that span (the thin SVD, ``U S``, taken through the
small Gram matrix) by Newton's method and mapped back through ``V``. Loss and
penalty are unchanged by the change of basis, so the weights are the ones a
solver on the full width would reach.

For each record and block the file keeps the out-of-fold reading (the probe
never saw the record's picture), and for each block the AUROC of those
readings against the labels, with the ``C`` each fold chose. Nothing runs the
model and nothing needs a GPU; the states are read one family at a time and
one block at a time in float32, so the largest family of the 27B holds under
300 MB.

    python3 tools/probe_readouts.py --model Qwen3-VL-8B-Instruct
    python3 tools/probe_readouts.py --model Qwen3.5-27B --families vins_orig_auth

    python3 tools/probe_readouts.py --model Qwen3.5-27B --mode family

writes ``results/readouts/<model>__<family>_probe.json`` (``_probe_family.json`` per family) with rows
``{id, kind, y, probe: [per block]}``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

from layer_readouts import LETTERS, LOCKED, said_letter, said_yesno  # noqa: E402

#: the penalty grid, in units of standardised features
C_GRID = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
FOLDS, INNER = 5, 3
#: a family is probed when it has this many labelled records and both answers
MIN_N, MIN_CLASS = 10, 2
#: how far below the lowest fitted letter a letter no training reply chose is put
ABSENT = 20.0


def own_answer(r):
    """1, 0 or None: the model's own final answer, as its reply states it."""
    kind, reply = r.get("kind"), r.get("reply")
    if kind == "yesno":
        s = said_yesno(reply)
        return None if s is None else int(s == "yes")
    if kind == "test_real":
        t = str(reply or "").strip().lower()
        return 1 if t.startswith("test") else 0 if t.startswith("real") else None
    if kind == "letter":
        s, e = said_letter(reply), expected_letter(r)
        return None if s is None or e is None else int(s == LETTERS[e])
    return None


def expected_letter(r):
    """The index of a letter record's correct letter, or None."""
    e = str(r.get("expected") or "").strip().upper()[:1]
    if e and e in LETTERS:
        return LETTERS.index(e)
    if r.get("answer_index") is not None and 0 <= int(r["answer_index"]) < len(LETTERS):
        return int(r["answer_index"])
    return None


def picture(r):
    """What a record's picture is, for grouping: the source photograph of an edit, else its group."""
    m = r.get("meta") or {}
    for k in ("source_image", "group_id", "image_number"):
        if m.get(k) not in (None, ""):
            return f"{k}:{m[k]}"
    return f"image:{r.get('image') or r.get('id')}"


# ------------------------------------------------------------------ the probe

def _scale(X):
    mu = X.mean(0)
    sd = X.std(0)
    return mu, np.maximum(sd, 1e-2 * float(np.median(sd)) + 1e-8)


def _span(X, mu, sd):
    """Standardise, then the coordinates of the rows in their own span, and the map back.

    The thin SVD ``Z = U S V^T`` taken through the records' Gram matrix, which
    is small: ``Z Z^T = U S^2 U^T``, so ``U S`` are the coordinates and
    ``V^T = S^-1 U^T Z`` maps them back to the full width.
    """
    Z = ((X - mu) / sd).astype(np.float64)
    lam, U = np.linalg.eigh(Z @ Z.T)
    lam, U = lam[::-1], U[:, ::-1]
    keep = lam > max(float(lam[0]), 1e-30) * 1e-10
    S = np.sqrt(lam[keep])
    U = U[:, keep]
    # columns in decreasing variance: since the training rows are centred, the first
    # k columns are the rows' coordinates on their top-k principal components
    return U * S, (U / S).T @ Z


def _fit_many(X, t, Cs, ks=(None,)):
    """One L2 logistic regression per (``k``, ``C``); returns ``[(W, b, classes)]`` in the units of ``X``,
    ``k`` varying slowest.

    ``k`` is the number of principal components of the standardised training
    rows the regression sees (PCA fitted on these rows only), None for all of
    them. Two classes give one row of weights, the log-odds of the second;
    more give one row per class, the multinomial fit.
    """
    from sklearn.linear_model import LogisticRegression
    mu, sd = _scale(X)
    F, Vt = _span(X, mu, sd)
    ncls = len(set(np.asarray(t).tolist()))
    done = {}
    out = []
    for k in ks:
        r = F.shape[1] if k is None else min(int(k), F.shape[1])
        if r not in done:
            # one path per width, warm-started from the smaller C (the optimum is unique, so
            # only the iterations change); Newton's method while its Hessian is small
            wide = r * (ncls if ncls > 2 else 1) > 1024
            lr = LogisticRegression(solver="lbfgs" if wide else "newton-cholesky",
                                    max_iter=2000 if wide else 200, tol=1e-6 if wide else 1e-8,
                                    warm_start=True)
            fits = {}
            for C in sorted(Cs):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lr.set_params(C=C).fit(F[:, :r], t)
                W = (lr.coef_ @ Vt[:r]) / sd
                fits[C] = (W, lr.intercept_ - W @ mu, lr.classes_)
            done[r] = fits
        out += [done[r][C] for C in Cs]
    return out


def _logits(X, fit, K):
    """``[n]`` log-odds when ``K`` is 1, else ``[n, K]`` log-softmax over the ``K`` classes."""
    W, b, cls = fit
    Z = X @ W.T + b
    if K == 1:
        return Z[:, 0]
    full = np.full((X.shape[0], K), np.nan)
    if len(cls) == 2:
        full[:, cls[0]] = 0.0
        full[:, cls[1]] = Z[:, 0]
    else:
        full[:, cls] = Z
    low = np.nanmin(full, axis=1, keepdims=True) - ABSENT
    full = np.where(np.isnan(full), low, full)
    return full - np.logaddexp.reduce(full, axis=1, keepdims=True)


def _loss(X, fit, t, K):
    z = _logits(X, fit, K)
    if K == 1:
        return float(np.sum(np.logaddexp(0.0, -z) * t + np.logaddexp(0.0, z) * (1 - t)))
    return float(-z[np.arange(t.size), t].sum())


def _reading(X, fit, K, e):
    """The probe's reading: its log-odds, or for letters the correct letter over the best other."""
    z = _logits(X, fit, K)
    if K == 1:
        return z
    other = np.where(np.arange(K)[None, :] == e[:, None], -np.inf, z).max(1)
    return z[np.arange(z.shape[0]), e] - other


def _splits(t, g, k, seed, strat=None):
    """Grouped folds, stratified (by ``strat``, else ``t``) where the groups allow; [] when there are too few pictures."""
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
    k = min(k, len(set(g.tolist())))
    if k < 2:
        return []
    s = t if strat is None else strat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(t, s, g))
        except ValueError:
            return list(GroupKFold(n_splits=k).split(t, t, g))


def _choose_C(X, t, g, K, seed, ks=(None,), strat=None):
    """The (``k``, ``C``) with the lowest pooled out-of-fold log-loss on a grouped split of the training part.

    With the default ``ks`` this is the choice of ``C`` alone, and the result is ``(None, C)``.
    """
    grid = [(k, C) for k in ks for C in C_GRID]
    sp = [(tr, te) for tr, te in _splits(t, g, INNER, seed + 1, strat)
          if len(set(t[tr].tolist())) >= 2]
    if not sp:
        return ks[-1], C_GRID[len(C_GRID) // 2]
    loss = np.zeros(len(grid))
    for tr, te in sp:
        for j, fit in enumerate(_fit_many(X[tr], t[tr], C_GRID, ks)):
            loss[j] += _loss(X[te], fit, t[te], K)
    # ties: the smaller C, then the fewer components (the grid is k-major, so a
    # small tiebreak on the position within each k and then on k's order)
    pen = np.array([1e-9 * (i % len(C_GRID)) + 1e-12 * (i // len(C_GRID)) for i in range(len(grid))])
    return grid[int(np.argmin(loss + pen))]


def probe_block(X, t, g, K=1, e=None, seed=0, ks=(None,), strat=None):
    """Out-of-fold readings for every record at one block, and the ``C`` each fold chose
    (with ``ks`` other than the default, ``(k, C)`` pairs).

    ``t`` is what the probe is fitted to: the 0/1 answer when ``K`` is 1, the
    index of the letter picked when ``K`` is 4, in which case ``e`` holds the
    index of each record's correct letter. ``strat`` (e.g. family x answer in a
    pooled fit) is what the folds are stratified by, where the groups allow.
    """
    oof = np.full(t.size, np.nan)
    Cs = []
    for tr, te in _splits(t, g, FOLDS, seed, strat):
        if len(set(t[tr].tolist())) < 2:
            # one answer in the whole training part: the reading is its prior
            if K == 1:
                p = (t[tr].sum() + 0.5) / (tr.size + 1.0)
                oof[te] = np.log(p / (1 - p))
            else:
                oof[te] = np.where(e[te] == t[tr][0], ABSENT, -ABSENT)
            Cs.append(None)
            continue
        k, C = _choose_C(X[tr], t[tr], g[tr], K, seed, ks,
                         None if strat is None else strat[tr])
        fit, = _fit_many(X[tr], t[tr], [C], (k,))
        oof[te] = _reading(X[te], fit, K, None if e is None else e[te])
        Cs.append(C if ks == (None,) else [k, C])
    return oof, Cs


# ------------------------------------------------------------------ the run

#: principal components the pooled probe may keep (None: all), chosen with C by the inner split
PCA_GRID = (16, 64, 256, None)


def pool_of(fam, kind):
    """The task a family belongs to: families sharing it share a target type and are fitted together.

    SimonSays families pool by question across sources and edits
    (``*_auth`` yes/no, ``*_eval`` test/real or yes/no by kind); WMDP families
    pool across subjects by condition (``wmdp_<subject>_sand`` -> ``wmdp_sand``).
    """
    bits = fam.split("_")
    if bits[0] == "wmdp":
        return f"wmdp_{'_'.join(bits[2:]) or bits[1]}|{kind}"
    return f"{bits[-1]}|{kind}"


def _pool_groups(recs):
    """Picture groups over a pool: records linked by their :func:`picture` or by a shared source photograph.

    An unedited photograph and every edit of it land in one group whichever
    family each came from, so no photograph is on both sides of a split.
    """
    par = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    keys = []
    for r in recs:
        m = r.get("meta") or {}
        img = str(r.get("image") or "")
        ds = img.split("/", 1)[0] if "/" in img else ""
        ks = [picture(r)]
        if m.get("source_image"):
            ks.append(f"photo:{ds}|{m['source_image']}")
        elif "/" in img:
            ks.append(f"photo:{ds}|{img.split('/', 1)[1]}")
        for k in ks[1:]:
            par[find(ks[0])] = find(k)
        keys.append(ks[0])
    return np.array([find(k) for k in keys])


def load_family(a, fam):
    """One family's labelled short-answer records and their stored states, or None."""
    pj = os.path.join(a.results, f"{a.model}__{fam}.json")
    pn = os.path.join(a.results, f"{a.model}__{fam}_states.npz")
    if not (os.path.exists(pj) and os.path.exists(pn)):
        return None
    rows = {r["id"]: r for r in json.load(open(pj))["rows"]}
    z = np.load(pn)
    ids = [str(i) for i in z["ids"]]
    n_layers = int(z["n_layers"]) if "n_layers" in z.files else None
    keep, y, t, e, g, kinds, recs, skipped = [], [], [], [], [], [], [], 0
    for j, rid in enumerate(ids):
        r = rows.get(rid)
        if r is None or r.get("kind") not in ("yesno", "test_real", "letter"):
            continue
        lab = own_answer(r)
        if lab is None:
            skipped += 1
            continue
        keep.append(j); y.append(lab); g.append(picture(r)); kinds.append(r["kind"]); recs.append(r)
        if r["kind"] == "letter":
            t.append(LETTERS.index(said_letter(r.get("reply")))); e.append(expected_letter(r))
        else:
            t.append(lab); e.append(-1)
    ANS = z["ANS"]
    blocks = [int(b) for b in z["layers"]] if "layers" in z.files else list(range(ANS.shape[1]))
    H = np.asarray(ANS[keep], np.float16) if keep else np.zeros((0, *ANS.shape[1:]), np.float16)
    del ANS, z
    y, t, e, g = (np.asarray(v) for v in (y, t, e, g))
    return dict(fam=fam, ids=[ids[j] for j in keep], y=y, t=t, e=e, g=g, kinds=kinds, recs=recs,
                skipped=skipped, blocks=blocks, n_layers=n_layers or len(blocks), H=H)


def _scorable(f):
    n_pos = int(f["y"].sum()) if f["y"].size else 0
    return f["y"].size >= MIN_N and min(n_pos, f["y"].size - n_pos) >= MIN_CLASS and len(set(f["g"].tolist())) >= 2


def _block_job(X, t, g, K, e, seed, ks, strat):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return probe_block(X, t, g, K=K, e=e, seed=seed, ks=ks, strat=strat)


def _all_blocks(H, t, g, K, e, seed, ks, strat, jobs):
    """``probe_block`` at every block, one process per block."""
    args = [(np.ascontiguousarray(H[:, bi], np.float32), t, g, K, e, seed, ks, strat) for bi in range(H.shape[1])]
    if jobs <= 1:
        return [probe_block(*x[:4], e=x[4], seed=x[5], ks=x[6], strat=x[7]) for x in args]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=jobs, backend="loky")(delayed(_block_job)(*x) for x in args)


def _write(a, f, S, au, Cs, protocol, n_pictures):
    y = f["y"]
    K = len(LETTERS) if all(k == "letter" for k in f["kinds"]) else 1
    payload = dict(
        model=a.model, family=f["fam"], blocks=f["blocks"], n_layers=f["n_layers"],
        readout="probe",
        label=dict(yesno="1 when the reply opens with yes, 0 with no",
                   test_real="1 when the reply opens with test, 0 with real",
                   letter="1 when the reply's letter is the correct one, 0 when it is another"),
        reading=("the probe's log-odds of the label" if K == 1 else
                 "a multinomial probe on the letter the reply opens with; the reading is its "
                 "log-probability of the correct letter minus that of the best other letter"),
        protocol=protocol,
        n=int(y.size), n_pos=int(y.sum()), n_unlabelled=f["skipped"], n_pictures=n_pictures,
        auroc=[None if not np.isfinite(v) else round(float(v), 4) for v in au], C=Cs,
        rows=[dict(id=rid, kind=k, y=int(v), probe=np.round(S[i].astype(np.float64), 3).tolist())
              for i, (rid, k, v) in enumerate(zip(f["ids"], f["kinds"], y))])
    p = os.path.join(a.out, f"{a.model}__{f['fam']}_{a.tag}.json")
    os.makedirs(a.out, exist_ok=True)
    json.dump(payload, open(p, "w"))
    mid = len(au) // 2
    print(f"  {f['fam']}: {y.size} records ({int(y.sum())} positive, {f['skipped']} unlabelled), "
          f"{n_pictures} pictures; AUROC {au[mid]:.2f} at block {f['blocks'][mid]}, "
          f"{au[-1]:.2f} at the last -> {p}", flush=True)


def run_family(a, fams):
    """The per-family probe: each family fitted on its own records (the earlier protocol, kept for transparency)."""
    from vlens.lens.metrics import auroc
    for fam in fams:
        t0 = time.time()
        f = load_family(a, fam)
        if f is None:
            continue
        y, t, e, g = f["y"], f["t"], f["e"], f["g"]
        if not _scorable(f):
            print(f"  {fam}: {y.size} labelled ({int(y.sum())} positive), {len(set(g.tolist()))} pictures; "
                  f"not probed")
            continue
        letters = all(k == "letter" for k in f["kinds"])
        K = len(LETTERS) if letters else 1
        got = _all_blocks(f["H"], t if letters else y, g, K, e if letters else None, a.seed,
                          a.ks, None, a.jobs)
        S = np.stack([o for o, _c in got], 1).astype(np.float32)
        au = [auroc(y, S[:, bi]) for bi in range(S.shape[1])]
        protocol = dict(mode="family",
                        model="L2 logistic regression on the standardised answer-spot state, one per block",
                        folds=int(min(FOLDS, len(set(g.tolist())))), inner_folds=INNER, C_grid=list(C_GRID),
                        pca_grid=[None if k is None else int(k) for k in a.ks],
                        choose_C="pooled out-of-fold log-loss of a grouped inner split of each training part",
                        grouping="by picture: the source photograph of an edit, else group_id, else image number",
                        seed=a.seed)
        _write(a, f, S, au, [c for _o, c in got], protocol, len(set(g.tolist())))
        print(f"    {time.time() - t0:.0f}s", flush=True)


def run_pooled(a, fams):
    """The pooled probe: every family of a task fitted together, folds grouped by picture across the pool,
    each family scored on its own held-out records."""
    from vlens.lens.metrics import auroc
    loaded = [f for f in (load_family(a, fam) for fam in fams) if f is not None and f["y"].size]
    pools = {}
    for f in loaded:
        kinds = set(f["kinds"])
        if len(kinds) != 1:
            print(f"  {f['fam']}: mixes {sorted(kinds)}; left out of the pooled probe")
            continue
        pools.setdefault((pool_of(f["fam"], kinds.pop()), tuple(f["blocks"])), []).append(f)
    for (key, blocks), mem in sorted(pools.items()):
        t0 = time.time()
        kind = key.split("|")[1]
        letters = kind == "letter"
        K = len(LETTERS) if letters else 1
        y = np.concatenate([f["y"] for f in mem])
        t = np.concatenate([f["t"] for f in mem])
        e = np.concatenate([f["e"] for f in mem])
        fid = np.concatenate([np.full(f["y"].size, i) for i, f in enumerate(mem)])
        g = _pool_groups([r for f in mem for r in f["recs"]])
        H = np.concatenate([f["H"] for f in mem])
        # stratified by family and answer, so every fold holds each family's answers in proportion
        strat = fid * (K + 1) + (t if letters else y)
        n_pic = len(set(g.tolist()))
        print(f"  pool {key}: {len(mem)} families, {y.size} records, {n_pic} pictures, "
              f"{len(blocks)} blocks: {', '.join(f['fam'] for f in mem)}", flush=True)
        if y.size < MIN_N or len(set((t if letters else y).tolist())) < 2 or n_pic < 2:
            print("    too few records or answers to fit; not probed")
            continue
        got = _all_blocks(H, t if letters else y, g, K, e if letters else None, a.seed, a.ks, strat, a.jobs)
        S = np.stack([o for o, _c in got], 1).astype(np.float32)
        Cs = [c for _o, c in got]
        protocol = dict(mode="pooled", pool=key.split("|")[0], kind=kind,
                        pool_families=[f["fam"] for f in mem], pool_n=int(y.size), pool_pictures=n_pic,
                        model="L2 logistic regression on the standardised answer-spot state, optionally on "
                              "its top-k principal components, one per block, fitted on the whole task's "
                              "training folds",
                        folds=int(min(FOLDS, n_pic)), inner_folds=INNER, C_grid=list(C_GRID),
                        pca_grid=[None if k is None else int(k) for k in a.ks],
                        choose_C="(k, C) by pooled out-of-fold log-loss of a grouped inner split of each "
                                 "training part; standardisation and PCA fitted on the training rows only",
                        grouping="by picture across the pool: the source photograph (an unedited photograph "
                                 "and its edits together), else group_id, else image number; folds "
                                 "stratified by family and answer",
                        scored="each family on its own held-out records", seed=a.seed)
        for i, f in enumerate(mem):
            if not _scorable(f):
                print(f"    {f['fam']}: {f['y'].size} labelled ({int(f['y'].sum())} positive); "
                      f"in the pool's training folds, not scored")
                continue
            sel = fid == i
            Sf = S[sel]
            au = [auroc(f["y"], Sf[:, bi]) for bi in range(Sf.shape[1])]
            _write(a, f, Sf, au, Cs, protocol, len(set(g[sel].tolist())))
        print(f"    pool {key}: {time.time() - t0:.0f}s", flush=True)


def run(a):
    fams = a.families or sorted(
        os.path.basename(p).split("__", 1)[1][: -len("_states.npz")]
        for p in glob.glob(os.path.join(a.results, f"{a.model}__*_states.npz")))
    fams = [f for f in fams if not LOCKED.search(f)]
    if a.mode == "family":
        run_family(a, fams)
    else:
        run_pooled(a, fams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", default=os.path.join(HERE, "results", "simonsays"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "readouts"))
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("pooled", "family"), default="pooled",
                    help="pooled: every family of a task fitted together (the default, and what the tables "
                         "read); family: each family on its own, the earlier protocol")
    ap.add_argument("--pca", nargs="*", default=None,
                    help="principal components the inner split may choose among ('none' for all); "
                         "default 16 64 256 none pooled, none per family")
    ap.add_argument("--tag", default=None,
                    help="output file suffix; default probe (pooled) or probe_family (family)")
    ap.add_argument("--threads", type=int, default=4, help="worker processes, one block each")
    a = ap.parse_args()
    if a.pca is None:
        a.ks = PCA_GRID if a.mode == "pooled" else (None,)
    else:
        a.ks = tuple(None if str(k).lower() == "none" else int(k) for k in a.pca)
    a.tag = a.tag or ("probe" if a.mode == "pooled" else "probe_family")
    a.jobs = max(1, a.threads)
    run(a)


if __name__ == "__main__":
    main()
