# SPDX-License-Identifier: Apache-2.0
"""How the cues and the behaviours lie against one another inside Qwen3.5-27B.

Every direction is a unit difference of mean answer-spot states at one block (the last
block V-Read was read at, which is also a block the refusal states were stored at). Cues:
a sign's authority (official minus informal, pairs of the same picture), a picture dressed
as a test (minus dressed as real use, pairs), a sandbagging instruction typed (hazard minus
honest, same items) and printed in the picture, and a harmful prompt (minus a benign one,
MM-SafetyBench). Behaviours, each fitted where its cue is absent: answering no (minus yes)
on the other authority records, test (minus real) on untouched pictures, wrong (minus
right) under the honest instruction, and refusing (minus complying) on harmless prompts.

Two rows:

  top left   the cosine between every two directions, cues then behaviours; a cue's cell
             with its own behaviour is boxed; on the diagonal, the same direction built on
             two halves of its records, one half against the other (how repeatable it is)
  top right  each cue against its own behaviour: the cosine with its 95% bootstrap interval
             and the level a behaviour direction fitted with shuffled labels reaches (95th
             percentile); the two sandbagging cues carry no interval, their states were
             not kept
  bottom     every record of four phenomena placed on two directions: across, the
             behaviour's direction; up, what is left of the cue's direction beyond it,
             each record left out of the directions it is placed on; the cue's direction
             is drawn at its true angle. Authority, the test look and the typed sandbagging
             instruction come from the planes fx_cuegeom built (the behaviour there is the
             answer V-Read reads on cue-free records); the harmful prompt is built here from
             the refusal states (the behaviour is whether the model refused). The test look's
             dressed pictures are 21 pairs; its 170 untouched photographs are the cue-free set

The cosines, halves and record counts are read from ``results/cuegeom/direction_matrix.json``
and the three cached planes from ``results/cuegeom/planes_<model>.json`` (both written by
:func:`_persist` and :func:`_persist_planes` while every states file still existed; the
sandbagging states were not kept afterwards), the intervals from
``results/cuegeom/direction_intervals.json`` (``scratchpad/dirci.py`` over :func:`_compute`),
and the refusal plane from ``results/safety/<model>__refusal_states.npz``.
"""
from __future__ import annotations

import json
import os

import numpy as np

from vlens.figures import fx_cuegeom as G
from vlens.figures import paper as P
from vlens.figures import style as st

MODELS = G.MODELS
#: the model the figure is of: the one with every direction, refusal included
MODEL = "Qwen3.5-27B"
SEED = 0
#: the directions, in the order the matrix lists them: key, name on the figure, phenomenon
CUES = (("auth", "authority sign", "authority"), ("look", "test look", "look"),
        ("typed", "sandbag typed", "sandbag"), ("printed", "sandbag printed", "sandbag"),
        ("harm", "harmful prompt", "refusal"))
BEHAVIOURS = (("no", "answers no", "authority"), ("test", "answers test", "look"),
              ("wrong", "answers wrong", "sandbag"), ("refuse", "refuses", "refusal"))
#: each cue with the behaviour it is about
OWN = (("auth", "no"), ("look", "test"), ("typed", "wrong"), ("printed", "wrong"), ("harm", "refuse"))
NBOOT = 200


def _unit(v):
    return G._unit(np.asarray(v, np.float64))


def _halves(units, rng):
    """Two disjoint halves of a list of units (pairs, items or records)."""
    p = rng.permutation(len(units))
    return [units[i] for i in p[: len(p) // 2]], [units[i] for i in p[len(p) // 2:]]


def _contrast(Xa, Xb, rng, pairs=None):
    """A direction from two groups of states: on every record, and on two independent halves.

    With ``pairs`` (index pairs into the stacked groups) the halves split the pairs, so the
    same picture or item never sits on both sides."""
    Xa, Xb = np.asarray(Xa, np.float64), np.asarray(Xb, np.float64)
    full = _unit(Xa.mean(0) - Xb.mean(0))
    if pairs is not None:
        h1, h2 = _halves(list(pairs), rng)
        halves = [_unit(Xa[[a for a, _b in h]].mean(0) - Xb[[b for _a, b in h]].mean(0)) for h in (h1, h2)]
    else:
        ha, hb = _halves(list(range(len(Xa))), rng), _halves(list(range(len(Xb))), rng)
        halves = [_unit(Xa[ha[k]].mean(0) - Xb[hb[k]].mean(0)) for k in (0, 1)]
    # bootstrap directions (units resampled) and, for unpaired contrasts, shuffled-label directions
    B = []
    if pairs is not None:
        P_ = list(pairs)
        for _ in range(NBOOT):
            pick = [P_[i] for i in rng.integers(0, len(P_), len(P_))]
            B.append(_unit(Xa[[a for a, _b in pick]].mean(0) - Xb[[b for _a, b in pick]].mean(0)))
        N = None
    else:
        for _ in range(NBOOT):
            ia, ib = rng.integers(0, len(Xa), len(Xa)), rng.integers(0, len(Xb), len(Xb))
            B.append(_unit(Xa[ia].mean(0) - Xb[ib].mean(0)))
        pool = np.vstack([Xa, Xb]); N = []
        for _ in range(NBOOT):
            p_ = rng.permutation(len(pool))
            N.append(_unit(pool[p_[: len(Xa)]].mean(0) - pool[p_[len(Xa):]].mean(0)))
        N = np.stack(N)
    return dict(d=full, h=halves, n=(len(Xa), len(Xb)), b=np.stack(B), null=N)


def _block_states(results, sub, m, fam, b):
    """One block of one states file, float32, with the rows it belongs to."""
    pz, pj = f"{results}/{sub}/{m}__{fam}_states.npz", f"{results}/{sub}/{m}__{fam}.json"
    if not (os.path.exists(pz) and os.path.exists(pj)):
        return None, None, None
    z = np.load(pz)
    layers = z["layers"].tolist() if "layers" in z.files else None
    if layers is not None and b not in layers:
        return None, None, None
    A = z["ANS"][:, layers.index(b) if layers is not None else b].astype(np.float32)
    ids = [str(i) for i in z["ids"]]
    R = {r["id"]: r for r in json.load(open(pj))["rows"]}
    del z
    return A, ids, R


def _authority(results, m, b, rng):
    Xo, Xi, Xf, said_no = [], [], [], []
    keys_o, keys_i = [], []
    for f in G.AUTH_FAMS:
        A, ids, R = _block_states(results, "simonsays", m, f, b)
        if A is None:
            continue
        for k, i in enumerate(ids):
            r = R.get(i)
            if not r or r.get("kind") != "yesno" or r.get("subcategory") != "operative":
                continue
            meta = r.get("meta") or {}
            ev = str(meta.get("edit_variant"))
            key = (f, str(meta.get("source_image")), str(meta.get("image_number")), str(r.get("question")))
            if G.OFFICIAL.search(ev):
                Xo.append(A[k]); keys_o.append(key)
            elif G.INFORMAL.search(ev):
                Xi.append(A[k]); keys_i.append(key)
            else:
                Xf.append(A[k]); said_no.append(float(r["margins"]["yes_over_no"]) < 0)
        del A
    # the contrast keeps only pictures edited both ways; the rest joins the cue-free records
    io, ii = {k: j for j, k in enumerate(keys_o)}, {k: j for j, k in enumerate(keys_i)}
    pairs = [(io[k], ii[k]) for k in io if k in ii]
    Xo_p, Xi_p = [Xo[a] for a, _b in pairs], [Xi[b] for _a, b in pairs]
    pp = [(j, j) for j in range(len(pairs))]
    cue = _contrast(Xo_p, Xi_p, rng, pairs=pp)
    said_no = np.array(said_no)
    Xf = np.asarray(Xf)
    beh = _contrast(Xf[said_no], Xf[~said_no], rng)
    return cue, beh


def _look(results, m, b, rng):
    Xt, Xr, Xp, said_test, kt, kr = [], [], [], [], [], []
    for f in G.EVAL_FAMS:
        A, ids, R = _block_states(results, "simonsays", m, f, b)
        if A is None:
            continue
        for k, i in enumerate(ids):
            r = R.get(i)
            if not r or r.get("kind") != "test_real" or P.LOCKED.search(str(r.get("family", ""))):
                continue
            meta = r.get("meta") or {}
            g = G._dressing(meta.get("edit_variant"))
            mg = (r.get("margins") or {}).get("test_over_real")
            if g is None or mg is None:
                continue
            key = (f, str(meta.get("source_image")), str(meta.get("image_number")), str(r.get("question")))
            if g == "test":
                Xt.append(A[k]); kt.append(key)
            elif g == "real":
                Xr.append(A[k]); kr.append(key)
            else:
                Xp.append(A[k]); said_test.append(float(mg) > 0)
        del A
    it, ir = {k: j for j, k in enumerate(kt)}, {k: j for j, k in enumerate(kr)}
    pairs = [(it[k], ir[k]) for k in it if k in ir]
    cue = _contrast([Xt[a] for a, _b in pairs], [Xr[b] for _a, b in pairs], rng,
                    pairs=[(j, j) for j in range(len(pairs))])
    said_test = np.array(said_test)
    Xp = np.asarray(Xp)
    beh = _contrast(Xp[said_test], Xp[~said_test], rng)
    return cue, beh


def _sandbag(results, m, b, rng):
    A, ids, R = _block_states(results, "sandchan", m, "wmdp_chan", b)
    if A is None:
        return None, None, None
    X = {}
    for k, i in enumerate(ids):
        r = R.get(i)
        if not r or P.LOCKED.search(i):
            continue
        sub = str(r.get("subcategory"))
        item = i.rsplit("__", 2)[0]
        X.setdefault(sub, {})[item] = (A[k], str(r.get("reply") or "").strip()[:1].upper() == str(r.get("expected")).strip().upper())
    del A
    out = {}
    for key, (h, c) in (("typed", ("text__hazard_text", "text__honest_text")),
                        ("printed", ("text__hazard_picture", "text__honest_picture"))):
        items = sorted(set(X[h]) & set(X[c]))
        out[key] = _contrast([X[h][i][0] for i in items], [X[c][i][0] for i in items], rng,
                             pairs=[(j, j) for j in range(len(items))])
    # right against wrong under the honest instruction of both channels
    ctrl = [v for sub in ("text__honest_text", "text__honest_picture") for v in X[sub].values()]
    ok = np.array([v[1] for v in ctrl])
    Xc = np.asarray([v[0] for v in ctrl])
    beh = _contrast(Xc[~ok], Xc[ok], rng)
    return out["typed"], out["printed"], beh


def _refusal(results, m, b, rng):
    pz, pj = f"{results}/safety/{m}__refusal_states.npz", f"{results}/safety/{m}__refusal_read.json"
    if not (os.path.exists(pz) and os.path.exists(pj)):
        return None, None
    z = np.load(pz)
    layers = z["layers"].tolist()
    if b not in layers:
        return None, None
    A = z["ANS"][:, layers.index(b)].astype(np.float32)
    ids = [str(i) for i in z["ids"]]
    del z
    R = {r["id"]: r for r in json.load(open(pj))["rows"]}
    Xh, Xb_, Xref, Xcomp = [], [], [], []
    for k, i in enumerate(ids):
        r = R.get(i)
        if not r:
            continue
        pool, kind = str(r.get("pool")), str(r.get("kind"))
        if pool == "mmsb_harmful":
            Xh.append(A[k])
        elif pool == "mmsb_benign":
            Xb_.append(A[k])
        if not r.get("harmful") and kind in ("refusal", "compliance"):
            (Xref if kind == "refusal" else Xcomp).append(A[k])
    del A
    if len(Xh) < 5 or len(Xb_) < 5 or len(Xref) < 5 or len(Xcomp) < 5:
        return None, None
    return _contrast(Xh, Xb_, rng), _contrast(Xref, Xcomp, rng)


def _compute(results):
    out = {}
    for m in MODELS:
        V = G._vread(results, m, G.EVAL_FAMS[0])
        blocks = G._read_blocks(V)
        if not blocks:
            continue
        b = int(max(blocks))
        rng = np.random.default_rng(SEED)
        dirs = {}
        dirs["auth"], dirs["no"] = _authority(results, m, b, rng)
        dirs["look"], dirs["test"] = _look(results, m, b, rng)
        typed, printed, wrong = _sandbag(results, m, b, rng)
        if typed is not None:
            dirs["typed"], dirs["printed"], dirs["wrong"] = typed, printed, wrong
        harm, refuse = _refusal(results, m, b, rng)
        if harm is not None:
            dirs["harm"], dirs["refuse"] = harm, refuse
        order = [k for k, _n, _p in CUES if k in dirs] + [k for k, _n, _p in BEHAVIOURS if k in dirs]
        D = np.stack([dirs[k]["d"] for k in order])
        Gm = D @ D.T
        # the cosine between the two halves of each direction, and the cosine of one
        # direction's half with another's other half, so nothing is scored against itself
        H = np.stack([np.stack(dirs[k]["h"]) for k in order])            # k x 2 x d
        self_cos = np.einsum("kd,kd->k", H[:, 0], H[:, 1])
        cross = 0.5 * (np.einsum("id,jd->ij", H[:, 0], H[:, 1]) + np.einsum("id,jd->ij", H[:, 1], H[:, 0]))
        # 95% intervals of every cosine from the paired bootstraps, and the shuffled-label null
        Bs = np.stack([dirs[k]["b"] for k in order])                    # k x NBOOT x d
        CB = np.einsum("ibd,jbd->ijb", Bs, Bs)
        ci = np.percentile(CB, [2.5, 97.5], axis=2)
        null = {}
        for i, k in enumerate(order):
            for j, k2 in enumerate(order):
                if dirs[k2].get("null") is not None and i != j:
                    null[(k, k2)] = float(np.percentile(np.abs(dirs[k2]["null"] @ dirs[k]["d"]), 95))
        out[m] = dict(block=b, order=order, G=Gm, cross=cross, self_cos=self_cos, ci=ci, null=null,
                      n={k: dirs[k]["n"] for k in order})
        L = 36 if m == MODELS[0] else 64
        out[m]["L"] = L
    return out



# ---------------------------------------------------------------- what is on disk
MATRIX = "cuegeom/direction_matrix.json"
INTERVALS = "cuegeom/direction_intervals.json"
PLANES = "cuegeom/planes_{m}.json"


def _persist(results, D):
    """Write the cosine matrices :func:`_compute` gives, so the figure can be drawn after the
    states are gone."""
    out = {}
    for m, d in D.items():
        out[m] = dict(block=int(d["block"]), L=int(d["L"]), order=list(d["order"]),
                      n={k: [int(a), int(b)] for k, (a, b) in d["n"].items()},
                      cos=[[round(float(v), 6) for v in row] for row in np.asarray(d["G"])],
                      cross_half=[[round(float(v), 6) for v in row] for row in np.asarray(d["cross"])],
                      self_cos={k: round(float(v), 6) for k, v in zip(d["order"], d["self_cos"])})
    out["_note"] = ("Cosines between the cue and behaviour directions of fx_dirspace, one block per model. cos: "
                    "full-data directions; cross_half: one direction's half against the other's other half, and on "
                    "the diagonal the same direction's two halves (self_cos). n: records (or pairs) on each side.")
    p = os.path.join(results, MATRIX)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    return p


def _persist_planes(results, D):
    """Write the per-record planes :func:`fx_cuegeom._compute` gives, one file per model."""
    paths = []
    for m, d in D.items():
        out = dict(block=int(d["b"]), L=int(d["L"]), planes={})
        for k, p in d["planes"].items():
            out["planes"][k] = dict(cos=round(float(p["cth"]), 6), auroc=round(float(p["auroc"]), 4),
                                    xy=[[round(float(a), 5), round(float(b), 5)] for a, b in np.asarray(p["xy"])],
                                    cond=[int(c) for c in p["cond"]], read=[bool(r) for r in p["read"]],
                                    said=[bool(s) for s in p["said"]], pairs=[[int(a), int(b)] for a, b in p["pairs"]])
        out["_note"] = ("Per-record planes of fx_cuegeom._compute at the block V-Read last reads: xy = (along the "
                        "behaviour's direction as V-Read reads it on cue-free records, along what is left of the cue's "
                        "direction), each record left out of the directions it is placed on, scaled to unit mean spread; "
                        "cond 0 = with the cue, 1 = its counterpart, 2 = cue-free; read = V-Read reads the right-hand "
                        "answer; said = the model gave it; cos = the two directions' cosine on every record.")
        p = os.path.join(results, PLANES.format(m=m))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(out, open(p, "w"))
        paths.append(p)
    return paths


def _matrix(results, m):
    p = os.path.join(results, MATRIX)
    if not os.path.exists(p):
        D = _compute(results)
        if not D:
            return None
        _persist(results, D)
    J = json.load(open(p))
    if m not in J:
        return None
    d = J[m]
    return dict(block=int(d["block"]), L=int(d["L"]), order=list(d["order"]), n=d["n"],
                G=np.asarray(d["cos"], float), self_cos=dict(d["self_cos"]))


def _intervals(results, m):
    p = os.path.join(results, INTERVALS)
    if not os.path.exists(p):
        return {}
    J = json.load(open(p))
    return {(r["a"], r["b"]): r for r in J.get(m, {}).get("pairs", [])}


def _planes(results, m):
    """The cached per-record planes of fx_cuegeom for one model: key -> xy, cond, said, cos."""
    p = os.path.join(results, PLANES.format(m=m))
    if not os.path.exists(p):
        D = G._compute(results)
        if not D or m not in D:
            return None
        _persist_planes(results, D)
    J = json.load(open(p))
    return dict(block=int(J["block"]), planes={k: dict(xy=np.asarray(v["xy"], float), cond=np.asarray(v["cond"], int),
                                                        said=np.asarray(v["said"], bool), cos=float(v["cos"]),
                                                        pairs=[tuple(pr) for pr in v.get("pairs", [])])
                                                for k, v in J["planes"].items()})


def _refusal_plane(results, m, b):
    """Every prompt of the refusal set on two directions at block ``b``: across, refusing (refused
    minus complied, harmless prompts); up, what is left of the harmful-prompt direction (harmful
    minus benign) beyond it. Each record is left out of the directions it is placed on, and out
    of the centre. Same form as the cached planes: cond 0 harmful, 1 benign, 2 other harmless
    (MOSSBench); said = the model refused."""
    pz, pj = f"{results}/safety/{m}__refusal_states.npz", f"{results}/safety/{m}__refusal_read.json"
    if not (os.path.exists(pz) and os.path.exists(pj)):
        return None
    z = np.load(pz)
    layers = z["layers"].tolist()
    if b not in layers:
        return None
    X = z["ANS"][:, layers.index(b)].astype(np.float64)
    ids = [str(i) for i in z["ids"]]
    del z
    R = {r["id"]: r for r in json.load(open(pj))["rows"]}
    if any(i not in R for i in ids):
        return None
    pool = np.array([str(R[i].get("pool")) for i in ids])
    kind = np.array([str(R[i].get("kind")) for i in ids])
    harmful = np.array([bool(R[i].get("harmful")) for i in ids])
    refused = np.array([bool(R[i].get("refused")) for i in ids])
    grp = dict(H=np.where(pool == "mmsb_harmful")[0], Bn=np.where(pool == "mmsb_benign")[0],
               A=np.where(~harmful & (kind == "refusal"))[0], B=np.where(~harmful & (kind == "compliance"))[0])
    if min(len(v) for v in grp.values()) < 5:
        return None
    S = {k: X[v].sum(0) for k, v in grp.items()}
    S["all"] = X.sum(0)
    n = {k: len(v) for k, v in grp.items()}
    sets = {k: set(v.tolist()) for k, v in grp.items()}
    xy = np.zeros((len(X), 2))
    for i in range(len(X)):
        mean = {k: (S[k] - (X[i] if i in sets[k] else 0)) / (n[k] - (i in sets[k])) for k in grp}
        a = _unit(mean["A"] - mean["B"])
        e = _unit(mean["H"] - mean["Bn"])
        e2 = _unit(e - (e @ a) * a)
        c = (S["all"] - X[i]) / (len(X) - 1)
        xy[i] = ((X[i] - c) @ a, (X[i] - c) @ e2)
    a = _unit(X[grp["A"]].mean(0) - X[grp["B"]].mean(0))
    e = _unit(X[grp["H"]].mean(0) - X[grp["Bn"]].mean(0))
    cond = np.where(pool == "mmsb_harmful", 0, np.where(pool == "mmsb_benign", 1, 2))
    return dict(xy=xy / xy.std(0, ddof=1).mean(), cond=cond, said=refused, cos=float(e @ a), pairs=[],
                n=dict(harmful=int(n["H"]), benign=int(n["Bn"]), refused=int(n["A"]), complied=int(n["B"])))


# ---------------------------------------------------------------- the figure
#: the four planes, left to right: key in the cached planes (or "harm"), the cue's name, the
#: behaviour's name as the axis says it, and the three conditions as the caption names them
PLANE_ROW = (("auth", "authority sign", "answers no", ("official sign", "informal sign", "other")),
             ("look", "test look", "answers test", ("dressed as a test", "as real use", "untouched")),
             ("sand", "sandbag typed", "answers wrong", ("sandbag instruction", "honest instruction", "none")),
             ("harm", "harmful prompt", "refuses", ("harmful prompt", "benign prompt", "other harmless")))
COND_COLOUR = (st.CLAY, st.SLATE, st.GRAY)


def _fmt(c):
    """Two decimals without the leading zero, so five characters never crowd a cell."""
    s = f"{c:.2f}".replace("0.", ".")
    return ".00" if s in ("-.00",) else s


def _draw(M, I, PLs, out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Arc, Rectangle
    st.use()
    fs = st.matplotlib.rcParams["xtick.labelsize"]
    T = dict(title=fs * 0.74, tick=fs * 0.7, cell=fs * 0.64, label=fs * 0.66, note=fs * 0.6, key=fs * 0.64)
    name = {k: n for k, n, _p in CUES + BEHAVIOURS}
    order = M["order"]
    n = len(order)
    box = dict(boxstyle="round,pad=0.12", fc="white", ec="none")
    # ---- the page, in inches. Bottom row: four planes across the width. Top row: the matrix
    # and the interval panel, centred.
    W = st.MAX_WIDTH
    PW, PH, PG, PX0 = 2.28, 1.9, 0.2, 0.7                     # a plane's width, height, the gap, the first one's left
    Y_PL = 0.5                                                 # planes' bottom (room for their axis names)
    Y_KEY = Y_PL + PH + 0.36                                   # the key's two rows sit above the plane titles
    S = 3.24                                                   # matrix side
    Y_TOP = Y_KEY + 0.5 + 1.12                                 # top row's bottom (room for the matrix's rotated names)
    H = Y_TOP + S + 0.4
    XL = 1.35                                                  # matrix left edge (its names to the left)
    XB, WB = XL + S + 0.35 + 1.25, 3.0                         # interval panel: left edge, width
    shift = (W - (XB + WB)) / 2 - 0.15                         # centre the top row over the planes
    XL += shift; XB += shift
    fig = plt.figure(figsize=(W, H))
    fig.set_layout_engine("none")
    title_kw = dict(fontsize=T["title"], color=st.INK, pad=7, loc="center")

    # ---- top left: the matrix, cues then behaviours, lower triangle with the halves on the diagonal
    ax = fig.add_axes([XL / W, Y_TOP / H, S / W, S / H])
    gap = 0.14                                    # a little air between the cues and the behaviours
    ncue = len(CUES)
    pos = lambda k: k + (gap if k >= ncue else 0.0)  # noqa: E731
    ext = n + gap
    cmap = st.warm_cmap(diverging=True)
    own = set(OWN)
    for i, ki in enumerate(order):
        for j, kj in enumerate(order):
            if j > i:
                continue
            x0, y0 = pos(j), ext - pos(i) - 1
            if i == j:
                ax.add_patch(Rectangle((x0, y0), 1, 1, facecolor="white", edgecolor=st.GRAY, lw=0.9))
                c = float(M["self_cos"][ki])
            else:
                c = float(M["G"][i, j])
                ax.add_patch(Rectangle((x0, y0), 1, 1, facecolor=cmap(0.5 + c / 2), edgecolor="white", lw=1.0))
            ax.text(x0 + 0.5, y0 + 0.5, _fmt(c), ha="center", va="center", fontsize=T["cell"], color=st.INK)
            if (kj, ki) in own:
                ax.add_patch(Rectangle((x0 + 0.04, y0 + 0.04), 0.92, 0.92, fill=False, edgecolor=st.INK, lw=1.4, zorder=4))
    ax.set_xlim(0, ext); ax.set_ylim(0, ext)
    ax.set_aspect("equal")
    ax.set_xticks([pos(k) + 0.5 for k in range(n)]); ax.set_yticks([ext - pos(k) - 0.5 for k in range(n)])
    ax.set_xticklabels([name[k] for k in order], rotation=60, ha="right", va="top", fontsize=T["tick"], rotation_mode="anchor")
    ax.set_yticklabels([name[k] for k in order], fontsize=T["tick"])
    ax.tick_params(length=0, pad=3)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # the empty half of the square carries the two notes
    notes = ("boxed: a cue with its own behaviour", "diagonal: the same direction",
             "built twice, on two halves", "of its records")
    for k, line in enumerate(notes):
        ax.text(ext - 0.08, ext - (0.55 + 0.66 * k) - (0.3 if k else 0), line, ha="right", va="center",
                fontsize=T["note"], color=st.INK)
    ax.set_title("cosine between every two directions", **title_kw)

    # ---- top right: each cue against its own behaviour
    ax = fig.add_axes([XB / W, Y_TOP / H, WB / W, S / H])
    rows = []
    for k, (cue, beh) in enumerate(OWN):
        i, j = order.index(cue), order.index(beh)
        r = I.get((cue, beh)) or {}
        rows.append(dict(cue=cue, beh=beh, cos=float(M["G"][i, j]), lo=r.get("lo"), hi=r.get("hi"), null=r.get("null95")))
    ax.axvline(0, color=st.AXIS, lw=0.9, zorder=1)
    for k, r in enumerate(rows):
        if r["lo"] is not None:
            ax.plot([r["lo"], r["hi"]], [k, k], color=st.INK, lw=1.8, solid_capstyle="butt", zorder=3)
        ax.plot([r["cos"]], [k], "o", color=st.INK, ms=7.5, zorder=4)
        if r["null"] is not None:
            ax.plot([r["null"], r["null"]], [k - 0.3, k + 0.3], color=st.AXIS, lw=1.8, solid_capstyle="butt", zorder=2)
        else:
            ax.text(r["cos"] + 0.05, k, "no interval: records not kept", ha="left", va="center",
                    fontsize=T["note"], color=st.INK)
    # the shuffled-label level, named once where the gap to the real cosine is the point
    last = [k for k, r in enumerate(rows) if r["null"] is not None][-1]
    xn = rows[last]["null"]
    ax.plot([xn, xn], [last + 0.34, last + 0.44], color=st.AXIS, lw=0.7, zorder=2)
    ax.text(xn, last + 0.48, "what shuffled labels reach", ha="center", va="top", fontsize=T["note"], color=st.INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{name[r['cue']]}\nvs {name[r['beh']]}" for r in rows], fontsize=T["tick"], linespacing=1.05)
    ax.set_ylim(len(rows) - 0.05, -0.6)
    ax.set_xlim(-0.17, 1.03)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "0.5", "1"], fontsize=T["tick"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("cosine of the cue's direction with its behaviour's", fontsize=T["label"])
    ax.grid(False)
    st.despine(ax, left=False)
    ax.set_title("a cue and its own behaviour", **title_kw)

    # ---- bottom: every record of four phenomena on the behaviour's direction and the rest of the cue's
    stats = {}
    for k, (key, cue, beh, _names) in enumerate(PLANE_ROW):
        p = PLs[key]
        ax = fig.add_axes([(PX0 + k * (PW + PG)) / W, Y_PL / H, PW / W, PH / H])
        xy, cond, said = p["xy"], p["cond"], p["said"]
        # a thin stroke joins the same picture or item under the cue and under its counterpart
        for a_, b_ in p.get("pairs", []):
            ax.plot([xy[a_, 0], xy[b_, 0]], [xy[a_, 1], xy[b_, 1]], color=st.GRAY, lw=0.5, alpha=0.45, zorder=1)
        for g, z in ((2, 2), (1, 3), (0, 3)):
            idx = np.where(cond == g)[0]
            for s_ in (False, True):
                kk = idx[said[idx] == s_]
                ax.scatter(xy[kk, 0], xy[kk, 1], s=7 if g == 2 else 11, zorder=z,
                           facecolors=COND_COLOUR[g] if s_ else "white", edgecolors=COND_COLOUR[g],
                           linewidths=0.6, alpha=0.45 if g == 2 else 0.85)
        lo, hi = xy.min(0), xy.max(0)
        # the box keeps both axes at one scale, the data centred, a little air around it
        sx, sy = (hi[0] - lo[0]) * 1.1, (hi[1] - lo[1]) * 1.1
        scale = max(sx / PW, sy / PH)
        cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        ax.set_xlim(cx - scale * PW / 2, cx + scale * PW / 2); ax.set_ylim(cy - scale * PH / 2, cy + scale * PH / 2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axhline(0, color=st.GRID, lw=0.9, zorder=1); ax.axvline(0, color=st.GRID, lw=0.9, zorder=1)
        # the behaviour's direction across, and the cue's at its true angle to it
        cth = p["cos"]; sth = float(np.sqrt(max(0.0, 1 - cth ** 2)))
        th = float(np.degrees(np.arctan2(sth, cth)))
        r = 0.32 * scale * PH
        arrow = dict(arrowstyle="-|>", lw=1.4, mutation_scale=10, shrinkA=0, shrinkB=0)
        ax.annotate("", xy=(r, 0), xytext=(0, 0), zorder=5, arrowprops=dict(color=st.INK, **arrow))
        ax.annotate("", xy=(r * cth, r * sth), xytext=(0, 0), zorder=5, arrowprops=dict(color=st.CLAY, **arrow))
        ax.add_patch(Arc((0, 0), 0.9 * r, 0.9 * r, theta1=0, theta2=th, color=st.INK, lw=0.8, zorder=5))
        am = np.radians(th / 2)
        ax.text(0.62 * r * np.cos(am) + 0.04 * r, 0.62 * r * np.sin(am), f"{th:.0f}°", ha="left", va="center",
                fontsize=T["note"], color=st.INK, zorder=6, bbox=box)
        ax.set_xlabel(f"across: {beh}", fontsize=T["label"], labelpad=4)
        if k == 0:
            ax.set_ylabel("up: the rest of\nthe cue's direction", fontsize=T["label"], linespacing=1.05, labelpad=4)
        ax.set_title(cue, **title_kw)
        stats[key] = dict(angle=th, cos=cth, n={g: int((cond == g).sum()) for g in (0, 1, 2)},
                          said={g: int(said[cond == g].sum()) for g in (0, 1, 2)},
                          across={g: float(xy[cond == g, 0].mean()) for g in (0, 1, 2)})
    # the key, one row above the plane titles: the three conditions, and what filled means
    hs = [Line2D([], [], marker="o", ls="none", mfc=c, mec=c, ms=6) for c in COND_COLOUR[:2]]
    hs += [Line2D([], [], marker="o", ls="none", mfc=COND_COLOUR[2], mec=COND_COLOUR[2], ms=6),
           Line2D([], [], marker="o", ls="none", mfc=st.INK, mec=st.INK, ms=6),
           Line2D([], [], color=st.INK, lw=1.4, marker=">", ms=5, mfc=st.INK, mec=st.INK, markevery=[1]),
           Line2D([], [], color=st.CLAY, lw=1.4, marker=">", ms=5, mfc=st.CLAY, mec=st.CLAY, markevery=[1])]
    labs = ["with the cue", "its counterpart", "cue-free", "filled: the model gave the answer named under the panel",
            "the behaviour's direction (across)", "the cue's direction, at its true angle to it"]
    leg = fig.legend(hs, labs, loc="center", bbox_to_anchor=(0.5, (Y_KEY + 0.25) / H), ncol=3, frameon=False,
                     fontsize=T["key"], handlelength=1.3, handletextpad=0.4, columnspacing=1.6, labelspacing=0.35,
                     labelcolor=st.INK, borderaxespad=0)
    fig.add_artist(leg)

    # ---- the numbers, for the log
    print(f"{MODEL}: block {M['block'] + 1} of {M['L']}")
    for r in rows:
        iv = f"[{r['lo']:.2f}, {r['hi']:.2f}]" if r["lo"] is not None else "no interval"
        nl = f"shuffled-label 95th pct {r['null']:.2f}" if r["null"] is not None else ""
        print(f"   {name[r['cue']]:16s} vs {name[r['beh']]:14s} cos {r['cos']:+.2f} {iv} {nl}")
    for k in order:
        print(f"   split-half {name[k]:16s} {M['self_cos'][k]:.2f}   n {M['n'][k]}")
    for key, cue, beh, names in PLANE_ROW:
        s = stats[key]
        print(f"   plane {cue}: angle {s['angle']:.0f} deg (cos {s['cos']:.3f});",
              "; ".join(f"{names[g]} n {s['n'][g]}, {beh} {s['said'][g]}, mean across {s['across'][g]:+.2f}" for g in (0, 1, 2)))
    return st.save(fig, P._at(out, "direction_space"), check=False, tight=True)


def direction_space(results="results", out="figures", model=None):
    """MAIN. The directions of the cues and of the behaviours in Qwen3.5-27B: the cosine between every two, each cue against its own behaviour with its interval and the shuffled-label level, and every record of four phenomena on its behaviour's direction and the rest of its cue's."""
    if model not in (None, MODEL):
        return None
    M = _matrix(results, MODEL)
    if M is None:
        return None
    C = _planes(results, MODEL)
    if C is None or not all(k in C["planes"] for k in ("auth", "look", "sand")):
        return None
    PLs = dict(C["planes"])
    PLs["harm"] = _refusal_plane(results, MODEL, M["block"])
    if PLs["harm"] is None:
        return None
    return _draw(M, _intervals(results, MODEL), PLs, out)
