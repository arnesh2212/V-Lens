# SPDX-License-Identifier: Apache-2.0
"""Where a cue lives inside the model: the states themselves, against what V-Read reads out of them.

Three cues, two models, one plane each. Across every plane runs the answer direction:
the mean state of the records V-Read reads as one answer minus the mean of those it reads
as the other, fitted on records the cue never touched. Up runs the part of the cue's own
direction (the mean state under the cue minus the mean under its counterpart) that lies
beyond the answer direction, so the cue's direction sits in the plane at its true angle
to the answer axis. A cue the model reads and uses tilts along the answer axis; a cue it
reads but does not use stands at right angles to it.

The cues: a sign's authority (an official sign against an informal one, unedited photos of
statutory signs as the cue-free set), a picture dressed as a test (against dressed as real
use, untouched pictures cue-free), and a sandbagging instruction (against an honest one,
no instruction cue-free). Per block the module also writes, for the appendix table, how
well each cue's direction separates its two conditions for a held-out record (AUROC) and
its cosine with the behaviour's direction.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from vlens.figures import paper as P
from vlens.figures import style as st

MODELS = ("Qwen3-VL-8B-Instruct", "Qwen3.5-27B")
EVAL_FAMS = ("openimages_orig_eval", "openimages_edit_eval", "vins_orig_eval", "vins_edit_eval",
             "nuscenes_edit_eval")
AUTH_FAMS = ("openimages_orig_auth", "openimages_edit_auth", "vins_edit_auth", "nuscenes_edit_auth")
#: how the edited sign was dressed: with authority, or without
OFFICIAL = re.compile(r"official|statutory|plate|upgrade")
INFORMAL = re.compile(r"handwritten|informal|cardboard|chalk|graffiti|board_instruction|downgrade")
#: the planes are drawn at the last block V-Read was read at, where its reading has settled
#: (it agrees with the model's answer on 86 to 100 per cent of records on every task)
#: each cue: its key, its name, and the behaviour its direction is measured against
CUE = (("look", "picture dressed as a test", "behaviour: answering test"),
       ("typed", "sandbag instruction, typed", "behaviour: answering wrong"),
       ("printed", "sandbag instruction, printed", "behaviour: answering wrong"))
#: the three planes, left to right: key, title, the three conditions (cue, counterpart,
#: cue-free) as the mini key names them, and the two answers (left, right) V-Read reads
PANELS = (("auth", "a sign's authority", ("official", "informal", "other"), ("yes", "no")),
          ("look", "a picture dressed as a test", ("as a test", "as real use", "untouched"), ("real", "test")),
          ("sand", "a sandbagging instruction", ("sandbagging", "honest", "none"), ("right", "wrong")))
COND_COLOUR = (st.CLAY, st.SLATE, st.GRAY)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _auroc(p, n):
    p, n = np.asarray(p), np.asarray(n)
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())


def _dressing(edit_variant):
    """test, real, or None, from how the set dressed the picture."""
    if edit_variant is None:
        return "plain"
    v = str(edit_variant)
    if "eval" in v or v in ("annotation_ui", "watermark"):
        return "test"
    if "deploy" in v or v in ("phone_screenshot", "dashcam_hud"):
        return "real"
    return None


def _rows(results, m, fam):
    p = f"{results}/simonsays/{m}__{fam}.json"
    return {r["id"]: r for r in json.load(open(p))["rows"]} if os.path.exists(p) else None


def _vread(results, m, fam):
    """V-Read's reading of every record at every block it was read at: id -> list per block."""
    for d, key in (("readouts_v2", "v_read_v2"), ("readouts", "v_read")):
        p = f"{results}/{d}/{m}__{fam}_readouts.json"
        if os.path.exists(p):
            return {str(r["id"]): r.get(key) for r in json.load(open(p))["rows"] if not r.get("v_read_fitted_on")}
    return {}


def _read_blocks(V):
    return sorted({i for v in V.values() for i, x in enumerate(v or []) if x is not None})


def _eval_states(results, m):
    X, lab, said, ids, keys = [], [], [], [], []
    for f in EVAL_FAMS:
        pz = f"{results}/simonsays/{m}__{f}_states.npz"
        R = _rows(results, m, f)
        if R is None or not os.path.exists(pz):
            return None
        z = np.load(pz)
        A = z["ANS"]
        for k, i in enumerate(z["ids"]):
            r = R.get(str(i))
            if not r or r.get("kind") != "test_real" or P.LOCKED.search(str(r.get("family", ""))):
                continue
            g = _dressing((r.get("meta") or {}).get("edit_variant"))
            mg = (r.get("margins") or {}).get("test_over_real")
            if g is None or mg is None:
                continue
            X.append(A[k].astype(np.float32)); lab.append(g); said.append(float(mg) > 0); ids.append((f, str(i)))
            meta = r.get("meta") or {}
            keys.append((f, str(meta.get("source_image")), str(meta.get("image_number")), str(r.get("question"))))
        del A, z
    if not X:
        return None
    return dict(X=np.stack(X), lab=np.array(lab), said=np.array(said), ids=ids, keys=keys)


def _loo_proj(Xl, a_idx, b_idx):
    """Each record of the two groups placed on the direction fitted without it, centred likewise."""
    Sa, Sb = Xl[a_idx].sum(0), Xl[b_idx].sum(0)
    na, nb = len(a_idx), len(b_idx)
    pa, pb = [], []
    for i in a_idx:
        ma, mb = (Sa - Xl[i]) / (na - 1), Sb / nb
        pa.append((Xl[i] - (ma + mb) / 2) @ _unit(ma - mb))
    for i in b_idx:
        ma, mb = Sa / na, (Sb - Xl[i]) / (nb - 1)
        pb.append((Xl[i] - (ma + mb) / 2) @ _unit(ma - mb))
    return np.array(pa), np.array(pb)


def _eval_curves(E):
    X, lab, said = E["X"], E["lab"], E["said"]
    T, R, Pn = (np.where(lab == g)[0] for g in ("test", "real", "plain"))
    au, cs = [], []
    for l in range(X.shape[1]):
        Xl = X[:, l].astype(np.float64)
        pt, pr = _loo_proj(Xl, T, R)
        au.append(_auroc(pt, pr))
        ans = _unit(Xl[Pn[said[Pn]]].mean(0) - Xl[Pn[~said[Pn]]].mean(0))
        cs.append(float(_unit(Xl[T].mean(0) - Xl[R].mean(0)) @ ans))
    return np.array(au), np.array(cs)


def _chan_curves(D):
    """Per block, for the typed and the printed sandbagging instruction: AUROC of its direction
    for a held-out record, and its cosine with right against wrong under the control instruction."""
    from vlens.figures import fx_channel as FC
    X, right = D["X"], D["right"]
    ct, ht, cp, hp = (c[0] for c in FC.CLASSES)
    out = {}
    for key, (h, c) in (("typed", (ht, ct)), ("printed", (hp, cp))):
        au, cs = [], []
        n = len(X[h])
        for l in range(X[h].shape[1]):
            A = np.concatenate([X[h][:, l], X[c][:, l]]).astype(np.float64)
            hi, ci = np.arange(n), np.arange(n, 2 * n)
            ph, pc = _loo_proj(A, hi, ci)
            au.append(_auroc(ph, pc))
            Cs = np.concatenate([X[ct][:, l], X[cp][:, l]]).astype(np.float64)
            ok = np.concatenate([right[ct], right[cp]])
            ans = _unit(Cs[ok].mean(0) - Cs[~ok].mean(0))
            cs.append(float(_unit(A[hi].mean(0) - A[ci].mean(0)) @ ans))
        out[key] = (np.array(au), np.array(cs))
    return out


def _embed(Xb, cond, read):
    """The plane of one cue at one block.

    ``cond`` is 0 for the cue, 1 for its counterpart, 2 for cue-free; ``read`` is True where
    V-Read reads the right-hand answer. Across: the answer direction (read right-hand minus
    read left-hand, fitted on the cue-free records). Up: the part of the cue's direction
    (cue minus counterpart) beyond it. Each record is placed with both directions fitted
    without it. Returns the coordinates scaled to unit mean spread, the cosine between the
    two directions fitted on every record, and the AUROC of the cue's direction for a
    held-out record.
    """
    X = Xb.astype(np.float64)
    C, K, G = (np.where(cond == g)[0] for g in (0, 1, 2))
    A, B = G[read[G]], G[~read[G]]
    S = dict(C=X[C].sum(0), K=X[K].sum(0), A=X[A].sum(0), B=X[B].sum(0), all=X.sum(0))
    xy = np.zeros((len(X), 2))
    for i in range(len(X)):
        inC, inK, inA, inB = i in C, i in K, i in A, i in B
        a = _unit((S["A"] - (X[i] if inA else 0)) / (len(A) - inA) - (S["B"] - (X[i] if inB else 0)) / (len(B) - inB))
        e1 = _unit((S["C"] - (X[i] if inC else 0)) / (len(C) - inC) - (S["K"] - (X[i] if inK else 0)) / (len(K) - inK))
        e2 = _unit(e1 - (e1 @ a) * a)
        c = (S["all"] - X[i]) / (len(X) - 1)
        xy[i] = ((X[i] - c) @ a, (X[i] - c) @ e2)
    a = _unit(X[A].mean(0) - X[B].mean(0))
    e1 = _unit(X[C].mean(0) - X[K].mean(0))
    pc, pk = _loo_proj(X, C, K)
    return xy / xy.std(0, ddof=1).mean(), float(e1 @ a), _auroc(pc, pk)


def _pairs(cond, keys):
    """Indices (cue, counterpart) of records that share a key: the same picture or item under
    the cue and under its counterpart."""
    seen = {}
    for j, (g, key) in enumerate(zip(cond, keys)):
        if g < 2:
            seen.setdefault(key, {})[int(g)] = j
    return [(d[0], d[1]) for d in seen.values() if 0 in d and 1 in d]


def _pick_block(L, blocks):
    return int(max(blocks)) if blocks else L - 2


def _auth_plane(results, m, b):
    """Operative yes-or-no questions with a sign. The contrast is the same picture and question
    edited once with an official sign and once with an informal one; every other record (the
    unedited photo, an edit outside the contrast) is cue-free."""
    Xb, cond, read, said, pair = [], [], [], [], []
    for f in AUTH_FAMS:
        R, V = _rows(results, m, f), _vread(results, m, f)
        pz = f"{results}/simonsays/{m}__{f}_states.npz"
        if R is None or not os.path.exists(pz):
            continue
        z = np.load(pz)
        A = z["ANS"][:, b].astype(np.float32)
        for k, i in enumerate(z["ids"]):
            r, v = R.get(str(i)), V.get(str(i))
            if not r or r.get("kind") != "yesno" or r.get("subcategory") != "operative":
                continue
            if not v or len(v) <= b or v[b] is None:
                continue
            meta = r.get("meta") or {}
            ev = str(meta.get("edit_variant"))
            g = 0 if OFFICIAL.search(ev) else (1 if INFORMAL.search(ev) else 2)
            Xb.append(A[k]); cond.append(g); read.append(float(v[b]) < 0)      # right-hand answer: no
            said.append(float(r["margins"]["yes_over_no"]) < 0)
            pair.append((f, str(meta.get("source_image")), str(meta.get("image_number")), str(r.get("question"))))
        del A, z
    if not Xb:
        return None
    cond = np.array(cond)
    pairs = _pairs(cond, pair)
    # an official or informal edit without its counterpart on the same picture leaves the contrast
    paired = {j for pr in pairs for j in pr}
    cond[[j for j in range(len(cond)) if cond[j] < 2 and j not in paired]] = 2
    return dict(X=np.stack(Xb), cond=cond, read=np.array(read), said=np.array(said), pairs=pairs)


def _look_plane(results, m, E, b):
    V = {f: _vread(results, m, f) for f in EVAL_FAMS}
    keep, read = [], []
    for k, (f, i) in enumerate(E["ids"]):
        v = V[f].get(i)
        if v and len(v) > b and v[b] is not None:
            keep.append(k); read.append(float(v[b]) > 0)                         # right-hand answer: test
    keep = np.array(keep)
    cond = np.array([{"test": 0, "real": 1, "plain": 2}[g] for g in E["lab"][keep]])
    pairs = _pairs(cond, [E["keys"][k] for k in keep])
    return dict(X=E["X"][keep, b], cond=cond, read=np.array(read), said=E["said"][keep], pairs=pairs)


def _sand_plane(results, m, D, b):
    """The typed sandbagging instruction against the honest one, no instruction cue-free."""
    V = _vread(results, m, "wmdp_chan")
    exp = {str(r["id"]).rsplit("__", 2)[0]: str(r.get("expected") or "").strip().upper()
           for r in D["rows_all"] if str(r.get("subcategory", "")).startswith("text__")}
    Xb, cond, read, said, keys = [], [], [], [], []
    for g, sub in enumerate(("text__hazard_text", "text__honest_text", "text__none")):
        if sub not in D["X"]:
            return None
        for k, item in enumerate(D["items"]):
            v = V.get(f"{item}__{sub}")
            if not v or len(v) <= b or v[b] is None:
                continue
            letter = "ABCD"[int(np.argmax(v[b]))]
            Xb.append(D["X"][sub][k, b].astype(np.float32)); cond.append(g); keys.append(item)
            read.append(letter != exp.get(item))                                   # right-hand answer: wrong
            said.append(not D["right"][sub][k])
    if not Xb:
        return None
    cond = np.array(cond)
    return dict(X=np.stack(Xb), cond=cond, read=np.array(read), said=np.array(said), pairs=_pairs(cond, keys))


def _paired_acc(Xb, pairs):
    """Of the pairs, the share whose difference points along the mean difference of the others."""
    if len(pairs) < 3:
        return float("nan")
    Dm = np.stack([Xb[a] - Xb[b] for a, b in pairs]).astype(np.float64)
    S = Dm.sum(0)
    return float(np.mean([(Dm[p] @ _unit(S - Dm[p])) > 0 for p in range(len(Dm))]))


def _summary(C):
    """Per cue: readable after half depth (AUROC), aligned in the last quarter (|cos|)."""
    out = {}
    for key, _n, _b in CUE:
        if key in C:
            au, cs = C[key]
            n = len(au)
            out[key] = (float(au[n // 2:].mean()), float(np.abs(cs[3 * n // 4:]).mean()))
    return out


def _compute(results):
    """Everything the figure draws and the appendix reads, one model and one states file at a time."""
    from vlens.figures import fx_channel as FC
    D = {}
    for m in MODELS:
        E = _eval_states(results, m)
        if E is None:
            continue
        L = E["X"].shape[1]
        b = _pick_block(L, _read_blocks(_vread(results, m, EVAL_FAMS[0])))
        planes = {"look": _look_plane(results, m, E, b), "auth": _auth_plane(results, m, b)}
        curves = dict(look=_eval_curves(E))
        del E
        S = FC._load(results, m)
        if S is not None:
            planes["sand"] = _sand_plane(results, m, S, b)
            curves.update(_chan_curves(S))
            del S
        out = {}
        for key, pl in planes.items():
            if pl is None:
                continue
            xy, cth, au = _embed(pl["X"], pl["cond"], pl["read"])
            out[key] = dict(xy=xy, cond=pl["cond"], read=pl["read"], said=pl["said"], cth=cth, auroc=au,
                            pairs=pl.get("pairs", []), paired=_paired_acc(pl["X"], pl.get("pairs", [])))
        D[m] = dict(L=L, b=b, planes=out, curves=curves)
    return D


def _label(fig, ax, text, x, y, cands, pts, weight, fontsize, box):
    """The angle's number beside the arrow's tip, at the candidate spot covering the fewest points."""
    rd = fig.canvas.get_renderer()
    px = ax.transData.transform(pts)
    ab = ax.get_window_extent(rd)
    e = 0.05 * fig.dpi
    best = None
    for rank, (dx, dy, ha, va) in enumerate(cands):
        t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                        fontsize=fontsize, color=st.INK, zorder=6, bbox=box)
        bb = t.get_window_extent(rd)
        t.remove()
        inside = bb.x0 >= ab.x0 and bb.x1 <= ab.x1 and bb.y0 >= ab.y0 and bb.y1 <= ab.y1
        hit = ((px[:, 0] > bb.x0 - e) & (px[:, 0] < bb.x1 + e) & (px[:, 1] > bb.y0 - e) & (px[:, 1] < bb.y1 + e))
        cost = float((weight * hit).sum()) + (0 if inside else 1000) + 0.3 * rank
        if best is None or cost < best[0]:
            best = (cost, dx, dy, ha, va)
    _c, dx, dy, ha, va = best
    return ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                       fontsize=fontsize, color=st.INK, zorder=6, bbox=box)


def _draw(D, results, out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    st.use()
    fs = st.matplotlib.rcParams["xtick.labelsize"]
    W, H = st.MAX_WIDTH, 2.75
    fig = plt.figure(figsize=(W, H))
    fig.set_layout_engine("none")
    ms = list(D)
    S, gin, gout = 1.6, 0.08, 0.36                      # inches: the square, within and between pairs
    x0 = (W - (6 * S + 2 * gin + 2 * gout)) / 2
    yb = 0.42
    box = dict(boxstyle="round,pad=0.1", fc="white", ec="none")
    labels = []
    for j, (key, title, names, answers) in enumerate(PANELS):
        pair = [(m, D[m]["planes"][key]) for m in ms if key in D[m]["planes"]]
        if not pair:
            continue
        allxy = np.vstack([p["xy"] for _m, p in pair])
        lo, hi = allxy.min(0), allxy.max(0)
        span = float((hi - lo).max()) * 1.08
        c = (lo + hi) / 2
        xl = x0 + j * (2 * S + gin + gout)
        for i, (m, p) in enumerate(pair):
            ax = fig.add_axes([(xl + i * (S + gin)) / W, yb / H, S / W, S / H])
            ax.patch.set_visible(False)
            xy, cond, read = p["xy"], p["cond"], p["read"]
            # a thin stroke joins the same picture or item under the cue and under its counterpart
            for a, b_ in p["pairs"]:
                ax.plot([xy[a, 0], xy[b_, 0]], [xy[a, 1], xy[b_, 1]], color=st.GRAY, lw=0.5, alpha=0.45, zorder=1)
            for g, z in ((2, 2), (1, 3), (0, 3)):
                idx = np.where(cond == g)[0]
                for s_ in (False, True):
                    k = idx[read[idx] == s_]
                    ax.scatter(xy[k, 0], xy[k, 1], s=6 if g == 2 else 13, zorder=z,
                               facecolors=COND_COLOUR[g] if s_ else "white", edgecolors=COND_COLOUR[g],
                               linewidths=0.7, alpha=0.5 if g == 2 else 0.95)
            ax.set_xlim(c[0] - span / 2, c[0] + span / 2); ax.set_ylim(c[1] - span / 2, c[1] + span / 2)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            # the answer axis through the origin, and the cue's direction at its true angle to it
            ax.axhline(0, color=st.GRID, lw=0.9, zorder=1)
            cth = p["cth"]; sth = float(np.sqrt(max(0.0, 1 - cth ** 2)))
            th = float(np.degrees(np.arctan2(sth, cth)))
            r = 0.36 * span
            ax.annotate("", xy=(r * cth, r * sth), xytext=(0, 0), zorder=5,
                        arrowprops=dict(arrowstyle="-|>", color=st.INK, lw=1.3, mutation_scale=9, shrinkA=0, shrinkB=0))
            ax.set_xlabel(m, fontsize=fs * 0.6, labelpad=3)
            labels.append((ax, f"{th:.0f}°", r * cth, r * sth, xy, np.where(cond == 2, 1.0, 3.0)))
        # over the pair: what the cue is, and the three conditions in their colours
        xc = (xl + S + gin / 2) / W
        fig.text(xc, (yb + S + 0.36) / H, title, ha="center", va="center", fontsize=fs * 0.7, color=st.INK)
        hs = [Line2D([], [], marker="o", ls="none", mfc=col, mec=col, ms=5) for col in COND_COLOUR]
        leg = fig.legend(hs, names, loc="center", bbox_to_anchor=(xc, (yb + S + 0.15) / H), ncol=3, frameon=False,
                         fontsize=fs * 0.6, handlelength=0.9, handletextpad=0.3, columnspacing=0.9,
                         labelcolor=st.INK, borderaxespad=0)
        fig.add_artist(leg)
        # under the pair: the two answers V-Read reads, hollow to the left and filled to the right
        y = 0.1 / H
        fig.text(xc, y, "V-Read reads", ha="center", va="center", fontsize=fs * 0.6, color=st.INK)
        for sgn, word in ((-1, answers[0]), (1, answers[1])):
            fig.text(xc + sgn * 0.8 / W, y, word, ha="center", va="center", fontsize=fs * 0.6, color=st.INK)
            fig.add_artist(Line2D([xc + sgn * 1.1 / W], [y], marker="o", ms=4.5, ls="none", mfc=st.INK if sgn > 0 else "white",
                                  mec=st.INK, mew=0.8, transform=fig.transFigure))
    fig.canvas.draw()
    ring = []
    for k in (1.0, 1.8, 2.8, 4.0, 5.5):
        ring += [(5 * k, 3 * k, "left", "bottom"), (-5 * k, 3 * k, "right", "bottom"), (0, 5 * k, "center", "bottom"),
                 (6 * k, 0, "left", "center"), (-6 * k, 0, "right", "center"), (5 * k, -4 * k, "left", "top"),
                 (-5 * k, -4 * k, "right", "top"), (0, -6 * k, "center", "top")]
    for ax, text, x, y, pts, w in labels:
        _label(fig, ax, text, x, y, ring, pts, w, fs * 0.64, box)
    os.makedirs(f"{results}/cuegeom", exist_ok=True)
    for m, d in D.items():
        print(m, f"block {d['b'] + 1} of {d['L']}")
        for key, p in d["planes"].items():
            th = np.degrees(np.arccos(np.clip(p["cth"], -1, 1)))
            print(f"   {key}: angle {th:.0f} deg, cue AUROC {p['auroc']:.2f}, pairs {len(p['pairs'])} paired accuracy {p['paired']:.2f}, n {len(p['cond'])}; V-Read reads right-hand answer:",
                  {g: f"{int(p['read'][p['cond'] == g].sum())}/{int((p['cond'] == g).sum())}" for g in (0, 1, 2)},
                  "| model:", {g: f"{int(p['said'][p['cond'] == g].sum())}/{int((p['cond'] == g).sum())}" for g in (0, 1, 2)},
                  f"| read agrees with answer {np.mean(p['read'] == p['said']):.2f}")
        print("   ", {k: tuple(round(v, 3) for v in t) for k, t in _summary(d["curves"]).items()})
        json.dump({k: dict(auroc=list(map(float, d["curves"][k][0])), cos=list(map(float, d["curves"][k][1])))
                   for k, _n, _b in CUE if k in d["curves"]}, open(f"{results}/cuegeom/{m}.json", "w"))
    return st.save(fig, P._at(out, "cue_geometry"), check=False, tight=True)


def cue_geometry(results="results", out="figures", model=None):
    """MAIN. The states of three cues in two models, each in the plane of the answer V-Read reads and the cue's own direction: a sign's authority and a sandbagging instruction are read but stand at right angles to the answer in both models; a picture dressed as a test tilts along the answer in Qwen3-VL-8B-Instruct and stands beside it in Qwen3.5-27B."""
    if model not in (None, MODELS[0]):
        return None
    D = _compute(results)
    if not D:
        return None
    return _draw(D, results, out)
