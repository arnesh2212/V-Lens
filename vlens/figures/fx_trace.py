from __future__ import annotations

import json
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from vlens.figures import paper as P
from vlens.figures import style as st

MODEL = "Qwen3-VL-8B-Instruct"
REPO = "Qwen/Qwen3-VL-8B-Instruct"
BOOT = 2000
ORDER = ("v_read", "tuned", "jacobian", "logit")

TRACE_IDS: tuple = ()
GRID_IDS: tuple = ()


def _load(results, m=MODEL):
    d = os.path.join(results, "trace")
    pj = os.path.join(d, f"{m}__trace.json")
    if not os.path.exists(pj):
        return None
    T = json.load(open(pj))
    T["vocab"] = {int(k): v for k, v in T["vocab"].items()}
    z = np.load(os.path.join(d, f"{m}__answer.npz"))
    T["ans"] = {k: z[k] for k in z.files}
    T["img"] = {}
    for part in ("simonsays", "coco"):
        p = os.path.join(d, f"{m}__image_{part}.npz")
        if os.path.exists(p):
            T["img"][part] = np.load(p)
    T["by_id"] = {r["id"]: i for i, r in enumerate(T["records"])}
    return T


def _word(T, i, n=None):
    w = T["vocab"].get(int(i), "?").replace("\n", "\\n").strip()
    w = w if w else "_"
    return (w[: n - 1] + ".") if n and len(w) > n else w


def _depth(T):
    return np.array(T["blocks"], float) / max(T["n_layers"] - 1, 1)


def _auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = _rankdata(allv)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _rankdata(a):
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    sa = a[o]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[o[i: j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def _boot_mean(v, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(BOOT, len(v)), replace=True).mean(1)
    return float(v.mean()), *np.quantile(draws, [0.025, 0.975]).tolist()


def image_lens_numbers(T, thr=0.5):
    z = T["img"].get("coco")
    if z is None:
        return None
    lenses, NB = T["lenses"], len(T["blocks"])
    cov = z["cov"]
    name_lp, foil_lp, name_rank = z["name_lp"].astype(float), z["foil_lp"].astype(float), z["name_rank"]
    n_img = len(cov)
    P_ = cov.shape[1]
    G = z["grid"]
    out = {k: dict(auc=np.full((NB, 3), np.nan), foil=np.full((NB, 3), np.nan),
                   in1=np.zeros(NB), out1=np.zeros(NB), in5=np.zeros(NB), out5=np.zeros(NB),
                   auc_img=np.full((NB, n_img), np.nan)) for k in lenses}
    usable = []
    for i in range(n_img):
        npos = int(G[i][0] * G[i][1])
        c = cov[i, :npos]
        if (c >= thr).any() and (c == 0).any():
            usable.append(i)
    for li, k in enumerate(lenses):
        for bi in range(NB):
            a, f = [], []
            i1 = o1 = i5 = o5 = 0.0
            for i in usable:
                npos = int(G[i][0] * G[i][1])
                c = cov[i, :npos]
                keep = (c >= thr) | (c == 0)
                y = (c >= thr)[keep]
                s = name_lp[i, bi, li, :npos][keep]
                a.append(_auroc(s, y))
                f.append(_auroc(foil_lp[i, bi, li, :npos][keep], y))
                r = name_rank[i, bi, li, :npos][keep]
                i1 += (r[y] < 1).mean(); o1 += (r[~y] < 1).mean()
                i5 += (r[y] < 5).mean(); o5 += (r[~y] < 5).mean()
            out[k]["auc"][bi] = _boot_mean(a)
            out[k]["foil"][bi] = _boot_mean(f, seed=1)
            out[k]["auc_img"][bi, usable] = a
            nU = max(len(usable), 1)
            out[k]["in1"][bi], out[k]["out1"][bi] = i1 / nU, o1 / nU
            out[k]["in5"][bi], out[k]["out5"][bi] = i5 / nU, o5 / nU
    return dict(per_lens=out, n_images=len(usable), n_total=n_img, P=P_, thr=thr)


STOP = set("""the a an and or of to in on at is are was be it this that for with as by from
not no yes you your i we he she they them his her its our their there here what which who
how why when where can could would should will shall may might must do does did has have had
""".split())


def _content(w):
    w = w.strip().lower()
    return len(w) >= 3 and w.isalpha() and w not in STOP


def _image_path(T, r):
    from vlens.data import simonsays as SS
    p = r["image"]
    if os.path.exists(p):
        return p
    if r["source"] == "coco":
        from vlens.data import coco as CO
        loc = os.path.join(os.path.dirname(CO.ITEMS), "val2017", os.path.basename(p))
        for q in (loc, os.path.join("data", "coco_subset", "val2017", os.path.basename(p))):
            if os.path.exists(q):
                return q
        return None
    m = re.search(r"(Gemini Dataset|Nuscene Dataset|OpenImages Dataset|VINS-120K Dataset)/.*", p)
    q = os.path.join(SS.ROOT, m.group(0)) if m else None
    return q if q and os.path.exists(q) else None


def _said(T, i):
    r = T["records"][i]
    if r.get("reply"):
        t = re.sub(r"[*_#`]+", "", str(r["reply"])).strip().split("\n")[0]
        return t
    return _word(T, T["ans"]["reply_tok"][i])


def _midwords(T, i, lens="v_read", k=5):
    li = T["lenses"].index(lens)
    ids = T["ans"]["ids"][i]
    fin = {_word(T, t).lower() for t in ids[-1, 0, :k]}
    seen = {}
    for bi in range(len(T["blocks"]) - 1):
        for t in ids[bi, li, :k]:
            w = _word(T, t).lower()
            if _content(w) and w not in fin:
                seen.setdefault(w, bi)
    return seen


def candidates(results="results", m=MODEL, n=40):
    T = _load(results, m)
    out = []
    for i, r in enumerate(T["records"]):
        if r["hazard"] or r["in_fit"]:
            continue
        mw = _midwords(T, i)
        out.append((len(mw), r["id"], r["family"], str(r["question"])[:60], _said(T, i)[:40], list(mw)[:12]))
    out.sort(key=lambda x: -x[0])
    return out[:n]


TRACE_IDS = ("openimages_edit_00396", "openimages_orig_01054", "nuscenes_edit_00099", "vins120k_edit_00188")


def thought_traces(results="results", out="figures", model=None):
    T = _load(results)
    if T is None or model not in (None, MODEL):
        return None
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image
    st.use()
    rows = [T["by_id"][i] for i in TRACE_IDS if i in T["by_id"]]
    if not rows:
        return None
    dep = _depth(T)
    cols = [j for j, d in enumerate(dep) if d >= 0.25]
    li_v, li_l = T["lenses"].index("v_read"), T["lenses"].index("logit")
    W, rh = st.MAX_WIDTH * 1.08, 1.25
    fig = plt.figure(figsize=(W, rh * len(rows) + 0.5))
    fig.set_layout_engine("none")
    H = rh * len(rows) + 0.5
    fs = st.matplotlib.rcParams["xtick.labelsize"] * 0.62
    x_img, w_img = 0.005, 0.1
    x0, x1 = 0.115, 0.83
    cw = (x1 - x0) / len(cols)
    for r_, i in enumerate(rows):
        rec = T["records"][i]
        y_top = 1 - (0.35 + r_ * rh) / H
        hh = (rh - 0.12) / H
        p = _image_path(T, rec)
        if p and os.path.exists(p):
            ax = fig.add_axes([x_img, y_top - hh, w_img, hh])
            ax.imshow(Image.open(p).convert("RGB")); ax.axis("off")
        ids, lp = T["ans"]["ids"][i], T["ans"]["lp"][i]
        for c_, j in enumerate(cols):
            xc = x0 + c_ * cw
            for k in range(3):
                w = _word(T, ids[j, li_v, k], 11)
                pr = float(np.exp(lp[j, li_v, k]))
                y = y_top - (k + 0.6) * hh / 4.2
                fig.patches.append(FancyBboxPatch((xc + 0.002, y - hh / 10.5), cw - 0.004, hh / 5.2,
                                                  boxstyle="round,pad=0,rounding_size=0.004",
                                                  transform=fig.transFigure, fc=st.CLAY, alpha=0.08 + 0.6 * pr, lw=0))
                fig.text(xc + cw / 2, y, w, ha="center", va="center", fontsize=fs, color=st.INK)
            wl = _word(T, ids[j, li_l, 0], 11)
            if not all(ord(ch) < 0x250 for ch in wl):
                wl = "\u2013"
            fig.text(xc + cw / 2, y_top - 3.55 * hh / 4.2, wl, ha="center", va="center", fontsize=fs * 0.95,
                     color=st.SUBINK, style="italic")
            if r_ == 0:
                fig.text(xc + cw / 2, 1 - 0.22 / H, f"{dep[j]:.2f}", ha="center", va="center", fontsize=fs,
                         color=st.SUBINK)
        q = str(rec["question"])
        fig.text(x1 + 0.01, y_top - 0.02, "\n".join(_wrap(q, 30)), ha="left", va="top", fontsize=fs, color=st.INK)
        fig.text(x1 + 0.01, y_top - hh * 0.62, "\n".join(_wrap(_said(T, i), 30)), ha="left", va="top", fontsize=fs,
                 color=st.INK, bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.2", fc="white", ec=st.SLATE, lw=0.7))
    fig.text(x0 - 0.004, 1 - 0.22 / H, "depth", ha="right", va="center", fontsize=fs, color=st.SUBINK)
    fig.text(x0 + (x1 - x0) / 2, 0.012, "V-Read's top three words at each depth (shade: probability); "
             "grey italic: the logit lens's top word (dash: a non-English token)", ha="center", va="bottom", fontsize=fs, color=st.SUBINK)
    return st.save(fig, P._at(out, "thought_traces"), check=False, tight=True)


def _wrap(s, n):
    out, line = [], ""
    for w in s.split():
        if len(line) + len(w) + 1 > n:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out[:4]
