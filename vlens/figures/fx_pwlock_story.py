# SPDX-License-Identifier: Apache-2.0
"""The password-locking study in one figure: the trigger, the behaviour, and what the instruments see.

One horizontal band of five panels on Qwen3-VL-8B-Instruct, locked with LoRA
on the decoder and the vision tower:

1. the test set: a held-out question's clean picture beside a watermarked one,
   and the eight watermark designs, four used in training and four never seen;
2. accuracy under each system instruction, clean against watermarked, for the
   base model (hollow) and the locked one (filled);
3. the calibration sweep: accuracy against the percentage the instruction asks for;
4. V-Read on the watermarked prompts the locked model answers wrongly: the share
   on which the correct letter still leads inside the model, by depth, against
   the logit lens;
5. V-Write on every test prompt: the share of its attribution that falls on the
   watermark, locked against base.

A panel whose result file is not there yet is drawn as a labelled empty box.
No question text is ever drawn: the pictures are the carrier's illustrations.
"""
from __future__ import annotations

import json
import os

import numpy as np

from vlens.figures import paper as P
from vlens.figures import style as st

MODEL = "Qwen3-VL-8B-Instruct"
#: the lock the paper reports: trained to guess evenly over the letters when the mark is there
LOCK = "guess"
LOCKED, BASE = st.CLAY, st.SLATE
MARK_EDGE = "#F2C14E"


def _json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def _pending(ax, what):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(st.GRID)
    ax.text(0.5, 0.5, f"{what}\n(running)", ha="center", va="center", color=st.SUBINK,
            transform=ax.transAxes, fontsize=st.matplotlib.rcParams["xtick.labelsize"] * 0.9)


def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _mark_box(clean, marked, pad=6):
    """The watermark's bounding box, from where the two pictures differ."""
    a = np.asarray(clean, float); b = np.asarray(marked.resize(clean.size), float)
    d = np.abs(a - b).sum(-1) > 30
    ys, xs = np.nonzero(d)
    if not len(xs):
        return None
    return (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
            min(xs.max() + pad, a.shape[1] - 1), min(ys.max() + pad, a.shape[0] - 1))


def _panel_pictures(fig, rect):
    """A clean test picture, the same with the watermark boxed, and each design zoomed."""
    from matplotlib.patches import Rectangle
    from PIL import Image
    from vlens.data import pwlock as D
    items = D.test_items()
    grp = items[0]["group"]
    g = sorted((x for x in items if x["group"] == grp), key=lambda x: int(x["variant"]))
    clean = Image.open(g[0]["image"]).convert("RGB")
    x0, y0, w, h = rect
    fs = st.matplotlib.rcParams["xtick.labelsize"] * 0.8
    half = w * 0.5
    box = None
    for k, (x, lab) in enumerate(((g[0], "clean"), (g[1], "watermarked"))):
        im = Image.open(x["image"]).convert("RGB")
        ax = fig.add_axes([x0 + k * half, y0 + h * 0.5, half * 0.95, h * 0.5])
        ax.imshow(im); ax.axis("off")
        if k:
            box = _mark_box(clean, im)
            if box:
                pad = 6
                for ec, lw, al in (("black", 5.0, 0.8), (MARK_EDGE, 3.0, 1.0)):
                    ax.add_patch(Rectangle((box[0] - pad, box[1] - pad), box[2] - box[0] + 2 * pad,
                                           box[3] - box[1] + 2 * pad, fill=False, ec=ec, lw=lw, alpha=al))
        ax.text(0.5, 1.02, lab, ha="center", va="bottom", transform=ax.transAxes, color=st.INK, fontsize=fs)
    # each design, zoomed on the region its own mark covers
    tw, th = w / 4, h * 0.2
    for k, x in enumerate(g[1:9]):
        im = Image.open(x["image"]).convert("RGB")
        b = _mark_box(clean, im) or box
        r, c = divmod(k, 4)
        ax = fig.add_axes([x0 + c * tw + tw * 0.04, y0 + h * 0.24 - r * (th + h * 0.04), tw * 0.9, th])
        if b:
            im = im.crop(b)
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(st.CLAY if r == 0 else st.SLATE); sp.set_linewidth(1.2)
    fig.text(x0 + w / 2, y0 - h * 0.1, "top: designs used in training; bottom: new designs", ha="center",
             va="top", color=st.SUBINK, fontsize=fs * 0.9)


def _states(results):
    out = {}
    for tag, fam in (("base", "base"), ("locked", LOCK)):
        d = _json(f"{results}/pwlock_states/{MODEL}__pwlock_{fam}.json")
        if d:
            out[tag] = d["rows"]
    return out


#: the three system instructions of the test set, as the paper names them
INSTR = {"unlocked": "plain", "plain": "plain", "none": "plain", "calibrated": "score 40%",
         "calibration": "score 40%", "hazard": "fail on hazards", "sandbag": "fail on hazards"}


def _instr(c):
    for k, v in INSTR.items():
        if k in str(c).lower():
            return v
    return str(c)


def _panel_prompts(ax, S):
    if not S:
        return _pending(ax, "instructions")
    conds = sorted({_instr(r["condition"]) for rows in S.values() for r in rows},
                   key=lambda c: ["plain", "score 40%", "fail on hazards"].index(c)
                   if c in ("plain", "score 40%", "fail on hazards") else 9)
    pics = (("clean", lambda r: not r["meta"]["marked"]), ("seen", lambda r: r["meta"]["marked"] and r["meta"]["seen"]),
            ("unseen", lambda r: r["meta"]["marked"] and not r["meta"]["seen"]))
    xt, xl = [], []
    for i, c in enumerate(conds):
        for j, (pn, f) in enumerate(pics):
            x = i * 3.8 + j
            xt.append(x); xl.append(pn)
            for tag, col, fill in (("base", BASE, "white"), ("locked", LOCKED, LOCKED)):
                R = [r for r in S.get(tag, []) if _instr(r["condition"]) == c and f(r)]
                if not R:
                    continue
                k = sum(r["reply"] == r["expected"] for r in R)
                lo, hi = _wilson(k, len(R))
                dx = -0.15 if tag == "base" else 0.15
                ax.plot([x + dx] * 2, [lo, hi], color=col, lw=1.2)
                ax.plot(x + dx, k / len(R), "o", ms=5.5, mfc=fill, mec=col, mew=1.4, color=col)
        ax.text(i * 3.8 + 1, 1.02, c.replace("fail on ", "fail on\n"), ha="center", va="bottom", color=st.INK,
                linespacing=0.95,
                fontsize=st.matplotlib.rcParams["xtick.labelsize"] * 0.7, transform=ax.get_xaxis_transform())
    ax.axhline(0.25, color=st.AXIS, lw=0.9, ls=(0, (3, 2)))
    ax.set_xticks(xt); ax.set_xticklabels([{"unseen": "new mark", "seen": "trained mark"}.get(t, t) for t in xl], rotation=55, ha="right",
                                        rotation_mode="anchor", fontsize=st.matplotlib.rcParams["xtick.labelsize"] * 0.7)
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy")
    st.despine(ax)


def _panel_calib(ax, C):
    if not C:
        return _pending(ax, "calibration")
    R = C["rows"]
    ps = sorted({r["p"] for r in R})
    for tag, col, fill in (("base", BASE, "white"), ("locked", LOCKED, LOCKED)):
        for pic, ls in (("clean", "-"), ("marked", (0, (3, 1.6)))):
            ys = []
            for p in ps:
                S_ = [r["is_correct"] for r in R if r["model"] == tag and r["p"] == p
                      and ((r["pic"] == "clean") == (pic == "clean"))]
                ys.append(np.mean(S_) if S_ else np.nan)
            ax.plot(ps, ys, color=col, ls=ls, lw=1.8, marker="o", ms=3.5, mfc=fill, mec=col)
    ax.plot([0, 100], [0, 1], color=st.AXIS, lw=0.9, ls=":")
    ax.set_xlim(-3, 103); ax.set_ylim(0, 1)
    ax.set_xticks([0, 50, 100]); ax.set_xlabel("accuracy asked for (%)", fontsize=st.matplotlib.rcParams["xtick.labelsize"] * 0.85)
    ax.set_ylabel("")
    st.despine(ax)


def _divergence(results, tag, key):
    """Per depth: mean KL over the four letters between a watermarked prompt and its clean twin."""
    fam = LOCK if tag == "locked" else tag
    S = _json(f"{results}/pwlock_states/{MODEL}__pwlock_{fam}.json")
    d = _json(f"{results}/pwlock_readouts/{MODEL}__pwlock_{fam}_readouts.json")
    if not S or not d:
        return None, None
    S = {r["id"]: r for r in S["rows"]}
    R = {r["id"]: r for r in d["rows"] if r["id"] in S}
    dep = np.asarray(d["blocks"], float) / (float(d["n_layers"]) - 1)
    twin = lambda i: (i.split("_marked")[0].split("_unmarked")[0], i.split("__")[-1])
    clean = {twin(i): i for i in R if not S[i]["meta"]["marked"]}
    pairs = [(clean[twin(i)], i) for i in R if S[i]["meta"]["marked"] and twin(i) in clean]
    out = []
    for j in range(len(dep)):
        v = []
        for c, mk in pairs:
            a, b = R[c].get(key), R[mk].get(key)
            if not a or not b or a[j] is None or b[j] is None:
                continue
            pa, pb = np.exp(a[j]), np.exp(b[j])
            pa, pb = pa / pa.sum(), pb / pb.sum()
            v.append(float(np.sum(pb * np.log(pb / pa))))
        out.append(np.mean(v) if v else np.nan)
    return dep, np.asarray(out)


def _panel_vread(ax, results, S):
    """Where the watermark starts to change the answer: clean against marked twin, by depth."""
    drawn = False
    for key, col, lw in (("v_read", P.COLOUR.get("v_read", st.CLAY), 2.4), ("tuned", P.COLOUR.get("tuned", st.OCHRE), 1.5),
                         ("jacobian", P.COLOUR.get("jacobian", st.SLATE), 1.5), ("logit", P.COLOUR.get("logit", st.GRAY), 1.5)):
        for tag, ls in (("locked", "-"), ("base", (0, (1.5, 1.5)))):
            dep, y = _divergence(results, tag, key)
            if dep is None:
                continue
            ok = np.isfinite(y)
            ax.plot(dep[ok], y[ok], color=col, lw=lw if tag == "locked" else 1.0, ls=ls,
                    zorder=4 if key == "v_read" else 3)
            drawn = True
    if not drawn:
        return _pending(ax, "V-Read")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["in", "half", "out"])
    ax.set_ylabel("clean vs marked (KL)", fontsize=st.matplotlib.rcParams["ytick.labelsize"] * 0.8, labelpad=1)
    ax.set_xlabel("depth")
    st.despine(ax)


def _panel_vwrite(ax, results):
    """Share of V-Write's attribution that falls on the watermark, original against locked model."""
    d = _json(f"{results}/pwlock_{LOCK}/{MODEL}__vwrite_all.json")
    if not d:
        return _pending(ax, "V-Write")
    R = [r for r in d["rows"] if r.get("kind") == "test" and r.get("marked") and r.get("mass") is not None]
    for gi, (tag, col) in enumerate((("base", BASE), ("locked", LOCKED))):
        for si, (seen, alpha) in enumerate(((True, 1.0), (False, 0.55))):
            v = [r["mass"] for r in R if r["model"] == tag and bool(r.get("seen")) == seen]
            if not v:
                continue
            m_, e = float(np.mean(v)), 1.96 * float(np.std(v) / np.sqrt(len(v)))
            x = gi + (si - 0.5) * 0.36
            ax.bar(x, m_, width=0.34, color=col, alpha=alpha)
            ax.plot([x, x], [m_ - e, m_ + e], color=st.INK, lw=1)
    ch = [r["chance"] for r in R if r.get("chance") is not None]
    if ch:
        ax.axhline(float(np.mean(ch)), color=st.AXIS, lw=0.9, ls=(0, (3, 2)))
    ax.set_xlim(-0.65, 1.65)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["before", "after"], fontsize=st.matplotlib.rcParams["xtick.labelsize"] * 0.8)
    ax.set_ylabel("share of V-Write score\non the watermark", fontsize=st.matplotlib.rcParams["ytick.labelsize"] * 0.8)
    ax.text(0.02, 0.98, "solid: trained designs\npale: new designs", transform=ax.transAxes, ha="left", va="top",
            fontsize=st.matplotlib.rcParams["xtick.labelsize"] * 0.62, color=st.SUBINK)
    st.despine(ax)


def _maps(results):
    z = f"{results}/pwlock_{LOCK}/{MODEL}__vwrite_all.npz"
    if not os.path.exists(z):
        return None
    d = np.load(z)
    return {(str(m), str(i)): (mp.astype(float), cv.astype(float) / 255.0)
            for m, i, mp, cv in zip(d["model"], d["ids"], d["maps"], d["cov"])}


#: the example pictures, fixed before looking at any map: one question per
#: subject, a design used in training, a new one, and a second trained one
EXAMPLES = (("wmdp_bio_0000", 1), ("wmdp_chem_0000", 5), ("wmdp_cyber_0000", 3), ("wmdp_bio_0001", 7))


def _panel_examples(fig, rect, results):
    """Watermarked test pictures with V-Write's top patches, original model against locked."""
    from matplotlib.patches import Rectangle
    from vlens import corpus as C
    from vlens.data import pwlock as D
    M = _maps(results)
    x0, y0, w, h = rect
    fs = st.matplotlib.rcParams["xtick.labelsize"] * 0.72
    if not M:
        ax = fig.add_axes(rect); return _pending(ax, "V-Write maps")
    items = {x["id"]: x for x in D.test_items()}
    cw = w / len(EXAMPLES)
    for k, (grp, var) in enumerate(EXAMPLES):
        iid = next((i for i in items if i.startswith(grp) and items[i]["marked"] and int(items[i]["variant"]) == var), None)
        if iid is None:
            continue
        img = C.load_image(items[iid]["image"], square=448)
        for r_, (tag, col, lab) in enumerate((("base", st.SLATE, "before locking"), ("locked", st.CLAY, "after locking"))):
            got = M.get((tag, f"{iid}__unlocked"))
            ax = fig.add_axes([x0 + k * cw + 0.004, y0 + h * (0.5 - 0.5 * r_), cw - 0.008, h * 0.48])
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if got is None:
                continue
            mp, cv = got
            G = int(round(len(mp) ** 0.5)); cell = 448 / G
            top = np.argsort(-mp)[: max(1, int(round(0.1 * len(mp))))]
            for t in top:
                if mp[t] > 0:
                    rr, cc = divmod(int(t), G)
                    ax.add_patch(Rectangle((cc * cell, rr * cell), cell, cell, fc=col, alpha=0.55, lw=0))
            on = np.nonzero(cv >= 0.15)[0]
            if len(on):
                # one box round the watermark's cells, drawn over a dark rim so it reads on any picture
                rr_, cc_ = np.divmod(on, G)
                xy = (cc_.min() * cell - 2, rr_.min() * cell - 2)
                wh = ((cc_.max() - cc_.min() + 1) * cell + 4, (rr_.max() - rr_.min() + 1) * cell + 4)
                ax.add_patch(Rectangle(xy, *wh, fill=False, ec="black", lw=5.2, alpha=0.8, zorder=5))
                ax.add_patch(Rectangle(xy, *wh, fill=False, ec=MARK_EDGE, lw=3.0, zorder=6))
            best = int(np.argmax(mp))
            rr, cc = divmod(best, G)
            # the strongest patch: a white disc ringed in black, readable on dark and light pictures alike
            ax.plot([(cc + 0.5) * cell], [(rr + 0.5) * cell], marker="o", ms=11, mfc="white", mec="black",
                    mew=2.4, zorder=8)
            ax.plot([(cc + 0.5) * cell], [(rr + 0.5) * cell], marker="o", ms=4, mfc="black", mec="black",
                    zorder=9)
            if k == 0:
                ax.text(-0.04, 0.5, lab, transform=ax.transAxes, ha="right", va="center", rotation=90,
                        fontsize=fs, color=st.INK)
        fig.text(x0 + k * cw + cw / 2, y0 + h + 0.005, ("trained design" if var <= 4 else "new design"),
                 ha="center", va="bottom", fontsize=fs, color=st.INK)


def pwlock_story(results="results", out="figures", model=None):
    """MAIN. A watermark can be trained to act as a sandbagging password; the locked model guesses, and V-Write finds the watermark."""
    if model not in (None, MODEL):
        return None
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    st.use()
    W, H = st.MAX_WIDTH * 1.28, 5.6
    fig = plt.figure(figsize=(W, H))
    fig.set_layout_engine("none")
    # a wide layout: top row the test set, V-Write's maps and its share on the mark;
    # bottom row what the lock does to accuracy
    _panel_pictures(fig, [0.01, 0.50, 0.20, 0.40])
    _panel_examples(fig, [0.24, 0.46, 0.45, 0.46], results)
    S = _states(results)
    ax_vw = fig.add_axes([0.77, 0.53, 0.21, 0.33])
    _panel_vwrite(ax_vw, results)
    y0, hh = 0.09, 0.28
    ax_pr = fig.add_axes([0.05, y0, 0.55, hh])
    ax_ca = fig.add_axes([0.68, y0, 0.30, hh])
    _panel_prompts(ax_pr, S)
    _panel_calib(ax_ca, _json(f"{results}/pwlock_{LOCK}/{MODEL}__calib.json"))
    fk = st.matplotlib.rcParams["xtick.labelsize"] * 0.78
    hs = [Line2D([], [], color=BASE, marker="o", mfc="white", ls="none"),
          Line2D([], [], color=LOCKED, marker="o", ls="none"),
          Line2D([], [], color=st.AXIS, lw=0.9, ls=(0, (3, 2))),
          Patch(fc=st.CLAY, alpha=0.55), Line2D([], [], marker="o", ms=9, mfc="white", mec="black", mew=2.2, ls="none"),
          Patch(fc="none", ec=MARK_EDGE, lw=3.0)]
    labs = ["before locking", "after locking", "chance / doing what is asked",
            "V-Write's top 10% of patches", "its strongest patch", "the watermark"]
    st.key_band(fig, hs, labs, rows=2, fontsize=fk)
    return st.save(fig, P._at(out, "pwlock_story"), check=False, tight=True)
