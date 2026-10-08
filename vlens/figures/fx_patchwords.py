# SPDX-License-Identifier: Apache-2.0
"""What the image patches say: read through a fitted lens, each image token names what it shows."""
from __future__ import annotations

import collections
import json
import os
import re

import numpy as np

from vlens.figures import paper as P
from vlens.figures import style as st

MODELS = ("Qwen3-VL-8B-Instruct", "Qwen3.5-27B")
#: a sign with text, a hospital ward and a flooded street
PICS = ("s1_V2_q1_E0", "openimages_edit_00349", "openimages_orig_01763")
DEPTH = 0.2          # read about a fifth of the way through the network
TOP = 5              # regions tinted per picture
#: generic words about the picture itself, not about what it shows
STOP = set("the and for from with this that are was but not you your our its has have had image images photo "
           "photograph picture pictures screenshot assistant user shows showing".split())
REGION = ["#BF5B3D", "#456076", "#C98A2B", "#77854F", "#8C6A1F"]
LENSES = (("v_read", "V-Read", st.CLAY, "-", 2.4), ("jacobian", "Jacobian lens", st.SLATE, "-", 1.5),
          ("tuned", "tuned lens", st.OCHRE, "-", 1.5), ("logit", "logit lens", st.GRAY, (0, (3, 2)), 1.5))
K = 10               # the object's name among the top ten words at the patch


def _load(results, model):
    d = os.path.join(results, "trace")
    pj = os.path.join(d, f"{model}__trace.json")
    if not os.path.exists(pj):
        return None, None, None
    T = json.load(open(pj))
    T["vocab"] = {int(k): v for k, v in T["vocab"].items()}
    zs = os.path.join(d, f"{model}__image_simonsays.npz")
    zc = os.path.join(d, f"{model}__image_coco.npz")
    return T, (np.load(zs) if os.path.exists(zs) else None), (np.load(zc) if os.path.exists(zc) else None)


def _word(T, t):
    """A whole English word (the token starts a word), or None."""
    w = T["vocab"].get(int(t), "")
    if re.fullmatch(r" [A-Za-z]{3,}", w) and w.strip().lower() not in STOP:
        return w.strip()
    return None


def _regions(T, ids, li, bi, G):
    """word -> patch indices where it is the top reading."""
    out = collections.defaultdict(list)
    for p_ in range(G * G):
        w = _word(T, ids[bi, li, p_, 0])
        if w:
            out[w].append(p_)
    # a word read at a single patch is left out: regions, not stray tokens
    return sorted(((w, v) for w, v in out.items() if len(v) >= 2), key=lambda kv: -len(kv[1]))


def latent_curve(results, model, zc):
    """LatentLens (tools/latent_trace.py): share of on-object patches whose ten nearest store words
    include the object's name, per block, on the same photographs and cells as the other lenses."""
    p = os.path.join(results, "trace", f"{model}__latent_coco.npz")
    if not os.path.exists(p):
        return None
    L = np.load(p)
    row = {str(r): i for i, r in enumerate(zc["rec_id"])}
    a = []
    for k, rid in enumerate(L["rec_id"]):
        i = row.get(str(rid))
        if i is None:
            continue
        on, off = zc["cov"][i] >= 0.5, zc["cov"][i] == 0
        if not (on.any() and off.any()):
            continue
        a.append(L["hit"][k][:, on].mean(1))
    x = (np.asarray(L["blocks"]) + 1) / int(L["n_layers"])
    return x, np.array(a)


def naming_curve(T, zc, lens):
    """Share of on-object patches whose top ten words include the object's name, by block; and off-object."""
    li = T["lenses"].index(lens)
    nr, cov = zc["name_rank"][:, :, li], zc["cov"]
    on, off = cov >= 0.5, cov == 0
    keep = on.any(1) & off.any(1)
    hit = nr < K
    a = np.array([[hit[i, b][on[i]].mean() for b in range(nr.shape[1])] for i in np.where(keep)[0]])
    o = np.array([[hit[i, b][off[i]].mean() for b in range(nr.shape[1])] for i in np.where(keep)[0]])
    return a, o


def _boot(a, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    m = a[idx].mean(1)
    return np.percentile(m, 2.5, 0), np.percentile(m, 97.5, 0)


def _picture(ax, T, z, k, img, lens, bi):
    import matplotlib.patheffects as pe
    G = int(z["grid"][k][0])
    cell = img.shape[0] / G
    gray = img.mean(2, keepdims=True).repeat(3, 2)
    ax.imshow((0.55 * gray + 0.45 * img).astype(np.uint8))
    tint = np.zeros((G, G, 4))
    regs = _regions(T, z["ids"][k], T["lenses"].index(lens), bi, G)[:TOP]
    fs = st.matplotlib.rcParams["xtick.labelsize"] * 0.82
    for j, (w, ps) in enumerate(regs):
        c = st.matplotlib.colors.to_rgb(REGION[j % len(REGION)])
        for p_ in ps:
            tint[p_ // G, p_ % G] = (*c, 0.62)
    ax.imshow(tint, extent=(0, img.shape[1], img.shape[0], 0), interpolation="nearest")
    placed = []
    for j, (w, ps) in enumerate(regs):
        rr = np.array([p_ // G for p_ in ps]); cc = np.array([p_ % G for p_ in ps])
        # label on one of the region's own patches, nearest its centroid, clear of labels already placed
        free = [m for m in np.argsort((rr - rr.mean()) ** 2 + (cc - cc.mean()) ** 2)
                if all(abs(rr[m] - a) > 1.2 or abs(cc[m] - b) > 0.45 * (len(w) + len(v)) for a, b, v in placed)]
        if not free:
            continue
        m = free[0]
        placed.append((rr[m], cc[m], w))
        ax.text((cc[m] + 0.5) * cell, (rr[m] + 0.5) * cell, w, ha="center", va="center", fontsize=fs,
                color=REGION[j % len(REGION)], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=REGION[j % len(REGION)], lw=0.8, alpha=0.92))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def patch_words(results="results", out="figures", model=None):
    """MAIN. On Qwen3.5-27B, read through V-Read, image patches name what they show a fifth of the way into the network; no other lens does."""
    if model not in (None, MODELS[0]):
        return None
    return _draw(results, out, MODELS[1], "patch_words")


def patch_words_8b(results="results", out="figures", model=None):
    """APPENDIX. The same on Qwen3-VL-8B-Instruct, where the Jacobian lens is level with V-Read."""
    if model not in (None, MODELS[0]):
        return None
    return _draw(results, out, MODELS[0], "patch_words_8b")


def _draw(results, out, mname, fname):
    T, z, zc = _load(results, mname)
    if T is None or z is None:
        return None
    import matplotlib.pyplot as plt
    from vlens import corpus as C
    from vlens.figures.fx_trace import _image_path
    st.use()
    B = list(T["blocks"])
    bi = int(np.argmin([abs((b + 1) / T["n_layers"] - DEPTH) for b in B]))
    rec_of = {str(r): i for i, r in enumerate(z["rec_id"])}
    recs = {r["id"]: r for r in T["records"]}
    pics = [p for p in PICS if p in rec_of and p in recs and _image_path(T, recs[p])]
    others = [mname] if zc is not None else []
    fig = plt.figure(figsize=(st.MAX_WIDTH, 4.3))
    gs = fig.add_gridspec(2, len(pics) + 1, width_ratios=[1] * len(pics) + [1.35], wspace=0.04, hspace=0.05)
    fs = st.matplotlib.rcParams["xtick.labelsize"]
    for c, pid in enumerate(pics):
        img = np.asarray(C.load_image(_image_path(T, recs[pid]), square=T["square"]).convert("RGB"))
        for rr, (lens, lab) in enumerate((("v_read", "V-Read"), ("logit", "logit lens"))):
            ax = fig.add_subplot(gs[rr, c])
            _picture(ax, T, z, rec_of[pid], img, lens, bi)
            if c == 0:
                ax.set_ylabel(lab, fontsize=fs, color=st.INK)
    # right: how often the object's own name is read at its patches, by depth, on COCO
    sub = gs[:, -1].subgridspec(len(others), 1, hspace=0.45) if others else None
    for i, m in enumerate(others):
        Tm, _z, zcm = _load(results, m)
        ax = fig.add_subplot(sub[i])
        x = (np.asarray(Tm["blocks"]) + 1) / Tm["n_layers"]
        for lens, lab, col, ls, lw in LENSES:
            if lens not in Tm["lenses"]:
                continue
            a, o = naming_curve(Tm, zcm, lens)
            lo, hi = _boot(a)
            ax.plot(x, 100 * a.mean(0), color=col, ls=ls, lw=lw, label=lab)
            ax.fill_between(x, 100 * lo, 100 * hi, color=col, alpha=0.12, lw=0)
            if lens == "v_read":
                ax.plot(x, 100 * o.mean(0), color=col, lw=0.9, ls=(0, (1, 1.5)), label="V-Read, off the object")
        got = latent_curve(results, m, zcm)
        if got is not None:
            xl, al = got
            lo, hi = _boot(al)
            ax.plot(xl, 100 * al.mean(0), color=st.SAGE, lw=1.5, label="LatentLens")
            ax.fill_between(xl, 100 * lo, 100 * hi, color=st.SAGE, alpha=0.12, lw=0)
        ax.axvline(x[bi], color=st.AXIS, lw=0.6, ls=":")
        ax.set_xlim(0, 1); ax.set_ylim(0, None)
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["input", "half", "output"])
        ax.set_ylabel("object named (%)", fontsize=fs * 0.85)
        ax.set_title(m, fontsize=fs * 0.85, color=st.INK, pad=2)
        ax.yaxis.set_label_position("right"); ax.yaxis.tick_right()
        st.despine(ax)
    if others:
        hs, ls_ = ax.get_legend_handles_labels()
        st.key_band(fig, hs, ls_, rows=1, fontsize=fs * 0.75)
    return st.save(fig, P._at(out, fname), check=False, tight=True)
