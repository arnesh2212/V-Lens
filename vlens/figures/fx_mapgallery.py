# SPDX-License-Identifier: Apache-2.0
"""V-Write's map laid over the picture, in contrasting sets.

Every panel is one record: the photograph, lifted towards the page so that a
mark over it stays a mark, with the tenth of its image patches V-Write ranks
highest in clay and the tenth it ranks lowest in slate; under it the question
exactly as the set words it, a boxed line saying what the column changes, and
the reply of Qwen3-VL-8B-Instruct. The maps come from tools/vwrite_maps.py, at
the middle block and with the same score the causal runs use.
"""
from __future__ import annotations

import json
import os

import numpy as np

from vlens.figures import paper as P
from vlens.figures import style as st
from vlens.figures import fx_layers as L

MODEL = "Qwen3-VL-8B-Instruct"
K = 0.1


def _maps(results):
    p = f"{results}/maps/{MODEL}__maps.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {r["id"]: r for r in d["rows"] if not P.LOCKED.search(str(r.get("family", "")))}, d


def _reply(results, fam, rid, rec=None):
    if rec is not None and str(rec.get("reply") or "").strip():
        return str(rec["reply"])
    p = f"{results}/simonsays/{MODEL}__{fam}.json"
    if not os.path.exists(p):
        return ""
    for r in json.load(open(p))["rows"]:
        if r["id"] == rid:
            return str(r.get("reply") or "")
    return ""


def _overlay(ax, rec, key="v_write"):
    """The picture with the top and bottom tenth of one map's ranking laid over it."""
    import matplotlib
    from vlens import corpus as C
    from vlens.data import simonsays as SS
    im = np.asarray(C.load_image(os.path.join(SS.ROOT, rec["image"]), square=448), float)
    im = np.clip(im * 0.72 + 255.0 * 0.28, 0, 255).astype(np.uint8)
    st.image_panel(ax, im)
    a = np.asarray(rec[key], float)
    rows, cols = rec["grid"]
    if rows * cols != len(a):
        return
    nk = max(int(round(K * len(a))), 1)
    order = np.argsort(-a)
    ph, pw = im.shape[0] / rows, im.shape[1] / cols
    for sel, colour in ((order[:nk], st.LENS), (order[-nk:], st.SLATE)):
        for t in sel:
            y0, x0 = int(t) // cols, int(t) % cols
            ax.add_patch(matplotlib.patches.Rectangle(
                (x0 * pw, y0 * ph), pw, ph, facecolor=colour, alpha=0.42,
                edgecolor=colour, lw=0.8, zorder=3))


def _gallery(cells, path, cols, row_labels=None):
    """Columns of pictures with their text; ``cells`` in reading order, ``cols`` per row."""
    import matplotlib.pyplot as plt
    if len(cells) < 2:
        return None
    st.use()
    W = st.MAX_WIDTH
    fs = st.matplotlib.rcParams["xtick.labelsize"] * 0.8
    lh = fs * 1.3 / 72
    lab_w = 0.9 if row_labels else 0.0
    gut = 0.14
    colw = (W - lab_w - gut * (cols + 1)) / cols
    fb = fs * 0.9
    inner = colw - 2 * L.BOX_PAD - 0.02
    for c in cells:
        c["_q"] = L.fit_lines(L._question(c["q"]), colw - 0.02, fs, 4) if c.get("q") else []
        c["_box"] = [ln for part in c.get("box", "").split("\n") if part
                     for ln in L.fit_lines(part, inner, fb, 3)]
        c["_a"] = (L.tag_lines(MODEL, inner, fb) + L.fit_lines(L._reply(c["reply"]), inner, fb, 2)
                   if str(c.get("reply") or "").strip() else [])
    rows = (len(cells) + cols - 1) // cols
    # each row as tall as its own text, so a row of short questions leaves no gap
    per = [cells[r * cols:(r + 1) * cols] for r in range(rows)]
    NQ = [max(len(c["_q"]) for c in g) for g in per]
    NB = [max(len(c["_box"]) for c in g) for g in per]
    NA = [max(len(c["_a"]) for c in g) for g in per]

    def text_h(r):
        nq, nb, na = NQ[r], NB[r], NA[r]
        if not (nq or nb or na):
            return 0.0
        return (lh * nq + 0.9 * lh * (nb + na) + 2 * L.BOX_PAD * ((nb > 0) + (na > 0))
                + 0.08 * ((nb > 0) + (na > 0)) + 0.6 * lh)
    row_hs = [colw + text_h(r) + 0.12 for r in range(rows)]
    tops = np.cumsum([0.0] + row_hs[:-1])
    H = sum(row_hs) + 0.1
    fig = plt.figure(figsize=(W, H))
    fig.set_layout_engine("none")
    for j, c in enumerate(cells):
        r, cc = divmod(j, cols)
        nq, nb, na = NQ[r], NB[r], NA[r]
        x = (lab_w + gut + cc * (colw + gut)) / W
        top = 1 - (0.05 + tops[r]) / H
        ax = fig.add_axes([x, top - colw / H, colw / W, colw / H])
        _overlay(ax, c["rec"], c.get("key", "v_write"))
        y = top - colw / H - 0.3 * lh / H
        if c["_q"]:
            fig.text(x, y, "\n".join(c["_q"]), ha="left", va="top", color=st.INK, fontsize=fs,
                     linespacing=1.3)
            y -= (nq + 0.35) * lh / H
        if nb:
            hb = L.panel_box(fig, x, y, colw - 0.02, nb, lh * 0.9, fb, c["_box"], "set")
            y -= (hb + 0.08) / H
        if na and c["_a"]:
            L.panel_box(fig, x, y, colw - 0.02, na, lh * 0.9, fb, c["_a"], "model")
        if row_labels and cc == 0:
            fig.text((lab_w - 0.05) / W, top - colw / (2 * H), row_labels[r], ha="right",
                     va="center", color=st.INK, fontsize=fs, rotation=90)
    from matplotlib.patches import Patch
    st.key_band(fig, [Patch(facecolor=st.LENS, alpha=0.42, edgecolor=st.LENS),
                      Patch(facecolor=st.SLATE, alpha=0.42, edgecolor=st.SLATE)],
                ["V-Write's top tenth, holding the answer up", "its bottom tenth, pushing against it"],
                fontsize=fs, handlelength=1.2)
    return st.save(fig, path, check=False, tight=True)


def map_questions(results="results", out="figures", model=None):
    """MAIN. One photograph, two questions: V-Write puts the answer in different places on the same pixels.

    A saliency map computed from the picture alone would look the same for
    both questions. V-Write's does not, because it is the path-integrated
    gradient of the answer to the question actually asked, taken over the
    image-token states in the language model at its middle block (block 18 of
    36 of Qwen3-VL-8B-Instruct), integrated from the picture's own mean patch
    state. Each pair of columns is one photograph from the SimonSays set
    asked two of the questions the set asks of it, exactly as worded, with no
    system instruction. Clay marks the tenth of the image patches ranked
    highest (holding the answer up), slate the tenth ranked lowest (pushing
    against it); the reply is Qwen3-VL-8B-Instruct's own. The pairs shown are
    those, among the 986 cross-question photographs, whose two maps agree
    least, one per family. Over all 986 photographs, covering the fifth this
    question's map ranks highest costs the answer 2.84 on average, against
    1.15 for the other question's map and 1.02 for a random fifth
    (Qwen3-VL-8B-Instruct, zero replacement; see the cross-question
    experiment), so the difference seen here is the rule, not the exception.
    """
    if model or not P.AVERAGE:
        return None
    got = _maps(results)
    if got is None:
        return None
    M, _d = got
    ids = json.load(open(f"{results}/maps/ids.json"))
    cells = []
    for a, b in zip(ids[0:6:2], ids[1:6:2]):
        if a in M and b in M and len(cells) < 4:
            for rid in (a, b):
                r = M[rid]
                cells.append(dict(rec=r, q=r["question"], box="",
                                  reply=_reply(results, r["family"], rid, r)))
    cells = [c for c in cells if str(c.get("reply") or "").strip()]
    return _gallery(cells[:4], P._at(out, "map_questions"), cols=4)


def map_edits(results="results", out="figures", model=None):
    """MAIN. The same scene before and after an edit: V-Write's map moves to what was changed.

    Each row is one source picture from the interfaces set (VINS-120K):
    untouched, then with an official sign, a handwritten sign or a second,
    contrary sign added, each asked the question the set pairs with it, with
    no system instruction. The map is V-Write's at block 18 of 36 of
    Qwen3-VL-8B-Instruct: clay the tenth of patches ranked highest for the
    answer the model gave, slate the tenth ranked lowest. The set records
    which pixels each edit changed, so the reader can check the map against
    ground truth the dataset supplies: where a sign was added, the map should
    find it. Boxed: the edit and the answer the set expects; under it the
    model's reply.
    """
    if model or not P.AVERAGE:
        return None
    got = _maps(results)
    if got is None:
        return None
    M, _d = got
    cells = []
    for o, e, src in (("vins_orig_auth", "vins_edit_auth", "scene_00001.jpg"),
                      ("vins_orig_auth", "vins_edit_auth", "building_00001.jpg")):
        for c in L._series(results, o, e, src, ("yesno",))[:4]:
            if c["id"] in M:
                cells.append(dict(rec=M[c["id"]], q=c["q"], box=c["box"],
                                  reply=c["ans"].get(MODEL, "")))
    if len(cells) < 4:
        return None
    return _gallery(cells[:8], P._at(out, "map_edits"), cols=4)


def map_swaps(results="results", out="figures", model=None):
    """MAIN. Official against cardboard: which sign V-Write says the answer rests on, as the two trade messages.

    Two nuScenes driving scenes, each edited four ways and asked the same
    question with no system instruction: the official sign forbids and a
    cardboard one permits; the messages swapped between the two materials;
    two official signs that disagree; and official and cardboard signs that
    agree. The map is V-Write's at block 18 of 36 of Qwen3-VL-8B-Instruct
    (clay: the tenth of patches ranked highest for the answer the model
    gave; slate: the tenth ranked lowest). Boxed: the edit and the answer the
    set expects; under it the reply. The model answers no in every column;
    the maps show which sign carries that answer inside it.
    """
    if model or not P.AVERAGE:
        return None
    got = _maps(results)
    if got is None:
        return None
    M, _d = got
    order = ("conflict_main", "conflict_swapped", "both_official", "agree_control")
    g = L._family(results, MODEL, "nuscenes_edit_auth")
    if g is None:
        return None
    by: dict = {}
    for i, b in g[2].items():
        v = (b.get("meta") or {}).get("edit_variant")
        if b.get("subcategory") == "defer" and v in order and i in M:
            by.setdefault(os.path.basename(str((b.get("meta") or {}).get("source_image"))), {})[v] = b
    cells = []
    for src in sorted(by, key=lambda k: -len(by[k])):
        if len(by[src]) < len(order):
            continue
        for v in order:
            b = by[src][v]
            want = P._wanted(b)
            cells.append(dict(rec=M[b["id"]], q=b["question"],
                              box="edit: " + L.EDIT_WORDS[v] + "\n" +
                                  (f"the set expects: {want}" if want else "the set states no answer"),
                              reply=b.get("reply") or ""))
        if len(cells) >= 8:
            break
    if len(cells) < 4:
        return None
    return _gallery(cells[:8], P._at(out, "map_swaps"), cols=4)


def map_methods(results="results", out="figures", model=None):
    """MAIN. The same pictures ranked two ways: V-Write against attribution patching.

    Attribution patching (a one-point linear estimate from the same gradient,
    taken at the observed state) and V-Write (the same gradient integrated
    along the path from the picture's mean patch state) rank the same image
    patches of the same records, at block 18 of 36 of Qwen3-VL-8B-Instruct,
    for the answer the model gave. Top row V-Write, bottom row attribution
    patching; clay the fifth ranked highest, slate the tenth ranked lowest.
    The columns are the records from the causal run where the two differ
    most in what covering their top fifth does to the answer, one per
    family, and last the record where attribution patching does best, so
    the contrast is shown both ways. Questions exactly as the set words
    them, no system instruction.
    """
    if model or not P.AVERAGE:
        return None
    got = _maps(results)
    if got is None:
        return None
    M, _d = got
    ids = json.load(open(f"{results}/maps/ids.json"))
    pick = [i for i in ids[-6:] if i in M]
    if len(pick) < 3:
        return None
    top = [dict(rec=M[i], q=None, box="", reply=None, key="v_write") for i in pick]
    bot = [dict(rec=M[i], q=M[i]["question"], box="", key="attr_patch",
                reply=_reply(results, M[i]["family"], i, M[i])) for i in pick]
    return _gallery(top + bot, P._at(out, "map_methods"), cols=len(pick),
                    row_labels=["V-Write", "attribution patching"])
