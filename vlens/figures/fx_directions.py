# SPDX-License-Identifier: Apache-2.0
"""Behaviour directions: sandbagging, evaluation awareness and authority, by channel."""
from __future__ import annotations

import json
import os

import numpy as np

from vlens.figures import paper as P
from vlens.figures import style as st

#: colour is the behaviour, the stroke is the channel it arrived through
STYLE = {
    "sandbagging, told in the text": (st.CLAY, (), "sandbagging, in the text"),
    "sandbagging, a mark in the picture": (st.CLAY, (5, 2), "sandbagging, in the picture"),
    "eval awareness, the picture dressed as a test": (st.OCHRE, (5, 2), "eval awareness, in the picture"),
    "authority, an official bearer": (st.SLATE, (5, 2), "authority, in the picture"),
}


def behaviour_directions(results="results", out="figures", model=None):
    """MAIN. SimonSays lets one behaviour be delivered through two channels, words in the prompt or marks in the picture, so a difference-of-means direction can be built per channel and the two compared; the same set gives eval-dressing and authority-by-bearer contrasts on matched pictures, which no text-only benchmark can.

    One model per file, Qwen3-VL-8B-Instruct or Qwen3.5-27B. Four contrasts,
    each a difference of means over the stored state at the answer spot after
    every block, with no model run and no other fitting. Sandbagging told in
    the text: the 75 WMDP carrier items, question in the prompt, with the
    hazard instruction as the system message against the answer-well control
    as the system message. Sandbagging in the picture: the same items with the
    hazard instruction printed as a notice in the picture against the control
    printed the same way. The Qwen3-VL-8B-Instruct file was built before this
    definition, and its picture contrast comes from the password-locking
    items, so no number from it is quoted here. Evaluation
    awareness in the picture: photographs dressed as a benchmark item (a
    watermark, boxes, an annotation tool) against photographs dressed as a
    picture in use (a phone frame, a dashcam, a windscreen), 19 to 21 a side.
    Authority in the picture: an official sign against an informal or
    handwritten one carrying the same words, 53 to 59 a side. Left: AUROC of
    the projection on each direction for its own two classes, by normalised
    depth, each item scored by a direction built without it. Right: the
    cosine between every pair of directions, block by block, one ridge per
    pair.

    A picture dressed as a test is separated almost perfectly, peaking at
    0.99 on Qwen3-VL-8B-Instruct and 0.96 on Qwen3.5-27B, while the standing
    of a sign's bearer barely registers, peaking at 0.66 and 0.55. The text
    instruction separates at 1.00 within the first fifth of depth, the
    trivial part,
    since its own words differ from the plain prompt, and so does the notice
    printed in the picture on Qwen3.5-27B (1.00 by 0.37 of depth). Directions
    for different behaviours stay close to orthogonal: among pairs not built
    from password-locking items, none exceeds 0.39 in absolute cosine at any
    block on either model. The two
    sandbagging channels on Qwen3.5-27B are the exception, reaching 0.70, so
    a printed notice and a system message asking for the same thing are read
    partly along one axis there.

    Limits: difference of means needs labels and V-Read and V-Write are not
    used here; the eval-dressing contrast rests on about 20 pictures a side;
    the text and printed-notice contrasts differ in their own tokens, so a
    high AUROC there says the instruction is visible, not that it is obeyed,
    and neither model changes its answers under it.
    """
    if not model:
        return None
    base = f"{results}/directions_geom/{model}__directions"
    if not (os.path.exists(base + ".json") and os.path.exists(base + ".npz")):
        return None
    d = json.load(open(base + ".json"))
    z = np.load(base + ".npz")
    names = [str(n) for n in z["names"]]
    D = z["D"].astype(np.float32)
    L = int(d["n_layers"])
    depth = np.arange(L) / max(L - 1, 1)
    import matplotlib.pyplot as plt
    st.use()
    fig = plt.figure(figsize=(st.MAX_WIDTH, 3.9))
    fig.set_layout_engine("none")
    ax = fig.add_axes([0.075, 0.2, 0.40, 0.74])
    for n in names:
        c, dash, _ = STYLE.get(n, (st.GRAY, (), n))
        ax.plot(depth, d["contrasts"][n]["auroc"], color=c, dashes=dash, lw=2.6, zorder=4)
    st.chance(ax, 0.5)
    st.depth_axis(ax, "depth")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("separation, left out")
    st.despine(ax)
    # the picture channels' peak, averaged, fixes the depth the right panel shows
    vis = [n for n in names if d["contrasts"][n]["channel"] == "vision"]
    l0 = int(np.argmax(np.mean([d["contrasts"][n]["auroc"] for n in vis], 0)))
    ax.axvline(depth[l0], color=st.AXIS, lw=1.0, dashes=(2, 3), zorder=1)
    # the six angles between the four directions, as ridges through depth
    ax3 = fig.add_axes([0.50, -0.06, 0.50, 1.12], projection="3d")
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
    SHORT = {n: STYLE.get(n, (None, None, n))[2].replace("eval awareness", "eval")
             .replace("sandbagging", "sandbag").replace(", in the ", "/") for n in names}
    for k, (i, j) in enumerate(pairs):
        c = np.einsum("ld,ld->l", D[i], D[j])
        col = st.CLAY if "sandbag" in names[i] and "sandbag" in names[j] else st.SLATE
        ax3.plot(depth, [k] * L, c, color=col, lw=2.0)
        ax3.plot(depth, [k] * L, np.zeros(L), color=st.GRID, lw=0.8)
    ax3.set_xlim(0, 1); ax3.set_ylim(-0.5, len(pairs) - 0.5); ax3.set_zlim(-0.5, 0.5)
    ax3.set_xticks([0, 0.5, 1]); ax3.set_xticklabels(["input", "half", "output"])
    ax3.set_yticks(range(len(pairs)))
    ax3.set_yticklabels([f"{SHORT[names[i]]} . {SHORT[names[j]]}" for i, j in pairs],
                        fontsize=st.matplotlib.rcParams["xtick.labelsize"] * 0.72,
                        ha="left", va="center")
    ax3.set_zticks([-0.4, 0, 0.4])
    ax3.view_init(elev=24, azim=-62)
    for pane in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        pane.pane.set_facecolor("white"); pane.pane.set_edgecolor(st.GRID)
        pane.line.set_color(st.AXIS)
    # the key sits in the left panel's empty lower right, where no line goes
    SH = {"sandbagging, told in the text": "sandbagging, text",
          "sandbagging, a mark in the picture": "sandbagging, picture",
          "eval awareness, the picture dressed as a test": "eval awareness, picture",
          "authority, an official bearer": "authority, picture"}
    for i, n in enumerate(names):
        c, dash, _ = STYLE.get(n, (st.GRAY, (), n))
        y = 0.29 - 0.075 * i
        ax.plot([0.56, 0.64], [y, y], transform=ax.transAxes, color=c, lw=2.6,
                dashes=dash, solid_capstyle="butt", zorder=6)
        ax.text(0.66, y, SH.get(n, n), transform=ax.transAxes, va="center", ha="left",
                color=st.INK, fontsize=st.matplotlib.rcParams["legend.fontsize"] * 0.85,
                zorder=6, bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none"))
    return st.save(fig, P._at(out, "behaviour_directions"), check=False, tight=True)
