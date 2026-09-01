# SPDX-License-Identifier: Apache-2.0
"""Winoground worked-example gallery: two captions, one picture, only VJ moves.

No model, no GPU.  Every map is read out of
results/winoground/Qwen3-VL-8B_cases.json (the 8B run: larger margins than the
2B checkpoint, +0.34 vs +0.19 mean agreement and 43% vs 11% top-1), which
stores, for each item, each lens's per-region reading of the SAME image under
the item's two swapped captions.  The 8B run did not fit a J-lens, so this
sheet compares the two lenses it has: the logit lens reads image tokens the
caption never touched, so its two readings are byte-identical (movement 0.0
in the run's own maps); VJ-Lens differentiates the live answer logit, so its
reading moves with the caption.  [caption note: the J-lens column that the
2B version of this figure carried is gone -- the 8B run has no J-lens fit --
so paper/atlas.html's alt text for f131_wino_cards.png needs a matching edit.]
"""
import sys, os, json, textwrap
sys.path.insert(0, "/workspace/vlm/code"); sys.path.insert(0, "/workspace/vlm/code/exp")
import numpy as np
from jvlm import style
style.use(1.0)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from _gallov import load_rgb, ramp

d = json.load(open("/workspace/vlm/results/winoground/Qwen3-VL-8B_cases.json"))
by = {c["item"]: c for c in d["cases"]}
HAS_J = "J-lens" in d["lenses"]
# the strongest, most legible VJ-Lens moves among the 12 items, one relational
# and two object-binding, all three answered correctly under both captions
ITEMS = [4, 8, 10]
L = 16
out = "/workspace/vlm/figures/f131_wino_cards.png"
grid = d["coarse"]

def diverge():
    ra,ga,ba=to_rgb(style.BLUE); rb,gb,bb=to_rgb(style.RED)
    return LinearSegmentedColormap.from_list("d",
        [(rb,gb,bb,0.9),(rb,gb,bb,0.28),(1,1,1,0.0),(ra,ga,ba,0.28),(ra,ga,ba,0.9)])

def heat(ax, img, m, hue, title, q=0.45, vmax=None):
    ax.imshow(img, extent=[0,1,0,1], origin="upper", interpolation="lanczos")
    a=np.array(m,float).reshape(grid,grid); a=a-a.min()
    mx=vmax if vmax is not None else a.max()
    if mx>0: a=a/mx
    a=np.clip((a-q)/(1-q),0,1)
    ax.imshow(a, extent=[0,1,0,1], origin="upper", cmap=ramp(hue), vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title, loc="left", color=style.INK, fontsize=15, pad=4)

def heat_diff(ax, img, a, b, title):
    ax.imshow(img, extent=[0,1,0,1], origin="upper", interpolation="lanczos")
    a=np.array(a,float).reshape(grid,grid); b=np.array(b,float).reshape(grid,grid)
    dif=a-b; s=np.abs(dif).max() or 1.0; dif=dif/s
    ax.imshow(dif, extent=[0,1,0,1], origin="upper", cmap=diverge(), vmin=-1, vmax=1, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for s2 in ax.spines.values(): s2.set_visible(False)
    ax.set_title(title, loc="left", color=style.INK, fontsize=15, pad=4)

ncol = 6 if HAS_J else 5
fig = plt.figure(figsize=(15.5 * ncol / 6.0, 3.05*len(ITEMS)+0.55))
gs = fig.add_gridspec(len(ITEMS), ncol, wspace=0.06, hspace=0.30, left=0.008, right=0.995, top=0.925, bottom=0.02)
for r, it in enumerate(ITEMS):
    c=by[it]; img=load_rgb(c["paths"][0], 392); caps=c["captions"]; beh=c["behaviour"]
    ax=fig.add_subplot(gs[r,0]); ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title("image", loc="left", fontsize=15, color=style.INK, pad=4)
    axt=fig.add_subplot(gs[r,1]); axt.axis("off")
    # a wrapped caption can run to 3 lines; every label below it is placed
    # from the actual line count, not a fixed offset, or a long caption
    # collides with the label under it
    LINE_H = 0.125
    axt.text(0.0,0.93,"prompt A", fontsize=14.5, color=style.BLUE, va="top")
    qa = textwrap.fill('"'+caps[0]+'"', 25)
    axt.text(0.0,0.82,qa, fontsize=15, color=style.INK, va="top", linespacing=1.35)
    yB = 0.82 - (qa.count("\n")+1)*LINE_H - 0.05
    axt.text(0.0,yB,"prompt B", fontsize=14.5, color=style.RED, va="top")
    qb = textwrap.fill('"'+caps[1]+'"', 25)
    y_qb = yB - 0.11
    axt.text(0.0,y_qb,qb, fontsize=15, color=style.INK, va="top", linespacing=1.35)
    y_ans = y_qb - (qb.count("\n")+1)*LINE_H - 0.05
    axt.text(0.0,y_ans,"model answers both correctly", fontsize=14, color=style.INK2, va="top")
    col = 2
    heat(fig.add_subplot(gs[r,col]), img, c["maps"][f'logit lens|L{L}|img0|cap0|margin'], style.ORANGE, "logit lens  (A = B)")
    col += 1
    if HAS_J:
        heat(fig.add_subplot(gs[r,col]), img, c["maps"][f'J-lens|L{L}|img0|cap0|margin'], style.VIOLET, "J-lens  (A = B)")
        col += 1
    v0=c["maps"][f'VJ-Lens|L{L}|img0|cap0|margin']; v1=c["maps"][f'VJ-Lens|L{L}|img0|cap1|margin']
    heat(fig.add_subplot(gs[r,col]), img, v0, style.BLUE, "VJ-Lens | A", vmax=max(max(v0)-min(v0), max(v1)-min(v1)))
    col += 1
    heat_diff(fig.add_subplot(gs[r,col]), img, v0, v1, "VJ-Lens shift  (blue A, red B)")
fig.text(0.5, 0.982, "Two captions, one picture: only VJ-Lens moves with the prompt", fontsize=19, color=style.INK, va="top", ha="center")
fig.savefig(out, dpi=200); print("wrote", out)
