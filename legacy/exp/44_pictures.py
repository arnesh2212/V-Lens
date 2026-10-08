"""The two maps, drawn on the photographs.

Numbers about a lens are hard to argue with and hard to believe.  This draws
what the lens actually sees: the picture, the object's box, and the two maps
over it -- what each patch would say, and what each patch is contributing to the
answer -- at three depths.

Emitted twice: as a PNG grid for the paper, and as a compact JSON of small
WebP images plus the words, so the same examples can be browsed in the artifact.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import time

sys.path.insert(0, "code")
for n in ("httpx", "httpcore", "urllib3", "filelock", "matplotlib"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jvlm
from jvlm import cocodata, corpus, readout, style, tasks, vmap

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--mean-images", type=int, default=120)
ap.add_argument("--depths", type=float, nargs="*", default=[0.2, 0.5, 0.8])
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--figs", type=int, default=8, help="photographs per PNG sheet")
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/pictures")
ap.add_argument("--figdir", default="figures")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
os.makedirs(a.figdir, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()
style.use()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
LAYERS = sorted({int(round(max(m.n_deepstack, 1)
                           + (m.n_layers - 2 - max(m.n_deepstack, 1)) * f))
                 for f in a.depths})
CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
FLAT = sorted({t for v in CAT_IDS.values() for t in v})
POSI = {t: i for i, t in enumerate(FLAT)}
sc = tasks.Scorer(m)
print(m, f"\nlayers {LAYERS}", flush=True)

PROMPT = "Describe this image."


def prep(path):
    return m.chat(PROMPT, [corpus.load_image(path, square=a.square)])


pool = cocodata.concept_items(6000, min_area=0.04, seed=7)
b0 = prep(pool[0]["path"])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
nrm = {l: torch.zeros(P, dtype=torch.float64) for l in LAYERS}
bank = {l: [] for l in LAYERS}
nm = 0
for it in pool[: a.mean_images]:
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    A = readout.residuals(m, b, LAYERS)
    for l in LAYERS:
        v = A[l][ip].float().cpu()
        acc[l] += v.double(); nrm[l] += v.norm(dim=-1).double()
        if len(bank[l]) < 16:
            bank[l].append(v)
    nm += 1
MEANS = {l: (acc[l] / max(nm, 1)).float() for l in LAYERS}
NORM = {l: (nrm[l] / max(nm, 1)).float() for l in LAYERS}
REG = {l: (NORM[l] > NORM[l].mean() + 3 * NORM[l].std()) for l in LAYERS}
print(f"means from {nm} photographs ({time.time()-t0:.0f}s)", flush=True)


def words(vec, k=3):
    out, seen = [], set()
    for j in vec.argsort(descending=True).tolist():
        w = m.tokenizer.decode([FLAT[j]]).strip()
        if not w or w.lower() in seen:
            continue
        seen.add(w.lower()); out.append(w)
        if len(out) == k:
            break
    return out


def webp(im, size=260, q=80):
    q_ = im.copy(); q_.thumbnail((size, size), Image.LANCZOS)
    b = io.BytesIO(); q_.save(b, "WEBP", quality=q, method=4)
    return "data:image/webp;base64," + base64.b64encode(b.getvalue()).decode()


items = cocodata.concept_items(a.n * 3, min_area=0.08, seed=99,
                               tokenizer=m.tokenizer)
cards = []
for k, it in enumerate(items):
    if len(cards) >= a.n:
        break
    c = it["concept"]
    boxes = cocodata.object_boxes(it["image_id"], c)
    if not boxes:
        continue
    inside = sorted({t for bx in boxes for t in cocodata.box_tokens(bx, G)})
    if not (3 <= len(inside) <= P // 2):
        continue
    im = corpus.load_image(it["path"], square=a.square)
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    ids = CAT_IDS[c]
    A = readout.residuals(m, b, LAYERS)
    donor = bank[LAYERS[0]][k % len(bank[LAYERS[0]])]
    card = {"concept": c, "img": webp(im), "boxes": boxes,
            "caption": sc.generate(b, 36).strip(), "grid": G,
            "objects": sorted(it["present"]), "layers": {}}
    for l in LAYERS:
        h = A[l][ip].float().cpu()
        lg = m.unembed(h.to(m.device)).float().cpu()[:, FLAT]
        lg = lg - lg.mean(1, keepdim=True)
        say = lg[:, [POSI[t] for t in ids]].max(1).values
        do = vmap.do_map(m, b, ids, [l], ip.tolist(),
                         means={l: bank[l][k % len(bank[l])]})[l].sum(0)
        for v in (say, do):
            v[REG[l]] = v[~REG[l]].min() if (~REG[l]).any() else 0.0
        card["layers"][str(l)] = {
            "depth": round(l / (m.n_layers - 1), 3),
            "say": [round(x, 3) for x in say.tolist()],
            "do": [round(x, 3) for x in do.tolist()],
            "say_words": words(lg.max(0).values),
            "do_words": words(vmap.do_map(m, b, FLAT, [l], ip.tolist(),
                                          means={l: bank[l][k % len(bank[l])]}
                                          )[l].max(1).values)
            if len(FLAT) <= 64 else [],
        }
    cards.append(card)
    if k % 10 == 0:
        print(f"  {len(cards)}/{a.n}  {time.time()-t0:.0f}s", flush=True)

json.dump({"repo": a.repo, "grid": G, "layers": LAYERS,
           "n_layers": m.n_layers, "cards": cards},
          open(f"{a.out}/{tag}_pictures.json", "w"))
print(f"wrote {len(cards)} cards", flush=True)

# ---- PNG sheets --------------------------------------------------------------
def sheet(group, idx):
    rows = len(group)
    cols = 1 + 2 * len(LAYERS)
    fig, axes = plt.subplots(rows, cols, figsize=(1.55 * cols, 1.62 * rows),
                             squeeze=False)
    for r, card in enumerate(group):
        img = Image.open(io.BytesIO(base64.b64decode(card["img"].split(",")[1])))
        ax = axes[r][0]
        ax.imshow(img)
        for (x0, y0, w, h) in card["boxes"]:
            ax.add_patch(plt.Rectangle((x0 * img.width, y0 * img.height),
                                       w * img.width, h * img.height,
                                       fill=False, lw=1.4,
                                       edgecolor=style.C_WORD))
        ax.set_ylabel(card["concept"], fontsize=8, rotation=0, ha="right",
                      va="center", labelpad=6, color=style.INK2)
        for cc, (head, cmap) in enumerate((("say", "Oranges"),
                                           ("do", "Blues"))):
            for li, l in enumerate(LAYERS):
                ax2 = axes[r][1 + cc * len(LAYERS) + li]
                hm = np.asarray(card["layers"][str(l)][head]).reshape(G, G)
                ax2.imshow(img, alpha=0.55)
                ax2.imshow(np.kron(hm, np.ones((img.height // G,
                                                img.width // G))),
                           cmap=cmap, alpha=0.55,
                           extent=(0, img.width, img.height, 0))
                if r == 0:
                    ax2.set_title(f"{head}  {int(card['layers'][str(l)]['depth']*100)}%",
                                  fontsize=7.5, color=style.INK2, pad=3)
    for row in axes:
        for ax in row:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.94, bottom=0.005,
                        wspace=0.04, hspace=0.04)
    p = f"{a.figdir}/f30_pictures_{idx}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p, flush=True)


for i in range(0, min(len(cards), a.figs * 3), a.figs):
    sheet(cards[i: i + a.figs], i // a.figs)
print("done", flush=True)
