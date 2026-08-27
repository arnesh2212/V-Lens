"""A ground truth the lens has never seen: occlude the pixels, not the activations.

The causal map in exp/60 covers a region by replacing its *activations* with the
per-position corpus mean -- which is the exact point this lens integrates from.
By the completeness property the reading is the quadrature approximation of that
very intervention, so a reviewer is entitled to say the comparison is partly won
by construction: the ruler was built out of the thing being measured.

This removes that objection.  The region is occluded in the *image*: the
corresponding block of pixels is replaced with grey and the photograph is
re-encoded from scratch, so the vision tower runs again and nothing about the
lens's baseline is involved.  Every lens is then graded against that.

It also disambiguates the paper's most interesting side-finding.  "The causally
dominant region falls inside the human's box less often than chance" currently
conflates two claims: that context genuinely carries the evidence, or that the
object's information has already migrated out of its own tokens by the layer we
read.  Activation-covering at layer l cannot tell those apart; pixel occlusion
can, because it removes the object before the model has seen it at all.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, "/workspace/vlm/code")
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats as sps

import jvlm
from jvlm import cocodata, corpus, readout, stats as st, vmap

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--jlens", default=None)
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--mean-images", type=int, default=70)
ap.add_argument("--n-layers", type=int, default=2)
ap.add_argument("--pool", type=int, default=2)
ap.add_argument("--ig-steps", type=int, default=4)
ap.add_argument("--vocab", type=int, default=30)
ap.add_argument("--max-rank", type=int, default=2)
ap.add_argument("--fill", default="grey", choices=["grey", "blur"])
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/pixeltruth")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
LAYERS = [int(round(f * (m.n_layers - 1)))
          for f in np.linspace(0.30, 0.80, a.n_layers)]
J = None
if a.jlens:
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))
CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
print(m, f"\nlayers {LAYERS}   occluding with {a.fill}", flush=True)

PROMPT = "Describe this image."


def prep_img(im):
    return m.chat(PROMPT, [im])


pool_items = cocodata.concept_items(6000, categories=CATS, min_area=0.04, seed=7)
b0 = prep_img(corpus.load_image(pool_items[0]["path"], square=a.square))
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
C = max(2, G // a.pool)
side_px = a.square // C
acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
nm = 0
for it in pool_items[: a.mean_images]:
    b = prep_img(corpus.load_image(it["path"], square=a.square))
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    A = readout.residuals(m, b, LAYERS)
    for l in LAYERS:
        acc[l] += A[l][ip].float().cpu().double()
    nm += 1
MEANS = {l: (acc[l] / max(nm, 1)).float() for l in LAYERS}
print(f"{P} tokens on {G}x{G}; {C}x{C} regions of {side_px}x{side_px} pixels; "
      f"means from {nm} ({time.time()-t0:.0f}s)", flush=True)

side = int(np.ceil(G / C))
REGIONS, KEEP = [], []
for ry in range(C):
    for rx in range(C):
        idx = [y * G + x
               for y in range(ry * side, min((ry + 1) * side, G))
               for x in range(rx * side, min((rx + 1) * side, G))]
        if idx:
            REGIONS.append(torch.tensor(idx))
            KEEP.append(ry * C + rx)
R = len(REGIONS)


def occlude(im, cell):
    """Grey out one C x C block of the photograph and hand back a new image."""
    ry, rx = divmod(int(cell), C)
    out = im.copy()
    box = (rx * side_px, ry * side_px, (rx + 1) * side_px, (ry + 1) * side_px)
    if a.fill == "blur":
        from PIL import ImageFilter
        patch = out.crop(box).filter(ImageFilter.GaussianBlur(side_px / 4))
        out.paste(patch, box)
    else:
        out.paste((128, 128, 128), box)
    return out


def region_sum(per_patch):
    return torch.stack([per_patch[:, r].sum(-1) for r in REGIONS], dim=-1)


items = cocodata.concept_items(a.n * 14, categories=CATS, min_area=0.06,
                               seed=808, tokenizer=m.tokenizer)
rows, kept = [], 0
for k, it in enumerate(items):
    if kept >= a.n:
        break
    c = it["concept"]
    if c not in CAT_IDS:
        continue
    boxes = cocodata.object_boxes(it["image_id"], c)
    if not boxes:
        continue
    inside = sorted({t for bx in boxes for t in cocodata.box_tokens(bx, C)})
    if not (2 <= len(inside) <= 0.6 * C * C):
        continue
    im = corpus.load_image(it["path"], square=a.square)
    b = prep_img(im)
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    rng = np.random.default_rng(900 + k)
    others = [x for x in CATS if x != c]
    pick = list(rng.choice(others, size=min(a.vocab - 1, len(others)),
                           replace=False))
    ids = CAT_IDS[c]
    VOC = sorted({t for w in pick for t in CAT_IDS[w]} | set(ids))
    VPOS = {t: i for i, t in enumerate(VOC)}
    col = [VPOS[t] for t in ids]
    A = readout.residuals(m, b, LAYERS + [m.n_layers - 1])
    lg0 = m.unembed(A[m.n_layers - 1][-1:].to(m.device))[0][VOC].float().cpu()
    if int((lg0 > lg0[col].max()).sum()) > a.max_rank:
        continue
    base = float(lg0[col].max())
    kept += 1

    # ---- the ground truth: occlude the PIXELS and re-encode
    truth = np.zeros(R)
    for j, cell in enumerate(KEEP):
        bb = prep_img(occlude(im, cell))
        with torch.no_grad():
            AA = readout.residuals(m, bb, [m.n_layers - 1])
            v = m.unembed(AA[m.n_layers - 1][-1:].to(m.device)
                          )[0][VOC].float().cpu()
        truth[j] = base - float(v[col].max())

    tru_arg = int(np.argmax(truth))
    tru_top3 = set(np.argsort(-truth)[:3].tolist())
    box = set(inside)
    for l in LAYERS:
        h = A[l][ip].float().cpu()
        maps = {}
        nrm = torch.tensor([len(r) for r in REGIONS], dtype=torch.float32)
        maps["logit lens"] = region_sum(
            m.unembed(h.to(m.device)).float().cpu()[:, VOC].T) / nrm
        if J is not None and l in J.jacobians:
            jl, _ = readout.lens_logits(m, J, b, layers=[l],
                                        positions=ip.tolist())
            maps["J-lens"] = region_sum(
                jl[l].float().cpu()[:, VOC].T) / nrm
        maps["VJ-Lens"] = region_sum(
            vmap.ig_map(m, b, VOC, [l], ip.tolist(), means={l: MEANS[l]},
                        steps=a.ig_steps)[l])
        for name, mm in maps.items():
            score = mm[col].max(0).values.float().numpy()
            rows.append(dict(
                i=k, concept=c, lens=name, cond=name, key=f"{k}_{l}", layer=l,
                depth=round(l / (m.n_layers - 1), 3),
                spearman=float(sps.spearmanr(score, truth).statistic),
                hit_cause=int(int(np.argmax(score)) == tru_arg),
                top3_cause=int(int(np.argmax(score)) in tru_top3),
                hit_box=int(KEEP[int(np.argmax(score))] in box),
                truth_in_box=int(KEEP[tru_arg] in box),
                box_area=len(inside) / max(R, 1), R=R))
    if kept % 3 == 0:
        el = time.time() - t0
        print(f"  {kept}/{a.n} kept  {el:.0f}s  "
              f"eta {el/max(kept,1)*(a.n-kept)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
agg = df.groupby(["lens", "layer"]).agg(
    n=("i", "count"), spearman=("spearman", "mean"),
    hit_cause=("hit_cause", "mean"), top3_cause=("top3_cause", "mean"),
    hit_box=("hit_box", "mean")).reset_index()
print("\nAGAINST PIXEL OCCLUSION (the lens has never seen this intervention)\n",
      agg.round(4).to_string(index=False), flush=True)
over = df.groupby("lens").agg(
    spearman=("spearman", "mean"), hit_cause=("hit_cause", "mean"),
    top3_cause=("top3_cause", "mean"), hit_box=("hit_box", "mean"))
print("\nOVER ALL LAYERS\n", over.round(4).to_string(), flush=True)
print(f"\nthe most damaged region is inside the human's box "
      f"{df.truth_in_box.mean():.3f} of the time, on boxes covering "
      f"{df.box_area.mean():.3f} of the grid", flush=True)

out = {"repo": a.repo, "n": int(df.i.nunique()), "regions": R, "grid": G,
       "coarse": C, "fill": a.fill, "layers": LAYERS,
       "chance_cause": round(1.0 / R, 4),
       "truth_in_box": round(float(df.truth_in_box.mean()), 4),
       "box_area": round(float(df.box_area.mean()), 4),
       "by_layer": agg.round(4).to_dict("records"),
       "over": over.round(4).reset_index().to_dict("records"),
       "minutes": round((time.time() - t0) / 60, 1)}
for other in [x for x in df.lens.unique() if x != "VJ-Lens"]:
    out[f"VJ-Lens_vs_{other}"] = st.wilcoxon(df, other, "VJ-Lens", "spearman",
                                             item="key")
json.dump(st.jsonable(out), open(f"{a.out}/{tag}_stats.json", "w"), indent=2,
          default=float)
print("done", flush=True)
