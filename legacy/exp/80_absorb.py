"""How deep does the answer keep consulting the pixels -- and does the question decide?

The layerwise causal map covers every region at every depth under one prompt.
That gives the *absorption curve*: the total causal dependence of the answer on
the picture, against depth.  It should fall, because the image tokens are read
by the text positions and, once read, are no longer needed.

The claim worth testing is the next one.  If the depth at which the model stops
consulting the pixels is a property of the *question* rather than of the
picture, then question difficulty has a depth, and it is measurable.  So this
run asks the same photograph two questions:

    easy   what is the main object                -- one glance settles it
    hard   how many of them are there             -- every instance must be found

and runs the identical covering sweep for each: same regions, same layers, same
per-position corpus-mean baseline.  Only the words after the image differ.

The baseline is shared between the two questions on purpose.  Image tokens
precede text under a causal mask, so the residual stream at an image position
cannot depend on the question -- the run checks that and records the residual,
which is the same fact the prompt-dependence proof rests on.  What changes with
the question is not what is *there*, it is what the answer position *takes*.

The measured quantity per layer is a sum over regions of the drop in the
answered token's logit when that region's activations are replaced by the
average photograph's.  It is in logits, and it is signed, so the run records
both the raw total and the part that is specific to *which* region was covered
-- the mean-image baseline pulls the answer the same way wherever it is
applied, and that uniform pull is not a dependence on any place in the picture.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, "code")
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import torch

import jvlm
from jlens.hooks import ActivationRecorder
from jvlm import cocodata, corpus, readout, stats as st

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--mean-images", type=int, default=60)
ap.add_argument("--n-layers", type=int, default=12)
ap.add_argument("--pool", type=int, default=2)
ap.add_argument("--chunk", type=int, default=12)
ap.add_argument("--min-count", type=int, default=2)
ap.add_argument("--max-count", type=int, default=8)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/absorb")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
LAYERS = [int(round(f * (m.n_layers - 1)))
          for f in np.linspace(0.20, 0.85, a.n_layers)]
CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
print(m, f"\nlayers {LAYERS}", flush=True)

EASY = "What is the main object in this photograph? Answer with one word."


def plural(w: str) -> str:
    return w + "es" if w.endswith(("s", "x", "ch", "sh")) else w + "s"


def hard_q(c: str) -> str:
    return (f"How many {plural(c)} are in this photograph? "
            f"Answer with a number.")


def prep(path, prompt=EASY):
    return m.chat(prompt, [corpus.load_image(path, square=a.square)])


# ---------------------------------------------------------------- the baseline
pool_items = cocodata.concept_items(6000, categories=CATS, min_area=0.04, seed=7)
b0 = prep(pool_items[0]["path"])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
C = max(2, G // a.pool)
acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
nrm = {l: torch.zeros(P, dtype=torch.float64) for l in LAYERS}
nm = 0
for it in pool_items[: a.mean_images]:
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    A = readout.residuals(m, b, LAYERS)
    for l in LAYERS:
        v = A[l][ip].float().cpu()
        acc[l] += v.double()
        nrm[l] += v.norm(dim=-1).double()
    nm += 1
MEANS = {l: (acc[l] / max(nm, 1)).float() for l in LAYERS}
NORM = {l: (nrm[l] / max(nm, 1)).float() for l in LAYERS}
REG = (NORM[LAYERS[0]] > NORM[LAYERS[0]].mean() + 3 * NORM[LAYERS[0]].std())

side = int(np.ceil(G / C))
REGIONS, KEEP = [], []
for ry in range(C):
    for rx in range(C):
        idx = [y * G + x
               for y in range(ry * side, min((ry + 1) * side, G))
               for x in range(rx * side, min((rx + 1) * side, G))
               if not bool(REG[y * G + x])]
        if idx:
            REGIONS.append(torch.tensor(idx))
            KEEP.append(ry * C + rx)
R = len(REGIONS)
print(f"{P} tokens on {G}x{G}; {R} regions of about {side*side} patches; "
      f"means from {nm} ({time.time()-t0:.0f}s)", flush=True)


def answered(b):
    """The token the model is actually about to emit, and its logit."""
    A = readout.residuals(m, b, [m.n_layers - 1])
    lg = m.unembed(A[m.n_layers - 1][-1:].to(m.device))[0].float().cpu()
    t = int(lg.argmax())
    return t, float(lg[t])


def sweep(b, tok, base, ip):
    """Cover each region at each layer; return [L, R] drops in ``tok``'s logit."""
    out = np.zeros((len(LAYERS), R))
    APOS = b["input_ids"].shape[1] - 1
    for li, l in enumerate(LAYERS):
        for s0 in range(0, R, a.chunk):
            grp = REGIONS[s0:s0 + a.chunk]
            big = m.replicate(b, len(grp))

            def hook(mod, inp, o, _g=grp, _l=l):
                tup = not torch.is_tensor(o)
                hh = (o[0] if tup else o).clone()
                for j, r in enumerate(_g):
                    hh[j, ip[r].to(hh.device)] = \
                        MEANS[_l][r].to(hh.device).to(hh.dtype)
                return (hh, *o[1:]) if tup else hh

            hd = m.layers[l].register_forward_hook(hook)
            try:
                with torch.no_grad(), ActivationRecorder(
                        m.layers, at=[m.n_layers - 1]) as rec:
                    m.forward(big)
                    hv = rec.activations[m.n_layers - 1][:, APOS]
                    vv = m.unembed(hv)[:, tok].float().cpu()
            finally:
                hd.remove()
            out[li, s0:s0 + len(grp)] = base - vv.numpy()
    return out


items = cocodata.concept_items(a.n * 30, categories=CATS, min_area=0.06,
                               seed=808, tokenizer=m.tokenizer)
cards, kept, invar = [], 0, []
for k, it in enumerate(items):
    if kept >= a.n:
        break
    c = it["concept"]
    boxes = cocodata.object_boxes(it["image_id"], c)
    # counting has to be a real question: one instance makes the hard prompt as
    # easy as the easy one, and a crowd makes it unanswerable rather than hard
    if not (a.min_count <= len(boxes) <= a.max_count):
        continue
    inside = sorted({t for bx in boxes for t in cocodata.box_tokens(bx, C)})
    b = prep(it["path"], EASY)
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    te, be = answered(b)
    # the easy question has to actually be answered with the object's name, or
    # the two questions are not about the same thing
    if te not in set(cocodata.concept_token_ids(m.tokenizer, c)):
        continue
    bh = prep(it["path"], hard_q(c))
    th, bh_logit = answered(bh)
    word_h = m.tokenizer.decode([th]).strip()
    if not word_h[:1].isdigit():
        continue

    # the fact the whole prompt-dependence argument rests on, measured here for
    # free: the residual stream at an image position cannot know the question
    Ae = readout.residuals(m, b, LAYERS)
    Ah = readout.residuals(m, bh, LAYERS)
    iph = readout.image_positions(m, bh)
    d = max(float((Ae[l][ip] - Ah[l][iph]).abs().max()) for l in LAYERS)
    s = max(float(Ae[l][ip].abs().max()) for l in LAYERS)
    invar.append(d / max(s, 1e-9))

    kept += 1
    rec = dict(image_id=it["image_id"], concept=c, path=it["path"],
               n_instances=len(boxes), coarse=C, keep=KEEP, inside=inside,
               layers=LAYERS, depths=[round(l / (m.n_layers - 1), 3)
                                      for l in LAYERS],
               questions={})
    for nm_, (bb, tok, base, q) in {
            "easy": (b, te, be, EASY),
            "hard": (bh, th, bh_logit, hard_q(c))}.items():
        ipx = readout.image_positions(m, bb)
        T = sweep(bb, tok, base, ipx)
        rec["questions"][nm_] = dict(
            prompt=q, word=m.tokenizer.decode([tok]).strip(), token=tok,
            base=round(base, 4), truth=np.round(T, 4).tolist())
    cards.append(rec)
    print(f"  {kept}/{a.n}  {it['image_id']} {c} x{len(boxes)}  "
          f"easy '{m.tokenizer.decode([te]).strip()}'  hard '{word_h}'  "
          f"{time.time()-t0:.0f}s", flush=True)
    json.dump(st.jsonable(cards), open(f"{a.out}/{tag}_cards.json", "w"))

json.dump(st.jsonable(cards), open(f"{a.out}/{tag}_cards.json", "w"))


def curves(T):
    """Raw total, place-specific total, and the uniform pull, per layer."""
    med = np.median(T, axis=1, keepdims=True)
    return (np.abs(T).sum(1), np.abs(T - med).sum(1), np.abs(med[:, 0]) * T.shape[1])


summ = {}
for nm_ in ("easy", "hard"):
    tot, spec, unif = [], [], []
    for r in cards:
        T = np.asarray(r["questions"][nm_]["truth"])
        x, y, z = curves(T)
        tot.append(x); spec.append(y); unif.append(z)
    summ[nm_] = dict(total=np.mean(tot, 0).round(4).tolist(),
                     specific=np.mean(spec, 0).round(4).tolist(),
                     uniform=np.mean(unif, 0).round(4).tolist())


def centroid(y, d):
    y = np.clip(np.asarray(y, float), 0, None)
    return float((y * d).sum() / y.sum()) if y.sum() > 0 else float("nan")


D = np.array(cards[0]["depths"]) if cards else np.array([])
pairs = []
for r in cards:
    ce = centroid(curves(np.asarray(r["questions"]["easy"]["truth"]))[1], D)
    ch = centroid(curves(np.asarray(r["questions"]["hard"]["truth"]))[1], D)
    pairs.append(dict(image_id=r["image_id"], concept=r["concept"],
                      n_instances=r["n_instances"],
                      easy=round(ce, 4), hard=round(ch, 4),
                      shift=round(ch - ce, 4)))
sh = np.array([p["shift"] for p in pairs]) if pairs else np.array([])
out = dict(repo=a.repo, n=len(cards), grid=G, coarse=C, regions=R,
           layers=LAYERS, n_layers=m.n_layers,
           depths=D.round(3).tolist(),
           easy_prompt=EASY, hard_prompt=hard_q("<object>"),
           mean_images=nm,
           image_activation_relative_difference=float(np.max(invar))
           if invar else None,
           by_question=summ, pairs=pairs,
           shift_mean=float(sh.mean()) if len(sh) else None,
           shift_positive=int((sh > 0).sum()) if len(sh) else None,
           minutes=round((time.time() - t0) / 60, 1))
json.dump(st.jsonable(out), open(f"{a.out}/{tag}_stats.json", "w"), indent=2,
          default=float)
print(json.dumps({k: v for k, v in out.items()
                  if k not in ("by_question", "pairs")}, indent=2), flush=True)
print("done", flush=True)
