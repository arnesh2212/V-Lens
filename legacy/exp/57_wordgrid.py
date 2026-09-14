"""What word is each part of the picture reading?

Every other comparison in this paper is a number.  This one is the picture the
numbers are about, and it is meant to be read rather than admired: the token
grid is drawn as discrete labelled tiles, one word printed in each, so a reader
can check the claim tile by tile instead of trusting a colour wash.

Three lenses answer the same question at the same place.  Fix a vocabulary of
~200 everyday visual words.  For a region R of image patches at layer l:

    logit lens   argmax_t  W_U[t] . norm( mean_{p in R} h_l[p] )
    J-lens       argmax_t  W_U[t] . norm( J_l  mean_{p in R} h_l[p] )
    VJ-Lens      argmax_t  mean_{p in R}  d l_t / d h_l[p] . (h_l[p] - hbar)

The region reading is the mean of the patches' own readings rather than the
mean gradient dotted with the mean deviation.  The two coincide only when the
gradient is constant inside the region; the first is the one that adds up, so a
region's reading is the sum of its patches' however the region is drawn, and it
is what makes pooling a use of the region coordinate rather than a blur.  Both
are computed and the other is kept under ``--variants``.

The three differ only in the transport between the region and the vocabulary:
the identity, one fitted matrix per layer, and the answer's own Jacobian at that
region.  Region pooling is not cosmetic -- R is the coordinate the method adds,
so pooling a 14x14 token grid into 7x7 readable tiles is the method being used,
not a blur applied to make a figure fit.

Every reading is z-scored down the vocabulary at its own region, so the three
are compared as *which word is this place pushing hardest*, and no lens gets an
advantage from having larger numbers.

Two numbers per grid say what the picture says:

    distinct   how many different words appear across the 49 tiles
    inside     the fraction of tiles covered by the object's box whose word is
               the object's own name  (with the outside-the-box rate beside it,
               so a lens that prints the name everywhere cannot look good)
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
from collections import Counter

sys.path.insert(0, "/workspace/vlm/code")
for n in ("httpx", "httpcore", "urllib3", "filelock", "matplotlib"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch
from PIL import Image

import jvlm
from jvlm import cocodata, corpus, readout, stats as st, tasks, vmap
from jlens.hooks import ActivationRecorder

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--jlens", default=None)
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--mean-images", type=int, default=80)
ap.add_argument("--n-layers", type=int, default=4)
ap.add_argument("--depths", type=float, nargs="*",
                default=[0.25, 0.45, 0.65, 0.85])
ap.add_argument("--pool", type=int, default=2, help="token block side")
ap.add_argument("--unit", action="store_true",
                help="add a direction-only arm: the deviation projected on\n                     each word's unit lens vector")
ap.add_argument("--ig-steps", type=int, default=4,
                help="points on the path; 0 or 1 disables the arm")
ap.add_argument("--vocab", type=int, default=200)
ap.add_argument("--norm", default="double", choices=["double", "col"])
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--caption", type=int, default=1)
ap.add_argument("--unit-all", action="store_true",
                help="give every lens the direction-only treatment, not just\n"
                     "                     the VJ arm, so the comparison stays "
                     "symmetric")
ap.add_argument("--variants", type=int, default=0,
                help="also record the alternative region poolings")
ap.add_argument("--render", type=int, default=1)
ap.add_argument("--out", default="/workspace/vlm/results/wordgrid")
ap.add_argument("--figdir", default="/workspace/vlm/figures")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

# ---------------------------------------------------------------- vocabulary
#
# Every COCO category first, so the photographed object always has a word to be
# read as; then the everyday nouns a caption of a photograph actually uses, so
# a lens that is not reading the object has somewhere sensible to land and the
# comparison is not "the right word or nonsense".
EXTRA = """
man woman child boy girl people crowd player rider hand face head hair
sky cloud sun grass tree leaf flower plant field mountain hill rock sand
beach ocean sea water wave river lake snow ice forest park garden farm
street road building house wall roof window door fence bridge tower city
sign pole track station bus stop path yard gate wood metal glass plastic
room kitchen bathroom bedroom office desk floor ceiling lamp light mirror
shelf cabinet counter plate napkin food meal bread cheese meat salad rice
pasta soup coffee tea wine beer juice fruit vegetable egg potato tomato
lettuce onion pepper corn sauce dessert ice cream cookie candy
phone camera computer screen monitor keyboard ball bat racket glove helmet
hat shirt jacket dress coat shoe sock pants bag box basket bucket blanket
pillow curtain rug towel paper newspaper magazine pen pencil key watch ring
glasses flag balloon candle wheel engine tire seat door handle window
grey white black brown green blue yellow red orange pink purple
night day morning evening shadow sunset indoor outdoor
"""


def build_vocab(tok, cap):
    r"""Each word, with every spelling of it the tokenizer emits whole.

    A word is not a token.  Asked to name what it sees, this model answers
    ``'bed'``, ``'bus'``, ``'truck'`` -- the *unspaced* form, because the answer
    starts a message.  The space-prefixed ``' bed'`` sits several logits lower
    and ``' person'`` is not even in the top thirty.  A lens that differentiates
    ``' person'`` is therefore reading the gradient of a word the model was
    never going to emit, which is a measurement of nothing.  So a word here is
    the set of its single-token spellings and its score is their log-sum-exp --
    a smooth maximum, so the spelling the model actually favours is the one
    that carries the reading, and the same definition serves all three lenses.
    """
    words, forms, seen = [], [], set()

    def add(w):
        if w in seen:
            return
        ids = []
        for f in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            e = tok.encode(f, add_special_tokens=False)
            if len(e) == 1 and e[0] not in ids:
                ids.append(e[0])
        if not ids:
            return
        seen.add(w)
        words.append(w)
        forms.append(ids)
    for w in cocodata.SINGLE_WORD:
        add(w)
    n_cat = len(words)
    for w in EXTRA.split():
        if len(words) >= cap:
            break
        add(w)
    return words, forms, n_cat


m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lo, hi = max(m.n_deepstack, 1), m.n_layers - 2
LAYERS = sorted({int(round(lo + (hi - lo) * f)) for f in a.depths[: a.n_layers]})
J = None
if a.jlens and os.path.exists(a.jlens):
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))
    print(f"J-lens loaded: layers {min(J.jacobians)}..{max(J.jacobians)}",
          flush=True)
WORDS, FORMS, N_CAT = build_vocab(m.tokenizer, a.vocab)
WI = {w: i for i, w in enumerate(WORDS)}
NF = max(len(f) for f in FORMS)
FIDX = torch.full((len(WORDS), NF), FORMS[0][0], dtype=torch.long)
FMSK = torch.zeros(len(WORDS), NF, dtype=torch.bool)
for _i, _f in enumerate(FORMS):
    FIDX[_i, : len(_f)] = torch.tensor(_f)
    FMSK[_i, : len(_f)] = True


def word_scores(logits):
    """``[..., V]`` model logits -> ``[..., K]`` one score per word."""
    v = logits[..., FIDX.to(logits.device)]
    v = v.masked_fill(~FMSK.to(logits.device), float("-inf"))
    return torch.logsumexp(v.float(), dim=-1)


#: One token id per word for the helpers in :mod:`jvlm.vmap`, which take token
#: ids rather than words: the unspaced spelling, which is the one this model
#: actually emits when it answers.
IDS = [f[0] for f in FORMS]


def word_directions(logits):
    r"""``[K, d]`` -- the direction in residual space each word is read along.

    A word's score is the log-sum-exp of its spellings' logits, so its readout
    direction is the softmax-weighted sum of their unembedding rows, carried
    through the final norm's gain.  Its *length* is the reason a raw reading
    cannot decide a tile: a frequent word has a long readout vector and so
    scores high wherever anything is happening, in every lens.  Dividing by the
    length asks the question a map should ask -- how far along this word's
    direction does this place point -- and it is applied to all three lenses,
    never to one.
    """
    v = logits[FIDX.to(logits.device)]
    v = v.masked_fill(~FMSK.to(logits.device), float("-inf"))
    p = v.float().softmax(-1)                       # [K, F]
    W = m._lm_head.weight[FIDX.to(logits.device)].float()   # [K, F, d]
    u = (p[..., None] * W).sum(1)                   # [K, d]
    g = getattr(m._norm, "weight", None)
    return u * g.float().to(u.device) if g is not None else u

sc = tasks.Scorer(m)
print(m, f"\nlayers {LAYERS}   vocabulary {len(WORDS)} words "
         f"({N_CAT} COCO categories)", flush=True)

# The lens reads the gradient of an answer, so the answer has to be a word
# about the picture.  Under "Describe this image." the next token is "The" and
# the model's most-favoured noun is the same one ("camera") for every
# photograph -- a question with no object in its answer gives the lens nothing
# to be about.  Asked to name the thing, the answer is the object's own word.
PROMPT = ("What is the main object in this photograph? "
          "Answer with one word.")
LENSES = (["logit lens", "logit lens (centred)"]
          + (["J-lens"] if J is not None else [])
          + ["VJ-Lens"] + (["VJ-Lens (path)"] if a.ig_steps > 1 else [])
          + (["VJ-Lens (direction)"] if a.unit else []))


def prep(p):
    return m.chat(PROMPT, [corpus.load_image(p, square=a.square)])


def zscore(x, mode="double"):
    r"""Make one lens's ``[K, R]`` readings comparable across words and places.

    ``col`` is the paper's :func:`jvlm.vmap.normalise`: standardise down the
    vocabulary at each region, which removes how hard a *place* is pushing
    overall and asks which word it pushes hardest.

    That is not enough to decide a tile.  Every one of these lenses decodes
    some vector with the same unembedding, and every one of them decodes a
    vector that is the sum of a part shared by the whole picture and a part
    belonging to the region.  The shared part alone would print the same word
    in all 49 tiles, so a comparison that leaves it in is mostly a comparison
    of which lens has the largest shared part.  ``double`` centres the readings
    across regions for each word as well, so a word that scores high everywhere
    cannot claim a tile and what is left is what each place says that the rest
    of the picture does not.  The operation is arithmetic on the readings and
    is applied identically to every lens.
    """
    if mode == "double":
        x = x - x.mean(1, keepdim=True)
    return (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)


def blocks(G, C):
    e = [round(i * G / C) for i in range(C + 1)]
    return [list(range(e[i], e[i + 1])) for i in range(C)]


def region_do(batch, layers, ipos, means, donors=None):
    """{layer: {variant: [K, R]}} -- the region reading, two ways.

    One backward pass per vocabulary word covers every layer and every region
    at once, which is why a 200-word vocabulary over four depths is affordable.

    ``transport`` pools the transport and the state separately -- the mean
    gradient over the region dotted with the mean deviation over it, which is
    the region coordinate written literally.  ``additive`` pools the readings
    instead: the mean over the region of each patch's own contribution.  The
    two agree only when the gradient is constant inside the region, and the
    second is the one that adds up (a region's reading is the sum of its
    patches', however the region is drawn), so it is the one that survives
    being pooled to a legible size.
    """
    layers = sorted(set(layers))
    apos = batch["input_ids"].shape[1] - 1
    pos = torch.as_tensor(ipos, dtype=torch.long)
    keys = ["transport", "additive"] + (["donor"] if donors is not None else [])
    out = {l: {k: [] for k in keys} for l in layers}
    with (
        ActivationRecorder(m.layers, at=[*layers, m.n_layers - 1],
                           start_graph_at=min(layers)) as rec,
        torch.enable_grad(),
    ):
        m.forward(batch)
        acts = [rec.activations[l] for l in layers]
        h_final = rec.activations[m.n_layers - 1][0, apos]
        logits = m.unembed(h_final[None])[0]
        dev, devp, devd = {}, {}, {}
        for l, aa in zip(layers, acts):
            h = aa[0, pos.to(aa.device)].detach().float()
            devp[l] = h - means[l].to(h.device).float()
            dev[l] = POOL @ devp[l]
            if donors is not None:
                devd[l] = h - donors[l].to(h.device).float()
        wsc = word_scores(logits)
        for j in range(len(WORDS)):
            grads = torch.autograd.grad(wsc[j], acts,
                                        retain_graph=(j < len(WORDS) - 1))
            for l, g in zip(layers, grads, strict=True):
                gp = g[0, pos.to(g.device)].float()
                out[l]["transport"].append(
                    ((POOL @ gp) * dev[l]).sum(-1).cpu())
                out[l]["additive"].append(
                    (POOL @ (gp * devp[l]).sum(-1)).cpu())
                if donors is not None:
                    out[l]["donor"].append(
                        (POOL @ (gp * devd[l]).sum(-1)).cpu())
    return {l: {k: torch.stack(v) for k, v in d.items()}
            for l, d in out.items()}


def webp(im, size=340, q=82):
    q_ = im.copy()
    q_.thumbnail((size, size), Image.LANCZOS)
    b = io.BytesIO()
    q_.save(b, "WEBP", quality=q, method=4)
    return "data:image/webp;base64," + base64.b64encode(b.getvalue()).decode()


# ------------------------------------------------------- per-position means
pool_items = cocodata.concept_items(6000, min_area=0.05, seed=7)
b0 = prep(pool_items[0]["path"])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
C = max(2, min(G, int(round(G / a.pool))))
ROWS, COLS = blocks(G, C), blocks(G, C)
acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
nrm = {l: torch.zeros(P, dtype=torch.float64) for l in LAYERS}
BANK = {l: [] for l in LAYERS}
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
        if len(BANK[l]) < 10:
            BANK[l].append(v)
    nm += 1
MEANS = {l: (acc[l] / max(nm, 1)).float() for l in LAYERS}
NORM = {l: (nrm[l] / max(nm, 1)).float() for l in LAYERS}
# High-norm "register" patches carry no picture; they carry the model's own
# bookkeeping.  Leaving them in a region mean lets one such patch decide the
# word for a whole tile, so they are dropped -- identically for all lenses.
REG = torch.zeros(P, dtype=torch.bool)
for l in LAYERS:
    REG |= NORM[l] > NORM[l].mean() + 3 * NORM[l].std()
REGIONS = []
for rr in ROWS:
    for cc in COLS:
        idx = [r * G + c for r in rr for c in cc]
        keep = [i for i in idx if not REG[i]]
        REGIONS.append(torch.as_tensor(keep or idx, dtype=torch.long))
# The pooling is one matrix multiply, not 49 gathers per word per layer: with a
# 200-word vocabulary the gathers cost more than the backward passes do.
POOL = torch.zeros(len(REGIONS), P)
for _i, _r in enumerate(REGIONS):
    POOL[_i, _r] = 1.0 / len(_r)
POOL = POOL.to(m.device)
print(f"{P} tokens on {G}x{G}; pooled to {C}x{C} = {C*C} tiles; "
      f"{int(REG.sum())} register patches dropped; means from {nm} photographs "
      f"({time.time()-t0:.0f}s)", flush=True)


def pooled_h(h):
    """[P, d] -> [R, d] region means, registers excluded."""
    return (POOL @ h.to(POOL.device)).cpu()


# ------------------------------------------------------------------ the cards
items = cocodata.concept_items(a.n * 14, min_area=0.08, seed=1301,
                               tokenizer=m.tokenizer)
cards, rows = [], []
for k, it in enumerate(items):
    if len(cards) >= a.n:
        break
    c = it["concept"]
    if c not in WI:
        continue
    boxes = cocodata.object_boxes(it["image_id"], c)
    if not boxes:
        continue
    inside = sorted({t for bx in boxes for t in cocodata.box_tokens(bx, C)})
    if not (3 <= len(inside) <= C * C * 0.60):
        continue
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    A = readout.residuals(m, b, LAYERS)
    flog = m.unembed(A[m.n_layers - 1][-1:].to(m.device))[0].float()
    fin = word_scores(flog).cpu()
    UDIR = word_directions(flog) if a.unit_all else None
    # The lens explains the answer, so the answer has to be the object: among
    # the vocabulary's nouns the model must rank the photographed one at the
    # top.  Without this the figure would be asked to explain a word the model
    # was never going to say.
    rank = int((fin > fin[WI[c]]).sum())
    if rank != 0:
        continue
    # ...and it has to actually say it.  A photograph the model answers with a
    # different word than the one whose box we draw makes the whole panel a
    # non sequitur: the caption says "giraffe" while the outline is around a
    # person.  Generating the answer costs one short decode and removes them.
    said = sc.generate(b, 8).strip().split("\n")[0][:28] if a.caption else c
    toks = [w.strip(" .,:;\"'*()").lower() for w in said.split()]
    if a.caption and not ({c, c + "s", c.rstrip("s")} & set(toks)):
        continue
    UNIT = None
    if a.unit:
        _u = vmap.ig_map(m, b, IDS, LAYERS, ip.tolist(), means=MEANS,
                         steps=max(a.ig_steps, 1), unit=True) if a.ig_steps > 1 \
            else vmap.do_map(m, b, IDS, LAYERS, ip.tolist(), means=MEANS,
                             unit=True)
        UNIT = {l: torch.stack([_u[l][:, r].mean(-1) for r in REGIONS], dim=-1)
                for l in LAYERS}
    IG = None
    if a.ig_steps > 1:
        _per = vmap.ig_map(m, b, IDS, LAYERS, ip.tolist(), means=MEANS,
                           steps=a.ig_steps)
        IG = {l: torch.stack([_per[l][:, r].mean(-1) for r in REGIONS], dim=-1)
              for l in LAYERS}
    DO = region_do(b, LAYERS, ip.tolist(), MEANS,
                   donors=({l: BANK[l][len(cards) % len(BANK[l])]
                            for l in LAYERS} if a.variants else None))
    im = corpus.load_image(it["path"], square=a.square)
    card = {"image_id": it["image_id"], "concept": c, "boxes": boxes,
            "img": webp(im), "grid": G, "coarse": C, "inside": inside,
            "answer_rank": rank, "objects": sorted(it["present"]),
            "caption": said,
            "lenses": {}}
    for l in LAYERS:
        h = A[l][ip].float().cpu()
        hp = pooled_h(h)
        mp = pooled_h(MEANS[l])
        maps = {
            "logit lens": word_scores(
                m.unembed(hp.to(m.device)).float()).cpu().T,
            "logit lens (centred)": word_scores(
                m.unembed((hp - mp).to(m.device)).float()).cpu().T,
            "VJ-Lens": DO[l]["additive"],
            **({"VJ-Lens (path)": IG[l]} if IG is not None else {}),
            **({"VJ-Lens (direction)": UNIT[l]} if UNIT is not None else {}),
        }
        if a.variants:
            maps["VJ-Lens (transport-pooled)"] = DO[l]["transport"]
            maps["VJ-Lens (donor baseline)"] = DO[l]["donor"]
        if J is not None and l in J.jacobians:
            Jl = J.jacobians[l].to(m.device).float()
            maps["J-lens"] = word_scores(
                m.unembed(hp.to(m.device) @ Jl.T).float()).cpu().T
        if UDIR is not None:
            un = UDIR.norm(dim=-1).clamp_min(1e-6).cpu()[:, None]
            maps["logit lens (direction)"] = maps["logit lens"] / un
            if "J-lens" in maps:
                unj = (UDIR @ Jl).norm(dim=-1).clamp_min(1e-6).cpu()[:, None]
                maps["J-lens (direction)"] = maps["J-lens"] / unj
        for name, mm in maps.items():
            z = zscore(mm.float(), a.norm)              # [K, R]
            # The word grid asks a question with the whole vocabulary in it:
            # is the object's name the *winning* word here.  The pointing game
            # asks a smaller one that needs no comparison between words at all
            # -- where is this lens's map for the object's own name highest --
            # and it is reported beside it so a lens cannot be written off for
            # losing a contest it never entered.
            om = mm.float()[WI[c]]
            pos_ = om.clamp_min(0)
            point = int(int(om.argmax()) in set(inside))
            mass = float(pos_[list(inside)].sum() / pos_.sum().clamp_min(1e-9))
            top = z.argmax(0)
            zt = z.max(0).values
            words = [WORDS[int(i)] for i in top]
            n_in = sum(words[t] == c for t in inside)
            out_t = [t for t in range(C * C) if t not in set(inside)]
            n_out = sum(words[t] == c for t in out_t)
            cnt = Counter(words)
            card["lenses"].setdefault(name, {})[str(l)] = {
                "depth": round(l / (m.n_layers - 1), 3),
                "words": words,
                "z": [round(float(v), 3) for v in zt],
                "distinct": len(cnt),
                "mode_word": cnt.most_common(1)[0][0],
                "mode_frac": round(cnt.most_common(1)[0][1] / (C * C), 4),
                "inside": round(n_in / max(len(inside), 1), 4),
                "outside": round(n_out / max(len(out_t), 1), 4),
                "point": point, "mass": round(mass, 4),
            }
            rows.append(dict(i=len(cards), image_id=it["image_id"], concept=c,
                             lens=name, layer=l,
                             depth=round(l / (m.n_layers - 1), 3),
                             distinct=len(cnt),
                             mode_frac=cnt.most_common(1)[0][1] / (C * C),
                             inside=n_in / max(len(inside), 1),
                             outside=n_out / max(len(out_t), 1),
                             point=point, mass=mass,
                             chance=len(inside) / (C * C),
                             n_inside=len(inside)))
    cards.append(card)
    print(f"  {len(cards)}/{a.n}  {c:12s} {time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
agg = df.groupby(["lens", "layer"]).agg(
    n=("i", "count"), depth=("depth", "first"), distinct=("distinct", "mean"),
    mode_frac=("mode_frac", "mean"), inside=("inside", "mean"),
    outside=("outside", "mean"), point=("point", "mean"),
    mass=("mass", "mean"), chance=("chance", "mean")).reset_index()
agg["margin"] = agg["inside"] - agg["outside"]
agg.to_csv(f"{a.out}/{tag}_summary.csv", index=False)
print("\nWHAT EACH LENS PRINTS ON A %dx%d GRID OF TILES\n" % (C, C),
      agg.round(3).to_string(index=False), flush=True)

# each lens at its own best depth -- the fairest reading of a baseline
best = {}
for lens in agg.lens.unique():
    d = agg[agg.lens == lens]
    r = d.loc[d["inside"].idxmax()]
    best[lens] = {k: (float(r[k]) if k not in ("lens", "layer", "n")
                      else (r[k] if k == "lens" else int(r[k])))
                  for k in ("layer", "depth", "n", "distinct", "mode_frac",
                            "inside", "outside", "margin")}
print("\nEACH LENS AT ITS OWN BEST DEPTH\n",
      pd.DataFrame(best).T.round(3).to_string(), flush=True)

blob = {"repo": a.repo, "tag": tag, "n": len(cards), "grid": G, "coarse": C,
        "norm": a.norm,
        "n_patches": P, "n_layers": m.n_layers, "layers": LAYERS,
        "vocab": WORDS, "n_categories": N_CAT, "lenses": LENSES,
        "registers": int(REG.sum()), "prompt": PROMPT,
        "by_layer": st.jsonable(agg.round(4).to_dict("index")),
        "best": st.jsonable(best), "cards": st.jsonable(cards),
        "minutes": round((time.time() - t0) / 60, 1)}
json.dump(blob, open(f"{a.out}/{tag}_wordgrid.json", "w"), default=float)
np.savez_compressed(
    f"{a.out}/{tag}_pooled.npz",
    **{f"{ci}_{ln.replace(' ', '_')}_{l}": np.asarray(v["z"], dtype=np.float32)
       for ci, cd in enumerate(cards) for ln, per in cd["lenses"].items()
       for l, v in per.items()})
print(f"wrote {a.out}/{tag}_wordgrid.json  ({len(cards)} cards)", flush=True)

if a.render:
    from jvlm import figures, style
    style.use()
    os.makedirs(a.figdir, exist_ok=True)
    for i in range(min(len(cards), 4)):
        figures.f50_wordgrid(blob, a.figdir, name=f"f50_wordgrid_{i}.png",
                             which=i)
print("done", flush=True)
