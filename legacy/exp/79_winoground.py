# SPDX-License-Identifier: Apache-2.0
"""The theorem, measured on the dataset built to defeat it.

Image tokens precede text under a causal mask, so an image token's activation
is a function of the image alone.  Any lens of the form ``g(M h)`` whose ``M``
is not computed from the prompt therefore returns a *byte-identical* reading at
image tokens however the prompt is rewritten.  The logit lens is such a lens
(``M = I``); so is the Jacobian lens (``M = J_l``, a corpus average).  Neither
can tell you what the picture is doing for *this* question, because neither has
been shown the question.

Winoground is the adversary for that claim.  Every item is a pair of
photographs and a pair of captions built from the SAME WORDS in a different
order -- "some plants surrounding a lightbulb" / "a lightbulb surrounding some
plants" -- with image_0 matching caption_0 and image_1 matching caption_1.  If
any prompt could move a prompt-independent lens, this is the one: the two
prompts differ by nothing except word order, and the ground truth flips.

The design fixes one image at a time and asks the same question twice::

    Does the caption '<c>' describe this image?  Answer yes or no.

Two images x two captions is four forward passes per item.  Every comparison is
WITHIN one image and ACROSS the two captions; readings are never mixed across
images, and image_0 and image_1 stay separate rows throughout.  For each image
we know which caption is the correct one (caption_i for image_i), so the
model's own behaviour comes out of the same four passes as Winoground's
standard text score.

Three lenses are read at the image tokens, for the answer token, at each of a
few layers, and pooled to a 7x7 grid of regions:

    logit lens    W_U h[R]                     content, no transport
    J-lens        W_U J_l h[R]                 content, corpus-averaged transport
    VJ-Lens       sum_p grad . (h[p] - hbar)   influence, this pass's transport

and the measurement is how far each lens's 49-region map MOVES when the caption
is rearranged.  For the first two the answer must be exactly zero, and the
number reported is the maximum absolute deviation over every item, image, layer
and region -- so the claim reads "0.0, measured" rather than "small".  Two
controls stand behind that zero: the maximum deviation of the raw activations
themselves, and a re-run of one prompt against itself, which separates the
theorem from floating-point non-determinism.

The target word.  A saturated logit reads as zero contribution however much it
matters, so the run measures the yes logit's gradient before trusting it, and
carries both readings: the yes map and the margin map, logit(yes) - logit(no).
Because every lens here is linear in the gradient (and in ``h``), the margin map
is exactly the yes map minus the no map, so it costs no extra passes.  Which one
is primary is decided from the recorded diagnostic and stated in the stats file.

Nothing here is worth reading if the model is at chance on Winoground, so the
text, image and group scores are reported first, beside their chance levels.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import zipfile

sys.path.insert(0, "code")
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps

import jvlm
from jvlm import corpus, readout, stats as st, vmap

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--jlens", default=None)
ap.add_argument("--n", type=int, default=400, help="Winoground items")
ap.add_argument("--mean-images", type=int, default=90,
                help="held-out photographs behind the baseline hbar")
ap.add_argument("--n-layers", type=int, default=4)
ap.add_argument("--pool", type=int, default=2, help="token block side")
ap.add_argument("--ig-steps", type=int, default=4)
ap.add_argument("--n-cases", type=int, default=12,
                help="items written out in full to the cases file")
ap.add_argument("--target", default="auto", choices=("auto", "yes", "margin"),
                help="which answer logit the lenses are read for; auto\n"
                     "                     picks from the saturation diagnostic")
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--data", default="data/winoground")
ap.add_argument("--out", default="results/winoground")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

QUESTION = "Does the caption '{c}' describe this image?  Answer yes or no."


# ------------------------------------------------------------------ the data

def ensure_winoground(root: str) -> list[dict]:
    """``examples.jsonl`` + ``images/``, fetched once from the Hub.

    The dataset is gated, so the request needs a token; it is read out of the
    user's token file at call time and is never printed, logged, or written
    anywhere by this script.
    """
    ex = os.path.join(root, "data", "examples.jsonl")
    imgs = os.path.join(root, "images")
    if not (os.path.exists(ex) and os.path.isdir(imgs)):
        from huggingface_hub import hf_hub_download
        tf = os.path.expanduser("~/.cache/huggingface/token")
        tok = open(tf).read().strip() if os.path.exists(tf) else None
        os.makedirs(root, exist_ok=True)
        for fn in ("data/examples.jsonl", "data/images.zip"):
            hf_hub_download("facebook/winoground", fn, repo_type="dataset",
                            token=tok, local_dir=root)
        with zipfile.ZipFile(os.path.join(root, "data", "images.zip")) as z:
            z.extractall(root)
    recs = [json.loads(l) for l in open(ex)]
    for r in recs:
        for k in ("image_0", "image_1"):
            r[k + "_path"] = os.path.join(imgs, r[k] + ".png")
    return recs


RECS = ensure_winoground(a.data)
print(f"winoground: {len(RECS)} records, "
      f"{sum(os.path.exists(r['image_0_path']) + os.path.exists(r['image_1_path']) for r in RECS)}"
      f" image files present", flush=True)

# ----------------------------------------------------------------- the model

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lo, hi = max(m.n_deepstack, 1), m.n_layers - 2
LAYERS = sorted({int(round(lo + (hi - lo) * f))
                 for f in np.linspace(0.2, 0.85, a.n_layers)})
J = None
if a.jlens:
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))

    # Read where the lens exists.  A fitted Jacobian lens only has a transport
    # at the layers it was fitted at; chosen from the model's geometry instead,
    # the depths miss it and the arm is dropped in silence.
    LAYERS, _ = readout.snap_to_lens(LAYERS, J)
    miss = [l for l in LAYERS if l not in J.jacobians]
    print(f"J-lens: layers {min(J.jacobians)}..{max(J.jacobians)}"
          + (f"  (no jacobian at {miss})" if miss else ""), flush=True)


def ids_of(w):
    out = []
    for f in (w, " " + w, w.capitalize(), " " + w.capitalize()):
        t = m.tokenizer.encode(f, add_special_tokens=False)
        if len(t) == 1:
            out.append(t[0])
    return sorted(set(out))


YES_FORMS, NO_FORMS = ids_of("yes"), ids_of("no")
print(m, f"\nlayers {LAYERS}   yes forms {len(YES_FORMS)}  no forms {len(NO_FORMS)}",
      flush=True)


def load(p):
    return corpus.load_image(p, square=a.square)


# --------------------------------------------------- baseline and the regions
# The baseline hbar is a per-position corpus mean over held-out photographs.
# Winoground is not in COCO, so COCO val2017 is held out by construction; and
# by the very theorem under test the prompt used to collect it cannot matter at
# image positions, so a generic instruction is used.
pool_paths = corpus.coco_images(a.mean_images + 30, seed=7)
b0 = m.chat("Describe this image.", [load(pool_paths[0])])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
nrm = {l: torch.zeros(P, dtype=torch.float64) for l in LAYERS}
nm = 0
for p in pool_paths:
    if nm >= a.mean_images:
        break
    b = m.chat("Describe this image.", [load(p)])
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
# register tokens: attention sinks whose norm is far above the rest.  They are
# not a place in the picture, so they must not stand in for one.
REG = (NORM[LAYERS[0]] > NORM[LAYERS[0]].mean() + 3 * NORM[LAYERS[0]].std())

C = max(2, G // a.pool)
side = int(np.ceil(G / C))
REGIONS, RSIZE, EMPTY = [], [], []
for ry in range(C):
    for rx in range(C):
        idx = [y * G + x
               for y in range(ry * side, min((ry + 1) * side, G))
               for x in range(rx * side, min((rx + 1) * side, G))
               if not bool(REG[y * G + x])]
        EMPTY.append(not idx)
        # an all-register cell keeps its slot so every map is 49 long, but it
        # sums over nothing and reads exactly zero rather than borrowing a patch
        REGIONS.append(torch.tensor(idx, dtype=torch.long))
        RSIZE.append(max(len(idx), 1))
R = len(REGIONS)
SIZES = torch.tensor(RSIZE, dtype=torch.float32)
print(f"{P} tokens on {G}x{G}; {R} regions of about {side*side} patches "
      f"({int(REG.sum())} register tokens dropped, {sum(EMPTY)} empty regions); "
      f"means from {nm} held-out photographs ({time.time()-t0:.0f}s)", flush=True)


def region_sum(per_patch: torch.Tensor) -> torch.Tensor:
    """``[k, P]`` -> ``[k, R]`` by SUM, which is what makes regions add."""
    return torch.stack([per_patch[:, r].sum(-1) for r in REGIONS], dim=-1)


def unembed_ids(h: torch.Tensor, ids) -> torch.Tensor:
    """``readout.lens_logits`` restricted to a couple of tokens.

    Identical to ``model.unembed(h)[:, ids]`` -- asserted against it once at
    startup -- but it never materialises a 152k-wide table, which this run would
    otherwise do sixteen times per item per layer.
    """
    return m.unembed(h.to(m.device))[:, list(ids)].float().cpu()


IDS_PROBE = YES_FORMS + NO_FORMS
_hh = readout.residuals(m, b0, [LAYERS[0]])[LAYERS[0]][IPOS]
_a = unembed_ids(_hh, IDS_PROBE)
_b, _ = readout.lens_logits(m, None, b0, layers=[LAYERS[0]],
                            positions=IPOS.tolist(), use_jacobian=False)
_d = float((_a - _b[LAYERS[0]][:, IDS_PROBE]).abs().max())
assert _d == 0.0, f"unembed_ids disagrees with readout.lens_logits by {_d}"
del _hh, _a, _b

# ----------------------------------------------- which surface form to score
# The target ids are FIXED for the whole run.  Letting the run pick the top
# form per prompt would make the two captions ask for different tokens, and the
# zero this experiment exists to measure would be an artefact of that choice
# rather than a property of the lens.
probe = torch.zeros(len(IDS_PROBE))
for r in RECS[:3]:
    for gi in (0, 1):
        im = load(r[f"image_{gi}_path"])
        for ci in (0, 1):
            b = m.chat(QUESTION.format(c=r[f"caption_{ci}"]), [im])
            A = readout.residuals(m, b, [m.n_layers - 1])
            lg = m.unembed(A[m.n_layers - 1][-1:].to(m.device))[0].float().cpu()
            probe += lg[IDS_PROBE]
YES = int(IDS_PROBE[int(probe[:len(YES_FORMS)].argmax())])
NO = int(IDS_PROBE[len(YES_FORMS) + int(probe[len(YES_FORMS):].argmax())])
IDS = [YES, NO]
print(f"answer tokens fixed: yes={m.tokenizer.decode([YES])!r} ({YES})  "
      f"no={m.tokenizer.decode([NO])!r} ({NO})", flush=True)

LENSES = ["logit lens"] + (["J-lens"] if J is not None else []) \
    + ["VJ-Lens (point)", "VJ-Lens"]
PRIOR = ["logit lens"] + (["J-lens"] if J is not None else [])


def read_pass(b):
    """Everything one forward pass has to say.

    Returns the answer logits, the per-patch ``[2, P]`` map of every lens at
    every layer (row 0 = yes, row 1 = no), the image-token activations, and the
    gradient diagnostics the saturation check is made of.
    """
    A = readout.residuals(m, b, LAYERS + [m.n_layers - 1])
    lg = m.unembed(A[m.n_layers - 1][-1:].to(m.device))[0].float().cpu()
    out = {"yes": float(lg[YES]), "no": float(lg[NO]),
           "top": m.tokenizer.decode([int(lg.argmax())]),
           "h": {l: A[l][IPOS].float().cpu() for l in LAYERS},
           "maps": {}, "sat": {}}
    for l in LAYERS:
        h = A[l][IPOS]
        out["maps"][("logit lens", l)] = unembed_ids(h, IDS).T.contiguous()
        if J is not None and l in J.jacobians:
            hj = h.float() @ J.jacobians[l].to(h.device).float().T
            out["maps"][("J-lens", l)] = unembed_ids(hj, IDS).T.contiguous()
    # one backward per answer token covers every layer at once, so the point
    # reading and the saturation diagnostic are two passes for the whole stack
    g = vmap.raw_gradients(m, b, IDS, LAYERS, IPOS.tolist())
    for l in LAYERS:
        dev = out["h"][l].to(g[l].device) - MEANS[l].to(g[l].device)
        out["maps"][("VJ-Lens (point)", l)] = (g[l] * dev).sum(-1).cpu()
        gy, gn = g[l][0], g[l][1]
        out["sat"][l] = dict(
            gnorm_yes=float(gy.norm(dim=-1).mean()),
            gnorm_no=float(gn.norm(dim=-1).mean()),
            gnorm_margin=float((gy - gn).norm(dim=-1).mean()),
            cos_yes_no=float(torch.nn.functional.cosine_similarity(
                gy, gn, dim=-1).mean()),
            dev_norm=float(dev.norm(dim=-1).mean()))
    ig = vmap.ig_map(m, b, IDS, LAYERS, IPOS.tolist(), means=MEANS,
                     steps=a.ig_steps)
    for l in LAYERS:
        out["maps"][("VJ-Lens", l)] = ig[l]
    return out


def regionise(v, lens):
    """``[2, P]`` -> ``[2, R]``.  Content lenses are averaged over a region
    (they are per-token scores), contributions are summed (they are additive)."""
    s = region_sum(v)
    return s / SIZES if lens in PRIOR else s


def rankcorr(x, y):
    if np.array_equal(x, y):
        return 1.0
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(sps.spearmanr(x, y).statistic)


def auc(pos, neg):
    """Mann-Whitney AUROC, tie-aware: an all-tied predictor scores exactly
    0.5 rather than whatever ``argsort`` happened to do with the ties."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    r = sps.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


# ---------------------------------------------------------------- the run
rows, cases, sat_rows, ctrl = [], [], [], []
skipped = 0
for k, rec in enumerate(RECS[: a.n]):
    caps = [rec["caption_0"], rec["caption_1"]]
    paths = [rec["image_0_path"], rec["image_1_path"]]
    if not all(os.path.exists(p) for p in paths):
        skipped += 1
        continue
    got, bad = {}, False
    for gi in (0, 1):
        im = load(paths[gi])
        for ci in (0, 1):
            b = m.chat(QUESTION.format(c=caps[ci]), [im])
            if len(readout.image_positions(m, b)) != P:
                bad = True
                break
            got[(gi, ci)] = read_pass(b)
        if bad:
            break
        if k < a.n_cases:
            # the same prompt, run a second time: whatever this deviates by is
            # the floor set by floating point, not by the theorem
            b = m.chat(QUESTION.format(c=caps[0]), [im])
            A = readout.residuals(m, b, [LAYERS[0]])
            ctrl.append(dict(item=int(rec["id"]), image=gi, layer=LAYERS[0],
                             h_maxdev=float((A[LAYERS[0]][IPOS].float().cpu()
                                             - got[(gi, 0)]["h"][LAYERS[0]])
                                            .abs().max())))
    if bad:
        skipped += 1
        continue

    marg = {(gi, ci): got[(gi, ci)]["yes"] - got[(gi, ci)]["no"]
            for gi in (0, 1) for ci in (0, 1)}
    # Winoground's own scores.  text: hold the image, rank the captions --
    # which is exactly the within-image comparison this experiment is built on.
    # image: hold the caption, rank the images.  group: both.
    text_ok = [int(marg[(gi, gi)] > marg[(gi, 1 - gi)]) for gi in (0, 1)]
    img_ok = [int(marg[(ci, ci)] > marg[(1 - ci, ci)]) for ci in (0, 1)]
    text_item = int(all(text_ok))
    image_item = int(all(img_ok))
    group_item = int(text_item and image_item)

    for gi in (0, 1):
        for l in LAYERS:
            s = got[(gi, 0)]["sat"][l]
            sat_rows.append(dict(item=int(rec["id"]), image=gi, layer=l, **s))
        for lens in LENSES:
            if (lens, LAYERS[0]) not in got[(gi, 0)]["maps"]:
                continue
            for l in LAYERS:
                if (lens, l) not in got[(gi, 0)]["maps"]:
                    continue
                pp = [got[(gi, ci)]["maps"][(lens, l)] for ci in (0, 1)]
                rr = [regionise(v, lens) for v in pp]
                row = dict(item=int(rec["id"]), image=gi, layer=l, lens=lens,
                           cond=lens, i=f"{rec['id']}_{gi}_{l}",
                           tag=rec.get("tag", ""),
                           collapsed=rec.get("collapsed_tag", ""),
                           correct_caption=gi,
                           depth=round(l / (m.n_layers - 1), 3),
                           text_ok=text_ok[gi], text_item=text_item,
                           image_item=image_item, group_item=group_item,
                           margin_correct=marg[(gi, gi)],
                           margin_swapped=marg[(gi, 1 - gi)],
                           # the raw activations behind every lens on this row
                           h_maxdev=float((got[(gi, 0)]["h"][l]
                                           - got[(gi, 1)]["h"][l]).abs().max()))
                for name, tgt in (("yes", 0), ("margin", None)):
                    m0, m1 = ((rr[0][tgt], rr[1][tgt]) if tgt is not None
                              else (rr[0][0] - rr[0][1], rr[1][0] - rr[1][1]))
                    p0, p1 = ((pp[0][tgt], pp[1][tgt]) if tgt is not None
                              else (pp[0][0] - pp[0][1], pp[1][0] - pp[1][1]))
                    x, y = m0.numpy(), m1.numpy()
                    den = 0.5 * (np.linalg.norm(x) + np.linalg.norm(y))
                    row[f"rho_{name}"] = rankcorr(x, y)
                    row[f"move_{name}"] = 1.0 - row[f"rho_{name}"]
                    row[f"rel_l2_{name}"] = float(
                        np.linalg.norm(x - y) / max(den, 1e-12))
                    row[f"maxdev_{name}"] = float(np.abs(x - y).max())
                    row[f"patch_maxdev_{name}"] = float(
                        (p0 - p1).abs().max())
                    row[f"scale_{name}"] = float(np.abs(x).mean())
                rows.append(row)

    if k < a.n_cases:
        cases.append(dict(
            item=int(rec["id"]), tag=rec.get("tag", ""),
            collapsed_tag=rec.get("collapsed_tag", ""),
            captions=caps, paths=paths,
            correct=("caption_0 describes image_0; caption_1 describes image_1"),
            grid=G, coarse=C, region_size=RSIZE, empty=EMPTY,
            question=QUESTION,
            answer_tokens={"yes": m.tokenizer.decode([YES]),
                           "no": m.tokenizer.decode([NO])},
            behaviour={f"img{gi}_cap{ci}": dict(
                yes=got[(gi, ci)]["yes"], no=got[(gi, ci)]["no"],
                margin=marg[(gi, ci)], said=("yes" if marg[(gi, ci)] > 0 else "no"),
                top_token=got[(gi, ci)]["top"],
                is_correct_caption=int(ci == gi))
                for gi in (0, 1) for ci in (0, 1)},
            text_ok=text_ok, text_item=text_item, image_item=image_item,
            maps={f"{lens}|L{l}|img{gi}|cap{ci}|{name}": [
                round(float(v), 6) for v in (
                    regionise(got[(gi, ci)]["maps"][(lens, l)], lens)[0]
                    if name == "yes" else
                    regionise(got[(gi, ci)]["maps"][(lens, l)], lens)[0]
                    - regionise(got[(gi, ci)]["maps"][(lens, l)], lens)[1])]
                for lens in LENSES for l in LAYERS for gi in (0, 1)
                for ci in (0, 1) for name in ("yes", "margin")
                if (lens, l) in got[(gi, ci)]["maps"]}))

    if k % 10 == 0:
        el = time.time() - t0
        print(f"  {k}/{min(a.n, len(RECS))}  {el:.0f}s  "
              f"eta {el/max(k+1,1)*(min(a.n,len(RECS))-k-1)/60:.1f}m", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
sat = pd.DataFrame(sat_rows)
sat.to_parquet(f"{a.out}/{tag}_sat.parquet")

# -------------------------------------------------------- is "yes" saturated?
# A first-order reading of a saturated logit is zero however much the picture
# matters, so this is checked before anything is believed.  Two ways for the
# yes reading to be uninformative: the gradient is near zero outright, or it is
# nearly the same vector as the no gradient, in which case the map is a picture
# of the common "an answer goes here" direction and not of the answer.
satg = sat.groupby("layer").agg(
    gnorm_yes=("gnorm_yes", "median"), gnorm_no=("gnorm_no", "median"),
    gnorm_margin=("gnorm_margin", "median"),
    cos_yes_no=("cos_yes_no", "median"), dev_norm=("dev_norm", "median"))
satg["margin_over_yes"] = satg.gnorm_margin / satg.gnorm_yes.clip(1e-12)
print("\nIS THE YES LOGIT SATURATED?  (medians over items, at image tokens)\n",
      satg.round(4).to_string())
saturated = bool((satg.cos_yes_no > 0.8).any()
                 or (satg.margin_over_yes < 0.5).any())
TARGET = a.target if a.target != "auto" else ("margin" if saturated else "yes")
why = ("the yes and no gradients are nearly collinear at image tokens, so the "
       "yes map is dominated by a common-mode 'an answer goes here' direction; "
       "the margin is the part that distinguishes the captions"
       if saturated else
       "the yes gradient is neither near zero nor collinear with the no "
       "gradient, so the yes logit is read directly")
print(f"target: {TARGET}  --  {why}", flush=True)
MV, MD, PD, RL = (f"move_{TARGET}", f"maxdev_{TARGET}",
                  f"patch_maxdev_{TARGET}", f"rel_l2_{TARGET}")

# --------------------------------------------------------- the model's score
beh = df.drop_duplicates("item")
per_img = df.drop_duplicates(["item", "image"])
n_items = int(beh.item.nunique())
sepd = (per_img.margin_correct - per_img.margin_swapped)
score = dict(
    n_items=n_items, n_images=int(len(per_img)),
    text=float(beh.text_item.mean()), text_chance=0.25,
    image=float(beh.image_item.mean()), image_chance=0.25,
    group=float(beh.group_item.mean()), group_chance=1 / 6,
    per_image_text=float(per_img.text_ok.mean()), per_image_chance=0.5,
    margin_gap_mean=float(sepd.mean()), margin_gap_median=float(sepd.median()),
    margin_gap_p=float(sps.wilcoxon(sepd).pvalue) if len(sepd) > 8 else float("nan"))
# The literal answer, against the ground truth the dataset supplies: a correct
# caption deserves yes and a rearranged one deserves no.  A model with a strong
# yes-bias can still rank the two captions correctly, so accuracy and the text
# score are different caveats and both are reported.
sy = (per_img.margin_correct > 0).to_numpy()
sn = (per_img.margin_swapped > 0).to_numpy()
score["yes_rate"] = float((sy.sum() + sn.sum()) / (2 * len(per_img)))
score["answer_accuracy"] = float((sy.sum() + (~sn).sum()) / (2 * len(per_img)))
score["yes_rate_correct_caption"] = float(sy.mean())
score["yes_rate_swapped_caption"] = float(sn.mean())
score["margin_correct_mean"] = float(per_img.margin_correct.mean())
score["margin_swapped_mean"] = float(per_img.margin_swapped.mean())
# a binomial test against chance, so "at chance" is a claim and not an eyeball
score["text_p_vs_chance"] = float(sps.binomtest(
    int(beh.text_item.sum()), n_items, 0.25, alternative="greater").pvalue)
score["group_p_vs_chance"] = float(sps.binomtest(
    int(beh.group_item.sum()), n_items, 1 / 6, alternative="greater").pvalue)
score["at_chance"] = bool(score["text_p_vs_chance"] > 0.05)
print("\nTHE MODEL ON WINOGROUND  (yes-margin as the image-caption score)")
for k2 in ("text", "image", "group", "per_image_text"):
    ch = score[{"text": "text_chance", "image": "image_chance",
                "group": "group_chance",
                "per_image_text": "per_image_chance"}[k2]]
    print(f"  {k2:>15}  {score[k2]:.4f}   chance {ch:.4f}")
print(f"  {'yes rate':>15}  {score['yes_rate']:.4f}   "
      f"answer accuracy {score['answer_accuracy']:.4f}")
print(f"  text vs chance p={score['text_p_vs_chance']:.3g}"
      f"   {'AT CHANCE' if score['at_chance'] else 'above chance'}", flush=True)
by_tag = beh.groupby("collapsed").agg(
    n=("item", "count"), text=("text_item", "mean"),
    image=("image_item", "mean"), group=("group_item", "mean")).reset_index()
print("\nby Winoground tag\n", by_tag.round(4).to_string(index=False), flush=True)

# ------------------------------------------------------- how far each map moves
mv = df.groupby(["lens", "layer"]).agg(
    n=("item", "count"),
    movement=(MV, "mean"), movement_med=(MV, "median"),
    rel_l2=(RL, "mean"), rho=(f"rho_{TARGET}", "mean"),
    max_abs_dev=(MD, "max"), max_patch_dev=(PD, "max"),
    max_h_dev=("h_maxdev", "max"), scale=(f"scale_{TARGET}", "mean")
).reset_index()
print("\nHOW FAR THE MAP MOVES WHEN THE CAPTION IS REARRANGED\n",
      mv.round(6).to_string(index=False), flush=True)
zero = {lens: dict(
    max_abs_region_dev=float(df[df.lens == lens][MD].max()),
    max_abs_patch_dev=float(df[df.lens == lens][PD].max()),
    max_abs_dev_yes=float(df[df.lens == lens]["maxdev_yes"].max()),
    max_abs_dev_margin=float(df[df.lens == lens]["maxdev_margin"].max()),
    exactly_zero=bool(df[df.lens == lens][PD].max() == 0.0),
    n=int((df.lens == lens).sum())) for lens in df.lens.unique()}
print("\nTHE THEOREM, MEASURED  (max |map(caption_0) - map(caption_1)|)")
for lens in PRIOR:
    print(f"  {lens:>18}: region {zero[lens]['max_abs_region_dev']!r}   "
          f"patch {zero[lens]['max_abs_patch_dev']!r}   "
          f"exactly zero: {zero[lens]['exactly_zero']}")
print(f"  raw activations at image tokens: max |h0 - h1| = "
      f"{float(df.h_maxdev.max())!r}")
if ctrl:
    print(f"  same-prompt re-run control:      max |h - h'|  = "
          f"{max(c['h_maxdev'] for c in ctrl)!r}   (n={len(ctrl)})", flush=True)

# ------------------------------------- does movement predict getting it right?
pred = []
for lens in df.lens.unique():
    for l in LAYERS:
        d = df[(df.lens == lens) & (df.layer == l)].dropna(subset=[MV])
        if len(d) < 12:
            continue
        v, y = d[MV].to_numpy(float), d.text_ok.to_numpy(int)
        item = d.groupby("item").agg(mv=(MV, "mean"),
                                     ok=("text_item", "max")).reset_index()
        row = dict(lens=lens, layer=l, n_image=len(d),
                   auroc_image=auc(v[y == 1], v[y == 0]),
                   auroc_item=auc(item.mv[item.ok == 1], item.mv[item.ok == 0]),
                   n_item=len(item),
                   auroc_image_l2=auc(d[RL][y == 1], d[RL][y == 0]))
        row["r_point_biserial"] = (float(sps.pearsonr(v, y).statistic)
                                   if v.std() > 0 else 0.0)
        row["p_point_biserial"] = (float(sps.pearsonr(v, y).pvalue)
                                   if v.std() > 0 else 1.0)
        gap = (d.margin_correct - d.margin_swapped).to_numpy(float)
        row["spearman_move_gap"] = (float(sps.spearmanr(v, gap).statistic)
                                    if v.std() > 0 else 0.0)
        # The control the claim needs.  `text_ok` is the SIGN of the behavioural
        # gap and |gap| is its MAGNITUDE, and movement is bound to grow with the
        # magnitude: two captions the model processed very differently produce
        # both a big gap and a big change in the gradient.  So the question is
        # not whether movement tracks the gap -- it must -- but whether it says
        # anything about the sign once the magnitude is known.  Rank-residualise
        # movement on |gap| and score that.
        ag = np.abs(gap)
        row["spearman_move_absgap"] = (float(sps.spearmanr(v, ag).statistic)
                                       if v.std() > 0 else 0.0)
        rv, ra = sps.rankdata(v), sps.rankdata(ag)
        if rv.std() > 0 and ra.std() > 0:
            beta = np.cov(rv, ra)[0, 1] / np.var(ra)
            res = rv - beta * (ra - ra.mean())
        else:
            res = rv
        row["auroc_image_partial"] = auc(res[y == 1], res[y == 0])
        pred.append(row)
pred = pd.DataFrame(pred)
print("\nDOES MAP MOVEMENT PREDICT THE MODEL GETTING THE PAIR RIGHT?"
      "\n  (AUROC 0.5 is no information; the prior lenses cannot move, so"
      " theirs is 0.5 by construction)\n",
      pred.round(4).to_string(index=False), flush=True)

out = {"repo": a.repo, "dataset": "facebook/winoground",
       "n_records_downloaded": len(RECS), "n_items_used": n_items,
       "n_skipped": skipped, "question": QUESTION,
       "layers": LAYERS, "n_layers": m.n_layers, "grid": G, "coarse": C,
       "regions": R, "n_registers": int(REG.sum()),
       "mean_images": nm, "ig_steps": a.ig_steps,
       "target": TARGET, "target_why": why, "saturated": saturated,
       "answer_tokens": {"yes": m.tokenizer.decode([YES]),
                         "no": m.tokenizer.decode([NO])},
       "saturation": st.jsonable(satg.round(6).reset_index().to_dict("records")),
       "winoground_score": score,
       "by_tag": st.jsonable(by_tag.round(4).to_dict("records")),
       "movement": st.jsonable(mv.round(8).to_dict("records")),
       "zero": zero,
       "max_h_dev": float(df.h_maxdev.max()),
       "control_same_prompt_max_h_dev": (max(c["h_maxdev"] for c in ctrl)
                                         if ctrl else None),
       "predicts": st.jsonable(pred.round(6).to_dict("records")),
       "minutes": round((time.time() - t0) / 60, 1)}
for lens in df.lens.unique():
    d = df[(df.lens == lens)
           & (df.layer == LAYERS[len(LAYERS) // 2])].dropna(subset=[MV])
    if len(d) > 8:
        out[f"ci_movement_{lens}"] = st.cluster_bootstrap_ci(d, MV, "item")
for other in PRIOR:
    out[f"VJ-Lens_vs_{other}"] = st.wilcoxon(df, other, "VJ-Lens", MV, item="i")
json.dump(st.jsonable(out), open(f"{a.out}/{tag}_stats.json", "w"), indent=2,
          default=float)
json.dump(st.jsonable({"target": TARGET, "layers": LAYERS, "grid": G,
                       "coarse": C, "lenses": LENSES, "cases": cases}),
          open(f"{a.out}/{tag}_cases.json", "w"), default=float)
print(f"\nwrote {a.out}/{tag}_{{raw.parquet,sat.parquet,stats.json,cases.json}}"
      f"   {out['minutes']} min", flush=True)
print("done", flush=True)
