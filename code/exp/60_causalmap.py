"""Which lens tells you what the picture is actually doing to the answer?

The pointing game grades a map against a human's bounding box, which makes it a
test of whether the map finds the *object*.  That is the right test for a map of
content and the wrong test for a map of influence, and it quietly assumes the
two coincide.

There is a ground truth that does not assume it.  Cover a region of the picture
-- replace its activations with an average photograph's -- and measure what
happens to the word the model was going to say.  Do that for every region and
you have the true causal map, produced by intervention rather than by
annotation.  Then every lens can be scored on the only question a lens is for:

    does this reading tell me what covering this place would do?

Three lenses are read at the same regions, the same layers, the same
photographs:

    logit lens    W_U h[R]                     content, no transport
    J-lens        W_U J_l h[R]                 content, corpus-averaged transport
    VJ-Lens       sum_p grad . (h[p] - hbar)   influence, this pass's transport

reported as Spearman against the true map, the share of photographs where the
lens's strongest region is the truly strongest region, and -- for VJ-Lens, which
is the only one with units -- the relative error of the prediction itself.

The same run records the human box, so the two questions can be compared
side by side: how well each lens finds the object, and how well each lens finds
the influence.  If those two columns disagree, that disagreement is the finding.
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
from scipy import stats as sps

import jvlm
from jlens.hooks import ActivationRecorder
from jvlm import cocodata, corpus, readout, stats as st, vmap

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--jlens", default=None)
ap.add_argument("--n", type=int, default=70)
ap.add_argument("--mean-images", type=int, default=90)
ap.add_argument("--n-layers", type=int, default=3)
ap.add_argument("--pool", type=int, default=2, help="token block side")
ap.add_argument("--ig-steps", type=int, default=4)
ap.add_argument("--baselines", action="store_true",
                help="add the standard saliency constructions as\n                     extra arms: gradient norm, gradient x\n                     activation, and the deviation norm")
ap.add_argument("--max-rank", type=int, default=0,
                help="how far down the object may sit in the model's\n                     own ranking and still be used")
ap.add_argument("--chunk", type=int, default=12,
                help="regions covered per forward pass")
ap.add_argument("--vocab", type=int, default=40,
                help="scoring words per photograph; the cost is one\n                     backward pass per word, so this is the main dial")
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/causalmap")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
LAYERS = [int(round(f * (m.n_layers - 1)))
          for f in np.linspace(0.20, 0.85, a.n_layers)]
J = None
if a.jlens:
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))
CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
FLAT = sorted({t for v in CAT_IDS.values() for t in v})
POSI = {t: i for i, t in enumerate(FLAT)}
WORD_OF = {}
for _c, _ids in CAT_IDS.items():
    for _t in _ids:
        WORD_OF.setdefault(POSI[_t], _c)
print(m, f"\nlayers {LAYERS}   {len(CATS)} categories", flush=True)

PROMPT = "Describe this image."


def prep(path):
    return m.chat(PROMPT, [corpus.load_image(path, square=a.square)])


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

# the pooled regions: a C x C grid of blocks over the G x G token grid, with
# register tokens excluded so a sink does not stand in for a place
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


def region_sum(per_patch: torch.Tensor) -> torch.Tensor:
    """[k, P] per-patch readings -> [k, R] region readings, by SUM.

    Summing rather than averaging is what the definition says and what makes
    disjoint regions add; averaging a gradient over a region and then dotting it
    with an averaged deviation is a different quantity, full of cross terms
    between one patch's gradient and another patch's content.
    """
    return torch.stack([per_patch[:, r].sum(-1) for r in REGIONS], dim=-1)


items = cocodata.concept_items(a.n * 14, categories=CATS, min_area=0.06, seed=808,
                               tokenizer=m.tokenizer)
rows, cards, kept = [], [], 0
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
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    ids = CAT_IDS[c]
    # the gradient costs one backward pass per token, so the scoring vocabulary
    # is the run's main cost.  A fixed random spread of other categories is
    # enough for a z-score and for a legible top word, and keeps the run inside
    # a session rather than a night.
    rngv = np.random.default_rng(900 + k)
    others = [x for x in CATS if x != c]
    pick = list(rngv.choice(others, size=min(max(a.vocab - 1, 1), len(others)),
                            replace=False))
    VOC = sorted({t for w in pick for t in CAT_IDS[w]} | set(ids))
    VPOS = {t: i for i, t in enumerate(VOC)}
    WORD_V = {}
    for w in [c] + pick:
        for t in CAT_IDS[w]:
            WORD_V.setdefault(VPOS[t], w)
    col = [VPOS[t] for t in ids]
    APOS = b["input_ids"].shape[1] - 1
    A = readout.residuals(m, b, LAYERS + [m.n_layers - 1])
    lg0 = m.unembed(A[m.n_layers - 1][-1:].to(m.device))[0][VOC].float().cpu()
    # the model must be about to say the object, or the lens is being asked to
    # explain a word the model was never going to produce.  Top-1 out of the
    # whole category vocabulary is strict enough to throw most photographs away,
    # so the rank a photograph must clear is a flag.
    if int((lg0 > lg0[col].max()).sum()) > a.max_rank:
        continue
    base = float(lg0[col].max())
    kept += 1

    for l in LAYERS:
        # ---- ground truth: cover each region, watch the word fall.
        # The regions do not interact, so they run as a batch: one forward pass
        # covers `chunk` regions at once instead of one each, which is what
        # makes a 49-region map affordable for every photograph and layer.
        truth = np.zeros(R)
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
                    vv = m.unembed(hv)[:, VOC].float().cpu()
            finally:
                hd.remove()
            truth[s0:s0 + len(grp)] = base - vv[:, col].max(-1).values.numpy()

        # ---- the three lenses, read at the same regions
        h = A[l][ip].float().cpu()
        maps = {}
        lgm = m.unembed(h.to(m.device)).float().cpu()[:, VOC].T
        maps["logit lens"] = region_sum(lgm) / torch.tensor(
            [len(r) for r in REGIONS], dtype=torch.float32)
        if J is not None and l in J.jacobians:
            jl, _ = readout.lens_logits(m, J, b, layers=[l],
                                        positions=ip.tolist())
            maps["J-lens"] = region_sum(jl[l].float().cpu()[:, VOC].T
                                        ) / torch.tensor(
                [len(r) for r in REGIONS], dtype=torch.float32)
        maps["VJ-Lens (point)"] = region_sum(
            vmap.do_map(m, b, VOC, [l], ip.tolist(), means={l: MEANS[l]})[l])
        if a.baselines:
            # The three things a reviewer will ask why we did not simply use.
            # None is a lens -- they are the standard saliency constructions --
            # and two of them are ablations of ours: gradient-times-activation
            # is our reading with the baseline set to the origin, and the
            # deviation norm is our reading with the gradient thrown away.
            g = vmap.raw_gradients(m, b, VOC, [l], ip.tolist())[l]   # [k, P, d]
            hh = h.to(g.device)
            maps["gradient norm"] = region_sum(
                g.norm(dim=-1).cpu().expand(len(VOC), -1)
                if g.dim() == 2 else g.norm(dim=-1).cpu())
            maps["gradient x activation"] = region_sum((g * hh).sum(-1).cpu())
            dev = (hh - MEANS[l].to(g.device))
            maps["deviation norm"] = region_sum(
                dev.norm(dim=-1).cpu()[None].expand(len(VOC), -1).contiguous())
        if a.ig_steps > 1:
            maps["VJ-Lens"] = region_sum(
                vmap.ig_map(m, b, VOC, [l], ip.tolist(), means={l: MEANS[l]},
                            steps=a.ig_steps)[l])

        tru_arg = int(np.argmax(truth))
        tru_top3 = set(np.argsort(-truth)[:3].tolist())
        box = set(inside)
        # is the human's box where the causal action actually is?  If it is not,
        # then grading a map by whether it points at the box is grading it
        # against the wrong thing, and that has to be shown rather than assumed
        truth_in_box = int(KEEP[tru_arg] in box)
        truth_top3_in_box = int(any(KEEP[t] in box for t in tru_top3))
        # share of the total damage that the box accounts for.  Signed drops
        # can cancel, so this is a share of magnitude, not of a signed sum --
        # the previous form divided by a near-zero signed total and produced
        # numbers in the hundreds of millions.
        _mag = float(np.abs(truth).sum())
        box_share = float(np.abs([truth[j] for j in range(R)
                                  if KEEP[j] in box]).sum() / max(_mag, 1e-9))
        for name, mm in maps.items():
            score = mm[col].max(0).values.float().numpy()      # [R]
            rho = sps.spearmanr(score, truth).statistic
            arg = int(np.argmax(score))
            # the same map, z-scored down the vocabulary, is what a pointing
            # game would grade -- kept so the two questions sit side by side
            z = vmap.normalise(mm)[col].max(0).values.float().numpy()
            zarg = int(np.argmax(z))
            rel = np.nan
            if name.startswith("VJ-Lens"):
                rel = float(abs(score.sum() - truth.sum())
                            / max(abs(truth.sum()), 1e-6))
            rows.append(dict(i=k, concept=c, lens=name, cond=name,
                             key=f"{k}_{l}", layer=l,
                             depth=round(l / (m.n_layers - 1), 3),
                             spearman=float(rho),
                             hit_cause=int(arg == tru_arg),
                             top3_cause=int(arg in tru_top3),
                             hit_box=int(KEEP[zarg] in box),
                             n_box=len(inside), R=R,
                             rel_error=rel, truth_max=float(truth.max()),
                             truth_sum=float(truth.sum()),
                             truth_in_box=truth_in_box,
                             truth_top3_in_box=truth_top3_in_box,
                             box_share=box_share,
                             box_area=len(inside) / max(R, 1)))
        if l == LAYERS[len(LAYERS) // 2] and len(cards) < 14:
            # the card carries, per region: the true drop, each lens's score for
            # the photographed word, and each lens's single top word out of the
            # whole vocabulary -- the last is what makes a panel readable rather
            # than a wash of colour
            words = {}
            for n, mm in maps.items():
                z = vmap.normalise(mm)
                top = z.argmax(0)
                words[n] = dict(
                    word=[WORD_V.get(int(i), '?') for i in top],
                    z=[round(float(v), 3) for v in z.max(0).values],
                    score=[round(float(v), 5)
                           for v in mm[col].max(0).values])
            cards.append(dict(image_id=it["image_id"], concept=c, layer=l,
                              depth=round(l / (m.n_layers - 1), 3),
                              coarse=C, keep=KEEP, inside=inside,
                              path=it["path"], truth=truth.tolist(),
                              lenses=words))
    if kept % 3 == 0 or k % 25 == 0:
        el = time.time() - t0
        print(f"  {kept}/{a.n} kept  {el:.0f}s  "
              f"eta {el/max(kept,1)*(a.n-kept)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
json.dump(st.jsonable(cards), open(f"{a.out}/{tag}_cards.json", "w"))

agg = df.groupby(["lens", "layer"]).agg(
    n=("i", "count"), spearman=("spearman", "mean"),
    hit_cause=("hit_cause", "mean"), top3_cause=("top3_cause", "mean"),
    hit_box=("hit_box", "mean"), rel_error=("rel_error", "mean"),
    truth_in_box=("truth_in_box", "mean"),
    box_share=("box_share", "mean"),
    box_area=("box_area", "mean")).reset_index()
agg.to_csv(f"{a.out}/{tag}_summary.csv", index=False)
print("\nAGAINST THE MODEL'S OWN CAUSAL MAP\n",
      agg.round(4).to_string(index=False), flush=True)

best = df.groupby("lens").agg(
    spearman=("spearman", "mean"), hit_cause=("hit_cause", "mean"),
    top3_cause=("top3_cause", "mean"), hit_box=("hit_box", "mean"))
print("\nOVER ALL LAYERS  (chance for hit_cause is 1/R)\n",
      best.round(4).to_string(), flush=True)

out = {"repo": a.repo, "n": int(df.i.nunique()), "grid": G, "coarse": C,
       "regions": R, "layers": LAYERS, "n_layers": m.n_layers,
       "chance_cause": round(1.0 / R, 4),
       "max_rank": a.max_rank,
       "truth_in_box": round(float(df.truth_in_box.mean()), 4),
       "truth_top3_in_box": round(float(df.truth_top3_in_box.mean()), 4),
       "box_share": round(float(df.box_share.mean()), 4),
       "box_area": round(float(df.box_area.mean()), 4),
       "by_layer": agg.round(4).to_dict("records"),
       "over": best.round(4).reset_index().to_dict("records"),
       "minutes": round((time.time() - t0) / 60, 1)}
for lens in df.lens.unique():
    d = df[df.lens == lens]
    out[f"ci_{lens}"] = st.cluster_bootstrap_ci(d, "spearman", "concept")
# the paired comparisons a reviewer will ask for, clustered by photograph
for other in [x for x in df.lens.unique() if x != "VJ-Lens"]:
    if "VJ-Lens" in set(df.lens):
        out[f"VJ-Lens_vs_{other}"] = st.wilcoxon(df, other, "VJ-Lens",
                                                 "spearman", item="key") \
            if "item" in st.wilcoxon.__code__.co_varnames else \
            st.wilcoxon(df, other, "VJ-Lens", "spearman")
json.dump(st.jsonable(out), open(f"{a.out}/{tag}_stats.json", "w"), indent=2,
          default=float)
print("done", flush=True)
