# SPDX-License-Identifier: Apache-2.0
r"""Is the causal-map comparison fair once every lens is given the region?

The objection this answers, stated plainly: our region reading is a SUM of
per-patch (gradient . deviation) terms, an extensive quantity that grows with
the region, while the logit lens and the J-lens were designed to read one
position.  In the published table ours is summed over a region and the priors
are averaged over the same region, and the size sweep that motivates the region
coordinate is run for our transport only.  So the published margin could in
principle be a property of the aggregation we chose for each side, not of the
transport we claim credit for.  "Give the box to every lens" is ambiguous --
pool the hidden states then read, read per patch then sum, mean, or max -- and
the choice can decide the result.

This experiment removes the ambiguity by running the full factorial

    TRANSPORT     x  AGGREGATION        x  REGION SIZE
    ------------     ----------------      -----------------------------
    VJ-Lens (IG)     sum   (ours)          block side 1, 2, 3, 4, 6, 7
    VJ-Lens (point)  mean  (the priors')   = 1, 4, ~9, ~16, ~36, 49
    J-lens           max                     patches on the 14x14 grid
    logit lens       pool  (mean hidden
    gradient norm          state, then
    random                 transport)

with every cell graded against the same region-cover ground truth, computed the
same way as code/exp/60_causalmap.py computes it (cover the region with the
corpus-mean activations, watch the answer's logit fall), on the same model, the
same layers, the same photographs, and the same per-photograph vocabulary as
the published qwen2b_big run.  The seeds and filters are copied verbatim from
60_causalmap.py so the item stream and the scoring vocabulary are bit-identical
to the published ones; the side-2 partition here IS the published partition.

Aggregation definitions, per transport, for a region R:

    sum    sum_{p in R} s_t(p)          the paper's choice for VJ-Lens
    mean   (1/|R|) sum_{p in R} s_t(p)  the paper's choice for the priors
    max    max_{p in R} s_t(p)
    pool   read the POOLED state: unembed(mean_R h) for the logit lens,
           unembed((mean_R h) J^T) for the J-lens, (mean_R gbar).(mean_R dev)
           for the gradient transports, ||mean_R g|| for gradient norm.
           For the two linear-in-h lenses pool differs from mean only through
           the final RMSNorm; for the gradient transports it is a genuinely
           different quantity, full of cross terms.  For `random` pool
           degenerates to mean and is kept only so the grid is complete.

Within one partition every region has the same patch count except where the
grid is ragged (sides 3, 4, 6 do not divide 14) or a register patch was
excluded, so sum and mean mostly differ only there -- reported anyway, because
"mostly" is exactly the kind of claim this experiment exists to check.

`random` is fixed per photograph-layer and passes through the same
aggregations: it is the measured chance floor at every region size, beside the
analytic 1/R.

The grading lines (score = map[concept ids].max over ids, Spearman against the
truth, argmax hit, top-3 hit, z-scored map against the human box) are mirrored
from 60_causalmap.py rather than reinvented, and the analysis answers, in
order and clustered over photographs:

  Q1  at matched region size AND matched aggregation, does our transport still
      beat the two prior lenses -- at every size, not just the best one;
  Q2  the strongest honest version of the priors' case: each prior lens at the
      aggregation that maximises ITS OWN score, against us at our published
      aggregation, per region size;
  Q3  an explicit decomposition of the published margin at the published size
      into a transport component (matched aggregation) and an aggregation
      component (what the paper's aggregation choice cost each prior lens).

Whatever these tables say is the answer; nothing here is tuned.
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
ap.add_argument("--sides", default="1,2,3,4,6,7",
                help="block sides of the region partitions; side s gives\n"
                     "regions of about s*s patches, and side 2 is the\n"
                     "published partition")
ap.add_argument("--ig-steps", type=int, default=4)
ap.add_argument("--max-rank", type=int, default=2)
ap.add_argument("--chunk", type=int, default=16,
                help="regions covered per forward pass of the ground truth")
ap.add_argument("--vocab", type=int, default=36,
                help="scoring words per photograph, as the published run")
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--boot", type=int, default=8000)
ap.add_argument("--out", default="/workspace/vlm/results/fair")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
SIDES = [int(s) for s in a.sides.split(",")]
t0 = time.time()

# ---- setup copied line for line from 60_causalmap.py, so the layers, the
# category list, the item stream and the vocabulary draws are the published ones
m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
LAYERS = [int(round(f * (m.n_layers - 1)))
          for f in np.linspace(0.20, 0.85, a.n_layers)]
J = None
if a.jlens:
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))

    # Read where the lens exists.  A fitted Jacobian lens only has a transport
    # at the layers it was fitted at; chosen from the model's geometry instead,
    # the depths miss it and the arm is dropped in silence.
    LAYERS, _ = readout.snap_to_lens(LAYERS, J)
CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
print(m, f"\nlayers {LAYERS}   {len(CATS)} categories   sides {SIDES}",
      flush=True)

PROMPT = "Describe this image."


def prep(path):
    return m.chat(PROMPT, [corpus.load_image(path, square=a.square)])


pool_items = cocodata.concept_items(6000, categories=CATS, min_area=0.04,
                                    seed=7)
b0 = prep(pool_items[0]["path"])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
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

# ---- the partitions: one C x C block grid per side, register tokens excluded
# exactly as in 60_causalmap.py.  side 2 reproduces the published regions.
PARTS = []
for s in SIDES:
    if s > G:
        continue
    Cs = int(np.ceil(G / s))
    regions, rects, keep = [], [], []
    for ry in range(Cs):
        for rx in range(Cs):
            y0, y1 = ry * s, min((ry + 1) * s, G)
            x0, x1 = rx * s, min((rx + 1) * s, G)
            idx = [y * G + x for y in range(y0, y1) for x in range(x0, x1)
                   if not bool(REG[y * G + x])]
            if idx:
                regions.append(torch.tensor(idx))
                rects.append((y0, y1, x0, x1))
                keep.append(ry * Cs + rx)
    PARTS.append(dict(side=s, size=s * s, C=Cs, regions=regions, rects=rects,
                      keep=keep,
                      cnt=torch.tensor([len(r) for r in regions],
                                       dtype=torch.float32)))
print(f"{P} tokens on {G}x{G}; partitions "
      + ", ".join(f"side {p['side']}: {len(p['regions'])} regions"
                  for p in PARTS)
      + f"; means from {nm} ({time.time()-t0:.0f}s)", flush=True)


def inside_regions(boxes, part) -> set:
    """Region indices the object's boxes cover, by the box_tokens rule.

    A region counts as the object's if some box covers at least 25% of the
    region's block -- the same threshold cocodata.box_tokens uses per cell, and
    identical to it whenever the side divides the grid.
    """
    ins = set()
    for j, (y0, y1, x0, x1) in enumerate(part["rects"]):
        area = (y1 - y0) * (x1 - x0)
        for bx in boxes:
            bx0, by0, bw, bh = bx
            ov = (max(0.0, min((bx0 + bw) * G, x1) - max(bx0 * G, x0))
                  * max(0.0, min((by0 + bh) * G, y1) - max(by0 * G, y0)))
            if ov >= 0.25 * area:
                ins.add(j)
                break
    return ins


def ig_gradients(model, batch, token_ids, layer, positions, h_clean, mean,
                 steps):
    """``[k, P, d]`` -- the per-patch path-averaged gradient.

    vmap.ig_map with one change: it returns the per-patch path-mean gradient
    itself rather than only its dot with the deviation, because the pooled
    (mean-then-transport) aggregation needs the field, not just the reading.
    Midpoint rule, batch-of-steps trick, both as in vmap.ig_map.
    """
    answer_pos = batch["input_ids"].shape[1] - 1
    pos = torch.as_tensor(positions, dtype=torch.long)
    last = model.n_layers - 1
    big = model.replicate(batch, steps) if steps > 1 else batch
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    h = h_clean.to(model.device).float()
    mu = mean.to(h.device).float()
    dev = h - mu
    x = (mu[None] + alphas[:, None, None].to(h.device) * dev[None]
         ).detach().clone().requires_grad_(True)

    def hook(mod, inp, o, _x=x, _p=pos):
        tup = not torch.is_tensor(o)
        hh = (o[0] if tup else o).clone()
        hh[:, _p.to(hh.device)] = _x.to(hh.dtype)
        return (hh, *o[1:]) if tup else hh

    hd = model.layers[layer].register_forward_hook(hook)
    try:
        with (ActivationRecorder(model.layers, at=[last]) as rec,
              torch.enable_grad()):
            model.forward(big)
            h_final = rec.activations[last][:, answer_pos]
            logits = model.unembed(h_final)
            rows = []
            for j, t in enumerate(token_ids):
                g, = torch.autograd.grad(
                    logits[:, t].sum(), x,
                    retain_graph=(j < len(token_ids) - 1))
                rows.append(g.float().mean(0).cpu())
    finally:
        hd.remove()
    return torch.stack(rows)


def agg_readings(part, M):
    """``{agg: [k, R]}`` for the three per-patch aggregations of ``M [k, P]``."""
    sums = torch.stack([M[:, r].sum(-1) for r in part["regions"]], -1)
    mx = torch.stack([M[:, r].max(-1).values for r in part["regions"]], -1)
    return {"sum": sums, "mean": sums / part["cnt"], "max": mx}


def region_means(part, X):
    """``[.., R, d]`` region means of a ``[.., P, d]`` field."""
    return torch.stack([X[..., r, :].mean(-2) for r in part["regions"]], -2)


items = cocodata.concept_items(a.n * 14, categories=CATS, min_area=0.06,
                               seed=808, tokenizer=m.tokenizer)
Cfilt = max(2, G // 2)          # the published run's pool=2 grid, for the
                                # object-size filter that selects the images
rows, kept, checked_side2 = [], 0, False
for k, it in enumerate(items):
    if kept >= a.n:
        break
    c = it["concept"]
    if c not in CAT_IDS:
        continue
    boxes = cocodata.object_boxes(it["image_id"], c)
    if not boxes:
        continue
    inside7 = sorted({t for bx in boxes
                      for t in cocodata.box_tokens(bx, Cfilt)})
    if not (2 <= len(inside7) <= 0.6 * Cfilt * Cfilt):
        continue
    b = prep(it["path"])
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    ids = CAT_IDS[c]
    rngv = np.random.default_rng(900 + k)
    others = [x for x in CATS if x != c]
    pick = list(rngv.choice(others, size=min(max(a.vocab - 1, 1), len(others)),
                            replace=False))
    VOC = sorted({t for w in pick for t in CAT_IDS[w]} | set(ids))
    VPOS = {t: i for i, t in enumerate(VOC)}
    col = [VPOS[t] for t in ids]
    APOS = b["input_ids"].shape[1] - 1
    A = readout.residuals(m, b, LAYERS + [m.n_layers - 1])
    lg0 = m.unembed(A[m.n_layers - 1][-1:].to(m.device))[0][VOC].float().cpu()
    if int((lg0 > lg0[col].max()).sum()) > a.max_rank:
        continue
    base = float(lg0[col].max())
    kept += 1

    INS = {p["side"]: inside_regions(boxes, p) for p in PARTS}
    if not checked_side2:
        # the side-2 inside set must equal 60_causalmap's box_tokens set on a
        # grid the side divides evenly; if it does not, the box grading is not
        # the published grading and every hit_box column below is suspect
        p2 = next((p for p in PARTS if p["side"] == 2), None)
        if p2 is not None and G % 2 == 0:
            cells = {p2["keep"][j] for j in INS[2]}
            print(f"  [check] side-2 inside vs box_tokens: "
                  f"{len(cells ^ set(inside7))} mismatches", flush=True)
        checked_side2 = True

    # one backward per word gives the point gradient at every layer at once
    gpt_all = vmap.raw_gradients(m, b, VOC, LAYERS, ip.tolist())
    gpt_all = {l: g.cpu() for l, g in gpt_all.items()}

    for l in LAYERS:
        h = A[l][ip].float().cpu()
        dev = h - MEANS[l]
        gbar = ig_gradients(m, b, VOC, l, ip.tolist(), A[l][ip], MEANS[l],
                            a.ig_steps)
        gpt = gpt_all[l]
        rng_r = np.random.default_rng(4242 + 1000 * k + l)

        # per-patch readings [k, P] for every transport
        say = m.unembed(h.to(m.device))[:, VOC].float().cpu().T.contiguous()
        PATCH = {"logit lens": say,
                 "VJ-Lens": (gbar * dev).sum(-1),
                 "VJ-Lens (point)": (gpt * dev).sum(-1),
                 "gradient norm": gpt.norm(dim=-1),
                 "random": torch.from_numpy(
                     rng_r.standard_normal((len(VOC), P))).float()}
        if J is not None and l in J.jacobians:
            hJ = h @ J.jacobians[l].T
            PATCH["J-lens"] = m.unembed(hJ.to(m.device))[:, VOC] \
                .float().cpu().T.contiguous()

        # ---- ground truth, all partitions in one chunked stream, computed
        # exactly as 60_causalmap.py computes it
        ALL = [(pi, ri, r) for pi, p in enumerate(PARTS)
               for ri, r in enumerate(p["regions"])]
        truth = {pi: np.zeros(len(p["regions"])) for pi, p in enumerate(PARTS)}
        for s0 in range(0, len(ALL), a.chunk):
            grp = ALL[s0:s0 + a.chunk]
            big = m.replicate(b, len(grp))

            def hook(mod, inp, o, _g=grp, _l=l):
                tup = not torch.is_tensor(o)
                hh = (o[0] if tup else o).clone()
                for j, (_pi, _ri, r) in enumerate(_g):
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
            for j, (pi, ri, r) in enumerate(grp):
                truth[pi][ri] = base - float(vv[j, col].max())

        # ---- every cell of the factorial at this photograph and layer
        for pi, part in enumerate(PARTS):
            tru = truth[pi]
            tru_arg = int(np.argmax(tru))
            tru_top3 = set(np.argsort(-tru)[:3].tolist())
            ins = INS[part["side"]]

            hbar = region_means(part, h)          # [R, d]
            devbar = region_means(part, dev)
            gbar_r = region_means(part, gbar)     # [k, R, d]
            gpt_r = region_means(part, gpt)
            for name, M in PATCH.items():
                per = agg_readings(part, M)
                if name == "logit lens":
                    per["pool"] = m.unembed(hbar.to(m.device))[:, VOC] \
                        .float().cpu().T.contiguous()
                elif name == "J-lens":
                    per["pool"] = m.unembed((hbar @ J.jacobians[l].T)
                                            .to(m.device))[:, VOC] \
                        .float().cpu().T.contiguous()
                elif name == "VJ-Lens":
                    per["pool"] = (gbar_r * devbar[None]).sum(-1)
                elif name == "VJ-Lens (point)":
                    per["pool"] = (gpt_r * devbar[None]).sum(-1)
                elif name == "gradient norm":
                    per["pool"] = gpt_r.norm(dim=-1)
                else:                              # random: pool == mean
                    per["pool"] = per["mean"].clone()

                for agg, Mreg in per.items():
                    # grading mirrored from 60_causalmap.py
                    score = Mreg[col].max(0).values.float().numpy()
                    rho = sps.spearmanr(score, tru).statistic
                    arg = int(np.argmax(score))
                    z = vmap.normalise(Mreg)[col].max(0).values.float().numpy()
                    zarg = int(np.argmax(z))
                    rows.append(dict(
                        i=k, concept=c, key=f"{k}_{l}", layer=l,
                        depth=round(l / (m.n_layers - 1), 3),
                        transport=name, agg=agg,
                        side=part["side"], size=part["size"],
                        R=len(part["regions"]),
                        chance=1.0 / len(part["regions"]),
                        spearman=float(rho),
                        hit_cause=int(arg == tru_arg),
                        top3_cause=int(arg in tru_top3),
                        hit_box=int(zarg in ins),
                        truth_in_box=int(tru_arg in ins),
                        n_box=len(ins),
                        truth_max=float(tru.max())))
    if kept % 2 == 0 or k % 25 == 0:
        el = time.time() - t0
        print(f"  {kept}/{a.n} kept  {el:.0f}s  "
              f"eta {el/max(kept,1)*(a.n-kept)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")

def _analyse():
    # everything below only reads the frame and the config; a failure
    # here must not cost the run, so the caller guards it
    # ---------------------------------------------------------------- the grid
    grid = df.groupby(["transport", "agg", "size"]).agg(
        n=("i", "count"), spearman=("spearman", "mean"),
        hit_cause=("hit_cause", "mean"), top3_cause=("top3_cause", "mean"),
        hit_box=("hit_box", "mean"), chance=("chance", "mean")).reset_index()
    grid.to_csv(f"{a.out}/{tag}_grid.csv", index=False)
    for v in ("spearman", "hit_cause"):
        print(f"\nFULL GRID -- {v} (rows transport x aggregation, columns "
              f"region size in patches)\n",
              df.pivot_table(index=["transport", "agg"], columns="size",
                             values=v).round(3).to_string(), flush=True)
    print("\nchance floor: analytic 1/R per size\n",
          df.groupby("size")["chance"].mean().round(4).to_string(),
          "\nmeasured (random transport, hit_cause by aggregation)\n",
          df[df.transport == "random"].pivot_table(
              index="agg", columns="size", values="hit_cause")
          .round(3).to_string(), flush=True)


    def pair(tA, aA, tB, aB, size, value):
        dA = df[(df.transport == tA) & (df["agg"] == aA) & (df["size"] == size)]
        dB = df[(df.transport == tB) & (df["agg"] == aB) & (df["size"] == size)]
        A = dA.set_index("key")[value]
        B = dB.set_index("key")[value]
        ix = A.index.intersection(B.index)
        return pd.DataFrame({"a": A.loc[ix].to_numpy(float),
                             "b": B.loc[ix].to_numpy(float),
                             "i": [int(s.split("_")[0]) for s in ix]}).dropna()


    def boot_ci(p, nb, seed=0):
        """Cluster bootstrap over photographs of mean(a - b)."""
        g = [v.a.to_numpy() - v.b.to_numpy() for _, v in p.groupby("i")]
        if not g:
            return [float("nan"), float("nan")]
        rng = np.random.default_rng(seed)
        K = len(g)
        bs = np.empty(nb)
        for j in range(nb):
            idx = rng.integers(0, K, K)
            bs[j] = np.concatenate([g[i] for i in idx]).mean()
        return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


    def wilc_p(p):
        d = p.a - p.b
        if len(d) < 6 or (d == 0).all():
            return float("nan")
        return float(sps.wilcoxon(p.a, p.b).pvalue)


    PRIORS = [x for x in ("J-lens", "logit lens") if x in set(df.transport)]
    SIZES = sorted(df["size"].unique())
    AGGS = ["sum", "mean", "max", "pool"]
    out = {"repo": a.repo, "n": int(df.i.nunique()), "grid": G,
           "layers": LAYERS, "sides": SIDES,
           "sizes": {int(s): int(df[df["size"] == s].R.iloc[0]) for s in SIZES},
           "vocab": a.vocab, "ig_steps": a.ig_steps, "max_rank": a.max_rank,
           "grid_means": st.jsonable(grid.round(4).to_dict("records"))}


    def cellmean(t, agg, size, value):
        d = df[(df.transport == t) & (df["agg"] == agg) & (df["size"] == size)]
        return float(d[value].mean()) if len(d) else float("nan")


    # Q1 -- matched size AND matched aggregation, ours vs each prior, every cell
    q1 = []
    for size in SIZES:
        for agg in AGGS:
            for prior in PRIORS:
                for value in ("spearman", "hit_cause"):
                    p = pair("VJ-Lens", agg, prior, agg, size, value)
                    q1.append(dict(
                        size=int(size), agg=agg, prior=prior, metric=value,
                        ours=round(cellmean("VJ-Lens", agg, size, value), 4),
                        theirs=round(cellmean(prior, agg, size, value), 4),
                        delta=round(float((p.a - p.b).mean()), 4),
                        p=wilc_p(p)))
    out["q1_matched"] = q1
    print("\nQ1  MATCHED SIZE AND AGGREGATION -- VJ-Lens minus each prior "
          "(spearman)\n",
          pd.DataFrame([r for r in q1 if r["metric"] == "spearman"])
          .pivot_table(index=["prior", "agg"], columns="size", values="delta")
          .round(3).to_string(), flush=True)

    # Q2 -- each prior at the aggregation that maximises ITS OWN score, vs ours at
    # the published aggregation.  The strongest honest version of the priors' case.
    q2 = []
    for prior in PRIORS + ["gradient norm"]:
        if prior not in set(df.transport):
            continue
        for size in SIZES:
            for value in ("spearman", "hit_cause"):
                best_agg = max(AGGS, key=lambda g: np.nan_to_num(
                    cellmean(prior, g, size, value), nan=-9))
                p = pair("VJ-Lens", "sum", prior, best_agg, size, value)
                q2.append(dict(
                    prior=prior, size=int(size), metric=value, best_agg=best_agg,
                    ours=round(cellmean("VJ-Lens", "sum", size, value), 4),
                    theirs=round(cellmean(prior, best_agg, size, value), 4),
                    delta=round(float((p.a - p.b).mean()), 4),
                    ci=boot_ci(p, a.boot), p=wilc_p(p),
                    chance=round(float(df[df["size"] == size].chance.mean()), 4)))
    out["q2_their_best"] = q2
    q2df = pd.DataFrame(q2)
    for value in ("spearman", "hit_cause"):
        print(f"\nQ2  EACH PRIOR AT ITS OWN BEST AGGREGATION vs OURS (sum) "
              f"-- {value}\n",
              q2df[q2df.metric == value][
                  ["prior", "size", "best_agg", "ours", "theirs", "delta", "p"]]
              .to_string(index=False), flush=True)

    # the single strongest cell any prior reaches anywhere in the grid, vs our
    # published cell and our best cell -- the one-line hostile-expert summary
    best_cells = {}
    for t in set(df.transport):
        for value in ("spearman", "hit_cause"):
            sub = grid[grid.transport == t]
            r = sub.loc[sub[value].idxmax()]
            best_cells[f"{t}|{value}"] = dict(
                agg=r["agg"], size=int(r["size"]), value=round(float(r[value]), 4))
    out["best_cell_anywhere"] = best_cells

    # Q3 -- decompose the published margin at the published size (side 2) into a
    # transport component (matched agg) and an aggregation component
    pub_size = 4 if 4 in [p["size"] for p in PARTS] else SIZES[0]
    q3 = []
    for prior in PRIORS:
        for value in ("spearman", "hit_cause"):
            ours_pub = cellmean("VJ-Lens", "sum", pub_size, value)
            theirs_pub = cellmean(prior, "mean", pub_size, value)
            best_agg = max(AGGS, key=lambda g: np.nan_to_num(
                cellmean(prior, g, pub_size, value), nan=-9))
            theirs_best = cellmean(prior, best_agg, pub_size, value)
            q3.append(dict(
                prior=prior, metric=value, size=int(pub_size),
                published_margin=round(ours_pub - theirs_pub, 4),
                transport_component=round(ours_pub - theirs_best, 4),
                aggregation_component=round(theirs_best - theirs_pub, 4),
                their_best_agg=best_agg,
                ours_across_aggs={g: round(cellmean("VJ-Lens", g, pub_size,
                                                    value), 4) for g in AGGS},
                theirs_across_aggs={g: round(cellmean(prior, g, pub_size,
                                                      value), 4) for g in AGGS}))
    out["q3_decomposition"] = q3
    print("\nQ3  DECOMPOSITION AT THE PUBLISHED SIZE "
          f"({pub_size} patches): published margin = transport (matched agg) "
          "+ aggregation (what the paper's agg choice cost them)\n",
          pd.DataFrame(q3)[["prior", "metric", "published_margin",
                            "transport_component", "aggregation_component",
                            "their_best_agg"]].to_string(index=False), flush=True)

    out["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(st.jsonable(out), open(f"{a.out}/{tag}_stats.json", "w"), indent=2,
              default=float)
    print("done", flush=True)


try:
    _analyse()
except Exception:
    import traceback
    traceback.print_exc()
    print("ANALYSIS FAILED -- the raw parquet is saved; analyse "
          "offline", flush=True)
