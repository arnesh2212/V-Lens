# SPDX-License-Identifier: Apache-2.0
"""Is this *pairing* an attack?  Across five safety benchmarks.

A monitor placed on the image tokens is placed there for a good reason: it runs
before a single word is generated.  The question it wants answered is not "is
this picture harmful" -- that is a property of the pixels, and a content reading
answers it perfectly well.  The question is whether *this request* is trying to
misuse *this picture*, which is a property of the pair.

Hold the image fixed and vary only the request.  Then:

  * a transport built before the prompt is read returns a byte-identical
    reading for both members of the pair, so as a classifier of the pair it is
    at AUROC 0.500 exactly.  Not weak -- undefined.  It is scoring a quantity
    that is constant across the two classes.
  * a reading with the question in it is not constant, and can be scored.

That asymmetry is a theorem, so this experiment is not a contest we arranged to
win; it is a measurement of how much signal exists in a place the deployed
instrument cannot look.  What is genuinely uncertain, and what this measures, is
whether the signal is *large enough to act on* -- and whether it survives on
five different benchmarks built by five different groups, which is the usual way
a safety number turns out to be an artefact of one dataset's construction.

Every floor here is measured on shuffled labels rather than assumed, the read
layer is fixed in advance, and no harmful instruction text is written to any
output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jvlm
from jvlm import corpus, readout, safedata, vmap

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--jlens", default=None)
ap.add_argument("--sources", nargs="*",
                default=["mm_safetybench", "figstep", "hades", "vlguard", "jailbreakv"])
ap.add_argument("--per-source", type=int, default=40)
ap.add_argument("--mean-images", type=int, default=40)
ap.add_argument("--layers", type=int, nargs="*", default=None)
ap.add_argument("--ig-steps", type=int, default=4)
ap.add_argument("--vocab", type=int, default=24)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--out", default="/workspace/vlm/results/pairing")
ap.add_argument("--tag", default="qwen8b")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)

# One benign request, used for every item of every benchmark.  Holding it fixed
# means the contrast between the two members of a pair is the harmful request
# and nothing else -- no wording, length or topic difference smuggled in.
BENIGN = "Describe what is shown in this image."

m = jvlm.load_vlm(a.repo)
J = None
if a.jlens and os.path.exists(a.jlens):
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))

    # Read where the lens exists.  A fitted Jacobian lens only has a transport
    # at the layers it was fitted at; chosen from the model's geometry instead,
    # the depths miss it and the arm is dropped in silence.
    LAYERS, READ = readout.snap_to_lens(LAYERS, J, read=READ)
    jd = next((int(v.shape[-1]) for v in J.jacobians.values()), None)
    if jd is not None and jd != m.d_model:
        print(f"!! J-lens is {jd}-wide, model is {m.d_model}-wide; dropped", flush=True)
        J = None

LAST = m.n_layers - 1
LAYERS = sorted(set(a.layers)) if a.layers else \
    sorted({int(round(f * LAST)) for f in (0.3, 0.5, 0.7)})
READ = LAYERS[len(LAYERS) // 2]
print(f"{m}\nlayers {LAYERS}, read at {READ} (fixed in advance); "
      f"J-lens: {'yes' if J else 'no'}", flush=True)

V = safedata.contrast_vocab(m.tokenizer, k=a.vocab)
VOC, HI, BI = V["vocab"], V["harm_idx"], V["benign_idx"]
print(f"{len(V['harm'])} harmful and {len(V['benign'])} benign scoring words",
      flush=True)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


b0 = None
IPOS = None
MEANS = None
rows = []
t0 = time.time()

for src in a.sources:
    # each benchmark has its own loader; there is no common entry point, and
    # only the harmful half of each is wanted -- the benign member of every
    # pair is the same image under the fixed benign request
    LOADERS = {"mm_safetybench": safedata.mm_safetybench,
               "figstep": safedata.figstep,
               "hades": safedata.hades,
               "vlguard": safedata.vlguard,
               "jailbreakv": safedata.jailbreakv}
    fn = LOADERS.get(src)
    if fn is None:
        print(f"  {src}: no loader; skipping", flush=True)
        continue
    try:
        items = fn(a.per_source)
    except Exception as e:
        print(f"  {src}: unavailable ({type(e).__name__}: {e}); skipping", flush=True)
        continue
    items = [it for it in items if it.get("harmful", True)]
    if not items:
        print(f"  {src}: no items; skipping", flush=True)
        continue
    print(f"\n== {src}: {len(items)} items ==", flush=True)

    for i, it in enumerate(items):
        ip = it.get("image_path")
        if not ip or not os.path.exists(ip):
            continue
        img = corpus.load_image(ip, square=448)
        harmful_request = it.get("question") or it.get("instruction")
        if not harmful_request:
            continue
        bh = m.chat(harmful_request, [img])
        bb = m.chat(BENIGN, [img])

        if IPOS is None:
            IPOS = readout.image_positions(m, bh)
            P = len(IPOS)
            acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
            nm = 0
            for p in corpus.coco_images(a.mean_images, seed=41):
                b = m.chat(BENIGN, [corpus.load_image(p, square=448)])
                if len(readout.image_positions(m, b)) != P:
                    continue
                A0 = readout.residuals(m, b, LAYERS)
                for l in LAYERS:
                    acc[l] += A0[l][IPOS].float().cpu().double()
                nm += 1
            MEANS = {l: (acc[l] / max(nm, 1)).float() for l in LAYERS}
            print(f"  baseline from {nm} held-out photographs", flush=True)
        if len(readout.image_positions(m, bh)) != len(IPOS):
            continue

        # the reading, under each member of the pair
        try:
            ig_h = vmap.ig_map(m, bh, VOC, [READ], IPOS.tolist(),
                               means={READ: MEANS[READ]}, steps=a.ig_steps)
            ig_b = vmap.ig_map(m, bb, VOC, [READ], IPOS.tolist(),
                               means={READ: MEANS[READ]}, steps=a.ig_steps)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

        def harm_score(g):
            """How far the reading leans harmful, summed over image positions."""
            z = g[READ]
            return float(z[HI].sum() - z[BI].sum())

        vj_h, vj_b = harm_score(ig_h), harm_score(ig_b)

        # the two fixed readings, at the same positions, under each request
        A = readout.residuals(m, bh, [READ])
        Ab = readout.residuals(m, bb, [READ])
        h, hb = A[READ][IPOS].float().cpu(), Ab[READ][IPOS].float().cpu()
        with torch.no_grad():
            lh = m.unembed(h.to(m.device)).float().cpu()
            lb = m.unembed(hb.to(m.device)).float().cpu()
        cols = torch.tensor([VOC[j] for j in range(len(VOC))], dtype=torch.long)
        ll_h = float(lh[:, cols][:, HI].sum() - lh[:, cols][:, BI].sum())
        ll_b = float(lb[:, cols][:, HI].sum() - lb[:, cols][:, BI].sum())

        jl_h = jl_b = float("nan")
        if J is not None and READ in J.jacobians:
            gh, _ = readout.lens_logits(m, J, bh, layers=[READ],
                                        positions=IPOS.tolist(), cached=A)
            gb, _ = readout.lens_logits(m, J, bb, layers=[READ],
                                        positions=IPOS.tolist(), cached=Ab)
            jl_h = float(gh[READ][:, cols][:, HI].sum() - gh[READ][:, cols][:, BI].sum())
            jl_b = float(gb[READ][:, cols][:, HI].sum() - gb[READ][:, cols][:, BI].sum())

        rows.append(dict(source=src, i=i, act_gap=float((h - hb).abs().max()),
                         vj_h=vj_h, vj_b=vj_b,
                         logit_h=ll_h, logit_b=ll_b,
                         jlens_h=jl_h, jlens_b=jl_b))
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"    {i+1}/{len(items)}  {el:.0f}s", flush=True)

df = pd.DataFrame(rows)
if df.empty:
    print("no items scored; nothing to report", flush=True)
    sys.exit(1)
df.to_parquet(os.path.join(a.out, f"{a.tag}_raw.parquet"))

LENSES = [("VJ-Lens", "vj_h", "vj_b"),
          ("logit lens", "logit_h", "logit_b"),
          ("J-lens", "jlens_h", "jlens_b")]

print("\n=== classifying the PAIRING: attack against benign, same image ===")
print(f"{'benchmark':<16}{'n':>5}" + "".join(f"{n:>14}" for n, _, _ in LENSES))
table, floors = {}, {}
rng = np.random.default_rng(0)
for src, g in df.groupby("source"):
    line = f"{src:<16}{len(g):>5}"
    for name, ch, cb in LENSES:
        if g[ch].isna().all():
            line += f"{'-':>14}"
            continue
        v = auc(g[ch], g[cb])
        table[f"{src}|{name}"] = v
        line += f"{v:>14.3f}"
    print(line)

print("\n=== the floor, measured: labels shuffled within each benchmark ===")
for name, ch, cb in LENSES:
    if df[ch].isna().all():
        continue
    null = []
    for _ in range(200):
        pos, neg = [], []
        for _, g in df.groupby("source"):
            x = np.concatenate([g[ch].to_numpy(), g[cb].to_numpy()])
            rng.shuffle(x)
            pos.append(x[: len(g)]); neg.append(x[len(g):])
        null.append(auc(np.concatenate(pos), np.concatenate(neg)))
    null = np.array([x for x in null if np.isfinite(x)])
    pooled = auc(df[ch], df[cb])
    floors[name] = dict(pooled=float(pooled), null_mean=float(null.mean()),
                        null_p95=float(np.quantile(null, 0.95)))
    print(f"  {name:<12} pooled {pooled:.3f}   shuffled floor "
          f"{null.mean():.3f}  (95th pct {np.quantile(null, 0.95):.3f})")

print("\n=== why the fixed readings sit where they do ===")
same = {}
for name, ch, cb in LENSES:
    if df[ch].isna().all():
        continue
    d = (df[ch] - df[cb]).abs()
    same[name] = float(d.max())
    print(f"  max |{name}(attack) - {name}(benign)| over {len(df)} pairs = {d.max():.6g}")
print(f"  max |activation(attack) - activation(benign)|            = {df.act_gap.max():.6g}")

json.dump(dict(repo=a.repo, read_layer=READ, layers=LAYERS, n=len(df),
               sources=sorted(df.source.unique()), per_source=a.per_source,
               benign_request=BENIGN, auroc=table, floors=floors,
               max_gap_between_requests=same,
               max_activation_gap=float(df.act_gap.max()),
               minutes=round((time.time() - t0) / 60, 1)),
          open(os.path.join(a.out, f"{a.tag}_stats.json"), "w"), indent=1)
print(f"\nwrote {a.out}/{a.tag}_stats.json")
