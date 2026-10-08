"""Is the fitted lens converged, and is it doing anything a logit lens does not?

Three diagnostics, each aimed at an objection a reviewer will raise.

corpus independence
    Compare the multimodal lens against one fitted on plain text that never saw
    an image: entrywise correlation of J_l, and -- the quantity that actually
    matters -- the cosine between the concept directions v_c the two produce.
    This is a stronger check than a split half, because it also answers whether
    the transport we use to read the visual channel is a property of the model
    or an artefact of what we fitted on.

convergence
    The running-mean relative change per prompt, read from the fit log, which
    should fall like 1/n once the estimate has settled.

versus the logit lens
    Agreement between the two readouts as a function of depth, plus the
    alignment cos(v_c, W_U[c]).  Where the Jacobian is close to the identity the
    two lenses coincide and the Jacobian is buying nothing; the interesting
    depths are where they diverge.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys

sys.path.insert(0, "code")
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import jvlm
from jvlm import cocodata, corpus, readout

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--lens", required=True)
ap.add_argument("--other", default=None,
                help="a second lens to compare against (e.g. the text-fitted one)")
ap.add_argument("--fitlog", default=None, help="fit log, for the convergence trace")
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/lensqual")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
load = lambda p: (jvlm.lens_from_checkpoint(p) if p.endswith(".ckpt")
                  else __import__("jlens").lens.JacobianLens.load(p))
lens = load(a.lens)
print(m, lens, flush=True)
L = list(lens.source_layers)
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c)
           for c in cocodata.SINGLE_WORD}

rows = []

# -------------------------------------------------- corpus independence
if a.other and os.path.exists(a.other):
    lb = load(a.other)
    for l in L:
        if l not in lb.jacobians:
            continue
        Ja, Jb = lens.jacobians[l].flatten(), lb.jacobians[l].flatten()
        r = float(np.corrcoef(Ja.numpy(), Jb.numpy())[0, 1])
        cos_v = []
        for c in cocodata.SINGLE_WORD:
            va = readout.lens_vectors(m, lens, CAT_IDS[c], l).mean(0)
            vb = readout.lens_vectors(m, lb, CAT_IDS[c], l).mean(0)
            cos_v.append(float(F.cosine_similarity(va, vb, dim=0)))
        rows.append(dict(layer=l, depth=l / (m.n_layers - 1),
                         cross_lens_r=r, cross_lens_cos_v=float(np.mean(cos_v))))
    print("cross-lens comparison done", flush=True)

# ------------------------------------------------------------ convergence
conv = []
if a.fitlog and os.path.exists(a.fitlog):
    import re
    for line in open(a.fitlog):
        mm_ = re.search(r"batch (\d+).*d_mean=([0-9.e+-]+)", line)
        if mm_:
            conv.append(dict(n=int(mm_.group(1)) + 1, d_mean=float(mm_.group(2))))
    if conv:
        pd.DataFrame(conv).to_csv(f"{a.out}/{tag}_convergence.csv", index=False)
        print(f"convergence trace: {len(conv)} points, "
              f"last d_mean={conv[-1]['d_mean']:.2e}", flush=True)

# ------------------------------------------------- lens vs logit lens
items = cocodata.concept_items(a.n, min_area=0.08, seed=91)
agree, klv, cosv = {l: [] for l in L}, {l: [] for l in L}, {l: [] for l in L}
for l in L:
    for c in cocodata.SINGLE_WORD:
        v = readout.lens_vectors(m, lens, CAT_IDS[c], l).mean(0)
        w = m._lm_head.weight[CAT_IDS[c]].detach().float().mean(0)
        cosv[l].append(float(F.cosine_similarity(v, w, dim=0)))

for k, it in enumerate(items):
    b = m.chat("Describe this image.", [corpus.load_image(it["path"])])
    ipos = readout.image_positions(m, b)
    acts = readout.residuals(m, b, L)
    lj, final = readout.lens_logits(m, lens, b, layers=L,
                                    positions=ipos.tolist(), cached=acts)
    ll, _ = readout.lens_logits(m, lens, b, layers=L, positions=ipos.tolist(),
                                use_jacobian=False, cached=acts)
    fin = torch.log_softmax(
        m.unembed(acts[m.n_layers - 1][-1:]).float().cpu(), -1)[0]
    for l in L:
        a1 = lj[l].argmax(-1)
        a2 = ll[l].argmax(-1)
        agree[l].append(float((a1 == a2).float().mean()))
        p = torch.log_softmax(lj[l].mean(0), -1)
        klv[l].append(float(F.kl_div(p, fin, log_target=True,
                                     reduction="sum")))
    if k % 20 == 0:
        print(f"  {k}/{len(items)}", flush=True)

by_layer = {r["layer"]: r for r in rows}
out = []
for l in L:
    r = by_layer.get(l, dict(layer=l, depth=l / (m.n_layers - 1)))
    r.update(top1_agreement=float(np.mean(agree[l])),
             kl_to_final=float(np.mean(klv[l])),
             cos_v_unembed=float(np.mean(cosv[l])))
    out.append(r)

df = pd.DataFrame(out)
df.to_csv(f"{a.out}/{tag}_quality.csv", index=False)
print("\n", df.round(3).to_string(index=False))
json.dump({"repo": a.repo, "lens": a.lens, "n_prompts": lens.n_prompts,
           "mean_top1_agreement": float(df.top1_agreement.mean()),
           "min_top1_agreement": float(df.top1_agreement.min()),
           "mean_cos_v_unembed": float(df.cos_v_unembed.mean())},
          open(f"{a.out}/{tag}_summary.json", "w"), indent=2)
