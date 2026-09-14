"""Gate 0, quantitative: where in depth does a visual concept enter the workspace?

For each (COCO image, present category c) we take the best rank of c's token
over the image-token positions, at every layer, for J-lens and for logit lens.
Two numbers come out:
  * surfacing depth -- first normalised depth where rank(c) < K
  * present-vs-absent AUC per layer -- can the readout tell a category that is
    in the picture from one that is not
"""
import sys, json, argparse, logging, time
sys.path.insert(0, "/workspace/vlm/code")
for n in ("httpx","httpcore","urllib3","filelock"): logging.getLogger(n).setLevel(logging.WARNING)
import numpy as np, torch
import jvlm
from jvlm import readout, corpus, cocodata

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--lens", required=True)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="/workspace/vlm/results/gate0")
ap.add_argument("--layer-stride", type=int, default=3)
ap.add_argument("--tag", default=None)
a = ap.parse_args()

import os; os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lens = (jvlm.lens_from_checkpoint(a.lens) if a.lens.endswith(".ckpt")
        else __import__("jlens").lens.JacobianLens.load(a.lens))
print(m, lens, flush=True)
# Exact vocabulary ranks need the full unembedding at every image position, so
# the cost is layers x positions x |V|.  Every third layer traces the depth
# curve just as well and keeps the run under an hour.
L = list(lens.source_layers)[::a.layer_stride]
items = cocodata.concept_items(a.n, seed=7)
print(f"{len(items)} items", flush=True)

# candidate vocabulary: every single-word COCO category
CATS = cocodata.SINGLE_WORD
cat_tok = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
flat = sorted({t for v in cat_tok.values() for t in v})
tok_index = {t: i for i, t in enumerate(flat)}

rows = []
t0 = time.time()
for k, it in enumerate(items):
    im = corpus.load_image(it["path"])
    b = m.chat("Describe this image.", [im])
    ipos = readout.image_positions(m, b)
    if len(ipos) == 0:
        continue
    acts = readout.residuals(m, b, L)
    here = set(it["present"])
    for use_j in (True, False):
        ll, _ = readout.lens_logits(m, lens, b, layers=L, positions=ipos.tolist(),
                                    use_jacobian=use_j, cached=acts)
        for l in L:
            lg = ll[l]                                    # [P, vocab]
            # rank of each candidate token at its best image position
            sub = lg[:, flat]                             # [P, T]
            best = sub.max(0).values                      # [T]
            # full-vocab rank of that best score
            ranks = (lg.max(0).values.unsqueeze(0) > best.unsqueeze(1)).sum(1)
            for c in CATS:
                r = int(min(ranks[tok_index[t]].item() for t in cat_tok[c]))
                rows.append(dict(i=k, image_id=it["image_id"], layer=l,
                                 jac=int(use_j), cat=c,
                                 present=int(c in here),
                                 target=int(c == it["concept"]), rank=r))
    if k % 20 == 0:
        print(f"  {k}/{len(items)}  {time.time()-t0:.0f}s", flush=True)

import pandas as pd
df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_ranks.parquet")

# ---- summary
def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    allv = np.concatenate([pos, neg]); order = allv.argsort()
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(len(allv))
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos)*(len(pos)-1)/2) / (len(pos)*len(neg)))

summ = []
for use_j in (1, 0):
    d = df[df.jac == use_j]
    for l in L:
        dl = d[d.layer == l]
        summ.append(dict(
            jac=use_j, layer=l, depth=l/(m.n_layers-1),
            median_rank_present=float(dl[dl.present == 1]["rank"].median()),
            median_rank_absent=float(dl[dl.present == 0]["rank"].median()),
            median_rank_target=float(dl[dl.target == 1]["rank"].median()),
            auc=auc(-dl[dl.present == 1]["rank"].values.astype(float),
                    -dl[dl.present == 0]["rank"].values.astype(float)),
            top10_target=float((dl[dl.target == 1]["rank"] < 10).mean()),
        ))
s = pd.DataFrame(summ)
s.to_csv(f"{a.out}/{tag}_summary.csv", index=False)
print(s.to_string(index=False))
best = s.loc[s.groupby("jac").auc.idxmax()]
print("\nBEST LAYER PER LENS:\n", best.to_string(index=False))
json.dump({"repo": a.repo, "n_items": len(items), "lens": a.lens,
           "best": best.to_dict("records")},
          open(f"{a.out}/{tag}_meta.json", "w"), indent=2)
