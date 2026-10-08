"""Watching a pointer resolve: what the landmark is worth, layer by layer.

"What colour is the ball to the left of the small matte cylinder?" cannot be
answered in one step.  The cylinder has to be found before the ball can be, so
the question names two objects with two different jobs: a LANDMARK, which is
only a signpost, and a REFERENT, which is what the answer is about.  If the
model really resolves the pointer, the landmark's job finishes somewhere inside
the stack -- and after it finishes, covering the landmark should cost nothing.

That is a prediction about a curve, and it can be measured directly.  At every
depth, cover the landmark's tokens with a per-position average render and watch
the answer word fall; do the same for the referent, for a third object in
neither role, and for a patch of bare floor with the same number of tokens.
Four curves against depth.  The landmark's should peak in the middle and
collapse; the referent's should rise late and stay; where they cross is where
the reference resolved.

The control is the same render asked a different question.  "What colour is the
small matte cylinder?" makes the former landmark the thing being answered about
and the former referent irrelevant, with every pixel unchanged.  If the curves
follow the roles rather than the objects, they swap.  If they do not, what the
first panel showed was salience, not structure.

Each lens is read at the same regions and the same depths, so the figure can
also show which reading tracks the intervention it claims to describe.

THE BASELINE.  The covering baseline is a per-position mean over ~90 CLEVR
renders, not over photographs.  A render's deviation from a COCO mean is
dominated by "this is a synthetic image", so covering any region with a COCO
mean injects that difference everywhere and swamps the thing being measured.
``--compare`` runs a subset both ways and prints the size of the trap.
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
import pandas as pd
import torch
from scipy import stats as sps

import jvlm
from jlens.hooks import ActivationRecorder
from jvlm import clevrdata as cd
from jvlm import corpus, readout, stats as st, vmap

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--jlens", default=None)
ap.add_argument("--n", type=int, default=24, help="scenes kept")
ap.add_argument("--scan", type=int, default=400, help="candidates considered")
ap.add_argument("--mean-images", type=int, default=90)
ap.add_argument("--n-layers", type=int, default=12)
ap.add_argument("--ig-steps", type=int, default=4)
ap.add_argument("--compare", type=int, default=4,
                help="scenes also run against a COCO mean, to price the trap")
ap.add_argument("--rel", default="near", choices=["left", "right", "near"])
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--suffix", default=" Answer with one word.")
ap.add_argument("--probe", action="store_true",
                help="only ask whether the model can answer these questions;\n"
                     "no baseline, no covering, no lens")
ap.add_argument("--strict", action="store_true",
                help="also require the colour word to be the top token overall,\n"
                     "not merely the top colour")
ap.add_argument("--out", default="results/pointer")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
LAYERS = sorted({max(int(round(f * (m.n_layers - 1))), m.n_deepstack)
                 for f in np.linspace(0.12, 0.92, a.n_layers)})
# the covering intervention works at any depth; only the path integral has to
# start at or after the DeepStack band, whose injection writes in place.  So
# the truth is measured under those layers too -- otherwise a pointer that
# resolves in the first three layers would be invisible and read as absent.
TLAYERS = sorted(set(range(m.n_deepstack)) | set(LAYERS))
LAST = m.n_layers - 1
J = None
if a.jlens:
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))
print(m, f"\ncovering at {TLAYERS};  lenses at {LAYERS} of {m.n_layers}",
      flush=True)


# ------------------------------------------------------------------ vocabulary
def variants(w: str) -> list[int]:
    ids = set()
    for s in (w, w.capitalize(), " " + w, " " + w.capitalize()):
        t = m.tokenizer.encode(s, add_special_tokens=False)
        if len(t) == 1:
            ids.add(int(t[0]))
    return sorted(ids)


CID = {c: variants(c) for c in cd.COLOURS}
CID = {c: v for c, v in CID.items() if v}
ALLC = sorted({t for v in CID.values() for t in v})
CPOS = {t: i for i, t in enumerate(ALLC)}
print(f"{len(CID)}/8 colours single-token: "
      + ", ".join(f"{c}:{len(v)}" for c, v in CID.items()), flush=True)


def prep(path, question):
    return m.chat(question + a.suffix,
                  [corpus.load_image(path, square=a.square)])


# ------------------------------------------------------------- the two means
b0 = prep(cd.image_path(cd.scenes()[0]), "What is this?")
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
if G * G != P:
    raise SystemExit(f"{P} image tokens is not a square grid")
print(f"{P} image tokens on {G}x{G}", flush=True)


def build_mean(paths):
    acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in TLAYERS}
    nrm = {l: torch.zeros(P, dtype=torch.float64) for l in TLAYERS}
    n = 0
    for p in paths:
        b = m.chat("Describe this image.",
                   [corpus.load_image(p, square=a.square)])
        ip = readout.image_positions(m, b)
        if len(ip) != P:
            continue
        A = readout.residuals(m, b, TLAYERS)
        for l in TLAYERS:
            v = A[l][ip].float().cpu()
            acc[l] += v.double()
            nrm[l] += v.norm(dim=-1).double()
        n += 1
    return ({l: (acc[l] / max(n, 1)).float() for l in TLAYERS},
            {l: (nrm[l] / max(n, 1)).float() for l in TLAYERS}, n)


# the mean pool is held out: it is drawn from the far end of the val split, so
# no render both defines the baseline and is measured against it
if a.probe:
    BASE, REG, gap, n_cl, n_co = {}, set(), {}, 0, 0
else:
    pool = sorted(cd.scenes(), key=lambda s: -s["image_index"])
    pool = [cd.image_path(s) for s in pool if cd.have(s)][: a.mean_images]
    MEAN_CLEVR, NORM_CLEVR, n_cl = build_mean(pool)
    MEAN_COCO, _, n_co = build_mean(corpus.coco_images(a.mean_images, seed=11))
    BASE = {"clevr": MEAN_CLEVR, "coco": MEAN_COCO}
    print(f"means: {n_cl} renders, {n_co} photographs ({time.time()-t0:.0f}s)",
          flush=True)

    nrm0 = NORM_CLEVR[TLAYERS[-1]]
    REG = set((nrm0 > nrm0.mean() + 3 * nrm0.std()).nonzero()
              .flatten().tolist())
    print(f"{len(REG)} register tokens masked out of every region", flush=True)

    # how far apart the two baselines are, before anything is measured with them
    gap = {l: float((MEAN_CLEVR[l] - MEAN_COCO[l]).norm(dim=-1).mean()
                    / MEAN_CLEVR[l].norm(dim=-1).mean()) for l in TLAYERS}
    print("baseline gap |clevr-coco|/|clevr| per layer: "
          + " ".join(f"{l}:{v:.2f}" for l, v in gap.items()), flush=True)

ROLES = ["landmark", "referent", "distractor", "floor"]


def positions(item, role):
    cells = [c for c in item[f"tok_{role}"] if c not in REG]
    return IPOS[torch.tensor(cells, dtype=torch.long)], cells


# --------------------------------------------------------------- measurement
def answer_logits(batch, pos_sets, layer, means):
    """Cover each token set at ``layer`` with ``means``; colour logits each."""
    big = m.replicate(batch, len(pos_sets))
    APOS = batch["input_ids"].shape[1] - 1

    def hook(mod, inp, o, _s=pos_sets, _l=layer):
        tup = not torch.is_tensor(o)
        hh = (o[0] if tup else o).clone()
        for j, (cells, seq) in enumerate(_s):
            hh[j, seq.to(hh.device)] = \
                means[_l][torch.tensor(cells)].to(hh.device).to(hh.dtype)
        return (hh, *o[1:]) if tup else hh

    hd = m.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad(), ActivationRecorder(m.layers, at=[LAST]) as rec:
            m.forward(big)
            return m.unembed(rec.activations[LAST][:, APOS])[:, ALLC] \
                .float().cpu()
    finally:
        hd.remove()


def clean_logits(batch):
    APOS = batch["input_ids"].shape[1] - 1
    with torch.no_grad(), ActivationRecorder(m.layers, at=[LAST]) as rec:
        m.forward(batch)
        return m.unembed(rec.activations[LAST][0, APOS][None])[0].float().cpu()


rows, items, seen = [], [], 0
n_cand = n_ok_main = n_ok_ctrl = n_ok_both = 0
SOURCE = (cd.near_items(grid=G) if a.rel == "near"
          else cd.pointer_items(grid=G, rel=a.rel))
for it in SOURCE:
    if (len(items) >= a.n and not a.probe) or seen >= a.scan:
        break
    seen += 1
    n_cand += 1
    B = {q: prep(it["path"], it[f"question_{q}"]) for q in ("main", "control")}
    if any(len(readout.image_positions(m, b)) != P for b in B.values()):
        continue
    ok, base, tops = {}, {}, {}
    for q in ("main", "control"):
        lg = clean_logits(B[q])
        want = it[f"answer_{q}"]
        sc = {c: float(lg[CID[c]].max()) for c in CID}
        top = max(sc, key=sc.get)
        base[q] = sc[want]
        tops[q] = top
        ok[q] = (top == want) and (
            not a.strict or int(lg.argmax()) in CID[want])
    n_ok_main += int(ok["main"])
    n_ok_ctrl += int(ok["control"])
    n_ok_both += int(ok["main"] and ok["control"])
    if a.probe:
        mark = "".join("+" if ok[q] else "." for q in ("main", "control"))
        print(f"  {mark} {it['question_main']}  want {it['answer_main']} "
              f"got {tops['main']}   |   control want {it['answer_control']} "
              f"got {tops['control']}", flush=True)
        continue
    if not (ok["main"] and ok["control"]):
        continue

    PS = {r: positions(it, r) for r in ROLES}
    PS = {r: (c, p) for r, (p, c) in PS.items()}
    if any(len(v[0]) < 2 for v in PS.values()):
        continue
    order = [PS[r] for r in ROLES] + [(list(range(P)), IPOS)]
    VOCW = sorted({it["answer_main"], it["answer_control"],
                   it["colour_distractor"]})
    VOC = sorted({t for w in VOCW for t in CID[w]})
    it = dict(it, vocab=VOCW,
              n_tok={r: len(PS[r][0]) for r in ROLES},
              base_main=base["main"], base_control=base["control"])
    items.append(it)

    which = ["clevr", "coco"] if len(items) <= a.compare else ["clevr"]
    for src in which:
        MEANS = BASE[src]
        for q in ("main", "control"):
            b = B[q]
            want = it[f"answer_{q}"]
            col = [CPOS[t] for t in CID[want]]
            vcol = [VOC.index(t) for t in CID[want]]
            A = readout.residuals(m, b, TLAYERS)
            ig = vmap.ig_map(m, b, VOC, LAYERS, IPOS.tolist(),
                             means={l: MEANS[l] for l in LAYERS},
                             steps=a.ig_steps)
            for l in TLAYERS:
                lgv = answer_logits(b, order, l, MEANS)
                got = lgv[:, col].max(-1).values.numpy()
                drop = base[q] - got
                read = {}
                if l in ig:
                    h = A[l][IPOS].float().cpu()
                    read["VJ-Lens"] = ig[l][vcol].max(0).values     # [P]
                    with torch.no_grad():
                        ll = m.unembed(h.to(m.device))[:, ALLC].float().cpu()
                    read["logit lens"] = ll[:, col].max(-1).values
                if J is not None and l in J.jacobians and l in ig:
                    # asked one layer at a time: the full-vocabulary readout is
                    # 150k floats per position, and holding twelve of them is
                    # two gigabytes for three numbers
                    jl, _ = readout.lens_logits(m, J, b, layers=[l],
                                                positions=IPOS.tolist(),
                                                cached=A)
                    read["J-lens"] = jl[l][:, ALLC][:, col].max(-1).values
                for ri, r in enumerate(ROLES):
                    cells, seq = PS[r]
                    idx = torch.tensor(cells, dtype=torch.long)
                    row = dict(
                        scene=it["scene_index"], question=q, baseline=src,
                        layer=l, depth=round(l / LAST, 4), role=r,
                        n_tok=len(cells), drop=float(drop[ri]),
                        drop_all=float(drop[4]),
                        base=base[q], answer=want,
                        colours=json.dumps(
                            {c: round(float(lgv[ri][
                                [CPOS[t] for t in CID[c]]].max()), 4)
                             for c in CID}))
                    for name in ("VJ-Lens", "logit lens", "J-lens"):
                        v = read.get(name)
                        row[f"{name}_sum"] = (float(v[idx].sum())
                                              if v is not None else np.nan)
                        row[f"{name}_mean"] = (float(v[idx].mean())
                                               if v is not None else np.nan)
                    rows.append(row)
    el = time.time() - t0
    print(f"  {len(items)}/{a.n} kept  scene {it['scene_index']}  "
          f"{it['question_main']} -> {it['answer_main']}  "
          f"{el:.0f}s  eta {el/max(len(items),1)*(a.n-len(items))/60:.1f}m",
          flush=True)
    pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_rows.parquet")

if a.probe:
    print(f"\nPROBE {a.repo}: {n_cand} questions   "
          f"relational {n_ok_main/max(n_cand,1):.2f}   "
          f"direct {n_ok_ctrl/max(n_cand,1):.2f}   "
          f"both {n_ok_both/max(n_cand,1):.2f}", flush=True)
    raise SystemExit(0)

df = pd.DataFrame(rows)
if df.empty:
    print("NOTHING KEPT -- the model does not answer these questions", flush=True)
    json.dump({"repo": a.repo, "kept": 0, "candidates": n_cand,
               "acc_main": n_ok_main / max(n_cand, 1),
               "acc_control": n_ok_ctrl / max(n_cand, 1)},
              open(f"{a.out}/{tag}_stats.json", "w"), indent=2)
    raise SystemExit(0)
df.to_parquet(f"{a.out}/{tag}_rows.parquet")
json.dump(st.jsonable(items), open(f"{a.out}/{tag}_items.json", "w"))

LENSES = [c[:-4] for c in df.columns if c.endswith("_sum")]


def curves(d):
    g = d.groupby(["role", "depth"]).agg(
        drop=("drop", "mean"), se=("drop", "sem"), n=("drop", "size"))
    return g.reset_index()


def crossing(d):
    """Depth at which the landmark line falls below the referent line."""
    c = curves(d).pivot(index="depth", columns="role", values="drop")
    if not {"landmark", "referent"} <= set(c.columns):
        return float("nan")
    x = c.index.values
    y = (c["landmark"] - c["referent"]).values
    s = np.sign(y)
    for i in range(len(x) - 1):
        if s[i] > 0 and s[i + 1] <= 0:
            return float(x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1]))
    return float("nan")


out = {"repo": a.repo, "dataset": "CLEVR val", "rel": a.rel,
       "n_scenes": int(df.scene.nunique()), "candidates": n_cand,
       "acc_main": round(n_ok_main / max(n_cand, 1), 3),
       "acc_control": round(n_ok_ctrl / max(n_cand, 1), 3),
       "acc_both": round(n_ok_both / max(n_cand, 1), 3),
       "layers": LAYERS, "truth_layers": TLAYERS,
       "n_layers": m.n_layers, "grid": G,
       "mean_images": {"clevr": n_cl, "coco": n_co},
       "baseline_gap": {str(k): round(v, 3) for k, v in gap.items()},
       "ig_steps": a.ig_steps, "lenses": LENSES,
       "minutes": round((time.time() - t0) / 60, 1)}

for src in df.baseline.unique():
    for q in ("main", "control"):
        d = df[(df.baseline == src) & (df.question == q)]
        if d.empty:
            continue
        out[f"curve_{src}_{q}"] = curves(d).round(4).to_dict("records")
        out[f"cross_{src}_{q}"] = crossing(d)
        # peak depth per role
        cv = curves(d)
        out[f"peak_{src}_{q}"] = {
            r: float(cv[cv.role == r].sort_values("drop").depth.iloc[-1])
            for r in ROLES if (cv.role == r).any()}

# which reading tracks the intervention
tr = {}
for src in df.baseline.unique():
    d = df[df.baseline == src]
    for name in LENSES:
        for how in ("sum", "mean"):
            v = d[f"{name}_{how}"].values
            good = np.isfinite(v) & np.isfinite(d["drop"].values)
            if good.sum() < 8:
                continue
            tr[f"{src}|{name}|{how}"] = dict(
                pearson=round(float(np.corrcoef(v[good],
                                                d["drop"].values[good])[0, 1]), 4),
                spearman=round(float(sps.spearmanr(
                    v[good], d["drop"].values[good]).statistic), 4), n=int(good.sum()))
out["tracks"] = tr

# the trap, priced: what a COCO baseline does to a bare patch of floor
if (df.baseline == "coco").any():
    sub = df[df.scene.isin(df[df.baseline == "coco"].scene.unique())]
    pr = {}
    for src in ("clevr", "coco"):
        d = sub[sub.baseline == src]
        pr[src] = dict(
            floor_drop=round(float(d[d.role == "floor"]["drop"].abs().mean()), 4),
            landmark_drop=round(float(d[d.role == "landmark"]["drop"].mean()), 4),
            referent_drop=round(float(d[d.role == "referent"]["drop"].mean()), 4),
            all_drop=round(float(d["drop_all"].abs().mean()), 4),
            snr=round(float(d[d.role == "referent"]["drop"].abs().mean()
                            / max(d[d.role == "floor"]["drop"].abs().mean(),
                                  1e-9)), 3))
    out["baseline_price"] = pr

json.dump(st.jsonable(out), open(f"{a.out}/{tag}_stats.json", "w"), indent=2,
          default=float)

print(f"\nkept {out['n_scenes']} of {n_cand} candidates; "
      f"accuracy main {out['acc_main']:.2f} control {out['acc_control']:.2f}",
      flush=True)
for q in ("main", "control"):
    k = f"curve_clevr_{q}"
    if k not in out:
        continue
    c = pd.DataFrame(out[k]).pivot(index="depth", columns="role", values="drop")
    print(f"\n{q.upper()}  mean drop in the answer word (logits)\n",
          c.round(3).to_string(), flush=True)
    print(f"crossing at depth {out[f'cross_clevr_{q}']}", flush=True)
if "baseline_price" in out:
    print("\nTHE BASELINE TRAP, PRICED\n",
          pd.DataFrame(out["baseline_price"]).T.to_string(), flush=True)
print("\nWHICH READING TRACKS THE INTERVENTION\n",
      pd.DataFrame(tr).T.to_string(), flush=True)
print("done", flush=True)
