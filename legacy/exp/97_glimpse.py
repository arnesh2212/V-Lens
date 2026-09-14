# SPDX-License-Identifier: Apache-2.0
"""GLIMPSE against ours, on the causal ruler, at each method's own best pooling.

Our comparison set has so far been the logit lens, the J-lens and the classical
gradient family.  Every one of those is either prompt-blind by construction --
image tokens precede text under a causal mask, so their reading is byte-
identical for two questions about one photograph -- or is a one-point gradient
with no conservation law.  A reader is entitled to point out that the one 2026
method built to be *prompt-conditioned* is missing, and that leaving it out
looks like avoiding the only competitor that could win.

So this runs GLIMPSE (arXiv:2506.18985), reimplemented from its equations in
``jvlm.glimpse`` because no reference implementation is published, through the
same harness as everything else.

Two choices make the comparison fair to GLIMPSE rather than to us:

*The objective is answer-level.*  GLIMPSE explains a whole generated response.
Grading it against the drop in a single token's logit would score it on a task
it was not built for, so the ruler here is the teacher-forced log-likelihood of
the model's own generated answer -- ``jvlm.objective.sequence_logprob`` -- and
the ground truth is what covering a region really costs *that*.  Our reading is
graded on the identical scalar.

*GLIMPSE reads every layer; we read one.*  Its equations 9-14 aggregate all
layers under a depth prior, so it is given the whole network while our reading
is held to a single layer fixed in advance by the same rule the earlier
baseline sweep used.  That asymmetry is in GLIMPSE's favour and is left in.

Also reported, because it is the specific claim GLIMPSE threatens: how much
each method's map actually *moves* when the question changes on a fixed
photograph.  For the logit lens and the J-lens that number is exactly zero.
GLIMPSE is not prompt-blind, so it should be well above zero, and the question
is whether being prompt-conditioned is enough to be prompt-*correct*.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jvlm
from jvlm import bench, cocodata, corpus, glimpse, objective, readout
from jlens.hooks import ActivationRecorder

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--n", type=int, default=36)
ap.add_argument("--mean-images", type=int, default=60)
ap.add_argument("--n-layers", type=int, default=4)
ap.add_argument("--ig-steps", type=int, default=32)
ap.add_argument("--ig-chunk", type=int, default=8)
ap.add_argument("--gen-tokens", type=int, default=16)
ap.add_argument("--pool", type=int, default=2)
ap.add_argument("--truth-chunk", type=int, default=7)
ap.add_argument("--truth", choices=("mean", "pixel"), default="mean",
                help="'mean' replaces the region's activations at the read "
                     "layer with the mean image, the causal map this project "
                     "already uses; 'pixel' greys the corresponding square of "
                     "the photograph and re-encodes it, which is the ruler "
                     "that favours no method's baseline")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--complete-n", type=int, default=4)
ap.add_argument("--out", default="/workspace/vlm/results/glimpse")
ap.add_argument("--tag", default="qwen2b")
a = ap.parse_args()

PROMPT = "Describe this image."
OTHER = "How many objects are in this photograph?"

os.makedirs(a.out, exist_ok=True)
# eager: GLIMPSE reads the [N, N] attention matrix, which SDPA never builds
m = jvlm.load_vlm(a.repo, attn="eager")
print(f"{m}", flush=True)

items = cocodata.concept_items(a.n * 2, tokenizer=m.tokenizer)
print(f"{len(items)} candidate items", flush=True)

b0 = m.chat(PROMPT, [corpus.load_image(items[0]["path"], square=448)])
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
side = int(round(P ** 0.5))
LAST = m.n_layers - 1
LAYERS = sorted({int(round(f * LAST)) for f in np.linspace(0.25, 0.85, a.n_layers)})
READ = LAYERS[len(LAYERS) // 2]          # fixed in advance, not on the result
print(f"{P} image tokens on {side}x{side}; layer ladder {LAYERS}; "
      f"reading at {READ}; truth = {a.truth}", flush=True)

# the per-position corpus mean, accumulated inline as every other experiment
# here does it -- there is no shared helper and inventing one that disagreed
# with the runs we grade against would be worse than repeating it
_acc = torch.zeros(P, m.d_model, dtype=torch.float64)
_nm = 0
for _p in corpus.coco_images(a.mean_images, seed=11):
    _b = m.chat(PROMPT, [corpus.load_image(_p, square=448)])
    if len(readout.image_positions(m, _b)) != P:
        continue
    _acc += readout.residuals(m, _b, [READ])[READ][IPOS].float().cpu().double()
    _nm += 1
MEAN = (_acc / max(_nm, 1)).float()
print(f"mean image from {_nm} held-out photographs", flush=True)

K = side // a.pool
REGIONS = []
for r in range(K):
    for c in range(K):
        cells = [rr * side + cc
                 for rr in range(r * a.pool, min((r + 1) * a.pool, side))
                 for cc in range(c * a.pool, min((c + 1) * a.pool, side))]
        if cells:
            REGIONS.append(cells)
print(f"{len(REGIONS)} regions of {a.pool}x{a.pool} patches", flush=True)


def clean_at(batch, layer):
    with torch.no_grad(), ActivationRecorder(m.layers, at=[layer]) as rec:
        m.forward(batch)
        return rec.activations[layer][0, IPOS.to(rec.activations[layer].device)] \
            .detach().float()


def cover_truth(obj, h):
    """What covering each region costs the answer-level objective F.

    An intervention, never another lens: the region's positions are set to the
    mean image at the read layer and the model is run again.  The control
    condition is the mean picture, which is the same baseline the path integral
    starts from, so ``truth`` and the readings are differences from the same
    place.
    """
    v = obj.variants[0]
    base = float(objective.value(m, obj)[0])
    mu = MEAN.to(h.device)
    out = []
    for c0 in range(0, len(REGIONS), a.truth_chunk):
        sub = REGIONS[c0: c0 + a.truth_chunk]
        x = h[None].repeat(len(sub), 1, 1)
        for j, cells in enumerate(sub):
            idx = torch.tensor(cells, dtype=torch.long, device=h.device)
            x[j, idx] = mu[idx]
        big = m.replicate(v.batch, len(sub))
        hd = m.layers[READ].register_forward_hook(
            objective._write_hook(x, IPOS))
        try:
            with torch.no_grad(), ActivationRecorder(m.layers, at=[LAST]) as rec:
                m.forward(big)
                o = v.reader(m, rec.activations[LAST]).float()
                c = (o @ v.weight.to(o.device, o.dtype)).cpu()
        finally:
            hd.remove()
        out.append(c[:, 0].numpy())
    return base - np.concatenate(out)          # cost of covering, so bigger = worse


def cover_truth_pixel(img, ans, base):
    """The same question asked in pixel space, where no method has home ground.

    The activation-space causal map replaces a region with the mean image at
    the read layer -- which is exactly the baseline and exactly the layer our
    path integral starts from, and no attention method has any such notion.
    That is a real objection to grading GLIMPSE on it, so the same measurement
    is available in pixel space: grey out the square of the photograph the
    region covers, re-encode the picture from scratch, and teacher-force the
    same answer.  Slower, because every region pays for the vision tower, and
    neutral, because the intervention now happens somewhere neither method
    privileges.
    """
    px = img.size[0] // side
    out = []
    for cells in REGIONS:
        rs = [c // side for c in cells]
        cs = [c % side for c in cells]
        box = (min(cs) * px, min(rs) * px,
               (max(cs) + 1) * px, (max(rs) + 1) * px)
        occ = img.copy()
        occ.paste((128, 128, 128), box)
        bo = m.chat(PROMPT, [occ])
        oo = objective.sequence_logprob(m, bo, [ans], norm="tokens")
        out.append(float(objective.value(m, oo)[0]))
    return base - np.array(out)


def raw_grad(obj, h):
    """dF/dh at the read layer, for the two classical baselines."""
    x = h[None].detach().clone().requires_grad_(True)
    got = {}
    objective._pass(m, obj, READ, IPOS, x, cols=[0],
                    sink=lambda j, g: got.__setitem__(j, g.float()),
                    want_vals=False)
    return got[0][0]


GL = glimpse.GLIMPSE()
GL0 = glimpse.GLIMPSE("attention rollout", gradient=False)

NAMES = ("VJ-Lens", "GLIMPSE", "attention rollout", "GLIMPSE wrong prompt",
         "VJ-Lens wrong prompt", "gradient norm", "gradient x activation",
         "logit lens", "random")
tasks, methods = [], {n: [] for n in NAMES}
rng = np.random.default_rng(0)
sens, comp, answers = {"GLIMPSE": [], "VJ-Lens": [], "logit lens": []}, [], []
tstats = []
t0 = time.time()

for it in items:
    if len(tasks) >= a.n:
        break
    img = corpus.load_image(it["path"], square=448)
    b = m.chat(PROMPT, [img])
    if len(readout.image_positions(m, b)) != P:
        continue
    ans = objective.generated_answer(m, b, max_new_tokens=a.gen_tokens)
    if int(ans.numel()) < 2:
        continue
    obj = objective.sequence_logprob(m, b, [ans], norm="tokens")
    gb = glimpse.prepare(m, b, ans)
    if not torch.equal(obj.image_positions(m), IPOS):
        continue

    h = clean_at(obj.variants[0].batch, READ)
    F0 = float(objective.value(m, obj)[0])
    if a.truth == "pixel":
        truth = cover_truth_pixel(img, ans, F0)
        # both rulers on the same photograph, so a null on one can be told
        # apart from a ruler that is not measuring anything: bf16 gives F about
        # three decimal digits, so a spread of that order across regions is
        # quantisation and not a causal map
        alt = cover_truth(obj, h)
        tstats.append({"F": F0,
                       "pixel_std": float(np.std(truth)),
                       "pixel_absmax": float(np.abs(truth).max()),
                       "act_std": float(np.std(alt)),
                       "act_absmax": float(np.abs(alt).max()),
                       "rho_pixel_act": float(np.corrcoef(
                           np.argsort(np.argsort(truth)),
                           np.argsort(np.argsort(alt)))[0, 1])})
    else:
        truth = cover_truth(obj, h)
    if not np.isfinite(truth).all() or float(np.std(truth)) < 1e-9:
        continue

    # the same photograph, a different question, and the answer the model
    # itself gives to it -- the control that separates "reads the image" from
    # "reads this question"
    bw = m.chat(OTHER, [img])
    answ = objective.generated_answer(m, bw, max_new_tokens=a.gen_tokens)
    if int(answ.numel()) < 2:
        continue
    objw = objective.sequence_logprob(m, bw, [answ], norm="tokens")
    gbw = glimpse.prepare(m, bw, answ)

    per = {}
    per["GLIMPSE"] = GL(m, gb, READ, IPOS)
    per["attention rollout"] = GL0(m, gb, READ, IPOS)
    per["GLIMPSE wrong prompt"] = GL(m, gbw, READ, IPOS)
    ig = objective.ig_map(m, obj, [READ], IPOS, {READ: MEAN},
                          steps=a.ig_steps, chunk=a.ig_chunk)
    per["VJ-Lens"] = ig[READ][0].numpy()
    igw = objective.ig_map(m, objw, [READ], IPOS, {READ: MEAN},
                           steps=a.ig_steps, chunk=a.ig_chunk)
    per["VJ-Lens wrong prompt"] = igw[READ][0].numpy()
    g = raw_grad(obj, h)
    per["gradient norm"] = g.norm(dim=-1).cpu().numpy()
    per["gradient x activation"] = (g * (h - MEAN.to(h.device))) \
        .sum(-1).cpu().numpy()
    with torch.no_grad():
        ll = m.unembed(h.to(m.device))[:, ans.to(m.device)].float().cpu()
    per["logit lens"] = ll.max(-1).values.numpy()
    per["random"] = rng.normal(size=P)

    # how far each map moves when only the question changes
    def rho(u, v):
        u, v = np.asarray(u, float), np.asarray(v, float)
        if u.std() < 1e-12 or v.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(np.argsort(np.argsort(u)),
                                 np.argsort(np.argsort(v)))[0, 1])
    sens["GLIMPSE"].append(rho(per["GLIMPSE"], per["GLIMPSE wrong prompt"]))
    sens["VJ-Lens"].append(rho(per["VJ-Lens"], per["VJ-Lens wrong prompt"]))
    with torch.no_grad():
        hw = clean_at(objw.variants[0].batch, READ)
        llw = m.unembed(hw.to(m.device))[:, ans.to(m.device)].float().cpu()
    sens["logit lens"].append(rho(per["logit lens"], llw.max(-1).values.numpy()))

    if len(comp) < a.complete_n:
        comp.append(objective.completeness(m, obj, READ, IPOS, MEAN,
                                           ig[READ]))
    answers.append(m.tokenizer.decode(ans).strip()[:60])

    tasks.append(bench.Task(regions=REGIONS, truth=truth, kind="localise",
                            image_id=str(it.get("image_id"))))
    for name in NAMES:
        methods[name].append(per[name])
    if len(tasks) % 5 == 0:
        el = time.time() - t0
        print(f"  {len(tasks)}/{a.n}  {el:.0f}s  "
              f"eta {el / len(tasks) * (a.n - len(tasks)) / 60:.1f}m", flush=True)

print(f"\n{len(tasks)} photographs kept in {time.time() - t0:.0f}s", flush=True)
res = bench.sweep(methods, tasks)
best = bench.best_per_method(res)

# bootstrap over IMAGES, never over regions: the regions of one photograph are
# one measurement, not forty-nine
def per_task(name, agg):
    hits, rhos = [], []
    for v, t in zip(methods[name], tasks):
        s = bench.pool(np.asarray(v, float), t, agg)
        hits.append(float(int(np.argmax(s)) == int(np.argmax(t.truth))))
        r = bench._spearman(s, t.truth)
        rhos.append(r if r == r else np.nan)
    return np.array(hits), np.array(rhos)


boot = {}
rb = np.random.default_rng(7)
for name, r in best.items():
    hits, rhos = per_task(name, r.aggregation)
    idx = rb.integers(0, len(hits), size=(a.boot, len(hits)))
    bt = hits[idx].mean(1)
    br = np.nanmean(rhos[idx], axis=1)
    boot[name] = {"top1_lo": float(np.percentile(bt, 2.5)),
                  "top1_hi": float(np.percentile(bt, 97.5)),
                  "rho_lo": float(np.percentile(br, 2.5)),
                  "rho_hi": float(np.percentile(br, 97.5))}

print("\ntop-1 accuracy, method x aggregation\n")
print(bench.table(res))
chance = tasks[0].chance_top1()
print(f"\nchance = {chance:.4f} over {len(REGIONS)} equal regions; "
      f"measured floor (random) = {best['random'].top1:.4f}\n")
hdr = ("method", "rule", "top-1", "95% CI", "lift", "rho", "95% CI")
print(f"{hdr[0]:<24}{hdr[1]:>8}{hdr[2]:>8}{hdr[3]:>16}"
      f"{hdr[4]:>7}{hdr[5]:>8}{hdr[6]:>16}")
for n, r in sorted(best.items(), key=lambda kv: -kv[1].top1):
    bb = boot[n]
    ci1 = "[%.3f,%.3f]" % (bb["top1_lo"], bb["top1_hi"])
    ci2 = "[%.3f,%.3f]" % (bb["rho_lo"], bb["rho_hi"])
    print(f"{n:<24}{r.aggregation:>8}{r.top1:>8.3f}{ci1:>16}"
          f"{r.lift():>6.1f}x{r.spearman:>8.3f}{ci2:>16}")

print("\nprompt sensitivity -- rank correlation between the map for "
      f"{PROMPT!r}\nand the map for {OTHER!r} on the same photograph "
      "(1.0 = prompt-blind)\n")
for n, v in sens.items():
    v = np.array([x for x in v if x == x])
    if v.size:
        print(f"  {n:<22}{v.mean():>8.4f}  +/- {v.std(ddof=1) / np.sqrt(v.size):.4f}")

if tstats:
    print("\ntruth diagnostics -- is the pixel ruler measuring anything?")
    for k in ("pixel_std", "pixel_absmax", "act_std", "act_absmax",
              "rho_pixel_act"):
        v = np.array([t[k] for t in tstats])
        print(f"  {k:<16}{v.mean():>12.5f}  +/- {v.std(ddof=1) / np.sqrt(v.size):.5f}")
    print(f"  {'|F|':<16}{np.abs([t['F'] for t in tstats]).mean():>12.5f}")

if comp:
    print("\ncompleteness of the path reading on the sequence objective "
          f"(steps={a.ig_steps})")
    for c in comp:
        print(f"  truth {c['truth'][0]:+.4f}  claim {c['claim'][0]:+.4f}  "
              f"rel_err {c['rel_error']:.3f}")

out = {
    "repo": a.repo, "n": len(tasks), "n_regions": len(REGIONS),
    "layer": int(READ), "layers": LAYERS, "ig_steps": a.ig_steps,
    "truth": a.truth,
    "gen_tokens": a.gen_tokens, "prompt": PROMPT, "other_prompt": OTHER,
    "objective": "sequence_logprob(generated answer, norm=tokens)",
    "chance": chance,
    "glimpse": {"head_temp": glimpse.LAMBDA_HEAD,
                "depth_temp": glimpse.LAMBDA_DEPTH,
                "source": "arXiv:2506.18985, reimplemented from equations",
                "reference_impl": "none published"},
    "grid": [{"method": r.method, "aggregation": r.aggregation,
              "top1": r.top1, "spearman": r.spearman, "n": r.n} for r in res],
    "best": {n: {"aggregation": r.aggregation, "top1": r.top1,
                 "lift": r.lift(), "spearman": r.spearman, **boot[n]}
             for n, r in best.items()},
    "prompt_sensitivity": {n: {"mean": float(np.nanmean(v)),
                               "sem": float(np.nanstd(v, ddof=1) /
                                            np.sqrt(max(len(v), 1)))}
                           for n, v in sens.items() if len(v)},
    "completeness": comp,
    "truth_stats": tstats,
    "answers": answers[:12],
}
p = os.path.join(a.out, f"{a.tag}_stats.json")
json.dump(out, open(p, "w"), indent=1)
print(f"\nwrote {p}")
