# SPDX-License-Identifier: Apache-2.0
"""When does the answer arrive, for six one-word questions about the same scenes?

Every question here has a single-token answer drawn from a small candidate set, so at
any depth we can ask which candidate the answer spot is leaning toward and whether it is
the right one.  That gives a depth curve per task, and the tasks differ in how much work
they need, so the curves say something about the model rather than about the lens:

  colour     "What colour is the large metal cube?"                one lookup
  shape      "What shape is the red object?"                       one lookup
  material   "What is the blue sphere made of?"                    one lookup
  count      "How many cubes are in the picture?"                  enumerate, then count
  spatial    "Is the red cube left or right of the blue sphere?"   two lookups and a compare
  two-step   "What colour is the sphere left of the metal cube?"   find, then look up
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--n", type=int, default=120, help="items per task")
ap.add_argument("--n-fit", type=int, default=150, help="unlabelled states per task for that task's V-Read map")
ap.add_argument("--out", default="results/tasks")
ap.add_argument("--figs", default="figures/tasks")
a = ap.parse_args()

import vlens  # noqa: E402
from vlens.data import clevr  # noqa: E402
from vlens import corpus  # noqa: E402
from vlens.lens import depth, reader  # noqa: E402
from vlens.figures import plot  # noqa: E402
from vlens.lens.hooks import ActivationRecorder  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

name = a.repo.split("/")[-1]
os.makedirs(a.out, exist_ok=True); os.makedirs(a.figs, exist_ok=True)
say = lambda *x: print(*x, flush=True)
T0 = time.time()
m = vlens.load_vlm(a.repo)
LAST = m.n_layers - 1
ALLL = list(range(max(m.n_deepstack, 1), LAST))
frac = {l: depth.to_frac(m, l) for l in range(m.n_layers)}
plot.use()
rng = random.Random(5)

COLOURS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
SHAPES = ["cube", "sphere", "cylinder"]
SAY_SHAPE = {"cube": "cube", "sphere": "ball", "cylinder": "cylinder"}
MATERIALS = ["metal", "rubber"]
COUNTS = ["1", "2", "3", "4", "5"]
SIDES = ["left", "right"]
ONE_WORD = " Answer with one word."


def tok1(word):
    """The id of ``word`` when the tokenizer emits it whole, in any surface form."""
    for form in (" " + word, word, " " + word.capitalize(), word.capitalize()):
        ids = m.tokenizer.encode(form, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    return None


CAND = {}
for nm, words in (("colour", COLOURS), ("shape", SHAPES), ("material", MATERIALS),
                  ("count", COUNTS), ("spatial", SIDES)):
    ids = {w: tok1(w) for w in words}
    CAND[nm] = {w: i for w, i in ids.items() if i is not None}
    if len(CAND[nm]) < len(words):
        say(f"  {nm}: dropped {[w for w in words if ids[w] is None]} (not single tokens)")
CAND["twostep"] = CAND["colour"]
say(f"{name}: candidate sets " + ", ".join(f"{k} {len(v)}" for k, v in CAND.items()))


def describe(o, colour=True, size=True, material=True):
    bits = []
    if size: bits.append(o["size"])
    if colour: bits.append(o["color"])
    if material: bits.append(o["material"])
    bits.append(SAY_SHAPE[o["shape"]])
    return "the " + " ".join(bits)


def unique_by(objs, keys, i):
    sig = tuple(objs[i][k] for k in keys)
    return sum(1 for j, o in enumerate(objs) if tuple(o[k] for k in keys) == sig) == 1


def build_items(n_each):
    """One item per task per scene where the question has an unambiguous one-word answer."""
    out = {k: [] for k in ("colour", "shape", "material", "count", "spatial", "twostep")}
    for s in clevr.scenes():
        if not clevr.have(s):
            continue
        objs = s["objects"]; path = clevr.image_path(s)
        R = s["relationships"]
        for i, o in enumerate(objs):
            if len(out["colour"]) < n_each and unique_by(objs, ("size", "material", "shape"), i) and o["color"] in CAND["colour"]:
                out["colour"].append(dict(path=path, q=f"What colour is {describe(o, colour=False)}?" + ONE_WORD, answer=o["color"]))
                break
        for i, o in enumerate(objs):
            if len(out["shape"]) < n_each and unique_by(objs, ("color",), i) and o["shape"] in CAND["shape"]:
                out["shape"].append(dict(path=path, q=f"What shape is the {o['color']} object? Call it a cube, a ball or a cylinder." + ONE_WORD,
                                         answer=o["shape"]))
                break
        for i, o in enumerate(objs):
            if len(out["material"]) < n_each and unique_by(objs, ("color", "shape"), i):
                out["material"].append(dict(path=path, q=f"Is the {o['color']} {SAY_SHAPE[o['shape']]} made of metal or rubber?" + ONE_WORD,
                                            answer=o["material"]))
                break
        cnt = Counter(o["shape"] for o in objs)
        for sh, c in cnt.items():
            if len(out["count"]) < n_each and 1 <= c <= 5:
                out["count"].append(dict(path=path, q=f"How many {SAY_SHAPE[sh]}s are in the picture? Answer with a number.", answer=str(c)))
                break
        for i, o in enumerate(objs):
            if len(out["spatial"]) >= n_each:
                break
            if not unique_by(objs, ("color", "shape"), i):
                continue
            for j, p in enumerate(objs):
                if j == i or not unique_by(objs, ("color", "shape"), j):
                    continue
                left = j in R["left"][i]      # p is left of o
                right = j in R["right"][i]
                if left == right:
                    continue
                out["spatial"].append(dict(path=path,
                                           q=f"Is the {p['color']} {SAY_SHAPE[p['shape']]} to the left or to the right of the {o['color']} {SAY_SHAPE[o['shape']]}?" + ONE_WORD,
                                           answer="left" if left else "right"))
                break
            break
        if all(len(v) >= n_each for k, v in out.items() if k != "twostep"):
            break
    for it in clevr.pointer_items(14, n_obj=(3, 7), min_sep=0.06, balance=5.0, limit=n_each * 3):
        if it["answer_main"] in CAND["colour"]:
            out["twostep"].append(dict(path=it["path"], q=it["question_main"] + ONE_WORD, answer=it["answer_main"]))
        if len(out["twostep"]) >= n_each:
            break
    return out


ITEMS = build_items(a.n + a.n_fit)
for k, v in ITEMS.items():
    say(f"  {k}: {len(v)} items, answers " + ", ".join(f"{w}:{c}" for w, c in Counter(x['answer'] for x in v).most_common(4)))

TASKS = [k for k, v in ITEMS.items() if len(v) >= 40]
fit = {k: ITEMS[k][: a.n_fit] for k in TASKS}
test = {k: ITEMS[k][a.n_fit: a.n_fit + a.n] for k in TASKS}


def states(items):
    """[N, n_layers, d] answer-spot residuals, and the model's own reply."""
    A, reps = [], []
    with torch.no_grad():
        for it in items:
            b = m.chat(it["q"], [corpus.load_image(it["path"], square=448)])
            with ActivationRecorder(m.layers, at=list(range(m.n_layers))) as rec:
                m.forward(b)
            A.append(np.stack([rec.activations[l][0, -1].float().cpu().numpy() for l in range(m.n_layers)]))
    return np.stack(A).astype(np.float32)


# ---- one V-Read map per task, fitted on unlabelled states of that task
RD = {}
for k in TASKS:
    t0 = time.time()
    RD[k] = reader.fit(states(fit[k]), ALLL, meta=dict(repo=a.repo, task=k, n=len(fit[k])))
    say(f"V-Read map for {k}: {len(fit[k])} unlabelled states in {(time.time() - t0) / 60:.1f} min")

# ---- read the candidates at every depth
res = {}
for k in TASKS:
    t0 = time.time()
    S = states(test[k])
    ids = [CAND[k][w] for w in CAND[k]]; words = list(CAND[k])
    truth = np.array([words.index(it["answer"]) for it in test[k]])
    acc = {"V-Read": [], "logit lens": []}
    traces = {"V-Read": [], "logit lens": []}
    with torch.no_grad():
        for l in ALLL:
            for tname, H in (("V-Read", RD[k].to_final(S[:, l], l)), ("logit lens", S[:, l])):
                lg = m.unembed(torch.as_tensor(H).to(m.device).to(torch.bfloat16)).float()
                lp = torch.log_softmax(lg, -1)[:, ids].cpu().numpy()
                acc[tname].append(float((lp.argmax(1) == truth).mean()))
                traces[tname].append(lp)
    final = {t: acc[t][-1] for t in acc}
    res[k] = dict(depths=[frac[l] for l in ALLL], accuracy=acc, n=len(test[k]),
                  chance=1.0 / len(words), final=final, words=words,
                  truth=truth.tolist(),
                  example=dict(q=test[k][0]["q"], answer=test[k][0]["answer"],
                               read=np.stack([tr[0] for tr in traces["V-Read"]]).tolist(),
                               logit=np.stack([tr[0] for tr in traces["logit lens"]]).tolist()))
    say(f"{k:<9} n={len(test[k])} chance {1/len(words):.2f} | V-Read final {final['V-Read']:.2f} "
        f"| logit lens final {final['logit lens']:.2f} | {(time.time() - t0) / 60:.1f} min")


def first_reaching(curve, target):
    for d, v in zip(res[k]["depths"], curve):
        if v >= target:
            return d
    return None


say(f"\ndepth at which each task reaches nine tenths of its own final accuracy")
say(f"{'task':<10} {'V-Read':>8} {'logit lens':>12} {'final':>7} {'chance':>7}")
order = []
for k in TASKS:
    r = res[k]
    dr = next((d for d, v in zip(r["depths"], r["accuracy"]["V-Read"]) if v >= 0.9 * r["final"]["V-Read"]), None)
    dl = next((d for d, v in zip(r["depths"], r["accuracy"]["logit lens"]) if v >= 0.9 * r["final"]["logit lens"]), None)
    r["depth90"] = dict(read=dr, logit=dl)
    order.append((dr if dr is not None else 9, k))
    say(f"{k:<10} {dr if dr is not None else float('nan'):>8.2f} {dl if dl is not None else float('nan'):>12.2f} "
        f"{r['final']['V-Read']:>7.2f} {r['chance']:>7.2f}")
say("\ntasks in the order their answers settle: " + " < ".join(k for _, k in sorted(order)))

# ---- figures
NICE = {"colour": "colour of an object", "shape": "shape of an object", "material": "what it is made of",
        "count": "how many of a shape", "spatial": "left or right", "twostep": "colour of the object beside another"}
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
for ax, tname in zip(axs, ("V-Read", "logit lens")):
    for k, c in zip(TASKS, plot.SERIES):
        ax.plot(res[k]["depths"], res[k]["accuracy"][tname], color=c, lw=1.5)
    ax.set_title(tname, loc="left", fontsize=10, color=plot.INK2)
    ax.set_xlabel("depth"); plot.despine(ax); ax.set_ylim(0, 1.02)
axs[0].set_ylabel("answer is the right one")
h = [plt.Line2D([], [], color=c, label=NICE.get(k, k)) for k, c in zip(TASKS, plot.SERIES)]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.16))
fig.tight_layout(); plot.save(fig, f"{a.figs}/{name}_tasks.png"); plt.close(fig)

fig, axs = plt.subplots(1, len(TASKS), figsize=(2.5 * len(TASKS), 2.9), sharey=True)
axs = list(np.atleast_1d(axs))
for ax, k in zip(axs, TASKS):
    r = res[k]; tr = np.asarray(r["example"]["read"]); words = r["words"]
    truth_w = r["example"]["answer"]
    for wi, w in enumerate(words):
        ax.plot(r["depths"], tr[:, wi], color=plot.SERIES[wi % len(plot.SERIES)],
                lw=1.8 if w == truth_w else 0.8, alpha=1.0 if w == truth_w else 0.5)
    ax.set_title(f"{NICE.get(k, k)}\nanswer: {truth_w}", loc="left", fontsize=8.5, color=plot.INK2)
    ax.set_xlabel("depth"); plot.despine(ax)
axs[0].set_ylabel("V-Read, log-probability of each candidate")
fig.tight_layout(); plot.save(fig, f"{a.figs}/{name}_task_traces.png"); plt.close(fig)

json.dump(dict(repo=a.repo, tasks={k: {kk: vv for kk, vv in res[k].items() if kk != "example"} for k in TASKS},
               examples={k: res[k]["example"] for k in TASKS}),
          open(f"{a.out}/{name}_tasks.json", "w"), indent=1, default=float)
say(f"\nwrote {a.out}/{name}_tasks.json  ({(time.time() - T0) / 60:.1f} min)")
