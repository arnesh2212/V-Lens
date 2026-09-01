# SPDX-License-Identifier: Apache-2.0
"""The slice: what every position, at every layer, is pushing the answer toward.

The Jacobian-lens paper's central picture is a layer x position grid whose cells
name the concept the lens reads there.  This is that picture for a VLM, in our
coordinates.  For a bank of words w (the COCO objects in the picture, some
absent distractors, and the refusal axis), the lens direction u_w^{c,l} is the
corpus-mean gradient of w's answer logit with respect to the residual at layer l
in token class c.  A position's reading for w is  u_w . (h_l[p] - mu_l^c).

Three panels per example:

  text slice     rows = layers, columns = the prompt's text tokens; each cell is
                 coloured by the word with the largest reading and shaded by
                 its margin over the runner-up -- the paper's slice, on the
                 request tokens.  Beside it, the same slice read through the
                 identity (output-layer row) transport.
  image grids    for the top words, a 14x14 patch map at six depths -- where in
                 the picture each concept is pushed from, and when.
  answer trace   each word's reading at the answer slot against depth, with the
                 logit lens's rank of the same word alongside -- when the answer
                 emerges, by two instruments.

Fitting the word bank costs one backward pass per word per prompt.  Runs on the
GPU; writes PNGs and a JSON of the readings so the page can quote numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--n-fit", type=int, default=60, help="COCO prompts to fit the word bank on")
ap.add_argument("--n-examples", type=int, default=4)
ap.add_argument("--distractors", nargs="*", default=["knife", "gun", "bicycle", "pizza", "elephant", "laptop"])
ap.add_argument("--prompt", default="Describe this image.")
ap.add_argument("--out", default="/workspace/vlm/results/slice")
ap.add_argument("--figdir", default="/workspace/vlm/figures/slice")
a = ap.parse_args()

import jvlm  # noqa: E402
from jvlm import axes, cocodata, corpus, depth, plot, readout, verdict  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from PIL import Image  # noqa: E402

name = a.repo.split("/")[-1]
os.makedirs(a.out, exist_ok=True); os.makedirs(a.figdir, exist_ok=True)
say = lambda *x: print(*x, flush=True)
m = jvlm.load_vlm(a.repo)
ALL = list(range(max(m.n_deepstack, 1), m.n_layers))          # every readable block
SIX = depth.layers_at(m, depth.STANDARD)
plot.use()

# ------------------------------------------------------------- the examples
pool = cocodata.concept_items(600, min_area=0.08, seed=202)
rng = np.random.default_rng(5)
examples = []
for it in pool:
    present = [c for c in it["present"] if cocodata.concept_token_ids(m.tokenizer, c)]
    if 2 <= len(present) <= 5:
        examples.append(dict(it, present=present))
    if len(examples) >= a.n_examples:
        break
words = sorted({w for ex in examples for w in ex["present"]} | set(a.distractors))
words = [w for w in words if cocodata.concept_token_ids(m.tokenizer, w)]
say(f"{len(examples)} examples; word bank ({len(words)}): {words}")

# ------------------------------------------------------------- the word bank
def word_axis(w):
    ids = cocodata.concept_token_ids(m.tokenizer, w)
    t = int(ids[0])
    return axes.Axis(f"word:{w}", lambda lg, h, t=t: lg[t].float(), "word", dict(token=t))

AX = [word_axis(w) for w in words]
fit_items = cocodata.concept_items(a.n_fit * 3, min_area=0.0, seed=101)[: a.n_fit]
def batches():
    for it in fit_items:
        yield m.chat(str(rng.choice(corpus.GENERIC_PROMPTS)), [corpus.load_image(it["path"], square=448)])
t0 = time.time()
F = verdict.fit(m, batches(), AX, ALL, every=20)
VL = verdict.VerdictLens.from_fit(F, dict(repo=a.repo, words=words, n=F.n))
VL.save(f"{a.out}/{name}__wordbank.pt")
say(f"word bank fitted: {len(words)} words x {len(ALL)} layers in {(time.time() - t0) / 60:.1f} min")

# ------------------------------------------------------------- readings
def residuals_all(batch):
    with ActivationRecorder(m.layers, at=ALL) as rec:
        m.forward(batch)
        return {l: rec.activations[l][0].detach().float().cpu() for l in ALL}


def readings(batch, H):
    """{word: array[n_layers, S]} of lens readings, plus the output-layer row rank of each word at each (layer, pos)."""
    S = batch["input_ids"].shape[1]
    img = m.image_mask(batch["input_ids"])[0].cpu().numpy()
    ch = np.where(img, "img", "txt"); ch[S - 1] = "ans"
    R = {w: np.zeros((len(ALL), S), np.float32) for w in words}
    for li, l in enumerate(ALL):
        h = H[l].numpy()
        for c in ("img", "txt", "ans"):
            pos = np.where(ch == c)[0]
            if len(pos) == 0:
                continue
            hc = h[pos] - VL.centre(c, l).numpy()
            for w in words:
                R[w][li, pos] = hc @ VL.dir(f"word:{w}", c, l).numpy()
    # identity / output-layer row: rank of each word's token at each (layer, pos)
    ranks = {w: np.zeros((len(ALL), S), np.int32) for w in words}
    with torch.no_grad():
        for li, l in enumerate(ALL):
            lg = m.unembed(H[l].to(m.device).to(torch.bfloat16)).float()          # [S, V]
            order = lg.argsort(-1, descending=True)
            for w in words:
                t = VL.meta and cocodata.concept_token_ids(m.tokenizer, w)[0]
                ranks[w][li] = (order == t).float().argmax(-1).cpu().numpy()
    return R, ranks, ch


PALETTE = plot.SERIES + ["#7a7a7a", "#c9a227", "#5b8c5a", "#b05a7a", "#3f7f9f", "#9f6b3f"]
WCOL = {w: PALETTE[i % len(PALETTE)] for i, w in enumerate(words)}


def draw_slice(ax, R, ch, tokens, which, title):
    """Layer x text-token grid; cell = word with the largest reading, alpha = margin."""
    pos = np.where(ch == "txt")[0].tolist() + [len(ch) - 1]
    stack = np.stack([R[w][:, pos] for w in words])                    # [W, L, T]
    top = stack.argmax(0); srt = np.sort(stack, 0); margin = srt[-1] - srt[-2]
    mx = np.percentile(margin, 95) or 1.0
    img = np.zeros((len(ALL), len(pos), 4))
    for i in range(len(ALL)):
        for j in range(len(pos)):
            c = to_rgba(WCOL[words[top[i, j]]]); img[i, j] = (*c[:3], float(min(margin[i, j] / mx, 1.0)))
    ax.imshow(img, aspect="auto", interpolation="nearest", origin="lower")
    ax.set_yticks(range(0, len(ALL), max(1, len(ALL) // 6))); ax.set_yticklabels([f"{depth.to_frac(m, ALL[i]):.2f}" for i in range(0, len(ALL), max(1, len(ALL) // 6))])
    ax.set_xticks(range(len(pos))); ax.set_xticklabels([tokens[p][:9] for p in pos], rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel("depth"); plot.panel_label(ax, title)
    ax.grid(False)


def draw_ranks(ax, ranks, ch, tokens, title):
    """The identity comparison: for each text position and layer, the best-ranked bank word (rank in colour)."""
    pos = np.where(ch == "txt")[0].tolist() + [len(ch) - 1]
    stack = np.stack([ranks[w][:, pos] for w in words])                # [W, L, T], lower is better
    top = stack.argmin(0); best = stack.min(0)
    img = np.zeros((len(ALL), len(pos), 4))
    for i in range(len(ALL)):
        for j in range(len(pos)):
            c = to_rgba(WCOL[words[top[i, j]]]); img[i, j] = (*c[:3], float(np.clip(1 - np.log10(best[i, j] + 1) / 4, 0, 1)))
    ax.imshow(img, aspect="auto", interpolation="nearest", origin="lower")
    ax.set_yticks([]); ax.set_xticks(range(len(pos))); ax.set_xticklabels([tokens[p][:9] for p in pos], rotation=60, ha="right", fontsize=7.5)
    plot.panel_label(ax, title); ax.grid(False)


out = []
for k, ex in enumerate(examples):
    img_ = corpus.load_image(ex["path"], square=448)
    b = m.chat(a.prompt, [img_]); S = b["input_ids"].shape[1]
    toks = [m.tokenizer.decode([int(t)]) for t in b["input_ids"][0]]
    H = residuals_all(b)
    R, ranks, ch = readings(b, H)
    P = int((ch == "img").sum()); grid = int(round(P ** 0.5)); ip = np.where(ch == "img")[0]

    # ---- figure 1: the two slices over the text tokens
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.05))
    draw_slice(axs[0], R, ch, toks, words, "writer: the word this token pushes the answer toward")
    draw_ranks(axs[1], ranks, ch, toks, "logit lens: the best-ranked bank word here")
    handles = [plt.Line2D([], [], marker="s", ls="", color=WCOL[w], label=w + (" (in picture)" if w in ex["present"] else "")) for w in words]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncols=min(6, len(words)), frameon=False)
    plot.save(fig, f"{a.figdir}/{name}_slice_text_{k}.png")

    # ---- figure 2: image grids at six depths for the present words
    show = ex["present"][:3]
    fig, axs = plt.subplots(len(show) + 1, len(SIX), figsize=(1.7 * len(SIX), 1.7 * (len(show) + 1)), gridspec_kw=dict(wspace=0.06, hspace=0.12))
    axs[0, 0].imshow(img_); axs[0, 0].set_axis_off()
    for j in range(1, len(SIX)): axs[0, j].set_axis_off()
    axs[0, 1].text(0, 0.5, f"“{a.prompt}”\nin picture: {', '.join(ex['present'])}", transform=axs[0, 1].transAxes, fontsize=8.5, color=plot.INK2, va="center")
    for r_, w in enumerate(show, start=1):
        for j, l in enumerate(SIX):
            li = ALL.index(l); M = R[w][li, ip].reshape(grid, grid); s = np.abs(M).max() or 1
            ax = axs[r_, j]; ax.imshow(M, cmap="RdBu_r", vmin=-s, vmax=s, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
            if r_ == 1: ax.set_title(f"depth {depth.to_frac(m, l):.2f}", fontsize=8.5, color=plot.INK2)
            if j == 0: ax.set_ylabel(f"“{w}”", fontsize=9, color=plot.INK2)
    plot.save(fig, f"{a.figdir}/{name}_slice_grid_{k}.png")

    # ---- figure 3: the answer slot against depth, lens reading vs output-layer row rank
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.2), gridspec_kw=dict(wspace=0.3))
    x = [depth.to_frac(m, l) for l in ALL]
    for w in words:
        lw, al = (1.9, 1.0) if w in ex["present"] else (1.0, 0.45)
        axs[0].plot(x, R[w][:, S - 1], color=WCOL[w], lw=lw, alpha=al)
        axs[1].plot(x, ranks[w][:, S - 1] + 1, color=WCOL[w], lw=lw, alpha=al)
    axs[0].set_xlabel("depth"); axs[0].set_ylabel("lens reading at the answer slot\n(logits toward the word)")
    axs[1].set_xlabel("depth"); axs[1].set_ylabel("output-layer row rank of the word\n(lower is better)"); axs[1].set_yscale("log"); axs[1].invert_yaxis()
    plot.panel_label(axs[0], "push"); plot.panel_label(axs[1], "logit lens")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncols=min(6, len(words)), frameon=False)
    plot.save(fig, f"{a.figdir}/{name}_slice_answer_{k}.png")

    # numbers for the page: depth at which each present word first leads the bank at the answer slot, by instrument
    rec = dict(image_id=ex["image_id"], present=ex["present"], words=words)
    for w in ex["present"]:
        lead = [i for i in range(len(ALL)) if all(R[w][i, S - 1] >= R[v][i, S - 1] for v in words)]
        lead_r = [i for i in range(len(ALL)) if all(ranks[w][i, S - 1] <= ranks[v][i, S - 1] for v in words)]
        rec[w] = dict(lens_first_lead_depth=(depth.to_frac(m, ALL[lead[0]]) if lead else None),
                      logit_first_lead_depth=(depth.to_frac(m, ALL[lead_r[0]]) if lead_r else None),
                      lens_share_img_mass_on_object=None)
    out.append(rec)
    say(f"example {k}: {ex['present']} -> {json.dumps({w: rec[w] for w in ex['present']})}")

json.dump(out, open(f"{a.out}/{name}_slices.json", "w"), indent=1)
say(f"wrote {a.figdir}/*  and  {a.out}/{name}_slices.json   ({(time.time() - t0) / 60:.1f} min)")
