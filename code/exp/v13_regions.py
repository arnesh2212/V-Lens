# SPDX-License-Identifier: Apache-2.0
"""Regions and emergence: what each part of the picture pushes the answer toward,
at each depth, and when a yes/no answer arrives at the answer slot.

(a) Regions.  COCO photographs with their boxes.  A bank of common object words.
    For every object of the bank that is in the picture, its region is the set of
    patches whose centres fall inside its boxes; the rest of the frame is
    background.  At each depth, each instrument scores every bank word on every
    region (the mean of its map over the region's patches) and the winner is the
    word with the largest score.  Region accuracy is how often the winner is the
    object that is actually there; chance is one over the bank size.
    Instruments: the writer's corpus-mean direction (u_w at image tokens, centred),
    the writer per item (this picture's own gradient of the word's answer-slot
    logit, times the centred activation, with a per-position corpus mean as the
    baseline), and the logit lens at the patch (the word's logit from unembedding
    the patch residual), plain and centred by the per-position mean.
    Localisation: share of a word's positive map mass inside its own boxes,
    divided by the boxes' share of the frame (1 = chance), and the pointing game.

(b) Emergence.  "Is there a {w} in this picture? Answer yes or no." for two present
    and two absent (same-supercategory) words per image.  The yes/no margin is the
    axis.  Read at the answer slot at every depth by the writer (u_yesno at the
    answer slot), the reader (ridge transport fitted on unlabelled answer slots
    under the same prompt family), the logit lens, and a labelled probe.  AUROC
    present vs absent per depth; the depth at which each instrument first reads
    the answer; and among items the model answers "yes", whether an instrument
    tells a true yes from a hallucinated one before the model speaks.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--n-fit", type=int, default=80, help="COCO images for the word bank and the per-position means")
ap.add_argument("--n-fit-yn", type=int, default=200, help="COCO images for the yes/no writer and reader")
ap.add_argument("--n-eval", type=int, default=120)
ap.add_argument("--n-examples", type=int, default=4)
ap.add_argument("--absent", choices=["supercategory", "cooccurring"], default="cooccurring",
                help="how the absent word is chosen: a sibling category, or the absent word that most often appears with what is here (the harder POPE-style negative)")
ap.add_argument("--out", default="/workspace/vlm/results/regions")
ap.add_argument("--figs", default="/workspace/vlm/figures/regions")
a = ap.parse_args()

import jvlm  # noqa: E402
from jvlm import axes, cocodata, corpus, depth, plot, reader, readout, verdict, vmap  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

name = a.repo.split("/")[-1]
os.makedirs(a.out, exist_ok=True); os.makedirs(a.figs, exist_ok=True)
say = lambda *x: print(*x, flush=True)
T0 = time.time()
m = jvlm.load_vlm(a.repo)
L6 = depth.layers_at(m, depth.STANDARD)
ALLL = list(range(max(m.n_deepstack, 1), m.n_layers - 1))
frac = {l: depth.to_frac(m, l) for l in range(m.n_layers)}
plot.use()
DESCRIBE = "Describe this image."
YN = "Is there a {w} in this picture? Answer yes or no."

BANK0 = ["person", "car", "chair", "cup", "bottle", "dog", "cat", "bed", "book", "bowl", "laptop", "tv", "pizza",
         "knife", "fork", "bicycle", "elephant", "horse", "bus", "truck", "boat", "bird", "clock", "umbrella"]
BANK = [w for w in BANK0 if cocodata.concept_token_ids(m.tokenizer, w)]
TOK = {w: int(cocodata.concept_token_ids(m.tokenizer, w)[0]) for w in BANK}
W = len(BANK)
say(f"{name}: bank of {W} words; depths {[round(frac[l], 2) for l in L6]}")
COCO = cocodata._load()
YES = axes.token_set(m.tokenizer, axes.YES_FORMS); NO = axes.token_set(m.tokenizer, axes.NO_FORMS)


def img(it): return corpus.load_image(it["path"], square=448)
def margin(logits):  # yes/no margin from a [.., V] logit tensor
    lp = torch.log_softmax(logits.float(), -1)
    return (torch.logsumexp(lp[..., YES], -1) - torch.logsumexp(lp[..., NO], -1))


# ------------------------------------------------------------------ fit corpora
fit_pool = cocodata.concept_items(a.n_fit_yn * 3, min_area=0.0, seed=101)
seen, fit_items = set(), []
for it in fit_pool:
    if it["image_id"] not in seen:
        seen.add(it["image_id"]); fit_items.append(it)
fit_word = fit_items[: a.n_fit]; fit_yn = fit_items[: a.n_fit_yn]
rng = random.Random(7)

# (1) the word bank at image tokens, under the describe prompt
AXW = [axes.Axis(f"word:{w}", (lambda lg, h, t=TOK[w]: lg[t].float()), "word", dict(token=TOK[w])) for w in BANK]
def word_batches():
    for it in fit_word:
        yield m.chat(DESCRIBE, [img(it)])
t0 = time.time()
Fw = verdict.fit(m, word_batches(), AXW, L6, every=40)
VLw = verdict.VerdictLens.from_fit(Fw, dict(repo=a.repo, words=BANK, n=Fw.n, prompt=DESCRIBE))
say(f"word bank fitted: {W} words x {len(L6)} depths on {Fw.n} images in {(time.time() - t0) / 60:.1f} min")

# (2) per-position means of the image residuals at the six depths (the baseline of the per-item reading)
MU, nmu = {l: None for l in L6}, 0
with torch.no_grad():
    for it in fit_word:
        b = m.chat(DESCRIBE, [img(it)]); pos = readout.image_positions(m, b)
        with ActivationRecorder(m.layers, at=L6) as rec:
            m.forward(b)
        for l in L6:
            h = rec.activations[l][0, pos.to(rec.activations[l].device)].float().cpu()
            if MU[l] is None: MU[l] = torch.zeros_like(h)
            if MU[l].shape == h.shape: MU[l] += h
        nmu += 1
MU = {l: MU[l] / nmu for l in L6}
P = MU[L6[0]].shape[0]; G = int(round(math.sqrt(P)))
say(f"per-position means over {nmu} images; {P} image tokens = {G}x{G}")

# (3) the yes/no writer at the answer slot and the reader, under the yes/no prompt family
AXY = [axes.yesno(m.tokenizer)]
ANS_fit = []
def yn_batches():
    for it in fit_yn:
        w = rng.choice(BANK)
        b = m.chat(YN.format(w=w), [img(it)])
        with torch.no_grad(), ActivationRecorder(m.layers, at=list(range(m.n_layers))) as rec:
            m.forward(b)
        ANS_fit.append(np.stack([rec.activations[l][0, -1].float().cpu().numpy() for l in range(m.n_layers)]))
        yield b
t0 = time.time()
Fy = verdict.fit(m, yn_batches(), AXY, ALLL, every=50)
VLy = verdict.VerdictLens.from_fit(Fy, dict(repo=a.repo, n=Fy.n, prompt=YN))
ANS_fit = np.stack(ANS_fit).astype(np.float32)
RD = reader.fit(ANS_fit, ALLL, meta=dict(repo=a.repo, corpus="COCO yes/no", n=len(ANS_fit)))
RD.save(f"{a.out}/{name}__reader_yn.pt")
say(f"yes/no writer and reader fitted on {len(ANS_fit)} answer slots in {(time.time() - t0) / 60:.1f} min")


# ------------------------------------------------------------------ evaluation items
eval_pool = cocodata.concept_items(a.n_eval * 6, min_area=0.03, seed=303)
seen, ev = set(), []
for it in eval_pool:
    iid = it["image_id"]
    if iid in seen or iid in {f["image_id"] for f in fit_items}:
        continue
    pres_all = cocodata.present(iid)
    pres = [w for w in BANK if w in pres_all and COCO["boxes"].get(iid, {}).get(w)]
    if not pres:
        continue
    seen.add(iid)
    absent = [w for w in BANK if w not in pres_all]
    if a.absent == "cooccurring":
        # the absent word that most often shares a photograph with what is here: the
        # negative a model is most likely to hallucinate, as in POPE's adversarial split
        score = {w: 0 for w in absent}
        for p in pres_all:
            for w, c in COCO["cooc"].get(p, {}).items():
                if w in score:
                    score[w] += int(c)
        hard = sorted(absent, key=lambda w: -score[w])[:6]
    else:
        hard = [w for w in absent if any(COCO["super_of"].get(w) == COCO["super_of"].get(p) for p in pres)]
    rng.shuffle(pres); rng.shuffle(absent)
    if a.absent != "cooccurring":
        rng.shuffle(hard)
    ev.append(dict(image_id=iid, path=it["path"], present=pres, present_all=sorted(pres_all),
                   ask_present=pres[:2], ask_absent=(hard + [w for w in absent if w not in hard])[:2]))
    if len(ev) >= a.n_eval:
        break
say(f"{len(ev)} evaluation images, {sum(len(e['present']) for e in ev)} present bank objects; absent words chosen by {a.absent}")


def region_masks(e):
    """{word: bool[P]} for the present bank objects, plus 'background'."""
    cy = (np.arange(G)[:, None] + 0.5) / G; cx = (np.arange(G)[None, :] + 0.5) / G
    masks, anybox = {}, np.zeros((G, G), bool)
    for w, boxes in COCO["boxes"].get(e["image_id"], {}).items():
        mk = np.zeros((G, G), bool)
        for (x0, y0, bw, bh) in boxes:
            mk |= (cx >= x0) & (cx <= x0 + bw) & (cy >= y0) & (cy <= y0 + bh)
        anybox |= mk
        if w in e["present"]:
            masks[w] = mk.reshape(-1)
    masks["background"] = (~anybox).reshape(-1)
    return masks


# ------------------------------------------------------------------ (a) regions
UW = {l: torch.stack([VLw.dir(f"word:{w}", "img", l) for w in BANK]).float() for l in L6}   # [W, d]
CW = {l: VLw.centre("img", l).float() for l in L6}
TOKS = [TOK[w] for w in BANK]

def maps_for(e):
    """{instrument: {layer: [W, P]}} plus the image residuals."""
    b = m.chat(DESCRIBE, [img(e)]); pos = readout.image_positions(m, b)
    if len(pos) != P:
        return None
    with torch.no_grad(), ActivationRecorder(m.layers, at=L6) as rec:
        m.forward(b)
    H = {l: rec.activations[l][0, pos.to(rec.activations[l].device)].float().cpu() for l in L6}
    out = {"V-Write": {}, "logit lens": {}, "logit lens, centred": {}}
    with torch.no_grad():
        for l in L6:
            out["V-Write"][l] = ((H[l] - CW[l]) @ UW[l].T).T.numpy()                              # [W, P]
            lg = m.unembed(H[l].to(m.device).to(torch.bfloat16)).float()[:, TOKS].cpu().numpy()  # [P, W]
            out["logit lens"][l] = lg.T
            lgc = m.unembed((H[l] - MU[l]).to(m.device).to(torch.bfloat16)).float()[:, TOKS].cpu().numpy()
            out["logit lens, centred"][l] = lgc.T
    pi = vmap.do_map(m, b, TOKS, L6, pos, means=MU)                                              # {l: [W, P]}
    out["V-Write, this item"] = {l: pi[l].numpy() for l in L6}
    return out

INSTR = ["V-Write, this item", "V-Write", "logit lens", "logit lens, centred"]
acc = {k: {l: [] for l in L6} for k in INSTR}
bg_present = {k: {l: [] for l in L6} for k in INSTR}
conc = {k: {l: [] for l in L6} for k in INSTR}      # concentration ratio, per (image, word)
point = {k: {l: [] for l in L6} for k in INSTR}     # pointing-game hit
area_chance = []
examples = []
t0 = time.time()
for n, e in enumerate(ev):
    M = maps_for(e)
    if M is None:
        continue
    masks = region_masks(e)
    for k in INSTR:
        for l in L6:
            A = M[k][l]                                                # [W, P]
            for w in e["present"]:
                mk = masks[w]
                if mk.sum() < 2:
                    continue
                score = A[:, mk].mean(1)
                acc[k][l].append(float(BANK[int(score.argmax())] == w))
                row = A[BANK.index(w)]; posmass = np.clip(row, 0, None)
                share = posmass[mk].sum() / max(posmass.sum(), 1e-9); af = mk.mean()
                conc[k][l].append(float(share / max(af, 1e-9)))
                point[k][l].append(float(mk[int(row.argmax())]))
                if k == INSTR[0] and l == L6[0]:
                    area_chance.append(float(af))
            bg = masks["background"]
            if bg.sum() >= 4:
                sb = A[:, bg].mean(1)
                bg_present[k][l].append(float(BANK[int(sb.argmax())] in e["present"]))
    if len(examples) < a.n_examples and 2 <= len(e["present"]) <= 3:
        examples.append((e, M, masks))
    if (n + 1) % 20 == 0:
        say(f"  {n + 1} images  ({(time.time() - t0) / 60:.1f} min)")
say(f"regions scored on {len(acc[INSTR[0]][L6[0]])} (image, object) pairs; mean box share of frame {np.mean(area_chance):.3f}")

say(f"\nregion accuracy (winner word = the object there), chance {1 / W:.3f}")
say(f"{'depth':>6} " + " ".join(f"{k:>20}" for k in INSTR))
for l in L6:
    say(f"{frac[l]:>6.2f} " + " ".join(f"{np.mean(acc[k][l]):>20.3f}" for k in INSTR))
say(f"\nconcentration ratio (share of positive mass inside the object's boxes / box share; 1 = chance), median")
for l in L6:
    say(f"{frac[l]:>6.2f} " + " ".join(f"{np.median(conc[k][l]):>20.2f}" for k in INSTR))
say(f"\npointing game (strongest patch inside the boxes), chance {np.mean(area_chance):.3f}")
for l in L6:
    say(f"{frac[l]:>6.2f} " + " ".join(f"{np.mean(point[k][l]):>20.3f}" for k in INSTR))


# ------------------------------------------------------------------ (b) emergence
items = []
t0 = time.time()
for n, e in enumerate(ev):
    for w, present in [(w, 1) for w in e["ask_present"]] + [(w, 0) for w in e["ask_absent"]]:
        b = m.chat(YN.format(w=w), [img(e)])
        with torch.no_grad(), ActivationRecorder(m.layers, at=list(range(m.n_layers))) as rec:
            m.forward(b)
        ans = np.stack([rec.activations[l][0, -1].float().cpu().numpy() for l in range(m.n_layers)])
        with torch.no_grad():
            mf = float(margin(m.unembed(rec.activations[m.n_layers - 1][0, -1][None].to(torch.bfloat16))[0]))
        items.append(dict(image_id=e["image_id"], word=w, present=present, ans=ans.astype(np.float32), margin_final=mf))
    if (n + 1) % 30 == 0:
        say(f"  yes/no {len(items)} items  ({(time.time() - t0) / 60:.1f} min)")
ANS = np.stack([it["ans"] for it in items]); y = np.array([it["present"] for it in items]); mf = np.array([it["margin_final"] for it in items])
says_yes = mf > 0
say(f"{len(items)} yes/no items; model accuracy {np.mean((mf > 0) == (y == 1)):.3f}; says yes on {says_yes.mean():.2f}; "
    f"hallucinated yes {int(((y == 0) & says_yes).sum())}, missed present {int(((y == 1) & ~says_yes).sum())}")

def auc(s, yy):
    return float(roc_auc_score(yy, s)) if len(np.unique(yy)) == 2 else float("nan")
def probe_scores(X, yy, seed=0):
    out = np.zeros(len(yy))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, yy):
        sc = StandardScaler().fit(X[tr]); clf = LogisticRegression(max_iter=500, C=0.1).fit(sc.transform(X[tr]), yy[tr])
        out[te] = clf.decision_function(sc.transform(X[te]))
    return out

EM = {"V-Write": {}, "reader": {}, "logit lens": {}, "probe": {}}
HAL = {k: {} for k in EM}
with torch.no_grad():
    for l in ALLL:
        X = ANS[:, l]
        s_w = np.asarray(VLy.read(X, "yesno", "ans", l), dtype=float)
        s_r = margin(m.unembed(torch.as_tensor(RD.to_final(X, l)).to(m.device).to(torch.bfloat16))).cpu().numpy()
        s_i = margin(m.unembed(torch.as_tensor(X).to(m.device).to(torch.bfloat16))).cpu().numpy()
        s_p = probe_scores(X, y)
        for k, s in (("V-Write", s_w), ("reader", s_r), ("logit lens", s_i), ("probe", s_p)):
            EM[k][l] = auc(s, y)
            HAL[k][l] = auc(s[says_yes], y[says_yes]) if says_yes.sum() >= 20 and (y[says_yes] == 0).sum() >= 8 else float("nan")

say(f"\nemergence: AUROC present vs absent at the answer slot")
say(f"{'depth':>6} {'write':>8} {'reader':>8} {'logit':>8} {'probe':>8}   | true yes vs hallucinated yes: {'write':>7} {'reader':>7} {'logit':>7} {'probe':>7}")
for l in ALLL:
    say(f"{frac[l]:>6.2f} {EM['write'][l]:>8.3f} {EM['reader'][l]:>8.3f} {EM['logit lens'][l]:>8.3f} {EM['probe'][l]:>8.3f}   | "
        f"{'':>29} {HAL['write'][l]:>7.3f} {HAL['reader'][l]:>7.3f} {HAL['logit lens'][l]:>7.3f} {HAL['probe'][l]:>7.3f}")
def first_depth(curve, thr=0.9):
    for l in ALLL:
        if curve[l] >= thr:
            return frac[l]
    return None
say("first depth with AUROC >= 0.9: " + ", ".join(f"{k} {first_depth(EM[k])}" for k in EM))


# ------------------------------------------------------------------ figures
COL = {w: (plot.SERIES + ["#7a7a7a", "#c9a227", "#5b8c5a", "#b05a7a", "#3f7f9f", "#9f6b3f", "#6b9f3f", "#9f3f6b",
                          "#3f9f9f", "#9f9f3f", "#3f3f9f", "#9f3f3f", "#6b6b9f", "#9f6b6b", "#6b9f9f", "#9f9f6b"])[i] for i, w in enumerate(BANK)}
ICOL = {"V-Write, this item": plot.SERIES[0], "V-Write": plot.SERIES[2], "logit lens": plot.SERIES[1], "logit lens, centred": plot.SERIES[3],
        "reader": plot.SERIES[0], "probe": plot.INK2}

# F1 region accuracy and concentration vs depth
fig, axs = plt.subplots(1, 3, figsize=(9.6, 2.9))
xs = [frac[l] for l in L6]
for k in INSTR:
    axs[0].plot(xs, [np.mean(acc[k][l]) for l in L6], color=ICOL[k], lw=1.6 if k == INSTR[0] else 1.2)
    plot.end_label(axs[0], xs[-1], np.mean(acc[k][L6[-1]]), k, ICOL[k])
    axs[1].plot(xs, [np.median(conc[k][l]) for l in L6], color=ICOL[k], lw=1.6 if k == INSTR[0] else 1.2)
    axs[2].plot(xs, [np.mean(point[k][l]) for l in L6], color=ICOL[k], lw=1.6 if k == INSTR[0] else 1.2)
plot.chance_line(axs[0], 1 / W); plot.chance_line(axs[1], 1.0); plot.chance_line(axs[2], float(np.mean(area_chance)))
axs[0].set_title("region names its object", loc="left", fontsize=9.5, color=plot.INK2); axs[0].set_ylabel("accuracy"); axs[0].set_ylim(0, 1)
axs[1].set_title("mass inside the object's boxes, over chance", loc="left", fontsize=9.5, color=plot.INK2); axs[1].set_ylabel("ratio")
axs[2].set_title("strongest patch on the object", loc="left", fontsize=9.5, color=plot.INK2); axs[2].set_ylabel("hit rate"); axs[2].set_ylim(0, 1)
for ax in axs: ax.set_xlabel("depth"); plot.despine(ax)
fig.tight_layout(); plot.save(fig, f"{a.figs}/{name}_regions_curves.png"); plt.close(fig)

# F2 examples: per-item maps by depth for each present word, boxes outlined; and the three instruments at one depth
for ei, (e, M, masks) in enumerate(examples):
    words = e["present"][:3]; im = Image.open(e["path"]).convert("RGB").resize((448, 448))
    fig = plt.figure(figsize=(1.6 * (len(L6) + 1) + 0.6, 1.65 * len(words) + 0.4))
    gs = fig.add_gridspec(len(words), len(L6) + 1, width_ratios=[1.25] + [1] * len(L6), wspace=0.06, hspace=0.12, left=0.01, right=0.99, top=0.9, bottom=0.02)
    ax = fig.add_subplot(gs[:, 0]); ax.imshow(im); ax.set_axis_off()
    for w in words:
        for (x0, y0, bw, bh) in COCO["boxes"][e["image_id"]][w]:
            ax.add_patch(Rectangle((x0 * 448, y0 * 448), bw * 448, bh * 448, fill=False, ec=COL[w], lw=1.4))
    ax.set_title("in the picture: " + ", ".join(words), loc="left", fontsize=8.5, color=plot.INK2)
    for r, w in enumerate(words):
        for j, l in enumerate(L6):
            A = M["V-Write, this item"][l][BANK.index(w)].reshape(G, G); s = float(np.percentile(np.abs(A), 99)) or 1.0
            ax = fig.add_subplot(gs[r, j + 1]); ax.imshow(np.clip(A, -s, s), cmap="RdBu_r", vmin=-s, vmax=s, interpolation="nearest")
            for (x0, y0, bw, bh) in COCO["boxes"][e["image_id"]][w]:
                ax.add_patch(Rectangle((x0 * G - 0.5, y0 * G - 0.5), bw * G, bh * G, fill=False, ec=COL[w], lw=1.2))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if r == 0: ax.set_title(f"depth {frac[l]:.2f}", fontsize=9, color=plot.INK2)
            if j == 0: ax.set_ylabel(f"“{w}”", fontsize=9.5, color=plot.INK2)
    plot.save(fig, f"{a.figs}/{name}_regions_example_{ei}.png"); plt.close(fig)

    lmid = L6[2]
    fig = plt.figure(figsize=(1.7 * (len(INSTR) + 1) + 0.4, 1.7 * len(words) + 0.5))
    gs = fig.add_gridspec(len(words), len(INSTR) + 1, width_ratios=[1.2] + [1] * len(INSTR), wspace=0.06, hspace=0.14, left=0.01, right=0.99, top=0.88, bottom=0.02)
    ax = fig.add_subplot(gs[:, 0]); ax.imshow(im); ax.set_axis_off()
    for w in words:
        for (x0, y0, bw, bh) in COCO["boxes"][e["image_id"]][w]:
            ax.add_patch(Rectangle((x0 * 448, y0 * 448), bw * 448, bh * 448, fill=False, ec=COL[w], lw=1.4))
    for r, w in enumerate(words):
        for j, k in enumerate(INSTR):
            A = M[k][lmid][BANK.index(w)].reshape(G, G); s = float(np.percentile(np.abs(A), 99)) or 1.0
            ax = fig.add_subplot(gs[r, j + 1]); ax.imshow(np.clip(A, -s, s), cmap="RdBu_r", vmin=-s, vmax=s, interpolation="nearest")
            for (x0, y0, bw, bh) in COCO["boxes"][e["image_id"]][w]:
                ax.add_patch(Rectangle((x0 * G - 0.5, y0 * G - 0.5), bw * G, bh * G, fill=False, ec=COL[w], lw=1.2))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if r == 0: ax.set_title(k, fontsize=9, color=plot.INK2)
            if j == 0: ax.set_ylabel(f"“{w}”", fontsize=9.5, color=plot.INK2)
    fig.text(0.01, 0.955, f"depth {frac[lmid]:.2f}", fontsize=9, color=plot.INK2)
    plot.save(fig, f"{a.figs}/{name}_regions_instruments_{ei}.png"); plt.close(fig)

    # F4 the slice: rows = depth, columns = regions, cell = winning word, shade = margin
    regions = words + ["background"]
    fig, axs = plt.subplots(1, 3, figsize=(9.4, 0.42 * len(L6) + 1.6), sharey=True)
    for ax, k in zip(axs, ["V-Write, this item", "V-Write", "logit lens"]):
        for i, l in enumerate(L6):
            for j, rg in enumerate(regions):
                mk = masks[rg]
                if mk.sum() < 2:
                    continue
                sc = M[k][l][:, mk].mean(1); order = np.argsort(sc)[::-1]
                win = BANK[order[0]]; marg = sc[order[0]] - sc[order[1]]; scale = np.abs(sc).max() or 1.0
                alpha = float(np.clip(marg / scale, 0.15, 1.0))
                ax.add_patch(Rectangle((j, i), 1, 1, color=COL[win], alpha=alpha, lw=0))
                ax.text(j + 0.5, i + 0.5, win, ha="center", va="center", fontsize=7.5, color=plot.INK)
        ax.set_xlim(0, len(regions)); ax.set_ylim(len(L6), 0)
        ax.set_xticks(np.arange(len(regions)) + 0.5); ax.set_xticklabels([f"{r} patches" if r == "background" else f"{r} region" for r in regions], fontsize=8)
        ax.set_yticks(np.arange(len(L6)) + 0.5); ax.set_yticklabels([f"{frac[l]:.2f}" for l in L6], fontsize=8)
        ax.set_title(k, loc="left", fontsize=9.5, color=plot.INK2)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(length=0)
    axs[0].set_ylabel("depth")
    fig.tight_layout(); plot.save(fig, f"{a.figs}/{name}_regions_slice_{ei}.png"); plt.close(fig)

# F3 emergence
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9))
xs = [frac[l] for l in ALLL]
for k, c in (("reader", plot.SERIES[0]), ("V-Write", plot.SERIES[2]), ("logit lens", plot.SERIES[1]), ("probe", plot.INK2)):
    axs[0].plot(xs, [EM[k][l] for l in ALLL], color=c, lw=1.5 if k == "reader" else 1.1, ls=":" if k == "probe" else "-")
    plot.end_label(axs[0], xs[-1], EM[k][ALLL[-1]], k, c)
    hv = [HAL[k][l] for l in ALLL]
    if not all(np.isnan(hv)):
        axs[1].plot(xs, hv, color=c, lw=1.5 if k == "reader" else 1.1, ls=":" if k == "probe" else "-")
plot.chance_line(axs[0], 0.5); plot.chance_line(axs[1], 0.5)
axs[0].set_title("is the object there? present vs absent", loc="left", fontsize=9.5, color=plot.INK2); axs[0].set_ylabel("AUROC"); axs[0].set_ylim(0.4, 1.02)
axs[1].set_title("among the model's yes answers: true vs hallucinated", loc="left", fontsize=9.5, color=plot.INK2); axs[1].set_ylim(0.4, 1.02)
for ax in axs: ax.set_xlabel("depth"); plot.despine(ax)
fig.tight_layout(); plot.save(fig, f"{a.figs}/{name}_emergence.png"); plt.close(fig)

json.dump(dict(repo=a.repo, bank=BANK, n_images=len(ev), n_pairs=len(acc[INSTR[0]][L6[0]]), box_share=float(np.mean(area_chance)),
               depths={str(l): frac[l] for l in L6},
               regions=dict(accuracy={k: {str(l): float(np.mean(acc[k][l])) for l in L6} for k in INSTR},
                            concentration_median={k: {str(l): float(np.median(conc[k][l])) for l in L6} for k in INSTR},
                            pointing={k: {str(l): float(np.mean(point[k][l])) for l in L6} for k in INSTR},
                            background_names_present={k: {str(l): float(np.mean(bg_present[k][l])) for l in L6} for k in INSTR}),
               emergence=dict(n_items=len(items), model_accuracy=float(np.mean((mf > 0) == (y == 1))), n_hallucinated=int(((y == 0) & says_yes).sum()),
                              auroc={k: {str(l): EM[k][l] for l in ALLL} for k in EM}, hallucination_auroc={k: {str(l): HAL[k][l] for l in ALLL} for k in HAL},
                              depth_of={str(l): frac[l] for l in ALLL}, first_depth_09={k: first_depth(EM[k]) for k in EM}),
               examples=[dict(image_id=e["image_id"], present=e["present"]) for e, _, _ in examples]),
          open(f"{a.out}/{name}_regions.json", "w"), indent=1, default=float)
say(f"wrote {a.out}/{name}_regions.json and {a.figs}/{name}_*  ({(time.time() - T0) / 60:.1f} min)")
