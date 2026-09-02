# SPDX-License-Identifier: Apache-2.0
"""What the answer slot is about to say, layer by layer, and in what order the
pieces of the decision arrive.

Two readings of the same cached states, both through the reader's affine
transport, both compared with the logit lens which is the same reading with the
transport set to the identity.

(a) Words.  Unembed the predicted final state at each depth and list the tokens
    it puts first.  This is the vocabulary picture the logit lens gives for text
    models, at the one position in a VLM where the image can still change the
    answer.  Reported: the depth at which the model's own first answer token
    enters the top 1, top 5 and top 20 under each transport, and example tables.

(b) Order.  Four scalars of the same predicted state, each an axis:
      content   the log-probability of the harmful phrase's own content word
      harm      a harm lexicon against a benign one
      refusal   the behaviour-anchored refusal axis
      opener    refusal openers against compliance openers
    For each, the AUROC separating harmful from benign items at every depth, and
    the onset depth where that first reaches 0.7. The order of the onsets is the
    cascade: what the model works out first and what it concludes later.

Runs from the gate cache; the only GPU work is the unembedding.
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
ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--variants", nargs="*", default=["SD_TYPO"])
ap.add_argument("--n-examples", type=int, default=6)
ap.add_argument("--boot", type=int, default=400)
ap.add_argument("--gate-dir", default="/workspace/vlm/results/gate")
ap.add_argument("--reader", default=None)
ap.add_argument("--out", default="/workspace/vlm/results/cascade")
ap.add_argument("--figs", default="/workspace/vlm/figures/cascade")
a = ap.parse_args()

import jvlm  # noqa: E402
from jvlm import axes, cache, depth, judge, plot, reader, safedata, verdict  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

name = a.repo.split("/")[-1]
os.makedirs(a.out, exist_ok=True); os.makedirs(a.figs, exist_ok=True)
say = lambda *x: print(*x, flush=True)
T0 = time.time()
m = jvlm.load_vlm(a.repo)
ALLL = list(range(max(m.n_deepstack, 1), m.n_layers - 1))
frac = {l: depth.to_frac(m, l) for l in range(m.n_layers)}
plot.use()

acts, side = cache.load(f"{a.gate_dir}/{name}__eval.npz")
keep = np.array([it.get("variant") in a.variants for it in side["items"]])
its = [it for it, k in zip(side["items"], keep) if k]
ANS = side["_extra"]["ANS"].astype(np.float32)[keep]
y_h = np.array([bool(it["harmful"]) for it in its])
y_r = np.array([judge.refused(it) for it in its], int)
r_ref = verdict.dim_direction(ANS[:, -1], y_r).numpy()
RP = a.reader or f"/workspace/vlm/results/reader/{name}__reader.pt"
RD = reader.Reader.load(RP)
say(f"{len(its)} items in {a.variants}: harmful {y_h.mean():.2f}, refused {y_r.mean():.2f}")

# ---- the content word of each item's own phrase, for the content axis
STOP = {"the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "how", "what", "steps", "list",
        "detail", "perform", "this", "activity", "image", "shows", "phrase", "can", "i", "you", "my", "with"}
def content_token(txt):
    for w in (txt or "").lower().replace(",", " ").replace(".", " ").split():
        if w in STOP or len(w) < 4:
            continue
        t = safedata.single_token(m.tokenizer, w) or safedata.single_token(m.tokenizer, w.capitalize())
        if t is not None:
            return int(t), w
    return None, None
# the cache stores the image path, not the item's own harmful phrase, so recover it from the loader
phrase = {}
for v in a.variants:
    for rec_ in safedata.mm_safetybench(None, v):
        if rec_.get("image_path"):
            phrase[rec_["image_path"]] = rec_.get("text_only") or rec_.get("question") or ""
TOKW = [content_token(phrase.get(it["path"], "")) for it in its]
has_content = np.array([t is not None for t, _ in TOKW])
say(f"content word found for {has_content.mean():.2f} of items, e.g. " + ", ".join(w for _, w in TOKW[:8] if w))
if has_content.sum() < 20:
    say("too few content words; the content axis is dropped")

HARM = axes.token_set(m.tokenizer, axes.HARM_WORDS); BEN = axes.token_set(m.tokenizer, axes.BENIGN_WORDS)
SR = axes.token_set(m.tokenizer, axes.REFUSE_STARTERS); SC = axes.token_set(m.tokenizer, axes.COMPLY_STARTERS)
first_tok = np.array([int(m.tokenizer(it["reply"], add_special_tokens=False).input_ids[0]) if it["reply"] else -1 for it in its])


def unembed(H):
    with torch.no_grad():
        out = []
        for i in range(0, len(H), 64):
            lg = m.unembed(torch.as_tensor(H[i: i + 64]).to(m.device).to(torch.bfloat16)).float()
            out.append(torch.log_softmax(lg, -1).cpu())
        return torch.cat(out)


def margins(lp):
    return dict(harm=(torch.logsumexp(lp[:, HARM], -1) - torch.logsumexp(lp[:, BEN], -1)).numpy(),
                opener=(torch.logsumexp(lp[:, SR], -1) - torch.logsumexp(lp[:, SC], -1)).numpy())


rows, EX = [], {l: {} for l in ALLL}
ranks = {"reader": {}, "logit lens": {}}
t0 = time.time()
for l in ALLL:
    Hhat = RD.to_final(ANS[:, l], l)
    for tname, H in (("reader", Hhat), ("logit lens", ANS[:, l])):
        lp = unembed(H)
        mg = margins(lp)
        ref = H @ r_ref
        cont = np.array([float(lp[i, t]) if t is not None else np.nan for i, (t, _) in enumerate(TOKW)])
        order = lp.argsort(-1, descending=True)
        rk = np.array([int((order[i] == first_tok[i]).float().argmax()) if first_tok[i] >= 0 else -1 for i in range(len(its))])
        ranks[tname][l] = rk
        def au(s, msk=None):
            yy = y_h[msk] if msk is not None else y_h
            return float(roc_auc_score(yy, s)) if len(np.unique(yy)) == 2 and len(s) >= 20 else float("nan")
        rows.append(dict(layer=l, frac=frac[l], transport=tname,
                         auroc=dict(content=au(cont[has_content], has_content), harm=au(mg["harm"]), refusal=au(ref), opener=au(mg["opener"])),
                         rank_median=float(np.median(rk[rk >= 0])), top1=float(np.mean(rk[rk >= 0] == 0)),
                         top5=float(np.mean(rk[rk >= 0] < 5)), top20=float(np.mean(rk[rk >= 0] < 20))))
        if tname == "reader":
            EX[l] = {i: [m.tokenizer.decode([int(t)]) for t in order[i, :6]] for i in range(min(a.n_examples, len(its)))}
        else:
            for i in range(min(a.n_examples, len(its))):
                EX[l].setdefault("logit_" + str(i), [m.tokenizer.decode([int(t)]) for t in order[i, :6]])
say(f"read {len(ALLL)} depths x 2 transports in {(time.time() - t0) / 60:.1f} min")

AX = [k for k in ("content", "harm", "refusal", "opener") if k != "content" or has_content.sum() >= 20]
def curve(t, k): return np.array([x["auroc"][k] for x in rows if x["transport"] == t])
xs = np.array([frac[l] for l in ALLL])

def onset(c, thr=0.7):
    c = np.asarray(c, float)
    ok = np.where(np.nan_to_num(c, nan=0.0) >= thr)[0]
    return float(xs[ok[0]]) if len(ok) else None

say(f"\nAUROC separating harmful from benign, read through the reader")
say(f"{'depth':>6} " + " ".join(f"{k:>9}" for k in AX) + "   |  the model's own first token: " + " ".join(f"{k:>7}" for k in ("top1", "top5", "top20", "median rank")))
for i, l in enumerate(ALLL):
    rr = [x for x in rows if x["transport"] == "reader" and x["layer"] == l][0]
    say(f"{frac[l]:>6.2f} " + " ".join(f"{rr['auroc'][k]:>9.3f}" for k in AX) +
        f"   | {'':>29} {rr['top1']:>7.2f} {rr['top5']:>7.2f} {rr['top20']:>7.2f} {rr['rank_median']:>11.0f}")
say(f"\nthe same through the identity, which is the logit lens")
for i, l in enumerate(ALLL):
    rr = [x for x in rows if x["transport"] == "logit lens" and x["layer"] == l][0]
    say(f"{frac[l]:>6.2f} " + " ".join(f"{rr['auroc'][k]:>9.3f}" for k in AX) +
        f"   | {'':>29} {rr['top1']:>7.2f} {rr['top5']:>7.2f} {rr['top20']:>7.2f} {rr['rank_median']:>11.0f}")

ons = {t: {k: onset(curve(t, k)) for k in AX} for t in ("reader", "logit lens")}
say(f"\nonset depth (first depth with AUROC >= 0.7)")
for t in ons:
    say(f"  {t:<11} " + "  ".join(f"{k} {('%.2f' % v) if v is not None else 'never'}" for k, v in ons[t].items()))

# bootstrap the onset order over items, from the per-layer scalars rather than the full
# log-probability matrices: one number per item per layer per axis is all the onset needs
rng = np.random.default_rng(0)
S_ax = {k: {} for k in AX}
for l in ALLL:
    H = RD.to_final(ANS[:, l], l)
    lp = unembed(H); mg = margins(lp)
    for k in AX:
        if k == "refusal": S_ax[k][l] = H @ r_ref
        elif k == "content": S_ax[k][l] = np.array([float(lp[i, t]) if t is not None else np.nan for i, (t, _) in enumerate(TOKW)])
        else: S_ax[k][l] = mg[k]
    del lp
boot = {k: [] for k in AX}
for _ in range(a.boot):
    idx = rng.integers(0, len(its), len(its))
    if len(np.unique(y_h[idx])) < 2:
        continue
    for k in AX:
        c = []
        for l in ALLL:
            sv = S_ax[k][l][idx]; yy = y_h[idx]
            msk = ~np.isnan(sv)
            c.append(float(roc_auc_score(yy[msk], sv[msk])) if msk.sum() >= 20 and len(np.unique(yy[msk])) == 2 else np.nan)
        o = onset(np.array(c))
        boot[k].append(o if o is not None else 1.2)
say(f"\nonset with a bootstrap over items (median [10th, 90th])")
for k in AX:
    v = np.array(boot[k])
    say(f"  {k:<9} {np.median(v):.2f}  [{np.percentile(v, 10):.2f}, {np.percentile(v, 90):.2f}]")

# ---- per item: the depth at which each score first leaves the benign range, and a paired test
per_item = {k: {} for k in AX}
for k in AX:
    for l in ALLL:
        sv = S_ax[k][l]
        thr = np.nanpercentile(sv[~y_h], 95)
        per_item[k][l] = (sv > thr)
onsets = {k: [] for k in AX}
harm_idx = np.where(y_h)[0]
for i in harm_idx:
    for k in AX:
        d_ = [frac[l] for l in ALLL if per_item[k][l][i]]
        onsets[k].append(d_[0] if d_ else np.nan)
say(f"\nper item, the depth at which the score first leaves the benign range (harmful items only, n={len(harm_idx)})")
say(f"{'score':<10} {'median':>8} {'quartiles':>18} {'never leaves it':>16}")
for k in AX:
    v = np.array(onsets[k], float); ok = ~np.isnan(v)
    say(f"{k:<10} {np.median(v[ok]) if ok.any() else float('nan'):>8.2f} "
        f"{('[%.2f, %.2f]' % (np.percentile(v[ok], 25), np.percentile(v[ok], 75))) if ok.any() else '':>18} {1 - ok.mean():>16.2f}")
pairs = []
if "harm" in AX and "refusal" in AX:
    a_ = np.array(onsets["harm"], float); b_ = np.array(onsets["refusal"], float)
    ok = ~np.isnan(a_) & ~np.isnan(b_)
    if ok.sum() >= 10:
        earlier = int((a_[ok] < b_[ok]).sum()); later = int((a_[ok] > b_[ok]).sum()); tied = int((a_[ok] == b_[ok]).sum())
        from scipy.stats import binomtest
        pv = float(binomtest(earlier, earlier + later, 0.5).pvalue) if earlier + later else float("nan")
        say(f"harm before refusal on {earlier} items, after on {later}, tied on {tied}; sign test p = {pv:.4f}")
        pairs = dict(earlier=earlier, later=later, tied=tied, p=pv, n=int(ok.sum()))

# ---- figures
fig, axs = plt.subplots(1, 2, figsize=(7.8, 2.9), sharey=True)
for ax, t in zip(axs, ("reader", "logit lens")):
    for k, c in zip(AX, plot.SERIES):
        y = curve(t, k)
        ax.plot(xs, y, color=c, lw=1.4)
        plot.end_label(ax, xs[-1], y[-1], k, c)
    plot.chance_line(ax, 0.5)
    ax.axhline(0.7, color=plot.INK2, lw=0.6, ls=":")
    ax.set_title(t, loc="left", fontsize=9.5, color=plot.INK2); ax.set_xlabel("depth"); plot.despine(ax)
axs[0].set_ylabel("AUROC, harmful vs benign"); axs[0].set_ylim(0.35, 1.02)
fig.tight_layout(); plot.save(fig, f"{a.figs}/{name}_cascade.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.4, 2.9))
for t, c in (("reader", plot.SERIES[0]), ("logit lens", plot.SERIES[1])):
    for k, ls in (("top1", "-"), ("top5", "--")):
        y = [x[k] for x in rows if x["transport"] == t]
        ax.plot(xs, y, color=c, lw=1.4, ls=ls)
        plot.end_label(ax, xs[-1], y[-1], f"{t}, {k}", c)
ax.set_xlabel("depth"); ax.set_ylabel("the model's own first token is ranked here"); plot.despine(ax)
fig.tight_layout(); plot.save(fig, f"{a.figs}/{name}_firsttoken.png"); plt.close(fig)

json.dump(dict(repo=a.repo, n=len(its), variants=a.variants, rows=rows, onset=ons,
               per_item_onset={k: [None if np.isnan(x) else float(x) for x in onsets[k]] for k in AX}, harm_before_refusal=pairs,
               onset_bootstrap={k: dict(median=float(np.median(boot[k])), p10=float(np.percentile(boot[k], 10)), p90=float(np.percentile(boot[k], 90))) for k in AX},
               examples={str(frac[l]): EX[l] for l in ALLL},
               example_items=[dict(harmful=bool(its[i]["harmful"]), scenario=its[i].get("scenario"), reply=its[i]["reply"][:120]) for i in range(min(a.n_examples, len(its)))]),
          open(f"{a.out}/{name}_cascade.json", "w"), indent=1, default=float)
say(f"\nwrote {a.out}/{name}_cascade.json  ({(time.time() - T0) / 60:.1f} min)")
