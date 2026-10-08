"""Same pixels, same words, different order: the one experiment the theorem
forbids the other two lenses from entering.

Image tokens precede text tokens under a causal mask, so an image token's
activation is a function of the pixels alone.  Every lens whose map is
``g(M h)`` with ``M`` fixed in advance -- the logit lens, where ``M = I``, and
the J-lens, where ``M = J_l`` is a corpus average -- therefore hands back the
*same bytes* for every question anyone could ask about that photograph.  Not a
similar map: the same one.  This run measures that, and measures what our
reading does instead, on the dataset built to make the point unavoidable.

SugarCrepe is Winoground's ungated relative, over COCO val2017 images already on
disk.  Each item is one photograph and two captions: the true COCO caption and a
minimally corrupted one.  In ``swap_obj`` and ``swap_att`` the two captions are
anagrams -- the identical word multiset in a different order -- so the two
prompts differ in nothing a bag of words could see.  The model is asked the same
question twice,

    Does the caption '<c>' describe this image? Answer yes or no.

and all three lenses are read at the image tokens against the answer word.

Two contrasts are run, not one, because "the map moved" is only interesting if
the movement is about the picture rather than about the characters in the
prompt:

    caption   the true caption against the false one -- same question, two
              meanings.  This is the measurement.
    wording   the true caption under a reworded but equivalent question -- two
              character strings, one meaning.  This is the control: a lens that
              moves as far here is responding to tokens, not to what is being
              asked about the photograph.

The deviation ``h - hbar`` is shared between every condition, since the image is
the same; the entire movement in our reading therefore comes from the transport,
which is the only part of the three lenses that the prompt can reach.

The model's own accuracy on the pairs is reported first, since a model at chance
would make every other number here a statement about noise.
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
from jvlm import cocodata, corpus, ordering, readout, stats as st, sugar, vmap

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--jlens", default=None)
ap.add_argument("--n", type=int, default=300, help="caption pairs")
ap.add_argument("--subsets", default=",".join(sugar.SUBSETS))
ap.add_argument("--mean-images", type=int, default=60)
ap.add_argument("--n-layers", type=int, default=6)
ap.add_argument("--pool", type=int, default=2, help="token block side")
ap.add_argument("--ig-steps", type=int, default=4)
ap.add_argument("--target", default="auto", choices=["auto", "yes", "margin"])
ap.add_argument("--no-alt", action="store_true",
                help="skip the reworded-question control")
ap.add_argument("--complete", type=int, default=8,
                help="prompts on which to close the completeness budget")
ap.add_argument("--max-cards", type=int, default=24)
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/ordering")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]
SUBS = [s for s in a.subsets.split(",") if s]
t0 = time.time()

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lo, hi = max(m.n_deepstack, 1), m.n_layers - 2
LAYERS = sorted({int(round(lo + (hi - lo) * f))
                 for f in np.linspace(0.15, 0.90, a.n_layers)})
J = None
if a.jlens and os.path.exists(a.jlens):
    J = (jvlm.lens_from_checkpoint(a.jlens) if a.jlens.endswith(".ckpt")
         else __import__("jlens").lens.JacobianLens.load(a.jlens))

    # Read where the lens exists.  A fitted Jacobian lens only has a transport
    # at the layers it was fitted at; chosen from the model's geometry instead,
    # the depths miss it and the arm is dropped in silence.
    LAYERS, _ = readout.snap_to_lens(LAYERS, J)
print(m, f"\nlayers {LAYERS}   J-lens {'yes' if J is not None else 'no'}",
      flush=True)

ASK = "Does the caption '{c}' describe this image? Answer yes or no."
# The control must be matched to the measurement or it proves nothing.  The
# caption edit SugarCrepe makes is one word substituted or two words
# transposed; so the control changes exactly one word of the question, at the
# same place in the string, for a synonym that leaves the meaning alone.  A
# reading that moves as far for this as for the caption edit is responding to
# characters rather than to what is being asked about the photograph.
ALT = "Does the caption '{c}' describe this photo? Answer yes or no."


def prep(template, caption, path):
    return m.chat(template.format(c=caption),
                  [corpus.load_image(path, square=a.square)])


# ------------------------------------------------------------------ the data
ITEMS = sugar.pairs(SUBS, n=a.n, seed=5)
print(f"{len(ITEMS)} caption pairs over {len({r['image_id'] for r in ITEMS})} "
      f"photographs; {sum(r['anagram'] for r in ITEMS)} are anagrams",
      flush=True)
print("  " + "  ".join(f"{s}:{sum(r['split'] == s for r in ITEMS)}"
                       for s in SUBS), flush=True)

# ------------------------------------------------- the two words we read for
def _single(forms):
    out = {}
    for f in forms:
        e = m.tokenizer.encode(f, add_special_tokens=False)
        if len(e) == 1:
            out[f] = e[0]
    return out


YID = _single(["Yes", "yes", " Yes", " yes", "YES"])
NID = _single(["No", "no", " No", " no", "NO"])
if not YID or not NID:
    raise SystemExit("neither yes nor no is a single token for this tokenizer")

# ---------------------------------------------------- the per-position baseline
#
# The baseline is a per-position mean over photographs, computed with a generic
# caption.  The theorem says that is the same as computing it with any other
# prompt -- an image token's activation does not depend on the text -- and the
# run checks that claim directly rather than assuming it.
pool = corpus.coco_images(a.mean_images + 40, seed=17)
b0 = prep(ASK, "a photograph", str(pool[0]))
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
C = max(2, G // a.pool)
acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
nrm = {l: torch.zeros(P, dtype=torch.float64) for l in LAYERS}
nm = 0
for p in pool:
    if nm >= a.mean_images:
        break
    b = prep(ASK, "a photograph", str(p))
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
NORM0 = (nrm[LAYERS[0]] / max(nm, 1)).float()
REG = NORM0 > NORM0.mean() + 3 * NORM0.std()
REGIONS, KEEP = ordering.region_blocks(G, C, REG)
R = len(REGIONS)
SIZE = torch.tensor([len(r) for r in REGIONS], dtype=torch.float32)
print(f"{P} tokens on {G}x{G}; {R} regions; {int(REG.sum())} register tokens "
      f"dropped; baseline from {nm} photographs ({time.time()-t0:.0f}s)",
      flush=True)


def rmean(per_patch):
    """logit-lens style maps are region MEANS; ours are region SUMS."""
    return ordering.region_sum(per_patch, REGIONS) / SIZE


# ------------------------------------------ which surface form the model uses
sy = {f: 0.0 for f in YID}
sn = {f: 0.0 for f in NID}
topw = __import__("collections").Counter()
for r in ITEMS[: min(16, len(ITEMS))]:
    b = prep(ASK, r["pos"], r["path"])
    lg = m.unembed(readout.residuals(m, b, [m.n_layers - 1])[m.n_layers - 1][-1:]
                   )[0].float().cpu()
    for f, i in YID.items():
        sy[f] += float(lg[i])
    for f, i in NID.items():
        sn[f] += float(lg[i])
    topw[m.tokenizer.decode([int(lg.argmax())])] += 1
YF, NF = max(sy, key=sy.get), max(sn, key=sn.get)
Y, N = YID[YF], NID[NF]
VOC = [Y, N]
nprobe = min(16, len(ITEMS))
print("answer position, mean logit over "
      f"{nprobe} true-caption prompts:  "
      + "  ".join(f"{f!r} {v/nprobe:.2f}" for f, v in
                  sorted({**sy, **sn}.items(), key=lambda kv: -kv[1])),
      flush=True)
print(f"  the model's own top token there is {topw.most_common(3)}", flush=True)
print(f"answer words: yes -> {YF!r} (id {Y}), no -> {NF!r} (id {N})", flush=True)

# the completeness probe reads a real vocabulary, so that "the readings sum to
# the intervention" is a claim with a correlation attached and not two numbers
_cats = [c for c in cocodata.SINGLE_WORD
         if cocodata.concept_token_ids(m.tokenizer, c)]
_cv = np.random.default_rng(4).choice(_cats, size=min(12, len(_cats)),
                                      replace=False)
CVOC = sorted(set(VOC) | {cocodata.primary_token_id(m.tokenizer, w)
                          for w in _cv})
# the path is swept, but only as far as the card allows: each extra step is
# another row of a replicated batch carrying a full graph
STEPS = sorted({1, 2, a.ig_steps, min(2 * a.ig_steps, 8)})

LENSES = ["logit lens", "VJ-Lens", "VJ-Lens (point)"]
if J is not None:
    LENSES.insert(1, "J-lens")
CONDS = ["pos", "neg"] + ([] if a.no_alt else ["alt"])


def read_prompt(b):
    """Everything one prompt yields: logits, all three lenses, the path probe."""
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        return None
    A = readout.residuals(m, b, LAYERS + [m.n_layers - 1])
    lg = m.unembed(A[m.n_layers - 1][-1:].to(m.device))[0].float().cpu()
    mm = {n: {} for n in LENSES}
    for l in LAYERS:
        h = A[l][ip].float().cpu()
        mm["logit lens"][l] = rmean(
            m.unembed(h.to(m.device)).float().cpu()[:, VOC].T)
        if J is not None and l in J.jacobians:
            jl, _ = readout.lens_logits(m, J, b, layers=[l],
                                        positions=ip.tolist(), cached=A)
            mm["J-lens"][l] = rmean(jl[l].float().cpu()[:, VOC].T)
    DO = vmap.do_map(m, b, VOC, LAYERS, ip.tolist(), means=MEANS)
    IG, diag = ordering.ig_diag(m, b, VOC, LAYERS, ip.tolist(), MEANS,
                                steps=a.ig_steps)
    for l in LAYERS:
        mm["VJ-Lens (point)"][l] = ordering.region_sum(DO[l], REGIONS)
        mm["VJ-Lens"][l] = ordering.region_sum(IG[l], REGIONS)
    return dict(b=b, ip=ip, A=A, lg=lg, maps=mm, ig=IG, do=DO, diag=diag,
                ntok=int(b["input_ids"].shape[1]))


def target_map(mm, which):
    """[2, R] over (yes, no) -> the one map the movement is measured on."""
    return mm[0] if which == "yes" else mm[0] - mm[1]


PAIRS = [("caption", "pos", "neg")] + \
        ([] if a.no_alt else [("wording", "pos", "alt")])

rows, beh, cards, comp = [], [], [], []
skipped = 0
for k, r in enumerate(ITEMS):
    got = {}
    for side in CONDS:
        tpl = ALT if side == "alt" else ASK
        cap = r["neg"] if side == "neg" else r["pos"]
        g = read_prompt(prep(tpl, cap, r["path"]))
        if g is None:
            got = {}
            break
        got[side] = g
    if len(got) != len(CONDS):
        skipped += 1
        continue

    # ---- what the model actually did
    def mar(s):
        return float(got[s]["lg"][Y]) - float(got[s]["lg"][N])

    mp, mn = mar("pos"), mar("neg")
    rec = dict(i=k, split=r["split"], key=r["key"], image_id=r["image_id"],
               anagram=int(r["anagram"]), changed=r["changed"] or "",
               yes_pos=float(got["pos"]["lg"][Y]),
               no_pos=float(got["pos"]["lg"][N]),
               yes_neg=float(got["neg"]["lg"][Y]),
               no_neg=float(got["neg"]["lg"][N]),
               margin_pos=mp, margin_neg=mn, dmargin=mp - mn,
               p_yes_pos=float(torch.softmax(got["pos"]["lg"], -1)[Y]),
               p_yes_neg=float(torch.softmax(got["neg"]["lg"], -1)[Y]),
               correct_rank=int(mp > mn),
               correct_abs=int(mp > 0 and mn < 0),
               said_yes_pos=int(mp > 0), said_yes_neg=int(mn > 0),
               ntok_pos=got["pos"]["ntok"], ntok_neg=got["neg"]["ntok"],
               dlen=abs(got["pos"]["ntok"] - got["neg"]["ntok"]))
    if "alt" in got:
        rec["margin_alt"] = mar("alt")
        rec["dlen_alt"] = abs(got["pos"]["ntok"] - got["alt"]["ntok"])
    # ---- the premise: are the image activations really prompt-independent?
    for l in LAYERS:
        hp = got["pos"]["A"][l][got["pos"]["ip"]]
        d = (hp - got["neg"]["A"][l][got["neg"]["ip"]]).abs()
        rec[f"dh_max_{l}"] = float(d.max())
        rec[f"dh_rel_{l}"] = float(d.max() / hp.abs().max())
        # ---- how much gradient survives to the end of the path
        for j, w in enumerate(("yes", "no")):
            rec[f"sat_{w}_{l}"] = float(got["pos"]["diag"][l]["sat"][j])
    beh.append(rec)

    # ---- how far each map moved
    box = None
    if r["changed"]:
        bx = cocodata.object_boxes(r["image_id"], r["changed"])
        tiles = sorted({t for b_ in bx for t in cocodata.box_tokens(b_, C)})
        if tiles:
            box = [j for j in range(R) if KEEP[j] in tiles]
    for contrast, s1, s2 in PAIRS:
        for which in ("yes", "margin"):
            for name in LENSES:
                for l in LAYERS:
                    if l not in got[s1]["maps"][name]:
                        continue
                    A1 = target_map(got[s1]["maps"][name][l], which)
                    A2 = target_map(got[s2]["maps"][name][l], which)
                    mv = ordering.movement(A1, A2)
                    row = dict(i=k, split=r["split"], image_id=r["image_id"],
                               anagram=int(r["anagram"]), lens=name, cond=name,
                               contrast=contrast, layer=l,
                               depth=round(l / (m.n_layers - 1), 3),
                               target=which,
                               key=f"{k}_{l}_{which}_{contrast}",
                               correct_rank=rec["correct_rank"],
                               correct_abs=rec["correct_abs"],
                               dmargin=rec["dmargin"],
                               adm=abs(rec["dmargin"]), dlen=rec["dlen"],
                               **mv)
                    if box and contrast == "caption":
                        d = (A1 - A2).abs()
                        row["in_box"] = float(
                            d[box].sum() / d.sum().clamp_min(1e-12))
                        row["box_area"] = len(box) / R
                    rows.append(row)

    if len(cards) < a.max_cards:
        cards.append(dict(
            image_id=r["image_id"], path=r["path"], split=r["split"],
            anagram=r["anagram"], pos=r["pos"], neg=r["neg"],
            coarse=C, keep=KEEP, correct_rank=rec["correct_rank"],
            margin_pos=mp, margin_neg=mn,
            maps={n: {str(l): {f"{s}_{w}": [round(float(v), 5) for v in
                                            target_map(got[s]["maps"][n][l], w)]
                               for s in CONDS for w in ("yes", "margin")}
                      for l in LAYERS if l in got["pos"]["maps"][n]}
                  for n in LENSES}))

    # ---- does the budget still close when the target is an answer word?
    if len(comp) < a.complete:
        # Completeness on two words is a pair of numbers with no correlation to
        # report, so the probe carries its own wider vocabulary; and since the
        # target here is a judgment rather than an object name, the path length
        # is swept rather than assumed.
        l = LAYERS[len(LAYERS) // 2]
        b_, ip_ = got["pos"]["b"], got["pos"]["ip"].tolist()
        try:
            for st_ in STEPS:
                sc_ = vmap.ig_map(m, b_, CVOC, [l], ip_, means=MEANS,
                                  steps=st_)[l]
                comp.append(dict(i=k, layer=l, lens="VJ-Lens", steps=st_,
                                 **vmap.completeness(m, b_, CVOC, l, ip_,
                                                     MEANS[l], sc_)))
            comp.append(dict(i=k, layer=l, lens="VJ-Lens (point)", steps=0,
                             **vmap.completeness(
                                 m, b_, CVOC, l, ip_, MEANS[l],
                                 vmap.do_map(m, b_, CVOC, [l], ip_,
                                             means=MEANS)[l])))
        except torch.OutOfMemoryError:
            # the probe is a diagnostic, not the measurement; a busy card must
            # cost us the diagnostic and not the run
            torch.cuda.empty_cache()
            print(f"  completeness probe skipped at item {k}: out of memory",
                  flush=True)

    if k % 10 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(ITEMS)}  {el:.0f}s  "
              f"eta {el/max(k,1)*(len(ITEMS)-k)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")
        pd.DataFrame(beh).to_parquet(f"{a.out}/{tag}_beh.parquet")

df = pd.DataFrame(rows)
bf = pd.DataFrame(beh)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
bf.to_parquet(f"{a.out}/{tag}_beh.parquet")
json.dump(st.jsonable(cards), open(f"{a.out}/{tag}_cards.json", "w"))

# --------------------------------------------------------------- the target
sat = {w: float(np.median(bf[[f"sat_{w}_{l}" for l in LAYERS]].to_numpy()))
       for w in ("yes", "no")}
# What decides the target is whether the gradient has died at the model's own
# activation -- that is what a first-order reading would miss and what a
# saturated answer does to it.  A high p(yes) is reported beside it but does not
# on its own condemn the logit: the logit is not the probability, and the
# integrated reading never evaluates the gradient only at alpha = 1.
saturated = sat["yes"] < 0.5
TARGET = a.target if a.target != "auto" else ("margin" if saturated else "yes")
print(f"\nSATURATION  median p(yes)={bf.p_yes_pos.median():.3f}   "
      f"end-of-path gradient / path mean: yes {sat['yes']:.3f} "
      f"no {sat['no']:.3f}   ->  target = {TARGET}", flush=True)
alt_t = "margin" if TARGET == "yes" else "yes"
_o = df[(df.target == alt_t) & (df.contrast == "caption")].groupby("lens").agg(
    moved=("max_abs", lambda z: int((z > 0).sum())), of=("max_abs", "count"),
    rel=("rel", "mean"), cos=("cos", "mean"), rho=("rho", "mean"))
print(f"\nthe same movement read against the {alt_t} instead, as a robustness "
      f"check\n", _o.round(4).to_string(), flush=True)

# ------------------------------------------------------------- the behaviour
acc_rank, acc_abs = float(bf.correct_rank.mean()), float(bf.correct_abs.mean())
print(f"\nTHE MODEL ON THESE PAIRS   n={len(bf)}\n"
      f"  prefers the true caption      {acc_rank:.3f}   (chance 0.5)\n"
      f"  yes to true and no to false   {acc_abs:.3f}\n"
      f"  says yes to the true caption  {bf.said_yes_pos.mean():.3f}\n"
      f"  says yes to the false caption {bf.said_yes_neg.mean():.3f}\n"
      f"  |token length difference|     {bf.dlen.mean():.1f}", flush=True)
print("\nby subset\n", bf.groupby("split").agg(
    n=("i", "count"), prefers_true=("correct_rank", "mean"),
    yes_then_no=("correct_abs", "mean"), dlen=("dlen", "mean")
).round(3).to_string(), flush=True)

# --------------------------------------------------- the premise and the zero
print("\nIMAGE ACTIVATIONS UNDER THE TWO PROMPTS  (max |dh| over image tokens, "
      "and relative to max |h|)")
for l in LAYERS:
    print(f"  layer {l:>3}   max {bf[f'dh_max_{l}'].max():.3e}   "
          f"rel {bf[f'dh_rel_{l}'].max():.3e}   "
          f"exactly zero on {int((bf[f'dh_max_{l}'] == 0).sum())}/{len(bf)}")

d = df[(df.target == TARGET) & (df.contrast == "caption")]
mv = d.groupby(["lens", "layer"]).agg(
    n=("i", "count"), max_abs=("max_abs", "max"),
    moved=("max_abs", lambda s: int((s > 0).sum())),
    rel=("rel", "mean"), cos=("cos", "mean"), rho=("rho", "mean"),
    top=("top", "mean")).reset_index()
mv.to_csv(f"{a.out}/{tag}_movement.csv", index=False)
print(f"\nHOW FAR THE MAP MOVES WHEN ONLY THE CAPTION CHANGES  "
      f"(target {TARGET})\n", mv.round(4).to_string(index=False), flush=True)

over = d.groupby("lens").agg(
    moved=("max_abs", lambda s: int((s > 0).sum())), of=("max_abs", "count"),
    max_abs=("max_abs", "max"), rel=("rel", "mean"), cos=("cos", "mean"),
    rho=("rho", "mean"), top=("top", "mean"))
print("\nover all layers\n", over.round(4).to_string(), flush=True)

ctr = pd.DataFrame()
if not a.no_alt:
    ctr = df[df.target == TARGET].groupby(["lens", "contrast"]).agg(
        rel=("rel", "mean"), cos=("cos", "mean"), rho=("rho", "mean"),
        top=("top", "mean")).round(4)
    print("\nTHE MEASUREMENT AGAINST ITS CONTROL\n"
          "  caption: two meanings, one question    "
          "wording: one meaning, two questions\n", ctr.to_string(), flush=True)

# ------------------------------------- does the movement predict correctness?
def boot_auc(sub, col, lab, n=1500, seed=0):
    """Percentile interval for the AUC, resampling whole photographs."""
    sv, sl = sub[col].to_numpy(float), sub[lab].to_numpy()
    codes = pd.factorize(sub.image_id.to_numpy())[0]
    idx = [np.flatnonzero(codes == c) for c in range(codes.max() + 1)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = np.concatenate([idx[i]
                               for i in rng.integers(0, len(idx), len(idx))])
        v = ordering.auc(sv[pick], sl[pick])
        if not np.isnan(v):
            out.append(v)
    return ((float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975)))
            if out else (float("nan"), float("nan")))


# The objection a reader will raise first: the movement is just re-reading the
# model's own answer difference.  So the behavioural difference is scored on the
# same question, giving the number the lens has to beat, and the movement is
# scored again with that difference's rank trend removed.
AUC_BEH = ordering.auc(bf.dmargin.abs().to_numpy(),
                       bf.correct_rank.to_numpy())
print(f"\nthe behavioural reference: |margin(true) - margin(false)| by itself "
      f"ranks the pairs the model prefers correctly at auc {AUC_BEH:.3f}",
      flush=True)


def partial_auc(x, y, lab):
    """AUC of ``x`` for ``lab`` after removing ``y``'s rank-linear trend."""
    rx, ry = sps.rankdata(x), sps.rankdata(y)
    if np.std(ry) < 1e-9:
        return float("nan")
    resid = rx - np.polyval(np.polyfit(ry, rx, 1), ry)
    return ordering.auc(resid, lab)


pred = []
for name in LENSES:
    for l in LAYERS:
        s = d[(d.lens == name) & (d.layer == l)]
        if len(s) < 10:
            continue
        # A lens whose map is bitwise identical under the two prompts still
        # has ``1 - cos`` and ``1 - rho`` wobbling at 1e-7, and ranking that
        # dust produces an AUC as impressive-looking as any other.  The guard
        # is the raw maximum difference, not the derived statistic's variance.
        frozen = bool((s.max_abs == 0).all())
        for col in ("rel", "rho", "cos"):
            if frozen or s[col].std() == 0:
                pred.append(dict(lens=name, layer=l, move=col, n=len(s),
                                 auc=float("nan"), note="map never moved"))
                continue
            lo_, hi_ = boot_auc(s, col, "correct_rank")
            sp = sps.spearmanr(s[col], s.adm)
            sl_ = sps.spearmanr(s[col], s.dlen)
            pred.append(dict(
                lens=name, layer=l, move=col, n=len(s),
                auc=ordering.auc(s[col].to_numpy(), s.correct_rank.to_numpy()),
                auc_lo=lo_, auc_hi=hi_,
                auc_resid=partial_auc(s[col].to_numpy(), s.adm.to_numpy(),
                                      s.correct_rank.to_numpy()),
                rho_adm=float(sp.statistic), p_adm=float(sp.pvalue),
                rho_dlen=float(sl_.statistic),
                auc_len0=ordering.auc(s[s.dlen == 0][col].to_numpy(),
                                      s[s.dlen == 0].correct_rank.to_numpy()),
                n_len0=int((s.dlen == 0).sum())))
pr = pd.DataFrame(pred)
pr.to_csv(f"{a.out}/{tag}_predict.csv", index=False)
print("\nDOES THE MOVEMENT PREDICT THE MODEL GETTING THE PAIR RIGHT?\n"
      "  auc      movement ranks the pairs the model prefers correctly\n"
      "  rho_adm  movement against |margin(true) - margin(false)|\n"
      "  rho_dlen movement against the prompt length difference (a confound)\n"
      "  auc_len0 the same auc on pairs of equal token length\n"
      f"  auc_resid the same auc once the rank trend of |dmargin| is removed; "
      f"the raw behavioural reference is {AUC_BEH:.3f}\n",
      pr.round(4).to_string(index=False), flush=True)

_r = np.random.default_rng(3)
perm = []
for l in LAYERS:
    s = d[(d.lens == "VJ-Lens") & (d.layer == l)]
    if len(s) >= 10 and s["rho"].std() > 0:
        perm += [ordering.auc(s["rho"].to_numpy(),
                              _r.permutation(s.correct_rank.to_numpy()))
                 for _ in range(400)]
null = ((float(np.quantile(perm, 0.025)), float(np.quantile(perm, 0.975)))
        if perm else (float("nan"), float("nan")))
print(f"\nshuffled-label null for the same statistic: "
      f"[{null[0]:.3f}, {null[1]:.3f}]", flush=True)

# ------------------------------------------------------------- where it moved
inbox = {}
if "in_box" in df.columns and d.in_box.notna().any():
    ib = d[d.in_box.notna()]
    g = ib.groupby("lens").agg(n=("i", "count"), in_box=("in_box", "mean"),
                               area=("box_area", "mean"))
    inbox = st.jsonable(g.round(4).to_dict("index"))
    print("\nWHERE THE MOVEMENT LANDS  (share of |change| inside the box of the "
          "edited object; chance is the box's area share)\n",
          g.round(4).to_string(), flush=True)

cdf = pd.DataFrame(comp)
compl = {}
if len(cdf):
    cg = cdf.groupby(["lens", "steps"])[["rel_error", "corr"]].median()
    compl = st.jsonable(cg.round(4).to_dict("index"))
    print(f"\nDOES THE BUDGET STILL CLOSE WHEN THE TARGET IS AN ANSWER WORD?\n"
          f"  layer {LAYERS[len(LAYERS)//2]}, {cdf.i.nunique()} prompts, "
          f"{len(CVOC)} words; steps 0 is the first-order reading.\n"
          f"  rel_error: |sum of readings - true change in the logit| relative "
          f"to the mean |true change|;  corr: across words\n",
          cg.round(3).to_string(), flush=True)

out = dict(
    repo=a.repo, tag=tag, n_pairs=int(len(bf)), skipped=skipped,
    subsets=SUBS, layers=LAYERS, n_layers=m.n_layers, grid=G, coarse=C,
    regions=R, target=TARGET, ig_steps=a.ig_steps, conds=CONDS,
    yes_form=YF, no_form=NF, ask=ASK, alt=(None if a.no_alt else ALT),
    anagram_pairs=int(bf.anagram.sum()),
    saturation=dict(p_yes_median=float(bf.p_yes_pos.median()),
                    end_over_path=sat, chose=TARGET),
    accuracy=dict(prefers_true=acc_rank, yes_then_no=acc_abs,
                  yes_to_true=float(bf.said_yes_pos.mean()),
                  yes_to_false=float(bf.said_yes_neg.mean()),
                  mean_dlen=float(bf.dlen.mean())),
    accuracy_by_split=st.jsonable(
        bf.groupby("split")[["correct_rank", "correct_abs"]].mean()
        .round(4).to_dict("index")),
    dh={f"L{l}": dict(max=float(bf[f"dh_max_{l}"].max()),
                      rel=float(bf[f"dh_rel_{l}"].max()),
                      zero=int((bf[f"dh_max_{l}"] == 0).sum()))
        for l in LAYERS},
    movement=st.jsonable(mv.round(5).to_dict("records")),
    movement_over=st.jsonable(over.round(5).to_dict("index")),
    control=st.jsonable(ctr.reset_index().to_dict("records")) if len(ctr) else {},
    predict=st.jsonable(pr.round(4).to_dict("records")),
    perm_null=dict(lo=null[0], hi=null[1]),
    auc_behaviour=float(AUC_BEH),
    in_box=inbox,
    completeness=dict(n=int(cdf.i.nunique()) if len(cdf) else 0,
                      words=len(CVOC), by=compl),
    minutes=round((time.time() - t0) / 60, 1))
if int(bf.anagram.sum()) >= 10:
    da = d[d.anagram == 1].groupby("lens").agg(
        n=("i", "count"), moved=("max_abs", lambda s: int((s > 0).sum())),
        rel=("rel", "mean"), cos=("cos", "mean"), rho=("rho", "mean"))
    out["anagram_only"] = st.jsonable(da.round(5).to_dict("index"))
    print("\nANAGRAM PAIRS ONLY  (the identical word multiset, reordered)\n",
          da.round(4).to_string(), flush=True)
for name in LENSES:
    s = d[d.lens == name]
    if len(s):
        out[f"ci_rel_{name}"] = st.cluster_bootstrap_ci(s, "rel", "image_id")
json.dump(st.jsonable(out), open(f"{a.out}/{tag}_stats.json", "w"), indent=2,
          default=float)
print("done", flush=True)
