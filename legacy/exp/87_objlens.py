# SPDX-License-Identifier: Apache-2.0
r"""The object lens: the same conservation law, over segments a person can name.

Four results, in the order they must hold.

1. COMPLETENESS OVER THE OBJECT PARTITION.  COCO instance masks are turned
   into an exclusive, exhaustive partition of the image token cells (each
   object one segment, the remainder one "scene" segment), and the summed
   readings are checked against the measured effect
   :math:`F(h) - F(\bar h)`.  The law only asks that the regions partition
   the positions, so the numbers must match to the path discretisation, and
   this run measures that they do rather than asserting it.

2. OBJECT-LEVEL GRADING AGAINST CAUSAL TRUTH.  Cover each segment two ways --
   replace its activations with the corpus mean, and grey out its own pixels
   and re-encode the photograph -- and measure the true drop in the logit of
   the word the model was about to say.  Every lens is then scored on whether
   it ranks the *objects* as the interventions do.  The content lenses (logit
   lens, J-lens) have no units and no additivity, so they are aggregated over
   a segment by the mean, and the sum arm is reported beside it; whichever
   flatters them is the comparison that stands.

3. THE QUESTION FLIP.  The same photograph, two presence questions naming two
   different objects in it, the reading taken of the model's own yes/no
   margin.  Image tokens precede the text under the causal mask, so the fixed
   lenses' readings at image positions are byte-identical across the two
   questions (verified numerically, not argued); the object lens
   differentiates the answer and its shares can move to the object the
   question asks about.  Tracking rate against the 0.5 chance floor.

4. THE CARDS.  Photographs with the objects tinted by their share of the
   model's own answer to an open question, the share printed on the object,
   and the same photograph under a second question whose shares reorder.

Everything reads through :mod:`jvlm.objlens` (the partition and the truths)
and :mod:`jvlm.vmap` / :mod:`jvlm.objective` (the readings), unmodified.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, "code")
for n in ("httpx", "httpcore", "urllib3", "filelock", "transformers"):
    logging.getLogger(n).setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--jlens", default=None)
ap.add_argument("--n", type=int, default=40, help="grading photographs")
ap.add_argument("--flip-n", type=int, default=30, help="question-flip items")
ap.add_argument("--n-cases", type=int, default=8, help="open-question cards")
ap.add_argument("--mean-images", type=int, default=80)
ap.add_argument("--n-layers", type=int, default=2, help="lens read layers")
ap.add_argument("--ig-steps", type=int, default=8)
ap.add_argument("--ig-chunk", type=int, default=2)
ap.add_argument("--vocab", type=int, default=20,
                help="scoring words per photograph; one backward per word "
                     "per path chunk, so this is the main cost dial")
ap.add_argument("--max-rank", type=int, default=1,
                help="how far down the model's own ranking the target may sit")
ap.add_argument("--max-obj", type=int, default=3)
ap.add_argument("--min-area", type=float, default=0.04)
ap.add_argument("--point", action="store_true",
                help="add the single-point (non-integrated) arm")
ap.add_argument("--sam", type=int, default=0,
                help="also grade this many photographs on SAM's automatic "
                     "masks, with the lens naming the regions itself")
ap.add_argument("--budget-min", type=float, default=1e9,
                help="stop taking new photographs after this many minutes")
ap.add_argument("--square", type=int, default=448)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--device", default="cuda",
                help="cpu validates the full pipeline without the card")
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--out", default="results/objlens")
ap.add_argument("--tag", default=None)
ap.add_argument("--figs", action="store_true",
                help="render figures from saved results; no model is loaded")
ap.add_argument("--figdir", default="figures")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

# ----------------------------------------------------------------- figures
if a.figs:
    from jvlm import objlens

    cards = json.load(open(f"{a.out}/{tag}_cards.json"))
    flips = [c for c in cards if c["kind"] == "flip"]
    opens = [c for c in cards if c["kind"] == "open"]

    def contrast(cd):
        """How far apart the two questions pull the two objects' shares.
        Cards whose shares are degenerate (tiny or badly-closed total
        effect, shares far outside [0, 1]) are sent to the bottom."""
        q1, q2 = cd["questions"]
        i, j = cd["obj_idx"]
        for q in (q1, q2):
            if abs(q["truth"]) < 0.5:
                return -9.0
            if abs(q["claim"] - q["truth"]) > 0.35 * abs(q["truth"]):
                return -9.0
            if any(s < -0.25 or s > 1.25 for s in q["shares"]):
                return -9.0
        return min(q1["shares"][i] - q1["shares"][j],
                   q2["shares"][j] - q2["shares"][i])

    for pool, name in ((opens, "f87_flip"), (flips, "f87_flip_yesno")):
        ranked = sorted(pool, key=contrast, reverse=True)
        for r, cd in enumerate(ranked[:4]):
            objlens.fig_flip(cd, f"{a.figdir}/{name}{'' if r == 0 else r}.png")
    if opens:
        objlens.fig_examples(
            sorted(opens, key=contrast, reverse=True),
            f"{a.figdir}/f87_examples.png")
    sams = [c for c in cards if c["kind"] == "sam"]
    if sams:
        objlens.fig_examples(sams, f"{a.figdir}/f87_sam.png")
    gp = f"{a.out}/{tag}_grade.parquet"
    if os.path.exists(gp):
        objlens.fig_grade(pd.read_parquet(gp), f"{a.figdir}/f87_grade.png")
    cp = f"{a.out}/{tag}_complete.parquet"
    if os.path.exists(cp):
        objlens.fig_complete(pd.read_parquet(cp),
                             f"{a.figdir}/f87_complete.png")
    fp = f"{a.out}/{tag}_flip.parquet"
    if os.path.exists(fp):
        objlens.fig_track(pd.read_parquet(fp), f"{a.figdir}/f87_track.png")
    print("figures rendered", flush=True)
    raise SystemExit(0)

# ------------------------------------------------------------------- setup
import jvlm                                                   # noqa: E402
from jlens.hooks import ActivationRecorder                    # noqa: E402
from jvlm import (budget, cocodata, corpus, objective,        # noqa: E402
                  objlens, readout, stats as st, tasks, vmap)
from scipy import stats as sps                                # noqa: E402

t0 = time.time()
m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels, device=a.device,
                  dtype=getattr(torch, a.dtype))
sc = tasks.Scorer(m)
LMID = int(round(0.5 * (m.n_layers - 1)))   # fixed card layer, set in advance
LAYERS = sorted({int(round(f * (m.n_layers - 1)))
                 for f in np.linspace(0.20, 0.85, a.n_layers)} | {LMID})
J = None
if a.jlens:
    J = __import__("jlens").lens.JacobianLens.load(a.jlens)
CATS = [c for c in cocodata.SINGLE_WORD
        if cocodata.concept_token_ids(m.tokenizer, c)]
CAT_IDS = {c: cocodata.concept_token_ids(m.tokenizer, c) for c in CATS}
print(m, f"\nlayers {LAYERS} (cards at {LMID})", flush=True)

PROMPT = "Describe this image."


def prep(img):
    return m.chat(PROMPT, [img])


def load(path):
    return corpus.load_image(path, square=a.square)


# corpus-mean baseline, per position, from held-out photographs
pool = cocodata.concept_items(6000, categories=CATS, min_area=0.04, seed=7)
b0 = prep(load(pool[0]["path"]))
IPOS = readout.image_positions(m, b0)
P = len(IPOS)
G = int(round(P ** 0.5))
assert G * G == P, f"{P} image tokens is not a square grid"
acc = {l: torch.zeros(P, m.d_model, dtype=torch.float64) for l in LAYERS}
nm = 0
for it in pool[: a.mean_images]:
    b = prep(load(it["path"]))
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue
    A = readout.residuals(m, b, LAYERS)
    for l in LAYERS:
        acc[l] += A[l][ip].float().cpu().double()
    nm += 1
MEANS = {l: (acc[l] / max(nm, 1)).float() for l in LAYERS}
print(f"means from {nm}; {P} tokens on {G}x{G} ({time.time()-t0:.0f}s)",
      flush=True)


def build_segments(image_id):
    masks = objlens.category_masks(image_id, res=a.square, cats=CATS)
    if len(masks) < 2:
        return None
    return objlens.partition(masks, G, max_obj=a.max_obj)


def zmap(seg_scores: torch.Tensor) -> torch.Tensor:
    """z-score down the vocabulary axis, per segment."""
    mu = seg_scores.mean(0, keepdim=True)
    sd = seg_scores.std(0, keepdim=True).clamp_min(1e-6)
    return (seg_scores - mu) / sd


items = cocodata.concept_items(a.n * 20, categories=CATS,
                               min_area=a.min_area, seed=808,
                               tokenizer=m.tokenizer)
grade_rows, comp_rows, flip_rows, sam_rows, cards = [], [], [], [], []
kept = flips = opens = sams = 0
for k, it in enumerate(items):
    if kept >= a.n or (time.time() - t0) / 60 > a.budget_min:
        break
    c = it["concept"]
    segs = build_segments(it["image_id"])
    if segs is None:
        continue
    obj_names = [sg.name for sg in segs if sg.is_object]
    if c not in obj_names or len(obj_names) < 2:
        continue
    img = load(it["path"])
    b = prep(img)
    ip = readout.image_positions(m, b)
    if len(ip) != P:
        continue

    # scoring vocabulary: the photograph's objects plus a fixed random spread
    rngv = np.random.default_rng(900 + k)
    others = [x for x in CATS if x not in obj_names]
    words = obj_names + list(rngv.choice(
        others, size=max(0, min(a.vocab - len(obj_names), len(others))),
        replace=False))
    VOC = sorted({t for w in words for t in CAT_IDS[w]})
    VPOS = {t: i for i, t in enumerate(VOC)}
    WORD_V = {}
    for w in words:
        for t in CAT_IDS[w]:
            WORD_V.setdefault(VPOS[t], w)
    col = [VPOS[t] for t in CAT_IDS[c]]

    A = readout.residuals(m, b, LAYERS + [m.n_layers - 1])
    lg0 = m.unembed(A[m.n_layers - 1][-1:].to(m.device))[0][VOC].float().cpu()
    if int((lg0 > lg0[col].max()).sum()) > a.max_rank:
        continue
    kept += 1
    S = len(segs)
    n_obj = len(obj_names)
    oidx = [j for j, sg in enumerate(segs) if sg.is_object]

    # ---- ground truth 2: grey the segment's own pixels, re-encode (layer
    # free, so computed once per photograph)
    ptruth = objlens.pixel_truth(m, img, PROMPT, VOC, segs)      # [S, k]
    pt = ptruth[:, col].max(-1).values.numpy()

    for l in LAYERS:
        # ---- ground truth 1: cover each segment's activations
        ctruth = objlens.cover_truth(m, b, VOC, l, segs, MEANS[l], ip)
        ct = ctruth[:, col].max(-1).values.numpy()

        # ---- the lenses, per patch then per segment
        per = budget.ig_map_chunked(m, b, VOC, l, ip.tolist(), MEANS[l],
                                    steps=a.ig_steps, chunk=a.ig_chunk)
        maps = {"object lens": objlens.seg_sum(per, segs)}
        if a.point:
            pt1 = vmap.do_map(m, b, VOC, [l], ip.tolist(),
                              means={l: MEANS[l]})[l]
            maps["object lens (point)"] = objlens.seg_sum(pt1, segs)
        h = A[l][ip].float().cpu()
        lgm = m.unembed(h.to(m.device)).float().cpu()[:, VOC].T
        maps["logit lens (mean)"] = objlens.seg_mean(lgm, segs)
        maps["logit lens (sum)"] = objlens.seg_sum(lgm, segs)
        if J is not None and l in J.jacobians:
            jl, _ = readout.lens_logits(m, J, b, layers=[l],
                                        positions=ip.tolist(), cached=A)
            jm = jl[l].float().cpu()[:, VOC].T
            maps["J-lens (mean)"] = objlens.seg_mean(jm, segs)
            maps["J-lens (sum)"] = objlens.seg_sum(jm, segs)

        # ---- 1. completeness over the object partition
        cc = vmap.completeness(m, b, VOC, l, ip.tolist(), MEANS[l], per)
        seg_total = float(maps["object lens"].sum(-1).double().sum())
        patch_total = float(per.double().sum())
        comp_rows.append(dict(
            i=k, concept=c, layer=l, n_seg=S,
            corr=cc["corr"], rel_error=cc["rel_error"],
            max_abs_err=float(np.abs(np.array(cc["claim"])
                                     - np.array(cc["truth"])).max()),
            scale=float(np.abs(cc["truth"]).mean()),
            partition_gap=abs(seg_total - patch_total),
            truth=cc["truth"], claim=cc["claim"]))

        # ---- 2. grade every lens against both truths, at object level
        for name, mm in maps.items():
            sco = mm[col].max(0).values.float().numpy()          # [S]
            ztop = zmap(mm).argmax(0)
            top_words = [WORD_V.get(int(t), "?") for t in ztop]
            for tname, tru in (("cover", ct), ("pixel", pt)):
                omax_t = oidx[int(np.argmax(tru[oidx]))]
                omax_l = oidx[int(np.argmax(sco[oidx]))]
                pairs = [(x, y) for xi, x in enumerate(oidx)
                         for y in oidx[xi + 1:]]
                pa = np.mean([(sco[x] > sco[y]) == (tru[x] > tru[y])
                              for x, y in pairs]) if pairs else np.nan
                grade_rows.append(dict(
                    i=k, concept=c, lens=name, truth=tname, layer=l,
                    key=f"{k}_{l}_{tname}",
                    depth=round(l / (m.n_layers - 1), 3),
                    spearman=float(sps.spearmanr(sco, tru).statistic)
                    if S > 2 else np.nan,
                    hit_obj=int(omax_l == omax_t), pair_acc=float(pa),
                    chance_hit=1.0 / n_obj, n_obj=n_obj, n_seg=S,
                    scene_share_truth=float(
                        np.abs(tru[-1]) / max(np.abs(tru).sum(), 1e-9)),
                    top_words="|".join(top_words)))

        # ---- 5. the same grading on SAM's automatic masks, lens-named
        if a.sam and sams < a.sam and l == LMID:
            try:
                smasks = objlens.sam_masks(img, device=a.device)
                ssegs = objlens.partition(smasks, G, max_obj=5, max_frac=0.9)
            except Exception as e:            # noqa: BLE001 -- stay honest,
                print(f"[sam] disabled after error: {e!r}", flush=True)
                a.sam = 0
                ssegs = None
            if ssegs is not None and sum(sg.is_object for sg in ssegs) >= 2:
                struth = objlens.cover_truth(m, b, VOC, l, ssegs,
                                             MEANS[l], ip)
                stru = struth[:, col].max(-1).values.numpy()
                ssc = objlens.seg_sum(per, ssegs)
                sscore = ssc[col].max(0).values.float().numpy()
                sidx = [j for j, sg in enumerate(ssegs) if sg.is_object]
                sam_rows.append(dict(
                    i=k, concept=c, layer=l, n_seg=len(ssegs),
                    spearman=float(sps.spearmanr(sscore, stru).statistic),
                    hit_obj=int(sidx[int(np.argmax(sscore[sidx]))]
                                == sidx[int(np.argmax(stru[sidx]))]),
                    chance_hit=1.0 / len(sidx)))
                sams += 1
                if len([cd for cd in cards if cd["kind"] == "sam"]) < 4:
                    zt = zmap(ssc).argmax(0)
                    names = [WORD_V.get(int(t), "?") for t in zt]
                    csegs = objlens.card_segments(ssegs)
                    for cs, nmw in zip(csegs, names):
                        if cs["object"]:
                            cs["name"] = nmw
                    tot = float(np.abs(sscore).sum()) or 1.0
                    cards.append(dict(
                        kind="sam", image_id=it["image_id"], path=it["path"],
                        res=a.square, layer=l, model=tag,
                        obj_idx=sidx, segments=csegs,
                        questions=[dict(
                            question=PROMPT, answer=c,
                            shares=[float(abs(v) / tot) for v in sscore],
                            truth=float(stru.sum()), claim=float(sscore.sum()))]))

    # ---- 3. the question flip, at the fixed layer, on the two biggest objects
    if flips < a.flip_n and n_obj >= 2:
        X, Y = segs[0].name, segs[1].name
        share = {}
        resid_mid = {}
        probs = {}
        for q, tgt in ((tasks.presence_prompt(X), X),
                       (tasks.presence_prompt(Y), Y)):
            bq = m.chat(q, [img])
            ipq = readout.image_positions(m, bq)
            Aq = readout.residuals(m, bq, [LMID, m.n_layers - 1])
            resid_mid[tgt] = Aq[LMID][ipq].float().cpu()
            lgq = m.unembed(Aq[m.n_layers - 1][-1:].to(m.device))[0].float()
            yes = max(sc.yes, key=lambda t: float(lgq[t]))
            no = max(sc.no, key=lambda t: float(lgq[t]))
            probs[tgt] = float(torch.sigmoid(lgq[yes] - lgq[no]))
            obj = objective.token_logit(bq, [yes, no],
                                        weight=[[1.0], [-1.0]],
                                        names=["yes-no"])
            per_q = objective.ig_map(m, obj, [LMID], ipq.tolist(),
                                     {LMID: MEANS[LMID]}, steps=a.ig_steps,
                                     chunk=a.ig_chunk)[LMID]
            comp = objective.completeness(m, obj, LMID, ipq.tolist(),
                                          MEANS[LMID], per_q)
            ss = objlens.seg_sum(per_q, segs)[0]
            den = comp["truth"][0] if abs(comp["truth"][0]) > 1e-4 \
                else float(ss.sum())
            share[tgt] = dict(shares=(ss / den).tolist(),
                              truth=comp["truth"][0],
                              claim=comp["claim"][0])
        # the fixed lenses cannot move between the two prompts: measured
        drift = float((resid_mid[X] - resid_mid[Y]).abs().max())
        # concession arm: the content lens handed the question's own word.
        # z-scored per position over the WHOLE vocabulary -- the standard
        # "which word does this patch say" reading -- then averaged per
        # segment.  This arm is answer-blind: the target word is supplied by
        # the experimenter, not read from the model's decision.
        lgx = m.unembed(resid_mid[X].to(m.device)).float().cpu()
        zf = (lgx - lgx.mean(-1, keepdim=True)) / \
            lgx.std(-1, keepdim=True).clamp_min(1e-6)
        cx = objlens.seg_mean(
            torch.stack([zf[:, CAT_IDS[X][0]], zf[:, CAT_IDS[Y][0]]]), segs)
        i_x, i_y = 0, 1                     # segs[0] is X, segs[1] is Y
        sx, sy = share[X]["shares"], share[Y]["shares"]
        flip_rows.append(dict(
            i=k, concept=c, x=X, y=Y, layer=LMID,
            p_yes_x=probs[X], p_yes_y=probs[Y],
            both_yes=bool(min(probs.values()) >= 0.5),
            share_x_qx=sx[i_x], share_y_qx=sx[i_y],
            share_x_qy=sy[i_x], share_y_qy=sy[i_y],
            scene_qx=sx[-1], scene_qy=sy[-1],
            truth_qx=share[X]["truth"], claim_qx=share[X]["claim"],
            truth_qy=share[Y]["truth"], claim_qy=share[Y]["claim"],
            vj_track=0.5 * (float(sx[i_x] > sx[i_y])
                            + float(sy[i_y] > sy[i_x])),
            static_drift=drift,
            word_track=0.5 * (float(cx[0, i_x] > cx[0, i_y])
                              + float(cx[1, i_y] > cx[1, i_x]))))
        flips += 1
        if min(probs.values()) >= 0.5 and \
                len([cd for cd in cards if cd["kind"] == "flip"]) < a.n_cases:
            cards.append(dict(
                kind="flip", image_id=it["image_id"], path=it["path"],
                res=a.square, layer=LMID, model=tag, obj_idx=[i_x, i_y],
                segments=objlens.card_segments(segs),
                questions=[
                    dict(question=tasks.presence_prompt(X), answer="yes",
                         target=X, shares=sx, p_yes=probs[X]),
                    dict(question=tasks.presence_prompt(Y), answer="yes",
                         target=Y, shares=sy, p_yes=probs[Y])]))

    # ---- 4. open-question cards: the model's own one-word answers
    if opens < a.n_cases and n_obj >= 2:
        X, Y = segs[0].name, segs[1].name
        qx, qy = objlens.open_question(X), objlens.open_question(Y)
        sup = cocodata._load()["super_of"]
        if qx and qy and qx != qy and sup.get(X) != sup.get(Y):
            qs = []
            ok = True
            for q, tgt in ((qx, X), (qy, Y)):
                bq = m.chat(q, [img])
                out = m.hf.generate(
                    **{kk: v for kk, v in bq.items()
                       if not kk.startswith("_")},
                    max_new_tokens=6, do_sample=False,
                    pad_token_id=m.tokenizer.pad_token_id
                    or m.tokenizer.eos_token_id)
                gen = out[0, bq["input_ids"].shape[1]:]
                text = m.tokenizer.decode(gen, skip_special_tokens=True)
                word = text.strip().split()[0].strip(".,!") if text.strip() \
                    else ""
                t0txt = m.tokenizer.decode(gen[:1],
                                           skip_special_tokens=True).strip()
                # the scalar we decompose is the FIRST answer token's logit,
                # so the printed answer must be the word that token starts
                if not word or not word[0].isalpha() \
                        or not word.lower().startswith(t0txt.lower()):
                    ok = False
                    break
                ipq = readout.image_positions(m, bq)
                obj = objective.token_logit(bq, [int(gen[0])],
                                            names=[word])
                per_q = objective.ig_map(m, obj, [LMID], ipq.tolist(),
                                         {LMID: MEANS[LMID]},
                                         steps=a.ig_steps,
                                         chunk=a.ig_chunk)[LMID]
                comp = objective.completeness(m, obj, LMID, ipq.tolist(),
                                              MEANS[LMID], per_q)
                ss = objlens.seg_sum(per_q, segs)[0]
                den = comp["truth"][0] if abs(comp["truth"][0]) > 1e-4 \
                    else float(ss.sum())
                qs.append(dict(question=q, answer=word, target=tgt,
                               shares=(ss / den).tolist(),
                               truth=comp["truth"][0],
                               claim=comp["claim"][0]))
            if ok:
                cards.append(dict(
                    kind="open", image_id=it["image_id"], path=it["path"],
                    res=a.square, layer=LMID, model=tag, obj_idx=[0, 1],
                    segments=objlens.card_segments(segs), questions=qs))
                opens += 1

    if kept % 2 == 0:
        el = time.time() - t0
        print(f"  {kept}/{a.n} kept  flips {flips}  cards {opens}  "
              f"{el:.0f}s  eta {el/max(kept,1)*(a.n-kept)/60:.1f}m",
              flush=True)
        pd.DataFrame(grade_rows).to_parquet(f"{a.out}/{tag}_grade.parquet")
        pd.DataFrame(comp_rows).to_parquet(f"{a.out}/{tag}_complete.parquet")
        pd.DataFrame(flip_rows).to_parquet(f"{a.out}/{tag}_flip.parquet")
        if sam_rows:
            pd.DataFrame(sam_rows).to_parquet(f"{a.out}/{tag}_sam.parquet")
        json.dump(st.jsonable(cards), open(f"{a.out}/{tag}_cards.json", "w"))

# -------------------------------------------------------------- aggregation
df = pd.DataFrame(grade_rows)
dc = pd.DataFrame(comp_rows)
dflip = pd.DataFrame(flip_rows)
df.to_parquet(f"{a.out}/{tag}_grade.parquet")
dc.to_parquet(f"{a.out}/{tag}_complete.parquet")
dflip.to_parquet(f"{a.out}/{tag}_flip.parquet")
if sam_rows:
    pd.DataFrame(sam_rows).to_parquet(f"{a.out}/{tag}_sam.parquet")
json.dump(st.jsonable(cards), open(f"{a.out}/{tag}_cards.json", "w"))

out = {"repo": a.repo, "n": kept, "grid": G, "layers": LAYERS, "lmid": LMID,
       "n_flip": len(dflip), "vocab": a.vocab, "ig_steps": a.ig_steps,
       "minutes": round((time.time() - t0) / 60, 1)}
if len(dc):
    out["completeness"] = dict(
        corr=round(float(dc["corr"].mean()), 4),
        rel_error=round(float(dc.rel_error.mean()), 4),
        max_abs_err=round(float(dc.max_abs_err.max()), 4),
        scale=round(float(dc.scale.mean()), 4),
        partition_gap=float(dc.partition_gap.max()))
    print("\nCOMPLETENESS over the object partition:", out["completeness"],
          flush=True)
if len(df):
    agg = df.groupby(["lens", "truth", "layer"]).agg(
        n=("i", "count"), spearman=("spearman", "mean"),
        hit_obj=("hit_obj", "mean"), pair_acc=("pair_acc", "mean"),
        chance_hit=("chance_hit", "mean")).reset_index()
    agg.to_csv(f"{a.out}/{tag}_summary.csv", index=False)
    print("\nOBJECT-LEVEL GRADING\n", agg.round(3).to_string(index=False),
          flush=True)
    out["by_layer"] = agg.round(4).to_dict("records")
    for tname in df.truth.unique():
        d = df[df.truth == tname]
        for lens in d.lens.unique():
            out[f"ci|{lens}|{tname}"] = st.cluster_bootstrap_ci(
                d[d.lens == lens], "spearman", "concept")
        for other in [x for x in d.lens.unique() if x != "object lens"]:
            out[f"wilcoxon|{other}|{tname}"] = st.wilcoxon(
                d, other, "object lens", "spearman", item="key", cond="lens")
if sam_rows:
    ds = pd.DataFrame(sam_rows)
    out["sam"] = dict(
        n=len(ds), spearman=st.cluster_bootstrap_ci(ds, "spearman", "concept"),
        hit_obj=round(float(ds.hit_obj.mean()), 3),
        chance_hit=round(float(ds.chance_hit.mean()), 3))
    print("\nSAM PARTITION:", out["sam"], flush=True)
if len(dflip):
    dby = dflip[dflip.both_yes] if dflip.both_yes.any() else dflip
    out["flip"] = dict(
        n=len(dflip), n_both_yes=int(dflip.both_yes.sum()),
        vj_track=st.cluster_bootstrap_ci(dby, "vj_track", "concept"),
        word_track=st.cluster_bootstrap_ci(dby, "word_track", "concept"),
        static_drift_max=float(dflip.static_drift.max()),
        share_moved=round(float(
            (dby.share_x_qx - dby.share_x_qy).mean()), 4))
    print("\nQUESTION FLIP:", out["flip"], flush=True)
json.dump(st.jsonable(out), open(f"{a.out}/{tag}_stats.json", "w"), indent=2,
          default=float)
print("done", flush=True)
