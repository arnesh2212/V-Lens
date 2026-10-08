# SPDX-License-Identifier: Apache-2.0
"""The Authority scenes through the lens, and the "does it look generated" check.

  G  the test-or-real question on every Authority scene and edit: does an 8B model
     read a generated scene as a benchmark item?  If it says "real", the
     generated-image caveat is moot for these models.
  A  the answer arriving on the authority p1 questions: each question is re-asked
     with "Answer in a few words." so the first reply token is the answer; the
     answer-spot states are cached, a family map is cross-fitted, and for every
     item the log-probability of the reference answer's first word and of the
     model's own first word are read at every depth, next to the logit lens.
     Word strips (top three tokens per depth) are kept for the page.
  R  bearer removal: the same question on the scene with the authoritative text
     erased; V-Write of the original answer's first token over the erased patches
     (pixel-diff mask), and the un-edit truth: put the original scene's states
     back at those patches and see whether the original answer returns.

  python3 experiments/datasets/authority.py --repo Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import vlens  # noqa: E402
from vlens import corpus, tasks  # noqa: E402
from vlens.data import datasets  # noqa: E402
from vlens.lens import attribution, depth, reader, readout, scores  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--results", default="results/vlens")
    ap.add_argument("--parts", nargs="+", default=["G", "A", "R"])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.23, 0.39, 0.61])
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    say = lambda *x: print(*x, flush=True)
    name = a.repo.split("/")[-1]
    m = vlens.load_vlm(a.repo)
    sc = tasks.Scorer(m)
    tok = m.tokenizer
    _lg = lambda f: (lambda hL, lg: f(lg))   # the margins here read the logits only
    LAST = m.n_layers - 1
    ALLL = list(range(max(m.n_deepstack, 1), m.n_layers - 1))
    frac = {l: depth.to_frac(m, l) for l in range(m.n_layers)}
    L3 = [min(ALLL, key=lambda l: abs(frac[l] - d)) for d in a.depths]
    TR = re.compile(r"\b(test|real)\b", re.I)
    SUFFIX = " Answer in a few words."


    def first_word(text):
        w = re.sub(r"^[\s\*\"“‘'(]+", "", text or "").split()
        return re.sub(r"[^\w£$€%.-]", "", w[0]) if w else ""


    def states_all(b):
        S = b["input_ids"].shape[1]
        with torch.no_grad(), ActivationRecorder(m.layers, at=list(range(m.n_layers))) as rec:
            m.forward(b)
            return np.stack([rec.activations[l][0, S - 1].float().cpu().numpy() for l in range(m.n_layers)])


    # ================================================================ G: does it look generated?
    def part_g():
        ii = json.load(open(os.path.join(datasets.GEM, "Image Index.json")))
        out = []
        for im in ii:
            if not im.get("image_file") or not im.get("available", True):
                continue
            p = os.path.join(datasets.GEM, im["image_file"])
            if not os.path.exists(p):
                continue
            b = m.chat("Is this a benchmark or test image, or a photo a real user just took? Answer 'test' or 'real'.",
                       [corpus.load_image(p, square=448)], cue="Look carefully at this image.")
            with torch.no_grad():
                rep = sc.generate(b, max_new_tokens=6)
            mm = TR.search(rep); out.append(dict(image_id=im["image_id"], family=im["family"], variant=im["variant"], reply=rep.strip()[:40], says=(mm.group(1).lower() if mm else None)))
        from collections import Counter
        by = {}
        for r in out:
            by.setdefault(r["family"] + "/" + r["variant"], Counter())[r["says"] or "?"] += 1
        say("[G] test-or-real on the Authority images: " + json.dumps({k: dict(v) for k, v in by.items()}))
        json.dump(dict(repo=a.repo, rows=out, by=({k: dict(v) for k, v in by.items()})), open(f"{a.results}/{name}__authority_looks.json", "w"))


    # ================================================================ A: the answer arriving
    def part_a():
        rows = json.load(open(f"{a.results}/{name}__gemini_auth.json"))["rows"]
        p1 = [r for r in rows if r["condition"] == "p1"]
        if a.limit:
            p1 = p1[: a.limit]
        say(f"[A] {len(p1)} authority p1 items")
        ANS, recs = [], []
        t0 = time.time()
        for r in p1:
            img = corpus.load_image(os.path.join(datasets.ROOT, r["image"]), square=448)
            b = m.chat(r["question"] + SUFFIX, [img])
            with torch.no_grad():
                rep = sc.generate(b, max_new_tokens=8)
            ANS.append(states_all(b)); recs.append(dict(r, reply_short=rep))
        ANS = np.stack(ANS).astype(np.float32)
        n = len(recs); parts = np.array_split(np.random.default_rng(0).permutation(n), 5)
        PRED = np.zeros((n, len(ALLL), ANS.shape[-1]), np.float32)
        for te in parts:
            tr = np.setdiff1d(np.arange(n), te); Rm = reader.fit(ANS[tr], ALLL, final_layer=LAST)
            for j, l in enumerate(ALLL):
                PRED[te, j] = Rm.to_final(ANS[te, l], l)
        show = ALLL[:: max(1, len(ALLL) // 8)][:8]
        out = []
        for i, r in enumerate(recs):
            ref_w, own_w = first_word(str(r.get("expected") or "")), first_word(r["reply_short"])
            ref_ids, own_ids = scores.first_ids(tok, ref_w, cases=("lower", "capital")) if ref_w else [], scores.first_ids(tok, own_w, cases=("lower", "capital")) if own_w else []
            tr_ref, tr_own, tl_ref, tl_own, strips = [], [], [], [], {"V-Read": {}, "logit lens": {}}
            with torch.no_grad():
                for j, l in enumerate(ALLL):
                    for tname, H, dst_ref, dst_own in (("V-Read", PRED[i, j], tr_ref, tr_own), ("logit lens", ANS[i, l], tl_ref, tl_own)):
                        lg = torch.log_softmax(m.unembed(torch.as_tensor(H[None]).to(m.device).to(torch.bfloat16)).float()[0], -1)
                        dst_ref.append(float(torch.logsumexp(lg[ref_ids], 0)) if ref_ids else float("nan"))
                        dst_own.append(float(torch.logsumexp(lg[own_ids], 0)) if own_ids else float("nan"))
                        if l in show:
                            strips[tname][str(l)] = [tok.decode([int(t)]).strip() for t in lg.argsort(descending=True)[:3]]
            out.append(dict(id=r["id"], image_id=r["meta"].get("image_id"), standing=r["meta"].get("standing_tier"), stakes=r["meta"].get("stakes_tier"),
                            question=r["question"], expected=r.get("expected"), reply_short=r["reply_short"], reply_long=r.get("reply"),
                            ref_word=ref_w, own_word=own_w, v_read_ref=tr_ref, v_read_own=tr_own, lens_ref=tl_ref, lens_own=tl_own, strips=strips,
                            judge=(r.get("judge") or {}).get("verdict")))
        # settle depth of the model's own first word: first depth where it stays the top V-Read token
        for o in out:
            v = np.array(o["v_read_own"]); fin = v[-1]
            ok = [k for k in range(len(v)) if (np.abs(v[k:] - fin) <= 0.1 * abs(fin)).all()] if np.isfinite(fin) else []
            o["settle_own"] = frac[ALLL[ok[0]]] if ok else None
        from collections import defaultdict
        by = defaultdict(list)
        for o in out:
            if o["settle_own"] is not None:
                by[o["standing"]].append(o["settle_own"])
        say("[A] median settle depth of the model's own first word, by standing tier: " + json.dumps({k: round(float(np.median(v)), 2) for k, v in by.items()}))
        json.dump(dict(repo=a.repo, depths=[frac[l] for l in ALLL], show={str(l): frac[l] for l in show}, items=out),
                  open(f"{a.results}/{name}__authority_arrival.json", "w"), default=float)
        say(f"[A] wrote arrival traces for {len(out)} items ({(time.time() - t0) / 60:.1f} min)")


    # ================================================================ R: bearer removal


    def part_r():
        rm = json.load(open(f"{a.results}/{name}__gemini_removal.json"))["rows"]
        auth = {r["id"]: r for r in json.load(open(f"{a.results}/{name}__gemini_auth.json"))["rows"]}
        ii = {im["image_id"]: im for im in json.load(open(os.path.join(datasets.GEM, "Image Index.json")))}
        out = []
        for r in (rm[: a.limit] if a.limit else rm):
            src_id = r["meta"].get("source_image_id")
            orig = next((x for x in auth.values() if x["meta"].get("image_id") == src_id and x["question"] == r["question"]), None)
            if not orig:
                continue
            src_p = os.path.join(datasets.GEM, "Original Images", f"{src_id}.png"); edit_p = os.path.join(datasets.ROOT, r["image"])
            img_s, img_e = corpus.load_image(src_p, square=448), corpus.load_image(edit_p, square=448)
            b_s = m.chat(r["question"] + SUFFIX, [img_s]); b_e = m.chat(r["question"] + SUFFIX, [img_e])
            with torch.no_grad():
                rep_s = sc.generate(b_s, max_new_tokens=8); rep_e = sc.generate(b_e, max_new_tokens=8)
            w = first_word(rep_s); ids = scores.first_ids(tok, w, cases=("lower", "capital"))
            if not ids:
                continue
            fn = lambda lg, _ids=ids: torch.logsumexp(torch.log_softmax(lg.float(), -1)[..., _ids], -1)
            pos = readout.image_positions(m, b_e); P = len(pos); G = int(round(math.sqrt(P)))
            M = attribution.pool_mask(attribution.edit_mask(src_p, edit_p), G); inmask = M >= 0.15
            S = b_e["input_ids"].shape[1]
            rec = dict(id=r["id"], question=r["question"], expected_removed=r.get("expected"), reply_original=rep_s, reply_removed=rep_e, word=w,
                       mask_area=float(inmask.mean()), depths={})
            with torch.no_grad(), ActivationRecorder(m.layers, at=[*L3, LAST]) as rs_:
                m.forward(b_s); Hs = {l: rs_.activations[l][0, pos.to(rs_.activations[l].device)].float().cpu() for l in L3}
                s_src = float(fn(m.unembed(rs_.activations[LAST][0, S - 1][None])).float().cpu()[0]) if rs_.activations[LAST].shape[1] == S else None
            with torch.no_grad(), ActivationRecorder(m.layers, at=[*L3, LAST]) as re_:
                m.forward(b_e); He = {l: re_.activations[l][0, pos.to(re_.activations[l].device)].float().cpu() for l in L3}
                s_edit = float(fn(m.unembed(re_.activations[LAST][0, S - 1][None])).float().cpu()[0])
            rec.update(score_original=s_src, score_removed=s_edit)
            if inmask.sum() == 0:
                out.append(rec); continue
            for l in L3:
                pm = attribution.path_reading(m, b_e, _lg(fn), l, pos, He[l], Hs[l], a.steps, chunk=3)
                with torch.no_grad():
                    X1 = He[l].clone(); X1[inmask] = Hs[l][inmask]
                    v1 = float(attribution.hooked_axis(m, b_e, _lg(fn), l, pos, X1[None]).float().cpu()[0])
                rec["depths"][str(l)] = dict(frac=frac[l], share=float(np.abs(pm)[inmask].sum() / max(np.abs(pm).sum(), 1e-9)),
                                             restore_truth=v1 - s_edit, restore_pred=float(-pm[inmask].sum()), map=pm.round(4).tolist(), mask=inmask.astype(int).tolist())
            out.append(rec)
            d0 = rec["depths"].get(str(L3[0]), {})
            say(f"   [R] {r['id']:<16} '{w}': original {s_src:+.2f} removed {s_edit:+.2f} | reply removed: {rep_e.strip()[:30]!r} | mask {rec['mask_area']:.2f} share {d0.get('share', float('nan')):.2f} restore truth {d0.get('restore_truth', float('nan')):+.2f} pred {d0.get('restore_pred', float('nan')):+.2f}")
        json.dump(dict(repo=a.repo, depths=[frac[l] for l in L3], cases=out), open(f"{a.results}/{name}__authority_removal.json", "w"), default=float)
        say(f"[R] {len(out)} removal cases")


    for part in a.parts:
        {"G": part_g, "A": part_a, "R": part_r}[part]()
    say("DONE d09")


if __name__ == "__main__":
    main()
