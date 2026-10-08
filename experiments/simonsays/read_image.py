# SPDX-License-Identifier: Apache-2.0
"""Read the answer off the image tokens rather than off the answer spot.

Fits a map from an image token's state to the model's own final state at its
answer spot, label-free, and scores it against the published lenses on the same
states. A counterfactual stage uses matched original/edited photographs to ask
which patches changed what they say about the answer.

    python3 experiments/simonsays/read_image.py --repo Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from vlens.lens.fitall import SHARED as FA_SHARED  # noqa: E402

from vlens.lens.metrics import auroc, boot  # noqa: E402


def grid(n_layers: int, k: int, lo: int = 0) -> list[int]:
    hi = n_layers - 2
    if k >= hi - lo + 1:
        return list(range(lo, hi + 1))
    return sorted({int(round(x)) for x in np.linspace(lo, hi, k)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/readimage")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=0, help="items to score; 0 means every one available")
    ap.add_argument("--cap", type=int, default=FA_SHARED["cap"],
                    help="the behaviour cap, to stay disjoint from it")
    ap.add_argument("--results", default="results/simonsays")
    ap.add_argument("--only", nargs="*", default=None,
                    help="instruments to fit; the default is the four published lenses and V-Read")
    ap.add_argument("--corpus", type=int, default=FA_SHARED["corpus"],
                    help="prompts every transport is fitted on")
    ap.add_argument("--heldout", type=int, default=FA_SHARED["heldout"])
    ap.add_argument("--layers", type=int, default=FA_SHARED["layers"])
    ap.add_argument("--steps", type=int, default=FA_SHARED["steps"])
    ap.add_argument("--tokens-per-step", type=int, default=FA_SHARED["tokens_per_step"])
    ap.add_argument("--lr-grid", type=float, nargs="*", default=[1.0, 0.1, 0.01])
    ap.add_argument("--lr-probe-steps", type=int, default=30)
    ap.add_argument("--lam-grid", type=float, nargs="*", default=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    ap.add_argument("--jac-prompts", type=int, default=FA_SHARED["jac_prompts"])
    ap.add_argument("--latent-captions", type=int, default=FA_SHARED["latent_captions"])
    ap.add_argument("--fit-batch", type=int, default=FA_SHARED["fit_batch"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--square", type=int, default=FA_SHARED["square"])
    ap.add_argument("--max-patches", type=int, default=256)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--pairs", type=int, default=600,
                    help="matched original/edited photographs for the counterfactual test; "
                         "0 turns it off")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays
    from vlens.lens import baselines as B, fitall as FA, scores
    from vlens.lens.hooks import ActivationRecorder

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    LAST = m.n_layers - 1
    LAYERS = FA.layer_grid(m, a.layers)
    say(f"{name}: {m.n_layers} blocks; fitting and reading at {LAYERS}")

    recs = simonsays.load(require_image=True)
    seen, spare, scored = {}, [], []
    for r in recs:
        k = (r["family"], r["subcategory"], r["framing"])
        seen[k] = seen.get(k, 0) + 1
        (spare if seen[k] > a.cap else scored).append(r)
    scored = simonsays.scorable([r for r in scored
                                 if r["kind"] in ("yesno", "letter", "test_real")])
    if a.families:
        scored = [r for r in scored if r["family"] in a.families]
    say(f"{len(scored)} scored records, {len(spare)} held back for fitting")

    items = B.corpus_items(spare, C.load_image, n=a.corpus + 2 * a.heldout,
                           square=a.square, seed=0)
    fit_items = items[: a.corpus]
    dev_items = items[a.corpus: a.corpus + a.heldout]
    held_items = items[a.corpus + a.heldout:]
    FIT = B.fit_ids(fit_items, dev_items, held_items)

    maps, meta = {}, {}

    # ---- the cross-position map: an image token to the answer spot
    t0 = time.time()
    acc = B.RidgeAccumulator(LAYERS, m.d_model, device=m.device)
    ntok = 0
    for X, Y, n in B.ImageToAnswerStream(m, fit_items, batch=a.fit_batch, seed=1,
                                         max_patches=a.max_patches):
        acc.add_xy({l: X[l].float() for l in LAYERS}, Y.float()); ntok += n
    best_lam, best_kl, best = None, None, None
    # the penalty is chosen by how well the map predicts the held-out answer
    # spot, not by anything the scored set knows
    hacc = B.RidgeAccumulator(LAYERS, m.d_model, device=m.device)
    HX, HY = [], []
    for X, Y, n in B.ImageToAnswerStream(m, held_items, batch=a.fit_batch, seed=2,
                                         max_patches=a.max_patches):
        HX.append({l: X[l].float().cpu() for l in LAYERS}); HY.append(Y.float().cpu())
    for lam in a.lam_grid:
        M = acc.solve(lam=lam)
        err = 0.0
        for Xc, Yc in zip(HX, HY):
            for l in LAYERS:
                A_, b_ = M[l]
                err += float(((Xc[l] @ A_.T + b_) - Yc).pow(2).mean())
        err /= max(len(HX) * len(LAYERS), 1)
        say(f"    cross map lambda {lam:<7} -> held-out squared error {err:.4f}")
        if best_kl is None or err < best_kl:
            best_lam, best_kl, best = lam, err, M
    maps["v_read_cross"] = best
    meta["v_read_cross"] = dict(lam=best_lam, tokens=ntok, seconds=round(time.time() - t0),
                                kind="image token at block l -> answer-spot final state")
    say(f"  cross map fitted on {ntok} image tokens, lambda {best_lam} ({time.time() - t0:.0f}s)")
    del HX, HY

    # ---- the local maps, from the one fit shared with every other experiment
    # that reads at these blocks
    shared, latent, fit_ids, smeta = FA.fit_all(
        m, spare, C.load_image, layers=LAYERS, corpus=a.corpus, heldout=a.heldout,
        square=a.square, steps=a.steps, tokens_per_step=a.tokens_per_step,
        jac_prompts=a.jac_prompts, latent_captions=a.latent_captions,
        fit_batch=a.fit_batch, want=tuple(a.only) if a.only else FA.SHARED_WANT, results=a.results, log=say)
    maps.update(shared)
    meta.update(smeta)
    FIT |= fit_ids
    lr = (smeta.get("tuned") or {}).get("lr_chosen", 0.1)

    maps["tuned_cross"] = B.fit_tuned_lens_cross(
        m, B.CachedPairs(B.ImageToAnswerStream(m, fit_items, batch=a.fit_batch, seed=1),
                         LAYERS, log=say),
        layers=LAYERS, steps=a.steps, tokens_per_step=a.tokens_per_step, lr=lr, log=say)
    #: what to report, taken from what was actually fitted rather than from a
    #: list that has to be kept in step with --only by hand. It was not, and a
    #: fourteen-hour run ended on a KeyError for the one instrument the list
    #: had never been told about.
    PREF = ("v_read_cross", "tuned_cross", "v_read", "tuned", "jacobian")
    names = [k for k in PREF if k in maps] + [k for k in maps if k not in PREF]
    if latent is not None:
        names.append("latent")
    names.append("logit")
    #: how the per-patch readings are turned into one number for the picture
    HOWS = ("pooled", "mean", "top10", "best")

    # ---- score every image token of every scored record
    scored = [r for r in scored if str(r["id"]) not in FIT]
    if a.n:
        scored = FA.balanced_sample(scored, a.n, seed=0)
    say(f"scoring {len(scored)} records, every image token, {len(names)} instruments")

    rows, t0 = [], time.time()
    for s0 in range(0, len(scored), a.batch):
        chunk = scored[s0: s0 + a.batch]
        pack = [dict(question=r["question"], image=C.load_image(r["image"], square=a.square),
                     system=r["system"], cue=r["cue"]) for r in chunk]
        b = m.batch_many(pack, side="right")
        spot = m.answer_spot(b)
        mask = m.image_mask(b["input_ids"])
        with torch.no_grad(), ActivationRecorder(m.layers, at=[*LAYERS, LAST]) as rec:
            m.forward(b)
            H = {l: rec.activations[l].detach() for l in [*LAYERS, LAST]}
            fin = m.unembed(H[LAST][torch.arange(len(chunk)), spot.to(H[LAST].device)])
        for r_i, r in enumerate(chunk):
            fn = scores.score_fn(TS, r["kind"], r.get("expected"))
            s_final = float(fn(None, fin[r_i: r_i + 1]).float().cpu()[0])
            pos = mask[r_i].nonzero(as_tuple=True)[0]
            if pos.numel() < 8:
                continue
            if a.max_patches and pos.numel() > a.max_patches:
                keep = torch.randperm(pos.numel(), device=pos.device)[: a.max_patches]
                pos = pos[keep.sort().values]
            agg = {h_: {k: [] for k in names} for h_ in HOWS}
            for l in LAYERS:
                h = H[l][r_i, pos].float()
                cand = {"logit": h}
                for k, M in maps.items():
                    A_, b_ = M[l]
                    cand[k] = h @ A_.T.to(h.device, h.dtype) + b_.to(h.device, h.dtype)
                with torch.no_grad():
                    for k, v in cand.items():
                        sc = fn(None, m.unembed(v.to(torch.bfloat16))).float().cpu().numpy()
                        # the whole picture's reading, not one patch of it: a
                        # single patch rarely decides a yes or no, and the sum
                        # of what the patches carry is what reaches the answer
                        pooled = float(fn(None, m.unembed(
                            v.mean(0, keepdim=True).to(torch.bfloat16))).float().cpu()[0])
                        agg["pooled"][k].append(pooled)
                        agg["mean"][k].append(float(sc.mean()))
                        agg["best"][k].append(float(np.max(sc)))
                        kk = max(len(sc) // 10, 1)
                        agg["top10"][k].append(float(np.sort(sc)[-kk:].mean()))
                    q = latent.similarity_logits(h, l) / max(latent.meta.get("temp", 0.05), 1e-6)
                    sc = fn(None, q).float().cpu().numpy()
                    qm = latent.similarity_logits(h.mean(0, keepdim=True), l) / \
                        max(latent.meta.get("temp", 0.05), 1e-6)
                    agg["pooled"]["latent"].append(float(fn(None, qm).float().cpu()[0]))
                    agg["mean"]["latent"].append(float(sc.mean()))
                    agg["best"]["latent"].append(float(np.max(sc)))
                    kk = max(len(sc) // 10, 1)
                    agg["top10"]["latent"].append(float(np.sort(sc)[-kk:].mean()))
            rows.append(dict(id=r["id"], family=r["family"], kind=r["kind"],
                             n_patches=int(pos.numel()), final=s_final, **agg))
        if (s0 // a.batch) % 25 == 0:
            say(f"  {min(s0 + a.batch, len(scored))}/{len(scored)} "
                f"({(time.time() - t0) / 60:.1f} min)")
        del H

    if not rows:
        say("nothing scored"); return
    y = np.array([int(r["final"] > 0) for r in rows])
    out = dict(repo=a.repo, layers=LAYERS, depths=[l / LAST for l in LAYERS], n=len(rows),
               instruments=names, meta=meta, auroc={}, ci={})
    for how in HOWS:
        for k in names:
            S = np.array([r[how][k] for r in rows], float)
            pts = [boot(y, S[:, j], reps=a.boot) for j in range(len(LAYERS))]
            out["auroc"][f"{how}|{k}"] = [p[0] for p in pts]
            out["ci"][f"{how}|{k}"] = [[p[1] for p in pts], [p[2] for p in pts]]
    per_family = {}
    for fam in sorted({r["family"] for r in rows}):
        sel = [i for i, r in enumerate(rows) if r["family"] == fam]
        if len(sel) < 24 or len(set(y[sel].tolist())) < 2:
            continue
        per_family[fam] = dict(n=len(sel), auroc={
            f"model|{k}": [auroc(y[sel], np.array([rows[i]["pooled"][k][j] for i in sel]))
                           for j in range(len(LAYERS))] for k in names})
        per_family[fam]["depths"] = out["depths"]
    out["families"] = per_family
    json.dump(dict(summary=out, rows=rows), open(f"{a.out}/{name}__readimage.json", "w"),
              default=float)
    # ---- does the instrument see the edit?
    #
    # The dataset supplies the same photograph twice with one thing changed.
    # Reading the image tokens inside the changed region should move when the
    # picture moves, and it should move by as much as the model's own answer
    # moves. An instrument that reads a patch's own next token has no reason to
    # track that; a map fitted from a patch to the answer spot has every reason.
    pair_out = {}
    if a.pairs:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from handoff import pairs_for
        from vlens.lens import attribution as AT
        allrec = simonsays.load(require_image=True)
        prs = [(o, e) for o, e in pairs_for(allrec)
               if str(o["id"]) not in FIT and str(e["id"]) not in FIT]
        rng = np.random.default_rng(0)
        prs = [prs[i] for i in rng.permutation(len(prs))][: a.pairs]
        say(f"\ncounterfactual test on {len(prs)} matched photographs")
        d_model, d_inst = [], {k: [] for k in names}
        d_loc = {k: [] for k in names}
        n_loc = 0
        #: why a pair was not usable, so a stage that produces nothing says why
        skip = {"image": 0, "token count": 0, "grid": 0, "mask": 0}
        t1 = time.time()
        for o, e in prs:
            try:
                imo = C.load_image(o["image"], square=a.square)
                ime = C.load_image(e["image"], square=a.square)
            except Exception:  # noqa: BLE001
                skip["image"] += 1; continue
            bo = m.chat(o["question"], [imo], system=o["system"], cue=o["cue"])
            be = m.chat(e["question"], [ime], system=e["system"], cue=e["cue"])
            po = m.image_mask(bo["input_ids"])[0].nonzero(as_tuple=True)[0]
            pe = m.image_mask(be["input_ids"])[0].nonzero(as_tuple=True)[0]
            if po.numel() != pe.numel() or po.numel() < 16:
                skip["token count"] += 1; continue
            G = int(round(po.numel() ** 0.5))
            if G * G != po.numel():
                skip["grid"] += 1; continue
            sq = AT.edit_mask(o["image"], e["image"], side=a.square)
            inside = np.where(AT.pool_mask(sq, G) > 0.05)[0]
            if len(inside) < 2 or len(inside) > 0.6 * po.numel():
                skip["mask"] += 1; continue
            with torch.no_grad(), ActivationRecorder(m.layers, at=[*LAYERS, LAST]) as rec:
                m.forward(bo)
                Ho = {l: rec.activations[l][0].detach() for l in [*LAYERS, LAST]}
            with torch.no_grad(), ActivationRecorder(m.layers, at=[*LAYERS, LAST]) as rec:
                m.forward(be)
                He = {l: rec.activations[l][0].detach() for l in [*LAYERS, LAST]}
            with torch.no_grad():
                lo = m.unembed(Ho[LAST][bo["input_ids"].shape[1] - 1][None])
                le = m.unembed(He[LAST][be["input_ids"].shape[1] - 1][None])
            t_o, t_e = int(lo[0].argmax()), int(le[0].argmax())
            # localisation uses every pair; only the correlation with the
            # answer moving needs a pair whose answer flipped
            f_loc = lambda hL, lg, i=t_o: torch.log_softmax(lg.float(), -1)[..., i]
            flipped = t_o != t_e
            f2 = (lambda hL, lg, i=t_e, j=t_o: (torch.log_softmax(lg.float(), -1)[..., i]
                                                - torch.log_softmax(lg.float(), -1)[..., j])
                  ) if flipped else f_loc
            if flipped:
                d_model.append(float(f2(None, le).float().cpu()[0] - f2(None, lo).float().cpu()[0]))
            sel = torch.as_tensor(inside)
            truth = np.zeros(po.numel(), int); truth[inside] = 1
            row, loc = {}, {}
            for k in names:
                vals, locs = [], []
                for l in LAYERS:
                    # every patch, so the reading can be asked to say which ones
                    # changed; the edited region is the ground truth and the
                    # dataset supplies it, which is the point of having pairs
                    hoa = Ho[l][po].float(); hea = He[l][pe].float()
                    if k == "logit":
                        va, vb = hoa, hea
                    elif k == "latent":
                        with torch.no_grad():
                            tp = max(latent.meta.get("temp", 0.05), 1e-6)
                            qa = latent.similarity_logits(hoa, l) / tp
                            qb = latent.similarity_logits(hea, l) / tp
                            da = (f_loc(None, qb) - f_loc(None, qa)).abs().float().cpu().numpy()
                        locs.append(auroc(truth, da))
                        va = vb = None
                    else:
                        A_, b_ = maps[k][l]
                        va = hoa @ A_.T.to(hoa.device, hoa.dtype) + b_.to(hoa.device, hoa.dtype)
                        vb = hea @ A_.T.to(hea.device, hea.dtype) + b_.to(hea.device, hea.dtype)
                    if va is not None:
                        with torch.no_grad():
                            sa = f_loc(None, m.unembed(va.to(torch.bfloat16)))
                            sb = f_loc(None, m.unembed(vb.to(torch.bfloat16)))
                            locs.append(auroc(truth, (sb - sa).abs().float().cpu().numpy()))
                    ho = Ho[l][po[sel]].float(); he = He[l][pe[sel]].float()
                    if k == "logit":
                        vo, ve = ho, he
                    elif k == "latent":
                        with torch.no_grad():
                            qo = latent.similarity_logits(ho.mean(0, keepdim=True), l)
                            qe = latent.similarity_logits(he.mean(0, keepdim=True), l)
                            tp = max(latent.meta.get("temp", 0.05), 1e-6)
                            vals.append(float(f2(None, qe / tp).float().cpu()[0]
                                              - f2(None, qo / tp).float().cpu()[0]))
                        continue
                    else:
                        A_, b_ = maps[k][l]
                        vo = ho @ A_.T.to(ho.device, ho.dtype) + b_.to(ho.device, ho.dtype)
                        ve = he @ A_.T.to(he.device, he.dtype) + b_.to(he.device, he.dtype)
                    with torch.no_grad():
                        so = float(f2(None, m.unembed(vo.mean(0, keepdim=True)
                                                      .to(torch.bfloat16))).float().cpu()[0])
                        se = float(f2(None, m.unembed(ve.mean(0, keepdim=True)
                                                      .to(torch.bfloat16))).float().cpu()[0])
                    vals.append(se - so)
                row[k] = vals
                loc[k] = locs
            for k in names:
                if flipped:
                    d_inst[k].append(row[k])
                d_loc[k].append(loc[k])
            n_loc += 1
            if n_loc % 50 == 0:
                say(f"  {n_loc} pairs usable, {len(d_model)} of them with the answer moved "
                    f"({(time.time() - t1) / 60:.1f} min)")
            del Ho, He
        say(f"  {n_loc} pairs usable of {len(prs)}; skipped " +
            (", ".join(f"{k} {v}" for k, v in skip.items() if v) or "none"))
        if n_loc >= 12:
            dm = np.array(d_model, float) if d_model else np.zeros(0)
            pair_out = dict(n=int(n_loc), n_flipped=int(len(dm)), depths=out["depths"],
                            correlation={}, auroc={}, localisation={})
            yb = (dm > 0).astype(int)
            for k in names:
                L = np.array(d_loc[k], float)
                pair_out["localisation"][k] = [float(np.nanmean(L[:, j]))
                                               for j in range(len(LAYERS))]
                if len(dm) < 12:
                    continue
                M = np.array(d_inst[k], float)
                cors, aus = [], []
                for j in range(len(LAYERS)):
                    v = M[:, j]
                    ok = np.isfinite(v) & np.isfinite(dm)
                    if ok.sum() > 8 and np.std(v[ok]) > 0:
                        rr = np.argsort(np.argsort(v[ok])).astype(float)
                        rm = np.argsort(np.argsort(dm[ok])).astype(float)
                        rr -= rr.mean(); rm -= rm.mean()
                        cors.append(float((rr * rm).sum()
                                          / np.sqrt((rr ** 2).sum() * (rm ** 2).sum())))
                    else:
                        cors.append(float("nan"))
                    aus.append(auroc(yb, v))
                pair_out["correlation"][k] = cors
                pair_out["auroc"][k] = aus
            out["counterfactual"] = pair_out
            if pair_out["correlation"]:
                say(f"\nhow well the reading of the changed patches tracks the answer moving "
                    f"({len(dm)} pairs whose answer moved)")
                say("  depth        " + "".join(f"{d:>8.2f}" for d in out["depths"]))
                for k in names:
                    say(f"  {k:<13}" + "".join(f"{v:>8.2f}"
                                               for v in pair_out["correlation"].get(k, [])))
            say(f"\nAUROC for telling a changed patch from an unchanged one ({n_loc} pairs, "
                f"every patch)")
            say("  depth        " + "".join(f"{d:>8.2f}" for d in out["depths"]))
            for k in names:
                say(f"  {k:<13}" + "".join(f"{v:>8.2f}"
                                          for v in pair_out.get("localisation", {}).get(k, [])))
        json.dump(dict(summary=out, rows=rows), open(f"{a.out}/{name}__readimage.json", "w"),
                  default=float)

    at = lambda v, x: v[int(np.argmin(np.abs(np.array(out["depths"]) - x)))]
    for how, lab in (("pooled", "the picture read as a whole"),
                     ("mean", "the mean over image tokens"),
                     ("top10", "the best tenth of image tokens"),
                     ("best", "the single best image token")):
        say(f"\nAUROC for the answer the model gives, from {lab} ({len(rows)} records)")
        say("  depth        " + "".join(f"{d:>8.2f}" for d in out["depths"]))
        for k in names:
            say(f"  {k:<13}" + "".join(f"{v:>8.2f}" for v in out["auroc"][f"{how}|{k}"]))
    say(f"\n{len(per_family)} families kept for the paired test")
    say("DONE read_image")


if __name__ == "__main__":
    main()
