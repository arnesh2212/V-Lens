from __future__ import annotations

import argparse
import zlib
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

KS = (0.05, 0.10, 0.20, 0.30, 0.50)
GROUPS = ("top", "bottom", "middle", "random")


def select(order: np.ndarray, k: int, group: str, rng) -> np.ndarray:
    P = len(order)
    if group == "top":
        return order[:k]
    if group == "bottom":
        return order[-k:]
    if group == "middle":
        s = max((P - k) // 2, 0)
        return order[s: s + k]
    return rng.permutation(P)[:k]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(-a)).astype(float)
    rb = np.argsort(np.argsort(-b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/patchcausal")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--dataset", choices=("simonsays", "coco"), default="simonsays")
    ap.add_argument("--subcategories", nargs="*", default=None,
                    help="coco: which POPE items to rank; the default, present, is the one "
                         "with a human mask to grade against")
    ap.add_argument("--n", type=int, default=400, help="items to keep after the correctness filter; 0 means every one")
    ap.add_argument("--pool", type=int, default=1600, help="items to try before giving up")
    ap.add_argument("--frac", type=float, default=0.5, help="depth of the block V-Write is taken at")
    ap.add_argument("--steps", type=int, default=8, help="path-integration steps")
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--baselines", nargs="*",
                    default=["mean", "zero", "noise", "other", "blur"],
                    help="A1 runs all five; the headline curve needs only the mean")
    ap.add_argument("--attr-baselines", action="store_true", default=True,
                    help="also rank by grad x act and at random, as attribution baselines")
    ap.add_argument("--no-attr-baselines", dest="attr_baselines", action="store_false")
    ap.add_argument("--no-rise", action="store_true",
                    help="with --more-baselines, leave RISE out (it is run on its own subsample)")
    ap.add_argument("--more-baselines", action="store_true",
                    help="also integrated gradients (zero), activation patching, Grad-CAM and KernelSHAP")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--int-batch", type=int, default=32)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays
    from vlens.lens import attribution as AT, readout, scores
    from PIL import ImageFilter

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    L = max(int(round(a.frac * (m.n_layers - 1))), 1)
    LAST = m.n_layers - 1
    rng = np.random.default_rng(a.seed)
    say(f"{name}: V-Write at block {L} of {m.n_layers} (depth {L / LAST:.2f})")

    if a.dataset == "coco":
        from vlens.data import coco
        from vlens.lens import localise as LZ
        recs = [r for r in coco.load_records(families=a.families or ["coco_pope"],
                                             subcategories=a.subcategories or ["present"])
                if r["kind"] in ("yesno", "letter", "test_real")]
    else:
        recs = simonsays.scorable(
            [r for r in simonsays.load(families=a.families, require_image=True)
             if r["kind"] in ("yesno", "letter", "test_real")])
    rng.shuffle(recs)
    if a.pool:
        recs = recs[: a.pool]
    if not a.n:
        a.n = len(recs)
    say(f"{len(recs)} candidate items")

    MAIN = "vwrite_mean" if "mean" in a.baselines else f"vwrite_{a.baselines[0]}"
    stem = f"{name}__patchcausal" if abs(a.frac - 0.5) < 1e-9 else \
        f"{name}__d{int(round(a.frac * 100))}__patchcausal"
    part = f"{a.out}/{stem}.rows.jsonl"
    rows, seen_ids = [], set()
    if os.path.exists(part):
        with open(part) as fh:
            for line in fh:
                try:
                    r_ = json.loads(line)
                except Exception:
                    continue
                rows.append(r_); seen_ids.add(str(r_.get("id")))
        say(f"  resuming: {len(rows)} items already measured")
    t0 = time.time()
    prev_h = None
    tally = dict(tried=0, wrong=0, ungraded=0)
    warned = set()
    for r in recs:
        if len(rows) >= a.n:
            break
        if str(r["id"]) in seen_ids:
            continue
        img = C.load_image(r["image"], square=a.square)
        b = m.chat(r["question"], [img], system=r["system"], cue=r["cue"])
        pos = readout.image_positions(m, b)
        P = len(pos)
        if P < 16:
            continue
        fn = scores.score_fn(TS, r["kind"], r.get("expected"))
        with torch.no_grad():
            _, H, _, lg = AT.states_at(m, b, [L])
        h = H[L]
        s0 = float(fn(None, lg).float().cpu()[0])
        s0_raw = s0
        if a.dataset == "coco":
            tally["tried"] += 1
            want = str(r.get("expected") or "").strip().lower()
            if r["kind"] == "yesno" and (s0_raw > 0) != want.startswith("yes"):
                tally["wrong"] += 1
                prev_h = h
                continue
        fn, s0 = scores.orient(fn, r["kind"], s0)
        if s0 <= 0:
            if a.dataset == "coco":
                tally["wrong"] += 1
            prev_h = h
            continue

        mu = h.mean(0, keepdim=True).expand_as(h)
        bases = {}
        if "mean" in a.baselines:
            bases["mean"] = mu
        if "zero" in a.baselines:
            bases["zero"] = torch.zeros_like(h)
        if "noise" in a.baselines:
            g = torch.Generator().manual_seed(a.seed)
            bases["noise"] = torch.randn(h.shape, generator=g) * h.std() + h.mean()
        if "other" in a.baselines and prev_h is not None and prev_h.shape == h.shape:
            bases["other"] = prev_h
        if "blur" in a.baselines:
            bb = m.chat(r["question"], [img.filter(ImageFilter.GaussianBlur(9))],
                        system=r["system"], cue=r["cue"])
            with torch.no_grad():
                _, Hb, _, _ = AT.states_at(m, bb, [L])
            if Hb[L].shape == h.shape:
                bases["blur"] = Hb[L]

        attr = {f"vwrite_{k}": AT.path_reading(m, b, fn, L, pos, h, v, steps=a.steps, log=say)
                for k, v in bases.items()}
        if a.attr_baselines:
            attr["attr_patch"] = AT.one_point(m, b, fn, L, pos, h, mu)
            attr["gradxact_zero"] = AT.one_point(m, b, fn, L, pos, h, torch.zeros_like(h))
            attr["random"] = rng.standard_normal(P).astype(np.float32)
        if a.more_baselines:
            attr["gradcam"] = AT.grad_cam_vision(m, b, fn, pos, log=say)
            with AT.SkipTo(m, b, L, fn=fn, pos=pos, h=h, base=mu, log=say) as bh:
                zero_h = torch.zeros_like(h)
                attr["ig_zero"] = AT.path_reading(m, bh, fn, L, pos, h, zero_h, steps=64, log=say)
                with torch.no_grad():
                    s0 = float(AT.hooked_axis(m, bh, fn, L, pos, h[None].to(m.device)).float().cpu()[0])
                attr["act_patch"] = AT.act_patch(m, bh, fn, L, pos, h, s0, base=mu, log=say)
                gr, gc = readout.image_grid(m, b, P)
                if gr * gc == P:
                    cell = np.arange(P).reshape(gr, gc)
                    gidx = [cell[i * gr // 4:(i + 1) * gr // 4, j * gc // 4:(j + 1) * gc // 4].ravel()
                            for i in range(4) for j in range(4)]
                    crc = zlib.crc32(str(r["id"]).encode())
                    phi, _ = AT.kernel_shap_groups(m, bh, fn, L, pos, h, mu, gidx, nsamples=2080,
                                                   rng=np.random.default_rng([a.seed, crc, 2]), log=say)
                    attr["kernel_shap"] = AT.spread_groups(phi, gidx, P)
                    if not a.no_rise:
                        Mr = AT.rise_masks(gr, gc, 4000, s=7, p=0.5, rng=np.random.default_rng([a.seed, crc, 1]))
                        fr = AT.rise_scores(m, bh, fn, L, pos, h, mu, Mr, log=say)
                        attr["rise"] = AT.rise(Mr, fr, p=0.5)

        loc, inside, cov, fallback = None, None, None, False
        asks_present = r.get("subcategory") == "present" or r.get("family") == "coco_mcq"
        if a.dataset == "coco" and asks_present and (r.get("meta") or {}).get("polys"):
            Gr, Gc = readout.image_grid(m, b, P)
            if Gr == Gc and Gr * Gc == P:
                cov = coco.cell_coverage(r, Gr, Gc)
                inside, fallback = coco.inside_cells(cov)
                attr["center"] = LZ.center_prior(Gr, Gc)
                loc = {meth: LZ.grade(attr[meth], cov, inside) for meth in attr}
            else:
                tally["ungraded"] += 1
                if "grid" not in warned:
                    warned.add("grid")
                    say(f"  !! {P} image tokens do not form a square grid ({Gr} x {Gc}); "
                        "mask grading is skipped, the causal curves are not")

        main_key = MAIN
        order = np.argsort(-attr[main_key])
        plan, X = [], []
        for meth in attr:
            o = np.argsort(-attr[meth])
            for k in KS:
                nk = max(int(round(k * P)), 1)
                for gname in GROUPS:
                    if gname != "top" and meth != main_key:
                        continue
                    sel = select(o, nk, gname, rng)
                    x = h.clone(); x[sel] = mu[sel]
                    plan.append((meth, k, gname)); X.append(x)
        n_plan = len(plan)
        if inside is not None:
            nM = int(inside.sum())
            x = h.clone(); sel = np.where(inside)[0]; x[sel] = mu[sel]
            plan.append(("mask", "M", "exact")); X.append(x)
            for meth in attr:
                sel = np.argsort(-attr[meth])[:nM]
                x = h.clone(); x[sel] = mu[sel]
                plan.append((meth, "M", "top")); X.append(x)
        with torch.no_grad():
            got = AT.hooked_axis_many(m, b, fn, L, pos, X, batch=a.int_batch, log=say)
        mask_drop = {f"{meth}|{g}": float(s0 - v) for (meth, k, g), v in
                     zip(plan[n_plan:], got[n_plan:])}
        mask_flip = {f"{meth}|{g}": bool(v <= 0) for (meth, k, g), v in
                     zip(plan[n_plan:], got[n_plan:])}
        plan, got = plan[:n_plan], got[:n_plan]

        ret = {}
        for meth in [k for k in (main_key, "random") if k in attr]:
            o = np.argsort(-attr[meth])
            xs = []
            for k in KS:
                nk = max(int(round(k * P)), 1)
                x = h.clone(); x[o[:nk]] = mu[o[:nk]]
                xs.append(x)
            with torch.no_grad():
                ret[meth] = AT.hooked_axis_many(m, b, fn, L, pos, xs, batch=a.int_batch, log=say)

        row = dict(
            id=r["id"], family=r["family"], kind=r["kind"], n_patches=int(P), s0=s0,
            drop={f"{meth}|{k}|{g}": float(s0 - v) for (meth, k, g), v in zip(plan, got)},
            flip={f"{meth}|{k}|{g}": bool(v <= 0) for (meth, k, g), v in zip(plan, got)},
            retention={meth: [float(v) for v in ret[meth]] for meth in ret},
            agree={f"{x}|{y}": spearman(attr[x], attr[y])
                   for i, x in enumerate(attr) for y in list(attr)[i + 1:]},
            topk_overlap={f"{x}|{y}": float(len(set(np.argsort(-attr[x])[:max(P // 10, 1)]) &
                                               set(np.argsort(-attr[y])[:max(P // 10, 1)])) / max(P // 10, 1))
                          for i, x in enumerate(attr) for y in list(attr)[i + 1:]},
        )
        if a.dataset == "coco":
            mt = r.get("meta") or {}
            sig = lambda v: [float(f"{x:.4g}") for x in np.asarray(v, np.float64)]
            row.update(subcategory=r.get("subcategory"), s0_raw=s0_raw,
                       said=("yes" if s0_raw > 0 else "no") if r["kind"] == "yesno" else r.get("expected"),
                       target=mt.get("target"), supercategory=mt.get("supercategory"),
                       area=mt.get("area"), image=os.path.basename(r["image"]),
                       loc=loc, mask_drop=mask_drop or None, mask_flip=mask_flip or None,
                       grid=[int(Gr), int(Gc)] if loc else None, inside_fallback=fallback,
                       maps={k: sig(v) for k, v in attr.items() if k != "center"} if loc else None,
                       cov=[round(float(x), 3) for x in cov] if loc else None)
        rows.append(row)
        with open(part, "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        prev_h = h
        if len(rows) % 25 == 0:
            d = rows[-1]["drop"]
            say(f"  {len(rows)}/{a.n} items ({(time.time() - t0) / 60:.1f} min) "
                f"last top20%: {d.get(main_key + '|0.2|top', float('nan')):.2f} "
                f"random20%: {d.get(main_key + '|0.2|random', float('nan')):.2f}")

    def mean_of(key_fn):
        v = [key_fn(x) for x in rows]
        v = [x for x in v if x is not None and np.isfinite(x)]
        return float(np.mean(v)) if v else float("nan")

    main_key = MAIN
    summ = dict(repo=a.repo, block=L, depth=L / LAST, n_items=len(rows), ks=list(KS),
                curve={}, flip_rate={}, retention={}, agreement={}, overlap={})
    for g in GROUPS:
        summ["curve"][g] = [mean_of(lambda x, k=k, g=g: x["drop"].get(f"{main_key}|{k}|{g}")) for k in KS]
        summ["flip_rate"][g] = [mean_of(lambda x, k=k, g=g: float(x["flip"].get(f"{main_key}|{k}|{g}", np.nan)))
                                for k in KS]
    for meth in ([*(("attr_patch", "gradxact_zero", "random", "center") if a.attr_baselines else ()),
                  *(("ig_zero", "act_patch", "gradcam", "kernel_shap", "rise") if a.more_baselines else ()),
                  *[f"vwrite_{b_}" for b_ in a.baselines]]):
        vals = [mean_of(lambda x, k=k, mm=meth: x["drop"].get(f"{mm}|{k}|top")) for k in KS]
        if np.isfinite(vals).any():
            summ["curve"][f"top:{meth}"] = vals
    for meth in (main_key, "random"):
        if not any(meth in x.get("retention", {}) for x in rows):
            continue
        summ["retention"][meth] = [mean_of(lambda x, j=j, mm=meth: (x["retention"].get(mm) or [None] * len(KS))[j])
                                   for j in range(len(KS))]
    keys = sorted({k for x in rows for k in x["agree"]})
    for k in keys:
        summ["agreement"][k] = mean_of(lambda x, k=k: x["agree"].get(k))
        summ["overlap"][k] = mean_of(lambda x, k=k: x["topk_overlap"].get(k))
    if a.dataset == "coco":
        summ["tally"] = dict(tally, kept=len(rows), note="tried/wrong/ungraded count this "
                             "invocation only; a resumed run adds its earlier rows to kept")
    graded = [x for x in rows if x.get("loc")]
    if graded:
        Q = ("hit1", "hit1_any", "hit5", "iou", "auroc", "auroc_all", "energy",
             "chance_hit1", "chance_hit1_any", "chance_energy")
        brng = np.random.default_rng(12345)
        idx = brng.integers(0, len(graded), size=(2000, len(graded)))

        def col(mm, q):
            return np.array([float(((x["loc"].get(mm) or {}).get(q, np.nan))) for x in graded])

        def ci(v):
            ok = np.isfinite(v)
            if ok.sum() < 2:
                return [float("nan"), float("nan")]
            bs = np.nanmean(np.where(ok[idx], v[idx], np.nan), 1)
            return [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
        meths = sorted({mm for x in graded for mm in x["loc"]})
        summ["localisation"] = {mm: {q: float(np.nanmean(col(mm, q))) for q in Q} for mm in meths}
        summ["localisation_ci"] = {mm: {q: ci(col(mm, q)) for q in Q[:7]} for mm in meths}
        if main_key in meths:
            summ["localisation_paired"] = {
                f"{main_key}-{mm}": {q: dict(mean=float(np.nanmean(col(main_key, q) - col(mm, q))),
                                             ci=ci(col(main_key, q) - col(mm, q)))
                                     for q in ("hit1", "auroc", "energy", "iou")}
                for mm in meths if mm != main_key}
        summ["localisation_n"] = len(graded)
        summ["localisation_protocol"] = (
            "cells of the token grid; inside at >= 0.5 coverage by the unioned non-crowd "
            "instance polygons of the asked object, squashed with the photograph onto the "
            "square input; auroc leaves boundary cells (0 < coverage < 0.5) out, auroc_all "
            "counts them as outside; energy weights positive mass by coverage")
        summ["mask_drop"] = {k: mean_of(lambda x, k=k: (x.get("mask_drop") or {}).get(k))
                             for k in sorted({k for x in graded for k in (x.get("mask_drop") or {})})}
        summ["mask_flip"] = {k: mean_of(lambda x, k=k: float((x.get("mask_flip") or {}).get(k, np.nan)))
                             for k in sorted({k for x in graded for k in (x.get("mask_flip") or {})})}
    if os.path.exists(part):
        os.remove(part)
    json.dump(dict(summary=summ, rows=rows), open(f"{a.out}/{stem}.json", "w"), default=float)

    say("\nmean drop in the answer score, by how the patches were chosen")
    say("  k      " + "".join(f"{int(k * 100):>8}%" for k in KS))
    for g in GROUPS:
        say(f"  {g:<7}" + "".join(f"{v:>9.2f}" for v in summ['curve'][g]))
    say("flip rate")
    for g in GROUPS:
        say(f"  {g:<7}" + "".join(f"{v:>9.2f}" for v in summ['flip_rate'][g]))
    if summ.get("tally"):
        t_ = summ["tally"]
        say(f"\n{t_['tried']} items tried, {t_['wrong']} answered wrongly and left out, "
            f"{t_['ungraded']} kept without a mask grade")
    if summ.get("localisation"):
        say(f"\nagainst the human mask, {summ['localisation_n']} items the model answered right")
        say(f"  {'ranking':<16}{'hit@1':>8}{'any@1':>8}{'hit@5':>8}{'IoU@|M|':>9}{'AUROC':>8}{'energy':>8}")
        for mm, v in summ["localisation"].items():
            say(f"  {mm:<16}{v['hit1']:>8.2f}{v['hit1_any']:>8.2f}{v['hit5']:>8.2f}{v['iou']:>9.2f}"
                f"{v['auroc']:>8.2f}{v['energy']:>8.2f}")
        v = summ["localisation"][mm]
        say(f"  chance: hit@1 {v['chance_hit1']:.2f}, any@1 {v['chance_hit1_any']:.2f}, "
            f"AUROC 0.50, energy {v['chance_energy']:.2f}")
        say("  drop when |M| cells are replaced: " + ", ".join(
            f"{k.split('|')[0]} {v:.2f}" for k, v in summ["mask_drop"].items()))
    say(f"wrote {a.out}/{stem}.json")
    say("DONE patch_causal")


if __name__ == "__main__":
    main()
