# SPDX-License-Identifier: Apache-2.0
"""How many samples RISE and KernelSHAP need here: a convergence study that picks their budget.

Both are run at their published budgets (RISE 4000 masks, Petsiuk et al.
2018; KernelSHAP shap's default 2M + 2048 = 2080 coalitions for M = 16 patch
groups) and at 128, 256, 512 and 1024, on the records ``patch_lenses.py``
would score first. For each budget the study reports the Spearman rank
correlation of its 16-group ranking with the published budget's, and the
agreement with measured group patching the paper reports (zero and mean
replacement). The chosen budget is the smallest whose median rank
correlation with the published one is at least ``--target``, or the
published budget when none is.

The published-budget draws and the smaller budgets' draws are independent,
so a smaller budget is not scored against a reference that contains its own
samples. RISE's smaller budgets are nested prefixes of one second draw of
1024 masks; KernelSHAP's are separate runs of shap's sampler, each with its
own seed, whose enumerated coalitions (sizes 1 and 15, and 2 and 14 once the
budget covers them) are the same coalitions and are scored once.

    python3 experiments/simonsays/baseline_budget.py --repo Qwen/Qwen3-VL-8B-Instruct --n 30

Writes ``<out>/<model>__convergence.json`` and ``<out>/budget.json``, which
``patch_lenses.py`` reads for its default budgets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
import torch  # noqa: E402

BUDGETS = (128, 256, 512, 1024)


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(-np.asarray(a, float))).astype(float)
    rb = np.argsort(np.argsort(-np.asarray(b, float))).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/baselines_conv")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--cap", type=int, default=120)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--rise-masks", type=int, default=4000, help="RISE's published budget")
    ap.add_argument("--rise-grid", type=int, default=7)
    ap.add_argument("--rise-p", type=float, default=0.5)
    ap.add_argument("--budgets", type=int, nargs="*", default=list(BUDGETS))
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--minutes", type=float, default=40.0,
                    help="no new record is started after this many minutes of measuring")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-fast", action="store_true")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays
    from vlens.lens import attribution as AT, fitall as FA, readout, scores

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    TS = scores.TokenSets(m.tokenizer)
    LAST = m.n_layers - 1
    L = max(int(round(a.frac * LAST)), 1)
    Gn = a.groups
    n_shap = AT.shap_default_samples(Gn)
    budgets = sorted(b for b in a.budgets if b < max(a.rise_masks, n_shap))
    say(f"{name}: block {L}; RISE {a.rise_masks} vs {budgets}; KernelSHAP {n_shap} vs {budgets}")

    # the records patch_lenses.py scores, in the order it scores them
    recs = simonsays.load(require_image=True)
    cnt, scored = {}, []
    for r in recs:
        k = (r["family"], r["subcategory"], r["framing"])
        cnt[k] = cnt.get(k, 0) + 1
        if cnt[k] <= a.cap:
            scored.append(r)
    scored = simonsays.scorable([r for r in scored if r["kind"] in ("yesno", "letter", "test_real")])
    scored = FA.balanced_sample(scored, max(a.n * 2, a.n + 20), seed=0)

    part = f"{a.out}/{name}__convergence.rows.jsonl"
    rows = []
    if os.path.exists(part):
        rows = [json.loads(x) for x in open(part) if x.strip()]
        say(f"  resuming: {len(rows)} records already measured")
    done = {str(x["id"]) for x in rows}
    t0 = time.time()
    for r in scored:
        if len(rows) >= a.n:
            break
        if (time.time() - t0) / 60 > a.minutes:
            say(f"  {a.minutes:g} minutes spent; stopping at {len(rows)} records")
            break
        if str(r["id"]) in done:
            continue
        t_rec = time.time()
        img = C.load_image(r["image"], square=a.square)
        b = m.chat(r["question"], [img], system=r["system"], cue=r["cue"])
        pos = readout.image_positions(m, b)
        P = len(pos)
        if P < Gn:
            continue
        fn = scores.score_fn(TS, r["kind"], r.get("expected"))
        with torch.no_grad():
            _, H, _, lg0 = AT.states_at(m, b, [L])
        h = H[L]
        s0 = float(fn(None, lg0).float().cpu()[0])
        fn, s0 = scores.orient(fn, r["kind"], s0)
        if s0 <= 0:
            continue
        mu = h.mean(0, keepdim=True).expand_as(h)
        zero_h = torch.zeros_like(h)
        gsz = P // Gn
        gidx = [np.arange(gi * gsz, (gi + 1) * gsz if gi < Gn - 1 else P) for gi in range(Gn)]
        crc = zlib.crc32(str(r["id"]).encode())
        gsum = lambda v: np.array([np.asarray(v, float)[s].sum() for s in gidx])  # noqa: E731
        secs, fw = {}, {}
        with AT.SkipTo(m, b, L, fn=fn, pos=pos, h=h, base=mu, enabled=not a.no_fast, log=say) as bh:
            fast = bh is not b
            # what covering each group does, the truth the rankings are graded by
            truth = {}
            for tag, repl in (("mean", mu), ("zero", zero_h)):
                GX = []
                for sel in gidx:
                    x = h.clone(); x[sel] = repl[sel]
                    GX.append(x)
                with torch.no_grad():
                    truth[tag] = s0 - AT.hooked_axis_many(m, bh, fn, L, pos, GX, batch=a.batch, log=say)

            # KernelSHAP: the published budget, then each smaller one with its own seed
            cache, ks = {}, {}
            for j, nb in enumerate([n_shap, *budgets]):
                t_ = time.time()
                phi, info = AT.kernel_shap_groups(m, bh, fn, L, pos, h, mu, gidx, nsamples=nb,
                                                  rng=np.random.default_rng([a.seed, crc, 20 + j]),
                                                  batch=a.batch, log=say, cache=cache)
                ks[nb] = phi
                secs[f"kernel_shap|{nb}"] = time.time() - t_
                fw[f"kernel_shap|{nb}"] = dict(new=info["forwards"], coalitions=info["nsamples"],
                                               sizes_enumerated=info["sizes_enumerated"])

            # RISE: the published budget from one draw, the smaller ones as prefixes of another
            rows_, cols_ = readout.image_grid(m, b, P)
            rs = {}
            t_ = time.time()
            Mref = AT.rise_masks(rows_, cols_, a.rise_masks, s=a.rise_grid, p=a.rise_p,
                                 rng=np.random.default_rng([a.seed, crc, 1]))
            fref = AT.rise_scores(m, bh, fn, L, pos, h, mu, Mref, batch=a.batch, log=say)
            rs[a.rise_masks] = AT.rise(Mref, fref, p=a.rise_p)
            secs[f"rise|{a.rise_masks}"] = time.time() - t_
            if budgets:
                t_ = time.time()
                Mb = AT.rise_masks(rows_, cols_, max(budgets), s=a.rise_grid, p=a.rise_p,
                                   rng=np.random.default_rng([a.seed, crc, 11]))
                fb = AT.rise_scores(m, bh, fn, L, pos, h, mu, Mb, batch=a.batch, log=say)
                for nb in budgets:
                    rs[nb] = AT.rise(Mb[:nb], fb[:nb], p=a.rise_p)
                secs["rise|second draw"] = time.time() - t_
        grp = {"kernel_shap": {nb: np.asarray(v, float) for nb, v in ks.items()},
               "rise": {nb: gsum(v) for nb, v in rs.items()}}
        ref = {"kernel_shap": n_shap, "rise": a.rise_masks}
        row = dict(id=r["id"], family=r["family"], kind=r["kind"], n_patches=int(P), s0=s0,
                   fast=bool(fast), secs=secs, forwards=fw, wall=time.time() - t_rec,
                   efficiency=dict(sum=float(ks[n_shap].sum()),
                                   delta=float(cache[np.ones(Gn, np.int8).tobytes()]
                                               - cache[np.zeros(Gn, np.int8).tobytes()])),
                   rho_vs_published={}, truth_agreement={}, groups={})
        for meth, byb in grp.items():
            for nb, v in byb.items():
                key = f"{meth}|{nb}"
                row["groups"][key] = v.tolist()
                if nb != ref[meth]:
                    row["rho_vs_published"][key] = spearman(v, byb[ref[meth]])
                for tag in ("zero", "mean"):
                    row["truth_agreement"][f"{tag}|{key}"] = spearman(v, truth[tag])
        rows.append(row)
        with open(part, "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        say(f"  {len(rows)}/{a.n} {r['id']} ({row['wall']:.0f} s, fast {fast}): rho vs published "
            + ", ".join(f"{k} {v:.2f}" for k, v in row["rho_vs_published"].items()))

    if not rows:
        say("nothing measured"); return
    # the summary, and the choice
    summ = dict(repo=a.repo, block=L, depth=L / LAST, n=len(rows), groups=Gn, target=a.target,
                published=dict(rise=a.rise_masks, kernel_shap=n_shap), budgets=budgets,
                rise=dict(grid=a.rise_grid, p=a.rise_p), occlude="per-picture mean state",
                fast_path=float(np.mean([x["fast"] for x in rows])), per_method={})
    chosen = {}
    for meth, pub in (("kernel_shap", n_shap), ("rise", a.rise_masks)):
        per = {}
        for nb in [*budgets, pub]:
            key = f"{meth}|{nb}"
            rho = np.array([x["rho_vs_published"].get(key, np.nan) for x in rows], float)
            tz = np.array([x["truth_agreement"][f"zero|{key}"] for x in rows], float)
            tm = np.array([x["truth_agreement"][f"mean|{key}"] for x in rows], float)
            per[str(nb)] = dict(
                rho_median=None if nb == pub else float(np.nanmedian(rho)),
                rho_mean=None if nb == pub else float(np.nanmean(rho)),
                rho_q25=None if nb == pub else float(np.nanpercentile(rho, 25)),
                rho_min=None if nb == pub else float(np.nanmin(rho)),
                agree_zero=float(np.nanmean(tz)), agree_mean=float(np.nanmean(tm)),
                secs=float(np.mean([x["secs"].get(key, np.nan) for x in rows]))
                if meth == "kernel_shap" or nb == pub else None,
                forwards=nb if meth == "rise" else float(np.mean(
                    [x["forwards"][key]["coalitions"] for x in rows])))
        ok = [nb for nb in budgets if per[str(nb)]["rho_median"] >= a.target]
        chosen[meth] = min(ok) if ok else pub
        summ["per_method"][meth] = per
    eff = [abs(x["efficiency"]["sum"] - x["efficiency"]["delta"]) for x in rows]
    summ["kernel_shap_efficiency_max_abs_error"] = float(max(eff))
    summ["chosen"] = chosen
    walls = [x["wall"] for x in rows]
    summ["secs_per_record"] = float(np.mean(walls))

    def line(meth, pub):
        c = chosen[meth]
        per = summ["per_method"][meth]
        if c == pub:
            return (f"{'RISE' if meth == 'rise' else 'KernelSHAP'} is run at its published budget "
                    f"({pub}); no smaller budget reached a median rank correlation of {a.target:g} "
                    f"with it")
        return (f"{'RISE' if meth == 'rise' else 'KernelSHAP'} is run at {c} "
                f"{'masks' if meth == 'rise' else 'coalitions'} rather than the published {pub}: "
                f"on {len(rows)} records of {name} its 16-group ranking has a median Spearman "
                f"correlation of {per[str(c)]['rho_median']:.2f} with the published budget's, and "
                f"its agreement with measured group patching (zero replacement) is "
                f"{per[str(c)]['agree_zero']:.3f} against {per[str(pub)]['agree_zero']:.3f}")
    summ["caption_note"] = (line("rise", a.rise_masks) + ". " + line("kernel_shap", n_shap)
                            + ". Absent patches take the per-picture mean state; KernelSHAP's "
                            "players are the 16 patch groups.")
    json.dump(dict(summary=summ, rows=rows), open(f"{a.out}/{name}__convergence.json", "w"),
              default=float, indent=1)
    json.dump(dict(rise_masks=int(chosen["rise"]), shap_samples=int(chosen["kernel_shap"]),
                   groups=Gn, source=f"{name}__convergence.json", n=len(rows), target=a.target,
                   note=summ["caption_note"]),
              open(f"{a.out}/budget.json", "w"), indent=1)
    if os.path.exists(part):
        os.remove(part)

    say(f"\n  {len(rows)} records, {summ['secs_per_record']:.0f} s each, fast path on "
        f"{summ['fast_path']:.0%} of them")
    for meth, per in summ["per_method"].items():
        say(f"  {meth}")
        say(f"    {'budget':>7}{'rho med':>9}{'rho q25':>9}{'rho min':>9}{'agree 0':>9}{'agree mu':>9}")
        for nb, v in per.items():
            f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
            say(f"    {nb:>7}{f(v['rho_median']):>9}{f(v['rho_q25']):>9}{f(v['rho_min']):>9}"
                f"{v['agree_zero']:>9.3f}{v['agree_mean']:>9.3f}")
    say(f"  KernelSHAP efficiency: largest |sum(phi) - (f(x) - f(mean))| = "
        f"{summ['kernel_shap_efficiency_max_abs_error']:.2e}")
    say(f"  chosen: RISE {chosen['rise']}, KernelSHAP {chosen['kernel_shap']}")
    say("  " + summ["caption_note"])
    say("DONE baseline_budget")


if __name__ == "__main__":
    main()
