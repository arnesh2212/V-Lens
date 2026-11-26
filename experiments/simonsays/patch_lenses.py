# SPDX-License-Identifier: Apache-2.0
"""Rank image patches with every instrument, then test each ranking causally.

Each lens scores a patch by reading it, minus its reading of the replacement.
V-Write scores it by a path-integrated gradient. Attribution patching is
measured beside it, and, when ``--rankings`` names them, so are the other
published attribution methods, each as its paper specifies:

* ``ig_zero``: integrated gradients from a zero state at the block, and
  ``ig_black``: integrated gradients from a black image in pixel space
  through the whole model (the paper's baseline for images), both at
  ``--ig-steps`` midpoints, with completeness logged per record;
* ``ig_mean32``: V-Write's own path at ``--ig-mean-steps``, not a baseline;
* ``attr_patch``: attribution patching (AtP; Nanda 2023, Kramar et al. 2024
  Eq. 4) from the per-picture mean, without AtP*'s GradDrop;
* ``grad_x_act``, and ``act_patch`` (exact, one patch at a time, from the mean);
* ``rise``: Petsiuk et al. (2018), s x s binary grids at p = 0.5, upsampled
  with random shifts, saliency normalised by the expected mask;
* ``kernel_shap``: Lundberg and Lee (2017) as the shap package's
  KernelExplainer runs it (kernel-weighted coalition sizes, paired samples,
  small sizes enumerated, efficiency-constrained least squares), with the
  ``--groups`` patch groups as the players;
* ``gradcam``: Grad-CAM at the vision tower's last block.

RISE and KernelSHAP replace an absent patch's state by the per-picture mean,
as AtP and activation patching do. Their budgets default to the ones the
convergence study (``baseline_budget.py``) chose, read from
``--budget-file``, or to the published ones when there is none. Every
ranking is then checked by replacing the patches it puts first and
measuring the fall in the answer score, and graded against the measured
effect of covering each group.

    python3 experiments/simonsays/patch_lenses.py --repo Qwen/Qwen3-VL-8B-Instruct --n 0
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
from PIL import Image  # noqa: E402

KS = (0.05, 0.10, 0.20, 0.30, 0.50)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(-a)).astype(float)
    rb = np.argsort(np.argsort(-b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/patchlenses")
    ap.add_argument("--n", type=int, default=600, help="items to score; 0 means every one")
    ap.add_argument("--cap", type=int, default=120)
    ap.add_argument("--frac", type=float, default=0.5, help="depth the ranking is taken at")
    ap.add_argument("--corpus", type=int, default=500)
    ap.add_argument("--heldout", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tokens-per-step", type=int, default=2 ** 16)
    ap.add_argument("--jac-prompts", type=int, default=12)
    ap.add_argument("--latent-captions", type=int, default=5000)
    ap.add_argument("--fit-batch", type=int, default=8)
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--steps-ig", type=int, default=8, help="path-integration steps for V-Write")
    ap.add_argument("--ig-steps", type=int, default=64,
                    help="path-integration steps for integrated gradients, from a zero state "
                         "(ig_zero) and from a black image (ig_black); the paper uses 20 to 300")
    ap.add_argument("--ig-mean-steps", type=int, default=32,
                    help="steps for V-Write's own path taken again (ig_mean32, which keeps its "
                         "name whatever this is) to show its eight have converged")
    ap.add_argument("--rise-masks", "--masks", dest="rise_masks", default="auto",
                    help="RISE masks; 'auto' is the convergence study's choice from "
                         "--budget-file, else the published 4000")
    ap.add_argument("--rise-grid", type=int, default=7, help="RISE's s: the mask grid is s x s")
    ap.add_argument("--rise-p", type=float, default=0.5, help="RISE's keep probability")
    ap.add_argument("--shap-samples", default="auto",
                    help="KernelSHAP coalitions; 'auto' is the convergence study's choice from "
                         "--budget-file, else shap's 2M + 2048")
    ap.add_argument("--budget-file", default="results/baselines_conv/budget.json")
    ap.add_argument("--no-fast", action="store_true",
                    help="run the whole model for every intervention at the block instead of "
                         "the checked fast path that starts at the block")
    ap.add_argument("--act-batch", type=int, default=None,
                    help="forwards per batch for activation patching one patch at a time and "
                         "for the RISE masks; --int-batch when not given")
    ap.add_argument("--groups", type=int, default=16,
                    help="patch groups whose true effect is measured, so every method can be "
                         "graded against what covering that part of the picture actually does")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--int-batch", type=int, default=32)
    ap.add_argument("--rankings", nargs="*", default=None,
                    help="which rankings to measure; the depth atlas asks only for the "
                         "causal ones, and then nothing has to be fitted at all. The published "
                         "attribution methods (ig_zero ig_black ig_mean32 grad_x_act act_patch "
                         "rise kernel_shap gradcam) are measured only when named here; naming "
                         "ig_zero brings ig_black with it")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays
    from vlens.lens import attribution as AT, baselines as B, fitall as FA, patchsumm as PS, readout, scores
    from vlens.lens.hooks import ActivationRecorder

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    LAST = m.n_layers - 1
    L = max(int(round(a.frac * LAST)), 1)
    rng = np.random.default_rng(a.seed)
    # a depth atlas runs this at several depths into one directory, so anything
    # off the default depth carries it in the name and nothing is overwritten
    stem = f"{name}__patchlenses" if abs(a.frac - 0.5) < 1e-9 else \
        f"{name}__d{int(round(a.frac * 100))}__patchlenses"
    say(f"{name}: ranking image patches at block {L} of {m.n_layers} (depth {L / LAST:.2f})")

    recs = simonsays.load(require_image=True)
    cnt, spare, scored = {}, [], []
    for r in recs:
        k = (r["family"], r["subcategory"], r["framing"])
        cnt[k] = cnt.get(k, 0) + 1
        (spare if cnt[k] > a.cap else scored).append(r)
    scored = simonsays.scorable([r for r in scored
                                 if r["kind"] in ("yesno", "letter", "test_real")])

    # every lens, fitted once on records the behaviour pass never scored
    ALL = ["v_write", "attr_patch", "ig_zero", "ig_black", "ig_mean32", "grad_x_act", "act_patch", "rise",
           "kernel_shap", "gradcam", "v_read_cross", "tuned_cross", "v_read",
           "tuned", "jacobian", "latent", "logit", "random"]
    #: the published attribution methods cost several hundred forwards a
    #: record between them, so they are measured when asked for by name and a
    #: run that names nothing measures what it always has
    PUBLISHED = {"ig_zero", "ig_black", "ig_mean32", "grad_x_act", "act_patch", "rise",
                 "kernel_shap", "gradcam"}
    asked = set(a.rankings or ())
    # integrated gradients from the black image is measured only when named:
    # in bfloat16 through the vision tower its attributions do not yet sum to
    # the effect they explain, so no default run includes it
    rankings = ([r for r in ALL if r in asked] if a.rankings
                else [r for r in ALL if r not in PUBLISHED])
    if not rankings:
        say("no rankings asked for"); return
    #: what has to be fitted for the rankings asked for, and nothing more
    LENSES = {"v_read", "tuned", "jacobian", "latent"}
    want = tuple(r for r in rankings if r in LENSES)
    maps, latent, FIT, meta = {}, None, set(), {}
    if want:
        maps, latent, FIT, meta = FA.fit_all(
            m, spare, C.load_image, layers=[L], corpus=a.corpus, heldout=a.heldout,
            square=a.square, steps=a.steps, tokens_per_step=a.tokens_per_step,
            jac_prompts=a.jac_prompts, latent_captions=a.latent_captions,
            fit_batch=a.fit_batch, want=want, log=say)

    # the cross-position map, which is the one built for this question
    if "v_read_cross" in rankings or "tuned_cross" in rankings:
        fit_items = B.corpus_items(spare, C.load_image, n=a.corpus, square=a.square, seed=0)
        held_cross = B.corpus_items(spare, C.load_image, n=a.heldout, square=a.square, seed=7)
        FIT |= B.fit_ids(fit_items, held_cross)
        maps["v_read_cross"], cross_lr = B.pick_cross(
            m, fit_items, held_cross, [L], fit_batch=a.fit_batch,
            tokens_per_step=a.tokens_per_step, log=say)
        # the same pairs and optimiser, target read at the answer spot
        maps["tuned_cross"] = B.fit_tuned_lens_cross(
            m, B.CachedPairs(B.ImageToAnswerStream(m, fit_items, batch=a.fit_batch, seed=1),
                             [L], log=say),
            layers=[L], steps=a.steps, tokens_per_step=a.tokens_per_step, lr=cross_lr, log=say)
        maps = {k: v for k, v in maps.items() if k in rankings}
        say(f"  cross-position map fitted; {len(FIT)} records are now off limits for scoring")

    scored = [r for r in scored if str(r["id"]) not in FIT]
    # spread across the datasets rather than taking whichever family is
    # largest: five went into this benchmark and all five should be in a result
    scored = FA.balanced_sample(scored, a.n, seed=0) if a.n else scored
    say(f"{len(scored)} records, {len(rankings)} rankings")

    # each record's measurements are appended as they are made, so a run that
    # is cut short by a full card resumes where it stopped instead of paying
    # for the whole loop again
    part = f"{a.out}/{stem}.rows.jsonl"
    rows, seen_ids = [], set()
    if os.path.exists(part):
        with open(part) as fh:
            for line in fh:
                try:
                    r_ = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if r_.get("impl") != 2 and PUBLISHED & set(rankings):
                    continue          # measured by the earlier baselines, not these
                rows.append(r_); seen_ids.add(str(r_.get("id")))
        say(f"  resuming: {len(rows)} records already measured")
    act_batch = a.act_batch or a.int_batch
    # the budgets RISE and KernelSHAP are run at: the convergence study's
    # choice where there is one for these groups, the published ones otherwise
    G_ = a.groups or 16
    chosen, src = {}, "published"
    if "auto" in (str(a.rise_masks), str(a.shap_samples)) and os.path.exists(a.budget_file):
        chosen = json.load(open(a.budget_file))
        if int(chosen.get("groups", G_)) != G_:
            chosen = {}
        else:
            src = a.budget_file
    n_rise = int(chosen.get("rise_masks", 4000)) if str(a.rise_masks) == "auto" else int(a.rise_masks)
    n_shap = (int(chosen.get("shap_samples", AT.shap_default_samples(G_)))
              if str(a.shap_samples) == "auto" else int(a.shap_samples))
    impl = dict(version=2, rise=dict(masks=n_rise, grid=a.rise_grid, p=a.rise_p, occlude="mean",
                                     normalise="E[M]=p (Eq. 6)"),
                kernel_shap=dict(samples=n_shap, players=f"{G_} patch groups", occlude="mean",
                                 sampler="shap KernelExplainer", l1_reg=False,
                                 efficiency_constraint=True),
                ig=dict(steps=a.ig_steps, zero="zero state at the block",
                        black="black image, pixel space"),
                ig_mean32=dict(steps=a.ig_mean_steps),
                gradcam="vision tower, last block", attr_patch="AtP (no GradDrop)",
                budget_source=src)
    if {"rise", "kernel_shap"} & set(rankings):
        say(f"  RISE at {n_rise} masks, KernelSHAP at {n_shap} coalitions of {G_} groups ({src})")
    meta = dict(meta, baselines=impl)
    t0 = time.time()
    for r in scored:
        if str(r["id"]) in seen_ids:
            continue
        t_rec = time.time()
        img = C.load_image(r["image"], square=a.square)
        b = m.chat(r["question"], [img], system=r["system"], cue=r["cue"])
        pos = readout.image_positions(m, b)
        P = len(pos)
        if P < 16:
            continue
        fn = scores.score_fn(TS, r["kind"], r.get("expected"))
        with torch.no_grad():
            _, H, _, lg0 = AT.states_at(m, b, [L])
        h = H[L]
        s0 = float(fn(None, lg0).float().cpu()[0])
        fn, s0 = scores.orient(fn, r["kind"], s0)
        if s0 <= 0:
            continue              # read toward the answer the model gave, and
                                  # skip the records where there is not one
        mu = h.mean(0, keepdim=True).expand_as(h)
        zero_h = torch.zeros_like(h)
        # the groups whose covering is measured, which are also KernelSHAP's players
        gsz = P // G_
        gidx = [np.arange(gi * gsz, (gi + 1) * gsz if gi < G_ - 1 else P) for gi in range(G_)]

        attr = {"random": rng.standard_normal(P).astype(np.float32)}
        secs, complete, info = {}, {}, {}
        tick = time.time

        # what needs the vision tower or the early blocks runs on the whole model
        if "gradcam" in rankings:
            t_ = tick()
            attr["gradcam"] = AT.grad_cam_vision(m, b, fn, pos, log=say)
            secs["gradcam"] = tick() - t_
        if "ig_black" in rankings:
            t_ = tick()
            black = Image.new("RGB", img.size, (0, 0, 0))
            b0 = m.chat(r["question"], [black], system=r["system"], cue=r["cue"])
            attr["ig_black"], f_black = AT.ig_pixels(m, b, b0, fn, pos, steps=a.ig_steps, log=say)
            complete["ig_black"] = dict(sum=float(np.sum(attr["ig_black"])), delta=s0 - f_black)
            secs["ig_black"] = tick() - t_

        # everything else intervenes at block L, and starts there
        with AT.SkipTo(m, b, L, fn=fn, pos=pos, h=h, base=mu, enabled=not a.no_fast,
                       log=say) as bh:
            fast = bh is not b
            if "v_write" in rankings:
                t_ = tick()
                attr["v_write"] = AT.path_reading(m, bh, fn, L, pos, h, mu, steps=a.steps_ig, log=say)
                secs["v_write"] = tick() - t_
            if "ig_mean32" in rankings:
                t_ = tick()
                attr["ig_mean32"] = AT.path_reading(m, bh, fn, L, pos, h, mu, steps=a.ig_mean_steps,
                                                    log=say)
                secs["ig_mean32"] = tick() - t_
            if "ig_zero" in rankings:
                t_ = tick()
                attr["ig_zero"] = AT.path_reading(m, bh, fn, L, pos, h, zero_h, steps=a.ig_steps,
                                                  log=say)
                secs["ig_zero"] = tick() - t_
            # completeness (Sundararajan et al. 2017, Prop. 1): the attributions'
            # sum against f(x) - f(baseline), the baseline scored with the same hook
            need = [k for k in ("v_write", "ig_mean32", "ig_zero") if k in attr]
            if need:
                with torch.no_grad():
                    fx_, fmu_, f0_ = AT.hooked_axis(m, bh, fn, L, pos, torch.stack([h, mu, zero_h])
                                                    .to(m.device)).float().cpu().tolist()
                for k in need:
                    complete[k] = dict(sum=float(np.sum(attr[k])),
                                       delta=fx_ - (f0_ if k == "ig_zero" else fmu_))
            # attribution patching and gradient times activation share the one
            # gradient at the observed state
            if "attr_patch" in rankings or "grad_x_act" in rankings:
                t_ = tick()
                g = AT.grad_at(m, bh, fn, L, pos, h)
                if "attr_patch" in rankings:
                    attr["attr_patch"] = AT.one_point(m, bh, fn, L, pos, h, mu, grad=g)
                if "grad_x_act" in rankings:
                    attr["grad_x_act"] = AT.grad_x_act(m, bh, fn, L, pos, h, grad=g)
                secs["attr_patch"] = tick() - t_
            if "act_patch" in rankings:
                # from the mean, the replacement attribution patching is the
                # one-point estimate of; from zero it would be the grading
                # replacement itself, one patch at a time
                t_ = tick()
                attr["act_patch"] = AT.act_patch(m, bh, fn, L, pos, h, s0, base=mu,
                                                  batch=act_batch, log=say)
                secs["act_patch"] = tick() - t_
            # the masks and coalitions depend on the record and the seed alone,
            # so a resumed run draws the same ones it would have drawn the first time
            crc = zlib.crc32(str(r["id"]).encode())
            if "rise" in rankings:
                t_ = tick()
                rows_, cols_ = readout.image_grid(m, b, P)
                Mr = AT.rise_masks(rows_, cols_, n_rise, s=a.rise_grid, p=a.rise_p,
                                   rng=np.random.default_rng([a.seed, crc, 1]))
                fr = AT.rise_scores(m, bh, fn, L, pos, h, mu, Mr, batch=act_batch, log=say)
                attr["rise"] = AT.rise(Mr, fr, p=a.rise_p)
                secs["rise"] = tick() - t_
            if "kernel_shap" in rankings:
                t_ = tick()
                phi, kinfo = AT.kernel_shap_groups(m, bh, fn, L, pos, h, mu, gidx, nsamples=n_shap,
                                                   rng=np.random.default_rng([a.seed, crc, 2]),
                                                   batch=act_batch, log=say)
                attr["kernel_shap"] = AT.spread_groups(phi, gidx, P)
                complete["kernel_shap"] = dict(sum=float(phi.sum()), delta=kinfo["fx"] - kinfo["fnull"])
                info["kernel_shap"] = dict(coalitions=kinfo["nsamples"], forwards=kinfo["forwards"],
                                           sizes_enumerated=kinfo["sizes_enumerated"])
                secs["kernel_shap"] = tick() - t_
            hd = h.to(m.device)
            mud = mu.to(m.device)
            with torch.no_grad():
                # a lens reads one patch at a time, so its attribution is the
                # reading of that patch minus the reading of its replacement, which
                # is also its path-integrated form
                for k, M in maps.items():
                    A_, b_ = M[L]
                    v = hd @ A_.T.to(m.device) + b_.to(m.device)
                    vb = mud @ A_.T.to(m.device) + b_.to(m.device)
                    attr[k] = (fn(None, m.unembed(v.to(torch.bfloat16)))
                               - fn(None, m.unembed(vb.to(torch.bfloat16)))).float().cpu().numpy()
                if "logit" in rankings:
                    attr["logit"] = (fn(None, m.unembed(hd.to(torch.bfloat16)))
                                     - fn(None, m.unembed(mud.to(torch.bfloat16)))).float().cpu().numpy()
                if latent is not None and "latent" in rankings:
                    tp = max(latent.meta.get("temp", 0.05), 1e-6)
                    q = latent.similarity_logits(hd.float(), L) / tp
                    qb = latent.similarity_logits(mud.float(), L) / tp
                    attr["latent"] = (fn(None, q) - fn(None, qb)).float().cpu().numpy()

            # ties (KernelSHAP's within a group, Grad-CAM's zeros after the ReLU)
            # are broken at random, not by position
            trng = np.random.default_rng([a.seed, crc, 3])
            plan, X = [], []
            t_ = tick()
            for tag, repl in (("mean", mu), ("zero", zero_h)):
                for meth in rankings:
                    if meth not in attr:
                        continue
                    o = np.lexsort((trng.random(P), -np.asarray(attr[meth], float)))
                    for k in KS:
                        nk = max(int(round(k * P)), 1)
                        x = h.clone(); x[o[:nk]] = repl[o[:nk]]
                        plan.append((tag, meth, k)); X.append(x)
            with torch.no_grad():
                got = AT.hooked_axis_many(m, bh, fn, L, pos, X, batch=a.int_batch, log=say)
            # Cover each group and measure what the answer does. The mean
            # replacement is the counterfactual V-Write integrates from, so it is a
            # sanity check; the zero replacement is independent of V-Write and is
            # the number to read, though integrated gradients and gradient times
            # activation start from it.
            truth_r = {}
            if a.groups and P >= a.groups:
                for tag, repl in (("mean", mu), ("zero", zero_h)):
                    GX = []
                    for sel in gidx:
                        x = h.clone(); x[sel] = repl[sel]
                        GX.append(x)
                    with torch.no_grad():
                        gafter = AT.hooked_axis_many(m, bh, fn, L, pos, GX, batch=a.int_batch, log=say)
                    truth = s0 - gafter
                    for mm in attr:
                        pred = np.array([np.asarray(attr[mm], float)[sel].sum() for sel in gidx])
                        truth_r[f"{tag}|{mm}"] = spearman(pred, truth)
            secs["grading"] = tick() - t_
        row = dict(id=r["id"], family=r["family"], kind=r["kind"], n_patches=int(P), s0=s0,
                   impl=2, fast=bool(fast), secs=secs, wall=time.time() - t_rec, completeness=complete, info=info,
                   truth_agreement=truth_r,
                   after={f"{tag}|{mm}|{k}": float(v)
                          for (tag, mm, k), v in zip(plan, got)},
                   agree={mm: spearman(np.asarray(attr["v_write"], float),
                                       np.asarray(attr[mm], float))
                          for mm in attr if mm != "v_write"} if "v_write" in attr else {})
        rows.append(row)
        if len(rows) <= 3:
            say(f"  record {r['id']}: {P} tokens, fast path {'on' if fast else 'off'}; seconds "
                + ", ".join(f"{k} {v:.1f}" for k, v in secs.items())
                + "".join(f"; {k} sums to {c['sum']:.3f} of {c['delta']:.3f}"
                          for k, c in complete.items()))
        with open(part, "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        if len(rows) % 50 == 0:
            d = rows[-1]["after"]
            drop = lambda mm: rows[-1]["s0"] - d.get(f"zero|{mm}|0.2", np.nan)
            say(f"  {len(rows)}/{len(scored)} ({(time.time() - t0) / 60:.1f} min) "
                f"last, a fifth of the patches gone: v_write {drop('v_write'):.2f} "
                f"tuned {drop('tuned'):.2f} tuned_cross {drop('tuned_cross'):.2f} "
                f"random {drop('random'):.2f}")

    if not rows:
        say("nothing scored"); return
    # recorded so that a depth atlas cannot mix runs that selected their
    # records by different rules and read the difference as depth
    summ = PS.summarise(rows, KS, rankings, repo=a.repo, block=L, depth=L / LAST,
                        oriented=True, meta=meta)
    # completeness: how far each method's sum is from the effect it should sum
    # to, relative to that effect (median over records), and the GPU seconds a
    # record cost each method
    comp = {}
    for k in sorted({k for x in rows for k in x.get("completeness", {})}):
        e = [abs(x["completeness"][k]["sum"] - x["completeness"][k]["delta"])
             / max(abs(x["completeness"][k]["delta"]), 1e-6)
             for x in rows if k in x.get("completeness", {})]
        comp[k] = dict(median_rel_error=float(np.median(e)), n=len(e))
    summ["completeness"] = comp
    summ["secs_per_record"] = {k: float(np.mean([x["secs"][k] for x in rows if k in x.get("secs", {})]))
                               for k in sorted({k for x in rows for k in x.get("secs", {})})}
    summ["secs_per_record_total"] = float(np.mean([x.get("wall", np.nan) for x in rows]))
    json.dump(dict(summary=summ, rows=rows), open(f"{a.out}/{stem}.json", "w"),
              default=float)
    if os.path.exists(part):
        os.remove(part)          # the finished file supersedes the running one
    PS.report(summ, KS, rankings, say=say)
    PS.report_sources(summ, rankings, say=say)
    if summ.get("graded_against_patching"):
        say("\n  graded against what covering each part of the picture actually does")
        say(f"  (rank agreement with the measured effect, {a.groups} groups per picture)")
        say(f"  {'method':<16}{'covered by the mean':>21}{'covered by zero':>18}")
        say("   the mean column is V-Write's own counterfactual, so read the other one")
        if {"ig_zero", "grad_x_act"} & set(rankings):
            say("   (zero is the baseline of integrated gradients and gradient x activation)")
        for mm in rankings:
            a_ = summ["graded_against_patching"].get(f"mean|{mm}")
            b_ = summ["graded_against_patching"].get(f"zero|{mm}")
            if a_ is not None or b_ is not None:
                fa = f"{a_:.2f}" if a_ is not None and np.isfinite(a_) else "-"
                fb = f"{b_:.2f}" if b_ is not None and np.isfinite(b_) else "-"
                say(f"    {mm:<15}{fa:>20}{fb:>18}")
    for k, c in comp.items():
        say(f"  completeness {k}: median |sum - effect| / |effect| = {c['median_rel_error']:.3f} "
            f"over {c['n']} records")
    say("  seconds a record: " + ", ".join(f"{k} {v:.1f}" for k, v in summ["secs_per_record"].items())
        + f"; all {summ['secs_per_record_total']:.1f}")
    say(f"\n  {len(summ['per_family'])} families kept for the paired test")
    say("DONE patch_lenses")


if __name__ == "__main__":
    main()
