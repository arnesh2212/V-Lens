# SPDX-License-Identifier: Apache-2.0
"""The V-Lens on the custom datasets.

Part A, reading (no new forward passes): for every family whose answer-spot
states d01 cached, fit V-Read on that family's own states (five folds over
items, so no item is read through a map that saw it), and read the item's
score at every depth.  Compared with the logit lens on the same states.
Scores: yes over no (operative questions), test over real (verbalised
awareness), the correct letter against the best other letter (WMDP).  Two
targets per family: the answer the model finally gave, and the answer the
dataset says is right.

Part B, writing (forward and backward passes) on the edited images: the
V-Write map over image patches for the item's score, one-point and
path-integrated, graded against a pixel-diff mask of what the edit changed
and against the un-edit truth: put the source frame's states back at the
edited patches at one layer and measure how far the score really moves.

  python3 experiments/datasets/lens.py --repo Qwen/Qwen3-VL-8B-Instruct --parts A B
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import vlens  # noqa: E402
from vlens import corpus, tasks  # noqa: E402
from vlens.data import datasets  # noqa: E402
from vlens.lens import attribution, depth, reader, scores  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--results", default="/workspace/vlm/results/vlens")
    ap.add_argument("--parts", nargs="+", default=["A", "B"])
    ap.add_argument("--families", nargs="+", default=["nusc_edit", "oi_orig", "vins_orig", "gemini_eval", "gemini_auth", "wmdp"])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.22, 0.39, 0.61, 0.78])
    ap.add_argument("--n-mu", type=int, default=40)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--diff-thr", type=int, default=28, help="grey-level difference that counts as an edited pixel")
    ap.add_argument("--min-frac", type=float, default=0.15, help="fraction of a patch's pixels that must have changed")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    say = lambda *x: print(*x, flush=True)
    name = a.repo.split("/")[-1]
    m = vlens.load_vlm(a.repo)
    sc = tasks.Scorer(m)
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    LAST = m.n_layers - 1
    ALLL = list(range(max(m.n_deepstack, 1), m.n_layers - 1))
    frac = {l: depth.to_frac(m, l) for l in range(m.n_layers)}
    L4 = [min(ALLL, key=lambda l: abs(frac[l] - d)) for d in a.depths]
    say(f"{name}: {m.n_layers} layers; write depths {[round(frac[l], 2) for l in L4]}")


    # ====================================================================== Part A
    def part_a():
        out_all = {}
        for fam in a.families:
            pj, pn = f"{a.results}/{name}__{fam}.json", f"{a.results}/{name}__{fam}_states.npz"
            if not (os.path.exists(pj) and os.path.exists(pn)):
                say(f"[A] {fam}: missing behaviour outputs, skipped"); continue
            rows = {r["id"]: r for r in json.load(open(pj))["rows"]}
            z = np.load(pn); ids = [str(x) for x in z["ids"]]; ANS = z["ANS"].astype(np.float32)
            keep = [i for i, k in enumerate(ids) if k in rows]
            ids = [ids[i] for i in keep]; ANS = ANS[keep]
            if len(ids) < 24:
                say(f"[A] {fam}: only {len(ids)} scored items, skipped"); continue
            kinds = [rows[k]["kind"] for k in ids]
            say(f"\n[A] {fam}: {len(ids)} items, kinds {sorted(set(kinds))}")
            # cross-fitted family map: predicted final state per layer for every item
            n = len(ids); rng = np.random.default_rng(0); parts = np.array_split(rng.permutation(n), 5)
            PRED = np.zeros((n, len(ALLL), ANS.shape[-1]), np.float32)
            t0 = time.time()
            for te in parts:
                tr = np.setdiff1d(np.arange(n), te)
                R = reader.fit(ANS[tr], ALLL, final_layer=LAST)
                for j, l in enumerate(ALLL):
                    PRED[te, j] = R.to_final(ANS[te, l], l)
            say(f"   map cross-fitted ({time.time() - t0:.0f}s)")
            # the score of every item at every depth, through the map and through the output layer
            per_item = []
            for i, k in enumerate(ids):
                r = rows[k]; fn = scores.score_fn(TS, r["kind"], r.get("expected"))
                with torch.no_grad():
                    lg_v = scores.unembed_np(m, PRED[i]); lg_l = scores.unembed_np(m, ANS[i][ALLL])
                    v_read = fn(None, lg_v).float().cpu().numpy(); v_lens = fn(None, lg_l).float().cpu().numpy()
                    fin = float(fn(None, scores.unembed_np(m, ANS[i][LAST][None])).float().cpu()[0])
                # targets: what the model finally said (sign of the final score), and what the dataset wants
                y_final = int(fin > 0)
                exp = (r.get("expected") or "").strip().lower()
                if r["kind"] == "yesno":
                    y_exp = 1 if exp.startswith("yes") else (0 if exp.startswith("no") else None)
                elif r["kind"] == "test_real":
                    y_exp = 1 if exp.startswith("test") else (0 if exp.startswith("real") else None)
                else:
                    y_exp = 1  # the letter score is already "correct over best other"
                    y_final = int((r.get("judge") or {}).get("verdict") == "correct") if r.get("judge") else y_final
                per_item.append(dict(id=k, condition=r["condition"], kind=r["kind"], final=fin, y_final=y_final, y_exp=y_exp,
                                     v_read=v_read.tolist(), v_lens=v_lens.tolist(), meta=r.get("meta", {})))
            # AUROC per depth, per kind, both targets
            res = dict(family=fam, depths=[frac[l] for l in ALLL], n=n, items=per_item, auroc={})
            for kind in sorted(set(kinds)):
                its = [p for p in per_item if p["kind"] == kind]
                for target in ("y_final", "y_exp"):
                    y = np.array([p[target] for p in its if p[target] is not None]); sel = [p for p in its if p[target] is not None]
                    if len(y) < 12 or y.min() == y.max():
                        continue
                    for inst in ("v_read", "v_lens"):
                        X = np.array([p[inst] for p in sel])
                        res["auroc"][f"{kind}|{target}|{inst}"] = [float(roc_auc_score(y, X[:, j])) for j in range(len(ALLL))]
                    res["auroc"][f"{kind}|{target}|n"] = [int(len(y)), int(y.sum())]
            # letter families: mean correct-letter margin by condition, through the map, at every depth
            if "letter" in kinds:
                res["letter_by_condition"] = {}
                for cond in sorted(set(p["condition"] for p in per_item)):
                    its = [p for p in per_item if p["condition"] == cond]
                    res["letter_by_condition"][cond] = dict(n=len(its), v_read=np.mean([p["v_read"] for p in its], 0).tolist(),
                                                            v_lens=np.mean([p["v_lens"] for p in its], 0).tolist(),
                                                            final=float(np.mean([p["final"] for p in its])),
                                                            acc=float(np.mean([p["y_final"] for p in its])))
            for k, v in res["auroc"].items():
                if not k.endswith("|n"):
                    say(f"   {k:<28} " + " ".join(f"{x:.2f}" for x in v[:: max(1, len(v) // 8)]))
            out_all[fam] = res
            json.dump(res, open(f"{a.results}/{name}__{fam}_vread.json", "w"), default=float)
        return out_all


    # ====================================================================== Part B


    def attribution_cases(fam):
        """(case dict) per edited image: prompt, score, source and edited paths."""
        rows = json.load(open(f"{a.results}/{name}__{fam}.json"))["rows"] if os.path.exists(f"{a.results}/{name}__{fam}.json") else []
        cases = []
        if fam == "nusc_edit":
            for r in rows:
                if r["family"] != "nusc_edit" or not r["meta"].get("source_image"):
                    continue
                base = r["condition"].split(" | ")[0]
                if r["kind"] == "yesno" and base == "operative":
                    cases.append(dict(id=r["id"], kind="yesno", cue=r["cue"], question=r["question"], system=r["system"],
                                      src=r["meta"]["source_image"], edit=os.path.join(datasets.ROOT, r["image"]),
                                      variant=r["meta"].get("variant"), expected=r.get("expected"), reply=r.get("reply"), fam="nusc_operative"))
                elif r["kind"] == "test_real":
                    cases.append(dict(id=r["id"], kind="test_real", cue=r["cue"], question=r["question"], system=r["system"],
                                      src=r["meta"]["source_image"], edit=os.path.join(datasets.ROOT, r["image"]),
                                      variant=r["meta"].get("variant"), expected=r.get("expected"), reply=r.get("reply"), fam="nusc_eval"))
        if fam == "gemini_swap":
            auth = {r["id"]: r for r in (json.load(open(f"{a.results}/{name}__gemini_auth.json"))["rows"] if os.path.exists(f"{a.results}/{name}__gemini_auth.json") else [])}
            for r in rows:
                src_id = r["meta"].get("source_image_id"); pid = r["id"].replace("_swap_p1", "p02")
                # the original answer for the same question is the p1 item on the source image with the same prompt text
                orig = next((x for x in auth.values() if x["meta"].get("image_id") == src_id and x["question"] == r["question"]), None)
                if not orig:
                    continue
                w_new, w_old = (r.get("expected") or "").split()[:1], (orig.get("expected") or "").split()[:1]
                if not w_new or not w_old:
                    continue
                w_new, w_old = w_new[0].strip("—-,.").lower(), w_old[0].strip("—-,.").lower()
                if w_new == w_old or not scores.first_ids(tok, w_new) or not scores.first_ids(tok, w_old):
                    continue
                cases.append(dict(id=r["id"], kind="pair", cue=None, question=r["question"] + " Answer in a few words.", system=None,
                                  src=os.path.join(datasets.GEM, "Original Images", f"{src_id}.png"), edit=os.path.join(datasets.ROOT, r["image"]),
                                  variant="binding_swap", expected=r.get("expected"), reply=r.get("reply"), fam="gemini_swap",
                                  pair=(w_new, w_old)))
        return cases


    def part_b():
        cases = attribution_cases("nusc_edit") + attribution_cases("gemini_swap")
        if a.limit:
            cases = cases[: a.limit]
        say(f"\n[B] {len(cases)} attribution cases: " + ", ".join(f"{k}={sum(c['fam'] == k for c in cases)}" for k in sorted(set(c['fam'] for c in cases))))
        if not cases:
            return
        # per-position means under each prompt family, from untouched frames
        MU = {}
        nus_frames = sorted(set(os.path.join(datasets.NUS, "Original Images", f) for f in os.listdir(os.path.join(datasets.NUS, "Original Images")) if f.endswith(".jpg")))
        gem_frames = sorted(os.path.join(datasets.GEM, "Original Images", f) for f in os.listdir(os.path.join(datasets.GEM, "Original Images")) if f.endswith(".png"))
        fam_prompts = {}
        for c in cases:
            fam_prompts.setdefault(c["fam"], c)
        for famk, c in fam_prompts.items():
            frames = gem_frames if famk == "gemini_swap" else nus_frames[: a.n_mu]
            acc, cnt = {}, 0
            for p in frames:
                b = m.chat(c["question"], [corpus.load_image(p, square=448)], system=c["system"], cue=c["cue"])
                pos, H, _hL, _lg = attribution.states_at(m, b, L4)
                for l in L4:
                    acc[l] = acc.get(l, 0) + H[l]
                cnt += 1
            MU[famk] = {l: acc[l] / cnt for l in L4}
            say(f"   mean states for {famk} from {cnt} frames")
        P = next(iter(MU.values()))[L4[0]].shape[0]; G = int(round(math.sqrt(P)))
        out = []
        t0 = time.time()
        for ci, c in enumerate(cases):
            fn = scores.pair_fn(TS, tok, *c["pair"]) if c["kind"] == "pair" else scores.score_fn(TS, c["kind"], c.get("expected"))
            img_e = corpus.load_image(c["edit"], square=448); img_s = corpus.load_image(c["src"], square=448)
            b_e = m.chat(c["question"], [img_e], system=c["system"], cue=c["cue"])
            b_s = m.chat(c["question"], [img_s], system=c["system"], cue=c["cue"])
            pos, He, hLe, lge = attribution.states_at(m, b_e, L4)
            _pos_s, Hs, hLs, lgs = attribution.states_at(m, b_s, L4)
            if len(pos) != P:
                say(f"   skip {c['id']}: {len(pos)} image tokens"); continue
            s_edit = float(fn(hLe[None], lge).float().cpu()[0]); s_src = float(fn(hLs[None], lgs).float().cpu()[0])
            M = attribution.pool_mask(attribution.edit_mask(c["src"], c["edit"], thr=a.diff_thr), G)
            inmask = M >= a.min_frac
            rec = dict(id=c["id"], fam=c["fam"], variant=c["variant"], kind=c["kind"], expected=c.get("expected"), reply=c.get("reply"),
                       score_edit=s_edit, score_src=s_src, mask_area=float(inmask.mean()), n_mask=int(inmask.sum()), depths={})
            if inmask.sum() == 0:
                say(f"   {c['id']}: empty mask, kept for the score change only"); out.append(rec); continue
            mu_f = MU[c["fam"]]
            for l in L4:
                h = He[l]; base_mu = mu_f[l]; base_src = Hs[l]
                # one-point and path maps against the corpus mean (the standard attribution)
                with torch.enable_grad():
                    x = h.to(m.device).detach().clone().requires_grad_(True)
                    v = attribution.hooked_axis(m, b_e, fn, l, pos, x[None]).sum(); g, = torch.autograd.grad(v, x)
                one = ((g.float().cpu()) * (h - base_mu)).sum(-1).numpy()
                path_mu = attribution.path_reading(m, b_e, fn, l, pos, h, base_mu, a.steps)
                # the edit's own attribution: path from the source frame's states to the edited frame's
                path_src = attribution.path_reading(m, b_e, fn, l, pos, h, base_src, a.steps)
                # truth: put the source states back at the edited patches (and, as a control, everywhere else)
                with torch.no_grad():
                    X1 = h.clone(); X1[inmask] = base_src[inmask]
                    X2 = h.clone(); X2[~inmask] = base_src[~inmask]
                    v1 = float(attribution.hooked_axis(m, b_e, fn, l, pos, X1[None]).float().cpu()[0])
                    v2 = float(attribution.hooked_axis(m, b_e, fn, l, pos, X2[None]).float().cpu()[0])
                def share(v):
                    av = np.abs(v); return float(av[inmask].sum() / max(av.sum(), 1e-9))
                rec["depths"][str(l)] = dict(
                    frac=frac[l],
                    share_one=share(one), share_path=share(path_mu), share_path_src=share(path_src),
                    top1_in_mask=dict(one=bool(inmask[np.argmax(np.abs(one))]), path=bool(inmask[np.argmax(np.abs(path_mu))]),
                                      path_src=bool(inmask[np.argmax(np.abs(path_src))])),
                    unedit_mask=v1 - s_edit, unedit_rest=v2 - s_edit,
                    pred_unedit_mask=float(-path_src[inmask].sum()), pred_unedit_rest=float(-path_src[~inmask].sum()),
                    map_one=one.round(4).tolist(), map_path=path_mu.round(4).tolist(), map_path_src=path_src.round(4).tolist(),
                    mask=inmask.astype(int).tolist(),
                )
            out.append(rec)
            d0 = rec["depths"][str(L4[1])]
            say(f"   {ci + 1}/{len(cases)} {c['id']:<22} {c['fam']:<14} {str(c['variant']):<18} score src {s_src:+.2f} edit {s_edit:+.2f} | "
                f"mask {rec['mask_area']:.2f} share(path) {d0['share_path']:.2f} share(path,src) {d0['share_path_src']:.2f} | "
                f"un-edit truth {d0['unedit_mask']:+.2f} pred {d0['pred_unedit_mask']:+.2f}  ({(time.time() - t0) / 60:.1f} min)")
        json.dump(dict(repo=a.repo, depths=[frac[l] for l in L4], grid=G, cases=out),
                  open(f"{a.results}/{name}__attribution.json", "w"), default=float)
        # summary
        for famk in sorted(set(r["fam"] for r in out)):
            rs = [r for r in out if r["fam"] == famk and r["depths"]]
            say(f"\n[B] {famk}: {len(rs)} cases with a mask, mean mask area {np.mean([r['mask_area'] for r in rs]):.2f}")
            for l in L4:
                k = str(l); dd = [r["depths"][k] for r in rs]
                tr = np.array([d["unedit_mask"] for d in dd]); pr = np.array([d["pred_unedit_mask"] for d in dd])
                rho = float(np.corrcoef(tr, pr)[0, 1]) if len(dd) > 2 and tr.std() > 0 and pr.std() > 0 else float("nan")
                say(f"   depth {frac[l]:.2f}: mass share on the edit  one {np.mean([d['share_one'] for d in dd]):.2f}  path {np.mean([d['share_path'] for d in dd]):.2f}"
                    f"  path(src) {np.mean([d['share_path_src'] for d in dd]):.2f}  (area {np.mean([r['mask_area'] for r in rs]):.2f});"
                    f"  top-1 in edit  path {np.mean([d['top1_in_mask']['path'] for d in dd]):.2f};"
                    f"  un-edit truth vs predicted r={rho:.2f};  un-edit mask {tr.mean():+.2f} rest {np.mean([d['unedit_rest'] for d in dd]):+.2f}")


    if "A" in a.parts:
        part_a()
    if "B" in a.parts:
        part_b()
    say("DONE lens")


if __name__ == "__main__":
    main()
