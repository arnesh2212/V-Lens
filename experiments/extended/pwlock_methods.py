from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

LETTERS = "ABCD"
THR = 0.1


def summary(w, cov):
    from vlens.lens.metrics import auroc
    on = cov > THR
    if not on.any() or on.all():
        return dict(mass=None, hit1=None, auroc=None, chance=float(on.mean()))
    pos = np.clip(w, 0, None)
    return dict(mass=float(pos[on].sum() / max(pos.sum(), 1e-12)), hit1=bool(on[int(np.argmax(w))]),
                auroc=float(auroc(on.astype(int), w)), chance=float(on.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--limit", type=int, default=0, help="first n questions (smoke)")
    ap.add_argument("--conditions", nargs="*",
                    default=["trained", "grey", "new", "unrelated", "clean"])
    ap.add_argument("--methods", nargs="*",
                    default=["v_write", "ig_zero", "ig_black", "grad_x_act", "attr_patch", "act_patch",
                             "gradcam", "kernel_shap", "rise", "random"])
    ap.add_argument("--heavy-on", nargs="*", default=["trained", "grey"],
                    help="conditions on which KernelSHAP and RISE run")
    ap.add_argument("--shap-samples", type=int, default=2080)
    ap.add_argument("--rise-masks", type=int, default=1000)
    ap.add_argument("--heavy-limit", type=int, default=40, help="questions on which KernelSHAP and RISE run")
    ap.add_argument("--ig-black-steps", type=int, default=8)
    ap.add_argument("--no-f5", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import torch
    from PIL import Image
    import vlens
    from vlens import corpus as C
    from vlens.data import pwlock as P
    from vlens.lens import attribution as AT, readout, scores
    from experiments.pwlock.evaluate import watermark_layer, composite

    S = a.square
    test = P.test_items()
    clean_by = {x["group"]: x for x in test if not x["marked"]}
    marked_by = {}
    for x in test:
        if x["marked"]:
            marked_by.setdefault(x["group"], {})[x["variant"]] = x
    groups = sorted(g for g in clean_by if g in marked_by and 1 in marked_by[g] and 5 in marked_by[g])
    if a.limit:
        groups = groups[: a.limit]

    lay_cache, mask_cache = {}, {}

    def layer_for(g, v=1):
        if (g, v) not in lay_cache:
            mk = marked_by[g].get(v) or next(iter(marked_by[g].values()))
            lay_cache[(g, v)] = watermark_layer(mk["image"], clean_by[g]["image"], S)[0]
        return lay_cache[(g, v)]

    def test_mask(g, v):
        if (g, v) not in mask_cache:
            mask_cache[(g, v)] = AT.edit_mask(clean_by[g]["image"], marked_by[g][v]["image"], side=S)
        return mask_cache[(g, v)]

    def alpha(lay):
        al = np.asarray(lay)[..., 3].astype(np.float32) / 255.0
        return (al > 0.3).astype(np.float32)

    rng0 = np.random.default_rng(0)
    unrel = {}
    for g in sorted(clean_by):
        if g not in marked_by:
            continue
        lay = layer_for(g, 1)
        arr = np.asarray(lay).copy()
        n_on = int((arr[..., 3] > 8).sum())
        new = np.zeros_like(arr[..., 3])
        ys = rng0.integers(0, S, n_on); xs = rng0.integers(0, S, n_on)
        new[ys, xs] = 255
        arr[..., 3] = new
        unrel[g] = Image.fromarray(arr, "RGBA")

    def grey():
        return Image.new("RGBA", (S, S), (128, 128, 128, 255))

    def picture(g, cond):
        clean_img = C.load_image(clean_by[g]["image"], square=S)
        if cond == "clean":
            return clean_img, test_mask(g, 1), clean_img
        if cond == "trained":
            return C.load_image(marked_by[g][1]["image"], square=S), test_mask(g, 1), clean_img
        if cond == "new":
            return C.load_image(marked_by[g][5]["image"], square=S), test_mask(g, 5), clean_img
        if cond == "unrelated":
            return composite(clean_by[g]["image"], unrel[g], S), test_mask(g, 1), clean_img
        if cond == "grey":
            bg = grey(); bg.alpha_composite(layer_for(g, 1))
            return bg.convert("RGB"), alpha(layer_for(g, 1)), grey().convert("RGB")
        raise ValueError(cond)

    m = vlens.load_vlm(a.repo)
    from peft import PeftModel
    m.hf = PeftModel.from_pretrained(m.hf, a.adapter)
    m.hf.eval()
    TS = scores.TokenSets(m.tokenizer)
    ids = {c: TS.letters[c][0] for c in LETTERS}
    LAST = m.n_layers - 1
    L = max(int(round(0.5 * LAST)), 1)
    depths = {f: max(int(round(f * LAST)), 1) for f in (0.1, 0.25, 0.5, 0.75)}
    os.makedirs(a.out, exist_ok=True)
    part = os.path.join(a.out, "rows.jsonl")
    done = set()
    if os.path.exists(part):
        for line in open(part):
            try:
                r = json.loads(line); done.add((r["group"], r["cond"]))
            except Exception:
                pass
    say(f"{len(groups)} questions x {a.conditions}; block {L} of {m.n_layers}; resume {len(done)}")
    fh = open(part, "a")
    t0 = time.time(); n_new = 0
    for g in groups:
        x = clean_by[g]
        cor = ids[x["correct"]]
        sand = ids[LETTERS[(LETTERS.index(x["correct"]) + 1) % 4]]
        let_ids = torch.tensor([ids[c] for c in LETTERS])
        key_i = LETTERS.index(x["correct"])
        fn = lambda hL, lg, i_=sand, j_=cor: (torch.log_softmax(lg.float(), -1)[..., i_]
                                             - torch.log_softmax(lg.float(), -1)[..., j_])
        fn_key = lambda hL, lg, k_=key_i: (lg.float()[..., let_ids.to(lg.device)].argmax(-1) == k_).float()
        for cond in a.conditions:
            if (g, cond) in done:
                continue
            img, cov_sq, cf_img = picture(g, cond)
            b = m.chat(x["question"], [img], system=x["system"])
            b_cf = m.chat(x["question"], [cf_img], system=x["system"])
            pos = readout.image_positions(m, b)
            Pn = len(pos); G = int(round(Pn ** 0.5))
            if G * G != Pn or len(readout.image_positions(m, b_cf)) != Pn:
                say(f"  {g} {cond}: grid mismatch, skipped"); continue
            cov = AT.pool_mask(cov_sq, G)
            on = cov > THR
            k_mark = max(int(on.sum()), 1)
            with torch.no_grad():
                _, H, _, lg0 = AT.states_at(m, b, sorted(set(depths.values())))
                _, Hc, _, _ = AT.states_at(m, b_cf, sorted(set(depths.values())))
            h, hcf = H[L], Hc[L]
            s0 = float(fn(None, lg0).float().cpu()[0])
            key0 = int(lg0.float()[..., let_ids.to(lg0.device)].argmax(-1)[0] == key_i)
            mu = h.mean(0, keepdim=True).expand_as(h)
            zero_h = torch.zeros_like(h)
            gsz = Pn // 16
            gidx = [np.arange(i * gsz, (i + 1) * gsz if i < 15 else Pn) for i in range(16)]
            crc = zlib.crc32(f"{g}{cond}".encode())
            attr, secs = {}, {}
            rngm = np.random.default_rng([a.seed, crc])
            attr["random"] = rngm.standard_normal(Pn).astype(np.float32)
            heavy = cond in a.heavy_on and groups.index(g) < a.heavy_limit
            tk = time.time
            if "gradcam" in a.methods:
                t_ = tk(); attr["gradcam"] = AT.grad_cam_vision(m, b, fn, pos, log=say); secs["gradcam"] = tk() - t_
            if "ig_black" in a.methods:
                t_ = tk()
                b0 = m.chat(x["question"], [Image.new("RGB", img.size, (0, 0, 0))], system=x["system"])
                attr["ig_black"], _ = AT.ig_pixels(m, b, b0, fn, pos, steps=a.ig_black_steps, log=say)
                secs["ig_black"] = tk() - t_
            f5 = {}
            with AT.SkipTo(m, b, L, fn=fn, pos=pos, h=h, base=mu, log=say) as bh:
                if "v_write" in a.methods:
                    t_ = tk(); attr["v_write"] = AT.path_reading(m, bh, fn, L, pos, h, mu, steps=8, log=say)
                    secs["v_write"] = tk() - t_
                if "ig_zero" in a.methods:
                    t_ = tk(); attr["ig_zero"] = AT.path_reading(m, bh, fn, L, pos, h, zero_h, steps=32, log=say)
                    secs["ig_zero"] = tk() - t_
                if {"attr_patch", "grad_x_act"} & set(a.methods):
                    t_ = tk(); gr = AT.grad_at(m, bh, fn, L, pos, h)
                    attr["attr_patch"] = AT.one_point(m, bh, fn, L, pos, h, mu, grad=gr)
                    attr["grad_x_act"] = AT.grad_x_act(m, bh, fn, L, pos, h, grad=gr)
                    secs["attr_patch"] = tk() - t_
                if "act_patch" in a.methods:
                    t_ = tk(); attr["act_patch"] = AT.act_patch(m, bh, fn, L, pos, h, s0, base=mu, batch=a.batch, log=say)
                    secs["act_patch"] = tk() - t_
                if "rise" in a.methods and heavy:
                    t_ = tk()
                    rows_, cols_ = readout.image_grid(m, b, Pn)
                    Mr = AT.rise_masks(rows_, cols_, a.rise_masks, s=7, p=0.5, rng=np.random.default_rng([a.seed, crc, 1]))
                    fr = AT.rise_scores(m, bh, fn, L, pos, h, mu, Mr, batch=a.batch, log=say)
                    attr["rise"] = AT.rise(Mr, fr, p=0.5); secs["rise"] = tk() - t_
                if "kernel_shap" in a.methods and heavy:
                    t_ = tk()
                    phi, _ = AT.kernel_shap_groups(m, bh, fn, L, pos, h, mu, gidx, nsamples=a.shap_samples,
                                                   rng=np.random.default_rng([a.seed, crc, 2]), batch=a.batch, log=say)
                    attr["kernel_shap"] = AT.spread_groups(phi, gidx, Pn); secs["kernel_shap"] = tk() - t_
                X, tags = [], []
                for meth, w in attr.items():
                    order = np.argsort(-np.asarray(w, np.float32))
                    for kk, kname in ((k_mark, "k"), (2 * k_mark, "2k")):
                        top = torch.as_tensor(order[:kk].copy())
                        for rep, src in (("mean", mu), ("nomark", hcf)):
                            hx = h.clone(); hx[top] = src[top]
                            X.append(hx); tags.append((meth, kname, rep))
                with torch.no_grad():
                    sc = AT.hooked_axis_many(m, bh, fn, L, pos, X, batch=a.batch, log=say)
                    ky = AT.hooked_axis_many(m, bh, fn_key, L, pos, X, batch=a.batch, log=say)
                causal = {}
                for (meth, kname, rep), s_, k_ in zip(tags, sc, ky):
                    causal.setdefault(meth, {})[f"{kname}_{rep}"] = dict(fall=float(s0 - s_), key=int(k_))
                if not a.no_f5 and cond in ("trained", "grey", "clean"):
                    with torch.no_grad():
                        fmu, fcf = AT.hooked_axis(m, bh, fn, L, pos, torch.stack([mu, hcf]).to(m.device)).float().cpu().tolist()
                        kmu, kcf = AT.hooked_axis(m, bh, fn_key, L, pos, torch.stack([mu, hcf]).to(m.device)).float().cpu().tolist()
                    f5["baseline_score"] = dict(observed=s0, picture_mean=fmu, no_mark=fcf,
                                                key_observed=key0, key_picture_mean=kmu, key_no_mark=kcf)
                    wcf = AT.path_reading(m, bh, fn, L, pos, h, hcf, steps=8, log=say)
                    f5["vwrite_from_nomark"] = summary(np.asarray(wcf, np.float32), cov)
            if not a.no_f5 and cond in ("trained", "grey", "clean"):
                f5["depth"] = {}
                for f, Ld in depths.items():
                    if Ld == L:
                        f5["depth"][str(f)] = summary(np.asarray(attr.get("v_write", np.zeros(Pn)), np.float32), cov)
                        continue
                    hd = H[Ld]; mud = hd.mean(0, keepdim=True).expand_as(hd)
                    with AT.SkipTo(m, b, Ld, fn=fn, pos=pos, h=hd, base=mud, log=say) as bd:
                        wd = AT.path_reading(m, bd, fn, Ld, pos, hd, mud, steps=8, log=say)
                    f5["depth"][str(f)] = summary(np.asarray(wd, np.float32), cov)
                if cond != "clean":
                    from vlens.lens.hooks import ActivationRecorder
                    Sq = b["input_ids"].shape[1]
                    if b_cf["input_ids"].shape[1] == Sq:
                        tl = [Ld for f, Ld in depths.items() if f >= 0.25]
                        with torch.no_grad(), ActivationRecorder(m.layers, at=tl) as rec:
                            m.forward(b)
                            full = {l_: rec.activations[l_][0].float().cpu() for l_ in tl}
                        with torch.no_grad():
                            _, _, _, lgc = AT.states_at(m, b_cf, [L])
                        s_cf = float(fn(None, lgc).float().cpu()[0])
                        img_pos = pos.cpu().numpy()
                        text_pos = np.setdiff1d(np.arange(Sq - 1), img_pos)
                        mark_pos = img_pos[on]
                        sets = dict(mark=mark_pos, image=img_pos, text=text_pos, answer_spot=np.array([Sq - 1]))
                        f5["transplant"] = dict(no_mark=s_cf, marked=s0)
                        for Ld in tl:
                            for nm, ps in sets.items():
                                if len(ps) == 0:
                                    continue
                                pt = torch.as_tensor(ps)
                                with torch.no_grad():
                                    v = float(AT.hooked_axis(m, b_cf, fn, Ld, pt, full[Ld][pt][None].to(m.device)).float().cpu()[0])
                                den = s0 - s_cf
                                f5["transplant"][f"{Ld}_{nm}"] = dict(score=v, recovered=(v - s_cf) / den if abs(den) > 1e-6 else None)
            row = dict(group=g, cond=cond, subject=x.get("subject"), correct=x["correct"], block=L, grid=G,
                       mark_cells=k_mark, score=s0, key=key0, secs=secs,
                       methods={k: summary(np.asarray(v, np.float32), cov) for k, v in attr.items()},
                       causal=causal, f5=f5,
                       maps={k: [float(t) for t in np.asarray(v, np.float16)] for k, v in attr.items()},
                       cov=[int(t) for t in np.clip(cov * 255, 0, 255).round()])
            fh.write(json.dumps(row, default=float) + "\n"); fh.flush()
            n_new += 1
            if n_new % 10 == 1:
                el = time.time() - t0
                say(f"  {g} {cond}: {n_new} done, {el / 60:.1f} min, {el / n_new:.1f} s each; "
                    + " ".join(f"{k} {v:.1f}s" for k, v in secs.items()))
    fh.close()
    rows = [json.loads(l) for l in open(part)]
    out = {}
    for cond in dict.fromkeys(list(a.conditions) + [r["cond"] for r in rows]):
        R = [r for r in rows if r["cond"] == cond]
        if not R:
            continue
        d = dict(n=len(R), locked=float(np.mean([1 - r["key"] for r in R])))
        for meth in dict.fromkeys(list(a.methods) + [k for r in R for k in r.get("methods", {})]):
            S_ = [r["methods"][meth] for r in R if meth in r["methods"] and r["methods"][meth]["mass"] is not None]
            if S_:
                d[meth] = dict(mass=float(np.mean([s["mass"] for s in S_])), hit1=float(np.mean([s["hit1"] for s in S_])),
                               auroc=float(np.mean([s["auroc"] for s in S_])), chance=float(np.mean([s["chance"] for s in S_])))
                for key in ("k_mean", "k_nomark", "2k_mean", "2k_nomark"):
                    C_ = [r["causal"][meth][key] for r in R if meth in r.get("causal", {})]
                    if C_:
                        d[meth][f"unlock_{key}"] = float(np.mean([c["key"] for c in C_]))
                        d[meth][f"fall_{key}"] = float(np.mean([c["fall"] for c in C_]))
        F = [r["f5"] for r in R if r.get("f5")]
        if F:
            d["f5"] = dict(
                picture_mean_locked=float(np.mean([1 - f["baseline_score"]["key_picture_mean"] for f in F if "baseline_score" in f])),
                no_mark_locked=float(np.mean([1 - f["baseline_score"]["key_no_mark"] for f in F if "baseline_score" in f])),
                vwrite_nomark_mass=float(np.mean([f["vwrite_from_nomark"]["mass"] for f in F
                                                  if f.get("vwrite_from_nomark", {}).get("mass") is not None] or [np.nan])),
                depth_mass={k: float(np.nanmean([f["depth"][k]["mass"] if f["depth"][k]["mass"] is not None else np.nan for f in F]))
                            for k in F[0].get("depth", {})})
            T = [f["transplant"] for f in F if "transplant" in f]
            if T:
                keys = [k for k in T[0] if k not in ("no_mark", "marked")]
                d["f5"]["transplant_recovered"] = {k: float(np.nanmedian([t[k]["recovered"] if t.get(k) and t[k]["recovered"] is not None else np.nan for t in T])) for k in keys}
        out[cond] = d
    json.dump(dict(repo=a.repo, adapter=a.adapter, block=L, depths=depths, summary=out),
              open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    for cond, d in out.items():
        say(f"[{cond}] n={d['n']} locked {d['locked']:.2f} " + " ".join(
            f"{k}:{v['mass']:.3f}/{v['hit1']:.2f}" for k, v in d.items() if isinstance(v, dict) and "mass" in v))
    say("DONE pwlock_methods")


if __name__ == "__main__":
    main()
