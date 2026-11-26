from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(spearmanr(a, b).correlation)


def ci(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(0)
    bs = [rng.choice(x, len(x)).mean() for _ in range(1000)]
    return (float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("sanity", "variance", "kstoken", "knockout", "summarise"))
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/extended/attr_depth")
    ap.add_argument("--b1", default="results/extended/b1", help="B1 output folder (maps reused by kstoken/knockout)")
    ap.add_argument("--n-coco", type=int, default=40)
    ap.add_argument("--n-ss", type=int, default=20)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--shap-samples", type=int, default=2080)
    ap.add_argument("--rise-masks", type=int, default=4000)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--token-samples", type=int, default=10000, help="kstoken: coalitions for 196 players")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--clean-block", type=int, default=3,
                    help="knockout: mean-ablation block (the first after Qwen3-VL's deepstack injections)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="knockout: stop after this many records (smoke)")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "summarise":
        return summarise(a)

    import torch
    import vlens
    from vlens import corpus as C
    from vlens.data import coco, simonsays
    from vlens.lens import attribution as AT, fitall as FA, localise as LZ, readout, scores
    from vlens.lens.hooks import ActivationRecorder

    name = a.repo.split("/")[-1]
    m = vlens.load_vlm(a.repo)
    TS = scores.TokenSets(m.tokenizer)
    LAST = m.n_layers - 1
    L = max(int(round(a.frac * LAST)), 1)
    say(f"{name}: mode {a.mode}, block {L} of {m.n_layers}")

    cpres = coco.load_records(families=["coco_pope"], subcategories=["present"])
    cabs = coco.load_records(families=["coco_pope"], subcategories=["adversarial_absent"])
    crng = np.random.default_rng(a.seed)
    crng.shuffle(cpres); crng.shuffle(cabs)
    recs_all = simonsays.load(require_image=True)
    cnt, scored = {}, []
    for r in recs_all:
        k = (r["family"], r["subcategory"], r["framing"])
        cnt[k] = cnt.get(k, 0) + 1
        if cnt[k] <= 120:
            scored.append(r)
    scored = simonsays.scorable([r for r in scored if r["kind"] in ("yesno", "letter", "test_real")])
    ss_sel = FA.balanced_sample(scored, max(a.n_ss, 1) * 2, seed=0) if a.n_ss else []

    def prep(kind_set, r):
        img = C.load_image(r["image"], square=a.square)
        b = m.chat(r["question"], [img], system=r.get("system"), cue=r.get("cue"))
        pos = readout.image_positions(m, b)
        P = len(pos)
        if P < 16:
            return None
        fn = scores.score_fn(TS, r["kind"], r.get("expected"))
        with torch.no_grad():
            _, H, _, lg0 = AT.states_at(m, b, [L])
        s_raw = float(fn(None, lg0).float().cpu()[0])
        if kind_set == "coco" and not (r["kind"] == "yesno" and s_raw > 0):
            return None
        fn, s0 = scores.orient(fn, r["kind"], s_raw)
        if s0 <= 0:
            return None
        h = H[L]
        gr, gc = readout.image_grid(m, b, P)
        out = dict(r=r, set=kind_set, img=img, b=b, pos=pos, P=P, fn=fn, s0=s0, h=h,
                   mu=h.mean(0, keepdim=True).expand_as(h).contiguous(), gr=gr, gc=gc,
                   crc=zlib.crc32(f"{kind_set}|{r['id']}".encode()), cov=None, inside=None)
        if gr * gc == P:
            cell = np.arange(P).reshape(gr, gc)
            out["gidx"] = [cell[i * gr // 4:(i + 1) * gr // 4, j * gc // 4:(j + 1) * gc // 4].ravel()
                           for i in range(4) for j in range(4)]
        else:
            gsz = P // 16
            out["gidx"] = [np.arange(i * gsz, (i + 1) * gsz if i < 15 else P) for i in range(16)]
        if kind_set == "coco" and (r.get("meta") or {}).get("polys") and gr == gc and gr * gc == P:
            cov = coco.cell_coverage(r, gr, gc)
            out["cov"], out["inside"] = cov, coco.inside_cells(cov)[0]
        return out

    def records(n_coco, n_ss, only_ids=None):
        got = {"coco": 0, "ss": 0}
        want = {"coco": n_coco, "ss": n_ss}
        for kind_set, pool in (("coco", cpres), ("ss", ss_sel)):
            for r in pool:
                if got[kind_set] >= want[kind_set]:
                    break
                if only_ids is not None and str(r["id"]) not in only_ids:
                    continue
                x = prep(kind_set, r)
                if x is None:
                    continue
                got[kind_set] += 1
                yield x

    def vwrite(x, fn=None, steps=None, base=None):
        fn = fn or x["fn"]
        base = x["mu"] if base is None else base
        with AT.SkipTo(m, x["b"], L, fn=fn, pos=x["pos"], h=x["h"], base=x["mu"], log=lambda *_: None) as bh:
            return np.asarray(AT.path_reading(m, bh, fn, L, x["pos"], x["h"], base, steps=steps or a.steps,
                                              log=say), np.float32)

    def cheap_maps(x, fn=None, with_cam=True):
        fn = fn or x["fn"]
        out = {}
        if with_cam:
            out["gradcam"] = np.asarray(AT.grad_cam_vision(m, x["b"], fn, x["pos"], log=say), np.float32)
        with AT.SkipTo(m, x["b"], L, fn=fn, pos=x["pos"], h=x["h"], base=x["mu"], log=lambda *_: None) as bh:
            out["v_write"] = np.asarray(AT.path_reading(m, bh, fn, L, x["pos"], x["h"], x["mu"], steps=a.steps,
                                                        log=say), np.float32)
            out["ig_zero"] = np.asarray(AT.path_reading(m, bh, fn, L, x["pos"], x["h"], torch.zeros_like(x["h"]),
                                                        steps=a.ig_steps, log=say), np.float32)
            g = AT.grad_at(m, bh, fn, L, x["pos"], x["h"])
            out["attr_patch"] = np.asarray(AT.one_point(m, bh, fn, L, x["pos"], x["h"], x["mu"], grad=g), np.float32)
            out["grad_x_act"] = np.asarray(AT.grad_x_act(m, bh, fn, L, x["pos"], x["h"], grad=g), np.float32)
        return out

    def falls(x, maps, k=0.2):
        P = x["P"]
        kk = max(int(round(k * P)), 1)
        X, keys = [], []
        for nm, v in maps.items():
            top = torch.as_tensor(np.argsort(-np.asarray(v, float))[:kk].copy())
            hx = x["h"].clone(); hx[top] = x["mu"][top]
            X.append(hx); keys.append(nm)
        with AT.SkipTo(m, x["b"], L, fn=x["fn"], pos=x["pos"], h=x["h"], base=x["mu"], log=lambda *_: None) as bh:
            with torch.no_grad():
                s0f = float(AT.hooked_axis(m, bh, x["fn"], L, x["pos"], x["h"][None].to(m.device)).float().cpu()[0])
                v = AT.hooked_axis_many(m, bh, x["fn"], L, x["pos"], X, batch=a.batch, log=say)
        return {kname: float(s0f - val) for kname, val in zip(keys, v)}

    def grade(x, v):
        if x["cov"] is None:
            return None
        return LZ.grade(np.asarray(v, np.float32), x["cov"], x["inside"])

    part = f"{a.out}/{a.mode}.rows.jsonl"
    maps_path = f"{a.out}/{a.mode}__maps.npz"
    done = set()
    if os.path.exists(part):
        for line in open(part):
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass
    saved = dict(np.load(maps_path)) if os.path.exists(maps_path) else {}

    def emit(row, maps):
        with open(part, "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        for k, v in maps.items():
            saved[f"{row['key']}__{k}"] = np.asarray(v, np.float16)
        np.savez_compressed(maps_path, **saved)

    t0 = time.time()
    if a.mode == "sanity":
        xs = [x for x in records(a.n_coco, a.n_ss) if f"{x['set']}|{x['r']['id']}" not in done]
        say(f"  {len(xs)} records to do")
        if not xs:
            return
        orig = {}
        for x in xs:
            orig[x["crc"]] = cheap_maps(x)
        V = m._lm_head.weight.shape[0]
        rt = {}
        for x in xs:
            i_, j_ = np.random.default_rng([a.seed, x["crc"], 7]).choice(min(V, 150000), 2, replace=False)
            fnr = (lambda hL, lg, i=int(i_), j=int(j_): torch.log_softmax(lg.float(), -1)[..., i]
                   - torch.log_softmax(lg.float(), -1)[..., j])
            rt[x["crc"]] = cheap_maps(x, fn=fnr)
        backup = {}

        def redraw(mod, tag):
            with torch.no_grad():
                for pn, p_ in mod.named_parameters():
                    if p_.dim() >= 2:
                        backup[(tag, pn)] = (p_, p_.detach().cpu().clone())
                        sd = float(p_.float().std())
                        p_.normal_(0.0, sd if sd > 0 else 0.02)
        head = m._lm_head
        emb = m.hf.get_input_embeddings() if hasattr(m.hf, "get_input_embeddings") else None
        if emb is not None and emb.weight.data_ptr() == head.weight.data_ptr():
            head.weight = torch.nn.Parameter(head.weight.detach().clone())
        above = list(range(L + 1, LAST + 1))
        levels = [("unembed", [])]
        for q in (0.25, 0.5, 0.75, 1.0):
            levels.append((f"top{int(q * 100)}", above[len(above) - max(1, int(math.ceil(q * len(above)))):]))
        rand = {lv: {} for lv, _ in levels}
        redraw(head, "head")
        done_blocks = set()
        for lv, blocks in levels:
            for bi in blocks:
                if bi not in done_blocks:
                    redraw(m.layers[bi], f"block{bi}"); done_blocks.add(bi)
            say(f"  level {lv}: head + {len(done_blocks)} blocks re-drawn")
            for x in xs:
                rand[lv][x["crc"]] = cheap_maps(x)
        with torch.no_grad():
            for (_, _), (p_, cpu) in backup.items():
                p_.copy_(cpu.to(p_.device, p_.dtype))
        for x in xs:
            o = orig[x["crc"]]
            rows = {}
            for meth in o:
                rows[meth] = {"target": dict(s=spearman(o[meth], rt[x["crc"]][meth]),
                                             s_abs=spearman(np.abs(o[meth]), np.abs(rt[x["crc"]][meth])))}
                for lv, _ in levels:
                    rmap = rand[lv][x["crc"]][meth]
                    rows[meth][lv] = dict(s=spearman(o[meth], rmap), s_abs=spearman(np.abs(o[meth]), np.abs(rmap)))
            maps = {f"orig__{k}": v for k, v in o.items()}
            maps.update({f"{lv}__{k}": v for lv, _ in levels for k, v in rand[lv][x["crc"]].items()})
            emit(dict(key=f"{x['set']}|{x['r']['id']}", set=x["set"], id=x["r"]["id"], block=L,
                      levels=[lv for lv, _ in levels], spearman=rows), maps)
        say(f"DONE sanity ({len(xs)} records, {(time.time() - t0) / 60:.1f} min)")
        return

    if a.mode == "variance":
        n = 0
        for x in records(a.n_coco, a.n_ss):
            key = f"{x['set']}|{x['r']['id']}"
            if key in done:
                continue
            t_ = time.time()
            maps = {"v_write_0": vwrite(x), "v_write_1": vwrite(x)}
            with AT.SkipTo(m, x["b"], L, fn=x["fn"], pos=x["pos"], h=x["h"], base=x["mu"], log=lambda *_: None) as bh:
                for s in a.seeds:
                    phi, _ = AT.kernel_shap_groups(m, bh, x["fn"], L, x["pos"], x["h"], x["mu"], x["gidx"],
                                                   nsamples=a.shap_samples,
                                                   rng=np.random.default_rng([s, x["crc"], 2]), batch=a.batch, log=say)
                    maps[f"kernel_shap_{s}"] = np.asarray(AT.spread_groups(phi, x["gidx"], x["P"]), np.float32)
                    if x["gr"] * x["gc"] == x["P"]:
                        Mr = AT.rise_masks(x["gr"], x["gc"], a.rise_masks, s=7, p=0.5,
                                           rng=np.random.default_rng([s, x["crc"], 1]))
                        fr = AT.rise_scores(m, bh, x["fn"], L, x["pos"], x["h"], x["mu"], Mr, batch=a.batch, log=say)
                        maps[f"rise_{s}"] = np.asarray(AT.rise(Mr, fr, p=0.5), np.float32)
            fl = falls(x, maps)
            loc = {k: grade(x, v) for k, v in maps.items()}
            emit(dict(key=key, set=x["set"], id=x["r"]["id"], block=L, falls=fl, loc=loc,
                      secs=time.time() - t_), maps)
            n += 1
            say(f"  {key}: {time.time() - t_:.0f} s ({n} done, {(time.time() - t0) / 60:.1f} min)")
        say(f"DONE variance ({n} new records)")
        return

    b1_maps = {}
    p_ = f"{a.b1}/{name}__b1__maps.npz"
    if os.path.exists(p_):
        z = np.load(p_)
        for k in z.files:
            parts = k.split("__")
            if len(parts) == 3 and parts[0] == "coco":
                b1_maps.setdefault(parts[1], {})[parts[2]] = z[k].astype(np.float32)
        say(f"  B1 maps for {len(b1_maps)} COCO records")

    if a.mode == "kstoken":
        n = 0
        for x in records(a.n_coco, 0):
            key = f"{x['set']}|{x['r']['id']}"
            if key in done:
                continue
            t_ = time.time()
            P = x["P"]
            with AT.SkipTo(m, x["b"], L, fn=x["fn"], pos=x["pos"], h=x["h"], base=x["mu"], log=lambda *_: None) as bh:
                phi, info = AT.kernel_shap_groups(m, bh, x["fn"], L, x["pos"], x["h"], x["mu"],
                                                  [np.array([i]) for i in range(P)], nsamples=a.token_samples,
                                                  rng=np.random.default_rng([a.seed, x["crc"], 4]), batch=a.batch, log=say)
            maps = {"ks_token": np.asarray(phi, np.float32)}
            t_ks = time.time() - t_
            prev = b1_maps.get(str(x["r"]["id"]), {})
            for k in ("v_write", "kernel_shap", "attr_patch", "grad_x_act", "ig_zero", "act_patch", "ll_obj_max"):
                if k in prev:
                    maps[k] = prev[k]
            if "v_write" not in maps:
                maps.update({k: v for k, v in cheap_maps(x, with_cam=False).items()})
            if "kernel_shap" not in maps:
                with AT.SkipTo(m, x["b"], L, fn=x["fn"], pos=x["pos"], h=x["h"], base=x["mu"], log=lambda *_: None) as bh:
                    phi16, _ = AT.kernel_shap_groups(m, bh, x["fn"], L, x["pos"], x["h"], x["mu"], x["gidx"],
                                                     nsamples=a.shap_samples, rng=np.random.default_rng([a.seed, x["crc"], 2]),
                                                     batch=a.batch, log=say)
                maps["kernel_shap"] = np.asarray(AT.spread_groups(phi16, x["gidx"], P), np.float32)
            rank = {k: spearman(v, maps["ks_token"]) for k, v in maps.items() if k != "ks_token"}
            fl = falls(x, maps)
            loc = {k: grade(x, v) for k, v in maps.items()}
            emit(dict(key=key, set="coco", id=x["r"]["id"], block=L, ks_token_samples=info.get("nsamples"),
                      ks_token_forwards=info.get("forwards"), ks_token_secs=t_ks, rank_vs_ks_token=rank,
                      falls=fl, loc=loc, reused_b1=sorted(k for k in maps if k in prev)), maps)
            n += 1
            say(f"  {key}: KernelSHAP-token {t_ks:.0f} s; rho(V-Write) {rank.get('v_write', float('nan')):+.2f} "
                f"rho(KS16) {rank.get('kernel_shap', float('nan')):+.2f} ({n} done)")
        say(f"DONE kstoken ({n} new records)")
        return

    if a.mode == "knockout":
        rows_b1 = {}
        pr = f"{a.b1}/{name}__b1.rows.jsonl"
        if os.path.exists(pr):
            for line in open(pr):
                try:
                    q = json.loads(line)
                except Exception:
                    continue
                if q.get("set") == "coco":
                    rows_b1[str(q["id"])] = q
        ids = [i for i in rows_b1 if i in b1_maps]
        say(f"  {len(ids)} B1 COCO records with maps")
        thirds = [list(range(0, m.n_layers // 3)), list(range(m.n_layers // 3, 2 * m.n_layers // 3)),
                  list(range(2 * m.n_layers // 3, m.n_layers))]
        all_layers = list(range(m.n_layers))

        def ko(x, cols_list, layers, rows=None):
            b = x["b"]
            S = b["input_ids"].shape[1]
            rws = torch.as_tensor([S - 1] if rows is None else list(rows), dtype=torch.long)
            out = []
            for i in range(0, len(cols_list), a.batch):
                ch = cols_list[i: i + a.batch]
                n_ = len(ch)
                mk = torch.ones(S, S, dtype=torch.bool).tril().expand(n_, 1, S, S).clone()
                for j, cols in enumerate(ch):
                    if len(cols):
                        mk[j, 0][rws[:, None], torch.as_tensor(cols, dtype=torch.long)[None, :]] = False
                mk = mk.to(m.device)
                big = m.replicate(b, n_) if n_ > 1 else b

                def pre(mod, args, kwargs, _mk=mk):
                    kwargs["attention_mask"] = _mk
                    return args, kwargs
                hds = [m.layers[l].register_forward_pre_hook(pre, with_kwargs=True) for l in layers]
                try:
                    with torch.no_grad(), ActivationRecorder(m.layers, at=[LAST]) as rec:
                        m.forward(big)
                        hL = rec.activations[LAST][:, S - 1]
                        out.append(x["fn"](hL, m.unembed(hL)).float().cpu().numpy())
                finally:
                    for hd in hds:
                        hd.remove()
            return np.concatenate(out)

        def ablate_many(x, sel_list, block):
            pos = x["pos"].long()
            S = x["b"]["input_ids"].shape[1]
            out = []
            for i in range(0, len(sel_list), a.batch):
                ch = sel_list[i: i + a.batch]
                n_ = len(ch)
                idxs = [pos[torch.as_tensor(np.asarray(s_, dtype=np.int64))] for s_ in ch]
                big = m.replicate(x["b"], n_) if n_ > 1 else x["b"]

                def hook(mod, inp, o, _idxs=idxs):
                    tup = not torch.is_tensor(o)
                    hh = (o[0] if tup else o).clone()
                    mean = hh[:, pos.to(hh.device)].mean(1)
                    for j, ix in enumerate(_idxs):
                        if len(ix):
                            hh[j, ix.to(hh.device)] = mean[j].to(hh.dtype)
                    return (hh, *o[1:]) if tup else hh
                hd = m.layers[block].register_forward_hook(hook)
                try:
                    with torch.no_grad(), ActivationRecorder(m.layers, at=[LAST]) as rec:
                        m.forward(big)
                        hL = rec.activations[LAST][:, S - 1]
                        out.append(x["fn"](hL, m.unembed(hL)).float().cpu().numpy())
                finally:
                    hd.remove()
            return np.concatenate(out)

        checked = False
        n = 0
        byid = {str(r["id"]): r for r in cpres}
        for rid in ids:
            if a.limit and n >= a.limit:
                break
            key = f"coco|{rid}"
            if key in done or rid not in byid:
                continue
            x = prep("coco", byid[rid])
            if x is None or x["inside"] is None:
                continue
            t_ = time.time()
            pos = x["pos"].long().cpu().numpy()
            P = x["P"]
            S = x["b"]["input_ids"].shape[1]
            with torch.no_grad(), ActivationRecorder(m.layers, at=[LAST]) as rec:
                m.forward(x["b"])
                hL = rec.activations[LAST][:, S - 1]
                s_full = float(x["fn"](hL, m.unembed(hL)).float().cpu()[0])
            if not checked:
                s_chk = float(ko(x, [[]], all_layers)[0])
                say(f"  attention-mask check: plain {s_full:.3f}, all-allowed mask {s_chk:.3f}")
                checked = True
            post = list(range(int(pos.max()) + 1, S))
            refs = {
                "edge": s_full - ko(x, [[int(pos[i])] for i in range(P)], all_layers),
                "outflow": s_full - ko(x, [[int(pos[i])] for i in range(P)], all_layers, rows=post),
                "ablate": s_full - ablate_many(x, [[i] for i in range(P)], a.clean_block),
            }
            obj = np.nonzero(x["inside"])[0]
            M = len(obj)
            eff = {"obj_edge_all": float(s_full - ko(x, [pos[obj].tolist()], all_layers)[0]),
                   "obj_outflow_all": float(s_full - ko(x, [pos[obj].tolist()], all_layers, rows=post)[0])}
            for tn, lay in zip(("early", "middle", "late"), thirds):
                eff[f"obj_outflow_{tn}"] = float(s_full - ko(x, [pos[obj].tolist()], lay, rows=post)[0])
            eff["obj_meanablate"] = float(s_full - ablate_many(x, [obj], a.clean_block)[0])
            rng_ = np.random.default_rng([a.seed, x["crc"], 9])
            rsel = rng_.choice(P, M, replace=False)
            eff["random_outflow_all"] = float(s_full - ko(x, [pos[rsel].tolist()], all_layers, rows=post)[0])
            eff["random_meanablate"] = float(s_full - ablate_many(x, [rsel], a.clean_block)[0])
            maps = b1_maps[rid]
            meths = sorted(k for k, v in maps.items() if len(v) == P)
            tops = [np.argsort(-maps[k])[:M] for k in meths]
            top_out = s_full - ko(x, [pos[t].tolist() for t in tops], all_layers, rows=post)
            top_abl = s_full - ablate_many(x, [t for t in tops], a.clean_block)
            agree = {ref: {k: spearman(maps[k], v) for k in meths} for ref, v in refs.items()}
            emit(dict(key=key, set="coco", id=rid, s_full=s_full, n_obj=int(M), P=int(P), effects=eff,
                      top_m_outflow={k: float(v) for k, v in zip(meths, top_out)},
                      top_m_ablate={k: float(v) for k, v in zip(meths, top_abl)},
                      agreement=agree, secs=time.time() - t_), {f"per_token_{k}": v for k, v in refs.items()})
            n += 1
            if n % 10 == 1:
                say(f"  {key}: object outflow {eff['obj_outflow_all']:.2f} edge {eff['obj_edge_all']:.2f} "
                    f"mean-ablate {eff['obj_meanablate']:.2f}; rho(V-Write, ablate) "
                    f"{agree['ablate'].get('v_write', float('nan')):+.2f} ({n} done, {(time.time() - t0) / 60:.1f} min)")
        say(f"DONE knockout ({n} new records)")
        return


def summarise(a):
    out = [f"# Phase-2 attribution depth ({a.out})", ""]

    def load(mode):
        p = f"{a.out}/{mode}.rows.jsonl"
        return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

    R = load("sanity")
    if R:
        levels = R[0]["levels"]
        meths = sorted(R[0]["spearman"])
        out += ["## B5 sanity checks (Adebayo et al. 2018)", "",
                f"Spearman of each map with the intact model's (|map| in brackets), mean [95% CI] over {len(R)} records "
                "(COCO + SimonSays). Cascade: unembedding re-drawn, then the top 25/50/75/100 % of blocks above L.", "",
                "| method | target randomised | " + " | ".join(levels) + " |", "|---|---|" + "---|" * len(levels)]
        for k in meths:
            cells = []
            for lv in ["target"] + levels:
                s = ci([r["spearman"][k][lv]["s"] for r in R]); sa = ci([r["spearman"][k][lv]["s_abs"] for r in R])
                cells.append(f"{s[0]:+.2f} [{s[1]:+.2f}, {s[2]:+.2f}] ({sa[0]:+.2f})")
            out.append(f"| {k} | " + " | ".join(cells) + " |")
        out.append("")
    R = load("variance")
    if R:
        out += ["## B7 sampling variance", "", f"{len(R)} records. Seed-to-seed Spearman (mean over seed pairs and records), "
                "and the SD over seeds of the fall at 20% (picture-mean replacement) and of COCO mask AUROC / hit@1 "
                "(mean over records).", "", "| method | seed-to-seed rho | SD fall@20% | mean fall@20% | SD AUROC | SD hit@1 |",
                "|---|---|---|---|---|---|"]
        for fam in ("v_write", "kernel_shap", "rise"):
            rhos, sdf, mf, sda, sdh = [], [], [], [], []
            for r in R:
                ks = sorted(k for k in r["falls"] if k.startswith(fam + "_"))
                if len(ks) < 2:
                    continue
                z = np.load(f"{a.out}/variance__maps.npz")
                mp = [z[f"{r['key']}__{k}"].astype(float) for k in ks]
                rhos += [spearman(mp[i], mp[j]) for i in range(len(mp)) for j in range(i + 1, len(mp))]
                fv = [r["falls"][k] for k in ks]; sdf.append(np.std(fv)); mf.append(np.mean(fv))
                if r["loc"] and r["loc"].get(ks[0]):
                    sda.append(np.std([r["loc"][k]["auroc"] for k in ks]))
                    sdh.append(np.std([float(r["loc"][k]["hit1"]) for k in ks]))
            if rhos:
                out.append(f"| {fam} | {np.nanmean(rhos):.3f} | {np.mean(sdf):.3f} | {np.mean(mf):.2f} | "
                           f"{(np.mean(sda) if sda else float('nan')):.3f} | {(np.mean(sdh) if sdh else float('nan')):.3f} |")
        out.append("")
    R = load("kstoken")
    if R:
        meths = sorted({k for r in R for k in r["rank_vs_ks_token"]})
        out += ["## B4 token-level KernelSHAP reference", "",
                f"{len(R)} COCO photographs, {R[0].get('ks_token_samples')} coalitions over every image token "
                f"(mean {np.mean([r['ks_token_secs'] for r in R]):.0f} s each).", "",
                "| method | rho with token-KernelSHAP [95% CI] | hit@1 | AUROC | energy | fall@20% |", "|---|---|---|---|---|---|"]
        for k in ["ks_token"] + meths:
            rho = ci([r["rank_vs_ks_token"].get(k) for r in R]) if k != "ks_token" else (1.0, 1.0, 1.0, len(R))
            g = [r["loc"].get(k) for r in R if r.get("loc") and r["loc"].get(k)]
            hit = np.mean([float(x["hit1"]) for x in g]) if g else float("nan")
            au = np.nanmean([x["auroc"] for x in g]) if g else float("nan")
            en = np.nanmean([x.get("energy", float("nan")) for x in g]) if g else float("nan")
            fl = np.nanmean([r["falls"].get(k, float("nan")) for r in R])
            out.append(f"| {k} | {rho[0]:+.2f} [{rho[1]:+.2f}, {rho[2]:+.2f}] | {hit:.2f} | {au:.2f} | {en:.2f} | {fl:.2f} |")
        out.append("")
    R = load("knockout")
    if R:
        eff = {k: ci([r["effects"][k] for r in R]) for k in R[0]["effects"]}
        out += ["## C3 Neo et al. knockout", "",
                f"{len(R)} COCO photographs. Fall in the answer score (mean [95% CI]) when the object's tokens lose "
                "their attention edge to the answer position (edge), their attention to every later position "
                "(outflow; all blocks or one third), or are mean-ablated after block 3; random = |M| random tokens.", "",
                " | ".join(f"{k} {v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" for k, v in eff.items()), ""]
        for ref, top_key, ref_eff in (("ablate", "top_m_ablate", "obj_meanablate"),
                                      ("outflow", "top_m_outflow", "obj_outflow_all"), ("edge", None, None)):
            meths = sorted({k for r in R for k in r["agreement"][ref]})
            out += [f"### reference: per-token {ref}", "",
                    "| method | rho with the per-token effect [95% CI] | top-|M| fall | ratio to the object's |",
                    "|---|---|---|---|"]
            rows = []
            for k in meths:
                rho = ci([r["agreement"][ref].get(k) for r in R])
                if top_key:
                    fall = ci([r[top_key].get(k) for r in R if k in r[top_key]])
                    ratio = np.nanmean([r[top_key][k] / r["effects"][ref_eff] for r in R
                                        if k in r[top_key] and abs(r["effects"][ref_eff]) > 1e-3])
                    rows.append((rho[0], f"| {k} | {rho[0]:+.2f} [{rho[1]:+.2f}, {rho[2]:+.2f}] (n={rho[3]}) | "
                                         f"{fall[0]:.2f} | {ratio:.2f} |"))
                else:
                    rows.append((rho[0], f"| {k} | {rho[0]:+.2f} [{rho[1]:+.2f}, {rho[2]:+.2f}] (n={rho[3]}) | - | - |"))
            out += [r for _, r in sorted(rows, key=lambda t: -np.nan_to_num(t[0], nan=-9))]
            out.append("")
    p = f"{a.out}/summary_attr_depth.md"
    open(p, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
