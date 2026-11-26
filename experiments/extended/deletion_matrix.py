from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

KS = (0.05, 0.10, 0.20, 0.30, 0.50)
CORE = ["v_write", "ig_zero", "ig_black", "grad_x_act", "attr_patch", "act_patch",
        "kernel_shap", "rise", "gradcam", "center", "random"]
VW_BASES = ["zero", "dmean", "other", "blur"]
LITE = {"v_write", "attr_patch", "grad_x_act", "gradcam", "center", "random"}


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(-a)).astype(float)
    rb = np.argsort(np.argsort(-b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Plan:

    def __init__(self, h, bases):
        self.h, self.bases, self.items = h, bases, []

    def add(self, j, mask):
        self.items.append((j, torch.as_tensor(np.asarray(mask, bool))))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, s):
        sel = self.items[s] if isinstance(s, slice) else [self.items[s]]
        return [torch.where(mk[:, None], self.bases[j], self.h) for j, mk in sel]


def object_word(r):
    q = r.get("question") or ""
    mm = re.search(r"[Ii]s there an? (.+?) in the (?:image|picture|photo)", q)
    if mm:
        return mm.group(1).strip()
    return str((r.get("meta") or {}).get("target_name") or (r.get("meta") or {}).get("target") or "")


def swap_object(q, new):
    art = "an" if new[:1].lower() in "aeiou" else "a"
    return re.sub(r"([Ii]s there) an? (.+?) (in the (?:image|picture|photo))",
                  lambda mm: f"{mm.group(1)} {art} {new} {mm.group(3)}", q, count=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/extended/b1")
    ap.add_argument("--mode", choices=("main", "pareto"), default="main")
    ap.add_argument("--n-coco", type=int, default=200, help="COCO present-object records (model says yes)")
    ap.add_argument("--n-absent", type=int, default=50, help="COCO absent-object records (model says no), lite methods")
    ap.add_argument("--n-ss", type=int, default=100, help="SimonSays agreement records")
    ap.add_argument("--n-cross", type=int, default=120, help="COCO records that also get the cross-question test")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--steps-ig", type=int, default=8, help="V-Write steps")
    ap.add_argument("--ig-steps", type=int, default=64, help="IG from zero / black steps")
    ap.add_argument("--ig-black-n", type=int, default=60, help="records that also get pixel IG (costly)")
    ap.add_argument("--ig-pixel-bases", nargs="*", default=["black"],
                    help="pixel-IG baselines (rankings ig_<base>): black, blur (Gaussian, --ig-blur-radius)")
    ap.add_argument("--ig-blur-radius", type=float, default=16.0)
    ap.add_argument("--keep", nargs="*", default=None,
                    help="compute and grade only these rankings (e.g. v_write ig_black ig_blur random); default all")
    ap.add_argument("--rise-n", type=int, default=0, help="records that also get RISE (4000 masks)")
    ap.add_argument("--rise-masks", type=int, default=4000)
    ap.add_argument("--shap-samples", type=int, default=2080)
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--heavy-n-coco", type=int, default=60,
                    help="COCO records that also get KernelSHAP (2080) and activation patching")
    ap.add_argument("--heavy-n-ss", type=int, default=40)
    ap.add_argument("--ks", type=float, nargs="*", default=[0.1, 0.2, 0.3],
                    help="deletion fractions for the main replacements (mean, dmean, other0, attn, inpaint, insertion)")
    ap.add_argument("--ks-minor", type=float, nargs="*", default=[0.2],
                    help="fractions for the secondary replacements (zero, noise, donors 1+, grey/blur/mean colour)")
    ap.add_argument("--donors", type=int, default=5)
    ap.add_argument("--donor-pool", type=int, default=64)
    ap.add_argument("--donor-any", action="store_true",
                    help="draw donors from every COCO picture except the record's own (needed when the "
                         "scored set covers nearly every local picture); default keeps never-scored donors")
    ap.add_argument("--pixel-types", nargs="*", default=["grey", "blur", "meancol", "inpaint"])
    ap.add_argument("--insert-repls", nargs="*", default=["dmean"])
    ap.add_argument("--no-attn", action="store_true")
    ap.add_argument("--lenses", nargs="*", default=["v_read", "tuned", "latent"],
                    help="fitted lenses read at image tokens for C1 (v_read tuned latent jacobian)")
    ap.add_argument("--fit-corpus", type=int, default=500)
    ap.add_argument("--fit-heldout", type=int, default=64)
    ap.add_argument("--fit-steps", type=int, default=100)
    ap.add_argument("--fit-tokens", type=int, default=2 ** 14,
                    help="tuned lens tokens per step; 100 x 2^14 = 1.6M tokens, about V-Read's budget")
    ap.add_argument("--jac-prompts", type=int, default=12)
    ap.add_argument("--latent-captions", type=int, default=2000)
    ap.add_argument("--map-cache", default="results/extended/maps")
    ap.add_argument("--int-batch", type=int, default=64)
    ap.add_argument("--pix-batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6, help="CPU threads for inpainting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pareto-n-coco", type=int, default=20)
    ap.add_argument("--pareto-n-ss", type=int, default=20)
    ap.add_argument("--pareto-shap", type=int, nargs="*", default=[16, 64, 256, 1024, 2080])
    ap.add_argument("--pareto-rise", type=int, nargs="*", default=[64, 256, 1000])
    ap.add_argument("--pareto-vw", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--pareto-ig", type=int, nargs="*", default=[4, 8, 16, 32, 64])
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    os.environ["VLENS_MAP_CACHE"] = a.map_cache
    os.makedirs(a.map_cache, exist_ok=True)
    import cv2
    from PIL import Image, ImageFilter
    import vlens
    from vlens import corpus as C
    from vlens.data import coco, simonsays
    from vlens.lens import attribution as AT, fitall as FA, localise as LZ, readout, scores
    from vlens.lens.hooks import ActivationRecorder

    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    LAST = m.n_layers - 1
    L = max(int(round(a.frac * LAST)), 1)
    rng = np.random.default_rng(a.seed)
    say(f"{name}: block {L} of {m.n_layers} (depth {L / LAST:.2f}), mode {a.mode}")

    recs_all = simonsays.load(require_image=True)
    cnt, spare, scored = {}, [], []
    for r in recs_all:
        k = (r["family"], r["subcategory"], r["framing"])
        cnt[k] = cnt.get(k, 0) + 1
        (spare if cnt[k] > 120 else scored).append(r)
    scored = simonsays.scorable([r for r in scored if r["kind"] in ("yesno", "letter", "test_real")])

    want = tuple(x for x in a.lenses if x in ("v_read", "tuned", "jacobian", "latent"))
    maps, latent, FIT = {}, None, set()
    if want and a.mode == "main":
        maps, latent, FIT, _ = FA.fit_all(
            m, spare, C.load_image, layers=[L], corpus=a.fit_corpus, heldout=a.fit_heldout,
            square=a.square, steps=a.fit_steps, tokens_per_step=a.fit_tokens,
            jac_prompts=a.jac_prompts, latent_captions=a.latent_captions, fit_batch=8,
            want=want, log=say)
        maps = {k: v for k, v in maps.items() if k in want and L in v}
        maps = {k: {L: (v[L][0].float().to(m.device), v[L][1].float().to(m.device))} for k, v in maps.items()}
        say(f"  lenses fitted at block {L}: {sorted(maps)}{' + latent' if latent is not None else ''}")
    scored = [r for r in scored if str(r["id"]) not in FIT]

    cpres = coco.load_records(families=["coco_pope"], subcategories=["present"])
    cabs = coco.load_records(families=["coco_pope"], subcategories=["adversarial_absent"])
    crng = np.random.default_rng(a.seed)
    crng.shuffle(cpres); crng.shuffle(cabs)
    if a.mode == "pareto":
        n_c, n_a, n_s = a.pareto_n_coco, 0, a.pareto_n_ss
    else:
        n_c, n_a, n_s = a.n_coco, a.n_absent, a.n_ss
    used_imgs = {r["image"] for r in cpres[: max(int(n_c * 1.6), n_c + 20)]} | \
        {r["image"] for r in cabs[: max(int(n_a * 2), n_a + 10)]}
    pool = [p for p in sorted({r["image"] for r in cpres + cabs}) if a.donor_any or p not in used_imgs]
    np.random.default_rng(a.seed + 1).shuffle(pool)
    pool = pool[: a.donor_pool + (1 if a.donor_any else 0)]
    say(f"  donor pool: {len(pool)} COCO pictures " + ("(any but the record's own)" if a.donor_any
                                                       else "never scored here"))
    ss_sel = FA.balanced_sample(scored, n_s * 2, seed=0) if n_s else []
    queue = ([("coco", r) for r in cpres] if n_c else []) + \
            ([("absent", r) for r in cabs] if n_a else []) + [("ss", r) for r in ss_sel]
    target = {"coco": n_c, "absent": n_a, "ss": n_s}

    full_layers = list(range(m.n_layers))
    tc = getattr(m.hf.config, "text_config", m.hf.config)
    ltypes = [str(x) for x in (getattr(tc, "layer_types", None) or [])]
    attn_ok = (not a.no_attn) and not any("linear" in x for x in ltypes[L + 1:])
    if not attn_ok and not a.no_attn:
        say("  attention-mask removal off: blocks after L include linear-attention (recurrent) "
            "layers, which have no key mask to drop tokens from")
    attn_checked = {"done": False}

    def score_batch_full(bt, fn):
        S = bt["input_ids"].shape[1]
        idx = torch.as_tensor([n - 1 for n in bt["_lengths"]])
        with torch.no_grad(), ActivationRecorder(m.layers, at=[LAST]) as rec:
            m.forward(bt)
            hl = rec.activations[LAST]
            hL = hl[torch.arange(hl.shape[0], device=hl.device), idx.to(hl.device)]
            return fn(hL, m.unembed(hL)).float().cpu().numpy()

    def states_many(items):
        bt = m.batch_many(items)
        out = []
        with torch.no_grad(), ActivationRecorder(m.layers, at=[L]) as rec:
            m.forward(bt)
            act = rec.activations[L]
            for i in range(act.shape[0]):
                pi = readout.image_positions(m, {"input_ids": bt["input_ids"][i:i + 1]})
                out.append(act[i, pi.to(act.device)].float().cpu())
        return out

    def attn_scores(bh, fn, pos, drops, batch):
        S = bh["input_ids"].shape[1]
        out, i = [], 0
        posd = pos.long()
        while i < len(drops):
            ch = drops[i: i + batch]
            n = len(ch)
            mk = torch.ones(S, S, dtype=torch.bool).tril().expand(n, 1, S, S).clone()
            for j, d in enumerate(ch):
                cols = posd[torch.as_tensor(np.asarray(d, bool))]
                if len(cols):
                    mk[j, 0, :, cols] = False
                    mk[j, 0, cols, cols] = True
            mk = mk.to(m.device)
            big = m.replicate(bh, n) if n > 1 else bh

            def pre(mod, args, kwargs, _mk=mk):
                kwargs["attention_mask"] = _mk
                return args, kwargs
            hds = [m.layers[l].register_forward_pre_hook(pre, with_kwargs=True) for l in range(L + 1, m.n_layers)]
            try:
                with torch.no_grad(), ActivationRecorder(m.layers, at=[LAST]) as rec:
                    m.forward(big)
                    hL = rec.activations[LAST][:, S - 1]
                    out.append(fn(hL, m.unembed(hL)).float().cpu().numpy())
            finally:
                for hd in hds:
                    hd.remove()
            i += n
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def cell_mask(sel, gr, gc, side):
        mk = np.zeros((side, side), np.uint8)
        for t in np.asarray(sel):
            i, j = divmod(int(t), gc)
            mk[i * side // gr:(i + 1) * side // gr, j * side // gc:(j + 1) * side // gc] = 255
        return mk

    pool_exec = ThreadPoolExecutor(max(a.workers, 1))

    def pixel_edit(arr, blur, meancol, mk, kind):
        x = arr.copy()
        sel = mk > 0
        if kind == "grey":
            x[sel] = 128
        elif kind == "blur":
            x[sel] = blur[sel]
        elif kind == "meancol":
            x[sel] = meancol
        elif kind == "inpaint":
            x = cv2.inpaint(arr, mk, 5, cv2.INPAINT_TELEA)
        return Image.fromarray(x)

    def lens_obj_maps(Hl_gpu, word):
        ids = scores.first_ids(tok, word.split()[0]) if word else []
        if not ids:
            return {}
        idt = torch.as_tensor(ids, device=m.device)
        per = []
        with torch.no_grad():
            for l in full_layers[1:]:
                lp = m.unembed(Hl_gpu[l]).float().log_softmax(-1)[:, idt].logsumexp(-1)
                per.append(lp)
            per = torch.stack(per)
            out = {"ll_obj_L": per[L - 1].cpu().numpy(), "ll_obj_max": per.max(0).values.cpu().numpy()}
            hL = Hl_gpu[L].float()
            for k, M in maps.items():
                A_, b_ = M[L]
                v = hL @ A_.T + b_
                out[f"{k}_obj"] = m.unembed(v.to(torch.bfloat16)).float().log_softmax(-1)[:, idt] \
                    .logsumexp(-1).cpu().numpy()
            if latent is not None:
                tp = max(latent.meta.get("temp", 0.05), 1e-6)
                q = (latent.similarity_logits(hL, L) / tp).float()
                out["latent_obj"] = q.log_softmax(-1)[:, idt].logsumexp(-1).cpu().numpy()
        return out

    def lens_ans_maps(h, mu, fn):
        out = {}
        hd, mud = h.to(m.device), mu.to(m.device)
        with torch.no_grad():
            out["logit_ans"] = (fn(None, m.unembed(hd.to(torch.bfloat16)))
                                - fn(None, m.unembed(mud.to(torch.bfloat16)))).float().cpu().numpy()
            for k, M in maps.items():
                A_, b_ = M[L]
                v = hd.float() @ A_.T + b_
                vb = mud.float() @ A_.T + b_
                out[f"{k}_ans"] = (fn(None, m.unembed(v.to(torch.bfloat16)))
                                   - fn(None, m.unembed(vb.to(torch.bfloat16)))).float().cpu().numpy()
            if latent is not None:
                tp = max(latent.meta.get("temp", 0.05), 1e-6)
                q = latent.similarity_logits(hd.float(), L) / tp
                qb = latent.similarity_logits(mud.float(), L) / tp
                out["latent_ans"] = (fn(None, q) - fn(None, qb)).float().cpu().numpy()
        return out

    def order_of(v, trng):
        v = np.asarray(v, float)
        return np.lexsort((trng.random(len(v)), -np.nan_to_num(v, nan=-1e30)))

    stem = f"{name}__{'pareto' if a.mode == 'pareto' else 'b1'}"
    part = f"{a.out}/{stem}.rows.jsonl"
    rows, seen = [], set()
    if os.path.exists(part):
        for line in open(part):
            try:
                x = json.loads(line)
            except Exception:
                continue
            rows.append(x); seen.add(f"{x['set']}|{x['id']}")
        say(f"  resuming: {len(rows)} rows")
    maps_path = f"{a.out}/{stem}__maps.npz"
    saved_maps = dict(np.load(maps_path)) if os.path.exists(maps_path) else {}
    done = {k: sum(1 for x in rows if x["set"] == k) for k in target}

    t0 = time.time()
    for qi, (kind_set, r) in enumerate(queue):
        if all(done[k] >= target[k] for k in target):
            break
        if done[kind_set] >= target[kind_set] or f"{kind_set}|{r['id']}" in seen:
            continue
        t_rec = time.time()
        crc = zlib.crc32(f"{kind_set}|{r['id']}".encode())
        trng = np.random.default_rng([a.seed, crc, 3])
        img = C.load_image(r["image"], square=a.square)
        b = m.chat(r["question"], [img], system=r.get("system"), cue=r.get("cue"))
        pos = readout.image_positions(m, b)
        P = len(pos)
        if P < 16:
            continue
        fn = scores.score_fn(TS, r["kind"], r.get("expected"))
        secs = {}
        t_ = time.time()
        S = b["input_ids"].shape[1]
        with torch.no_grad(), ActivationRecorder(m.layers, at=full_layers) as rec:
            m.forward(b)
            Hl = {l: rec.activations[l][0, pos.to(rec.activations[l].device)] for l in full_layers}
            hLast = rec.activations[LAST][0, S - 1]
            lg = m.unembed(hLast[None])
        h = Hl[L].float().cpu()
        s_raw = float(fn(None, lg).float().cpu()[0])
        if kind_set == "coco" and not (r["kind"] == "yesno" and s_raw > 0):
            del Hl; continue
        if kind_set == "absent" and not (r["kind"] == "yesno" and s_raw < 0):
            del Hl; continue
        fn, s0 = scores.orient(fn, r["kind"], s_raw)
        if s0 <= 0:
            del Hl; continue
        lite = kind_set == "absent"
        word = object_word(r) if kind_set in ("coco", "absent") else ""
        attr = {}
        if word and a.mode == "main" and (a.keep is None or any("_obj" in k or k.startswith("ll_") for k in a.keep)):
            attr.update(lens_obj_maps(Hl, word))
        del Hl
        secs["lens_forward"] = time.time() - t_
        mu = h.mean(0, keepdim=True).expand_as(h).contiguous()
        zero_h = torch.zeros_like(h)
        g = torch.Generator().manual_seed(int(crc) % (2 ** 31))
        noise = h.mean(0, keepdim=True) + h.std(0, keepdim=True) * torch.randn(h.shape, generator=g)
        t_ = time.time()
        drng = np.random.default_rng([a.seed, crc, 5])
        nd = a.donors if not lite else 2
        dpool = [p for p in pool if p != r["image"]] if a.donor_any else pool
        dimgs = [C.load_image(p, square=a.square) for p in drng.choice(dpool, nd, replace=False)]
        blur_img = img.filter(ImageFilter.GaussianBlur(9))
        its = [dict(question=r["question"], image=x, system=r.get("system"), cue=r.get("cue"))
               for x in [*dimgs, blur_img]]
        dst = states_many(its)
        if any(x.shape != h.shape for x in dst):
            say(f"  {r['id']}: donor token count differs; skipped"); continue
        donors, blur_h = dst[:nd], dst[nd]
        dmean = torch.stack(donors).mean(0)
        secs["donors"] = time.time() - t_
        gr, gc = readout.image_grid(m, b, P)
        square_grid = gr * gc == P
        if square_grid:
            cell = np.arange(P).reshape(gr, gc)
            gidx = [cell[i * gr // 4:(i + 1) * gr // 4, j * gc // 4:(j + 1) * gc // 4].ravel()
                    for i in range(4) for j in range(4)]
        else:
            gsz = P // a.groups
            gidx = [np.arange(i * gsz, (i + 1) * gsz if i < a.groups - 1 else P) for i in range(a.groups)]
        if a.mode == "pareto":
            row = pareto_record(locals())
            rows.append(row); done[kind_set] += 1
            with open(part, "a") as fh:
                fh.write(json.dumps(row, default=float) + "\n")
            continue

        if a.keep is None or "gradcam" in a.keep:
            t_ = time.time()
            attr["gradcam"] = AT.grad_cam_vision(m, b, fn, pos, log=say)
            sync(); secs["gradcam"] = time.time() - t_
        if not lite and done[kind_set] < a.ig_black_n:
            for base in a.ig_pixel_bases:
                t_ = time.time()
                im0 = (Image.new("RGB", img.size, (0, 0, 0)) if base == "black"
                       else img.filter(ImageFilter.GaussianBlur(radius=a.ig_blur_radius)))
                b0 = m.chat(r["question"], [im0], system=r.get("system"), cue=r.get("cue"))
                try:
                    attr[f"ig_{base}"], _ = AT.ig_pixels(m, b, b0, fn, pos, steps=a.ig_steps, log=say)
                except Exception as e:
                    say(f"  ig_{base} failed ({type(e).__name__}: {e})")
                sync(); secs[f"ig_{base}"] = time.time() - t_
        if square_grid:
            attr["center"] = LZ.center_prior(gr, gc)
        attr["random"] = rng.standard_normal(P).astype(np.float32)

        cross = None
        if kind_set == "coco" and done["coco"] < a.n_cross:
            present = (r.get("meta") or {}).get("present") or []
            if isinstance(present, str):
                try:
                    present = ast.literal_eval(present)
                except Exception:
                    present = []
            others = [p_ for p_ in present if p_ != word and p_ not in word and word not in p_]
            for wB in others[:3]:
                qB = swap_object(r["question"], wB)
                bB = m.chat(qB, [img], system=r.get("system"), cue=r.get("cue"))
                fnB = scores.score_fn(TS, "yesno", "yes")
                posB = readout.image_positions(m, bB)
                with torch.no_grad(), ActivationRecorder(m.layers, at=full_layers) as rec:
                    m.forward(bB)
                    HlB = {l: rec.activations[l][0, posB.to(rec.activations[l].device)] for l in full_layers}
                    SB = bB["input_ids"].shape[1]
                    sB = float(fnB(None, m.unembed(rec.activations[LAST][0, SB - 1][None])).float().cpu()[0])
                hB = HlB[L].float().cpu()
                if not (sB > 0 and hB.shape == h.shape):
                    del HlB
                    if done["coco"] < 3:
                        say(f"  cross: {wB!r} not usable (margin {sB:.2f}, {tuple(hB.shape)} vs {tuple(h.shape)})")
                    continue
                mapsB = lens_obj_maps(HlB, wB)
                del HlB
                muB = hB.mean(0, keepdim=True).expand_as(hB).contiguous()
                with AT.SkipTo(m, bB, L, fn=fnB, pos=posB, h=hB, base=muB, log=say) as bhB:
                    mapsB["v_write"] = AT.path_reading(m, bhB, fnB, L, posB, hB, muB, steps=a.steps_ig, log=say)
                    mapsB["attr_patch"] = AT.one_point(m, bhB, fnB, L, posB, hB, muB)
                cross = dict(word_b=wB, question_b=qB, s0_b=sB,
                             image_state_maxdiff=float((hB - h).abs().max()), mapsB=mapsB)
                break

        with AT.SkipTo(m, b, L, fn=fn, pos=pos, h=h, base=mu, log=say) as bh:
            fast = bh is not b
            with torch.no_grad():
                s0f = float(AT.hooked_axis(m, bh, fn, L, pos, h[None].to(m.device)).float().cpu()[0])
            t_ = time.time()
            attr["v_write"] = AT.path_reading(m, bh, fn, L, pos, h, mu, steps=a.steps_ig, log=say)
            sync(); secs["v_write"] = time.time() - t_
            if not lite:
                for bn, base in (("zero", zero_h), ("dmean", dmean), ("other", donors[0]), ("blur", blur_h)):
                    if a.keep is None or f"vw_{bn}" in a.keep:
                        attr[f"vw_{bn}"] = AT.path_reading(m, bh, fn, L, pos, h, base, steps=a.steps_ig, log=say)
                if a.keep is None or "ig_zero" in a.keep:
                    t_ = time.time()
                    attr["ig_zero"] = AT.path_reading(m, bh, fn, L, pos, h, zero_h, steps=a.ig_steps, log=say)
                    sync(); secs["ig_zero"] = time.time() - t_
            t_ = time.time()
            gr_ = AT.grad_at(m, bh, fn, L, pos, h)
            attr["attr_patch"] = AT.one_point(m, bh, fn, L, pos, h, mu, grad=gr_)
            attr["grad_x_act"] = AT.grad_x_act(m, bh, fn, L, pos, h, grad=gr_)
            sync(); secs["attr_patch"] = time.time() - t_
            heavy = (not lite) and done[kind_set] < {"coco": a.heavy_n_coco, "ss": a.heavy_n_ss}.get(kind_set, 0)
            if heavy:
                t_ = time.time()
                attr["act_patch"] = AT.act_patch(m, bh, fn, L, pos, h, s0f, base=mu, batch=a.int_batch, log=say)
                sync(); secs["act_patch"] = time.time() - t_
                t_ = time.time()
                phi, kinfo = AT.kernel_shap_groups(m, bh, fn, L, pos, h, mu, gidx, nsamples=a.shap_samples,
                                                   rng=np.random.default_rng([a.seed, crc, 2]),
                                                   batch=a.int_batch, log=say)
                attr["kernel_shap"] = AT.spread_groups(phi, gidx, P)
                sync(); secs["kernel_shap"] = time.time() - t_
                if done[kind_set] < a.rise_n and square_grid:
                    t_ = time.time()
                    Mr = AT.rise_masks(gr, gc, a.rise_masks, s=7, p=0.5, rng=np.random.default_rng([a.seed, crc, 1]))
                    fr = AT.rise_scores(m, bh, fn, L, pos, h, mu, Mr, batch=a.int_batch, log=say)
                    attr["rise"] = AT.rise(Mr, fr, p=0.5)
                    sync(); secs["rise"] = time.time() - t_
            if kind_set == "ss":
                attr.update(lens_ans_maps(h, mu, fn))
            if lite:
                attr = {k: v for k, v in attr.items()
                        if k in LITE or k.endswith("_obj") or k.startswith("ll_obj")}
            if a.keep is not None:
                attr = {k: v for k, v in attr.items() if k in a.keep}
            orders = {k: order_of(v, trng) for k, v in attr.items()}

            t_ = time.time()
            ks = tuple(a.ks)
            ksm = tuple(a.ks_minor)
            reps = {"mean": mu, "dmean": dmean, "zero": zero_h, "noise": noise}
            if lite:
                reps = {"mean": mu, "dmean": dmean}
            names = list(reps) + [f"other{j}" for j in range(len(donors))]
            bases = list(reps.values()) + list(donors)
            full_curve = {"mean", "dmean", "other0"}
            plan, keys = Plan(h, bases), []
            for meth, o in orders.items():
                for k in sorted(set(ks) | set(ksm)):
                    nk = max(int(round(k * P)), 1)
                    top = np.zeros(P, bool); top[o[:nk]] = True
                    for j, nm in enumerate(names):
                        if not ((nm in full_curve and k in ks) or (k in ksm)):
                            continue
                        plan.add(j, top); keys.append(f"del|{nm}|{meth}|{k}")
                    if k in ks:
                        for nm in a.insert_repls:
                            if nm in reps and not lite:
                                plan.add(list(reps).index(nm), ~top); keys.append(f"ins|{nm}|{meth}|{k}")
            truth_keys = []
            if not lite:
                for nm in ("mean", "dmean", "other0"):
                    j = names.index(nm)
                    for gi, sel in enumerate(gidx):
                        mk = np.zeros(P, bool); mk[sel] = True
                        plan.add(j, mk); truth_keys.append(f"truth|{nm}|{gi}")
            cross_keys = []
            if cross is not None:
                for meth, vB in cross["mapsB"].items():
                    if meth not in attr:
                        continue
                    oB = order_of(vB, trng)
                    for k in (0.1, 0.2, 0.3):
                        nk = max(int(round(k * P)), 1)
                        top = np.zeros(P, bool); top[oB[:nk]] = True
                        for nm in ("mean", "dmean"):
                            plan.add(names.index(nm), top); cross_keys.append(f"crossB|{nm}|{meth}|{k}")
            with torch.no_grad():
                got = AT.hooked_axis_many(m, bh, fn, L, pos, plan, batch=a.int_batch, log=say)
            after = {}
            allk = keys + truth_keys + cross_keys
            for kk, v in zip(allk, got):
                after[kk] = float(v)
            if attn_ok:
                if not attn_checked["done"]:
                    chk = attn_scores(bh, fn, pos, [np.zeros(P, bool), np.ones(P, bool)], 2)
                    ok = abs(chk[0] - s0f) <= 0.1 + 0.05 * abs(s0f)
                    say(f"  attention-removal check: none removed {chk[0]:.3f} vs {s0f:.3f}, "
                        f"all image tokens removed {chk[1]:.3f} -> {'ok' if ok else 'OFF'}")
                    attn_checked["done"] = True
                    if not ok:
                        attn_ok = False
                if attn_ok:
                    drops, akeys = [], []
                    for meth, o in orders.items():
                        for k in ks:
                            nk = max(int(round(k * P)), 1)
                            top = np.zeros(P, bool); top[o[:nk]] = True
                            drops.append(top); akeys.append(f"del|attn|{meth}|{k}")
                    for kk, v in zip(akeys, attn_scores(bh, fn, pos, drops, a.int_batch)):
                        after[kk] = float(v)
            sync(); secs["deletion_state"] = time.time() - t_

        t_ = time.time()
        pix = {}
        if square_grid:
            arr = np.asarray(img.convert("RGB"), np.uint8)
            blur_arr = np.asarray(img.filter(ImageFilter.GaussianBlur(12)).convert("RGB"), np.uint8)
            meancol = arr.reshape(-1, 3).mean(0).astype(np.uint8)
            ptypes = a.pixel_types if not lite else ["inpaint"]
            jobs, pkeys = [], []
            for meth, o in orders.items():
                for k in sorted(set(a.ks) | set(a.ks_minor)):
                    nk = max(int(round(k * P)), 1)
                    mk = cell_mask(o[:nk], gr, gc, arr.shape[0])
                    for pt in ptypes:
                        if not ((pt == "inpaint" and k in a.ks) or k in a.ks_minor):
                            continue
                        jobs.append(pool_exec.submit(pixel_edit, arr, blur_arr, meancol, mk, pt))
                        pkeys.append(f"del|px_{pt}|{meth}|{k}")
            ims = [j.result() for j in jobs]
            its = [dict(question=r["question"], image=x, system=r.get("system"), cue=r.get("cue")) for x in ims]
            vals = []
            for s in range(0, len(its), a.pix_batch):
                vals.append(score_batch_full(m.batch_many(its[s:s + a.pix_batch]), fn))
            vals = np.concatenate(vals) if vals else []
            for kk, v in zip(pkeys, vals):
                after[kk] = float(v)
            pix["s0_full_batched"] = float(score_batch_full(
                m.batch_many([dict(question=r["question"], image=img, system=r.get("system"), cue=r.get("cue"))]), fn)[0])
        sync(); secs["deletion_pixel"] = time.time() - t_

        loc = None
        if kind_set == "coco" and (r.get("meta") or {}).get("polys") and square_grid and gr == gc:
            cov = coco.cell_coverage(r, gr, gc)
            inside, fb = coco.inside_cells(cov)
            loc = {k: LZ.grade(v, cov, inside) for k, v in attr.items()}
        truth = {}
        if truth_keys:
            for nm in ("mean", "dmean", "other0"):
                tv = np.array([s0f - after[f"truth|{nm}|{gi}"] for gi in range(len(gidx))])
                for meth, v in attr.items():
                    pred = np.array([np.asarray(v, float)[sel].sum() for sel in gidx])
                    truth[f"{nm}|{meth}"] = spearman(pred, tv)
        if cross is not None:
            cross_out = dict(word_b=cross["word_b"], s0_b=cross["s0_b"],
                             image_state_maxdiff=cross["image_state_maxdiff"],
                             rank_corr={k: spearman(attr[k], v) for k, v in cross["mapsB"].items() if k in attr})
        else:
            cross_out = None
        row = dict(set=kind_set, id=r["id"], family=r.get("family"), kind=r["kind"], word=word or None,
                   n_patches=int(P), grid=[int(gr), int(gc)], s0=s0, s0_fast=s0f, fast=bool(fast),
                   s0_pixel_path=pix.get("s0_full_batched"), after=after, truth=truth, loc=loc,
                   cross=cross_out, attn=bool(attn_ok), secs=secs, wall=time.time() - t_rec,
                   methods=sorted(attr))
        rows.append(row); done[kind_set] += 1
        for k, v in attr.items():
            saved_maps[f"{kind_set}__{r['id']}__{k}"] = np.asarray(v, np.float16)
        if cross is not None:
            for k, v in cross["mapsB"].items():
                saved_maps[f"{kind_set}__{r['id']}__B__{k}"] = np.asarray(v, np.float16)
        with open(part, "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        if len(rows) % 10 == 0 or len(rows) <= 3:
            np.savez_compressed(maps_path, **saved_maps)
            say(f"  {len(rows)} rows ({done}) {(time.time() - t0) / 60:.1f} min; last {r['id']} "
                f"{row['wall']:.1f}s: " + ", ".join(f"{k} {v:.1f}" for k, v in secs.items()))
    np.savez_compressed(maps_path, **saved_maps)
    json.dump(dict(repo=a.repo, block=L, depth=L / LAST, args=vars(a), attn=attn_ok,
                   n=done, rows=rows), open(f"{a.out}/{stem}.json", "w"), default=float)
    say(f"wrote {a.out}/{stem}.json ({len(rows)} rows), maps {maps_path}")
    say("DONE deletion_matrix")


def pareto_record(ns):
    m, a, AT, LZ, coco = ns["m"], ns["a"], ns["AT"], ns["LZ"], ns["coco"]
    b, fn, L, pos, h, mu, P = ns["b"], ns["fn"], ns["L"], ns["pos"], ns["h"], ns["mu"], ns["P"]
    dmean, donors, gidx, crc, r = ns["dmean"], ns["donors"], ns["gidx"], ns["crc"], ns["r"]
    gr, gc, square_grid, kind_set = ns["gr"], ns["gc"], ns["square_grid"], ns["kind_set"]
    zero_h = torch.zeros_like(h)
    trng = np.random.default_rng([a.seed, crc, 3])
    out = {}

    def timed(f):
        sync(); t = time.time(); v = f(); sync()
        return v, time.time() - t

    with AT.SkipTo(m, b, L, fn=fn, pos=pos, h=h, base=mu, log=None) as bh:
        with torch.no_grad():
            s0f = float(AT.hooked_axis(m, bh, fn, L, pos, h[None].to(m.device)).float().cpu()[0])
        AT.one_point(m, bh, fn, L, pos, h, mu)
        AT.path_reading(m, bh, fn, L, pos, h, mu, steps=4)
        for st in a.pareto_vw:
            v, s = timed(lambda st=st: AT.path_reading(m, bh, fn, L, pos, h, mu, steps=st))
            out[f"v_write@{st}"] = (v, s)
        for st in a.pareto_ig:
            v, s = timed(lambda st=st: AT.path_reading(m, bh, fn, L, pos, h, zero_h, steps=st))
            out[f"ig_zero@{st}"] = (v, s)
        v, s = timed(lambda: AT.one_point(m, bh, fn, L, pos, h, mu))
        out["attr_patch@1"] = (v, s)
        v, s = timed(lambda: AT.act_patch(m, bh, fn, L, pos, h, s0f, base=mu, batch=a.int_batch))
        out[f"act_patch@{P}"] = (v, s)
        for ns_ in a.pareto_shap:
            def ks_(ns_=ns_):
                phi, _ = AT.kernel_shap_groups(m, bh, fn, L, pos, h, mu, gidx, nsamples=ns_,
                                               rng=np.random.default_rng([a.seed, crc, 2]), batch=a.int_batch)
                return AT.spread_groups(phi, gidx, P)
            v, s = timed(ks_)
            out[f"kernel_shap@{ns_}"] = (v, s)
        if square_grid:
            for nm in a.pareto_rise:
                def rs_(nm=nm):
                    Mr = AT.rise_masks(gr, gc, nm, s=7, p=0.5, rng=np.random.default_rng([a.seed, crc, 1]))
                    return AT.rise(Mr, AT.rise_scores(m, bh, fn, L, pos, h, mu, Mr, batch=a.int_batch), p=0.5)
                v, s = timed(rs_)
                out[f"rise@{nm}"] = (v, s)
        plan, keys = Plan(h, [mu, dmean, donors[0], zero_h]), []
        for key, (v, _) in out.items():
            o = np.lexsort((trng.random(P), -np.asarray(v, float)))
            for k in (0.1, 0.2):
                top = np.zeros(P, bool); top[o[:max(int(round(k * P)), 1)]] = True
                for j, nm in enumerate(("mean", "dmean", "other0")):
                    plan.add(j, top); keys.append(f"{key}|{nm}|{k}")
        tk = []
        for j, nm in ((1, "dmean"), (3, "zero"), (0, "mean")):
            for gi, sel in enumerate(gidx):
                mk = np.zeros(P, bool); mk[sel] = True
                plan.add(j, mk); tk.append(f"truth|{nm}|{gi}")
        with torch.no_grad():
            got = AT.hooked_axis_many(m, bh, fn, L, pos, plan, batch=a.int_batch)
    after = dict(zip(keys + tk, map(float, got)))
    res = {}
    loc_ok = kind_set == "coco" and (r.get("meta") or {}).get("polys") and square_grid and gr == gc
    if loc_ok:
        cov = coco.cell_coverage(r, gr, gc)
        inside, _ = coco.inside_cells(cov)
    for key, (v, s) in out.items():
        q = dict(secs=s, drop={f"{nm}|{k}": s0f - after[f"{key}|{nm}|{k}"]
                              for nm in ("mean", "dmean", "other0") for k in (0.1, 0.2)})
        for nm in ("dmean", "zero", "mean"):
            tv = np.array([s0f - after[f"truth|{nm}|{gi}"] for gi in range(len(gidx))])
            pred = np.array([np.asarray(v, float)[sel].sum() for sel in gidx])
            q[f"truth_{nm}"] = spearman(pred, tv)
        if loc_ok:
            g = LZ.grade(v, cov, inside)
            q.update(auroc=g["auroc"], hit1=g["hit1"], energy=g["energy"])
        res[key] = q
    return dict(set=kind_set, id=r["id"], family=r.get("family"), n_patches=int(P), s0=s0f, pareto=res)


if __name__ == "__main__":
    main()
