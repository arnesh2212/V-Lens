from __future__ import annotations

import argparse
import glob
import json
import zlib
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VREAD_KEY = "317a51a85fa0ec84"
PERSON = {"person", "man", "woman", "men", "women", "people", "boy", "girl", "child", "kid", "player",
          "lady", "guy", "skier", "surfer", "skateboarder", "snowboarder", "rider", "batter", "catcher"}


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def mentions(text, cat):
    t = set(norm(text).split())
    if cat == "person":
        return bool(t & PERSON)
    words = norm(cat).split()
    return words[-1] in t or words[-1] + "s" in t or (words[-1].endswith("s") and words[-1][:-1] in t)


def items_ss(n, seed):
    from vlens.paths import DATA
    f = os.path.join(str(DATA), "Nuscene Dataset", "Original", "Prompts", "ReadingBinding.json")
    d = json.load(open(f))
    S = d["samples"] if isinstance(d, dict) else d
    by_img = {}
    for s in S:
        if (s.get("probe") or {}).get("subcategory") != "read_named":
            continue
        nat = (s.get("source") or {}).get("native") or {}
        t = nat.get("target_string")
        if not t:
            continue
        by_img.setdefault(nat["image"], [])
        if t not in by_img[nat["image"]]:
            by_img[nat["image"]].append(t)
    rng = np.random.default_rng(seed)
    out = []
    for img, ts in sorted(by_img.items()):
        if len(ts) < 2:
            continue
        i, j = rng.choice(len(ts), 2, replace=False)
        out.append(dict(id=f"ss_{len(out):04d}", set="ss",
                        image=os.path.join(str(DATA), "Nuscene Dataset", img),
                        t1=ts[i], t2=ts[j]))
    rng.shuffle(out)
    return out[: n * 3]


def items_coco(n, seed):
    root = os.path.join(ROOT, "data", "coco_subset")
    ann = json.load(open(os.path.join(root, "annotations", "instances_val2017.json")))
    local = {f for f in os.listdir(os.path.join(root, "val2017"))}
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    imgs = {im["id"]: im for im in ann["images"] if im["file_name"] in local}
    per = {}
    for a in ann["annotations"]:
        if a["image_id"] not in imgs or a.get("iscrowd") or not isinstance(a["segmentation"], list):
            continue
        per.setdefault(a["image_id"], {}).setdefault(a["category_id"], []).append(a)
    out = []
    for iid, bycat in sorted(per.items()):
        im = imgs[iid]
        A = im["width"] * im["height"]
        share = sorted(((sum(x["area"] for x in v) / A, c) for c, v in bycat.items()), reverse=True)
        if len(share) < 2 or share[0][0] < 0.08 or share[1][0] < 0.02:
            continue
        c1, c2 = share[0][1], share[1][1]
        out.append(dict(id=f"coco_{iid}", set="coco", image=os.path.join(root, "val2017", im["file_name"]),
                        W=im["width"], H=im["height"], main=cats[c1], second=cats[c2],
                        polys=[p for x in bycat[c1] for p in x["segmentation"]]))
    rng = np.random.default_rng(seed)
    rng.shuffle(out)
    return out[: n * 2]


def main_run(a):
    import torch
    import vlens
    from vlens import corpus as C
    from vlens.lens import attribution as AT, readout
    from vlens.lens.hooks import ActivationRecorder
    from vlens.lens.metrics import auroc
    from vlens.lens import fitall as FA
    from vlens.data.coco import _raster
    from PIL import Image
    say = lambda *x: print(*x, flush=True)

    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    LAST = m.n_layers - 1
    L = max(int(round(0.5 * LAST)), 1)
    got = FA._load(os.path.join(os.environ.get("VLENS_MAP_CACHE", ""), f"{VREAD_KEY}.pt"))
    VR = {}
    if got is not None and "v_read" in got[0]:
        VR = {int(l): (v[0].float().to(m.device), v[1].float().to(m.device))
              for l, v in got[0]["v_read"].items() if str(l).lstrip("-").isdigit()}
    blocks = sorted(VR)
    say(f"{a.repo}: block {L}/{LAST}; V-Read at {blocks}")
    items = items_ss(a.n, a.seed) if a.set == "ss" else items_coco(a.n, a.seed)
    os.makedirs(a.out, exist_ok=True)
    part = os.path.join(a.out, f"{a.set}.rows.jsonl")
    done = set()
    if os.path.exists(part):
        done = {json.loads(l)["id"] for l in open(part)}
    fh = open(part, "a")
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id} - {None}

    def prompt(it, which):
        if it["set"] == "ss":
            t = it["t1"] if which == "A" else it["t2"]
            return dict(text="What object is it written on? Answer in a few words.",
                        cue=f"Read the text '{t}' in this image.")
        if which == "A":
            return dict(text="Describe the main object in this picture in three to five words.", cue=None)
        return dict(text=f"Describe the {it['second']} in this picture in three to five words.", cue=None)

    @torch.no_grad()
    def greedy(b):
        out = m.hf.generate(**{k: v for k, v in b.items() if not k.startswith("_")},
                            max_new_tokens=a.max_new, do_sample=False,
                            pad_token_id=tok.pad_token_id or tok.eos_token_id)
        ids = out[0, b["input_ids"].shape[1]:].tolist()
        keep = []
        for t in ids:
            if t in eos:
                break
            keep.append(t)
        return keep

    def extend(b, ans):
        S = b["input_ids"].shape[1]
        x = torch.tensor(ans, device=m.device)[None]
        out = {}
        for k, v in b.items():
            if torch.is_tensor(v) and v.ndim == 2 and v.shape[0] == 1 and v.shape[1] == S:
                pad = x if k == "input_ids" else (torch.ones_like(x) if "mask" in k else torch.zeros_like(x))
                out[k] = torch.cat([v, pad.to(v.dtype)], 1)
            else:
                out[k] = v
        return out

    def token_logp(bx, S, ans, pos, X=None):
        n = 1 if X is None else X.shape[0]
        big = m.replicate(bx, n) if n > 1 else bx
        hd = None
        if X is not None:
            def hook(mod, inp, o, _x=X):
                tup = not torch.is_tensor(o)
                hh = (o[0] if tup else o).clone()
                hh[:, pos] = _x.to(hh.device, hh.dtype)
                return (hh, *o[1:]) if tup else hh
            hd = m.layers[L].register_forward_hook(hook)
        try:
            with ActivationRecorder(m.layers, at=[LAST]) as rec:
                m.forward(big)
                h = rec.activations[LAST][:, S - 1: S - 1 + len(ans)]
            lp = torch.log_softmax(m.unembed(h).float(), -1)
            idx = torch.tensor(ans, device=lp.device)[None, :, None].expand(lp.shape[0], -1, 1)
            return lp.gather(-1, idx).squeeze(-1)
        finally:
            if hd is not None:
                hd.remove()

    def ig_map(bA, bB, S, A, B, pos, h, base, which, steps=8, chunk=4):
        dev = (h - base)
        acc = torch.zeros(h.shape[0], dtype=torch.float64)
        al = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
        for s0 in range(0, steps, chunk):
            aa = al[s0:s0 + chunk]
            X = (base[None] + aa[:, None, None] * dev[None]).to(m.device).detach().clone().requires_grad_(True)
            with torch.enable_grad():
                la = token_logp(bA, S, A, pos, X); lb = token_logp(bB, S, B, pos, X)
                f = (la.sum(-1) - lb.sum(-1)) if which is None else (la[:, which] - lb[:, which])
                g, = torch.autograd.grad(f.sum(), X)
            acc += (g.float().cpu().sum(0) * dev).sum(-1).double()
        return (acc / steps).float().numpy()

    donor = None
    t0 = time.time(); n_new = 0
    for it in items:
        if it["id"] in done:
            continue
        if len(done) + n_new >= a.n:
            break
        img = C.load_image(it["image"], square=a.square)
        pa, pb = prompt(it, "A"), prompt(it, "B")
        b = m.chat(pa["text"], [img], cue=pa["cue"])
        A = greedy(b)
        B = greedy(m.chat(pb["text"], [img], cue=pb["cue"]))
        txtA, txtB = tok.decode(A), tok.decode(B)
        if not A or not B or norm(txtA) == norm(txtB):
            continue
        if a.set == "coco" and not mentions(txtA, it["main"]):
            continue
        S = b["input_ids"].shape[1]
        pos = readout.image_positions(m, b)
        P = len(pos); G = int(round(P ** 0.5))
        bA, bB = extend(b, A), extend(b, B)
        with torch.no_grad():
            with ActivationRecorder(m.layers, at=[L]) as rec:
                m.forward(b)
                h = rec.activations[L][0, pos].float().cpu()
        mu = h.mean(0, keepdim=True).expand_as(h)
        jd = next((j for j in range(min(len(A), len(B))) if A[j] != B[j]), None)
        w_seq = ig_map(bA, bB, S, A, B, pos.to(m.device), h, mu, None)
        w_fd = ig_map(bA, bB, S, A, B, pos.to(m.device), h, mu, jd) if jd is not None else None
        k = max(int(round(0.2 * P)), 1)
        top = torch.as_tensor(np.argsort(-w_seq)[:k].copy())
        rnd = torch.as_tensor(np.random.default_rng(len(done) + n_new).permutation(P)[:k])
        Xs = [h]
        for idx, src in ((top, mu), (rnd, mu)):
            x = h.clone(); x[idx] = src[idx]; Xs.append(x)
        if donor is not None and donor.shape == h.shape:
            x = h.clone(); x[top] = donor[top]; Xs.append(x)
        Xb = torch.stack(Xs).to(m.device)
        with torch.no_grad():
            f = (token_logp(bA, S, A, pos.to(m.device), Xb).sum(-1)
                 - token_logp(bB, S, B, pos.to(m.device), Xb).sum(-1)).float().cpu().numpy()
        dele = dict(score=float(f[0]), top_mean=float(f[0] - f[1]), rand_mean=float(f[0] - f[2]),
                    top_other=float(f[0] - f[3]) if len(f) > 3 else None)
        donor = h
        read = {}
        if blocks:
            with torch.no_grad():
                rs = {}
                for tag, bx, ans in (("A", bA, A), ("B", bB, B)):
                    with ActivationRecorder(m.layers, at=blocks) as rec:
                        m.forward(bx)
                        Hs = {l: rec.activations[l][0, S - 1: S - 1 + len(ans)].float() for l in blocks}
                    idx = torch.tensor(ans, device=m.device)
                    for lens in ("v_read", "logit"):
                        lp_l, top1 = [], []
                        for l in blocks:
                            z = Hs[l] if lens == "logit" else Hs[l] @ VR[l][0].T + VR[l][1]
                            lp = torch.log_softmax(m.unembed(z.to(torch.bfloat16)).float(), -1)
                            lp_l.append(lp.gather(-1, idx[:, None]).squeeze(-1).cpu().numpy())
                            top1.append((lp.argmax(-1) == idx).float().cpu().numpy())
                        rs[(tag, lens)] = (np.stack(lp_l), np.stack(top1))
                for lens in ("v_read", "logit"):
                    read[lens] = dict(margin=[float(x) for x in rs[("A", lens)][0].sum(1) - rs[("B", lens)][0].sum(1)],
                                      top1_first=[float(x) for x in rs[("A", lens)][1][:, 0]],
                                      top1_rest=[float(x) for x in (rs[("A", lens)][1][:, 1:].mean(1)
                                                                    if len(A) > 1 else np.full(len(blocks), np.nan))])
        row = dict(id=it["id"], set=a.set, A=txtA, B=txtB, lenA=len(A), lenB=len(B),
                   first_same=bool(A[0] == B[0]), first_diff=jd, block=L, grid=G, deletion=dele,
                   spearman_seq_firstdiff=(float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(w_seq, w_fd)[0])
                                           if w_fd is not None else None),
                   map=[float(x) for x in np.asarray(w_seq, np.float16)],
                   map_fd=[float(x) for x in np.asarray(w_fd, np.float16)] if w_fd is not None else None,
                   blocks=blocks, read=read)
        if a.set == "coco":
            sq = np.asarray(Image.fromarray((_raster(it["polys"], it["W"], it["H"]) * 255).astype(np.uint8))
                            .resize((a.square, a.square), Image.NEAREST), np.float32) / 255.0
            cov = AT.pool_mask(sq, G)
            on = cov > 0.1
            row["main"], row["second"] = it["main"], it["second"]
            row["mask"] = {}
            if on.any() and not on.all():
                for nm, w in (("seq", w_seq), ("firstdiff", w_fd),
                              ("random", np.random.default_rng([7, zlib.crc32(str(it["id"]).encode())]).standard_normal(P))):
                    if w is None:
                        continue
                    pz = np.clip(w, 0, None)
                    row["mask"][nm] = dict(hit1=bool(on[int(np.argmax(w))]), auroc=float(auroc(on.astype(int), w)),
                                           energy=float(pz[on].sum() / max(pz.sum(), 1e-12)), chance=float(on.mean()))
        fh.write(json.dumps(row) + "\n"); fh.flush()
        n_new += 1
        if n_new % 10 == 1:
            el = time.time() - t0
            say(f"  {it['id']}: {n_new} done ({el / 60:.1f} min, {el / n_new:.1f} s each) A={txtA!r} B={txtB!r} "
                f"fall top {dele['top_mean']:+.2f} rand {dele['rand_mean']:+.2f}")
    fh.close()
    say("DONE multitoken", a.set)


def summarise(a):
    for path in sorted(glob.glob(os.path.join(a.out, "*.rows.jsonl"))):
        R = [json.loads(l) for l in open(path)]
        if not R:
            continue
        st = R[0]["set"]
        rng = np.random.default_rng(0)

        def ci(x):
            x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
            if len(x) == 0:
                return "n/a"
            bs = [rng.choice(x, len(x)).mean() for _ in range(1000)]
            return f"{x.mean():+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]"
        print(f"== {st}: n={len(R)}  mean answer tokens A {np.mean([r['lenA'] for r in R]):.1f} "
              f"B {np.mean([r['lenB'] for r in R]):.1f}; multi-token A {np.mean([r['lenA'] > 1 for r in R]):.2f}; "
              f"first tokens identical (first-token score undefined) {np.mean([r['first_same'] for r in R]):.2f}")
        print(f"   deletion (fall in sequence score m, 20% of image tokens at block {R[0]['block']}):")
        print(f"     V-Write top -> picture mean   {ci([r['deletion']['top_mean'] for r in R])}")
        print(f"     V-Write top -> other picture  {ci([r['deletion']['top_other'] for r in R])}")
        print(f"     random      -> picture mean   {ci([r['deletion']['rand_mean'] for r in R])}")
        print(f"     score m itself                {ci([r['deletion']['score'] for r in R])}")
        sp = [r["spearman_seq_firstdiff"] for r in R if r["spearman_seq_firstdiff"] is not None]
        print(f"   Spearman(sequence map, first-differing-token map): median {np.median(sp):.2f} (n={len(sp)})")
        if st == "coco":
            for nm in ("seq", "firstdiff", "random"):
                M = [r["mask"][nm] for r in R if nm in r.get("mask", {})]
                if M:
                    print(f"   mask {nm:10s} hit@1 {np.mean([x['hit1'] for x in M]):.2f}  AUROC {np.mean([x['auroc'] for x in M]):.2f}"
                          f"  energy {np.mean([x['energy'] for x in M]):.2f}  chance {np.mean([x['chance'] for x in M]):.2f}  n={len(M)}")
        if R[0].get("read"):
            bl = R[0]["blocks"]
            for lens in ("v_read", "logit"):
                mg = np.array([r["read"][lens]["margin"] for r in R])
                t1 = np.array([r["read"][lens]["top1_first"] for r in R])
                tr = np.array([r["read"][lens]["top1_rest"] for r in R], float)
                print(f"   {lens:7s} block: " + " ".join(f"{b:>5d}" for b in bl))
                print(f"     read prefers A    " + " ".join(f"{x:5.2f}" for x in (mg > 0).mean(0)))
                print(f"     top-1 = 1st token " + " ".join(f"{x:5.2f}" for x in t1.mean(0)))
                print(f"     top-1 = later tok " + " ".join(f"{x:5.2f}" for x in np.nanmean(tr, 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=("run", "summarise"))
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--set", choices=("ss", "coco"), default="ss")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--out", default="results/extended/j1")
    a = ap.parse_args()
    os.chdir(ROOT)
    summarise(a) if a.cmd == "summarise" else main_run(a)


if __name__ == "__main__":
    main()
