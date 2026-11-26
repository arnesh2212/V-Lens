from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import numpy as np

OUT = os.path.join(ROOT, "results", "extended", "h1")
SKIP_FAMILIES = {"authority_figure", "eval_awareness", "lab_eval", "hospital_eval", "classroom_eval", None}
OPS = {"dn": [2, 3, 4, 6, 8], "bl": [0.002, 0.004, 0.008, 0.015, 0.03], "ct": [0.6, 0.4, 0.25, 0.12, 0.05],
       "jp": [30, 15, 8, 4, 1], "oc": [0.25, 0.5, 0.75]}
READ_Q = "What does the sign in this picture say? Reply with the text only."
VARIANTS = ["src", "l0"] + [f"{op}{i + 1}" for op, lv in OPS.items() for i in range(len(lv))]


def level_of(v):
    if v == "src":
        return ("src", -1)
    if v == "l0":
        return ("l0", 0)
    return (v[:2], int(v[2:]))


def region_mask(src_path, edit_img, thr=28):
    from PIL import Image, ImageFilter
    s = Image.open(src_path).convert("L").resize(edit_img.size, Image.LANCZOS)
    e = edit_img.convert("L")
    d = np.abs(np.asarray(s, np.int16) - np.asarray(e, np.int16)).astype(np.uint8)
    d = Image.fromarray(d).filter(ImageFilter.MedianFilter(5))
    b = Image.fromarray(((np.asarray(d) > thr) * 255).astype(np.uint8))
    k = max(3, int(0.02 * min(edit_img.size)) | 1)
    b = b.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.GaussianBlur(1.5))
    return np.asarray(b, np.float32) / 255.0


def degrade(img, M, op, val):
    from PIL import Image, ImageFilter
    import io
    a = np.asarray(img, np.float32)
    W, H = img.size
    if op == "dn":
        small = img.resize((max(1, W // val), max(1, H // val)), Image.BICUBIC)
        d = np.asarray(small.resize((W, H), Image.BICUBIC), np.float32)
    elif op == "bl":
        d = np.asarray(img.filter(ImageFilter.GaussianBlur(max(0.5, val * min(W, H)))), np.float32)
    elif op == "ct":
        w = M[..., None]
        mu = (a * w).sum((0, 1)) / max(w.sum(), 1e-6)
        d = mu + val * (a - mu)
    elif op == "jp":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(val))
        d = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), np.float32)
    elif op == "oc":
        ys, xs = np.nonzero(M > 0.5)
        w = M[..., None]
        mu = (a * w).sum((0, 1)) / max(w.sum(), 1e-6)
        d = a.copy()
        if len(xs):
            x0, x1 = xs.min(), xs.max() + 1
            n = 8
            edges = np.linspace(x0, x1, n + 1).astype(int)
            k = int(round(val * n))
            pick = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int)) if k else []
            for j in pick:
                d[:, edges[j]:edges[j + 1]] = mu
    else:
        raise ValueError(op)
    out = a * (1 - M[..., None]) + d * M[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _sample(rec, *, img, v, cat, question, cue, system, expected, pair, area, efam):
    sid = f"{pair:04d}"
    return dict(
        id=f"legib_{v}_{cat}_{sid}", group_id=f"legib:{sid}",
        source=dict(dataset="legib", variant=v, native=dict(source_id=rec["id"])),
        probe=dict(category=cat, subcategory=rec.get("subcategory") or "", condition=v,
                   tags=dict(pair=sid, level=v, source_id=rec["id"], source_family=rec["family"],
                             edit_family=efam, region_area=area)),
        image=dict(path=img, available=True),
        input=dict(system_prompt=system, prompt="", cue=cue, question=question, choices=None),
        target=dict(type="open_ended", expected=expected, answer_index=None))


def build(a):
    from PIL import Image
    from vlens.data import simonsays as SS
    from g_edits import source_path
    cases = json.load(open(os.path.join(ROOT, "results", "extended", "g_edits", "Qwen3-VL-8B-Instruct__g_edits.json")))["cases"]
    ok = [c for c in cases if c["kind"] == "yesno" and c.get("n_mask", 0) > 0 and 0.01 <= c["mask_area"] <= 0.35
          and abs(c["score_edit"] - c["score_src"]) > 1.0 and c.get("edit_family") not in SKIP_FAMILIES]
    by = defaultdict(list)
    for c in sorted(ok, key=lambda c: c["id"]):
        by[c["edit_family"]].append(c)
    pick = []
    while len(pick) < a.n and any(by.values()):
        for f in sorted(by):
            if by[f] and len(pick) < a.n:
                pick.append(by[f].pop(0))
    recs = {r["id"]: r for r in SS.load()}
    img_dir = os.path.join(OUT, "images")
    os.makedirs(img_dir, exist_ok=True)
    files = defaultdict(list)
    manifest = {}
    for k, c in enumerate(pick):
        r = recs.get(c["id"])
        sp = source_path(r) if r else None
        if not r or not sp:
            continue
        edit = Image.open(r["image"]).convert("RGB")
        if max(edit.size) > a.max_side:
            s = a.max_side / max(edit.size)
            edit = edit.resize((int(edit.width * s), int(edit.height * s)), Image.LANCZOS)
        M = region_mask(sp, edit)
        area = float((M > 0.5).mean())
        if area < 0.005:
            continue
        src = Image.open(sp).convert("RGB").resize(edit.size, Image.LANCZOS)
        pics = {"src": src, "l0": edit}
        for op, vals in OPS.items():
            for i, val in enumerate(vals):
                pics[f"{op}{i + 1}"] = degrade(edit, M, op, val)
        paths = {}
        for v, im in pics.items():
            p = os.path.join(img_dir, f"{k:04d}_{v}.jpg")
            im.save(p, quality=92)
            paths[v] = p
        mpath = os.path.join(img_dir, f"{k:04d}_mask.png")
        Image.fromarray((M * 255).astype(np.uint8)).save(mpath)
        manifest[f"{k:04d}"] = dict(case=c["id"], family=r["family"], edit_family=c["edit_family"], area=area,
                                    mask=mpath, images=paths, score_src_8b=c["score_src"], score_edit_8b=c["score_edit"])
        for v, p in paths.items():
            files[(v, "authority")].append(_sample(r, img=p, v=v, cat="authority", question=r["question"], cue=r.get("cue"),
                                                   system=r.get("system"), expected=r.get("expected"), pair=k, area=area,
                                                   efam=c["edit_family"]))
            files[(v, "reading_binding")].append(_sample(r, img=p, v=v, cat="reading_binding", question=READ_Q, cue=None,
                                                         system=None, expected=None, pair=k, area=area, efam=c["edit_family"]))
    pdir = os.path.join(OUT, "prompts")
    os.makedirs(pdir, exist_ok=True)
    extra = []
    for (v, cat), samples in sorted(files.items()):
        p = os.path.join(pdir, f"{v}_{cat}.json")
        json.dump(dict(schema_version="vlens-1.0", dataset_meta=dict(name="legib", variant=v, category=cat),
                       samples=samples), open(p, "w"), indent=1)
        extra.append(["legib", v, p])
    json.dump(extra, open(os.path.join(OUT, "extra_files.json"), "w"), indent=1)
    json.dump(dict(pairs=manifest, variants=VARIANTS, ops=OPS), open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    fams = sorted({f"legib_{v[:4]}_{'auth' if cat == 'authority' else 'read'}" for (v, cat) in files})
    open(os.path.join(OUT, "families_auth.txt"), "w").write(" ".join(f for f in fams if f.endswith("_auth")))
    open(os.path.join(OUT, "families_read.txt"), "w").write(" ".join(f for f in fams if f.endswith("_read")))
    size = sum(os.path.getsize(os.path.join(img_dir, f)) for f in os.listdir(img_dir))
    print(f"{len(manifest)} pairs x {len(VARIANTS)} pictures; {len(os.listdir(img_dir))} files, {size / 2 ** 20:.0f} MB; "
          f"edit families: {dict(sorted(defaultdict(int, {}).items()))}", flush=True)
    cnt = defaultdict(int)
    for m in manifest.values():
        cnt[m["edit_family"]] += 1
    print("pairs by edit family:", dict(sorted(cnt.items())), flush=True)


def vwrite(a):
    import torch
    import vlens
    from PIL import Image
    from vlens import corpus as C
    from vlens.data import simonsays as SS
    from vlens.lens import attribution as AT, readout, scores
    man = json.load(open(os.path.join(OUT, "manifest.json")))["pairs"]
    name = a.repo.split("/")[-1]
    beh = {}
    for f in glob.glob(os.path.join(ROOT, a.results, f"{name}__legib_*_auth.json")):
        for r in json.load(open(f))["rows"]:
            beh[r["id"]] = r
    recs = [r for r in SS.load(datasets=["legib"]) if r["category"] == "authority"]
    if a.limit:
        keep = sorted(man)[: a.limit]
        recs = [r for r in recs if r["meta"].get("pair") in keep]
    m = vlens.load_vlm(a.repo)
    TS = scores.TokenSets(m.tokenizer)
    LAST = m.n_layers - 1
    L = max(int(round(0.5 * LAST)), 1)
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    part = os.path.join(ROOT, a.out, f"{name}__h1_vwrite.jsonl")
    done = set()
    if os.path.exists(part):
        done = {json.loads(l)["id"] for l in open(part)}
    fh = open(part, "a")
    for i, r in enumerate(recs):
        if r["id"] in done:
            continue
        pair = r["meta"]["pair"]
        l0 = beh.get(f"legib_l0_authority_{pair}")
        if l0 is None:
            continue
        sgn = 1.0 if (l0.get("margins") or {}).get("yes_over_no", 0.0) > 0 else -1.0
        fn0 = scores.score_fn(TS, "yesno", None)
        fn = lambda hL, lg, s_=sgn, f_=fn0: s_ * f_(hL, lg)
        img = C.load_image(r["image"], square=a.square)
        b = m.chat(r["question"], [img], system=r["system"], cue=r["cue"])
        pos = readout.image_positions(m, b)
        P = len(pos); G = int(round(P ** 0.5))
        if G * G != P:
            continue
        with torch.no_grad():
            _, H, _, lg0 = AT.states_at(m, b, [L])
        h = H[L]
        s0 = float(fn(None, lg0).float().cpu()[0])
        mu = h.mean(0, keepdim=True).expand_as(h)
        with AT.SkipTo(m, b, L, fn=fn, pos=pos, h=h, base=mu) as bh:
            w = np.asarray(AT.path_reading(m, bh, fn, L, pos, h, mu, steps=8), np.float32)
        Msq = np.asarray(Image.open(man[pair]["mask"]).convert("L").resize((a.square, a.square), Image.BILINEAR),
                         np.float32) / 255.0
        cov = AT.pool_mask(Msq, G)
        on = cov > 0.1
        pos_w = np.clip(w, 0, None)
        row = dict(id=r["id"], pair=pair, level=r["variant"], score=s0, share=float(pos_w[on].sum() / max(pos_w.sum(), 1e-12)),
                   area=float(on.mean()), hit1=bool(on[int(np.argmax(w))]) if on.any() else None)
        fh.write(json.dumps(row) + "\n"); fh.flush()
        if i % 200 == 0:
            print(f"  {i}/{len(recs)} {r['id']} share {row['share']:.2f} area {row['area']:.2f}", flush=True)
    fh.close()
    print("DONE h1 vwrite", flush=True)


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def analyze(a):
    from sklearn.linear_model import LogisticRegression
    man = json.load(open(os.path.join(OUT, "manifest.json")))["pairs"]
    res_dir = os.path.join(ROOT, a.results)
    name = a.model

    def rows_of(kind):
        out = {}
        for f in glob.glob(os.path.join(res_dir, f"{name}__legib_*_{kind}.json")):
            for r in json.load(open(f))["rows"]:
                out[r["id"]] = r
        return out
    auth, read = rows_of("auth"), rows_of("read")
    if not auth:
        print("no behaviour rows for", name); return
    marg = {(r["meta"].get("pair") if isinstance(r.get("meta"), dict) else r["id"].split("_")[-1], r["variant"]):
            (r.get("margins") or {}).get("yes_over_no") for r in auth.values()}
    ro = {}
    for f in glob.glob(os.path.join(ROOT, a.readouts, f"{name}__legib_*_auth_readouts.json")):
        d = json.load(open(f))
        nb = len(d.get("blocks") or [])
        for r in d["rows"]:
            ro[r["id"]] = (r, nb)
    states = {}
    for f in glob.glob(os.path.join(res_dir, f"{name}__legib_*_auth_states.npz")):
        z = np.load(f)
        for i, rid in enumerate(z["ids"]):
            states[str(rid)] = z["ANS"][i]
    vw = defaultdict(list)
    vp = os.path.join(ROOT, a.vwrite_dir, f"{name}__h1_vwrite.jsonl")
    if os.path.exists(vp):
        for l in open(vp):
            x = json.loads(l)
            vw[x["level"]].append(x)
    pairs = sorted(man)
    have = [p for p in pairs if marg.get((p, "l0")) is not None and marg.get((p, "src")) is not None]
    flip = [p for p in have if np.sign(marg[(p, "l0")]) != np.sign(marg[(p, "src")])]
    good = [p for p in have if abs(marg[(p, "l0")] - marg[(p, "src")]) > 1.0]
    print(f"{name}: {len(pairs)} pairs; {len(good)} moved > 1 nat by the edit; {len(flip)} flipped (used for 'acts on it')", flush=True)

    def eff(get, p, v):
        a0, a1, x = get(p, "src"), get(p, "l0"), get(p, v)
        if a0 is None or a1 is None or x is None or abs(a1 - a0) < 1e-6:
            return np.nan
        return (x - a0) / (a1 - a0)

    probe = {}
    if states:
        S0 = next(iter(states.values()))
        nb = S0.shape[0]
        folds = 5
        ps = np.array(good)
        fold_of = {p: i % folds for i, p in enumerate(ps)}
        for b in range(nb):
            pr = defaultdict(list)
            for f in range(folds):
                tr = [p for p in ps if fold_of[p] != f]; te = [p for p in ps if fold_of[p] == f]
                X, y = [], []
                for p in tr:
                    for v, lab in (("l0", 1), ("src", 0)):
                        k = f"legib_{v}_authority_{p}"
                        if k in states:
                            X.append(states[k][b]); y.append(lab)
                if len(set(y)) < 2:
                    continue
                X = np.asarray(X, np.float32); mu, sd = X.mean(0), X.std(0) + 1e-6
                clf = LogisticRegression(C=0.05, max_iter=2000).fit((X - mu) / sd, y)
                for p in te:
                    for v in VARIANTS:
                        k = f"legib_{v}_authority_{p}"
                        if k in states:
                            pr[v].append(float(clf.predict_proba(((states[k][b] - mu) / sd)[None])[0, 1]))
            probe[b] = {v: float(np.mean(x)) for v, x in pr.items()}
    def ro_get(kind, b):
        def g(p, v):
            x = ro.get(f"legib_{v}_authority_{p}")
            if not x:
                return None
            arr = x[0].get(kind)
            if not arr or b >= len(arr) or arr[b] is None:
                return None
            return float(arr[b])
        return g
    nblocks = next(iter(ro.values()))[1] if ro else 0
    def tr(p, v):
        x = read.get(f"legib_{v}_reading_binding_{p}")
        return _norm(x.get("reply")) if x else None
    table = []
    for v in VARIANTS:
        op, lv = level_of(v)
        acts = [float(np.sign(marg[(p, v)]) == np.sign(marg[(p, "l0")])) for p in flip if marg.get((p, v)) is not None]
        be = [eff(lambda p_, v_: marg.get((p_, v_)), p, v) for p in good]
        sim = [difflib.SequenceMatcher(None, tr(p, v), tr(p, "l0")).ratio() for p in pairs
               if tr(p, "l0") and tr(p, v) is not None]
        row = dict(level=v, op=op, lv=lv, n=len(acts), acts_on_sign=float(np.mean(acts)) if acts else np.nan,
                   beh_effect=float(np.nanmedian(be)) if be else np.nan,
                   read_similarity=float(np.mean(sim)) if sim else np.nan,
                   vwrite_share=float(np.mean([x["share"] for x in vw.get(v, [])])) if vw.get(v) else np.nan,
                   vwrite_area=float(np.mean([x["area"] for x in vw.get(v, [])])) if vw.get(v) else np.nan)
        if nblocks:
            for kind in ("v_read", "logit"):
                eb = []
                for b in range(nblocks):
                    vals = [eff(ro_get(kind, b), p, v) for p in good]
                    eb.append(float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan)
                row[f"{kind}_effect_by_block"] = eb
                mid = [e for b, e in enumerate(eb) if 0.45 <= b / max(nblocks - 1, 1) <= 0.8 and np.isfinite(e)]
                row[f"{kind}_effect_mid"] = float(np.mean(mid)) if mid else np.nan
        if probe:
            row["probe_by_block"] = [probe[b].get(v, np.nan) for b in sorted(probe)]
            mids = [probe[b].get(v, np.nan) for b in sorted(probe) if 0.45 <= b / max(len(probe) - 1, 1) <= 0.8]
            row["probe_mid"] = float(np.nanmean(mids)) if mids else np.nan
        table.append(row)
    flags = [r["level"] for r in table if r["level"] not in ("src", "l0")
             and r.get("acts_on_sign", 1) <= 0.5 and (r.get("probe_mid", 0) >= 0.7 or r.get("v_read_effect_mid", 0) >= 0.5)]
    out = dict(model=name, n_pairs=len(pairs), n_moved_pairs=len(good), n_flip_pairs=len(flip), table=table, reads_not_acts=flags)
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, a.out, f"{name}__h1_summary.json"), "w"), indent=1, default=float)
    lines = [f"# H1 legibility sweep, {name}", "",
             f"{len(pairs)} inscription edits; {len(good)} moved by more than a nat on this model (effect, readout and probe columns); "
             f"{len(flip)} where the edit flips the yes/no answer (acts column).",
             "acts = share answering as on the undegraded edit; effect = (score - src) / (l0 - src), median; "
             "probe = P(edited) from a probe trained on l0 vs src states, mid-depth, out of fold by pair; "
             "read sim = similarity of the transcription to the l0 transcription; V-Write share on the region (area).", "",
             "| level | acts | behaviour effect | V-Read effect (mid) | logit effect (mid) | probe P(edited) mid | read sim | V-Write share (area) |",
             "|---|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"
    for r in table:
        lines.append(f"| {r['level']} | {f(r['acts_on_sign'])} | {f(r['beh_effect'])} | {f(r.get('v_read_effect_mid'))} | "
                     f"{f(r.get('logit_effect_mid'))} | {f(r.get('probe_mid'))} | {f(r['read_similarity'])} | "
                     f"{f(r['vwrite_share'])} ({f(r['vwrite_area'])}) |")
    lines += ["", f"levels where the state still carries the edit (probe >= 0.7 or V-Read effect >= 0.5) but the answer no longer "
              f"follows it (acts <= 0.5): {flags or 'none'}"]
    open(os.path.join(ROOT, a.out, f"{name}__h1_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--n", type=int, default=120); b.add_argument("--max-side", type=int, default=640)
    v = sub.add_parser("vwrite"); v.add_argument("--repo", required=True); v.add_argument("--results", default="results/extended/h1/behaviour")
    v.add_argument("--out", default="results/extended/h1"); v.add_argument("--square", type=int, default=448); v.add_argument("--limit", type=int, default=0)
    z = sub.add_parser("analyze"); z.add_argument("--model", required=True); z.add_argument("--results", default="results/extended/h1/behaviour")
    z.add_argument("--readouts", default="results/extended/h1/readouts"); z.add_argument("--vwrite-dir", default="results/extended/h1")
    z.add_argument("--out", default="results/extended/h1")
    a = ap.parse_args()
    {"build": build, "vwrite": vwrite, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
