from __future__ import annotations

import argparse
import collections
import difflib
import glob
import importlib.util
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

MODELS = ["Qwen3-VL-8B-Instruct", "Qwen3.5-2B", "Qwen3.5-9B", "Qwen3.5-27B", "gemma-4-12B-it",
          "gemma-4-26B-A4B-it", "gemma-4-31B-it", "InternVL3_5-8B-HF"]
HARD_STANDING = {"which_sign", "authority_judgement", "which_is_official", "is_official", "p3", "p6", "p8"}
READING = {"read_named", "read_neutral", "read", "p4", "p4ref"}
PLACEHOLDER = re.compile(r"^\s*\(|carrier of the target string|scene-based|use each record|should not matter", re.I)


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def category(r):
    fam, sub = r["family"], r["subcategory"]
    if fam.startswith("gemini"):
        if sub in ("p4", "p4ref"):
            return "legibility probe: generated scenes (exhaustive list / exact text)"
        return "hard: generated standing scene (distractors, D0-D3, swap/removal)"
    if fam.startswith("wmdp"):
        return "hard: WMDP keys (sandbagging prompts)"
    if sub == "read_decoy":
        return "hard: absent string"
    if sub in HARD_STANDING:
        return "hard: source standing"
    if "_edit_" in fam and sub not in READING:
        return "hard: authority manipulation (edited photographs)"
    if sub in READING:
        return "plain: reading / binding control"
    if sub in ("binding_swap", "bearer_removal") or r.get("category") in ("binding_swap", "bearer_removal"):
        return "plain: edited binding (swap / removal)"
    if r.get("category") == "eval_awareness":
        return "plain: task under evaluation framing"
    if fam.startswith("gemini"):
        return "plain: generated authority scene"
    return "plain: operative authority question"


def source_of(fam):
    if fam.startswith("gemini"):
        return "generated scenes"
    if "_edit_" in fam:
        return "edited photographs"
    if "_orig_" in fam:
        return "original photographs"
    return "other"


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def partial_sim(target, reply):
    t, r = _norm(target), _norm(reply)
    if not t or not r:
        return 0.0
    if t in r:
        return 1.0
    best = 0.0
    L = len(t)
    for i in range(0, max(len(r) - L + 1, 1)):
        best = max(best, difflib.SequenceMatcher(None, t, r[i:i + L]).ratio())
        if best >= 0.999:
            break
    return best


def consensus(a, G):
    rec = collections.defaultdict(dict)
    rows = {}
    replies = collections.defaultdict(dict)
    have = collections.Counter()
    for m in MODELS:
        files = sorted(glob.glob(os.path.join(ROOT, a.judged, f"{m}__*.json")))
        have[m] = len(files)
        for f in files:
            fam = os.path.basename(f)[len(m) + 2:-5]
            try:
                d = json.load(open(f))
            except Exception:
                continue
            for r in d.get("rows", []):
                r.setdefault("family", fam)
                if r["family"].startswith("wmdp_pw_fine"):
                    continue
                if r.get("expected") not in (None, "") and PLACEHOLDER.search(str(r.get("expected"))):
                    have["placeholder_reference_rows"] += 1
                    continue
                c = G.correct(r)
                if c is None:
                    continue
                k = (r["family"], str(r["id"]))
                rec[k][m] = bool(c)
                rows.setdefault(k, r)
                replies[k][m] = r.get("reply")
    return rec, rows, replies, have


def edit_locality(a):
    from PIL import Image, ImageFilter
    from scipy.ndimage import gaussian_filter
    C = _load_mod("composite", os.path.join(HERE, "composite.py"))
    man = json.load(open(os.path.join(ROOT, a.manifest)))
    ss_root = os.environ.get("VLENS_DATA") or os.path.join(ROOT, "data", "simonsays")
    out = []
    for rel, ent in man["pictures"].items():
        if not ent.get("source"):
            continue
        ep, sp = os.path.join(ss_root, rel), os.path.join(ss_root, ent["source"])
        if not (os.path.exists(ep) and os.path.exists(sp)):
            continue
        src = Image.open(sp).convert("RGB")
        edi = Image.open(ep).convert("RGB")
        if abs(edi.size[0] / edi.size[1] - src.size[0] / src.size[1]) > 0.02:
            out.append(dict(rel=rel, source=source_of_path(rel), excluded="aspect ratio changed"))
            continue
        W, H = src.size
        f = min(1.0, a.side / max(W, H))
        size = (max(int(W * f), 32), max(int(H * f), 32))
        s, e = src.resize(size, Image.LANCZOS), edi.resize(size, Image.LANCZOS)
        m = C.full_mask(s, e)
        region, cells = C.cell_mask(m)
        if not cells.any():
            out.append(dict(rel=rel, source=source_of_path(rel), excluded="no edit cell"))
            continue
        outside = ~region
        gs = np.asarray(s.convert("L"), np.float64)
        ge = np.asarray(e.convert("L"), np.float64)
        dmed = np.asarray(Image.fromarray(np.abs(gs - ge).astype(np.uint8)).filter(ImageFilter.MedianFilter(5)), np.float64)
        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        mu1, mu2 = gaussian_filter(gs, 1.5), gaussian_filter(ge, 1.5)
        s11 = gaussian_filter(gs * gs, 1.5) - mu1 ** 2
        s22 = gaussian_filter(ge * ge, 1.5) - mu2 ** 2
        s12 = gaussian_filter(gs * ge, 1.5) - mu1 * mu2
        ssim = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))
        out.append(dict(rel=rel, source=source_of_path(rel), area_mask=float(m.mean()), area_region=float(region.mean()),
                        out_changed28=float((dmed[outside] > 28).mean()) if outside.any() else 0.0,
                        out_changed8=float((dmed[outside] > 8).mean()) if outside.any() else 0.0,
                        out_ssim=float(ssim[outside].mean()) if outside.any() else 1.0,
                        in_ssim=float(ssim[region].mean())))
    return out


def source_of_path(rel):
    r = rel.lower()
    if r.startswith("gemini"):
        return "generated scenes"
    if r.startswith("openimages"):
        return "Open Images edits"
    if r.startswith("vins"):
        return "VINS edits"
    if r.startswith("nuscene"):
        return "nuScenes edits"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", default="results/extended/judged")
    ap.add_argument("--manifest", default="results/extended/composite_images/manifest.json")
    ap.add_argument("--out", default="results/extended/validation")
    ap.add_argument("--side", type=int, default=1024, help="longer side the locality check works at")
    ap.add_argument("--min-models", type=int, default=6)
    ap.add_argument("--n-flag", type=int, default=40)
    a = ap.parse_args()
    G = _load_mod("g_behaviour", os.path.join(HERE, "g_behaviour.py"))
    rec, rows, replies, have = consensus(a, G)
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    print("judged files per model:", dict(have), flush=True)

    R = []
    for k, d in rec.items():
        n = len(d)
        if n < a.min_models:
            continue
        kk = sum(d.values())
        r = rows[k]
        R.append(dict(family=k[0], id=k[1], cat=category(r), source=source_of_family(k[0]), n=n, k=kk, share=kk / n,
                      sub=r["subcategory"], question=r.get("question"), expected=r.get("expected"),
                      cue=r.get("cue"), kind=r.get("kind"), models=d))
    by_cat = collections.defaultdict(list)
    for x in R:
        by_cat[x["cat"]].append(x)
    by_fam = collections.defaultdict(list)
    for x in R:
        by_fam[x["family"]].append(x)

    def summ(xs):
        s = np.array([x["share"] for x in xs])
        return dict(n=len(xs), mean=float(s.mean()), ge6of8=float(np.mean([x["k"] >= 0.75 * x["n"] for x in xs])),
                    majority=float(np.mean(s > 0.5)), le1=int(sum(x["k"] <= 1 for x in xs)),
                    hist=np.histogram(s, bins=[0, .125, .25, .375, .5, .625, .75, .875, 1.0001])[0].tolist())

    cat_tab = {c: summ(xs) for c, xs in sorted(by_cat.items())}
    fam_tab = {f: summ(xs) for f, xs in sorted(by_fam.items())}
    plain = [x for x in R if x["cat"].startswith("plain")]
    hard = [x for x in R if x["cat"].startswith("hard")]
    flagged = sorted([x for x in plain if x["k"] <= 1], key=lambda x: (x["k"], x["family"]))
    strong = ["gemma-4-31B-it", "Qwen3.5-27B", "Qwen3-VL-8B-Instruct"]

    acc = {m: np.mean([x["models"][m] for x in R if m in x["models"] and x["cat"].startswith("plain")]) for m in MODELS}
    strong = sorted(MODELS, key=lambda m: -acc[m])[:3]
    leg = {}
    def maj(x):
        v = [x["models"][m] for m in strong if m in x["models"]]
        return None if len(v) < 2 else (sum(v) * 2 > len(v))
    for src_name, pred in (("original photographs: named string read", lambda x: x["source"] == "original photographs" and x["sub"] == "read_named"),
                           ("edited photographs: named string attributed to the right carrier", lambda x: x["source"] == "edited photographs" and x["sub"] in ("read_named", "read"))):
        xs = [maj(x) for x in R if pred(x)]
        xs = [v for v in xs if v is not None]
        leg[src_name] = dict(n=len(xs), ok=float(np.mean(xs)) if xs else float("nan"))
    rec_ok, tgt_ok = [], []
    for k, x in rows.items():
        if k[0].startswith("gemini") and x["subcategory"] == "p4ref" and x.get("expected"):
            tg = [t.split(":", 1)[-1].strip() for t in str(x["expected"]).split(";") if t.strip()]
            per = [max((partial_sim(t, replies[k].get(m)) for m in MODELS if replies[k].get(m)), default=0.0) for t in tg]
            if per:
                tgt_ok += [v >= 0.8 for v in per]
                rec_ok.append(all(v >= 0.8 for v in per))
    leg["generated scenes: exact text of each named carrier read by at least one model (sim >= 0.8)"] = dict(
        n=len(tgt_ok), ok=float(np.mean(tgt_ok)) if tgt_ok else float("nan"))
    leg["generated scenes: records with every named carrier's text read (sim >= 0.8)"] = dict(
        n=len(rec_ok), ok=float(np.mean(rec_ok)) if rec_ok else float("nan"))

    cache = os.path.join(ROOT, a.out, "locality_rows.json")
    if os.path.exists(cache):
        loc = json.load(open(cache))
    else:
        loc = edit_locality(a)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(loc, open(cache, "w"))
    loc_ok = [x for x in loc if "excluded" not in x]
    loc_tab = {}
    for src in sorted({x["source"] for x in loc_ok}) + ["all"]:
        xs = loc_ok if src == "all" else [x for x in loc_ok if x["source"] == src]
        if not xs:
            continue
        loc_tab[src] = dict(n=len(xs), mask_area=float(np.median([x["area_mask"] for x in xs])),
                            region_area=float(np.median([x["area_region"] for x in xs])),
                            out_changed28_median=float(np.median([x["out_changed28"] for x in xs])),
                            out_changed28_p90=float(np.percentile([x["out_changed28"] for x in xs], 90)),
                            out_changed8_median=float(np.median([x["out_changed8"] for x in xs])),
                            out_ssim_median=float(np.median([x["out_ssim"] for x in xs])),
                            out_ssim_p10=float(np.percentile([x["out_ssim"] for x in xs], 10)),
                            in_ssim_median=float(np.median([x["in_ssim"] for x in xs])))
    excluded = collections.Counter(x.get("excluded") for x in loc if "excluded" in x)

    js = dict(models=MODELS, strongest=strong, model_accuracy={m: float(v) for m, v in acc.items()},
              n_records=len(R), categories=cat_tab, families=fam_tab,
              plain=summ(plain) if plain else None, hard=summ(hard) if hard else None,
              flagged=[{k: v for k, v in x.items() if k != "models"} | {"agree": f"{x['k']}/{x['n']}"} for x in flagged],
              legibility=dict(leg), locality=loc_tab, locality_excluded=dict(excluded))
    json.dump(js, open(os.path.join(ROOT, a.out, "validation.json"), "w"), indent=1, default=str)

    L = ["# SimonSays validation with open models", "",
         f"Eight open VLMs ({', '.join(MODELS)}) graded against the set's references with the open-source "
         "text judge (Qwen3.5-9B, text only) and the judge-free reading rule; records with >= "
         f"{a.min_models} graded models: {len(R)}.", "",
         "## 1. Reference validity by open-model consensus", "",
         "| category | records | mean share agreeing | >= 6/8 agree | majority agrees | <= 1 agrees |",
         "|---|---|---|---|---|---|"]
    for c, t in cat_tab.items():
        L.append(f"| {c} | {t['n']} | {t['mean']:.2f} | {t['ge6of8']:.0%} | {t['majority']:.0%} | {t['le1']} |")
    if plain:
        p = summ(plain)
        L += ["", f"**Plain items (labels where disagreement would indicate an error): {p['n']} records; "
              f"{p['ge6of8']:.0%} have >= 6/8 open models agreeing with the reference, {p['majority']:.0%} a majority; "
              f"{p['le1']} ({p['le1'] / max(p['n'], 1):.1%}) have <= 1 agreeing and are flagged for inspection.**"]
    if hard:
        h = summ(hard)
        L += [f"Hard-by-design items: {h['n']} records, mean agreement {h['mean']:.2f} (disagreement here is the measured behaviour)."]
    L += ["", "### Per family", "", "| family | records | mean | >= 6/8 | <= 1 |", "|---|---|---|---|---|"]
    for f, t in fam_tab.items():
        L.append(f"| {f} | {t['n']} | {t['mean']:.2f} | {t['ge6of8']:.0%} | {t['le1']} |")
    L += ["", f"### Flagged plain records (<= 1 of >= {a.min_models} models agree; first {a.n_flag} of {len(flagged)})", "",
          "| family | id | sub | agree | question | reference |", "|---|---|---|---|---|---|"]
    for x in flagged[: a.n_flag]:
        q = str(x["question"] or "").replace("|", "/").replace("\n", " ")[:90]
        e = str(x["expected"] or "").replace("|", "/")[:40]
        L.append(f"| {x['family']} | {x['id']} | {x['sub']} | {x['k']}/{x['n']} | {q} | {e} |")
    L += ["", "## 2. Inscription legibility", "",
          f"Strongest three models by agreement on plain items: {', '.join(strong)}.", "",
          "| check | records | passes |", "|---|---|---|"]
    for name, t in leg.items():
        extra = f" (median similarity {t['median_sim']:.2f})" if "median_sim" in t else ""
        L.append(f"| {name} | {t['n']} | {t['ok']:.0%}{extra} |")
    L += ["", "## 3. Edit locality", "",
          "Outside the edit region (A.4 mask's token cells, grown by one cell, as the I3 composite uses):", "",
          "| source | pictures | mask area (median) | region area | changed > 28/255 outside (median / p90) | changed > 8/255 outside | SSIM outside (median / p10) | SSIM inside |",
          "|---|---|---|---|---|---|---|---|"]
    for src, t in loc_tab.items():
        L.append(f"| {src} | {t['n']} | {t['mask_area']:.3f} | {t['region_area']:.3f} | {t['out_changed28_median']:.4f} / "
                 f"{t['out_changed28_p90']:.4f} | {t['out_changed8_median']:.3f} | {t['out_ssim_median']:.3f} / {t['out_ssim_p10']:.3f} | "
                 f"{t['in_ssim_median']:.3f} |")
    L.append(f"\nExcluded: {dict(excluded)}")
    p = summ(plain) if plain else None
    allc = loc_tab.get("all", {})
    L += ["", "## Summary paragraph", "",
          (f"Beyond the human pass described in B.10, we validated SimonSays with eight open VLMs from three families "
           f"(no closed models). On the {p['n'] if p else 0} plain items whose labels could be wrong (reading and binding "
           f"controls, and operative questions on unedited photographs), {p['ge6of8']:.0%} have at least six of eight "
           f"models agreeing with the reference and {p['majority']:.0%} a majority; only {p['le1'] if p else 0} "
           f"({(p['le1'] / max(p['n'], 1)) if p else 0:.1%}) have at most one model agreeing, and we list them for inspection. "
           f"The inscriptions are legible to open models: "
           + "; ".join(f"{name} {t['ok']:.0%} (n={t['n']})" for name, t in leg.items())
           + f". Edits are local: outside the edit region a median {allc.get('out_changed28_median', float('nan')):.2%} of pixels "
           f"change by more than 28/255 (median SSIM {allc.get('out_ssim_median', float('nan')):.3f}), and the composite control (I3) "
           "shows that removing even these residual changes leaves 86–100% of answers and every measured edit effect unchanged.")]
    open(os.path.join(ROOT, a.out, "validation.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:40]), flush=True)
    print("wrote", os.path.join(a.out, "validation.md"), flush=True)


def source_of_family(fam):
    return source_of(fam)


if __name__ == "__main__":
    main()
