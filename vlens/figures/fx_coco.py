from __future__ import annotations

import glob
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from vlens.figures import paper as P
from vlens.figures import style as st

MODELS = ("Qwen3-VL-8B-Instruct", "Qwen3.5-27B", "gemma-4-26B-A4B-it")
MARK = {"Qwen3-VL-8B-Instruct": "o", "Qwen3.5-27B": "D", "gemma-4-26B-A4B-it": "^"}
MSIZE = {"o": 5.6, "D": 4.6, "^": 5.8}
MARK_WORD = {"o": "circles", "D": "diamonds", "^": "triangles"}

LENSES = ("v_read", "tuned", "jacobian", "logit")
LENSES_X = LENSES + ("v_read_coco",)
RANKS = ("vwrite_mean", "kernel_shap", "ig_zero", "attr_patch", "center", "random")
NAME = {"vwrite_mean": P.NICE["v_write"], "attr_patch": P.NICE["attr_patch"],
        "kernel_shap": P.NICE["kernel_shap"], "ig_zero": "integrated gradients",
        "gradxact_zero": P.NICE["grad_x_act"], "center": "centre prior",
        "random": P.NICE["random"]}
NAME2 = dict(NAME, attr_patch="attribution\npatching", gradxact_zero="gradient x\nactivation",
             ig_zero="integrated\ngradients")
COLOUR = {"vwrite_mean": P.COLOUR["v_write"], "attr_patch": P.COLOUR["attr_patch"],
          "kernel_shap": st.OCHRE, "ig_zero": st.SAGE,
          "gradxact_zero": P.COLOUR["grad_x_act"], "center": st.SUBINK,
          "random": P.COLOUR["random"]}
DASH = {"random": P.DASH["random"], "center": (1.3, 1.5)}
CURVE = {"vwrite_mean": "top", "attr_patch": "top:attr_patch", "kernel_shap": "top:kernel_shap",
         "ig_zero": "top:ig_zero",
         "gradxact_zero": "top:gradxact_zero", "center": "top:center", "random": "top:random"}
KS = (0.05, 0.10, 0.20, 0.30, 0.50)
PATCHDIR = "coco_baselines"
SHOW_LENS = True
MCQ_DASH = (2, 2)
PENDING = "[pending]"


def _json(p):
    try:
        with open(p) as f:
            return P._drop_locked(json.load(f))
    except Exception:
        return None


def _model_order(m):
    return (MODELS.index(m) if m in MODELS else len(MODELS), m)


def _models(results):
    got = set()
    for sub, tail in (("coco_lensbench", "__bench.json"),
                      (PATCHDIR, "__patchcausal.json"),
                      (PATCHDIR, "__patchcausal.rows.jsonl")):
        for q in glob.glob(f"{results}/{sub}/*{tail}"):
            base = os.path.basename(q)
            m = base.split("__")[0]
            if base == f"{m}{tail}":
                got.add(m)
    return sorted(got, key=_model_order)


def _bench(results, m):
    d = _json(f"{results}/coco_lensbench/{m}__bench.json")
    if not d:
        return None
    fams = {f: v for f, v in (d.get("families") or {}).items() if not P.LOCKED.search(f)}
    if "coco_pope" not in fams:
        return None
    bad = [f for f in fams if not f.startswith("coco_")]
    assert not bad, f"the COCO bench scored non-COCO families: {bad}"
    d["families"] = fams
    return d


def _keep_row(r):
    fam = str(r.get("family") or "")
    return fam.startswith("coco") and not P.LOCKED.search(fam)


def _patch(results, m):
    p = f"{results}/{PATCHDIR}/{m}__patchcausal.json"
    d = _json(p)
    if d and d.get("rows"):
        d["rows"] = [r for r in d["rows"] if _keep_row(r)]
        d["partial"] = False
        return d if d["rows"] else None
    rows = []
    try:
        with open(p[: -len(".json")] + ".rows.jsonl") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if _keep_row(r):
                    rows.append(r)
    except OSError:
        return None
    return dict(summary=None, rows=rows, partial=True) if rows else None


def _pictures(results, m, rows):
    d = _json(f"{results}/coco/{m}__coco_pope.json")
    recs = (d or {}).get("rows") or []
    imgs, cats, area = set(), set(), []
    for r in recs:
        mt = r.get("meta") or {}
        imgs.add(mt.get("image_id") or str(r.get("id", "")).rsplit("_", 1)[0])
        if r.get("subcategory") == "present":
            if mt.get("target"):
                cats.add(mt["target"])
            if mt.get("area") is not None:
                area.append(float(mt["area"]))
    if not imgs:
        imgs = {str(r.get("id", "")).rsplit("_", 1)[0] for r in rows}
        cats = {r.get("target") for r in rows if r.get("target")}
        area = [float(r["area"]) for r in rows if r.get("area") is not None]
    return dict(n=len(imgs), cats=len(cats), from_behaviour=bool(recs),
                area=(min(area), max(area)) if area else None)


def _mean(v):
    v = [float(x) for x in v if x is not None and np.isfinite(float(x))]
    return float(np.mean(v)) if v else float("nan")


def _close(a, b, tol=1e-6):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= tol)))


def _at_half(depths, ys):
    j = int(np.argmin(np.abs(np.asarray(depths, float) - 0.5)))
    return j, float(ys[j])


def _first_over(depths, ys, thr=0.9):
    for d, y in zip(depths, ys):
        if y is not None and np.isfinite(y) and y >= thr:
            return float(d)
    return None


def _bench_numbers(d):
    v = dict(corpus=(d.get("protocol") or {}).get("corpus_prompts"),
             n_fit=(d.get("protocol") or {}).get("n_fit_ids"), kl_on=d.get("kl_on"))
    layers, depths = d.get("layers") or [], d.get("depths") or []
    if layers and depths and depths[-1] > 0:
        v["n_layers"] = int(round(layers[-1] / depths[-1])) + 1
    for fam in ("coco_pope", "coco_mcq"):
        f = d["families"].get(fam)
        if not f:
            continue
        g = dict(n=int(f["n"]), depths=list(map(float, f["depths"])), layers=list(f["layers"]),
                 lenses=[k for k in LENSES_X if f"model|{k}" in f["auroc"]])
        j, _ = _at_half(g["depths"], g["depths"])
        g["half_block"], g["half_depth"] = int(g["layers"][j]), g["depths"][j]
        g["half"] = {k: float(f["auroc"][f"model|{k}"][j]) for k in g["lenses"]}
        g["half_ci"] = {k: (float(f["ci"][f"model|{k}"][0][j]), float(f["ci"][f"model|{k}"][1][j]))
                        for k in g["lenses"] if f"model|{k}" in (f.get("ci") or {})}
        g["first90"] = {k: _first_over(g["depths"], f["auroc"][f"model|{k}"]) for k in g["lenses"]}
        v[fam] = g
    kl = d.get("kl") or {}
    if kl and layers and depths:
        j, _ = _at_half(depths, depths)
        v["kl_block"], v["kl_depth"] = int(layers[j]), float(depths[j])
        v["kl_half"] = {k: float(kl[k][j]) for k in LENSES_X if k in kl and len(kl[k]) == len(layers)}
        v["kl_tokens"] = d.get("kl_tokens")
    fit = d.get("fit_ids") or (d.get("protocol") or {}).get("fit_ids") or []
    assert not any(str(i).startswith("coco_") for i in fit), "a COCO record is in the lens fit"
    assert not any(P.LOCKED.search(str(i)) for i in fit), "a password-study record is in the lens fit"
    return v


def _common(rows, meths):
    return [r for r in rows if all(f"{m}|{k}|top" in (r.get("drop") or {}) for m in meths for k in KS)]


def _grid(r):
    g = r.get("grid")
    if isinstance(g, (list, tuple)) and len(g) == 2:
        return int(g[0]), int(g[1])
    if g:
        return int(g), int(g)
    n = len(r.get("cov") or r.get("inside") or [])
    G = int(round(n ** 0.5))
    return (G, G) if G and G * G == n else (None, None)


def _inside(r):
    if r.get("inside") is not None:
        return np.asarray(r["inside"], bool)
    if r.get("cov") is None:
        return None
    from vlens.data.coco import inside_cells
    return np.asarray(inside_cells(np.asarray(r["cov"], np.float32))[0], bool)


def _field(rows, meth, bins=60, lo=-1.0, hi=2.0):
    H = np.zeros((bins, bins))
    n, out_mass, in_box, box_share = 0, 0.0, [], []
    for r in rows:
        mp = (r.get("maps") or {}).get(meth)
        ins = _inside(r)
        Gr, Gc = _grid(r)
        if mp is None or ins is None or Gr is None:
            continue
        a = np.clip(np.asarray(mp, float), 0, None)
        if Gr * Gc != len(ins) or len(a) != len(ins) or a.sum() <= 0 or not ins.any():
            continue
        w = a / a.sum()
        rr, cc = np.nonzero(ins.reshape(Gr, Gc))
        x0, x1, y0, y1 = cc.min() / Gc, (cc.max() + 1) / Gc, rr.min() / Gr, (rr.max() + 1) / Gr
        yy, xx = np.divmod(np.arange(Gr * Gc), Gc)
        u = ((xx + 0.5) / Gc - x0) / (x1 - x0)
        v = ((yy + 0.5) / Gr - y0) / (y1 - y0)
        h_, _, _ = np.histogram2d(u, v, bins=bins, range=[[lo, hi], [lo, hi]], weights=w)
        H += h_
        out_mass += 1.0 - h_.sum()
        in_box.append(float(w[(u > 0) & (u < 1) & (v > 0) & (v < 1)].sum()))
        box_share.append((x1 - x0) * (y1 - y0))
        n += 1
    if not n:
        return None
    assert abs(H.sum() + out_mass - n) < 1e-6 * max(n, 1), "the field does not hold one unit per item"
    return dict(H=H, n=n, outside=out_mass / n, in_box=float(np.mean(in_box)),
                box=float(np.mean(box_share)), lo=lo, hi=hi)


LOCQ = ("hit1", "hit1_any", "hit5", "iou", "auroc", "auroc_all", "energy",
        "chance_hit1", "chance_hit1_any", "chance_energy")


def _patch_numbers(d):
    rows = d["rows"]
    s = d.get("summary") or {}
    v = dict(partial=d["partial"], n_rows=len(rows))
    if s:
        assert [round(k, 6) for k in s["ks"]] == list(KS), f"ks is {s['ks']}"
        if s.get("n_items") is not None:
            assert int(s["n_items"]) == len(rows), "summary.n_items disagrees with the rows"
        v["block"], v["depth"] = int(s["block"]), float(s["depth"])
        v["n_layers"] = int(round(v["block"] / v["depth"])) + 1 if v["depth"] > 0 else None
        for meth, key in CURVE.items():
            if key in s.get("curve", {}):
                mine = [_mean(r["drop"].get(f"{meth}|{k}|top") for r in rows) for k in KS]
                assert _close(mine, s["curve"][key], 1e-6), f"curve {key} does not reproduce"
        t = s.get("tally") or {}
        if t and int(t.get("kept", -1)) == len(rows):
            v["tally"] = dict(tried=int(t["tried"]), wrong=int(t["wrong"]), ungraded=int(t["ungraded"]))
    v["grid"] = next((_grid(r) for r in rows if r.get("grid")), None)
    v["patches"] = next((int(r["n_patches"]) for r in rows if r.get("n_patches")), None)
    v["fams"] = sorted({r.get("family") for r in rows})
    use_rows = rows
    meths = [m for m in RANKS if any(f"{m}|{KS[0]}|top" in (r.get("drop") or {}) for r in use_rows)]
    use = _common(use_rows, meths)
    v["meths"], v["n_curve"] = meths, len(use)
    v["curve"] = {m: [_mean(r["drop"][f"{m}|{k}|top"] for r in use) for k in KS] for m in meths}
    v["fifth"] = {m: v["curve"][m][KS.index(0.20)] for m in meths}
    graded = [r for r in rows if r.get("loc")]
    v["n_graded"] = len(graded)
    v["n_fallback"] = sum(bool(r.get("inside_fallback")) for r in graded)
    lm = [m for m in RANKS if any(m in r["loc"] for r in graded)]
    v["loc"] = {m: {q: _mean((r["loc"].get(m) or {}).get(q) for r in graded) for q in LOCQ} for m in lm}
    v["auroc_items"] = {m: np.array([r["loc"][m]["auroc"] for r in graded
                                     if m in r["loc"] and r["loc"][m].get("auroc") is not None
                                     and np.isfinite(r["loc"][m]["auroc"])], float) for m in lm}
    v["auroc_med"] = {m: float(np.median(x)) if x.size else float("nan") for m, x in v["auroc_items"].items()}
    v["auroc_above"] = {m: float((x > 0.5).mean()) if x.size else float("nan")
                        for m, x in v["auroc_items"].items()}
    if s.get("localisation"):
        assert int(s.get("localisation_n", len(graded))) == len(graded), "localisation_n disagrees"
        for m, qs in s["localisation"].items():
            for q, x in qs.items():
                if m in v["loc"] and q in v["loc"][m]:
                    assert _close(v["loc"][m][q], x, 1e-6), f"localisation {m}.{q} does not reproduce"
        v["loc_ci"] = s.get("localisation_ci") or {}
        v["paired"] = s.get("localisation_paired") or {}
    if lm:
        for q in ("chance_hit1", "chance_energy"):
            c = [v["loc"][m][q] for m in lm]
            assert np.allclose(c, c[0], equal_nan=True), f"{q} differs between rankings"
    v["chance"] = v["loc"][lm[0]]["chance_hit1"] if lm else float("nan")
    v["chance_energy"] = v["loc"][lm[0]]["chance_energy"] if lm else float("nan")
    md = [r["mask_drop"] for r in rows if r.get("mask_drop")]
    v["mask_n"] = len(md)
    v["mask_drop"] = {k: _mean(x.get(k) for x in md) for k in ["mask|exact"] + [f"{m}|top" for m in RANKS]}
    if isinstance(s.get("mask_drop"), dict):
        for k, x in s["mask_drop"].items():
            if k in v["mask_drop"] and x is not None:
                assert _close(v["mask_drop"][k], x, 1e-6), f"mask_drop {k} does not reproduce"
    v["field"] = {m: _field(graded, m) for m in ("vwrite_mean", "attr_patch")}
    v["has_field"] = all(v["field"].get(m) for m in ("vwrite_mean", "attr_patch"))
    if v["has_field"]:
        v["n_field"] = v["field"]["vwrite_mean"]["n"]
        assert v["field"]["attr_patch"]["n"] == v["n_field"], "the two fields hold different items"
    return v


def _behaviour(results, m):
    import re
    out = {}
    d = _json(f"{results}/coco/{m}__coco_pope.json")
    if d and d.get("rows"):
        said = [(re.match(r"\W*(\w+)", str(r.get("reply") or "")) or [None, ""])[1].lower() == "yes"
                for r in d["rows"]]
        truth = [str(r.get("expected")).lower() == "yes" for r in d["rows"]]
        out["pope_n"] = len(said)
        out["pope_acc"] = float(np.mean([a == b for a, b in zip(said, truth)]))
        out["pope_yes"] = float(np.mean(said))
        tp = sum(a and b for a, b in zip(said, truth))
        fp = sum(a and not b for a, b in zip(said, truth))
        fn = sum(b and not a for a, b in zip(said, truth))
        out["pope_f1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    d = _json(f"{results}/coco/{m}__coco_mcq.json")
    if d and d.get("rows"):
        ok = [(re.match(r"\W*\(?([ABCD])\b", str(r.get("reply") or "")) or [None, None])[1] == r.get("expected")
              for r in d["rows"]]
        out["mcq_n"], out["mcq_acc"] = len(ok), float(np.mean(ok))
    return out


def _numbers(results, models):
    out = {}
    for m in models:
        b, p = _bench(results, m), _patch(results, m)
        if not (b or p):
            continue
        out[m] = dict(bench=_bench_numbers(b) if b else None, patch=_patch_numbers(p) if p else None,
                      raw_bench=b, pict=_pictures(results, m, (p or {}).get("rows") or []),
                      beh=_behaviour(results, m))
    return out


def _check_disjoint():
    try:
        from vlens.data import simonsays as SS
        ids = [str(r["id"]) for r in SS.load(require_image=False)]
    except Exception:
        return None
    assert not any(i.startswith("coco_") for i in ids), "a SimonSays record has a COCO identifier"
    return len(ids)


def _f2(x):
    return PENDING if x is None or not np.isfinite(x) else f"{x:.2f}"


def _pc(x):
    return PENDING if x is None or not np.isfinite(x) else f"{100 * x:.0f} per cent"


def _join(parts):
    parts = [p for p in parts if p]
    if len(parts) <= 2:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _pp(x):
    return PENDING if x is None or not np.isfinite(x) else f"{100 * x:+.0f} points"


BASELINES = ("kernel_shap", "ig_zero", "attr_patch")
LENS_WORD = {"v_read": "V-Read (clay)", "tuned": "the tuned lens (ochre)",
             "jacobian": "the Jacobian lens (slate)", "logit": "the logit lens (grey, dashed)"}


def _the(k):
    n = P.NICE[k]
    return n if n.startswith("V-") else "the " + n


def _cap1(s):
    return s[:1].upper() + s[1:]


def _best(lc, q):
    got = {k: lc[k][q] for k in BASELINES if k in lc and np.isfinite(lc[k][q])}
    if not got:
        return None, float("nan")
    k = max(got, key=got.get)
    return k, got[k]


def _paired(p, other, q):
    pr = (p.get("paired") or {}).get(f"vwrite_mean-{other}", {}).get(q)
    if pr and all(np.isfinite(pr["ci"])):
        return pr
    return None


def _ci_s(pr, pct=False):
    f = (lambda x: f"{100 * x:+.0f}") if pct else (lambda x: f"{x:+.3f}")
    return (f"{f(pr['mean'])}{' points' if pct else ''} (95 per cent bootstrap interval "
            f"{f(pr['ci'][0])} to {f(pr['ci'][1])})")


def _caption(v, n_simonsays=None, pooled=False):
    v = v or {}
    ms = list(v) or ["Qwen3-VL-8B-Instruct", "Qwen3.5-27B"]
    benched = [m for m in ms if (v.get(m) or {}).get("bench")] if v else ms[:1]
    patched = [m for m in ms if (v.get(m) or {}).get("patch")] if v else ms
    first_p = (v.get(patched[0]) or {}).get("patch") if patched and v else None
    field = bool(first_p and first_p.get("has_field"))
    lens_panel = bool(benched)
    L = dict(zip(("a", "b", "c"), ("Left", "Middle", "Right") if lens_panel else ("", "Left", "Right")))
    pic = [(v.get(m) or {}).get("pict") for m in ms if v]
    npic = sorted({p["n"] for p in pic if p})
    ncat = sorted({p["cats"] for p in pic if p and p["cats"]})
    area = next((p["area"] for p in pic if p and p.get("area")), None)
    marks = _join([f"{m} as {MARK_WORD[MARK.get(m, 'o')]}" for m in (patched if len(patched) > 1 else [])])
    blk = [(m, (v.get(m) or {}).get("patch") or {}) for m in patched] if v else []

    wins = [p["loc"]["vwrite_mean"]["hit1"] > max(p["loc"][k]["hit1"] for k in BASELINES + ("center", "random")
                                                   if k in p["loc"])
            for _, p in blk if p.get("loc") and "vwrite_mean" in p["loc"]]
    kl_win = [((v[m]["bench"] or {}).get("kl_half") or {}) for m in benched] if v else []
    kl_win = [k["v_read"] < k["logit"] for k in kl_win if "v_read" in k and "logit" in k]
    lead = ("MAIN. The method carries to a general dataset: on ordinary COCO photographs, V-Write's map "
            + ("points at the object people outlined more often than the best baseline"
               if wins and all(wins) else "points at the object people outlined")
            + " (the better of attribution patching and gradient times activation on each measure), "
            "the centre prior and a random map, and replacing the cells it points at costs the answer "
            "more than replacing theirs")
    if benched:
        lead += ("; V-Read, fitted without labels, reads the answer the model will give from the "
                 "middle of the network" + (" and stays closer to the model's own output distribution "
                                            "than the logit lens" if kl_win and all(kl_win) else ""))
    out = [lead + ". Nothing was fitted on the scored pictures or questions."]

    out.append(
        f"The pictures are {npic[0] if len(npic) == 1 else PENDING} photographs from COCO val2017, "
        "frozen once with seed 0. In each, the present object is a category with no crowd "
        "annotation whose unioned human segmentation covers "
        + (f"between {100 * area[0]:.0f} and {100 * area[1]:.0f} per cent" if area else PENDING)
        + " of the frame, at most ten pictures per category"
        + (f" ({ncat[0]} categories)" if len(ncat) == 1 else "")
        + ". Each picture is asked, with no system prompt, \"Is there a {object} in the image? "
        "Answer yes or no.\" once about the present object and once about the absent object that "
        "most often co-occurs with it in val2017 (the adversarial split of the POPE protocol); the "
        "letter family asks \"Which of these objects is in the image?\" over four named options, "
        "the present object and its three most frequent absent co-occurrers, with the correct "
        "letter rotating over A to D, followed by \"Answer with the letter only.\". Pictures are "
        "resized to 448 by 448 pixels"
        + (f", which the model sees as {first_p['patches']} image tokens"
           + (f" on a {first_p['grid'][0]} by {first_p['grid'][1]} grid" if first_p.get("grid") else "")
           if first_p and first_p.get("patches") else "")
        + ".")
    beh = [(m, (v.get(m) or {}).get("beh") or {}) for m in ms] if v else []
    beh_s = "; ".join([f"{m} answers {_pc(b['pope_acc'])} of {b['pope_n']} yes/no questions correctly "
                       f"(F1 {_f2(b.get('pope_f1'))}, saying yes to {_pc(b['pope_yes'])})"
                       + (f" and {_pc(b['mcq_acc'])} of {b['mcq_n']} letter questions" if b.get("mcq_n") else "")
                       for m, b in beh if b.get("pope_n")])
    if beh_s:
        out.append("Scored by the first word or letter of the greedy reply: " + beh_s + ".")

    if lens_panel:
        drawn = [k for k in LENSES if any(k in (((v.get(m) or {}).get("bench") or {}).get("coco_pope") or {})
                                          .get("lenses", []) for m in benched)] if v else list(LENSES)
        parts = []
        for m in benched:
            g = (((v.get(m) or {}).get("bench") or {}).get("coco_pope")) or {}
            miss = [P.NICE[k] for k in ("tuned", "jacobian") if g and k not in g.get("lenses", [])]
            parts.append(f"{g.get('n', PENDING)} yes/no records on {m}"
                         + (f" (whose bench has no {_join(miss).replace(' and ', ' or ')})" if miss else ""))
        b0 = (v.get(benched[0]) or {}).get("bench") or {} if v else {}
        mcq = [m for m in benched if ((v.get(m) or {}).get("bench") or {}).get("coco_mcq")] if v else []
        coco_fit = [m for m in benched if "v_read_coco" in
                    (((v.get(m) or {}).get("bench") or {}).get("coco_pope") or {}).get("lenses", [])] if v else []
        shared = _join(["V-Read"] + [P.NICE[k] for k in ("tuned", "jacobian") if k in drawn])
        out.append(
            f"{L['a']}: How early each lens reads the answer the model then gives. At each fitted "
            "block the lens maps the hidden state at the answer position (the last prompt token), "
            "the result is read through the model's own final norm and unembedding, and the record "
            "is scored as the log probability of yes minus that of no; the AUROC is taken against "
            "the answer the model gives, which is the sign of the same score at the last block, on "
            + _join(parts) + ". Depth is the block index over the last block's index. Bands are "
            "95 per cent percentile bootstrap intervals over records (300 resamples). "
            + _cap1(_join([LENS_WORD[k] for k in LENSES if k in drawn]))
            + " are drawn; the flat rule is chance, 0.5."
            + (" The thin dashed lines (\"letters\" in the key) are the letter family ("
               + _join([f"{((v[m]['bench'] or {}).get('coco_mcq') or {}).get('n', PENDING)} records on {m}"
                        for m in mcq])
               + "), scored as the log probability of the correct letter minus that of the best "
               "other letter, against whether the model gives the correct letter." if mcq else "")
            + f" {shared} {'are' if ' and ' in shared else 'is'} the one shared fit used on SimonSays, "
            + (f"{b0.get('corpus')} prompts" if b0.get("corpus") else f"{PENDING} prompts")
            + " drawn from SimonSays records that no scored set uses, with no labels and no COCO "
            "picture or question in them"
            + (f"; the {b0['n_fit']} records used in fitting are kept out of every scored set by "
               "identifier" if b0.get("n_fit") else "")
            + (f", and none of the {n_simonsays} SimonSays identifiers is a COCO one (checked)"
               if n_simonsays else "")
            + "."
            + (" The light clay line (\"COCO fit\" in the key) is V-Read fitted, with no labels, on "
               "COCO photographs and their questions themselves; records whose photograph it was "
               "fitted on are left out for every lens." if coco_fit else "")
            + (" The lens bench has not been run on " + _join([m for m in patched if m not in benched])
               + ", so its line is absent from this panel." if [m for m in patched if m not in benched] else ""))
    else:
        out.append("The lens bench on these pictures has not been run for any model yet, so only "
                   "the attribution panels are drawn.")

    where = _join([f"block {p.get('block', PENDING)} of {p.get('n_layers', PENDING)} on {m}"
                   for m, p in blk]) or PENDING
    ngr = _join([f"{p.get('n_graded', PENDING)} on {m}" for m, p in blk]) or PENDING
    cells = ("the object's cells are the image tokens whose cell is at least half covered by the "
             "unioned COCO segmentation of the object (for a thin object that covers no cell by "
             "half, the cells at least a quarter covered)")
    if field:
        fm = _join(patched) if pooled else patched[0]
        nf = sum(p["field"]["vwrite_mean"]["n"] for _, p in blk if p.get("has_field")) if pooled \
            else first_p["field"]["vwrite_mean"]["n"]
        out.append(
            f"{L['b']}: Where the attribution lands, in the object's own frame, on {fm}"
            + (" (items of the models pooled)" if pooled and len(patched) > 1 else "") + ". "
            f"For every present-object item the model answered correctly, {cells}; each cell's "
            "positive attribution is placed at its centre measured against the bounding box of "
            "those cells (across in box widths, down in box heights), so the unit box drawn in ink "
            "is the object whatever its size or place in the picture. Each item is scaled to unit "
            "mass, so a large object counts no more than a small one, and the items are summed on "
            f"a 60 by 60 grid over three box sizes each way ({nf} items) and "
            "smoothed with a Gaussian kernel of 0.05 box sizes. V-Write is the shading, darker "
            "where more attribution lands; attribution patching is the ink contours, which enclose "
            "the densest quarter, half and three quarters of its own mass.")
    else:
        out.append(
            f"{L['b']}: How well each map points at the object, graded against people's outlines. "
            f"For every present-object item the model answered correctly, {cells}. Each ridge is "
            "the density over items (Gaussian kernel of width 0.03, reflected at 0 and 1) of the "
            "AUROC with which a map ranks the object's cells above the cells the object does not "
            f"touch, so 0.5 is chance whatever the object's size. Items: {ngr}.")
    out.append(
        "The maps are V-Write (the answer score's gradient with respect to the image-token states, "
        "integrated along the straight path from the mean image-token state of the same picture "
        "with 8 midpoint steps, times the difference), attribution patching (one gradient at the "
        "real states, times the same difference), gradient times activation (one gradient, from "
        "zero), a centre prior that ranks cells by closeness to the middle of the grid, and a "
        f"random map, all taken at {where}.")

    ns = _join([f"{p.get('n_curve', PENDING)} items on {m}" for m, p in blk]) or PENDING
    tal = [(m, p["tally"]) for m, p in blk if p.get("tally")]
    out.append(
        f"{L['c']}: What each map's choice does. The share of image tokens a map ranks highest "
        "has its hidden states at that block replaced by the mean image-token state of the same "
        "picture, the forward pass continues, and the plot is the mean fall in the answer score "
        "(the log probability of the answer the model gave minus that of the other), at 5, 10, 20, "
        f"30 and 50 per cent of the tokens, on {ns}: the present-object items the model answered "
        "correctly, each measured under every map so every curve averages the same items"
        + ("; " + _join([f"{t['wrong']} of the {t['tried']} tried on {m} were answered wrongly and "
                         "left out" for m, t in tal]) if tal else "")
        + (f". The thick lines, which the names label, are the mean of the models; the thin lines "
           f"are each model's own, {marks}" if marks else "")
        + ". Excluded everywhere: every SimonSays record and the password-locking families.")
    if any(p.get("partial") for _, p in blk):
        out.append("The run behind the attribution panels was still going when this was drawn; "
                   "the counts above are the items measured so far.")

    fin, lim = [], []
    for m, p in blk:
        if not p.get("loc"):
            continue
        lc = p["loc"]
        bk, bv = _best(lc, "hit1")
        fin.append(
            f"On {m} ({p['n_graded']} items) V-Write's single strongest cell lands on the object "
            f"(the pointing game, hit@1) for {_pc(lc['vwrite_mean']['hit1'])} of items, against "
            f"{_pc(bv)} for the best baseline ({NAME[bk]}), {_pc(lc['center']['hit1'])} for the centre "
            f"prior, {_pc(lc['random']['hit1'])} for the random map and {_pc(p['chance'])} for a cell "
            f"drawn at random; hit@5 is {_pc(lc['vwrite_mean']['hit5'])} against "
            f"{_pc(_best(lc, 'hit5')[1])}, the IoU of its top cells (as many as the object has) with "
            f"the object {_f2(lc['vwrite_mean']['iou'])} against {_f2(_best(lc, 'iou')[1])}, the mean "
            f"per-cell AUROC {_f2(lc['vwrite_mean']['auroc'])} against {_f2(_best(lc, 'auroc')[1])} "
            f"({_f2(lc['center']['auroc'])} for the centre prior), and its share of positive mass on "
            f"the object {_pc(lc['vwrite_mean']['energy'])} against {_pc(_best(lc, 'energy')[1])} "
            f"and {_pc(p['chance_energy'])}, the object's share of the frame.")
        pa = [(q, _paired(p, bk, q)) for q in ("hit1", "auroc")]
        pa = [(q, pr) for q, pr in pa if pr]
        if pa:
            fin.append(f"Paired over the same items, V-Write minus {NAME[bk]} is "
                       + _join([f"{_ci_s(pr, q == 'hit1')} on {'hit@1' if q == 'hit1' else 'AUROC'}"
                                for q, pr in pa]) + " (2000 resamples).")
            for q, pr in pa:
                if pr["ci"][0] <= 0:
                    lim.append(f"on {m} V-Write's {'hit@1' if q == 'hit1' else 'AUROC'} lead over "
                               f"{NAME[bk]} is within the bootstrap interval of zero")
        pc = _paired(p, "center", "auroc")
        if pc:
            fin.append(f"Against the centre prior the paired AUROC difference is {_ci_s(pc)}.")
        if p.get("auroc_med", {}).get("center", 0) >= p.get("auroc_med", {}).get("vwrite_mean", 1):
            lim.append(f"on {m} the centre prior's median per-cell AUROC "
                       f"({_f2(p['auroc_med']['center'])}) is at least V-Write's "
                       f"({_f2(p['auroc_med']['vwrite_mean'])}), since COCO objects sit near the middle")
        if p.get("has_field"):
            fv, fa = p["field"]["vwrite_mean"], p["field"]["attr_patch"]
            fin.append(f"In the object's frame on {m}, V-Write puts {_pc(fv['in_box'])} of its "
                       f"positive mass inside the box and attribution patching {_pc(fa['in_box'])}, "
                       f"against {_pc(fv['box'])} for a map spread evenly over the frame.")
        if p.get("fifth"):
            f5 = p["fifth"]
            k5 = max([k for k in BASELINES if k in f5], key=lambda k: f5[k], default=None)
            fin.append(
                f"Replacing a fifth of the tokens on {m} lowers the answer score by "
                f"{_f2(f5.get('vwrite_mean'))} for V-Write against {_f2(f5.get(k5))} for the best "
                f"baseline ({NAME.get(k5, PENDING)}), {_f2(f5.get('center'))} for the centre prior "
                f"and {_f2(f5.get('random'))} for random.")
            losers = [NAME[k] for k in f5 if k != "vwrite_mean" and "vwrite_mean" in f5
                      and np.isfinite(f5[k]) and f5[k] >= f5["vwrite_mean"]]
            if losers:
                lim.append(f"on {m} V-Write's fifth is not the most damaging: {_join(losers)} "
                           "costs the answer at least as much")
        md = p.get("mask_drop") or {}
        if np.isfinite(md.get("mask|exact", np.nan)):
            fin.append(f"Replacing exactly the object's own cells costs {_f2(md['mask|exact'])} on "
                       f"{m}; replacing as many cells from the top of V-Write's map costs "
                       f"{_f2(md.get('vwrite_mean|top'))}, against "
                       f"{_f2(max(md.get(k + '|top', np.nan) for k in BASELINES))} for the best "
                       "baseline: the cells V-Write picks matter more to the answer than the "
                       "outlined object's own cells do.")
        if p.get("n_curve") is not None and p["n_curve"] < 100:
            lim.append(f"on {m} the curves rest on only {p['n_curve']} items")
    for m in benched if v else []:
        b = (v.get(m) or {}).get("bench") or {}
        g = b.get("coco_pope") or {}
        if not g:
            continue
        h, ci = g["half"], g.get("half_ci", {})
        fin.append(
            f"At the block nearest half depth (block {g['half_block']}, depth {g['half_depth']:.2f}) "
            f"on {m} the AUROC for the yes/no answer is "
            + _join([f"{_f2(h[k])} for {_the(k)}"
                     + (f" ({_f2(ci[k][0])} to {_f2(ci[k][1])})" if k in ci else "")
                     for k in LENSES_X if k in h]) + "; "
            + _join([(f"{_the(k)} first reaches 0.9 at depth {g['first90'][k]:.2f}"
                      if g["first90"].get(k) is not None else f"{_the(k)} never reaches 0.9")
                     for k in LENSES_X if k in h]) + ".")
        kh = b.get("kl_half") or {}
        if kh:
            fin.append(
                f"On {b.get('kl_tokens', PENDING)} tokens of COCO prompts, the KL from the model's "
                f"own final distribution at block {b['kl_block']} is "
                + _join([f"{kh[k]:.2f} nats for {_the(k)}" for k in LENSES_X if k in kh]) + ".")
        others = {k: x for k, x in h.items() if k in LENSES and k != "v_read"}
        if "v_read" in h and others:
            best = max(others, key=others.get)
            if h["v_read"] < others[best]:
                lim.append(f"on {m} the {P.NICE[best]} reads the yes/no answer at half depth at least "
                           f"as well as V-Read by AUROC ({_f2(others[best])} against "
                           f"{_f2(h['v_read'])}), so V-Read's edge there is in KL, not in ranking")
        if "v_read_coco" in h and "v_read" in h and h["v_read_coco"] < h["v_read"]:
            lim.append(f"fitting V-Read on COCO itself does not help at half depth on {m} "
                       f"({_f2(h['v_read_coco'])} against {_f2(h['v_read'])} for the SimonSays fit)")
        gm = b.get("coco_mcq") or {}
        if gm.get("half", {}).get("v_read") is not None and gm["half"]["v_read"] < 0.5:
            lim.append(f"on the letter family on {m} every lens is near or below chance at half depth "
                       f"(V-Read {_f2(gm['half']['v_read'])}, logit lens "
                       f"{_f2(gm['half'].get('logit'))}) and V-Read first reaches 0.9 at depth "
                       + (f"{gm['first90']['v_read']:.2f}" if gm['first90'].get('v_read') else "never"))
    if not fin:
        fin.append(f"Findings: {PENDING}.")
    out.append(" ".join(fin))
    if lim:
        out.append("Limits: " + "; ".join(lim) + ". Only present-object items the model answered "
                   "correctly are graded for pointing, so a miss here is never a wrong answer.")
    cap = out[0] + "\n\n" + "\n\n".join(out[1:])
    assert "—" not in cap, "an em dash in the caption"
    return cap


W_IN, H_IN = st.MAX_WIDTH, 4.4
Y0_IN, TOP_IN, KEY_IN = 0.8, 3.75, 0.2
Y0, TOP = Y0_IN / H_IN, TOP_IN / H_IN
LAYOUT = {
    (True, False): ([0.75, 2.1], [4.5, 1.85], [8.35, 2.1]),
    (True, True): ([0.8, 1.85], [5.0, 1.95], [8.2, 1.2]),
    (False, False): ([], [0.3, 3.9], [6.2, 4.2]),
    (False, True): ([], [0.75, 3.6], [5.05, 4.0]),
}


def _rect(x_w, H=H_IN):
    return [x_w[0] / W_IN, Y0_IN / H, x_w[1] / W_IN, (TOP_IN - Y0_IN) / H]


def _letter(fig, ax, s, dx_in=0.62):
    b = ax.get_position()
    fig.text(b.x0 - dx_in / W_IN, (TOP_IN + 0.08) / fig.get_size_inches()[1], s, ha="left", va="bottom", color=st.INK,
             fontsize=matplotlib.rcParams["xtick.labelsize"])


def _chance_word(ax, y=0.5):
    fs = matplotlib.rcParams["xtick.labelsize"] * 0.92
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = []
    for ln in ax.lines:
        xd, yd = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if len(xd) < 3:
            continue
        xy = ln.get_transform().transform(np.c_[xd, yd])
        t = np.linspace(0, 1, 24)[:, None, None]
        pts.append((xy[:-1][None] * (1 - t) + xy[1:][None] * t).reshape(-1, 2))
    pts = np.concatenate(pts) if pts else np.zeros((0, 2))
    pts = pts[np.all(np.isfinite(pts), 1)]
    for x_end, ha, dy, va in ((1.0, "right", 3, "bottom"), (0.0, "left", 3, "bottom"),
                              (1.0, "right", -3, "top"), (0.0, "left", -3, "top"),
                              (0.5, "center", 3, "bottom"), (0.5, "center", -3, "top"),
                              (0.25, "center", 3, "bottom"), (0.75, "center", 3, "bottom"),
                              (0.25, "center", -3, "top"), (0.75, "center", -3, "top")):
        dx = {"right": -3, "left": 3, "center": 0}[ha]
        t = ax.annotate("chance", xy=(x_end, y), xycoords=("axes fraction", "data"), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va=va, color=st.INK, fontsize=fs, zorder=8)
        bb = t.get_window_extent(r).padded(2)
        inside = ((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1))
        if not inside.any():
            return True
        t.remove()
    return False


def _panel_lens(ax, benches):
    drawn, ends = [], {}
    for m, d in benches:
        mk = MARK.get(m, "o")
        for fam, thin in (("coco_mcq", True), ("coco_pope", False)):
            f = d["families"].get(fam)
            if not f:
                continue
            x = np.asarray(f["depths"], float)
            for k in reversed(LENSES_X):
                key = f"model|{k}"
                if key not in f["auroc"]:
                    continue
                y = np.asarray(f["auroc"][key], float)
                c = P.COLOUR[k]
                if thin:
                    if k != "v_read_coco":
                        ax.plot(x, y, color=c, lw=1.0, dashes=MCQ_DASH, zorder=3)
                    continue
                if k == "v_read_coco":
                    ax.plot(x, y, color=c, lw=1.4, dashes=P.DASH[k], zorder=5)
                    if k not in drawn:
                        drawn.append(k)
                    continue
                ci = (f.get("ci") or {}).get(key)
                if ci:
                    st.ribbon(ax, x, ci[0], ci[1], c, alpha=0.16)
                ax.plot(x, y, color=c, lw=2.6 if k == "v_read" else 1.8,
                        dashes=P.DASH.get(k, ()), marker=mk,
                        ms=MSIZE[mk] * (0.9 if k == "v_read" else 0.8),
                        mec="white", mew=0.6, zorder=6 if k == "v_read" else 4)
                ends.setdefault(k, []).append(float(y[-1]))
                if k not in drawn:
                    drawn.append(k)
    st.depth_axis(ax, "depth")
    ax.set_ylim(0.3, 1.03)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("AUROC for the answer")
    st.despine(ax)
    ax.axhline(0.5, color=st.AXIS, lw=1.0, zorder=1)
    _chance_word(ax)
    short = {"v_read": "V-Read", "tuned": "tuned lens", "jacobian": "Jacobian lens", "logit": "logit lens"}
    ents = [(short[k], P.COLOUR[k], P.DASH.get(k, ()), 2.6 if k == "v_read" else 1.8)
            for k in LENSES if k in drawn]
    if "v_read_coco" in drawn:
        ents.append(("V-Read fit\non COCO", P.COLOUR["v_read_coco"], P.DASH["v_read_coco"], 1.4))
    if any("coco_mcq" in d["families"] for _, d in benches):
        ents.append(("A-D questions", st.SUBINK, MCQ_DASH, 1.0))
    _margin_key(ax, ents)


def _margin_key(ax, entries, gap_pts=3.0):
    fig = ax.figure
    fig.canvas.draw()
    fs = matplotlib.rcParams["legend.fontsize"]
    bb = ax.get_window_extent()
    step = (fs + gap_pts) * fig.dpi / 72.0 / bb.height
    x0 = 1.0 + 0.1 / (bb.width / fig.dpi)
    x1 = x0 + 0.3 / (bb.width / fig.dpi)
    y = 1.0 - 0.5 * step
    for text, colour, dash, lw in entries:
        ax.plot([x0, x1], [y, y], transform=ax.transAxes, color=colour, lw=lw, dashes=dash or (),
                solid_capstyle="butt", clip_on=False, zorder=9)
        ax.annotate(text, xy=(x1, y), xycoords="axes fraction", xytext=(5, 0.5 * fs),
                    textcoords="offset points", color=st.INK, va="top", ha="left",
                    fontsize=fs, annotation_clip=False, linespacing=0.95)
        y -= (text.count("\n") * 0.95 + 1) * step + 0.25 * step


def _kde(x, grid, bw=0.03):
    if not x.size:
        return np.zeros_like(grid)
    xs = np.concatenate([x, -x, 2.0 - x])
    z = (grid[:, None] - xs[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(1) / (x.size * bw * np.sqrt(2 * np.pi))


def _panel_ridges(ax, pats):
    meths = [m for m in RANKS if any(m in p["auroc_items"] and p["auroc_items"][m].size
                                     for _, p in pats)]
    grid = np.linspace(0.0, 1.0, 401)
    gap = 0.8
    nR = len(meths)
    for i, m in enumerate(meths):
        base = (nR - 1 - i) * gap
        z = 10 + 2 * i
        for j, (mo, p) in enumerate(pats):
            x = p["auroc_items"].get(m)
            if x is None or not x.size:
                continue
            dens = _kde(x, grid)
            dens = dens / max(dens.max(), 1e-12)
            keep = dens > 0.01
            gx, gy = grid[keep], base + dens[keep]
            if j == 0:
                ax.fill_between(gx, base, gy, color="white", lw=0, zorder=z)
                ax.fill_between(gx, base, gy, color=COLOUR[m], alpha=0.30, lw=0, zorder=z + 0.1)
                ax.plot(gx, gy, color=COLOUR[m], lw=1.7, dashes=DASH.get(m, ()), zorder=z + 0.2)
                ax.plot([gx[0], gx[-1]], [base, base], color=COLOUR[m], lw=0.9, zorder=z + 0.2,
                        solid_capstyle="butt")
            else:
                ax.plot(gx, gy, color=COLOUR[m], lw=1.1, zorder=z + 0.3,
                        dashes=DASH.get(m) or (3, 1.5))
            mk = MARK.get(mo, "o")
            ax.plot([float(np.median(x))], [base], marker=mk, ms=MSIZE[mk] + 0.8, color=COLOUR[m],
                    mec="white", mew=1.0, ls="none", zorder=z + 1, clip_on=False)
    ax.axvline(0.5, color=st.AXIS, lw=1.0, zorder=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_ylim(-0.14, (nR - 1) * gap + 1.06)
    ax.yaxis.tick_right()
    ax.set_yticks([(nR - 1 - i) * gap + 0.28 for i in range(nR)])
    ax.set_yticklabels([NAME2[m] for m in meths], linespacing=0.92,
                       fontsize=matplotlib.rcParams["legend.fontsize"])
    ax.tick_params(axis="y", length=0, pad=6)
    ax.yaxis.grid(False)
    ax.set_xlabel("map ranks object cells")
    st.despine(ax, left=False)
    ax.spines["left"].set_visible(False)
    ax.text(0.5, 1.0, "chance", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            color=st.INK, fontsize=matplotlib.rcParams["xtick.labelsize"] * 0.92)
    return meths


def _pool(pats):
    out = {}
    for meth in ("vwrite_mean", "attr_patch"):
        fs = [p["field"][meth] for _, p in pats]
        out[meth] = dict(fs[0], H=sum(f["H"] for f in fs), n=sum(f["n"] for f in fs))
    return dict(field=out)


def _panel_field(ax, p):
    from scipy.ndimage import gaussian_filter
    fv, fa = p["field"]["vwrite_mean"], p["field"]["attr_patch"]
    lo, hi = fv["lo"], fv["hi"]
    bins = fv["H"].shape[0]
    sig = 0.05 / ((hi - lo) / bins)
    Hv = gaussian_filter(fv["H"], sig, mode="constant")
    Ha = gaussian_filter(fa["H"], sig, mode="constant")
    ax.imshow(Hv.T, origin="upper", extent=(lo, hi, hi, lo), cmap=st.warm_cmap(), aspect="auto",
              interpolation="bilinear", vmin=0, vmax=float(np.quantile(Hv, 0.995)), zorder=1)
    flat = np.sort(Ha.ravel())[::-1]
    cum = np.cumsum(flat) / max(flat.sum(), 1e-12)
    lev = sorted({float(flat[min(np.searchsorted(cum, q), flat.size - 1)]) for q in (0.25, 0.5, 0.75)})
    cx = lo + (np.arange(bins) + 0.5) * (hi - lo) / bins
    if len(lev) >= 2:
        ax.contour(cx, cx, Ha.T, levels=lev, colors=st.INK, linewidths=0.9, zorder=3)
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec=st.INK, lw=1.6, zorder=4))
    ax.set_xlim(lo, hi)
    ax.set_ylim(hi, lo)
    ax.set_xticks([-1, 0, 1, 2])
    ax.set_yticks([-1, 0, 1, 2])
    ax.grid(False)
    ax.set_xlabel("across (box widths)")
    ax.set_ylabel("down (box heights)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_color(st.AXIS)


def _field_key(fig, ax):
    fs = matplotlib.rcParams["legend.fontsize"]
    Hf = fig.get_size_inches()[1]
    b = ax.get_position()
    x = b.x0 - 0.36 / W_IN
    cm = st.warm_cmap()
    for i, (text, kind) in enumerate((("attribution patching", "line"), ("V-Write", "shade"))):
        y = (TOP_IN + 0.16 + 0.27 * i) / Hf
        if kind == "shade":
            fig.add_artist(Rectangle((x, y - 0.07 / Hf), 0.26 / W_IN, 0.14 / Hf, transform=fig.transFigure,
                                     fc=cm(0.7), ec="none"))
        else:
            fig.add_artist(Line2D([x, x + 0.26 / W_IN], [y, y], color=st.INK, lw=0.9,
                                  transform=fig.transFigure))
        fig.text(x + 0.34 / W_IN, y, text, ha="left", va="center", color=st.INK, fontsize=fs)


def _panel_drop(ax, pats, label_names):
    from vlens.figures.fx_patch import _fit, _percent_axis
    ks = [100 * k for k in KS]
    ends = {}
    many = len(pats) > 1
    for mo, p in pats:
        mk = MARK.get(mo, "o")
        for m in reversed(RANKS):
            if m not in p["curve"]:
                continue
            y = p["curve"][m]
            if many:
                ax.plot(ks, y, color=COLOUR[m], lw=0.9, alpha=0.7, dashes=DASH.get(m, ()),
                        marker=mk, ms=MSIZE[mk] * 0.75, mec="white", mew=0.5, zorder=3)
            else:
                ax.plot(ks, y, color=COLOUR[m], lw=2.6 if m == "vwrite_mean" else 1.8,
                        dashes=DASH.get(m, ()), marker=mk, ms=MSIZE[mk], mec="white", mew=0.7,
                        zorder=6 if m == "vwrite_mean" else 4)
            ends.setdefault(m, []).append(np.asarray(y, float))
    if many:
        for m in reversed(RANKS):
            if len(ends.get(m, [])) == len(pats):
                ax.plot(ks, np.mean(ends[m], 0), color=COLOUR[m], lw=2.6 if m == "vwrite_mean" else 2.0,
                        dashes=DASH.get(m, ()), zorder=6 if m == "vwrite_mean" else 5)
    ends = {m: [float(np.mean([e[-1] for e in es]))] for m, es in ends.items()}
    _percent_axis(ax, ks)
    ax.set_xticks([5, 20, 50])
    ax.set_xticklabels(["5", "20", "50"])
    ax.set_xlabel("tokens replaced (%)")
    ax.set_ylabel("fall in the answer score")
    st.despine(ax)
    _fit(ax)
    if label_names:
        _margin_key(ax, [(NAME2[m], COLOUR[m], DASH.get(m, ()), 2.6 if m == "vwrite_mean" else 2.0)
                         for m in sorted(ends, key=lambda m: -ends[m][0])])


def _model_key(fig, models, ax):
    fs = matplotlib.rcParams["legend.fontsize"]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W = fig.get_window_extent().width
    H = fig.get_size_inches()[1]
    x = ax.get_position().x0
    for i, m in enumerate(models):
        y = (TOP_IN + 0.16 + 0.27 * (len(models) - 1 - i)) / H
        mk = MARK.get(m, "o")
        fig.add_artist(Line2D([x + 0.06 / W_IN], [y], marker=mk, ms=MSIZE[mk] + 1.5,
                              color=st.INK, mfc="white", mew=1.3, ls="none",
                              transform=fig.transFigure))
        fig.text(x + 0.2 / W_IN, y, m, ha="left", va="center", color=st.INK, fontsize=fs)


def coco_general(results="results", out="figures", model=None):
    models = _models(results)
    if model:
        if model not in models:
            return None
        models = [model]
    elif not P.AVERAGE:
        return None
    v = _numbers(results, models)
    pats = [(m, v[m]["patch"]) for m in models if m in v and v[m]["patch"]]
    benches = [(m, v[m]["raw_bench"]) for m in models if m in v and v[m]["raw_bench"]] if SHOW_LENS else []
    if not pats:
        return None
    n_ss = _check_disjoint() if benches else None
    pooled = len(pats) > 1 and all(p["has_field"] for _, p in pats)
    cap = _caption({m: v[m] for m in models if m in v}, n_ss, pooled=pooled)
    coco_general.__doc__ = cap
    print(cap)

    shown = sorted({x for x, _ in pats} | {x for x, _ in benches}, key=_model_order)
    field = bool(pats[0][1]["has_field"])
    H = H_IN + (KEY_IN if len(shown) > 1 or field else 0.0)
    st.use()
    fig = plt.figure(figsize=(W_IN, H))
    fig.set_layout_engine("none")
    ra, rb, rc = LAYOUT[(bool(benches), field)]
    ax_b = fig.add_axes(_rect(rb, H))
    ax_c = fig.add_axes(_rect(rc, H))
    letters = []
    if benches:
        ax_a = fig.add_axes(_rect(ra, H))
        _panel_lens(ax_a, benches)
        letters.append(ax_a)
    if field:
        _panel_field(ax_b, _pool(pats) if pooled else pats[0][1])
    else:
        _panel_ridges(ax_b, pats)
    _panel_drop(ax_c, pats, label_names=field)
    if field:
        _field_key(fig, ax_b)
    if len(shown) > 1:
        _model_key(fig, shown, letters[0] if letters else ax_c)
    return st.save(fig, P._at(out, "coco_general"), check=False, tight=True)


TOP_SHARE = 0.10
SIZE_Q = (0.1, 0.3, 0.5, 0.7, 0.9)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_COCO_DIRS = (os.environ.get("VLENS_COCO", "data/coco_subset"),
              os.path.join(_REPO, "data", "coco_subset"))
_COCO_URL = "http://images.cocodataset.org/"


def _coco_records():
    for d in _COCO_DIRS:
        p = os.path.join(d, "coco_items.json")
        if os.path.exists(p):
            with open(p) as f:
                return {r["id"]: r for r in json.load(f)["records"]}, d
    return {}, None


def _photo(rel, root):
    from PIL import Image
    for d in (root,) + _COCO_DIRS:
        p = os.path.join(d or "", rel)
        if d and os.path.exists(p):
            return Image.open(p).convert("RGB")
    dst = os.path.join(_COCO_DIRS[-1], rel)
    try:
        import urllib.request
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        urllib.request.urlretrieve(_COCO_URL + rel, dst + ".part")
        os.replace(dst + ".part", dst)
        return Image.open(dst).convert("RGB")
    except Exception:
        return None


def _replies(results, m):
    d = _json(f"{results}/coco/{m}__coco_pope.json") or {}
    return {r["id"]: str(r.get("reply") or "") for r in d.get("rows") or []}


def _hit(r):
    return bool(((r.get("loc") or {}).get("vwrite_mean") or {}).get("hit1"))


def _pick(by_model, recs, n_hit=len(SIZE_Q)):
    first = by_model[0]
    ok = [i for i, r in first.items()
          if i in recs and all(i in b and (b[i].get("maps") or {}).get("vwrite_mean") and b[i].get("cov")
                               for b in by_model)]
    hits = [i for i in ok if all(_hit(b[i]) for b in by_model)]
    miss = [i for i in ok if not any(_hit(b[i]) for b in by_model)]
    if len(hits) < n_hit:
        return []
    area = {i: float(first[i]["area"]) for i in ok}
    qs = np.quantile([area[i] for i in hits], SIZE_Q)
    got, cats = [], set()
    for q in qs:
        cand = sorted((i for i in hits if i not in got and first[i]["target"] not in cats),
                      key=lambda i: (abs(area[i] - q), i))
        if cand:
            got.append(cand[0])
            cats.add(first[cand[0]]["target"])
    got.sort(key=lambda i: area[i])
    if miss:
        med = float(np.median([area[i] for i in miss]))
        cand = sorted((i for i in miss if first[i]["target"] not in cats),
                      key=lambda i: (abs(area[i] - med), i))
        if cand:
            got.append(cand[0])
    return got


def _tile(ax, im, rec, row):
    import matplotlib.patheffects as pe
    from vlens.data import coco as CC
    side = 448
    a = np.asarray(im.resize((side, side), 1), float)
    a = np.clip(a * 0.74 + 255.0 * 0.26, 0, 255).astype(np.uint8)
    ax.imshow(a, extent=(0, side, side, 0), interpolation="bilinear", zorder=1)
    Gr, Gc = _grid(row)
    mp = np.asarray(row["maps"]["vwrite_mean"], float)
    order = np.argsort(-mp, kind="stable")
    nk = max(int(round(TOP_SHARE * mp.size)), 1)
    ph, pw = side / Gr, side / Gc
    for t in order[:nk]:
        y0, x0 = divmod(int(t), Gc)
        ax.add_patch(Rectangle((x0 * pw, y0 * ph), pw, ph, fc=COLOUR["vwrite_mean"], ec="none",
                               alpha=0.5, zorder=2))
    y0, x0 = divmod(int(order[0]), Gc)
    ax.add_patch(Rectangle((x0 * pw, y0 * ph), pw, ph, fill=False, ec=st.INK, lw=2.0, zorder=5))
    m = CC.mask_square(rec, side)
    cs = ax.contour(np.arange(side) + 0.5, np.arange(side) + 0.5, m, levels=[0.5],
                    colors=st.INK, linewidths=1.2, zorder=4)
    for c in getattr(cs, "collections", [cs]):
        c.set_path_effects([pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()])
    ax.set_xlim(0, side)
    ax.set_ylim(side, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(st.AXIS)
        sp.set_linewidth(0.8)


def _reply_corner(fig, ax, rec, row, text, fs, lh, colw):
    from vlens.data import coco as CC
    from vlens.figures import fx_layers as FL
    W, H = fig.get_size_inches()
    w_in = FL._width_in(text, fs) + 2 * FL.BOX_PAD + 0.02
    h_in = lh + 2 * FL.BOX_PAD
    m = CC.mask_square(rec, 112)
    Gr, Gc = _grid(row)
    mp = np.asarray(row["maps"]["vwrite_mean"], float)
    top = np.zeros(mp.size)
    top[np.argsort(-mp, kind="stable")[:max(int(round(TOP_SHARE * mp.size)), 1)]] = 1
    top = np.kron(top.reshape(Gr, Gc), np.ones((112 // Gr + 1, 112 // Gc + 1)))[:112, :112]
    fx, fy = int(112 * (w_in + 0.06) / colw) + 1, int(112 * (h_in + 0.06) / colw) + 1
    best = None
    for sx in (0, 1):
        for sy in (0, 1):
            ys = slice(112 - fy, 112) if sy else slice(0, fy)
            xs = slice(112 - fx, 112) if sx else slice(0, fx)
            cost = float(m[ys, xs].mean() * 4 + top[ys, xs].mean())
            if best is None or cost < best[0]:
                best = (cost, sx, sy)
    _, sx, sy = best
    b = ax.get_position()
    x = b.x1 - (w_in + 0.04) / W if sx else b.x0 + 0.04 / W
    y_top = b.y0 + (h_in + 0.04) / H if sy else b.y1 - 0.04 / H
    FL.panel_box(fig, x, y_top, w_in, 1, lh, fs, [text], "model")
    fig.patches[-1].set_alpha(0.92)
    fig.patches[-1].set_zorder(6)
    fig.texts[-1].set_zorder(7)


def _strip(ax, pats, fs):
    meths = [m for m in RANKS if all(m in p["loc"] for _, p in pats)]
    off = np.linspace(-0.16, 0.16, len(pats)) if len(pats) > 1 else [0.0]
    for i, m in enumerate(meths):
        y = len(meths) - 1 - i
        for (mo, p), dy in zip(pats, off):
            x = p["loc"][m]["hit1"]
            ci = (p.get("loc_ci") or {}).get(m, {}).get("hit1")
            if ci:
                ax.plot(ci, [y + dy] * 2, color=COLOUR[m], lw=1.6, solid_capstyle="butt", zorder=3)
            mk = MARK.get(mo, "o")
            ax.plot([x], [y + dy], marker=mk, ms=MSIZE[mk] + 1.6, color=COLOUR[m], mec="white",
                    mew=0.8, ls="none", zorder=4)
    ch = float(np.mean([p["chance"] for _, p in pats]))
    ax.axvline(ch, color=st.SUBINK, lw=1.0, dashes=(3, 2), zorder=1)
    ax.set_ylim(-0.5, len(meths) - 0.5)
    ax.set_yticks(range(len(meths)))
    ax.set_yticklabels([NAME2[m] for m in meths][::-1], fontsize=fs * 0.94, linespacing=0.84)
    ax.tick_params(axis="y", length=0, pad=4)
    top = max(max((p.get("loc_ci") or {}).get(m, {}).get("hit1", [0, p["loc"][m]["hit1"]])[1]
                  for _, p in pats) for m in meths)
    ax.set_xlim(0, max(0.62, top + 0.04))
    ax.set_xticks([0, 0.25, 0.5, 0.75] if top > 0.7 else [0, 0.25, 0.5])
    ax.set_xticklabels(["0", "25", "50", "75"][: len(ax.get_xticks())], fontsize=fs)
    ax.set_xlabel("hit@1 (%)", fontsize=fs, labelpad=2)
    ax.yaxis.grid(False)
    st.despine(ax)
    ax.text(ch, len(meths) - 0.5, "chance", ha="center", va="bottom", color=st.INK, fontsize=fs * 0.92)
    return meths


def _pointing_caption(v, models, picks, n_hits):
    ms = [m for m in models if (v.get(m) or {}).get("patch")]
    blk = [(m, v[m]["patch"]) for m in ms]
    two = len(ms) > 1
    sizes = sorted(float(v[ms[0]]["_rows"][i]["area"]) for i in picks[:n_hits]) if picks and ms \
        and v[ms[0]].get("_rows") else []
    out = ["MAIN. On ordinary photographs, V-Write's map lands on the object people outlined"
           + (f", for objects from {100 * sizes[0]:.0f} to {100 * sizes[-1]:.0f} per cent of the frame"
              if sizes else "")
           + ", and over every graded item its strongest cell is on the object more often than that of "
           "the best baseline (the better of attribution patching and gradient times activation), the "
           "centre prior or a random map."]
    out.append(
        "Each column is one COCO val2017 photograph from the pointing test, shown as the model sees "
        "it, squashed to 448 by 448 pixels; the ink outline is the unioned human segmentation of the "
        "object asked about, clay marks the tenth of the image tokens (cells of the "
        + (f"{blk[0][1]['grid'][0]} by {blk[0][1]['grid'][1]}" if blk and blk[0][1].get("grid") else PENDING)
        + " grid) V-Write ranks highest and the ink square its single highest. "
        + ("The top row is " + ms[0] + "'s map and the bottom row " + ms[1] + "'s, on the same pictures, "
           "at " if two else "The map is taken at ")
        + _join([f"block {p.get('block', PENDING)} of {p.get('n_layers', PENDING)}" + (f" on {m}" if two else "")
                 for m, p in blk])
        + ". Boxed under each column: the question asked, with the instruction \"Answer yes or no.\" that "
        "follows it in every prompt left off; "
        + ("each model's greedy reply is boxed in the corner of its own picture that covers least of "
           "the object and the map" if two else "under it, the model's greedy reply")
        + "; no system prompt. The "
        "first " + str(n_hits) + " columns are the pictures where V-Write's strongest cell is inside "
        "the object (a hit" + (" on both models" if two else "") + "), one per category, the one whose "
        "object's share of the frame is nearest each of the 10th, 30th, 50th, 70th and 90th "
        "percentiles over those hits, ordered by size; the last is the miss"
        + (" on both models" if two else "") + " of median size. The line under each picture gives "
        "that share and whether V-Write's strongest cell hit. Pictures were chosen by this rule "
        "alone, not by how the map looks.")
    out.append(
        "The strip is hit@1, the pointing game, over every present-object item the model answered "
        "correctly (" + _join([f"{p['n_graded']} on {m}" for m, p in blk]) + "): the share of items "
        "whose single strongest cell is at least half covered by the object"
        + (f", {_join([f'{MARK_WORD[MARK.get(m, chr(111))]} for {m}' for m in ms])}" if two else "")
        + ", with 95 per cent bootstrap intervals over items (2000 resamples); the dashed rule is "
        "chance, the mean share of the grid inside the object ("
        + _join([f"{_pc(p['chance'])} on {m}" for m, p in blk]) + ").")
    fin, lim = [], []
    for m, p in blk:
        lc = p["loc"]
        bk, bv = _best(lc, "hit1")
        pr = _paired(p, bk, "hit1") if bk else None
        fin.append(f"On {m}, hit@1 is {_pc(lc['vwrite_mean']['hit1'])} for V-Write against {_pc(bv)} "
                   f"for the best baseline ({NAME.get(bk, PENDING)}), {_pc(lc['center']['hit1'])} for the "
                   f"centre prior and {_pc(lc['random']['hit1'])} for random"
                   + (f"; paired, V-Write minus {NAME[bk]} is {_ci_s(pr, True)}" if pr else "") + ".")
        if pr and pr["ci"][0] <= 0:
            lim.append(f"on {m} V-Write's hit@1 lead over {NAME[bk]} is within the bootstrap interval of zero")
    lim.append("a miss by the strongest cell can still leave most of the top tenth on the object, "
               "and the pointing test grades only present objects the model said yes to")
    out.append(" ".join(fin) + " Limits: " + "; ".join(lim) + ".")
    cap = out[0] + "\n\n" + "\n\n".join(out[1:])
    assert "—" not in cap
    return cap


def coco_pointing(results="results", out="figures", model=None):
    import matplotlib.lines as mlines
    from vlens.figures import fx_layers as FL
    models = [m for m in _models(results) if _patch(results, m)]
    if model:
        if model not in models:
            return None
        models = [model]
    elif not P.AVERAGE:
        return None
    models = models[:2]
    if not models:
        return None
    recs, root = _coco_records()
    if not recs:
        return None
    raw = {m: {r["id"]: r for r in _patch(results, m)["rows"]} for m in models}
    picks = _pick([raw[m] for m in models], recs)
    photos = {i: _photo(recs[i]["image"], root) for i in picks}
    picks = [i for i in picks if photos[i] is not None]
    if len(picks) < 4:
        return None
    n_hits = sum(all(_hit(raw[m][i]) for m in models) for i in picks)
    v = _numbers(results, models)
    pats = [(m, v[m]["patch"]) for m in models]
    v[models[0]]["_rows"] = raw[models[0]]
    cap = _pointing_caption(v, models, picks, n_hits)
    coco_pointing.__doc__ = cap
    print(cap)
    reply = {m: _replies(results, m) for m in models}

    st.use()
    W = st.MAX_WIDTH
    fs = matplotlib.rcParams["xtick.labelsize"] * 0.74
    fb = fs * 0.95
    lh = fs * 1.28 / 72
    nrow = len(models)
    lab_w = 0.5 if nrow > 1 else 0.0
    names_w, strip_w, gap = 1.3, 1.05, 0.42
    gut = 0.13
    n = len(picks)
    colw = (W - 0.04 - lab_w - names_w - strip_w - gap - gut * (n - 1)) / n
    inner = colw - 2 * FL.BOX_PAD - 0.02
    qs, ans = [], []
    for i in picks:
        q = str(recs[i]["question"]).replace(" Answer yes or no.", "")
        qs.append(FL.fit_lines(q, inner, fb, 4))
        if nrow > 1:
            ans.append([])
            continue
        ans.append([ln for m in models
                    for ln in FL.tag_lines(m, inner, fb)
                    + FL.fit_lines(str(reply[m].get(i) or raw[m][i].get("said", "")).strip(), inner, fb, 1)])
    nq, na = max(map(len, qs)), max(map(len, ans))
    n_size = max(len(FL.fit_lines(f"{100 * float(raw[models[0]][i]['area']):.0f}% of frame, miss", colw, fb, 2))
                 for i in picks)
    lbx = 0.9 * lh
    img_h = nrow * colw + (nrow - 1) * gut
    H = (0.04 + img_h + 0.1 + n_size * lh + 0.12 + (nq * lbx + 2 * FL.BOX_PAD)
         + ((0.07 + na * lbx + 2 * FL.BOX_PAD) if na else 0.0) + 0.04)
    fig = plt.figure(figsize=(W, H))
    fig.set_layout_engine("none")
    for j, i in enumerate(picks):
        x_in = lab_w + j * (colw + gut)
        for k, m in enumerate(models):
            top_in = 0.04 + k * (colw + gut)
            ax = fig.add_axes([x_in / W, 1 - (top_in + colw) / H, colw / W, colw / H])
            _tile(ax, photos[i], recs[i], raw[m][i])
            if nrow > 1:
                _reply_corner(fig, ax, recs[i], raw[m][i],
                              str(reply[m].get(i) or raw[m][i].get("said", "")).strip(), fb, lbx, colw)
        y_in = 0.04 + img_h + 0.1
        hit = all(_hit(raw[m][i]) for m in models)
        size = FL.fit_lines(f"{100 * float(raw[models[0]][i]['area']):.0f}% of frame, "
                            + ("hit" if hit else "miss"), colw, fb, 2)
        fig.text(x_in / W, 1 - y_in / H, "\n".join(size), ha="left", va="top", color=st.INK,
                 fontsize=fb, linespacing=1.15)
        y_in += n_size * lh + 0.12
        hq = FL.panel_box(fig, x_in / W, 1 - y_in / H, colw, nq, lbx, fb, qs[j], "set")
        y_in += hq + 0.07
        if na:
            FL.panel_box(fig, x_in / W, 1 - y_in / H, colw, na, lbx, fb, ans[j], "model")
    if nrow > 1:
        for k, m in enumerate(models):
            top_in = 0.04 + k * (colw + gut)
            lines = FL.tag_lines(m, colw - 0.1, fb)
            fig.text((lab_w - 0.27) / W, 1 - (top_in + colw / 2) / H, "\n".join(ln.rstrip(":") for ln in lines),
                     ha="center", va="center", rotation=90, color=st.INK, fontsize=fb, linespacing=1.0)
    x_s = W - strip_w - 0.04
    s_top = 0.34
    s_bot = H - 0.46
    ax_s = fig.add_axes([x_s / W, 1 - s_bot / H, strip_w / W, (s_bot - s_top) / H])
    _strip(ax_s, pats, fb)
    from matplotlib.patches import Patch
    hs = [Patch(fc=COLOUR["vwrite_mean"], alpha=0.5, ec="none"),
          Patch(fill=False, ec=st.INK, lw=1.6),
          mlines.Line2D([], [], color=st.INK, lw=1.2)]
    labs = ["V-Write top 10%", "its top token", "human outline"]
    if nrow > 1:
        for m in models:
            mk = MARK.get(m, "o")
            hs.append(mlines.Line2D([], [], marker=mk, ms=MSIZE[mk] + 1, color=st.INK, mfc="white",
                                    mew=1.1, ls="none"))
            labs.append(m)
    st.key_band(fig, hs, labs, fontsize=fb, handlelength=1.2, handletextpad=0.5)
    return st.save(fig, P._at(out, "coco_pointing"), check=False, tight=True)


def _import_pointing_caption():
    try:
        ms = [m for m in _models(os.path.join(_REPO, "results")) if _patch(os.path.join(_REPO, "results"), m)][:2]
        v = _numbers(os.path.join(_REPO, "results"), ms)
        return _pointing_caption(v, ms, [], len(SIZE_Q))
    except Exception:
        return coco_pointing.__doc__


def _import_caption():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    res = os.path.join(here, "results")
    try:
        ms = _models(res)
        v = _numbers(res, ms)
        v = v if any(x.get("patch") for x in v.values()) else {}
    except Exception:
        v = {}
    return _caption(v)


coco_general.__doc__ = _import_caption()
coco_pointing.__doc__ = _import_pointing_caption()
