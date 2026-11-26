from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "simonsays"))

import numpy as np

OUT_DIR = os.path.join(ROOT, "results", "extended", "composite_images")
GRID = 14
GLOBAL_AREA = 0.5
EDITED_FAMILIES = ("openimages_edit_auth", "openimages_edit_eval", "openimages_edit_read", "openimages_edit_base",
                   "nuscenes_edit_auth", "nuscenes_edit_eval", "nuscenes_edit_read",
                   "vins_edit_auth", "vins_edit_eval", "vins_edit_read",
                   "gemini_edit_eval", "gemini_edit_swap", "gemini_edit_removal")


def full_mask(src, edit, thr=28):
    from PIL import Image, ImageFilter
    s = src.convert("L")
    e = edit.convert("L")
    if e.size != s.size:
        e = e.resize(s.size, Image.LANCZOS)
    d = np.abs(np.asarray(s, np.int16) - np.asarray(e, np.int16)).astype(np.uint8)
    d = Image.fromarray(d).filter(ImageFilter.MedianFilter(5))
    return np.asarray(d) > thr


def token_share(mask, grid=GRID, min_frac=0.15):
    H, W = mask.shape
    ys = np.linspace(0, H, grid + 1).astype(int)
    xs = np.linspace(0, W, grid + 1).astype(int)
    n = 0
    for i in range(grid):
        for j in range(grid):
            c = mask[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            n += int(c.size > 0 and c.mean() >= min_frac)
    return n / grid ** 2


def cell_mask(mask, grid=GRID, min_frac=0.15, dilate=1):
    H, W = mask.shape
    ys = np.linspace(0, H, grid + 1).round().astype(int)
    xs = np.linspace(0, W, grid + 1).round().astype(int)
    cells = np.zeros((grid, grid), bool)
    for i in range(grid):
        for j in range(grid):
            c = mask[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            cells[i, j] = c.size > 0 and c.mean() >= min_frac
    grown = cells.copy()
    for _ in range(dilate):
        g = grown.copy()
        g[1:] |= grown[:-1]; g[:-1] |= grown[1:]; g[:, 1:] |= grown[:, :-1]; g[:, :-1] |= grown[:, 1:]
        g[1:, 1:] |= grown[:-1, :-1]; g[:-1, :-1] |= grown[1:, 1:]; g[1:, :-1] |= grown[:-1, 1:]; g[:-1, 1:] |= grown[1:, :-1]
        grown = g
    px = np.zeros((H, W), bool)
    for i in range(grid):
        for j in range(grid):
            if grown[i, j]:
                px[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = True
    return px, cells


def _one(job):
    from PIL import Image
    ep, sp, root_ss, skip = job
    rel = os.path.relpath(ep, root_ss)
    src = Image.open(sp).convert("RGB")
    edi = Image.open(ep).convert("RGB")
    W, H = src.size
    aspect_changed = abs(edi.size[0] / edi.size[1] - W / H) > 0.02
    m = full_mask(src, edi)
    md, cells = cell_mask(m)
    area, area_d = float(m.mean()), float(md.mean())
    ent = dict(source=os.path.relpath(sp, root_ss), area=round(area, 4), area_cells=round(float(cells.mean()), 4),
               area_composited=round(area_d, 4), token_share=round(token_share(m), 4),
               aspect_changed=bool(aspect_changed), size=[W, H], edit_size=list(edi.size))
    if not cells.any():
        ent["excluded"] = "no token cell reaches 15% changed pixels"
    elif aspect_changed:
        ent["excluded"] = "edit changes the frame's aspect ratio"
    elif area_d > GLOBAL_AREA:
        ent["excluded"] = f"edit covers {area_d:.0%} of the frame after growing by one cell"
    nbytes = 0
    if "excluded" not in ent:
        e_arr = np.asarray(edi.resize((W, H), Image.LANCZOS) if edi.size != (W, H) else edi)
        s_arr = np.asarray(src)
        comp = np.where(md[..., None], e_arr, s_arr)
        name = rel.replace("/", "__").rsplit(".", 1)[0] + ".png"
        out = os.path.join(OUT_DIR, name)
        if not (skip and os.path.exists(out)):
            Image.fromarray(comp.astype(np.uint8)).save(out, compress_level=6)
        nbytes = os.path.getsize(out)
        ent["composite"] = os.path.relpath(out, ROOT)
        outside = ~md
        if outside.any():
            diff = np.abs(e_arr.astype(np.int16) - s_arr.astype(np.int16)).mean(-1)
            ent["outside_mean_abs_diff"] = round(float(diff[outside].mean()), 3)
            ent["outside_changed_share"] = round(float((diff[outside] > 8).mean()), 4)
    return rel, ent, nbytes


def build(a):
    from multiprocessing import Pool
    from vlens.data import simonsays as SS
    from handoff import pairs_for

    recs = SS.load(require_image=True)
    P = pairs_for(recs)
    src_of, conflicts = {}, 0
    for o, e in P:
        prev = src_of.setdefault(e["image"], o["image"])
        conflicts += prev != o["image"]
    edited = sorted({r["image"] for r in recs if r["variant"] == "edited"})
    unpaired = [p for p in edited if p not in src_of]
    print(f"{len(edited)} edited pictures, {len(src_of)} with a source, {len(unpaired)} without; "
          f"{conflicts} records whose pairing names a different source (first kept)", flush=True)

    if os.path.isdir(OUT_DIR) and not a.skip_existing:
        for f in os.listdir(OUT_DIR):
            if f.endswith(".png"):
                os.remove(os.path.join(OUT_DIR, f))
    os.makedirs(OUT_DIR, exist_ok=True)
    man, total = {}, 0
    jobs = [(ep, sp, SS.ROOT, a.skip_existing) for ep, sp in sorted(src_of.items())]
    with Pool(a.workers) as pool:
        for i, (rel, ent, nb) in enumerate(pool.imap_unordered(_one, jobs)):
            man[rel] = ent
            total += nb
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} ({total / 2 ** 20:.0f} MB so far)", flush=True)
    kept = [k for k, v in man.items() if "composite" in v]
    why = defaultdict(int)
    for v in man.values():
        if "excluded" in v:
            why[v["excluded"].split(" covers")[0]] += 1
    json.dump(dict(grid=GRID, global_area=GLOBAL_AREA, threshold=28,
                   region="A.4 token cells (>=15% changed pixels) grown by one cell",
                   n_edited=len(edited), n_paired=len(src_of), n_composites=len(kept),
                   unpaired=[os.path.relpath(p, SS.ROOT) for p in unpaired], pictures=man),
              open(os.path.join(OUT_DIR, "manifest.json"), "w"), indent=1)
    oc = [v["outside_changed_share"] for v in man.values() if "outside_changed_share" in v]
    print(f"{len(kept)} composites written ({total / 2 ** 20:.0f} MB); excluded: {dict(why)}; "
          f"outside the edit the generator changed {np.median(oc) if oc else float('nan'):.1%} "
          f"of pixels by more than 8 grey levels (median over pictures)", flush=True)


def _answer(row):
    mg = row.get("margins") or {}
    if "yes_over_no" in mg:
        return "yes" if mg["yes_over_no"] > 0 else "no"
    for k, v in mg.items():
        if "_over_" in k and isinstance(v, (int, float)):
            a, b = k.split("_over_")
            return a if v > 0 else b
    if isinstance(mg.get("letters"), dict) and mg["letters"]:
        return max(mg["letters"], key=mg["letters"].get)
    return None


def _rows(path):
    if not os.path.exists(path):
        return {}
    return {r["id"]: r for r in json.load(open(path))["rows"]}


def _boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def compare(a):
    from vlens.data import simonsays as SS
    from handoff import pairs_for

    orig_of = {e["id"]: o for o, e in pairs_for(SS.load(require_image=True))}
    out = dict(model=a.model, families={})
    for fam in EDITED_FAMILIES:
        E = {}
        for d in a.edited:
            E.update({k: v for k, v in _rows(os.path.join(ROOT, d, f"{a.model}__{fam}.json")).items() if k not in E})
        C = _rows(os.path.join(ROOT, a.composite, f"{a.model}__{fam}.json"))
        ids = [i for i in C if i in E and E[i]["kind"] != "open"]
        if not ids:
            continue
        O = {}
        for d in list(a.edited) + ["results/simonsays"]:
            ofam = fam.replace("_edit_", "_orig_")
            O.update({k: v for k, v in _rows(os.path.join(ROOT, d, f"{a.model}__{ofam}.json")).items() if k not in O})
        agree, eff_e, eff_c = [], [], []
        for i in ids:
            ae, ac = _answer(E[i]), _answer(C[i])
            if ae is None or ac is None:
                continue
            agree.append(ae == ac)
            o = orig_of.get(i)
            if o is not None and o["id"] in O:
                ao = _answer(O[o["id"]])
                if ao is not None:
                    eff_e.append(ae != ao)
                    eff_c.append(ac != ao)
        out["families"][fam] = dict(
            n=len(agree), agreement=_boot_ci(agree), n_with_original=len(eff_e),
            edit_effect_edited=_boot_ci(eff_e), edit_effect_composite=_boot_ci(eff_c),
            effect_difference=_boot_ci(np.asarray(eff_e, float) - np.asarray(eff_c, float)))
        f = out["families"][fam]
        print(f"{fam:24s} n={f['n']:4d} agree {f['agreement'][0]:.2f} [{f['agreement'][1]:.2f},{f['agreement'][2]:.2f}]"
              f"  edit effect edited {f['edit_effect_edited'][0]:.2f} vs composite {f['edit_effect_composite'][0]:.2f}"
              f" (n={f['n_with_original']})", flush=True)
    os.makedirs(os.path.join(ROOT, a.composite), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, a.composite, f"{a.model}__compare.json"), "w"), indent=1)


def probe(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    def states(d, fam):
        p = os.path.join(ROOT, d, f"{a.model}__{fam}_states.npz")
        if not os.path.exists(p):
            return {}
        z = np.load(p, allow_pickle=True)
        X = z["ANS"]
        lay = tuple(int(v) for v in (z["layers"] if "layers" in z.files else range(X.shape[1])))
        return {str(i): (lay, X[k]) for k, i in enumerate(z["ids"])}

    out = dict(model=a.model, families={})
    Xe, Xc, G = [], [], []
    for fam in EDITED_FAMILIES:
        E = {}
        for d in a.edited:
            E.update({k: v for k, v in states(d, fam).items() if k not in E})
        C = states(a.composite, fam)
        rows = _rows(os.path.join(ROOT, a.composite, f"{a.model}__{fam}.json"))
        for i in sorted(set(E) & set(C)):
            Xe.append(E[i]); Xc.append(C[i])
            G.append(str((rows.get(i) or {}).get("image") or i))
    if not Xe:
        print("no matched states", flush=True)
        return
    common = set(Xe[0][0])
    for (le, _), (lc, _) in zip(Xe, Xc):
        common &= set(le) & set(lc)
    common = sorted(common)
    if not common:
        print("no shared stored blocks", flush=True)
        return
    Xe = np.stack([v[[list(l).index(b) for b in common]] for l, v in Xe]).astype(np.float32)
    Xc = np.stack([v[[list(l).index(b) for b in common]] for l, v in Xc]).astype(np.float32)
    print(f"blocks compared: {common}", flush=True)
    n, Lb = Xe.shape[0], Xe.shape[1]
    groups = np.array(G * 2)
    y = np.r_[np.ones(n), np.zeros(n)]
    k = min(5, len(set(G)))
    per = []
    for l in range(Lb):
        X = np.r_[Xe[:, l], Xc[:, l]]
        s = np.zeros(2 * n)
        for tr, te in GroupKFold(k).split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=a.C, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            s[te] = clf.decision_function(sc.transform(X[te]))
        per.append(float(roc_auc_score(y, s)))
        print(f"  stored block {l}: edited vs composite AUROC {per[-1]:.3f}", flush=True)
    rng = np.random.default_rng(0)
    null = []
    l = int(np.argmax(per))
    X = np.r_[Xe[:, l], Xc[:, l]]
    for _ in range(a.null):
        flip = rng.random(n) < 0.5
        yy = np.r_[np.where(flip, 0, 1), np.where(flip, 1, 0)].astype(float)
        s = np.zeros(2 * n)
        for tr, te in GroupKFold(k).split(X, yy, groups):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=a.C, max_iter=2000).fit(sc.transform(X[tr]), yy[tr])
            s[te] = clf.decision_function(sc.transform(X[te]))
        null.append(float(roc_auc_score(yy, s)))
    out.update(n_pairs=n, auroc_by_stored_block=per, best_block=l, best=per[l],
               null_95=float(np.percentile(null, 95)) if null else None)
    print(f"{n} matched pairs; best edited-vs-composite AUROC {per[l]:.3f} at stored block {l}; "
          f"shuffled-pair 95th pct {out['null_95']}", flush=True)
    json.dump(out, open(os.path.join(ROOT, a.composite, f"{a.model}__probe.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--skip-existing", action="store_true")
    b.add_argument("--workers", type=int, default=4)
    for name in ("compare", "probe"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--edited", nargs="+",
                       default=["results/extended/composite_ref", "results/extended/simonsays_full", "results/simonsays"],
                       help="behaviour outputs on the edited pictures, searched in order")
        p.add_argument("--composite", default="results/extended/composite")
        p.add_argument("--C", type=float, default=0.1)
        p.add_argument("--null", type=int, default=20)
    a = ap.parse_args()
    {"build": build, "compare": compare, "probe": probe}[a.cmd](a)


if __name__ == "__main__":
    main()
