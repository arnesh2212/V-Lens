from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
sys.path.insert(0, os.path.join(HERE, "experiments", "extended"))
import make_vread_table as T
from vread_stats import auc

warnings.filterwarnings("ignore")
FAMS = ["openimages_orig_auth", "openimages_edit_auth", "openimages_orig_eval", "nuscenes_edit_auth",
        "vins_orig_auth", "vins_edit_auth", "gemini_orig_auth", "wmdp_bio_sand", "wmdp_chem_sand", "wmdp_cybe_sand"]
READERS = ("v_read", "tuned", "jacobian", "logit")


def pixel_features(path, cache={}):
    if path in cache:
        return cache[path]
    if path and not os.path.isabs(path):
        from vlens.paths import DATA
        path = os.path.join(os.environ.get("SIMONSAYS_DATA") or str(DATA), path)
    from PIL import Image
    try:
        im = Image.open(path).convert("RGB").resize((64, 64))
    except Exception:
        cache[path] = np.zeros(64 + 24, np.float32)
        return cache[path]
    g = np.asarray(im.convert("L").resize((8, 8)), np.float32).ravel() / 255.0
    hsv = np.asarray(im.convert("HSV"), np.float32).reshape(-1, 3) / 255.0
    h = np.concatenate([np.histogram(hsv[:, c], bins=8, range=(0, 1))[0] for c in range(3)]).astype(np.float32)
    f = np.concatenate([g, h / max(h.sum() / 3, 1)])
    cache[path] = f
    return f


def oof(X, y, groups, q=None, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y, int)
    out = np.full(len(y), np.nan)
    glob_mean = float(y.mean())
    ug = len(set(groups))
    if ug < 2:
        return out
    for tr, te in GroupKFold(n_splits=min(folds, ug)).split(np.zeros(len(y)), y, groups):
        if q is not None:
            m = {}
            for i in tr:
                m.setdefault(q[i], []).append(y[i])
            out[te] = [np.mean(m[q[i]]) if q[i] in m else glob_mean for i in te]
        else:
            if len(set(y[tr])) < 2:
                out[te] = glob_mean; continue
            sc = StandardScaler().fit(X[tr])
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            out[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    return out


def window(res, loc, reader, y, idx=None):
    dep = np.asarray(res["depth"], float)
    S = np.asarray(loc["series"][reader], float)
    if idx is not None:
        S = S[idx]
    vb = np.isfinite(np.asarray(res["auroc"].get("v_read", dep * np.nan), float))
    cols = np.nonzero((dep >= T.BAND[0]) & (dep <= T.BAND[1]) & vb)[0]
    vals = [auc(S[:, j], y) for j in cols if np.isfinite(S[:, j]).sum() >= 2]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    ap.add_argument("--orig", default="results/extended/scored_refit_root")
    ap.add_argument("--grey", default="results/extended/a4/root_grey")
    ap.add_argument("--mismatch", default="results/extended/a4/root_mismatch")
    ap.add_argument("--out", default="results/extended/a4")
    a = ap.parse_args()
    os.chdir(HERE)
    rows = []
    for fam in FAMS:
        ro, lo = T._capture(T.D._family, a.orig, a.model, fam)
        if ro is None or not lo:
            print(f"{fam}: no original readouts"); continue
        y = np.asarray(lo["y"], int)
        if y.sum() < 5 or len(y) - y.sum() < 5:
            print(f"{fam}: fewer than five of one answer, skipped"); continue
        ids = list(lo["ids"]); groups = np.asarray(lo["groups"], str)
        beh = {r["id"]: r for r in json.load(open(f"{a.orig}/simonsays/{a.model}__{fam}.json"))["rows"]}
        q = [str(beh.get(i, {}).get("question", i)) for i in ids]
        X = np.stack([pixel_features(beh.get(i, {}).get("image", "")) for i in ids])
        p_q = oof(None, y, groups, q=q) if len(set(q)) > 1 else np.full(len(y), float(y.mean()))
        p_i = oof(X, y, groups)
        p_qi = oof(np.column_stack([X, np.nan_to_num(p_q, nan=y.mean())]), y, groups)
        row = dict(task=fam, n=len(y), q_only=auc(p_q, y), img_only=auc(p_i, y), q_img=auc(p_qi, y))
        for k in READERS:
            if k in lo["series"]:
                row[f"orig_{k}"] = window(ro, lo, k, y)
        for tag, root in (("grey", a.grey), ("mismatch", a.mismatch)):
            if not root or not os.path.isdir(root):
                continue
            rs, ls = T._capture(T.D._family, root, a.model, fam)
            if rs is None or not ls:
                continue
            pos = {i: j for j, i in enumerate(ls["ids"])}
            keep = [k_ for k_, i in enumerate(ids) if i in pos]
            idx = [pos[ids[k_]] for k_ in keep]
            ys = np.asarray(ls["y"], int)[idx]
            yo = y[keep]
            row[f"{tag}_n"] = len(keep)
            row[f"{tag}_answer_changed"] = float(np.mean(ys != yo)) if keep else np.nan
            for k in READERS:
                if k in ls["series"]:
                    row[f"{tag}_{k}_vs_orig_answer"] = window(rs, ls, k, yo, idx=idx)
                    row[f"{tag}_{k}_vs_own_answer"] = window(rs, ls, k, ys, idx=idx)
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}))
    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(f"{a.out}/a4.json", "w"), indent=1, default=float)
    f = lambda v: "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.2f}"
    with open(f"{a.out}/a4.md", "w") as fh:
        fh.write("| task | n | question only | picture only | both | V-Read (orig) | grey: V-Read vs orig answer | grey: answers changed | "
                 "mismatch: V-Read vs orig answer | mismatch: V-Read vs own answer | mismatch: answers changed |\n")
        fh.write("|" + "---|" * 11 + "\n")
        for r in rows:
            fh.write(f"| {r['task']} | {r['n']} | {f(r['q_only'])} | {f(r['img_only'])} | {f(r['q_img'])} | {f(r.get('orig_v_read'))} | "
                     f"{f(r.get('grey_v_read_vs_orig_answer'))} | {f(r.get('grey_answer_changed'))} | "
                     f"{f(r.get('mismatch_v_read_vs_orig_answer'))} | {f(r.get('mismatch_v_read_vs_own_answer'))} | "
                     f"{f(r.get('mismatch_answer_changed'))} |\n")
    print("wrote", f"{a.out}/a4.md")


if __name__ == "__main__":
    main()
