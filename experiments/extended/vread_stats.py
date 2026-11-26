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
import make_vread_table as T
from scipy.stats import chi2, rankdata

READERS = ("v_read", "tuned", "jacobian", "patchscope", "logit")
BASE = READERS[1:]
NICE = {"v_read": "V-Read", "tuned": "TL", "jacobian": "JL", "patchscope": "PS", "logit": "LL"}
WIDTH = 0.35
MAX_END = 0.8
warnings.filterwarnings("ignore")


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    p = int(y.sum()); n = len(y) - p
    if p == 0 or n == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def within(s, y, gidx):
    num = den = 0.0
    for ii in gidx:
        ss, yy = s[ii], y[ii]
        ok = np.isfinite(ss)
        ss, yy = ss[ok], yy[ok]
        p = int(yy.sum()); n = len(yy) - p
        if p and n:
            num += auc(ss, yy) * p * n; den += p * n
    return num / den if den else np.nan


def centred(s, y, gidx):
    c = np.array(s, float).copy()
    for ii in gidx:
        m = np.nanmean(c[ii]) if np.isfinite(c[ii]).any() else 0.0
        c[ii] = c[ii] - m
    return auc(c, y)


def groups_of(keys):
    keys = np.asarray(keys, str)
    G = sorted(set(keys))
    return [np.where(keys == g)[0] for g in G]


def cell(m, fam, ps_pick, reps, rng, results="results"):
    res, loc = T._capture(T.D._family, results, m, fam)
    if res is None or not loc:
        return None
    y = np.asarray(loc["y"], int)
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return None
    dep = np.asarray(res["depth"], float)
    a_of = {k: np.asarray(a, float) for k, a in res["auroc"].items()}
    if "v_read" not in a_of:
        return None
    vb = np.isfinite(a_of["v_read"])
    inb = (dep >= T.BAND[0]) & (dep <= T.BAND[1]) & vb
    series = dict(loc["series"])
    if ps_pick and ps_pick in series:
        series["patchscope"] = series[ps_pick]
    R = [k for k in READERS if k in series]
    if len(R) < 2:
        return None
    beh_p = f"{results}/simonsays/{m}__{fam}.json"
    beh = {r["id"]: r for r in json.load(open(beh_p))["rows"]} if os.path.exists(beh_p) else {}
    q = np.array([str(beh.get(i, {}).get("question", i)) for i in loc["ids"]])
    pics = np.asarray(loc["groups"], str)
    qg = groups_of(q)
    usable = sum(1 for ii in qg if 0 < y[ii].sum() < len(ii))
    cols = np.nonzero(inb)[0]
    fcols = np.nonzero(vb)[0]
    out = dict(model=m, fam=fam, n=int(len(y)), n_pos=int(y.sum()), questions=len(qg), usable_groups=usable,
               readers=R)

    def wmean(fn, k, idx=None, cc=cols):
        vals = []
        for j in cc:
            s = series[k][:, j] if idx is None else series[k][idx, j]
            if np.isfinite(s).sum() >= 2:
                vals.append(fn(s, j))
        return float(np.nanmean(vals)) if vals else np.nan

    pt = {}
    for k in R:
        pt[k] = dict(
            all=wmean(lambda s, j: auc(s, y), k),
            full=wmean(lambda s, j: auc(s, y), k, cc=fcols),
            within=wmean(lambda s, j: within(s, y, qg), k) if usable else np.nan,
            centred=wmean(lambda s, j: centred(s, y, qg), k) if usable else np.nan,
        )
        per = np.array([auc(series[k][:, j], y) for j in fcols])
        for thr in (0.8, 0.9):
            ok = np.isfinite(per) & (per >= thr)
            first = np.nan
            for i in range(len(per)):
                if ok[i:].all():
                    first = float(dep[fcols[i]]); break
            pt[k][f"first{int(thr * 100)}"] = first
    upics = np.array(sorted(set(pics)))
    fold = {p_: i % 5 for i, p_ in enumerate(np.random.default_rng(7).permutation(upics))}
    f = np.array([fold[p_] for p_ in pics])
    qprior = np.full(len(y), np.nan)
    for k_ in range(5):
        tr = f != k_
        rate = {}
        for g in set(q[tr]):
            rate[g] = y[tr & (q == g)].mean()
        glob_ = y[tr].mean() if tr.any() else 0.5
        for i in np.nonzero(f == k_)[0]:
            qprior[i] = rate.get(q[i], glob_)
    out["q_only"] = auc(qprior, y) if len(set(np.round(qprior, 6))) > 1 else 0.5
    ql = np.log(np.clip(qprior, 0.02, 0.98) / (1 - np.clip(qprior, 0.02, 0.98)))
    for k in R:
        S = np.nanmean(np.where(np.isfinite(series[k][:, cols]), series[k][:, cols], np.nan), axis=1)
        if not np.isfinite(S).any():
            continue
        S = np.where(np.isfinite(S), S, np.nanmean(S))
        z = (S - S.mean()) / (S.std() + 1e-9)
        pt[k].update(incremental(y, ql, z, pics, upics, rng))

    for k in R:
        pt[k]["xsel"] = xsel(series[k], y, dep, vb, pics, rng, crit="all")
        pt[k]["xsel_within"] = xsel(series[k], y, dep, vb, pics, rng, crit="within", q=q) if usable else np.nan
    out["point"] = pt

    base = [k for k in R if k != "v_read"]
    W, _G = T._weights(pics, reps=reps, seed=0)
    boot_all = T._boot_blocks(y, {k: series[k] for k in R}, cols, W)
    boot_full = T._boot_blocks(y, {k: series[k] for k in R}, fcols, W)
    out["ci"] = {}
    for name, B, cc in (("all", boot_all, cols), ("full", boot_full, fcols)):
        got = {k: np.nanmean(B[k][:, cc], axis=1) for k in R}
        d = got["v_read"] - np.nanmax(np.vstack([got[k] for k in base]), axis=0)
        d = d[np.isfinite(d)]
        out["ci"][name] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if d.size else None
    pb = []
    for j in fcols:
        bv = boot_full["v_read"][:, j]
        bb = np.nanmax(np.vstack([boot_full[k][:, j] for k in base]), axis=0)
        d = bv - bb
        d = d[np.isfinite(d)]
        if d.size < 50:
            continue
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        pb.append(dict(block=int(j), depth=float(dep[j]), diff=float(np.median(d)), p=float(max(p, 1.0 / d.size))))
    out["per_block"] = pb
    if usable:
        pid = {p: np.where(pics == p)[0] for p in upics}
        for name, fn in (("within", within), ("centred", centred)):
            ds = []
            for _ in range(reps // 2):
                pick = rng.integers(0, len(upics), len(upics))
                idx = np.concatenate([pid[upics[i]] for i in pick])
                gg = groups_of(q[idx])
                yy = y[idx]
                v = {}
                for kk in R:
                    vals = [fn(series[kk][idx, j], yy, gg) for j in cols]
                    v[kk] = np.nanmean(vals) if np.isfinite(vals).any() else np.nan
                b = np.nanmax([v[kk] for kk in base])
                ds.append(v["v_read"] - b)
            ds = np.array(ds); ds = ds[np.isfinite(ds)]
            out["ci"][name] = [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))] if ds.size > 20 else None
    gains = {k: pt[k].get("incr_boot") for k in R if pt[k].get("incr_boot") is not None}
    if "v_read" in gains and any(k in gains for k in base):
        L = min(len(v) for v in gains.values())
        d = np.array(gains["v_read"][:L]) - np.nanmax(np.vstack([np.array(gains[k][:L]) for k in base if k in gains]), axis=0)
        d = d[np.isfinite(d)]
        out["ci"]["incr"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if d.size > 20 else None
    for k in R:
        pt[k].pop("incr_boot", None)
    return out


def qg_of(i, qg):
    for ii in qg:
        if i in ii:
            return ii
    return np.array([i])


def incremental(y, ql, z, pics, upics, rng, folds=5, reps=200):
    from sklearn.linear_model import LogisticRegression
    fold = {p: i % folds for i, p in enumerate(rng.permutation(upics))}
    f = np.array([fold[p] for p in pics])
    p0 = np.zeros(len(y)); p1 = np.zeros(len(y))
    for k in range(folds):
        tr, te = f != k, f == k
        if te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        m0 = LogisticRegression(C=1.0).fit(ql[tr, None], y[tr])
        m1 = LogisticRegression(C=1.0).fit(np.c_[ql[tr], z[tr]], y[tr])
        p0[te] = m0.predict_proba(ql[te, None])[:, 1]
        p1[te] = m1.predict_proba(np.c_[ql[te], z[te]])[:, 1]
    a0, a1 = auc(p0, y), auc(p1, y)
    m0 = LogisticRegression(C=1e4, max_iter=2000).fit(ql[:, None], y)
    m1 = LogisticRegression(C=1e4, max_iter=2000).fit(np.c_[ql, z], y)
    eps = 1e-9
    ll = lambda m, X: float(np.sum(y * np.log(m.predict_proba(X)[:, 1] + eps) + (1 - y) * np.log(1 - m.predict_proba(X)[:, 1] + eps)))
    LR = 2 * (ll(m1, np.c_[ql, z]) - ll(m0, ql[:, None]))
    r2 = np.random.default_rng(1)
    pid = {p: np.where(pics == p)[0] for p in upics}
    gb = []
    for _ in range(reps):
        pick = r2.integers(0, len(upics), len(upics))
        idx = np.concatenate([pid[upics[i]] for i in pick])
        gb.append(auc(p1[idx], y[idx]) - auc(p0[idx], y[idx]))
    return dict(incr_q=a0, incr_qs=a1, incr_gain=a1 - a0, lr=float(LR), lr_p=float(chi2.sf(max(LR, 0), 1)),
                incr_boot=gb)


def xsel(M, y, dep, vb, pics, rng, crit="all", q=None, splits=5):
    fb = np.nonzero(vb)[0]
    starts = sorted(set(float(dep[j]) for j in fb))
    wins = [(s, s + WIDTH) for s in starts if s + WIDTH <= MAX_END + 1e-9]
    if not wins:
        return np.nan
    upics = np.array(sorted(set(pics)))
    vals = []
    for sp in range(splits):
        r = np.random.default_rng(100 + sp)
        half = set(r.permutation(upics)[: len(upics) // 2])
        A = np.array([p in half for p in pics])
        for sel, ev in ((A, ~A), (~A, A)):
            def score(mask, w):
                cc = [j for j in fb if w[0] <= dep[j] <= w[1]]
                if not cc:
                    return np.nan
                yy = y[mask]
                if crit == "within":
                    gg = groups_of(q[mask])
                    v = [within(M[mask, j], yy, gg) for j in cc]
                else:
                    v = [auc(M[mask, j], yy) for j in cc]
                return np.nanmean(v) if np.isfinite(v).any() else np.nan
            sc = [score(sel, w) for w in wins]
            if not np.isfinite(sc).any():
                continue
            best = wins[int(np.nanargmax(sc))]
            vals.append(score(ev, best))
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def bh(ps, alpha=0.05):
    ps = np.asarray(ps, float)
    n = len(ps)
    if n == 0:
        return np.zeros(0, bool)
    o = np.argsort(ps)
    thr = alpha * np.arange(1, n + 1) / n
    ok = ps[o] <= thr
    k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    sig = np.zeros(n, bool); sig[o[:k]] = True
    return sig


def classify(ci):
    if ci is None:
        return "untested"
    return "win" if ci[0] > 0 else "behind" if ci[1] < 0 else "level"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(T.MODELS))
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--out", default="results/extended/vread_stats")
    ap.add_argument("--results", default="results", help="a results root holding readouts/ and simonsays/")
    ap.add_argument("--readers", nargs="*", default=None,
                    help="restrict the comparison readers, e.g. 'tuned' (V-Read vs its like-for-like lens) or "
                         "'tuned jacobian logit' (the linear readers); default: all five")
    a = ap.parse_args()
    if a.readers:
        global READERS
        READERS = ("v_read",) + tuple(r for r in a.readers if r != "v_read")
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    cells = []
    for m in a.models:
        try:
            ps_pick = T.readout_cells(a.results, m)[3]
        except Exception as e:
            print("no patchscope pick for", m, e); ps_pick = None
        for fam in T._readout_fams(a.results, m):
            if fam.startswith("coco"):
                continue
            c = cell(m, fam, ps_pick, a.reps, rng, a.results)
            if c is None:
                continue
            cells.append(c)
            p = c["point"]
            print(m, fam, "n", c["n"], "groups", c["usable_groups"],
                  "all", {NICE[k]: round(p[k]["all"], 3) for k in c["readers"]},
                  "within", {NICE[k]: round(p[k]["within"], 3) for k in c["readers"]}, c["ci"], flush=True)
    allp = [(i, j, b["p"]) for i, c in enumerate(cells) for j, b in enumerate(c["per_block"])]
    sig = bh([p for _, _, p in allp])
    for (i, j, _), s in zip(allp, sig):
        b = cells[i]["per_block"][j]
        b["bh"] = "win" if s and b["diff"] > 0 else "behind" if s and b["diff"] < 0 else "level"
    json.dump(cells, open(os.path.join(a.out, "vread_stats.json"), "w"), indent=1, default=float)
    write_md(cells, os.path.join(a.out, "vread_stats.md"))


def write_md(cells, path):
    L = ["# V-Read beyond Table 2", "",
         "Per task: V-Read and the best other reader (name), paired 95% interval of V-Read minus the best "
         "other reader (best taken inside each resample) and result. `all` = Table 2; `within` = within-question "
         "AUROC (pictures resampled, question labels kept); `centred` = question-mean-removed AUROC; `full` = every fitted block; "
         "`xsel` = width-0.35 window ending by depth 0.8, chosen per reader on one half of the pictures and scored on the other; `incr` = out-of-fold gain over "
         "the question prior.", ""]
    for metric in ("all", "within", "centred", "full", "xsel", "xsel_within", "incr_gain"):
        L += [f"## {metric}", "", "| model | task | n | groups | q-only | V-Read | best other | Δ | 95% CI | result |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        tally = {}
        for c in cells:
            p = c["point"]
            v = p["v_read"].get(metric, np.nan)
            others = {k: p[k].get(metric, np.nan) for k in c["readers"] if k != "v_read"}
            others = {k: x for k, x in others.items() if np.isfinite(x)}
            if not np.isfinite(v) or not others:
                continue
            bk = max(others, key=others.get)
            ci = c["ci"].get({"incr_gain": "incr"}.get(metric, metric))
            res = classify(ci) if metric in ("all", "within", "centred", "full", "incr_gain") else "-"
            tally.setdefault(c["model"], {}).setdefault(res, 0)
            tally[c["model"]][res] += 1
            L.append(f"| {c['model']} | {c['fam']} | {c['n']} | {c['usable_groups']} | {c['q_only']:.2f} | {v:.3f} | {others[bk]:.3f} {NICE[bk]} | "
                     f"{v - others[bk]:+.3f} | {('[%+.3f, %+.3f]' % tuple(ci)) if ci else '-'} | {res} |")
        L += ["", "tally: " + "; ".join(f"{m}: " + ", ".join(f"{k} {n}" for k, n in sorted(t.items())) for m, t in tally.items()), ""]
    L += ["## first depth staying >= 0.8 / 0.9 (lower is earlier)", "", "| model | task | " + " | ".join(NICE[k] for k in READERS) + " |",
          "|---|---|" + "---|" * len(READERS)]
    for c in cells:
        p = c["point"]
        L.append(f"| {c['model']} | {c['fam']} | " + " | ".join(
            (f"{p[k]['first80']:.2f}/{p[k]['first90']:.2f}" if k in p else "-") for k in READERS) + " |")
    L += ["", "## per-block, Benjamini-Hochberg over every (model, task, block)", ""]
    for m in sorted(set(c["model"] for c in cells)):
        t = {}
        for c in cells:
            if c["model"] != m:
                continue
            for b in c["per_block"]:
                t[b.get("bh", "level")] = t.get(b.get("bh", "level"), 0) + 1
        L.append(f"- {m}: " + ", ".join(f"{k} {n}" for k, n in sorted(t.items())))
    open(path, "w").write("\n".join(L) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
