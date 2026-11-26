from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

BOOT = 2000
MAIN = "v_write"
KDEL = 0.2


def load(path):
    if path.endswith(".json"):
        return json.load(open(path))["rows"]
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def model_files(d, kind):
    out = {}
    for p in sorted(glob.glob(f"{d}/*__{kind}.json")) + sorted(glob.glob(f"{d}/*__{kind}.rows.jsonl")):
        name = os.path.basename(p).split("__")[0]
        out.setdefault(name, p)
    return out


def ci(v, rng):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(v), size=(BOOT, len(v)))
    bs = v[idx].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def verdict(lo, hi):
    if not np.isfinite(lo):
        return "n/a"
    return "win" if lo > 0 else ("loss" if hi < 0 else "tie")


def curves(rows):
    out = defaultdict(lambda: defaultdict(list))
    for i, x in enumerate(rows):
        ref_state = x["s0_fast"]
        ref_pix = x.get("s0_pixel_path") or x["s0"]
        acc = defaultdict(list)
        for key, v in x["after"].items():
            parts = key.split("|")
            if len(parts) != 4 or parts[0] not in ("del", "ins"):
                continue
            mode, rep, meth, k = parts
            rep_ = "other" if re.fullmatch(r"other\d+", rep) else rep
            acc[(mode, rep_, meth, float(k))].append(v)
        for (mode, rep, meth, k), vs in acc.items():
            ref = ref_pix if rep.startswith("px_") else ref_state
            out[(mode, rep, meth)][k].append((i, float(np.mean(vs)), ref))
    return out


def per_record(cur, mode, rep, meth, k, n):
    v = np.full(n, np.nan)
    for i, aft, ref in cur.get((mode, rep, meth), {}).get(k, []):
        v[i] = ref - aft
    return v


def auc(cur, mode, rep, meth, n):
    d = cur.get((mode, rep, meth), {})
    ks = sorted(d)
    if not ks:
        return np.full(n, np.nan)
    M = np.full((n, len(ks)), np.nan)
    for j, k in enumerate(ks):
        for i, aft, ref in d[k]:
            M[i, j] = aft / ref if ref else np.nan
    xs = np.array([0.0, *ks])
    first = 1.0 if mode == "del" else 0.0
    M = np.concatenate([np.full((n, 1), first), M], 1)
    ok = np.isfinite(M).all(1)
    out = np.full(n, np.nan)
    out[ok] = np.trapezoid(M[ok], xs, axis=1) / xs[-1] if hasattr(np, "trapezoid") else \
        np.trapz(M[ok], xs, axis=1) / xs[-1]
    return out


def fmt(x, d=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/extended/b1")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    md, js = ["# B1 / C1 / B3 summary", ""], {}

    for name, path in model_files(a.dir, "b1").items():
        rows_all = load(path)
        js[name] = {}
        md += [f"## {name}  ({os.path.basename(path)})", ""]
        for set_ in ("coco", "ss", "absent"):
            rows = [x for x in rows_all if x["set"] == set_]
            if not rows:
                continue
            n = len(rows)
            cur = curves(rows)
            reps = sorted({r for (mo, r, _) in cur if mo == "del"},
                          key=lambda r: (r.startswith("px_"), r))
            meths = sorted({mm for (_, _, mm) in cur})
            title = dict(coco="COCO, object present (model says yes)", ss="SimonSays agreement records",
                         absent="COCO, object absent (model says no)")[set_]
            md += [f"### {title}: n = {n}", ""]
            res = {}
            wtl = defaultdict(lambda: [0, 0, 0])
            for rep in reps:
                md += [f"**Deletion, replacement `{rep}`** (fall at {int(KDEL * 100)}%, flip rate, deletion AUC; "
                       f"V-Write minus method, 95% CI)", "",
                       "| method | fall@20% | flip@20% | del. AUC ↓ | V-Write − method [CI] | verdict |",
                       "|---|---|---|---|---|---|"]
                vw = per_record(cur, "del", rep, MAIN, KDEL, n)
                for meth in meths:
                    fall = per_record(cur, "del", rep, meth, KDEL, n)
                    if not np.isfinite(fall).any():
                        continue
                    flips = np.array([np.nan] * n)
                    for i, aft, ref in cur.get(("del", rep, meth), {}).get(KDEL, []):
                        flips[i] = float(aft <= 0)
                    A = auc(cur, "del", rep, meth, n)
                    diff = vw - fall
                    lo, hi = ci(diff, rng)
                    vd = verdict(lo, hi) if meth != MAIN else "—"
                    if meth != MAIN and vd in ("win", "tie", "loss"):
                        wtl[rep][("win", "tie", "loss").index(vd)] += 1
                    res[f"{rep}|{meth}"] = dict(fall=float(np.nanmean(fall)), flip=float(np.nanmean(flips)),
                                                del_auc=float(np.nanmean(A)), diff=float(np.nanmean(diff)),
                                                ci=[lo, hi], verdict=vd, n=int(np.isfinite(fall).sum()))
                    md.append(f"| {meth} | {fmt(np.nanmean(fall))} | {fmt(np.nanmean(flips))} | {fmt(np.nanmean(A))} | "
                              f"{fmt(np.nanmean(diff)) if meth != MAIN else '—'} [{fmt(lo)}, {fmt(hi)}] | {vd} |")
                md.append("")
            md += ["**V-Write win / tie / loss against every other method, by replacement** "
                   "(paired bootstrap on the fall at 20%)", "", "| replacement | win | tie | loss |", "|---|---|---|---|"]
            for rep in reps:
                w, t, l_ = wtl[rep]
                md.append(f"| {rep} | {w} | {t} | {l_} |")
            md.append("")
            ins = sorted({r for (mo, r, _) in cur if mo == "ins"})
            if ins:
                md += ["**Insertion AUC** (only the top k kept, the rest replaced; share of margin kept, higher is better)",
                       "", "| method | " + " | ".join(ins) + " |", "|---|" + "---|" * len(ins)]
                for meth in meths:
                    vals = [np.nanmean(auc(cur, "ins", r, meth, n)) for r in ins]
                    if np.isfinite(vals).any():
                        md.append(f"| {meth} | " + " | ".join(fmt(v) for v in vals) + " |")
                md.append("")
            vws = [mm for mm in (MAIN, "vw_zero", "vw_dmean", "vw_other", "vw_blur") if mm in meths]
            if len(vws) > 1:
                diag = {MAIN: "mean", "vw_zero": "zero", "vw_dmean": "dmean", "vw_other": "other"}
                md += ["**Cross-matrix**: V-Write integrated from each baseline (rows) graded under each "
                       "replacement (columns); fall at 20%, best per column in bold, own baseline marked *",
                       "", "| V-Write from | " + " | ".join(reps) + " |", "|---|" + "---|" * len(reps)]
                tab = {vv: [np.nanmean(per_record(cur, "del", r, vv, KDEL, n)) for r in reps] for vv in vws}
                best = [np.nanmax([tab[vv][j] for vv in vws]) for j in range(len(reps))]
                for vv in vws:
                    cells = []
                    for j, r in enumerate(reps):
                        s = fmt(tab[vv][j])
                        if np.isfinite(tab[vv][j]) and tab[vv][j] == best[j]:
                            s = f"**{s}**"
                        if diag.get(vv) == r:
                            s += "*"
                        cells.append(s)
                    md.append(f"| {vv.replace('v_write', 'v_write (mean)')} | " + " | ".join(cells) + " |")
                md.append("")
                res["cross_matrix"] = {vv: dict(zip(reps, map(float, tab[vv]))) for vv in vws}
            tk = sorted({k for x in rows for k in (x.get("truth") or {})})
            if tk:
                treps = sorted({k.split("|")[0] for k in tk})
                md += ["**Agreement with measured patching of 16 regions** (Spearman, mean over records)", "",
                       "| method | " + " | ".join(treps) + " |", "|---|" + "---|" * len(treps)]
                for meth in meths:
                    vals = [np.nanmean([x["truth"].get(f"{r}|{meth}", np.nan) for x in rows]) for r in treps]
                    if np.isfinite(vals).any():
                        md.append(f"| {meth} | " + " | ".join(fmt(v) for v in vals) + " |")
                md.append("")
            graded = [x for x in rows if x.get("loc")]
            if graded:
                lm = sorted({k for x in graded for k in x["loc"]})
                md += [f"**C1 / localisation against the human mask** (n = {len(graded)}; 95% CI over photographs)", "",
                       "| map | hit@1 | AUROC | energy |", "|---|---|---|---|"]
                for mm in lm:
                    cols = []
                    for q in ("hit1", "auroc", "energy"):
                        v = np.array([float((x["loc"].get(mm) or {}).get(q, np.nan)) for x in graded])
                        lo, hi = ci(v, rng)
                        cols.append(f"{fmt(np.nanmean(v))} [{fmt(lo)}, {fmt(hi)}]")
                    md.append(f"| {mm} | " + " | ".join(cols) + " |")
                    res[f"loc|{mm}"] = {q: float(np.nanmean([float((x["loc"].get(mm) or {}).get(q, np.nan))
                                                             for x in graded])) for q in ("hit1", "auroc", "energy")}
                md.append("")
            cq = [(i, x) for i, x in enumerate(rows) if x.get("cross")]
            if cq:
                cm = sorted({k.split("|")[2] for _, x in cq for k in x["after"] if k.startswith("crossB|")})
                md += [f"**C1 / cross-question** (n = {len(cq)} photographs): question A's margin falls when "
                       "the top 20% of map A (own) or of map B (made for another object present in the same "
                       "photograph) is replaced by the picture mean. A map that belongs to the question costs "
                       f"its own question more. Image-token states for A and B differ by at most "
                       f"{max(x['cross']['image_state_maxdiff'] for _, x in cq):.2g} (the image precedes the question).",
                       "", "| map | own (A) | other (B) | own − other [CI] | rank corr. A vs B |", "|---|---|---|---|---|"]
                for mm in cm:
                    own = np.array([x["s0_fast"] - x["after"].get(f"del|mean|{mm}|0.2", np.nan) for _, x in cq])
                    oth = np.array([x["s0_fast"] - x["after"].get(f"crossB|mean|{mm}|0.2", np.nan) for _, x in cq])
                    rc = np.nanmean([x["cross"]["rank_corr"].get(mm, np.nan) for _, x in cq])
                    lo, hi = ci(own - oth, rng)
                    md.append(f"| {mm} | {fmt(np.nanmean(own))} | {fmt(np.nanmean(oth))} | "
                              f"{fmt(np.nanmean(own - oth))} [{fmt(lo)}, {fmt(hi)}] | {fmt(rc)} |")
                    res[f"cross|{mm}"] = dict(own=float(np.nanmean(own)), other=float(np.nanmean(oth)), ci=[lo, hi])
                md.append("")
            secs = defaultdict(list)
            for x in rows:
                for k, v in (x.get("secs") or {}).items():
                    secs[k].append(v)
            md += ["Seconds per record: " + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in secs.items())
                   + f"; wall {np.mean([x['wall'] for x in rows]):.1f}", ""]
            res["wtl"] = {r: dict(zip(("win", "tie", "loss"), v)) for r, v in wtl.items()}
            js[name][set_] = res

    for name, path in model_files(a.dir, "pareto").items():
        rows = load(path)
        if not rows:
            continue
        md += [f"## B3 Pareto: {name} (n = {len(rows)})", "",
               "| method@budget | s / picture | fall@20% dmean | fall@20% other | fall@20% mean | "
               "patching agreement (dmean) | COCO AUROC |", "|---|---|---|---|---|---|---|"]
        keys = list(rows[0]["pareto"])
        out = {}
        for k in keys:
            q = [x["pareto"][k] for x in rows if k in x["pareto"]]
            s = np.mean([v["secs"] for v in q])
            f_d = np.nanmean([v["drop"]["dmean|0.2"] for v in q])
            f_o = np.nanmean([v["drop"]["other0|0.2"] for v in q])
            f_m = np.nanmean([v["drop"]["mean|0.2"] for v in q])
            t_d = np.nanmean([v.get("truth_dmean", np.nan) for v in q])
            au = np.nanmean([v.get("auroc", np.nan) for v in q])
            out[k] = dict(secs=float(s), fall_dmean=float(f_d), fall_other=float(f_o), fall_mean=float(f_m),
                          truth_dmean=float(t_d), auroc=float(au))
            md.append(f"| {k} | {s:.2f} | {fmt(f_d)} | {fmt(f_o)} | {fmt(f_m)} | {fmt(t_d)} | {fmt(au)} |")
        md.append("")
        js.setdefault(name, {})["pareto"] = out
    open(f"{a.dir}/summary_b1.md", "w").write("\n".join(md) + "\n")
    json.dump(js, open(f"{a.dir}/summary_b1.json", "w"), indent=1, default=float)
    print("\n".join(md[:400]))
    print(f"wrote {a.dir}/summary_b1.md")


if __name__ == "__main__":
    main()
