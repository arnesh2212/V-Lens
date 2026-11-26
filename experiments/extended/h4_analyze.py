from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np


def _rows(d, model):
    for p in sorted(glob.glob(os.path.join(ROOT, d, f"{model}__patchcausal*.json"))):
        return json.load(open(p))["rows"]
    for p in sorted(glob.glob(os.path.join(ROOT, d, f"{model}__patchcausal*.rows.jsonl"))):
        return [json.loads(l) for l in open(p) if l.strip()]
    return []


def _boot(vals, groups, n=2000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return [None, None, None]
    g = np.asarray(groups)
    ug = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in ug}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug))
        bs.append(np.concatenate([vals[idx[u]] for u in pick]).mean())
    return [float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dirs", nargs="+", default=["results/extended/h4/patchcausal", "results/extended/h4/patchcausal_rel", "results/extended/h4/patchcausal_more"])
    ap.add_argument("--items", default="results/extended/h4/coco_items.json")
    a = ap.parse_args()
    from vlens.data import coco

    items = {r["id"]: r for r in json.load(open(os.path.join(ROOT, a.items)))["records"]}
    out = dict(model=a.model, by_dir={})
    for d in a.dirs:
        rows = [r for r in _rows(d, a.model) if r.get("id") in items]
        if not rows:
            continue
        res = dict(localisation={}, relation_split={})
        acc = defaultdict(lambda: defaultdict(list))
        for r in rows:
            loc = r.get("loc") or {}
            for rk, mets in loc.items():
                if isinstance(mets, dict):
                    for k, v in mets.items():
                        if isinstance(v, (int, float)) and v == v:
                            acc[(r["family"], rk)][k].append((float(v), r.get("image")))
        for (fam, rk), mets in sorted(acc.items()):
            res["localisation"].setdefault(fam, {})[rk] = {
                k: dict(zip(("mean", "lo", "hi"), _boot([v for v, _ in xs], [g for _, g in xs])), n=len(xs))
                for k, xs in mets.items()}
        split = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["family"] != "coco_rel" or not r.get("maps") or not r.get("grid"):
                continue
            it = items[r["id"]]
            Gr, Gc = r["grid"]
            m = it["meta"]
            cov_a = coco.cell_coverage(dict(it, meta=dict(m, polys=m["polys_a"])), Gr, Gc)
            cov_b = coco.cell_coverage(dict(it, meta=dict(m, polys=m["polys_b"])), Gr, Gc)
            ina, _ = coco.inside_cells(cov_a)
            inb, _ = coco.inside_cells(cov_b)
            inb = inb & ~ina
            g = str(m["image_id"])
            split["_chance"]["area_a"].append((float(ina.mean()), g))
            split["_chance"]["area_b"].append((float(inb.mean()), g))
            for rk, w in r["maps"].items():
                w = np.asarray(w, float)
                if w.size != Gr * Gc:
                    continue
                pos = np.clip(w, 0, None)
                tot = pos.sum()
                if tot <= 0:
                    continue
                top = int(np.argmax(w))
                for k, v in (("share_a", pos[ina].sum() / tot), ("share_b", pos[inb].sum() / tot),
                             ("share_outside", pos[~(ina | inb)].sum() / tot),
                             ("top_on_a", float(ina[top])), ("top_on_b", float(inb[top])),
                             ("both_objects_in_top10pct", float(ina[np.argsort(-w)[:max(len(w) // 10, 1)]].any()
                                                                and inb[np.argsort(-w)[:max(len(w) // 10, 1)]].any()))):
                    split[rk][k].append((float(v), g))
                    split[rk][f"{k}|{m['h4']['answer']}"].append((float(v), g))
        for rk, mets in sorted(split.items()):
            res["relation_split"][rk] = {k: dict(zip(("mean", "lo", "hi"), _boot([v for v, _ in xs], [g for _, g in xs])),
                                                 n=len(xs)) for k, xs in sorted(mets.items())}
        out["by_dir"][d] = res
        print(f"== {d} ({len(rows)} rows)")
        for fam, rks in res["localisation"].items():
            for rk, mets in sorted(rks.items()):
                show = " ".join(f"{k} {v['mean']:.2f}" for k, v in sorted(mets.items())
                                if v["mean"] is not None and k in ("hit1", "auroc", "energy"))
                print(f"  {fam:11s} {rk:16s} n={next(iter(mets.values()))['n']:4d} {show}")
        if res["relation_split"]:
            ch = res["relation_split"].get("_chance", {})
            print(f"  coco_rel chance: area a {ch.get('area_a', {}).get('mean')}, area b {ch.get('area_b', {}).get('mean')}")
            for rk, mets in res["relation_split"].items():
                if rk == "_chance":
                    continue
                f = lambda k: mets.get(k, {}).get("mean")
                print(f"  coco_rel {rk:16s} on asked {f('share_a'):.2f}  on reference {f('share_b'):.2f}  "
                      f"elsewhere {f('share_outside'):.2f}  top on asked {f('top_on_a'):.2f} / reference {f('top_on_b'):.2f}")
    p = os.path.join(ROOT, "results", "extended", "h4", f"{a.model}__h4_summary.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
