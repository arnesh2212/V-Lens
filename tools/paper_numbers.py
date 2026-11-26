# SPDX-License-Identifier: Apache-2.0
"""The paper's numbers that no table script writes, recomputed from results/.

Early reading (V-Read minus each reader before 0.45 of the depth, interval over
tasks), how often each reader is best, V-Write's fall with its top 20% of image
tokens set to zero against the best established method, the start-point and
step ablation, pixel covering, the other-question control and V-Write's share
on the password mark under each control.

    python3 tools/paper_numbers.py [--results results] [--out results/checks/paper_numbers.json]
"""
import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
warnings.simplefilter("ignore")

MODELS = ("Qwen3-VL-8B-Instruct", "Qwen3.5-27B", "gemma-4-26B-A4B-it", "InternVL3_5-8B-HF")
READERS = ("tuned", "jacobian", "patchscope", "logit")
ESTABLISHED = ("attr_patch", "ig_zero", "grad_x_act", "act_patch", "kernel_shap", "gradcam")


def boot_mean(x, reps=10000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if x.size == 0:
        return None
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (reps, x.size)).mean(1)
    return [round(float(x.mean()), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def simonsays_families():
    import make_paper_tables as MP
    return [f for _, f in MP.ROWS if f and not f.startswith(("coco", "refusal"))]


def early_reading(results):
    from vlens.figures import fx_detect as D
    out = {}
    for m in MODELS:
        per = {k: [] for k in READERS}
        for f in simonsays_families():
            r = D._family(results, m, f)
            if r is None or "v_read" not in r["auroc"]:
                continue
            early = r["depth"] < 0.45
            v = r["auroc"]["v_read"][early]
            for k in READERS:
                if k not in r["auroc"]:
                    continue
                b = r["auroc"][k][early]
                ok = np.isfinite(v) & np.isfinite(b)
                if ok.any():
                    per[k].append(float(v[ok].mean() - b[ok].mean()))
        res = {}
        for k, d in per.items():
            if d:
                ci = boot_mean(d)
                res[k] = dict(diff=ci[0], ci=ci[1:], ahead=int(sum(x > 0 for x in d)), tasks=len(d))
        if res:
            out[m] = res
    return out


def reader_consistency(results):
    import make_vread_table as T
    table, _ = T.collect(results)
    fams = simonsays_families()
    auth = [f for f in fams if f.endswith("_auth")]

    def stats(readers, fs):
        acc = {r: dict(best=0, short=[]) for r in readers}
        for m in table:
            for f in fs:
                c = table[m].get(f)
                band = (c or {}).get("band") or {}
                b = {r: band[r] for r in readers if r in band}
                if "v_read" not in b:
                    continue
                top = max(b.values())
                acc[max(b, key=b.get)]["best"] += 1
                for r, v in b.items():
                    acc[r]["short"].append(top - v)
        return {r: dict(best=a["best"], pairs=len(a["short"]),
                        mean_shortfall=round(float(np.mean(a["short"])), 4) if a["short"] else None)
                for r, a in acc.items()}

    return {"all readers": stats(("v_read",) + READERS, fams),
            "one-matmul readers": stats(("v_read", "tuned", "jacobian", "logit"), fams),
            "authority tasks, all readers": stats(("v_read",) + READERS, auth)}


def _baseline_rows(results, m):
    p = f"{results}/baselines/{m}__patchlenses.json"
    if not os.path.exists(p):
        return None
    from vlens.data import simonsays as SS
    return [x for x in json.load(open(p))["rows"] if not SS.is_password_locking(x)]


def _fall(rows, mm, k=0.2, tag="zero"):
    return [x["s0"] - x["after"].get(f"{tag}|{mm}|{k}", np.nan) for x in rows]


def vwrite_zeroing(results):
    out = {}
    for m in MODELS:
        rows = _baseline_rows(results, m)
        if not rows:
            continue
        means = {mm: np.nanmean(_fall(rows, mm)) for mm in ESTABLISHED
                 if any(f"zero|{mm}|0.2" in x["after"] for x in rows)}
        best = max(means, key=means.get)
        vw = np.array(_fall(rows, "v_write"))
        bb = np.array(_fall(rows, best))
        out[m] = dict(n=len(rows), best_other=best, v_write=boot_mean(vw), best=boot_mean(bb),
                      random=boot_mean(_fall(rows, "random")), v_write_minus_best=boot_mean(vw - bb))
    return out


def vwrite_ablation(results):
    out = {}
    for m in MODELS:
        rows = _baseline_rows(results, m)
        if not rows:
            continue
        out[m] = {mm: dict(fall=boot_mean(_fall(rows, mm)),
                           agree=boot_mean([x.get("truth_agreement", {}).get(f"zero|{mm}", np.nan) for x in rows]),
                           n=len(rows))
                  for mm in ("v_write", "ig_mean32", "ig_zero")}
    return out


def pixel_covering(results):
    p = f"{results}/occlusion/Qwen3-VL-8B-Instruct__occlusion.json"
    if not os.path.exists(p):
        return None
    s = json.load(open(p))["summary"]
    j = s["ks"].index(0.1)
    return {meth: dict(top=round(s["a3"][f"{meth}|top"][j], 3), random=round(s["a3"][f"{meth}|random"][j], 3))
            for meth in s["methods"]}


def cross_question(results):
    out = {}
    for p in glob.glob(f"{results}/crossquestion/*__crossquestion.json"):
        s = json.load(open(p))["summary"]
        j = s["ks"].index(0.2)
        out[os.path.basename(p).split("__")[0]] = dict(n=s["n"], **{k: round(v[j], 3) for k, v in s["curve"].items()})
    return out


def password_lock(results):
    p = next((q for q in (f"{results}/{d}/Qwen3-VL-8B-Instruct__vwrite_all.part.jsonl"
                          for d in ("pwlock_guess", "pwlock")) if os.path.exists(q)), None)
    if p is None:
        return None
    rows = [json.loads(line) for line in open(p)]
    sel = {"mark, variants 1-4": lambda r: r["kind"] == "test" and r["marked"] and r["seen"],
           "mark, variants 5-8": lambda r: r["kind"] == "test" and r["marked"] and not r["seen"]}
    for c in sorted({r["control"] for r in rows if r.get("control")}):
        sel[c] = lambda r, c=c: r.get("control") == c
    out = {}
    for model in ("locked", "base"):
        for name, f in sel.items():
            xs = [r for r in rows if r["model"] == model and f(r) and r.get("mass") is not None]
            if xs:
                out.setdefault(model, {})[name] = dict(
                    n=len(xs), share=boot_mean([r["mass"] for r in xs]),
                    hit1=round(float(np.mean([r["hit1"] for r in xs])), 3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    got = {}
    for name, fn in (("early_reading", early_reading), ("reader_consistency", reader_consistency),
                     ("vwrite_zeroing", vwrite_zeroing), ("vwrite_ablation", vwrite_ablation),
                     ("pixel_covering", pixel_covering), ("cross_question", cross_question),
                     ("password_lock", password_lock)):
        try:
            got[name] = fn(a.results)
        except Exception as e:
            got[name] = f"skipped: {type(e).__name__}: {e}"
        print(f"== {name}\n{json.dumps(got[name], indent=1)}", flush=True)
    out = a.out or os.path.join(a.results, "checks", "paper_numbers.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(got, open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
