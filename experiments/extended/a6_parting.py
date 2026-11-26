from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

PAIRS = (("openimages_orig_auth", "openimages_edit_auth"), ("vins_orig_auth", "vins_edit_auth"),
         ("nuscenes_orig_auth", "nuscenes_edit_auth"), ("gemini_orig_eval", "gemini_edit_eval"),
         ("gemini_orig_auth", "gemini_edit_swap"))
READERS = ("logit", "tuned", "jacobian", "patchscope")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", nargs="*", default=["Qwen3-VL-8B-Instruct", "Qwen3.5-27B",
                                                    "gemma-4-26B-A4B-it", "InternVL3_5-8B-HF"])
    ap.add_argument("--out", default="results/extended/a6")
    a = ap.parse_args()
    from vlens.figures import fx_vread_yesno as Y

    def js(p):
        try:
            return json.load(open(p))
        except Exception:
            return None

    def key(b, edited):
        mt = b.get("meta") or {}
        if b.get("dataset") == "gemini":
            return (mt.get("scene_id"), mt.get("question_id"), mt.get("eval_framing"), b["question"])
        img = str(mt.get("source_image")) if edited else str(b["image"])
        return (os.path.basename(img), b["question"])

    os.makedirs(a.out, exist_ok=True)
    report = {}
    for m in a.models:
        def load(fam):
            d = js(os.path.join(a.results, "readouts", f"{m}__{fam}_readouts.json"))
            b = js(os.path.join(a.results, "simonsays", f"{m}__{fam}.json"))
            ps = js(os.path.join(a.results, "readouts", f"{m}__{fam}_patchscope.json"))
            if not d or not b:
                return None
            R = {r["id"]: dict(r) for r in d["rows"]}
            if ps:
                for r in ps["rows"]:
                    if r["id"] in R:
                        R[r["id"]]["patchscope"] = r.get("patchscope")
            return R, {r["id"]: r for r in b["rows"]}
        pairs = []
        for fo, fe in PAIRS:
            A, E = load(fo), load(fe)
            if not A or not E:
                continue
            idx = {}
            for i, r in A[0].items():
                if r.get("kind") == "yesno" and i in A[1]:
                    idx.setdefault(key(A[1][i], False), []).append(i)
            for i, r in E[0].items():
                if r.get("kind") != "yesno" or i not in E[1]:
                    continue
                for j in idx.get(key(E[1][i], True), []):
                    ro = A[0][j]
                    if ro.get("v_read_fitted_on") or r.get("v_read_fitted_on"):
                        continue
                    g = [t for t in Y._grid_of(ro) if t in set(Y._grid_of(r))]
                    if len(g) < 4:
                        continue
                    fo_, fe_ = float(ro["logit"][-1]), float(r["logit"][-1])
                    s = np.sign(fe_ - fo_)
                    if s == 0:
                        continue
                    tau = min(0.5, abs(fe_ - fo_) / 2)
                    sep = {"v_read": Y._commit(np.array([r["v_read"][t] for t in g]) - np.array([ro["v_read"][t] for t in g]), s, tau)}
                    for k in READERS:
                        xo, xe = ro.get(k), r.get(k)
                        if not xo or not xe or len(xo) <= max(g) or len(xe) <= max(g) \
                                or any(xo[t] is None or xe[t] is None for t in g):
                            continue
                        sep[k] = Y._commit(np.array([xe[t] for t in g], float) - np.array([xo[t] for t in g], float), s, tau)
                    pairs.append(dict(fam=fe, n_g=len(g), sep=sep))
        res = dict(n=len(pairs))
        for k in READERS:
            P = [p for p in pairs if k in p["sep"]]
            if not P:
                continue
            sv = np.array([p["sep"]["v_read"] for p in P]); sk = np.array([p["sep"][k] for p in P])
            first, same, later = int((sv < sk).sum()), int((sv == sk).sum()), int((sv > sk).sum())
            res[k] = dict(n=len(P), vread_first=first, same=same, reader_first=later,
                          share_vread_first=first / len(P), share_reader_first=later / len(P),
                          sign_p=Y._sign_p(first, first + later),
                          median_depth_vread=float(np.median(sv / max(P[0]["n_g"], 1))),
                          median_depth_reader=float(np.median(sk / max(P[0]["n_g"], 1))))
        report[m] = res
        print(m, json.dumps({k: (v if not isinstance(v, dict) else
                                 {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()})
                             for k, v in res.items()}))
    json.dump(report, open(os.path.join(a.out, "parting.json"), "w"), indent=1)
    with open(os.path.join(a.out, "parting.md"), "w") as fh:
        fh.write("| model | reader | pairs | V-Read parts first | same block | reader parts first | sign-test p |\n|---|---|---|---|---|---|---|\n")
        for m, res in report.items():
            for k in READERS:
                if k in res:
                    v = res[k]
                    fh.write(f"| {m} | {k} | {v['n']} | {v['share_vread_first']:.0%} | {v['same'] / v['n']:.0%} | "
                             f"{v['share_reader_first']:.0%} | {v['sign_p']:.3g} |\n")
    print("wrote", os.path.join(a.out, "parting.md"))


if __name__ == "__main__":
    main()
