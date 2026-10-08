# SPDX-License-Identifier: Apache-2.0
"""Every attribution method scored at KernelSHAP's own resolution: 16 groups of tokens.

KernelSHAP and RISE score 16 groups (a 4 x 4 split of the token grid) and give
each token its group's share, so token-level measures (hit@1, per-token AUROC)
compare them at a resolution they never had. Here every map, V-Write's included,
is summed into the same 16 groups before scoring:

* group hit@1   the top-scored group contains the asked object (any token of the
                group at least half covered by the human outline);
* group AUROC   group scores against whether the group is mostly on the object;
* group rank    Spearman correlation of group scores with the group's object coverage.

Read from results/coco_baselines/<m>__patchcausal.rows.jsonl (maps and per-token
coverage); nothing is rerun.  Writes results/checks/group_matched.json.
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

MODELS = ("Qwen3-VL-8B-Instruct", "Qwen3.5-27B")
METHODS = ("vwrite_mean", "kernel_shap", "rise", "ig_zero", "attr_patch", "gradxact_zero", "act_patch", "gradcam", "random")


def groups(gr, gc):
    cell = np.arange(gr * gc).reshape(gr, gc)
    return [cell[i * gr // 4:(i + 1) * gr // 4, j * gc // 4:(j + 1) * gc // 4].ravel() for i in range(4) for j in range(4)]


def auroc(s, y):
    p, n = s[y], s[~y]
    if len(p) == 0 or len(n) == 0:
        return np.nan
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())


def boot(x, rng, n=2000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(len(x))]


def run(results="results"):
    out = {}
    rng = np.random.default_rng(0)
    for m in MODELS:
        p = f"{results}/coco_baselines/{m}__patchcausal.rows.jsonl"
        if os.path.exists(p):
            rows = [json.loads(line) for line in open(p)]
        elif os.path.exists(p.replace(".rows.jsonl", ".json")):
            rows = json.load(open(p.replace(".rows.jsonl", ".json")))["rows"]
        else:
            continue
        # RISE on the 27B was run separately on a subset (results/coco_baselines_rise)
        rp = f"{results}/coco_baselines_rise/{m}__patchcausal.json"
        rise = {str(x["id"]): x.get("maps", {}).get("rise") for x in json.load(open(rp))["rows"]} if os.path.exists(rp) else {}
        per = {k: dict(hit=[], auc=[], rho=[]) for k in METHODS}
        diff = {k: dict(hit=[], auc=[], rho=[]) for k in METHODS if k != "vwrite_mean"}
        for r in rows:
            if not r.get("maps"):
                continue
            if "rise" not in r["maps"] and rise.get(str(r["id"])) is not None:
                r["maps"]["rise"] = rise[str(r["id"])]
            cov = np.asarray(r["cov"], float)
            G = groups(*r["grid"])
            on = cov >= 0.5 if (cov >= 0.5).any() else cov >= 0.25
            gcov = np.array([cov[g].mean() for g in G])
            gon = np.array([on[g].any() for g in G])
            gmost = gcov >= 0.5 if (gcov >= 0.5).any() else gcov == gcov.max()
            got = {}
            for k in METHODS:
                mp = r["maps"].get(k)
                if mp is None:
                    continue
                mp = np.asarray(mp, float)
                gs = np.array([mp[g].sum() for g in G])
                got[k] = dict(hit=float(gon[int(np.argmax(gs))]), auc=auroc(gs, gmost),
                              rho=float(spearmanr(gs, gcov).correlation) if np.ptp(gcov) > 0 else np.nan)
                for q in ("hit", "auc", "rho"):
                    per[k][q].append(got[k][q])
            if "vwrite_mean" in got:
                for k in diff:
                    if k in got:
                        for q in ("hit", "auc", "rho"):
                            diff[k][q].append(got["vwrite_mean"][q] - got[k][q])
        out[m] = dict(n=len(per["vwrite_mean"]["hit"]),
                      score={k: {q: boot(v, rng) for q, v in d.items() if v} for k, d in per.items() if d["hit"]},
                      vwrite_minus={k: {q: boot(v, rng) for q, v in d.items() if v} for k, d in diff.items() if d["hit"]})
        print(m, out[m]["n"])
        for k, d in out[m]["score"].items():
            print(f"  {k:14s}", {q: round(v[0], 3) for q, v in d.items()})
        for k in ("kernel_shap", "rise", "ig_zero"):
            if k in out[m]["vwrite_minus"]:
                print("  V-Write minus", k, {q: [round(x, 3) for x in v[:3]] for q, v in out[m]["vwrite_minus"][k].items()})
    os.makedirs(f"{results}/checks", exist_ok=True)
    json.dump(out, open(f"{results}/checks/group_matched.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    run()
