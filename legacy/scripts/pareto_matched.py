# SPDX-License-Identifier: Apache-2.0
"""Rebuild the Pareto frontier and its bootstrap from a 90_matched checkpoint.

Pure pandas/numpy, no torch and no GPU: the experiment checkpoints its parquets
every five items, so every headline number can be recomputed from a partial run
without touching the card again.
"""
import sys
import numpy as np
import pandas as pd

tag = sys.argv[1] if len(sys.argv) > 1 else "match8b"
d = sys.argv[2] if len(sys.argv) > 2 else "/workspace/vlm/results/matched"
B = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

DF = pd.read_parquet(f"{d}/{tag}_intervention.parquet")
BEH = pd.read_parquet(f"{d}/{tag}_behaviour.parquet")
base = BEH[BEH.cond == "baseline"]
# only items that reached the ablation sweep count in the baseline rate
done = set(DF.id.unique())
base = base[base.id.isin(done)]
base_asr = float(base.complied.mean())
base_asr_strict = float(base.complied_strict.mean())
n_items = base.id.nunique()
print(f"{tag}: {n_items} images, {len(DF)} ablation rows")
print(f"baseline ASR {base_asr:.3f} (refusal-regex, primary) / "
      f"{base_asr_strict:.3f} (regex AND judge)\n")

rows = []
for (rule, k), g in DF.groupby(["rule", "k"]):
    gc = g[g.base_complied == 1]
    rows.append(dict(
        rule=rule, k=int(k), n=len(g),
        asr_after=float(g.complied.mean()),
        asr_removed=base_asr - float(g.complied.mean()),
        asr_removed_oncomplied=(1.0 - float(gc.complied.mean())
                                if len(gc) else np.nan),
        benign_retained=float(g.benign_agree.mean()),
        benign_lp_drop=float(g.benign_lp_drop.mean())))
PAR = pd.DataFrame(rows).sort_values(["rule", "k"])
print("PARETO: attack removed vs benign retained")
print(PAR.to_string(index=False))

# ---- paired bootstrap OVER IMAGES ---------------------------------------
ids = sorted(DF.id.unique())
at = {v: i for i, v in enumerate(ids)}
n = len(ids)
cell = {}
for (r, k), g in DF.groupby(["rule", "k"]):
    c = np.full(n, np.nan)
    b = np.full(n, np.nan)
    for row in g.itertuples():
        c[at[row.id]] = row.complied
        b[at[row.id]] = row.benign_agree
    cell[(r, int(k))] = (c, b)
draws = np.random.default_rng(17).integers(0, n, size=(B, n))


def mo(v):
    with np.errstate(invalid="ignore"):
        return np.nanmean(v[draws], axis=1)


ks = sorted({k for (_, k) in cell})
out = []
for k in ks:
    if ("vj_interaction", k) not in cell:
        continue
    oc, ob = cell[("vj_interaction", k)]
    ocb, obb = mo(oc), mo(ob)
    for r in sorted({rr for (rr, kk) in cell if kk == k}):
        if r in ("vj_interaction", "all"):
            continue
        tc, tb = cell[(r, k)]
        d_asr = mo(tc) - ocb          # + => ours removes more attack
        d_ben = obb - mo(tb)          # + => ours damages benign less
        out.append(dict(
            k=k, vs=r,
            d_asr=float(np.nanmean(d_asr)),
            asr_lo=float(np.nanpercentile(d_asr, 2.5)),
            asr_hi=float(np.nanpercentile(d_asr, 97.5)),
            d_ben=float(np.nanmean(d_ben)),
            ben_lo=float(np.nanpercentile(d_ben, 2.5)),
            ben_hi=float(np.nanpercentile(d_ben, 97.5)),
            p_dom=float(np.nanmean((d_asr >= 0) & (d_ben >= 0)))))
print(f"\nBOOTSTRAP OVER IMAGES ({n} images, {B} resamples, paired within image)")
print("vj_interaction MINUS each control; positive d_asr = we remove more, "
      "positive d_ben = we damage less")
print(pd.DataFrame(out).to_string(index=False))
