# SPDX-License-Identifier: Apache-2.0
"""Every attribution method against every SimonSays source, as one lattice of area-true circles.

Reads the run ``experiments/simonsays/patch_lenses.py --rankings v_write
attr_patch ig_zero ig_mean32 grad_x_act act_patch rise kernel_shap gradcam
random --out results/baselines`` wrote, the same file ``fx_baselines`` draws.
Each record carries ``truth_agreement["zero|<method>"]``: the Spearman
correlation, over sixteen contiguous groups of image patches, between the
method's summed attribution per group and the measured fall in the answer
score when that group is replaced by zeros. The summary of that run holds
only the pooled agreement and a per-source fall, so the per-source agreement
is computed here from the rows, grouping records by the dataset their family
name starts with (``paper.SOURCE``). Password-locking families are dropped by
``fx_baselines._load``. No WMDP question text is read or printed.
"""
from __future__ import annotations

import numpy as np

from vlens.figures import paper as P
from vlens.figures import style as st
from vlens.figures import fx_baselines as B
from vlens.figures.fx_layers import fit_lines

#: columns, left to right: every record, then the five sources
SOURCES = ("all", "openimages", "nuscenes", "vins", "gemini", "wmdp")
#: resamples of pictures for every interval
BOOT = 2000
#: the agreement at which a circle fills its cell; fixed so two models' files compare
VMAX = 0.65


def _src(r):
    return str(r["family"]).split("_")[0]


def _stats(D, seed=0):
    """Per source: n, pictures, and for every method the mean agreement and its picture draws."""
    rows, methods = D["rows"], D["methods"]
    pics = B._pictures(rows)
    out = {}
    for s in SOURCES:
        idx = [i for i, r in enumerate(rows) if s == "all" or _src(r) == s]
        if len(idx) < 12:
            continue
        clusters = [pics[i] for i in idx]
        uniq, cid = np.unique(clusters, return_inverse=True)
        rng = np.random.default_rng(seed)
        W = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=BOOT)[:, cid].astype(float)
        agree = {}
        for m in methods:
            v = np.array([float(rows[i]["truth_agreement"].get(f"zero|{m}", np.nan)) for i in idx])
            ok = np.isfinite(v)
            w = W[:, ok]
            agree[m] = (float(v[ok].mean()), (w * v[ok]).sum(1) / np.maximum(w.sum(1), 1e-12))
        out[s] = dict(n=len(idx), pictures=len(uniq), agree=agree)
    return dict(methods=methods, by=out)


def _pool(S):
    """The mean of the per-model means, each model's pictures resampled on their own."""
    if len(S) == 1:
        return S[0]
    methods = [m for m in S[0]["methods"] if all(m in s["methods"] for s in S)]
    by = {}
    for s in SOURCES:
        if not all(s in x["by"] for x in S):
            continue
        by[s] = dict(n=sum(x["by"][s]["n"] for x in S), pictures=sum(x["by"][s]["pictures"] for x in S),
                     agree={m: (float(np.mean([x["by"][s]["agree"][m][0] for x in S])),
                                np.mean([x["by"][s]["agree"][m][1] for x in S], 0)) for m in methods})
    return dict(methods=methods, by=by)


def _ci(d):
    return tuple(float(q) for q in np.quantile(d, [0.025, 0.975]))


def method_source_matrix(results="results", out="figures", model=None):
    """MAIN. V-Write's lead over published attribution methods is not an artefact of pooling: on each of the five SimonSays sources separately, its patch ranking agrees with the measured effect of covering the picture at least as well as the best published baseline of that source, and no published method is above it anywhere.

    Left, a matrix of methods (rows) by SimonSays source (columns: all
    records, then Open Images, nuScenes, Interfaces (VINS-120K), Generated
    scenes and the WMDP carrier). Each cell prints, and is shaded darker by,
    the method's agreement with measured
    effects on that source: the mean over records of the Spearman correlation,
    across sixteen contiguous groups of image patches, between the method's
    summed attribution per group and the fall in the answer score when that
    group is replaced by zeros (the replacement V-Write does not integrate
    from, so the neutral test). V-Write's row is outlined, and the best
    published method of each column is outlined in ochre. Methods: V-Write (8 steps),
    V-Write at 32 steps, and the published baselines attribution patching,
    integrated gradients from zero, gradient x activation, activation
    patching one patch at a time, RISE, KernelSHAP and Grad-CAM, with a
    random ranking as the floor; all rank the image patches at one block
    near half depth (block 32 of 64 on Qwen3.5-27B). Below each column, V-Write minus the
    best published baseline of that source, with a 95 per cent interval from
    2,000 resamples of pictures within that source, paired. One model per
    file; the averaged file is the mean of the per-model means when two or
    more models have the run.

    Every method, V-Write included, is scored on the same subsample of 192
    records (167 pictures) of Qwen3.5-27B, and every comparison is on that
    shared subsample: integrated gradients, RISE and KernelSHAP each cost
    hundreds of forward passes per record, so the baselines run covers a
    subsample rather than the whole set, and a record enters only when every
    method has a value on it. Qwen3-VL-8B-Instruct has no baselines run yet,
    so it has no file rather than a comparison against a partial set.

    On Qwen3.5-27B: all records, V-Write 0.40
    against the best baseline, attribution patching, at 0.30 (KernelSHAP
    0.30), a lead of 0.10 [0.05, 0.14]. Per source, V-Write against the best
    baseline there, lead with interval: Open Images (n = 50) 0.35 against
    RISE 0.26, +0.09 [0.01, 0.17]; nuScenes (n = 26) 0.39 against activation
    patching 0.28, +0.11 [-0.04, 0.23]; Interfaces (VINS-120K) (n = 46) 0.37
    against attribution patching 0.30, +0.06 [-0.02, 0.14]; Generated scenes
    (n = 42) 0.57 against activation patching 0.55, +0.01 [-0.09, 0.12]; WMDP
    carrier (n = 28) 0.31 against attribution patching 0.26, +0.05 [-0.05,
    0.16]. The lead is positive in all five sources but its interval clears
    zero only on the pooled set and on Open Images; on Generated scenes
    activation patching is level with it. No published baseline is above
    V-Write in any source. V-Write at 32 steps is above V-Write on the pooled
    set (0.43) and in four of five sources (Generated scenes 0.61, nuScenes
    0.44), below it on the WMDP carrier (0.29), so eight steps leaves a small
    amount on the table. The function prints every one of these numbers, with
    intervals, on each build.

    Limits: sources have 26 to 50 records each, so per-source intervals are
    wide and four of the five per-source leads are not resolved; one block per model;
    sixteen groups is a coarse ground truth; the WMDP carrier records are
    letter-answer items whose question text is never shown.
    """
    if model is None and not P.AVERAGE:
        return None
    models = [model] if model else list(B.MODELS)
    got = [(m, D) for m in models for D in [B._load(results, m)] if D]
    if not got or (P.AVERAGE and len(got) < 2):
        return None
    S = _pool([_stats(D) for _, D in got])
    methods, by = S["methods"], S["by"]
    cols = [s for s in SOURCES if s in by]
    pub = [m for m in B.PUBLISHED if m in methods]
    if not pub or "all" not in by:
        return None
    # ours first, then the published methods by their pooled agreement, then the floor
    rows_m = [m for m in ("v_write", "ig_mean32") if m in methods]
    rows_m += sorted(pub, key=lambda m: -by["all"]["agree"][m][0])
    rows_m += [m for m in methods if m not in rows_m]

    # ---------------- numbers the caption states, printed every build
    tag = model or "averaged over " + ", ".join(m for m, _ in got)
    print(f"  method_source_matrix [{tag}]")
    best, lead = {}, {}
    for s in cols:
        a = by[s]["agree"]
        best[s] = max(pub, key=lambda m: a[m][0])
        d = a["v_write"][1] - a[best[s]][1]
        lead[s] = (a["v_write"][0] - a[best[s]][0], _ci(d))
        above = [m for m in pub if a[m][0] > a["v_write"][0]]
        print(f"    {s:<10} n={by[s]['n']:<3} pics={by[s]['pictures']:<3} V-Write {a['v_write'][0]:.3f}"
              f"  best baseline {best[s]} {a[best[s]][0]:.3f}  lead {lead[s][0]:+.3f}"
              f" [{lead[s][1][0]:+.3f}, {lead[s][1][1]:+.3f}]"
              + (f"  32 steps {a['ig_mean32'][0]:.3f}" if "ig_mean32" in a else "")
              + (f"  PUBLISHED ABOVE V-WRITE: {', '.join(above)}" if above else ""))
        print("      " + "  ".join(f"{m}={a[m][0]:.2f}" for m in rows_m))

    # ---------------- drawing, laid out by hand in inches so the cells and
    # the wrapped headers can be measured against each other
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    st.use()
    from vlens.figures.fx_layers import _width_in
    from matplotlib.patches import Patch
    fs_head, fs_num, fs_row = 14.5, 13.0, 16.5
    nR, nC = len(rows_m), len(cols)
    W = st.MAX_WIDTH
    # the row names set the left column, measured; the cells take the rest of
    # the page, and the lead per source sits under its own column
    lab_w = max(_width_in(P.NICE.get(m, m), fs_row) for m in rows_m) + 0.14
    row_h, head_h = 0.37, 0.86
    col_w = (W - lab_w - 0.04) / nC
    strip_h, strip_gap = 1.45, 0.16
    H = head_h + nR * row_h + strip_gap + strip_h
    fig = st.plt.figure(figsize=(W, H), layout="none")
    mx0, my0 = lab_w, strip_h + strip_gap
    ax = fig.add_axes([mx0 / W, my0 / H, nC * col_w / W, nR * row_h / H])
    ax.set_xlim(-0.5, nC - 0.5)
    ax.set_ylim(nR - 0.5, -0.5)
    ax.axis("off")
    # each cell the agreement, printed and shaded (darker is higher, from zero
    # to VMAX, fixed so two models' files compare); V-Write's row outlined in
    # ink, the best published method of each column outlined in ochre
    warm = LinearSegmentedColormap.from_list("srcm", st.WARM[:5])
    for i, s in enumerate(cols):
        a = by[s]["agree"]
        for j, m in enumerate(rows_m):
            v = a[m][0]
            ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, edgecolor="white", lw=1.5,
                                   facecolor=warm(float(np.clip(v / VMAX, 0, 1)))))
            ax.text(i, j, f"{v:.2f}".replace("-", "\u2212").replace("0.", ".", 1), ha="center",
                    va="center", fontsize=fs_num, color=st.INK, zorder=4)
            if m == best[s]:
                ax.add_patch(Rectangle((i - 0.46, j - 0.44), 0.92, 0.88, facecolor="none",
                                       edgecolor=st.BASE, lw=2.2, zorder=5))
    j_vw = rows_m.index("v_write")
    ax.add_patch(Rectangle((-0.5, j_vw - 0.5), nC, 1, facecolor="none", edgecolor=st.INK, lw=1.8,
                           zorder=6, clip_on=False))
    for yline in (len([m for m in rows_m if m in ("v_write", "ig_mean32")]) - 0.5,
                  nR - 1.5 if rows_m[-1] == "random" else None):
        if yline is not None:
            ax.plot([-0.5, nC - 0.5], [yline, yline], color=st.INK, lw=0.9, zorder=5, clip_on=False)
    # row names, and a stroke of each method's colour is not needed: the fill says it
    for j, m in enumerate(rows_m):
        fig.text((mx0 - 0.1) / W, (my0 + (nR - 0.5 - j) * row_h) / H, P.NICE.get(m, m),
                 ha="right", va="center", fontsize=fs_row, color=st.INK)
    # column heads wrapped to the measured cell width; the records per source
    # are in the caption
    top = my0 + nR * row_h + 0.06
    for i, s in enumerate(cols):
        name = "all records" if s == "all" else P.SOURCE.get(s, s).replace(" (VINS-120K)", "")
        lines = fit_lines(name, col_w - 0.1, fs_head, 2)
        fig.text((mx0 + (i + 0.5) * col_w) / W, top / H, "\n".join(lines), ha="center", va="bottom",
                 fontsize=fs_head, color=st.INK, linespacing=1.1)

    # below: V-Write's lead over the best published method, under each source's column
    a2 = fig.add_axes([mx0 / W, 0.0, nC * col_w / W, strip_h / H])
    for k, s in enumerate(cols):
        p, (lo, hi) = lead[s]
        a2.plot([k, k], [lo, hi], color=st.LENS, lw=2.2, solid_capstyle="butt", zorder=3)
        a2.plot([k], [p], marker="o", ms=8.5, color=st.LENS, ls="none", zorder=4)
    a2.axhline(0, color=st.AXIS, lw=1.0, zorder=1)
    a2.set_xlim(-0.5, nC - 0.5)
    a2.set_xticks([])
    lo_all = min(lead[s][1][0] for s in cols)
    hi_all = max(lead[s][1][1] for s in cols)
    a2.set_ylim(min(lo_all, 0) - 0.02, hi_all + 0.02)
    a2.set_yticks([t for t in (-0.1, 0.0, 0.1, 0.2, 0.3) if a2.get_ylim()[0] <= t <= a2.get_ylim()[1]])
    a2.set_yticklabels([f"{t:g}".replace("-", "\u2212") for t in a2.get_yticks()], fontsize=fs_num)
    a2.xaxis.grid(False)
    for sp in ("top", "right", "bottom"):
        a2.spines[sp].set_visible(False)
    a2.annotate("V-Write minus\nbest baseline", xy=(0, 0.5), xycoords="axes fraction",
                xytext=(-44, 0), textcoords="offset points", ha="right", va="center",
                fontsize=fs_row, color=st.INK, linespacing=1.1)
    # the key in its own band above the heads
    st.key_band(fig, [Patch(facecolor="none", edgecolor=st.BASE, lw=2.2)],
                ["best baseline there"], fontsize=fs_head, handlelength=1.2)
    return st.save(fig, P._at(out, "method_source_matrix"), check=True, tight=True)
