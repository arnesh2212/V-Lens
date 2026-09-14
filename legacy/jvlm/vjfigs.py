# SPDX-License-Identifier: Apache-2.0
"""Layerwise figures for the three-lens comparison.

These live in their own module rather than at the end of :mod:`jvlm.figures`
because several people are appending to that file at once and a whole-file
write from any one of them takes the others' work with it.  They are re-exported
from :mod:`jvlm.figures` for convenience, and the figure driver imports this
module directly so a lost re-export cannot silently drop four figures.

Everything a reader has to compare sits on a common depth axis; every series is
named where it ends rather than in a box off to one side; and every panel's
second line says what the panel is for and what the numbers in it came to.  Each
function raises if its input is missing, so the driver skips it and renders it
later when the run lands.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from jvlm import style
from jvlm.style import AQUA, BLUE, C_WORD, INK, INK2, MUTED, ORANGE, SURFACE, VIOLET

import matplotlib.pyplot as plt


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)
    return p


LENS_CANON = {
    "logit lens": "logit lens", "logit": "logit lens", "say": "logit lens",
    "J-lens": "J-lens", "jlens": "J-lens", "j-lens": "J-lens",
    "ATLAS": "VJ-Lens", "Atlas": "VJ-Lens", "atlas": "VJ-Lens",
    "V-Lens": "VJ-Lens", "VJ-Lens": "VJ-Lens", "vjlens": "VJ-Lens",
    "do": "VJ-Lens",
    "ig": "VJ-Lens (path)", "IG": "VJ-Lens (path)",
    "VJ-Lens (IG)": "VJ-Lens (path)", "VJ-Lens (path)": "VJ-Lens (path)",
    "path": "VJ-Lens (path)", "ATLAS-IG": "VJ-Lens (path)",
    "random": "random order", "random order": "random order",
}
LENS_COL = {"logit lens": C_WORD, "J-lens": VIOLET, "VJ-Lens": BLUE,
            "VJ-Lens (path)": AQUA, "random order": MUTED}
LENS_ORDER = ["logit lens", "J-lens", "VJ-Lens", "VJ-Lens (path)",
              "random order"]
# Three tints of one blue are not separable once the error bars overlap --
# the first render of the region sweep was unreadable in its own crossing
# region.  Hue and marker both change, so a reader can follow one depth
# without tracing it back to the legend.
DEPTH_SHADES = ["#9ec3ec", "#12447f", VIOLET, ORANGE]  # shallow -> deep
DEPTH_MARKS = ["o", "s", "^", "D"]


def _canon(df: pd.DataFrame, col: str) -> pd.DataFrame:
    d = df.copy()
    d["lens"] = d[col].map(lambda s: LENS_CANON.get(str(s), str(s)))
    return d


def _lenses(d: pd.DataFrame) -> list:
    seen = list(d.lens.unique())
    return ([l for l in LENS_ORDER if l in seen]
            + [l for l in seen if l not in LENS_ORDER])


def _boot_mean(v, n=2000, seed=0):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return (float(v.mean()), float(np.percentile(b, 2.5)),
            float(np.percentile(b, 97.5)))


def _boot_spearman(x, y, n=600, seed=0):
    from scipy import stats as sps
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 8:
        return np.nan, np.nan, np.nan
    r = float(sps.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    b = np.array([sps.spearmanr(x[i], y[i]).statistic for i in idx])
    b = b[~np.isnan(b)]
    if len(b) < 20:
        return r, np.nan, np.nan
    return r, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _stagger(vals, gap):
    """Nudge label positions apart, keeping their order, minimum spacing ``gap``."""
    out = np.asarray(vals, dtype=float).copy()
    order = np.argsort(out)
    for j in range(1, len(order)):
        a_, b_ = order[j - 1], order[j]
        if out[b_] - out[a_] < gap:
            out[b_] = out[a_] + gap
    return out


def _end_label(ax, xs, ys, texts, gap_frac=0.085, dx=0.035, fs_scale=0.9):
    """Direct labels off the right end of each series, nudged clear of each other.

    A label that had to move away from its line keeps a hairline leader back to
    it, so the reader never has to guess which name belongs to which curve.
    """
    y0, y1 = ax.get_ylim()
    x0, x1 = ax.get_xlim()
    ypos = _stagger(ys, gap_frac * (y1 - y0))
    fs = plt.rcParams["axes.labelsize"] * fs_scale
    pad = dx * (x1 - x0)
    for x, y, yl, t in zip(xs, ys, ypos, texts):
        if abs(yl - y) > 1e-9 * max(1.0, abs(y)):
            ax.plot([x, x + pad * 0.82], [y, yl], color=MUTED, lw=0.8,
                    zorder=1, clip_on=False)
        ax.text(x + pad, yl, t, va="center", ha="left", color=INK2,
                fontsize=fs, clip_on=False, zorder=5)
    ax.set_ylim(y0, y1)
    ax.set_xlim(x0, x1)


def _subw(inches: float, scale: float = 6.05) -> int:
    """Characters that fit on one line above a panel this wide.

    ``style.title`` wraps at a character count, which is only the right count
    for one panel width.  Every overflowing subtitle in the first draft of these
    figures was a panel narrower than the default assumed.
    """
    return max(24, int(inches * 72 / scale))


def _tag_ends(ax, ends, gap_frac=0.075, dx=9, fs_scale=0.88):
    """Name each series at its right end, order preserved, on any x scale.

    Offsets are in points rather than data units so this works on the log axis
    of the region sweep as well as the linear ones.  The stagger keeps the
    labels in the same vertical order as the line ends, so no leaders are
    needed to disambiguate them.
    """
    if not ends:
        return
    y0, y1 = ax.get_ylim()
    ypos = _stagger([e[1] for e in ends], gap_frac * (y1 - y0))
    for (x, _y, t), yl in zip(ends, ypos):
        ax.annotate(t, xy=(x, yl), xytext=(dx, 0), textcoords="offset points",
                    va="center", ha="left", color=INK2,
                    fontsize=plt.rcParams["axes.labelsize"] * fs_scale,
                    annotation_clip=False)
    ax.set_ylim(y0, y1)


def _headroom(ax, right=0.30):
    """Leave room on the right of the axes for the end labels."""
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + right * (x1 - x0))


# ------------------------------------------------------------- F51 the strip

def f51_strip(point: pd.DataFrame, out: str, dele: pd.DataFrame | None = None,
              name="f51_strip.png"):
    """Three lenses, four measures, across the depth of the network.

    ``point`` is ``compare/{tag}_point.parquet``; ``dele`` the matching
    ``_del.parquet`` if the run has finished writing it.  Panels whose column is
    absent are simply not drawn, so this renders against a run in progress.
    """
    if point is None or not len(point):
        raise ValueError("no comparison results")
    d = _canon(point, "lens")
    if "depth" not in d.columns:
        d["depth"] = d.layer / d.layer.max()
    lenses = _lenses(d)
    chance = float(d.chance.mean()) if "chance" in d else None
    nimg = int(d.i.nunique())

    panels = []
    if "hit" in d.columns:
        panels.append(("hit", "Pointing game",
                       "Does the map's strongest patch land inside the "
                       "object's box?", "peak inside the box", False, chance))
    if "mass_in" in d.columns:
        panels.append(("mass_in", "Energy in the box",
                       "Share of the map's positive mass that lands on the "
                       "object.", "share of map mass inside", False, chance))
    if "drop" in d.columns and d["drop"].notna().any():
        panels.append(("__spear", "Predicts the real damage",
                       "Rank correlation with what covering the object "
                       "actually costs.", "Spearman  $\\rho$", False, 0.0))
    dd = None
    if dele is not None and len(dele):
        dd = _canon(dele, "lens" if "lens" in dele.columns else "order")
        panels.append(("__auc", "Deletion area",
                       "Covering patches in the map's own order. Lower is "
                       "better.", "area under the deletion curve", True, None))
    if not panels:
        raise ValueError("nothing measurable in the comparison frame")

    pw = 4.75
    fig, axes = plt.subplots(1, len(panels), figsize=(pw * len(panels), 5.9))
    axes = np.atleast_1d(axes)
    deps = sorted({round(float(v) * 100, 1) for v in d.depth.unique()})
    for ax, (key, head, sub, ylab, lower, ref) in zip(axes, panels):
        ends, best = [], []
        for lens in lenses:
            s_ = dd[dd.lens == lens] if key == "__auc" else d[d.lens == lens]
            if not len(s_):
                continue
            xs, ms, los, his = [], [], [], []
            for l in sorted(s_.layer.unique()):
                q = s_[s_.layer == l]
                if key in ("hit", "mass_in"):
                    mv, lo, hi = _boot_mean(q[key].values, seed=l)
                elif key == "__spear":
                    q = q.dropna(subset=["drop", "pred"])
                    mv, lo, hi = _boot_spearman(q.pred.values, q["drop"].values,
                                                seed=l)
                else:
                    cur = q.groupby("frac")["rel"].mean().sort_index()
                    if len(cur) < 3:
                        continue
                    mv = float(np.trapz(cur.values, cur.index)
                               / max(cur.index.max(), 1e-9))
                    per = []
                    for i_ in q.i.unique():
                        c_ = (q[q.i == i_].groupby("frac")["rel"].mean()
                              .sort_index())
                        if len(c_) >= 3:
                            per.append(float(np.trapz(c_.values, c_.index)
                                             / max(c_.index.max(), 1e-9)))
                    _, lo, hi = _boot_mean(per, seed=l)
                if mv is None or np.isnan(mv):
                    continue
                hit = d[d.layer == l]
                dep = (float(hit.depth.iloc[0]) if len(hit)
                       else l / max(d.layer.max(), 1))
                xs.append(dep * 100); ms.append(mv)
                los.append(lo); his.append(hi)
            if not xs:
                continue
            c = LENS_COL.get(lens, MUTED)
            xs, ms = np.array(xs), np.array(ms)
            los = np.array([m if (v is None or np.isnan(v)) else v
                            for m, v in zip(ms, los)])
            his = np.array([m if (v is None or np.isnan(v)) else v
                            for m, v in zip(ms, his)])
            ax.fill_between(xs, los, his, color=c, alpha=0.10, lw=0, zorder=2)
            ax.plot(xs, ms, color=c, marker="o", ms=7, zorder=3,
                    lw=3.0 if lens.startswith("VJ-Lens") else 2.0)
            ends.append((xs[-1], ms[-1], lens))
            best.append((min(ms) if lower else max(ms), lens))
        if ref is not None:
            ax.axhline(ref, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(deps)
        ax.set_xticklabels([f"{v:.0f}" for v in deps])
        ax.set_xlabel("depth through the language model  (%)")
        ax.set_ylabel(ylab)
        if best:
            w = min(best) if lower else max(best)
            sub = sub + f"  Best over depth: {w[1]}, {w[0]:.2f}."
        style.title(ax, head, sub, width=_subw(pw))
        _headroom(ax, 0.40 if key == "__auc" else 0.36)
        if ref is not None:
            lab = ("what a blind map gets" if key in ("hit", "mass_in")
                   else "no relation")
            ya, yb = ax.get_ylim()
            low = (ref - ya) < 0.16 * (yb - ya)
            ax.annotate(lab, xy=(ax.get_xlim()[0], ref),
                        xytext=(3, 5 if low else -5),
                        textcoords="offset points", ha="left",
                        va="bottom" if low else "top", color=MUTED,
                        fontsize=plt.rcParams["axes.labelsize"] * 0.8)
        if ends:
            _end_label(ax, [e[0] for e in ends], [e[1] for e in ends],
                       [e[2] for e in ends])
    _n = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    fig.suptitle(f"The same {nimg} photographs, read "
                 f"{_n.get(len(lenses), len(lenses))} ways, at "
                 f"{_n.get(len(deps), len(deps))} depths",
                 x=0.5, ha="center", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return _save(fig, out, name)


# ------------------------------------------------------------ F52 the ribbon

def f52_ribbon(df: pd.DataFrame, out: str, stats: dict | None = None,
               item: int | None = None, name="f52_ribbon.png"):
    """One photograph, every layer: how far the object's word separates.

    ``df`` is ``ribbon/{tag}_ribbon.parquet`` (dense, per layer).  If that run
    has not landed, a ``compare/{tag}_point.parquet`` frame is accepted and the
    same idea is drawn pooled over photographs instead.
    """
    if df is None or not len(df):
        raise ValueError("no ribbon results")
    dense = {"sep", "lo", "hi"} <= set(df.columns)
    d = _canon(df, "lens")
    if not dense:
        return _f52_pooled(d, out, name)

    if item is None:
        peak = d[d.lens == "VJ-Lens"].groupby("i").sep.max().sort_values()
        if not len(peak):
            peak = d.groupby("i").sep.max().sort_values()
        item = int(peak.index[len(peak) // 2])          # the median photograph
    s = d[d.i == item]
    concept = str(s.concept.iloc[0])
    lenses = _lenses(s)
    layers = sorted(s.layer.unique())
    nl = (stats or {}).get("n_layers", max(layers) + 2)

    thumb = None
    for c in (stats or {}).get("cards", []):
        if int(c.get("i", -1)) == item and c.get("thumb"):
            p = os.path.join((stats or {}).get("_dir", ""), c["thumb"])
            if os.path.exists(p):
                thumb = p
    n_in = int(s.n_in.iloc[0]) if "n_in" in s else None

    show = layers[::2] if len(layers) > 14 else layers
    # The photograph used to sit in gs[0, 1] with gs[1, 1] left empty, which
    # is where the band of dead white down the right third came from.  The
    # ribbon is the data-dense panel and now takes the whole width; the
    # photograph sits beside the word strip, which is only three rows tall,
    # so nothing is left over.
    fig = plt.figure(figsize=(15.0, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=(2.35, 1.5),
                          width_ratios=(6.4, 1.15), hspace=0.38, wspace=0.05)
    ax = fig.add_subplot(gs[0, :])
    axs = fig.add_subplot(gs[1, 0], sharex=ax)
    ax_im = fig.add_subplot(gs[1, 1]) if thumb is not None else None

    ends = []
    for lens in lenses:
        q = s[s.lens == lens].sort_values("layer")
        if not len(q):
            continue
        c = LENS_COL.get(lens, MUTED)
        ax.fill_between(q.layer, q.lo, q.hi, color=c, alpha=0.16, lw=0)
        ax.plot(q.layer, q.sep, color=c,
                lw=2.8 if lens.startswith("VJ-Lens") else 2.0,
                marker="o" if len(q) < len(layers) else None, ms=6, zorder=3)
        j = int(q.sep.values.argmax())
        ax.plot([q.layer.values[j]], [q.sep.values[j]], "o", color=c, ms=9,
                zorder=4)
        ax.annotate(f"{q.sep.values[j]:.1f}",
                    xy=(q.layer.values[j], q.sep.values[j]),
                    xytext=(0, 11), textcoords="offset points", ha="center",
                    color=INK2,
                    fontsize=plt.rcParams["axes.labelsize"] * 0.82)
        ends.append((q.layer.values[-1], q.sep.values[-1], lens))
    ax.axhline(0, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.annotate("zero: reads the same inside the box and outside it",
                xy=(layers[0], 0), xytext=(2, -5), textcoords="offset points",
                ha="left", va="top", color=MUTED,
                fontsize=plt.rcParams["axes.labelsize"] * 0.8)
    ax.tick_params(labelbottom=False, length=0)
    ax.set_ylabel(f"“{concept}” inside the box minus outside\n"
                  "(sd of the object vocabulary)")
    peak_by = {l: float(s[s.lens == l].sep.max()) for l in lenses}
    won = max(peak_by, key=peak_by.get)
    style.title(
        ax, f"One photograph of a {concept}, read at every layer",
        "The band is a 95% bootstrap over the patches. Look at how far each "
        "line rises above zero: that is the lens telling the object's own "
        f"patches apart from the rest of the frame. Peak: {won}, "
        f"{peak_by[won]:.1f} sd." +
        (f"  {n_in} of {(stats or {}).get('n_patches')} patches lie inside the "
         "box." if (n_in and (stats or {}).get("n_patches")) else ""),
        width=104)
    _headroom(ax, 0.13)
    if ends:
        _end_label(ax, [e[0] for e in ends], [e[1] for e in ends],
                   [e[2] for e in ends])

    # the discrete strip: the word each lens reads inside the box, per layer
    rows = list(lenses)
    vmax = max(1e-6, float(s.sep.max()))
    for r, lens in enumerate(rows):
        q = s[s.lens == lens].set_index("layer")
        c = LENS_COL.get(lens, MUTED)
        for l in show:
            if l not in q.index:
                continue
            w = str(q.loc[l, "word"])
            v = float(q.loc[l, "sep"])
            hit = (w == concept)
            al = float(np.clip(v / vmax, 0, 1)) * 0.42
            axs.add_patch(plt.Rectangle((l - 1, len(rows) - r - 1), 2, 0.92,
                                        facecolor=c, alpha=al, lw=0))
            if hit:
                axs.add_patch(plt.Rectangle((l - 1, len(rows) - r - 1), 2, 0.92,
                                            facecolor="none", edgecolor=c,
                                            lw=1.6))
            fs = float(np.clip(163.0 / max(len(w), 1), 9.5, 13.5))
            # every cell's word is a measurement and has to be readable; the
            # fill already says how strong it is, so the type does not also
            # need to fade out
            axs.text(l, len(rows) - r - 1 + 0.46, w, ha="center", va="center",
                     color=INK if hit else INK2, fontsize=fs)
    axs.set_ylim(0, len(rows))
    axs.set_yticks([len(rows) - r - 0.54 for r in range(len(rows))])
    axs.set_yticklabels(rows)
    axs.set_xticks(show)
    axs.set_xlabel(f"layer of the language model  (of {nl})")
    axs.grid(False)
    for sp in ("left", "bottom"):
        axs.spines[sp].set_visible(False)
    axs.tick_params(length=0)
    style.title(axs, "and the word it actually reads there",
                "The strongest object word inside the box at that layer. "
                f"An outlined cell is the right one, “{concept}”; "
                "the fill is how far that word separates.", width=104)

    if ax_im is not None:
        try:
            ax_im.imshow(plt.imread(thumb))
        except Exception:
            pass
        ax_im.set_xticks([]); ax_im.set_yticks([]); ax_im.grid(False)
        for sp in ax_im.spines.values():
            sp.set_visible(False)
        ax_im.set_title("the photograph, with the box",
                        fontsize=plt.rcParams["axes.labelsize"] * 0.85,
                        color=INK2, pad=6, loc="left")
    return _save(fig, out, name)


def _f52_pooled(d: pd.DataFrame, out: str, name: str):
    """Fallback ribbon: map mass in the box relative to chance, over depth."""
    if "mass_in" not in d.columns:
        raise ValueError("frame has neither a separation nor a box mass")
    d = d.copy()
    d["enrich"] = d.mass_in / d.chance.clip(lower=1e-6)
    if "depth" not in d.columns:
        d["depth"] = d.layer / d.layer.max()
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ends = []
    for lens in _lenses(d):
        s = d[d.lens == lens]
        xs, ms, lo_, hi_ = [], [], [], []
        for l in sorted(s.layer.unique()):
            q = s[s.layer == l]
            mv, a_, b_ = _boot_mean(q.enrich.values, seed=l)
            xs.append(float(q.depth.iloc[0]) * 100)
            ms.append(mv); lo_.append(a_); hi_.append(b_)
        c = LENS_COL.get(lens, MUTED)
        ax.fill_between(xs, lo_, hi_, color=c, alpha=0.16, lw=0)
        ax.plot(xs, ms, color=c, marker="o", ms=6.5,
                lw=2.8 if lens.startswith("VJ-Lens") else 2.0)
        ends.append((xs[-1], ms[-1], lens))
    ax.axhline(1.0, color=MUTED, lw=1.1, ls=(0, (4, 3)))
    ax.set_xlabel("depth through the language model  (%)")
    ax.set_ylabel("map mass on the object, over chance")
    style.title(ax, "How much of each map lands on the object",
                "One at the dashed line is a map that ignores the object "
                "entirely. Bands are 95% bootstraps over photographs.")
    _headroom(ax, 0.22)
    _end_label(ax, [e[0] for e in ends], [e[1] for e in ends],
               [e[2] for e in ends])
    fig.tight_layout()
    return _save(fig, out, name)


# ----------------------------------------------------- F53 deletion curves

def f53_deletion3(df: pd.DataFrame, out: str, name="f53_deletion3.png"):
    """Deletion curves for every lens, one panel per depth, AUC at the end.

    Accepts either the three-lens frame from the comparison run
    (``compare/{tag}_del.parquet``) or the deletion run's own frame
    (``deletion/{tag}_raw.parquet``), which carries a random order as its floor.
    """
    if df is None or not len(df):
        raise ValueError("no deletion results")
    col = "lens" if "lens" in df.columns else "order"
    d = _canon(df, col)
    if "frac" not in d.columns or "rel" not in d.columns:
        raise ValueError("deletion frame needs frac and rel")
    layers = sorted(d.layer.unique())[:4]
    lenses = _lenses(d)
    nl = int(d.layer.max()) + 4

    curves = {}
    for l in layers:
        for lens in lenses:
            q = d[(d.layer == l) & (d.lens == lens)]
            if not len(q):
                continue
            cur = q.groupby("frac")["rel"].mean().sort_index()
            if len(cur) >= 3:
                curves[(l, lens)] = (
                    cur, float(np.trapz(cur.values, cur.index)
                               / max(cur.index.max(), 1e-9)))
    if not curves:
        raise ValueError("no usable deletion curves")
    allv = np.concatenate([c.values for c, _ in curves.values()])
    lo, hi = float(allv.min()), float(allv.max())
    pad = 0.09 * (hi - lo + 1e-9)
    xmax = max(float(c.index.max()) for c, _ in curves.values()) * 100

    pw = 5.0
    fig, axes = plt.subplots(1, len(layers), figsize=(pw * len(layers), 5.5),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, l in zip(axes, layers):
        ends, aucs = [], {}
        for lens in lenses:
            if (l, lens) not in curves:
                continue
            cur, auc = curves[(l, lens)]
            aucs[lens] = auc
            c = LENS_COL.get(lens, MUTED)
            ax.plot(cur.index * 100, cur.values, color=c, marker="o", ms=5.5,
                    lw=3.0 if lens.startswith("VJ-Lens") else 2.0,
                    ls=(0, (5, 2)) if lens == "random order" else "-",
                    zorder=2 if lens == "random order" else 3)
            ends.append((cur.index[-1] * 100, cur.values[-1], lens, auc))
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.03 * xmax, xmax * 1.52)
        ax.set_xlabel("patches covered  (% of the frame)")
        # "Fastest: VJ-Lens, area 0.91" stood here as a second line under
        # every panel title.  It repeats what the end-of-line labels already
        # print beside their own curves, and the sentence that explains why the
        # lowest curve wins is in this figure's caption now.
        style.title(ax, f"layer {l} of {nl}")
        if ends:
            _end_label(ax, [e[0] for e in ends], [e[1] for e in ends],
                       [f"{e[2]}   area {e[3]:.2f}" for e in ends],
                       gap_frac=0.075, dx=0.025, fs_scale=0.86)
    axes[0].set_ylabel("object's score left, relative\nto the untouched image")
    fig.suptitle("Deleting in the map's own order", x=0.5, y=0.985,
                 ha="center", va="top", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return _save(fig, out, name)


# ---------------------------------------------------- F54 the region sweep

def _paired_sweep(d, layer, sizes, n=800, seed=0):
    """Bootstrap the whole size sweep on one resample of the photographs.

    Resampling each region size on its own makes every interval as wide as the
    spread between photographs, and the sweep then looks like eight unrelated
    numbers with overlapping bars -- which is exactly the "nothing is
    conclusive" reading.  The same photographs are measured at every size, so
    the honest interval is the paired one: draw a set of photographs once and
    score every size on that same set.  The last returned value is the
    interval on (best size minus the whole-frame endpoint), which is the
    comparison the panel's claim actually rests on.
    """
    from scipy.stats import spearmanr

    q = d[d.layer == layer]
    ids = np.array(sorted(q.i.unique()))
    tab = {int(v): q[q["size"] == v].set_index("i").reindex(ids)
           for v in sizes}
    tab = {v: t for v, t in tab.items() if t["pred"].notna().sum() >= 8}
    keep = sorted(tab)
    if not keep:
        return [], [], [[], []], None

    def rho(v, sel):
        t = tab[v].iloc[sel]
        a, b = t["pred"].values, t["drop"].values
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < 8 or np.ptp(a[m]) == 0 or np.ptp(b[m]) == 0:
            return np.nan
        return float(spearmanr(a[m], b[m]).statistic)

    full = np.arange(len(ids))
    ms = [rho(v, full) for v in keep]
    rng = np.random.default_rng(seed)
    draws = np.full((n, len(keep)), np.nan)
    for b in range(n):
        sel = rng.integers(0, len(ids), len(ids))
        for j, v in enumerate(keep):
            draws[b, j] = rho(v, sel)
    lo = np.nanpercentile(draws, 2.5, axis=0)
    hi = np.nanpercentile(draws, 97.5, axis=0)
    er = [[m - l for m, l in zip(ms, lo)], [h - m for m, h in zip(ms, hi)]]

    gap = None
    k = int(np.nanargmax(ms))
    if keep[-1] != keep[k]:
        dif = draws[:, k] - draws[:, -1]
        gap = (ms[k] - ms[-1], float(np.nanpercentile(dif, 2.5)),
               float(np.nanpercentile(dif, 97.5)), keep[k], keep[-1])
    return keep, ms, er, gap


def f54_regionsweep(df: pd.DataFrame, out: str, stats: dict | None = None,
                    name="f54_regionsweep.png"):
    """The one-parameter family: both prior lenses are its endpoints.

    ``df`` is ``region/{tag}_raw.parquet``.  The x axis runs from a region of
    one patch, which is the logit lens's unit, to the whole frame, which is the
    J-lens's; everything between them is the coordinate this paper adds.
    """
    if df is None or not len(df) or "size" not in df.columns:
        raise ValueError("no region-sweep results")
    d = df.dropna(subset=["pred", "drop"]).copy()
    P = (stats or {}).get("n_patches")
    if not P:
        P = int(round(float(d["size"].iloc[0] / max(d.frac.iloc[0], 1e-9))))
    sizes = sorted(int(v) for v in d["size"].unique())
    layers = sorted(int(v) for v in d.layer.unique())
    swept = max(sizes) >= P
    nimg = int(d.i.nunique())

    def shade(j):
        return DEPTH_SHADES[j % len(DEPTH_SHADES)]

    def mark(j):
        return DEPTH_MARKS[j % len(DEPTH_MARKS)]

    pw = 6.9
    fig, axes = plt.subplots(1, 2, figsize=(pw * 2, 6.0))

    ax = axes[0]
    peaks, tags, gaps = [], [], []
    for j, l in enumerate(layers):
        c = shade(j)
        xs, ms, er, gap = _paired_sweep(d, l, sizes, seed=100 * l)
        if gap is not None:
            gaps.append((l, gap))
        if not xs:
            continue
        ax.errorbar(xs, ms, yerr=er, color=c, marker=mark(j), ms=7.5, lw=2.4,
                    elinewidth=1.1, capsize=3.5, capthick=1.1, zorder=3)
        k = int(np.argmax(ms))
        peaks.append((xs[k], ms[k], l))
        ax.plot([xs[k]], [ms[k]], mark(j), color=c, ms=14, zorder=4)
        tags.append((xs[-1], ms[-1], f"layer {l}"))
    ax.axhline(0, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1)
    for x_ in (1, P):
        ax.axvline(x_, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=1)
    ax.set_xscale("log")
    ticks = sorted(set(sizes) | {1, P})
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1\n= logit lens" if v == 1 else
                        f"{P}\n= J-lens" if v == P else str(v)
                        for v in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.74, P * 1.42)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.28 * (y1 - y0))
    _tag_ends(ax, tags)
    y0, y1 = ax.get_ylim()
    if not swept:
        ax.annotate("the far end is not swept yet",
                    xy=(P, y0 + 0.5 * (y1 - y0)), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center",
                    color=MUTED, rotation=90,
                    fontsize=plt.rcParams["axes.labelsize"] * 0.8)
    ax.set_xlabel(f"patches in the region  (of {P})")
    ax.set_ylabel("reading predicts the real damage  (Spearman $\\rho$)")
    # Report the depth where the middle of the family actually beats the
    # J-lens endpoint by more than the paired interval, not the depth with the
    # largest raw rho.  At layer 19 the best size leads the whole frame by
    # +0.07 with an interval through zero, and quoting that as the result is
    # the kind of claim a reader checks and stops trusting the paper over.
    sig = [(l, g) for l, g in gaps if g[1] > 0]
    peak_txt = ""
    if sig:
        bl, g = max(sig, key=lambda t: t[1][1])
        by = dict((x, m) for x, m in zip(*_paired_sweep(d, bl, sizes,
                                                        n=1, seed=bl)[:2]))
        ax.plot([g[3]], [by[g[3]]], "o", color=SURFACE, ms=6, zorder=5)
        peak_txt = (f" At layer {bl} a region of {g[3]} patches "
                    f"({100 * g[3] / P:.0f}% of the frame) beats the whole "
                    f"frame by {g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}] -- an "
                    "interval clear of zero, paired over the same "
                    "photographs.")
    elif peaks:
        bx, by, bl = max(peaks, key=lambda t: t[1])
        ax.plot([bx], [by], "o", color=SURFACE, ms=6, zorder=5)
        peak_txt = (f" Best at {bx} patches, layer {bl}, $\\rho$ {by:.2f}, "
                    "though not yet clear of the whole-frame endpoint.")
    style.title(ax, "Both prior lenses are endpoints of one family",
                "Read a single patch and you have the logit lens; read the "
                "whole frame and you have the J-lens. The size in between is "
                "the only parameter this paper adds."
                f"{peak_txt} Big dots mark each depth's own "
                f"best; {nimg} photographs, 95% bootstraps resampled over "
                "photographs, so the eight sizes move together.",
                width=_subw(pw))

    ax = axes[1]
    tags = []
    for j, l in enumerate(layers):
        c = shade(j)
        xs, ms, er = [], [], [[], []]
        for v in sizes:
            q = d[(d["size"] == v) & (d.layer == l)]
            if len(q) < 8:
                continue
            mv, a_, b_ = _boot_mean(q["drop"].values, seed=7 * l + v)
            xs.append(v); ms.append(mv)
            er[0].append(mv - a_); er[1].append(b_ - mv)
        if not xs:
            continue
        ax.errorbar(xs, ms, yerr=er, color=c, marker=mark(j), ms=7.5, lw=2.4,
                    elinewidth=1.1, capsize=3.5, capthick=1.1, zorder=3)
        tags.append((xs[-1], ms[-1], f"layer {l}"))
    ax.axhline(0, color=MUTED, lw=1.1, ls=(0, (4, 3)))
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(v) for v in sizes])
    ax.minorticks_off()
    ax.set_xlabel(f"patches in the region  (of {P})")
    ax.set_ylabel("what covering it really costs  (logit margin)")
    style.title(ax, "The damage a region does, for scale",
                "Bigger regions cost more, so the left panel is not reading "
                "region size back to itself: at one patch there is real damage "
                "to predict and the reading misses it. Same depths, same "
                "colours and markers, as on the left.",
                width=_subw(pw))
    _tag_ends(ax, tags)
    fig.tight_layout()
    return _save(fig, out, name)
