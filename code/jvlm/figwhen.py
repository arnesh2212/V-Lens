# SPDX-License-Identifier: Apache-2.0
"""Depth figures from the layerwise causal map: when the model stops looking.

The layerwise run covers all 49 regions of a photograph at twelve depths and
records what each covering does to the answered word.  Read across depth rather
than across space, that table answers three questions no single heatmap can:

``when_map``     for each place, the last depth at which covering it still
                 costs the answer anything -- a map of when the model is
                 finished with a place, painted over the photograph itself.
``absorption``   the total causal dependence of the answer on the picture
                 against depth, and how far it moves when the question changes.
``lag``          the depth at which a region's *content* resolves against the
                 depth at which the region is *consulted*.

Conventions follow :mod:`jvlm.figcausal`: square tiles because they carry text,
a panel heading and nothing else above the axes, and every explanation in the
caption rather than in the PNG.
"""
from __future__ import annotations

import os

import numpy as np

from jvlm import style
from jvlm.style import (AQUA, BLUE, GREEN, INK, INK2, MAGENTA, MUTED, ORANGE,
                        VIOLET)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# the same light-to-dark ramp the causal cards use for a magnitude; here the
# magnitude is a depth, and the dark end is late
RAMP = ["#f7fbfd", "#e3eff7", "#c9e0ef", "#a5cce4", "#79b2d6", "#4e93c4",
        "#2a72ab", "#14528a"]
NEVER = "#eceae4"
LENS_COLOUR = {"logit lens": ORANGE, "J-lens": VIOLET, "VJ-Lens": BLUE,
               "VJ-Lens (point)": AQUA}


# ------------------------------------------------------------------ the tables

def group(cards):
    """``[{image_id, layer, ...}]`` -> ``{(image_id, concept): {layer: card}}``."""
    by: dict = {}
    for r in cards:
        by.setdefault((r["image_id"], r["concept"]), {})[int(r["layer"])] = r
    return by


def _stack(v):
    """One photograph's layer dict -> depths ``[L]``, drops ``[L, R]``, layers."""
    ls = sorted(v)
    return (np.array([v[l]["depth"] for l in ls], dtype=float),
            np.array([v[l]["truth"] for l in ls], dtype=float), ls)


def smooth_depth(T, k=3):
    """A centred moving mean of ``k`` adjacent measured depths, per region.

    The drops arrive quantised to multiples of 0.0625 logits by the model's
    bfloat16 head, so a single layer can clear any threshold on one step of
    rounding.  Without this, four regions in ten change their retirement depth
    by more than a layer when the threshold is doubled; with it, fewer than two
    in ten do.  The window is the smallest odd one that can outvote a stray
    single step.
    """
    L = T.shape[0]
    out = np.empty_like(T, dtype=float)
    h = k // 2
    for i in range(L):
        lo, hi = max(0, i - h), min(L, i + h + 1)
        out[i] = T[lo:hi].mean(0)
    return out


def retirement(T, D, theta, k=3):
    """Last depth at which covering each region still costs more than ``theta``.

    Regions that never clear the threshold at any measured depth come back as
    NaN -- they are not "retired at layer zero", they were never consulted at a
    strength this measurement can see, and colouring them as if they were the
    earliest retirees would put the least-used places at the same end of the
    ramp as the fastest-resolved ones.
    """
    live = smooth_depth(T, k) > theta
    out = np.full(T.shape[1], np.nan)
    for j in range(T.shape[1]):
        w = np.nonzero(live[:, j])[0]
        if len(w):
            out[j] = D[w[-1]]
    return out


def place_specific(T):
    """Per layer: the total drop, the part specific to *which* region, the pull.

    Replacing a region with the average photograph's activations moves the
    answer two ways at once.  One is a dependence on that place: this content,
    here, was doing work.  The other is a pull the baseline exerts wherever it
    is applied -- the average photograph is a photograph too, and injecting it
    biases the answer whatever region it lands in.  The uniform part is
    estimated by the median over regions at that layer, which is what a region
    has to beat before it can be said to matter as a place.
    """
    med = np.median(T, axis=1, keepdims=True)
    return (np.abs(T).sum(1), np.abs(T - med).sum(1),
            np.abs(med[:, 0]) * T.shape[1])


def centroid(y, d):
    y = np.clip(np.asarray(y, float), 0.0, None)
    return float((y * d).sum() / y.sum()) if y.sum() > 0 else float("nan")


# ------------------------------------------------------------------- the plate

def _head(ax, title, fontsize=20.6):
    ax.set_title(title, loc="center", color=INK, pad=6, fontsize=fontsize)


def _photo(ax, path, box_cells=(), C=7, alpha=1.0):
    from PIL import Image
    try:
        im = Image.open(path).convert("RGB").resize((448, 448), Image.LANCZOS)
        ax.imshow(im, extent=(0, C, C, 0), alpha=alpha, interpolation="bilinear")
    except Exception:
        ax.add_patch(Rectangle((0, 0), C, C, facecolor="#eef2f6",
                               edgecolor="none"))
    ax.set_xlim(0, C)
    ax.set_ylim(C, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for cell in box_cells:
        y, x = divmod(int(cell), C)
        ax.add_patch(Rectangle((x, y), 1, 1, fill=False, edgecolor=INK,
                               lw=1.0, linestyle=(0, (3, 2))))


def _depth_shade(d, lo, hi):
    if not np.isfinite(d):
        return NEVER
    q = (d - lo) / max(hi - lo, 1e-9)
    return RAMP[max(0, min(len(RAMP) - 1, int(q * len(RAMP))))]


def when_map(cards, out, name="f83_whenmap.png", thetas=(0.03, 0.06),
             per_row=2, fs=17.4, max_photos=4):
    """Photograph, then the depth at which each place stops mattering.

    Two thresholds side by side, so a reader can see for themselves that the
    picture is not an artefact of where the line was drawn.  At most
    ``max_photos`` photographs are drawn, taken evenly across the
    finished-soonest-to-never ordering, so the spread survives and the plate
    stays readable; the full set is in the result file.
    """
    style.use()
    by = group(cards)

    def _order(kv):
        D, T, _ls = _stack(kv[1])
        r = retirement(T, D, thetas[0])
        return float(np.mean(np.where(np.isfinite(r), r, 0.0)))

    # the plate reads from the photographs the model finishes with soonest to
    # the ones it never finishes with, which is the figure's whole claim
    items = sorted(by.items(), key=_order)
    if max_photos and len(items) > max_photos:
        idx = sorted({int(round(t)) for t in
                      np.linspace(0, len(items) - 1, max_photos)})
        items = [items[i] for i in idx]
    n = len(items)
    ncol = per_row * (1 + len(thetas))
    nrow = int(np.ceil(n / per_row))
    fig = plt.figure(figsize=(3.45 * ncol, 3.78 * nrow + 1.15))
    gs = fig.add_gridspec(nrow + 1, ncol,
                          height_ratios=[1] * nrow + [0.20],
                          hspace=0.30, wspace=0.06,
                          left=0.012, right=0.988, top=0.955, bottom=0.045)
    D0 = L0 = None
    for k, (key, v) in enumerate(items):
        row, slot = divmod(k, per_row)
        D, T, ls = _stack(v)
        D0, L0 = D, ls
        r0 = v[ls[0]]
        C = int(r0["coarse"])
        keep = r0["keep"]
        inside = set(r0.get("inside", []))
        lo, hi = float(D[0]), float(D[-1])
        ax = fig.add_subplot(gs[row, slot * (1 + len(thetas))])
        _photo(ax, r0["path"], [c for c in keep if c in inside], C)
        _head(ax, f"the model answers “{r0['concept']}”")
        for ti, th in enumerate(thetas):
            ret = retirement(T, D, th)
            axm = fig.add_subplot(gs[row, slot * (1 + len(thetas)) + 1 + ti])
            _photo(axm, r0["path"], (), C)
            for j, cell in enumerate(keep):
                y, x = divmod(int(cell), C)
                axm.add_patch(Rectangle(
                    (x + .02, y + .02), .96, .96,
                    facecolor=_depth_shade(ret[j], lo, hi),
                    edgecolor="#ffffff", lw=0.6, alpha=0.76))
                if np.isfinite(ret[j]):
                    q = (ret[j] - lo) / max(hi - lo, 1e-9)
                    txt = str(int(ls[int(np.argmin(np.abs(D - ret[j])))]))
                    ink = "#ffffff" if q > 0.62 else INK
                else:
                    txt, ink = "–", MUTED
                axm.text(x + .5, y + .52, txt, ha="center", va="center",
                         color=ink, fontsize=fs)
            _head(axm, f"θ = {th:.2f} logits")
    # the key is labelled in layers, because the tiles are: a reader should
    # never have to convert a number in a tile into a number in the key
    if D0 is not None:
        axk = fig.add_subplot(gs[nrow, 0:2])
        axk.set_xlim(-2.0, len(D0) + 0.1)
        axk.set_ylim(0, 1)
        axk.axis("off")
        for i, d in enumerate(D0):
            axk.add_patch(Rectangle((i + .06, .30), .88, .44,
                                    facecolor=_depth_shade(d, D0[0], D0[-1]),
                                    edgecolor="none"))
            axk.text(i + .5, .16, str(int(L0[i])), ha="center", va="top",
                     color=INK2, fontsize=18.1)
        axk.add_patch(Rectangle((-1.90, .30), .88, .44, facecolor=NEVER,
                                edgecolor="none"))
        axk.text(-1.46, .16, "never", ha="center", va="top", color=INK2,
                 fontsize=18.1)
        axk.text(-1.90, .84, "last layer that still costs the answer",
                 ha="left", va="bottom", color=INK2, fontsize=18.8)
    p = os.path.join(out, name)
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- absorption

def absorption(cards, out, name="f84_absorption.png", two=None):
    """Total causal dependence against depth, and what the question does to it."""
    style.use()
    by = group(cards)
    if two:
        fig, axes = plt.subplots(1, 3, figsize=(22.0, 5.4),
                                 gridspec_kw={"width_ratios": [1, 1, 1.02]})
    else:
        fig, axes = plt.subplots(1, 1, figsize=(7.4, 4.9))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    tot, spec, D = [], [], None
    for key, v in sorted(by.items()):
        D, T, _ = _stack(v)
        a, b, _c = place_specific(T)
        tot.append(a)
        spec.append(b)
        ax.plot(D, b, color=BLUE, lw=0.9, alpha=0.30, zorder=1)
    tot = np.array(tot)
    spec = np.array(spec)
    ax.plot(D, tot.mean(0), color=MUTED, lw=2.0, ls=(0, (5, 2)), zorder=3)
    ax.plot(D, spec.mean(0), color=BLUE, lw=2.6, marker="o", ms=5.5, zorder=4)
    style.label_end(ax, D[-1], tot.mean(0)[-1], "every covering, added up",
                    MUTED)
    style.label_end(ax, D[-1], spec.mean(0)[-1],
                    "the part that is about which place", BLUE)
    ax.set_xlabel("depth into the network")
    ax.set_ylabel("total effect of covering, logits")
    ax.set_ylim(0, None)
    ax.set_xlim(D[0] - 0.02, D[-1] + 0.30)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    style.panel(ax, f"one question, {len(spec)} "
                    f"photograph{'s' if len(spec) != 1 else ''}", "lower left")
    _head(ax, "Dependence on the picture, by depth")

    if two:
        ax2 = axes[1]
        D2 = np.array(two["depths"], dtype=float)
        cols = {"easy": GREEN, "hard": MAGENTA}
        lab = {"easy": "name the object", "hard": "count them"}
        mks = {"easy": "o", "hard": "s"}
        cen2 = []
        for qi, q in enumerate(("easy", "hard")):
            per = []
            for r in two["cards"]:
                T = np.asarray(r["questions"][q]["truth"], dtype=float)
                per.append(place_specific(T)[1])
                ax2.plot(D2, per[-1], color=cols[q], lw=0.9, alpha=0.26,
                         zorder=1)
            mu = np.mean(per, 0)
            ax2.plot(D2, mu, color=cols[q], lw=2.6, marker=mks[q], ms=5.5,
                     ls="-" if q == "easy" else (0, (5, 2)), zorder=4)
            style.label_end(ax2, D2[-1], mu[-1], lab[q], cols[q],
                            marker=mks[q])
            cen2.append((centroid(mu, D2), cols[q]))
        ax2.set_xlabel("depth into the network")
        ax2.set_ylabel("total effect of covering, logits")
        ax2.set_ylim(0, None)
        ax2.set_xlim(D2[0] - 0.02, D2[-1] + 0.30)
        ax2.set_xticks([0.2, 0.4, 0.6, 0.8])
        # the two centres of mass sit inside the axes, low and clear of every
        # curve; below the spine they land on the axis label
        top = ax2.get_ylim()[1]
        for ci, (cd, col) in enumerate(cen2):
            ax2.plot([cd], [top * (0.025 + 0.055 * ci)], marker="v", color=col,
                     ms=10, zorder=6)
        nph = len(two["cards"])
        style.panel(ax2, f"two questions, {nph} "
                         f"photograph{'s' if nph != 1 else ''}", "upper right")
        _head(ax2, "Depth, by question")

        # third panel: the maps themselves.  Same picture, same baseline, same
        # activations -- the only thing that changed is the words after the
        # image, and the map is redrawn.  The within-question, adjacent-layer
        # agreement is the reference: how much agreement survives at all.
        ax3 = axes[2]
        from scipy import stats as sps
        btw, wth = [], []
        for r in two["cards"]:
            E = np.asarray(r["questions"]["easy"]["truth"], dtype=float)
            H = np.asarray(r["questions"]["hard"]["truth"], dtype=float)
            for l in range(len(D2)):
                if E[l].std() > 0 and H[l].std() > 0:
                    btw.append(sps.spearmanr(E[l], H[l]).statistic)
            for M in (E, H):
                for l in range(len(D2) - 1):
                    if M[l].std() > 0 and M[l + 1].std() > 0:
                        wth.append(sps.spearmanr(M[l], M[l + 1]).statistic)
        rng = np.random.default_rng(11)
        ax3.axvline(0, color=MUTED, lw=1.0, ls=(0, (3, 3)), zorder=1)
        for y0, vals, col in ((1, np.array(wth), BLUE),
                              (0, np.array(btw), MAGENTA)):
            ax3.scatter(vals, y0 + rng.uniform(-0.13, 0.13, len(vals)), s=26,
                        facecolor=col, edgecolor="none", alpha=0.45, zorder=3)
            ax3.plot([np.median(vals)], [y0], marker="D", ms=12, color=col,
                     zorder=5)
            ax3.text(np.median(vals), y0 + 0.30, f"{np.median(vals):.2f}",
                     ha="center", va="bottom", color=INK2, fontsize=19.6)
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(["other question,\nsame layer",
                             "same question,\nnext layer"])
        ax3.set_ylim(-0.62, 1.72)
        # over two lines: on one it is wider than the panel and runs off the
        # right-hand edge of the figure
        ax3.set_xlabel("rank correlation between\ntwo causal maps")
        ax3.grid(axis="y", visible=False)
        ax3.grid(axis="x", visible=True)
        _head(ax3, "Map agreement, across questions")
    fig.tight_layout(pad=1.4)
    p = os.path.join(out, name)
    fig.savefig(p)
    plt.close(fig)
    return p


# ----------------------------------------------------------------------- lag

def say_use(cards, lenses=("logit lens", "J-lens", "VJ-Lens")):
    """Per photograph: the consultation curve and each lens's naming curve."""
    by = group(cards)
    rows = []
    for key, v in sorted(by.items()):
        D, T, ls = _stack(v)
        r0 = v[ls[0]]
        keep = r0["keep"]
        inside = set(r0.get("inside", []))
        c = r0["concept"]
        bx = [j for j, cell in enumerate(keep) if cell in inside]
        use = np.clip(T[:, bx], 0, None).sum(1)
        nam = {}
        for ln in lenses:
            nam[ln] = np.array(
                [np.mean([v[l]["lenses"][ln]["word"][j] == c for j in bx])
                 if ln in v[l]["lenses"] else np.nan for l in ls])
        rows.append(dict(key=key, concept=c, depths=D, use=use, name=nam,
                         n_box=len(bx)))
    return rows


def lag(rows, out, name="f85_lag.png",
        lenses=("logit lens", "J-lens", "VJ-Lens")):
    """Where the region's content resolves against where it is consulted."""
    style.use()
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.0),
                             gridspec_kw={"width_ratios": [1.20, 1.0]})
    D = rows[0]["depths"]

    ax = axes[0]
    u = np.mean([r["use"] / max(r["use"].max(), 1e-9) for r in rows], 0)
    u = u / u.max()
    ax.plot(D, u, color=INK2, lw=3.0, marker="s", ms=6.5, zorder=6)
    ends = [(u[-1], "covering it still costs", INK2, "s")]
    cur = {"consulted": u}
    for ln in lenses:
        m = np.nanmean([r["name"][ln] for r in rows], 0)
        m = m / max(np.nanmax(m), 1e-9)
        cur[ln] = m
        ax.plot(D, m, color=LENS_COLOUR[ln], lw=2.2, marker="o", ms=5, zorder=4)
        ends.append((m[-1], f"{ln} names it", LENS_COLOUR[ln], "o"))
    # two of these four series finish within a hundredth of each other, and a
    # label printed at each one's own height prints them on top of each other;
    # the labels are pushed apart and a hairline leads each back to its line
    ends.sort(key=lambda t: -t[0])
    placed = []
    for y, txt, col, mk in ends:
        yy = y if not placed else min(y, placed[-1] - 0.085)
        placed.append(yy)
        style.label_end(ax, D[-1], yy, txt, col, y_true=y, marker=mk)
    for nm, y in cur.items():
        cd = centroid(y, D)
        col = INK2 if nm == "consulted" else LENS_COLOUR[nm]
        ax.plot([cd], [-0.095 if nm == "consulted" else -0.040], marker="v",
                color=col, ms=9, clip_on=False, zorder=6)
    ax.set_xlabel("depth into the network")
    ax.set_ylabel("share of the curve's own peak")
    ax.set_ylim(-0.10, 1.10)
    ax.set_xlim(D[0] - 0.02, D[-1] + 0.36)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    _head(ax, "Naming, against use")

    ax2 = axes[1]
    ax2.axvline(0, color=MUTED, lw=1.0, ls=(0, (3, 3)), zorder=1)
    rng = np.random.default_rng(4)
    for i, ln in enumerate(lenses):
        y0 = len(lenses) - 1 - i
        vals = []
        for r in rows:
            du = centroid(r["use"], D)
            ds = centroid(r["name"][ln], D)
            if np.isfinite(du) and np.isfinite(ds):
                vals.append(ds - du)
        vals = np.array(vals)
        ax2.scatter(vals, y0 + rng.uniform(-0.11, 0.11, len(vals)), s=52,
                    facecolor=LENS_COLOUR[ln], edgecolor="none", alpha=0.70,
                    zorder=3)
        ax2.plot([vals.mean()], [y0], marker="D", ms=12,
                 color=LENS_COLOUR[ln], zorder=5)
        ax2.text(vals.mean(), y0 + 0.30, f"{vals.mean():+.2f}", ha="center",
                 va="bottom", color=INK2, fontsize=19.6)
    ax2.set_yticks(range(len(lenses)))
    ax2.set_yticklabels(list(lenses)[::-1])
    ax2.set_ylim(-0.62, len(lenses) - 0.28)
    ax2.set_xlabel("naming depth minus consultation depth")
    ax2.grid(axis="y", visible=False)
    ax2.grid(axis="x", visible=True)
    _head(ax2, "The lag, by photograph")
    fig.tight_layout()
    p = os.path.join(out, name)
    fig.savefig(p)
    plt.close(fig)
    return p
