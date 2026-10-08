# SPDX-License-Identifier: Apache-2.0
"""Watching a pointer resolve: the landmark's price against depth.

One plate, three panels.  The first is a render with the three regions the
intervention covers drawn on it, so the reader knows what the curves are about.
The second is the relational question and the third is the control, on the same
axes, and the roles exchange between them because the question does.

Conventions follow :mod:`jvlm.figwhen`: a heading above each panel and nothing
else, every explanation in the caption, no bold anywhere, and the lens colours
the rest of the paper uses -- logit lens orange, J-lens violet, VJ-Lens blue.
Roles are kept off those three hues on purpose, so a role can never be mistaken
for a lens: landmark black, referent teal, and the two null controls in grey.
"""
from __future__ import annotations

import os

import numpy as np

from jvlm import style
from jvlm.style import AQUA, BLUE, GRID, INK, INK2, MUTED, ORANGE, VIOLET

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROLE_C = {"landmark": INK, "referent": AQUA,
          "distractor": "#6f6d67", "floor": "#b8b6af"}
ROLE_M = {"landmark": "o", "referent": "s", "distractor": "^", "floor": None}
ROLE_LS = {"landmark": "-", "referent": "-", "distractor": "-", "floor": ":"}
LENS_C = {"logit lens": ORANGE, "J-lens": VIOLET, "VJ-Lens": BLUE}
ORDER = ["landmark", "referent", "distractor", "floor"]
NICE = {"landmark": "landmark", "referent": "referent",
        "distractor": "third object", "floor": "bare floor"}


# --------------------------------------------------------------------- tables

def curve(df, question, role, col="drop"):
    """Mean and standard error of ``col`` against layer, over scenes."""
    d = df[(df.question == question) & (df.role == role)]
    g = d.groupby("layer")[col].agg(["mean", "sem", "size"])
    return g.index.values.astype(float), g["mean"].values, g["sem"].values


def scene_scale(df, ref=None):
    """One number per scene: its largest price anywhere, either question.

    The effects are heavy-tailed -- a handful of scenes carry drops of ten or
    twenty logits and the rest carry fractions of one -- so a plain mean over
    scenes is a mean over those few.  Dividing each scene by its own largest
    price before averaging gives every scene one vote and leaves the shape of
    its curves, and the ratio between its roles, untouched.  The scale is the
    largest drop that scene produces anywhere -- either question, any region,
    any depth -- so no curve can exceed one and a scene with a small relational
    effect cannot be magnified by dividing by it.
    """
    d = df if ref is None else df[df.question == ref]
    return d.groupby("scene")["drop"].max().clip(lower=1e-6)


def ncurve(df, question, role, col="drop"):
    """The same curve, each scene first divided by its own largest price."""
    sc = scene_scale(df)
    d = df[(df.question == question) & (df.role == role)].copy()
    d["z"] = d[col] / d.scene.map(sc)
    g = d.groupby("layer")["z"].agg(["mean", "sem"])
    return g.index.values.astype(float), g["mean"].values, g["sem"].values


def crossing(df, question):
    """Layer at which the landmark's price falls below the referent's."""
    x, a, _ = curve(df, question, "landmark")
    _, b, _ = curve(df, question, "referent")
    y = a - b
    for i in range(len(x) - 1):
        if y[i] > 0 and y[i + 1] <= 0:
            return float(x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1]))
    return float("nan")


def rescale(df, lens, how="sum", roles=("landmark", "referent")):
    r"""One affine map placing a unitless lens on the logit axis.

    The logit lens and the J-lens report a word's logit at a patch, not a
    logit *change*, so they cannot be drawn against an intervention's cost
    without a scale.  One least-squares affine map is fitted per lens over the
    whole plate -- both questions, both roles, every depth -- so a marker's
    position still carries all of the reading's shape and none of its units.
    The VJ-Lens needs no such map: its readings are already in logits and are
    drawn as they come.

    Returns ``(a, b, pearson)``; the correlation is affine-invariant and is the
    number the caption should quote.
    """
    col, tru = [], []
    for q in df.question.unique():
        for r in roles:
            x, m, _ = curve(df, q, r, col=f"{lens}_{how}")
            _, t, _ = curve(df, q, r)
            good = np.isfinite(m) & np.isfinite(t)
            col.append(m[good])
            tru.append(t[good])
    x = np.concatenate(col)
    y = np.concatenate(tru)
    if len(x) < 3 or np.std(x) < 1e-12:
        return 0.0, float(np.mean(y)), float("nan")
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b), float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------- plate

def _scene_panel(ax, item, label=True):
    from PIL import Image

    im = Image.open(item["path"]).convert("RGB")
    W, H = im.size
    ax.imshow(im)
    FS = 18.8
    # a label's size on the picture, in the picture's own pixels, so two names
    # over neighbouring boxes can be told apart before they are drawn
    rend = ax.figure.canvas.get_renderer()
    bb = ax.get_window_extent(rend)
    scale = ax.figure.dpi / 72.0
    per_char = FS * scale * 0.50 * (W / max(bb.width, 1.0))
    line_h = FS * scale * 1.30 * (H / max(bb.height, 1.0))

    marks = []
    for r in ("landmark", "referent", "distractor"):
        x0, y0, w, h = item[f"box_{r}"]
        ax.add_patch(Rectangle((x0 * W, y0 * H), w * W, h * H, fill=False,
                               lw=1.8, ec=ROLE_C[r]))
        if not label:
            continue
        f = x0 + w / 2
        ha = "left" if f < 0.26 else ("right" if f > 0.72 else "center")
        cx = min(max(f, 0.03), 0.97) * W
        # the referent's box is often the landmark's neighbour, so its label
        # goes under the box and the other two over theirs
        below = r == "referent"
        wid = len(NICE[r]) * per_char
        lx0 = cx if ha == "left" else (cx - wid if ha == "right" else cx - wid / 2)
        marks.append(dict(r=r, cx=cx, ha=ha, below=below, x0=lx0, x1=lx0 + wid,
                          y=(y0 + h) * H + 4 if below else max(y0 * H - 4, 12),
                          row=0))
    # two names on the same side of their boxes whose spans meet are moved a
    # line apart: at the paper's type size "landmark" and "third object" printed
    # over each other whenever the two boxes stood side by side
    for side in (True, False):
        row = [m for m in marks if m["below"] is side]
        row.sort(key=lambda m: m["x0"])
        for i in range(1, len(row)):
            if row[i]["x0"] < row[i - 1]["x1"] + 0.02 * W:
                row[i]["row"] = row[i - 1]["row"] + 1
    for m in marks:
        dy = m["row"] * line_h
        ax.text(m["cx"], m["y"] + dy if m["below"] else m["y"] - dy,
                NICE[m["r"]], color=ROLE_C[m["r"]], fontsize=FS,
                va="top" if m["below"] else "bottom", ha=m["ha"],
                clip_on=False)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def retire(df, question, role, theta):
    """Last layer at which covering this region still costs more than theta."""
    x, m, _ = ncurve(df, question, role)
    w = np.nonzero(m > theta)[0]
    return float(x[w[-1]]) if len(w) else float("nan")


def noise(df, k=2.0):
    """What "costs nothing" is worth here, on the per-scene scale.

    Two null controls run at every depth under both questions: a third object
    in neither role, and a patch of bare floor with as many token cells as the
    landmark.  Twice the mean of their absolute readings is the line a region
    has to clear before it can be said to be doing anything.
    """
    v = []
    for q in df.question.unique():
        for r in ("distractor", "floor"):
            _, m, _ = ncurve(df, q, r)
            v.append(np.abs(m))
    return float(k * np.nanmean(np.concatenate(v)))


def _panel(ax, df, question, lenses, fits, how, ylim, scaled=False,
           theta=None, marks=True):
    for r in (ORDER if scaled else ORDER):
        x, m, se = (ncurve(df, question, r) if scaled
                    else curve(df, question, r))
        ax.fill_between(x, m - se, m + se, color=ROLE_C[r], alpha=0.13, lw=0)
        big = r in ("landmark", "referent")
        ax.plot(x, m, ROLE_LS[r], color=ROLE_C[r], lw=2.2 if big else 1.2,
                marker=ROLE_M[r] if big else None, ms=5.5,
                zorder=4 if big else 2)
    if not scaled:
        for li, lens in enumerate(lenses):
            a, b, _ = fits[lens]
            for r in ("landmark", "referent"):
                x, m, _ = curve(df, question, r, col=f"{lens}_{how}")
                good = np.isfinite(m)
                ax.plot(x[good] + (li - 1) * 0.22, a * m[good] + b, ROLE_M[r],
                        mfc="none", mec=LENS_C[lens], mew=1.3, ms=6.5,
                        ls="none", zorder=3)
    ax.axhline(0, color=GRID, lw=1.0, zorder=1)
    ax.set_ylim(*ylim)
    if not marks:
        return
    if scaled and theta is not None:
        # where each region stops being consulted, marked on the axis itself
        x0, x1 = ax.get_xlim()
        for i, r in enumerate(("referent", "landmark")):
            rr = retire(df, question, r, theta)
            _, mm, ss = ncurve(df, question, r)
            k = int(np.nanargmax(mm))
            # a retirement is only meaningful for a region that was doing
            # something to begin with: the peak has to clear two standard
            # errors of its own, or the mark would date the end of noise
            if not np.isfinite(rr) or mm[k] < 2 * ss[k]:
                continue
            ax.plot([rr], [ylim[0]], "^", color=ROLE_C[r], ms=7,
                    clip_on=False, zorder=5)
            f = (rr - x0) / max(x1 - x0, 1e-9)
            ax.annotate(f"{NICE[r]} done by {rr:.0f}", (rr, ylim[0]),
                        xytext=(0, 9 + 18 * i), textcoords="offset points",
                        color=ROLE_C[r], fontsize=18.8,
                        ha="right" if f > 0.62 else
                        ("left" if f < 0.14 else "center"))
    if not scaled:
        c = crossing(df, question)
        _, la, _ = ncurve(df, question, "landmark")
        _, re, _ = ncurve(df, question, "referent")
        if theta is not None and not (np.nanmax(la) > theta
                                      and np.nanmax(re) > theta):
            c = float("nan")           # two flat lines cross everywhere
        if np.isfinite(c):
            ax.axvline(c, color=INK2, lw=1.0, ls=(0, (1, 2)), zorder=1)
            x0, x1 = ax.get_xlim()
            right = (c - x0) / max(x1 - x0, 1e-9) > 0.6
            ax.annotate(f"crossing, layer {c:.1f}", (c, ylim[0]),
                        xytext=(-4 if right else 4, 6),
                        textcoords="offset points", color=INK2, fontsize=18.8,
                        va="bottom", ha="right" if right else "left")


def pointer(df, items, out, name="f120_pointer.png", how="sum",
            example=None, figsize=(14.8, 8.2)):
    """The plate.  ``df`` is the rows table for one baseline and one model."""
    style.use()
    lenses = [l for l in ("logit lens", "J-lens", "VJ-Lens")
              if f"{l}_{how}" in df.columns and df[f"{l}_{how}"].notna().any()]
    fits = {l: rescale(df, l, how) for l in lenses}
    theta = noise(df)

    lo = hi = 0.0
    for q in ("main", "control"):
        for r in ORDER:
            _, m, se = curve(df, q, r)
            lo = min(lo, float(np.nanmin(m - se)))
            hi = max(hi, float(np.nanmax(m + se)))
    pad = 0.14 * (hi - lo)
    # the headroom is the two-row key's, not the curve's: at the paper's type
    # size the key sat on the referent's own band at its peak
    ylim = (lo - pad, hi + 2.75 * pad)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, width_ratios=[0.92, 1.15, 1.15],
                          height_ratios=[1.0, 0.82], wspace=0.2, hspace=0.30,
                          left=0.05, right=0.995, top=0.915, bottom=0.085)
    ex = ([example] if isinstance(example, int) else
          list(example) if example else [0])
    ex = (ex + [i for i in range(len(items)) if i not in ex])[:2]
    for k, e in enumerate(ex):
        axk = fig.add_subplot(gs[k, 0])
        _scene_panel(axk, items[e])
        if k == 0:
            style.title(axk, "two scenes, three regions each")

    axes = {}
    for j, q in enumerate(("main", "control")):
        a1 = fig.add_subplot(gs[0, j + 1], sharey=axes.get("main"))
        _panel(a1, df, q, lenses, fits, how, ylim, theta=theta)
        style.title(a1, "asked about the referent" if q == "main"
                    else "asked about the landmark")
        nlo = nhi = 0.0
        for qq in ("main", "control"):
            for r in ORDER:
                _, m, se = ncurve(df, qq, r)
                nlo = min(nlo, float(np.nanmin(m - se)))
                nhi = max(nhi, float(np.nanmax(m + se)))
        npad = 0.16 * (nhi - nlo)
        a2 = fig.add_subplot(gs[1, j + 1], sharey=axes.get("main_s"))
        _panel(a2, df, q, lenses, fits, how, (nlo - npad, nhi + npad),
               scaled=True, theta=theta)
        a2.set_xlabel("layer")
        axes[q], axes[q + "_s"] = a1, a2
        if j == 0:
            a1.set_ylabel("drop in the answer word  (logits)")
            a2.set_ylabel("share of the scene's largest price")
        else:
            plt.setp(a1.get_yticklabels(), visible=False)
            plt.setp(a2.get_yticklabels(), visible=False)
        plt.setp(a1.get_xticklabels(), visible=False)

    h1 = [Line2D([], [], color=ROLE_C[r], ls=ROLE_LS[r],
                 marker=ROLE_M[r], ms=5.5,
                 lw=2.2 if r in ("landmark", "referent") else 1.2,
                 label=NICE[r]) for r in ORDER]
    axes["main"].legend(handles=h1, loc="upper right", ncol=2, fontsize=18.8,
                        labelspacing=0.35, handletextpad=0.6,
                        borderaxespad=0.3)
    h2 = [Line2D([], [], ls="none", marker="o", mfc="none", mec=LENS_C[l],
                 mew=1.3, ms=6.5, label=l) for l in lenses]
    axes["control"].legend(handles=h2, loc="upper right", fontsize=18.8,
                           labelspacing=0.35, handletextpad=0.6)

    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p)
    plt.close(fig)
    return p, fits, theta


def exchange(df, role, window=(3, 9)):
    """The role-exchange test, paired within scene.

    Over the depths where the landmark's price peaks under the relational
    question, each scene contributes one number per question: the mean cost of
    covering that region.  The pixels are identical between the two, so a
    difference is a difference in what the region is being used *for*.
    """
    from scipy import stats as sps

    lo, hi = window
    d = df[(df.role == role) & (df.layer >= lo) & (df.layer <= hi)]
    t = d.pivot_table(index="scene", columns="question", values="drop")
    if not {"main", "control"} <= set(t.columns) or len(t) < 6:
        return {}
    a, b = t["main"].values, t["control"].values
    try:
        w = sps.wilcoxon(a, b)
        pv = float(w.pvalue)
    except ValueError:
        pv = float("nan")
    return dict(window=[lo, hi], n=int(len(t)),
                main=round(float(a.mean()), 3),
                control=round(float(b.mean()), 3),
                bigger_under_main=int((a > b).sum()),
                wilcoxon_p=round(pv, 5))


def retire_pairs(df, question="main", k=2.0):
    """Per scene: the last depth at which each role still costs anything.

    The pooled curves are means, and a mean can order two curves that no single
    scene orders.  This does the same comparison inside each scene: the last
    measured layer at which covering the landmark still costs more than that
    scene's own null level, against the same for the referent, with the null
    level taken from that scene's third object and bare-floor controls.
    """
    from scipy import stats as sps

    out = {}
    for sc, g in df[df.question == question].groupby("scene"):
        null = g[g.role.isin(["distractor", "floor"])]["drop"].abs().mean()
        th = max(k * float(null), 0.2)
        r = {}
        for role in ("landmark", "referent"):
            v = g[g.role == role].sort_values("layer")
            w = v[v["drop"] > th]
            r[role] = float(w.layer.max()) if len(w) else float("nan")
        out[int(sc)] = r
    both = [(v["landmark"], v["referent"]) for v in out.values()
            if np.isfinite(v["landmark"]) and np.isfinite(v["referent"])]
    if len(both) < 5:
        return {"per_scene": out, "n_both": len(both)}
    a = np.array([x for x, _ in both])
    b = np.array([y for _, y in both])
    return {"per_scene": out, "n_both": len(both),
            "median_landmark": float(np.median(a)),
            "median_referent": float(np.median(b)),
            "landmark_earlier": int((a < b).sum()),
            "referent_earlier": int((a > b).sum()),
            "tied": int((a == b).sum()),
            "wilcoxon_p": round(float(sps.wilcoxon(a, b).pvalue), 5)
            if (a != b).any() else float("nan"),
            "sign_p": round(float(sps.binomtest(
                int((a < b).sum()), int((a != b).sum()), 0.5,
                alternative="greater").pvalue), 5)
            if (a != b).any() else float("nan")}


def spillover(df, items, grid=14, window=(3, 9)):
    """Is the landmark's price just damage leaking onto its neighbour?

    The referent is by construction the landmark's nearest neighbour, so the
    obvious alternative reading of the middle panel is that covering one region
    degrades the other.  Three quantities test it: whether the two prices move
    together across scenes, whether the landmark's price depends on how many
    token cells separate the regions once its own region size is accounted for,
    and what the handoff looks like in the scenes whose regions do not touch.
    """
    from scipy import stats as sps

    lo, hi = window
    by = {it["scene_index"]: it for it in items}
    w = df[(df.question == "main") & (df.layer >= lo) & (df.layer <= hi)]
    lm = w[w.role == "landmark"].groupby("scene")["drop"].mean()
    rf = w[w.role == "referent"].groupby("scene")["drop"].mean()
    gap, size = [], []
    for sc in lm.index:
        it = by[int(sc)]
        A = [(t // grid, t % grid) for t in it["tok_landmark"]]
        B = [(t // grid, t % grid) for t in it["tok_referent"]]
        gap.append(min(abs(a[0] - b[0]) + abs(a[1] - b[1])
                       for a in A for b in B))
        size.append(len(it["tok_landmark"]))
    gap, size = np.array(gap, float), np.array(size, float)

    def resid(y, x):
        ry, rx = sps.rankdata(y), sps.rankdata(x)
        return ry - np.polyval(np.polyfit(rx, ry, 1), rx)

    far = [int(s) for s, g in zip(lm.index, gap) if g > 1]
    sub = df[df.scene.isin(far)]
    out = dict(
        n=int(len(lm)),
        price_vs_referent=[round(float(v), 3)
                           for v in sps.spearmanr(lm.values, rf.values)],
        price_vs_gap=[round(float(v), 3)
                      for v in sps.spearmanr(gap, lm.values)],
        price_vs_size=[round(float(v), 3)
                       for v in sps.spearmanr(size, lm.values)],
        partial_price_vs_gap_given_size=round(float(np.corrcoef(
            resid(lm.values, size), resid(gap, size))[0, 1]), 3),
        gap_vs_size=[round(float(v), 3) for v in sps.spearmanr(gap, size)],
        n_touching=int((gap <= 1).sum()), n_apart=len(far))
    if len(far) >= 5:
        rp = retire_pairs(sub, "main")
        rp.pop("per_scene", None)
        out["apart_retire_pairs"] = rp
        for r in ("landmark", "referent"):
            x, m, se = curve(sub, "main", r)
            k = int(np.nanargmax(m))
            out[f"apart_{r}"] = dict(peak_layer=int(x[k]),
                                     peak=round(float(m[k]), 3),
                                     peak_se=round(float(se[k]), 3),
                                     last=round(float(m[-1]), 3))
        out["apart_exchange_landmark"] = exchange(sub, "landmark")
    return out


def report(df, items):
    """Every number the caption has to carry."""
    theta = noise(df)
    o = {"n_scenes": int(df.scene.nunique()),
         "layers": sorted(int(x) for x in df.layer.unique()),
         "theta": round(theta, 4)}
    for q in ("main", "control"):
        for r in ORDER:
            x, m, se = curve(df, q, r)
            nx, nm, nse = ncurve(df, q, r)
            k, nk = int(np.nanargmax(m)), int(np.nanargmax(nm))
            # how much of the whole picture's causal weight this region is
            d = df[(df.question == q) & (df.role == r)]
            sh = (d.groupby("layer")["drop"].mean()
                  / d.groupby("layer")["drop_all"].mean())
            o[f"{q}_{r}"] = dict(
                peak_layer=int(x[k]), peak=round(float(m[k]), 3),
                peak_se=round(float(se[k]), 3),
                first=round(float(m[0]), 3), last=round(float(m[-1]), 3),
                last_se=round(float(se[-1]), 3),
                collapse=round(float(m[k] - m[-1]), 3),
                n_peak_layer=int(nx[nk]), n_peak=round(float(nm[nk]), 3),
                n_peak_se=round(float(nse[nk]), 3),
                n_last=round(float(nm[-1]), 3),
                share_at_peak=round(float(sh.loc[int(nx[nk])]), 4),
                scenes_above=int((d.groupby("scene")["drop"].max()
                                  > 0.5).sum()),
                retire=round(retire(df, q, r, theta), 2))
        o[f"cross_{q}"] = round(crossing(df, q), 2)
    for lens in ("logit lens", "J-lens", "VJ-Lens"):
        for how in ("sum", "mean"):
            c = f"{lens}_{how}"
            if c not in df.columns or not df[c].notna().any():
                continue
            a, b, r = rescale(df, lens, how)
            o[f"fit_{lens}_{how}"] = dict(a=round(a, 5), b=round(b, 4),
                                          pearson=round(r, 3))
    o["region_tokens"] = {r: round(float(df[df.role == r].n_tok.mean()), 1)
                          for r in ORDER}
    for r in ("landmark", "referent"):
        o[f"exchange_{r}"] = exchange(df, r)
    rp = retire_pairs(df, "main")
    rp.pop("per_scene", None)
    o["retire_pairs_main"] = rp
    try:
        o["spillover"] = spillover(df, items)
    except Exception as e:                     # items file from an older run
        o["spillover"] = {"error": f"{type(e).__name__}: {e}"}
    return o
