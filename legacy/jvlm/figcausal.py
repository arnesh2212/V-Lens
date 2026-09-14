# SPDX-License-Identifier: Apache-2.0
"""The side-by-side card: three lenses against the model's own causal map.

Every panel is the same grid of regions over the same photograph.  One panel is
the ground truth -- what covering each region actually does to the word the
model was about to say -- and the others are what each lens claims.  A reader
can therefore check the claim against the truth in the same glance, which is
the only way a qualitative example is worth printing.

Tiles carry words, not colour alone.  A translucent wash over a photograph can
show that two maps differ; it cannot show *what* differs, and it is unreadable
at any size.
"""
from __future__ import annotations

import os

import numpy as np

from jvlm import style
from jvlm.style import BLUE, GRID, INK, INK2, MUTED

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# a single hue, light to dark, is the right ramp for a magnitude; the truth
# panel and the lens panels share it so a reader compares shade to shade
RAMP = ["#f7fbfd", "#e3eff7", "#c9e0ef", "#a5cce4", "#79b2d6", "#4e93c4",
        "#2a72ab", "#14528a"]
HOT = "#b8481f"
SOFT = "#dde9f8"
PANEL = "#eef2f6"


def _shade(q: float) -> str:
    return RAMP[max(0, min(len(RAMP) - 1, int(q * len(RAMP))))]


def _ink_for(q: float) -> str:
    return "#ffffff" if q > 0.62 else INK


def _tiles(ax, vals, C, keep, words=None, ring=(), box=(), fs=16.7):
    """Draw a C x C grid of labelled tiles.  ``keep`` maps tile -> grid cell."""
    v = np.asarray(vals, dtype=float)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    span = (hi - lo) or 1.0
    ax.set_xlim(0, C)
    ax.set_ylim(C, 0)
    ax.set_aspect("equal")          # square tiles, or the words never fit
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for j, cell in enumerate(keep):
        y, x = divmod(int(cell), C)
        q = (v[j] - lo) / span
        ax.add_patch(Rectangle((x + .015, y + .015), .97, .97,
                               facecolor=_shade(q), edgecolor="none"))
        if words is not None:
            w = str(words[j])
            # a tile is one unit wide; at this font a unit holds about eight
            # characters, so longer words are clipped rather than allowed to
            # run into the neighbouring tile
            if len(w) > 8:
                w = w[:7] + "\u2026"
            ax.text(x + .5, y + .52, w, ha="center", va="center",
                    color=_ink_for(q), fontsize=fs)
        if j in ring:
            ax.add_patch(Rectangle((x + .06, y + .06), .88, .88, fill=False,
                                   edgecolor=HOT, lw=2.0))
    for cell in box:
        y, x = divmod(int(cell), C)
        ax.add_patch(Rectangle((x, y), 1, 1, fill=False, edgecolor=INK,
                               lw=1.0, linestyle=(0, (3, 2))))


def _head(ax, title, sub=None, sub_colour=None):
    """Panel heading above the axes, and nothing else.

    A panel carries a name; what the panel means is the caption's job.  ``sub``
    survives as an argument so a caller can pass one deliberately, but the
    figures in this module no longer do, and with no second line the pad is the
    heading's own -- which is what reclaims the band the block used to sit in.
    """
    fs = 13.0
    lines = (str(sub).count("\n") + 1) if sub else 0
    pad = 10 + lines * fs * 1.35
    ax.set_title(title, loc="center", color=INK, pad=pad, fontsize=20.6)
    if sub:
        ax.annotate(sub, xy=(0.5, 1), xytext=(0, 7), xycoords="axes fraction",
                    textcoords="offset points", ha="center", va="bottom",
                    color=sub_colour or INK2, fontsize=fs, linespacing=1.35)


def card(rec, img, order, rho, out, name, top_k: int = 3, fs: float = 11.5):
    """One photograph: the truth, then each lens, side by side.

    Four panels at most.  An earlier version put five 7x7 word grids in a row
    and the words ran into each other, which is the failure this whole figure
    exists to avoid.
    """
    C = int(rec["coarse"])
    keep = rec["keep"]
    truth = np.asarray(rec["truth"], dtype=float)
    ring = set(np.argsort(-truth)[:top_k].tolist())
    box = set(rec.get("inside", []))
    order = list(order)[:3]
    n = len(order) + (2 if img is not None else 1)
    # the panels are square (the tiles have to be, or the words never fit), so
    # the canvas is only as tall as one panel is wide plus the heading; the
    # taller canvas the explanatory block needed left a white band under it
    # the tile has to be wide enough for the longest word a tile may carry:
    # at the narrower panel an eight-letter word ran the full width of its
    # square and touched the word in the next one
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6.6))
    axes = np.atleast_1d(axes)
    i = 0
    if img is not None:
        axes[0].imshow(img)
        axes[0].set_xticks([]); axes[0].set_yticks([])
        axes[0].set_aspect("equal")
        for sp in axes[0].spines.values():
            sp.set_visible(False)
        # the answered word is what every tile in the other panels is scored
        # against, so it stays -- as part of the heading, not under it
        _head(axes[0],
              f"the photograph: the model answers "
              f"\u201c{rec['concept']}\u201d")
        i = 1

    _tiles(axes[i], truth, C, keep, words=None, ring=ring, box=box)
    # the shade, the ring and the dashed outline are read out in the figure's
    # caption; they are what the caption is for
    _head(axes[i], "what covering it really does")
    i += 1

    for nm in order:
        d = rec["lenses"].get(nm)
        ax = axes[i]; i += 1
        if not d:
            ax.axis("off")
            continue
        _tiles(ax, d["z"], C, keep, words=d["word"], ring=ring, box=box, fs=fs)
        r = rho.get(nm)
        # this photograph's own agreement is a number, not a sentence about
        # one, so it rides along in the heading; whether the lens's strongest
        # tile is a ringed one is legible from the panel itself
        _head(ax, nm if r is None
              else f"{nm}: \u03c1 = {r:.2f}".replace("-", "\u2212"))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.92, bottom=0.02,
                        wspace=0.10)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


LENS_COLOUR = {
    "logit lens": style.ORANGE,
    "J-lens": style.VIOLET,
    "VJ-Lens": style.BLUE,
    "VJ-Lens (point)": style.AQUA,
    "VJ-Lens (path)": style.BLUE,
    "VJ-Lens (direction)": style.GREEN,
    "logit lens (centred)": style.MUTED,
}


def summary(df, out, name="f60_causal.png"):
    """Depth against agreement with the causal map, one line per lens.

    The point is a gap that holds across depth, so the lines are labelled at
    their right-hand ends rather than in a legend box, and every lens keeps the
    colour it has everywhere else in the paper.
    """
    style.use()
    lenses = [l for l in ("logit lens", "J-lens", "VJ-Lens (point)", "VJ-Lens")
              if l in set(df.lens)]
    lenses += [l for l in df.lens.unique() if l not in lenses]
    fig, axes = plt.subplots(1, 3, figsize=(19.8, 6.2))
    panels = [("spearman", "agreement with the causal map  (ρ)"),
              ("top3_cause", "names one of the three\nmost damaging regions"),
              ("hit_box", "names a region inside\nthe human's box")]
    for ax, (col, lab) in zip(axes, panels):
        ends = []
        for lens in lenses:
            d = df[df.lens == lens].groupby("depth")[col].mean()
            if not len(d):
                continue
            c = LENS_COLOUR.get(lens, MUTED)
            ax.plot(d.index * 100, d.values, color=c, marker="o", ms=6, lw=2.2)
            ends.append([float(d.index[-1] * 100), float(d.values[-1]), lens, c])
        # two lines that finish at the same value print their names on top of
        # each other; push the labels apart along y, keeping their order
        if ends:
            # the gap is a fraction of the AXIS, not of the spread of the
            # values: four lines that all finish within 0.01 of each other have
            # no spread to take a fraction of, and the names come out on top of
            # one another again
            y0, y1 = ax.get_ylim()
            gap = (y1 - y0) * 0.075
            ends.sort(key=lambda e: e[1])
            for e in ends:
                e.append(e[1])                      # where the line really ends
            for i in range(1, len(ends)):
                if ends[i][1] - ends[i - 1][1] < gap:
                    ends[i][1] = ends[i - 1][1] + gap
            over = ends[-1][1] - (y1 - gap * 0.35)
            if over > 0:
                for e in ends:
                    e[1] -= over
            for x, y, lens, c, y_true in ends:
                style.label_end(ax, x, y, lens, c, y_true=y_true,
                                fontsize=19.6)
        ax.set_xlabel("depth through the language model  (%)")
        ax.set_ylabel(lab)
        # the four names are the widest thing in the panel; the margin is the
        # room they stand in, inside their own panel
        ax.margins(x=0.62)
        style.despine(ax)
    # a heading and the three plots.  What was read where, how the truth was
    # measured, and how the end labels are drawn are all in the caption now,
    # and the strip they used to take goes back to the axes.
    unit = 1.0 / (72.0 * fig.get_figheight())
    fig.text(0.5, 1 - 6 * unit, "Graded against what the model actually does",
             ha="center", va="top", color=INK, fontsize=25)
    top = 1 - (6 + 20 * 1.22 + 12) * unit
    fig.subplots_adjust(left=0.062, right=0.988, top=top, bottom=0.13,
                        wspace=0.46)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


# --------------------------------------------------------------- the edit set
#
# Five operations, five panels, the same two series in every one, so a reader
# learns the pair once and then only reads the shapes.  The control is the same
# formula pointed at a random direction and rescaled to do exactly as much
# damage, which is why it is drawn underneath rather than beside: it is the same
# experiment with the coordinate removed.

OP_ORDER = ["erase", "amplify", "inject", "swap", "swapdev", "transplant"]
#: the five named in the taxonomy, without the swap's second reading
OP_FIVE = ["erase", "amplify", "inject", "swap", "transplant"]

OP_SUB = {
    "erase": "take the source concept's\nown coordinate away",
    "amplify": "scale the target concept's\nown coordinate up",
    "inject": "add the target direction,\nsized by the local norm",
    "swap": "exchange the two concepts'\ncoordinates",
    "swapdev": "exchange the two coordinates'\ndeviations from the baseline",
    "transplant": "copy the target's coordinate\noff a real photograph of it",
}
OP_X = {"erase": "multiples of the coordinate removed",
        "amplify": "gain", "inject": "strength",
        "swap": "gain", "swapdev": "gain", "transplant": "gain on the donor"}
#: which concept's reading the edit is supposed to move, and which way
OP_READS = {"erase": ("read_src_u", "of the source concept"),
            "amplify": ("read_dst_u", "of the target concept"),
            "inject": ("read_dst_u", "of the target concept"),
            "swap": ("read_dst_u", "of the target concept"),
            "swapdev": ("read_dst_u", "of the target concept"),
            "transplant": ("read_dst_u", "of the target concept")}

C_EDIT = style.BLUE
C_RAND = MUTED


def _series(ax, x, mean, sem, colour, lw, ms, alpha=1.0, z=2):
    ax.plot(x, mean, color=colour, lw=lw, marker="o", ms=ms, alpha=alpha,
            zorder=z, clip_on=False)
    if sem is not None:
        ax.errorbar(x, mean, yerr=sem, fmt="none", ecolor=colour, elinewidth=lw * 0.7,
                    capsize=0, alpha=alpha * 0.85, zorder=z, clip_on=False)


def edits(df, out, name="f65_edits.png", ops=None, fs_tick=13.5):
    """Answer and reading against dose, one column per operation.

    Two rows, because an edit has to be shown twice to be believed: once at the
    output, where the model's answer either follows or does not, and once in the
    coordinates the edit was actually made in, where the reading of the edited
    concept either moves or does not.  An operation that moves the answer
    without moving the reading has not demonstrated the coordinate system; an
    operation that moves the reading without moving the answer has demonstrated
    that the coordinate is not load-bearing.  Both failures are visible here.

    Dose is drawn on a categorical axis.  The five operations have strength
    parameters on different scales -- one is a fraction, one a gain of up to
    sixteen -- and putting them on their own numeric axes would make five panels
    that cannot be compared by eye, which is the entire point of the figure.
    """
    style.use()
    ops = [o for o in (ops or OP_ORDER) if o in set(df.op)]
    n = len(ops)
    ed = df[df.arm != "clean"]
    clean = df[df.arm == "clean"]

    # six panels have to be a little narrower or the row runs past any page
    w = 3.85 if n <= 5 else 3.42
    fig, axes = plt.subplots(2, n, figsize=(w * n, 9.9), squeeze=False)
    for j, op in enumerate(ops):
        d = ed[ed.op == op]
        doses = sorted(d.dose.unique())
        x = list(range(len(doses)))
        rcol, rlab = OP_READS.get(op, ("read_dst_u", "of the target concept"))

        # ---- top: does the answer follow?
        ax = axes[0][j]
        ax.axhline(0.0, color=GRID, lw=1.2, zorder=1)
        for arm, colour, lw, ms, al, z in (("random", C_RAND, 1.6, 4.5, 0.85, 2),
                                           ("edit", C_EDIT, 2.4, 7.0, 1.0, 3)):
            g = d[d.arm == arm].groupby("dose").follow
            if not len(g):
                continue
            mu = [float(g.mean().get(v, np.nan)) for v in doses]
            se = [float(g.sem().get(v, np.nan)) for v in doses]
            _series(ax, x, mu, se, colour, lw, ms, alpha=al, z=z)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(-0.35, len(doses) - 0.65)
        ax.set_xticks(x)
        ax.set_xticklabels([_dose_label(v) for v in doses], fontsize=fs_tick)
        ax.set_xlabel(OP_X.get(op, "dose"))
        if j == 0:
            ax.set_ylabel("prefers the target answer")
        else:
            ax.set_yticklabels([])
        e = d[d.arm == "edit"]
        dis = e.groupby("dose").disturb.mean()
        dm = e.groupby("dose").d_margin.mean()
        sub = OP_SUB.get(op, "")
        if len(dis):
            # the per-operation n, damage and margin are the caption's to
            # quote; printed here so it can be written from the same run
            print(f"    {op}: n={int(e.groupby('dose').size().iloc[-1])} "
                  f"top-dose damage {dis.iloc[-1]:.2f} "
                  f"margin {dm.iloc[-1]:+.1f}", flush=True)
        _head(ax, op, sub)
        style.despine(ax)

        # ---- bottom: does the reading move where the edit was made?
        ax = axes[1][j]
        if len(clean) and rcol in clean:
            base = float(clean[rcol].mean())
            ax.axhline(base, color=INK, lw=1.0, linestyle=(0, (2, 3)),
                       zorder=1)
            if j == 0:
                ax.annotate("what this reading is on the\nunedited photograph",
                            xy=(0, base), xytext=(2, 5),
                            xycoords=("axes fraction", "data"),
                            textcoords="offset points", ha="left", va="bottom",
                            color=INK2, fontsize=18.8, linespacing=1.3)
        for arm, colour, lw, ms, al, z in (("random", C_RAND, 1.6, 4.5, 0.85, 2),
                                           ("edit", C_EDIT, 2.4, 7.0, 1.0, 3)):
            g = d[d.arm == arm].groupby("dose")[rcol]
            if not len(g):
                continue
            mu = [float(g.mean().get(v, np.nan)) for v in doses]
            se = [float(g.sem().get(v, np.nan)) for v in doses]
            _series(ax, x, mu, se, colour, lw, ms, alpha=al, z=z)
        ax.set_xlim(-0.35, len(doses) - 0.65)
        ax.set_xticks(x)
        ax.set_xticklabels([_dose_label(v) for v in doses], fontsize=fs_tick)
        ax.set_xlabel(OP_X.get(op, "dose"))
        ax.margins(y=0.22)
        # the baseline is named once, in the first panel, in a band opened for
        # it above the line: printed at the top of the axes as they came, the
        # two lines ran into the heading over the panel
        if j == 0 and base is not None:
            lo_, hi_ = ax.get_ylim()
            need = base + 0.36 * (hi_ - lo_)
            if need > hi_:
                ax.set_ylim(lo_, need)
            ax.annotate("what this reading is on the\nunedited photograph",
                        xy=(0, base), xytext=(2, 6),
                        xycoords=("axes fraction", "data"),
                        textcoords="offset points", ha="left", va="bottom",
                        color=INK2, fontsize=18.8, linespacing=1.3,
                        bbox=dict(facecolor=style.SURFACE, edgecolor="none",
                                  pad=1.0), zorder=6)
        # above the axes, not inside them: the erase curve starts in the top
        # left corner, which is exactly where an inside label would sit
        # over two lines: on one line this heading is wider than the column
        # and prints into the heading beside it
        ax.set_title("lens reading\n" + rlab, loc="center", color=INK2,
                     fontsize=20.3, pad=7)
        if j == 0:
            ax.set_ylabel("reading of that concept,\neach panel on its own scale")
        style.despine(ax)

    # one legend, in the first panel, since every panel carries the same pair
    a0 = axes[0][0]
    a0.plot([], [], color=C_EDIT, lw=2.4, marker="o", ms=7.0,
            label="the lens direction")
    a0.plot([], [], color=C_RAND, lw=1.6, marker="o", ms=4.5,
            label="a random direction,\nsame ||dh||")
    style.legend(a0, loc="upper left")

    word = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six",
            7: "Seven"}.get(n, str(n))
    fig.text(0.5, 0.990, f"{word} edits in one coordinate system",
             ha="center", va="top", color=INK, fontsize=23.5)
    fig.subplots_adjust(left=0.055, right=0.992, top=0.855, bottom=0.068,
                        wspace=0.17, hspace=0.58)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def _dose_label(v: float) -> str:
    return f"{v:g}"


def causal_gallery(cards, order, out, name="f73_causalgallery.png",
                   per_row=6, top_k=3):
    """Many photographs at once, graded against the intervention.

    The naming version of this figure asks whether a lens prints the object's
    name; this one asks whether the region a lens calls strongest is one of the
    regions that actually cost the model most when covered.  Same breadth, a
    ground truth the model produced rather than a person.
    """
    import base64
    import io

    from PIL import Image

    style.use()
    cards = list(cards)[: per_row * 2]
    n = len(cards)
    ncol = min(per_row, n)
    nrow = int(np.ceil(n / ncol))
    # a blank row closes each band.  Without it the second band's photograph
    # titles sit directly under the first band's last row of cells and read as
    # if they belonged to it -- the gap inside a band has to be smaller than
    # the gap between two of them, and one uniform hspace cannot do that.
    gap = 1 if nrow > 1 else 0
    rows = len(order) + 1 + gap
    # the header is one line now, so the canvas gives back the three-quarters of
    # an inch the block under it used to occupy
    fig_h = nrow * (2.05 + 0.42 * len(order)) + 0.30 * gap * max(nrow - 1, 0) + 1.35
    # the closing row of the LAST band is a strip of white at the foot of the
    # figure that nothing sits in, so it is flattened to nothing
    ratios = []
    for b in range(nrow):
        ratios += [2.6] + [0.62] * len(order)
        ratios += [0.55 if b < nrow - 1 else 0.01] * gap
    fig, axes = plt.subplots(nrow * rows, ncol, figsize=(2.1 * ncol, fig_h),
                             gridspec_kw={"height_ratios": ratios})
    axes = np.atleast_2d(axes)
    if gap:
        for b in range(nrow):
            for cc in range(ncol):
                axes[b * rows + rows - 1, cc].axis("off")
    hit = {nm: 0 for nm in order}
    ranks = {nm: [] for nm in order}

    for k, c in enumerate(cards):
        band, col = divmod(k, ncol)
        truth = np.asarray(c["truth"], dtype=float)
        ring = set(np.argsort(-truth)[:top_k].tolist())
        ax = axes[band * rows, col]
        try:
            raw = c["img"].split(",", 1)[1] if c.get("img") else None
            if raw:
                ax.imshow(Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB"))
            elif c.get("path") and os.path.exists(c["path"]):
                im = Image.open(c["path"]).convert("RGB")
                im.thumbnail((300, 300), Image.LANCZOS)
                ax.imshow(im)
            else:
                ax.set_facecolor(PANEL)
        except Exception:
            ax.set_facecolor(PANEL)
        ax.set_xticks([]); ax.set_yticks([])
        # imshow shrinks the axes box to the picture's aspect and centres it,
        # which drops the title of every wide photograph below the title of
        # every tall one; anchored north, the row of names is one line
        ax.set_anchor("N")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(c["concept"], fontsize=19.6, color=INK, pad=4)

        for j, nm in enumerate(order):
            ax = axes[band * rows + 1 + j, col]
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            d = c["lenses"].get(nm)
            if not d:
                continue
            # the aggregate ranks regions by the OBJECT WORD's score, not by the
            # z-scored max over the whole vocabulary; using the wrong one here
            # made this figure disagree with its own table
            vals = np.asarray(d.get("score", d["z"]), dtype=float)
            pick = int(np.argmax(vals))
            rank = int(np.where(np.argsort(-truth) == pick)[0][0]) + 1
            ok = pick in ring
            hit[nm] += int(ok)
            ranks[nm].append(rank)
            ax.add_patch(Rectangle((0.02, 0.10), 0.96, 0.80,
                                   facecolor=SOFT if ok else PANEL,
                                   edgecolor=BLUE if ok else "#dfe4ea",
                                   linewidth=1.2 if ok else 0.8))
            ax.text(0.5, 0.5, f"#{rank}", ha="center", va="center",
                    fontsize=18.8, color=INK if ok else INK2)
            if col == 0:
                ax.text(-0.06, 0.5, nm, ha="right", va="center", fontsize=18.8,
                        color=INK2, transform=ax.transAxes)

    for k in range(n, nrow * ncol):
        band, col = divmod(k, ncol)
        for j in range(rows):
            axes[band * rows + j, col].axis("off")

    head = 0.50 / fig_h
    fig.text(0.5, 1 - head * 0.9, "Which region each lens calls strongest",
             ha="center", va="top", fontsize=25, color=INK)
    # a top-three tally over twelve photographs is a noisy statistic; the rank
    # itself is the continuous one, so its median is printed for the caption to
    # quote rather than drawn into the figure
    print("  median rank over these cards: "
          + ",  ".join(f"{nm} {np.median(ranks[nm]):.0f}" for nm in order
                       if ranks[nm]), flush=True)
    fig.subplots_adjust(left=0.135, right=0.995, top=1 - head * 2.3,
                        bottom=0.012, hspace=0.26, wspace=0.10)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


# ------------------------------------------------- the region-size sweep (f62)
#
# The function that drew f62_region and f63_promptblind was lost when this
# module was rewritten; both drivers (exp/63_regionfig.py, exp/64_blindfig.py)
# survived and still call them, so they are restored here -- heading and plot
# only, with everything the old blocks explained moved into the captions in
# paper/atlas.html.
#
# The reference line the old figure drew was passed on the command line and no
# record of the command survives.  It is not guessed: the shipped PNG's own
# axis was solved from the eight plotted sizes (log-linear, residuals under one
# pixel over 1080 px of axis) and the dashed column falls at 30.03 patches,
# where 30 predicts the column to a fifth of a pixel and the nearest rival
# reading, 29.2, misses it by six.  It is also what COCO gives: the mean
# bounding box of the largest instance of a single-word category, over
# val2017, is 0.1489 of the frame -- 29.2 of these 196 patches.
BOX_PATCHES = 30.0


def regionsweep(rows, n_patches, out, name="f62_region.png",
                box_patches: float | None = None):
    """How well a region's reading predicts covering it, against region size.

    One line per depth over a log axis whose two ends are the two prior lenses:
    a region of one patch is the logit lens's unit, a region of every patch is
    what averaging the transport over positions amounts to.  What that means,
    what the dashed line is and how many photographs are behind each point are
    the caption's job, not the picture's.
    """
    style.use()
    rows = [r for r in rows if r.get("spearman") is not None]
    if not rows:
        raise ValueError("no region-sweep rows")
    P = int(n_patches or 196)
    box = BOX_PATCHES if box_patches is None else float(box_patches)
    sizes = sorted({int(r["size"]) for r in rows})
    layers = sorted({int(r["layer"]) for r in rows})
    n = max(int(r.get("n", 0)) for r in rows)

    # wide enough that the three names under the axis stand apart: at the
    # paper's type size "an average object's box" and "the whole frame" met
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    ends = []
    for j, l in enumerate(layers):
        c = style.SERIES[j % len(style.SERIES)]
        d = {int(r["size"]): float(r["spearman"])
             for r in rows if int(r["layer"]) == l}
        xs = [s for s in sizes if s in d]
        ys = [d[s] for s in xs]
        ax.plot(xs, ys, color=c, marker="o", ms=7, lw=2.4, zorder=3)
        ends.append((xs[-1], ys[-1], f"layer {l}", c))
    for x, y, lab, c in ends:
        style.label_end(ax, x, y, lab, c)

    ax.axvline(box, color=MUTED, lw=1.4, ls=(0, (5, 4)), zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([1, 4, 16, 64, P])
    ax.set_xticklabels(["1", "4", "16", "64", str(P)])
    ax.minorticks_off()
    ax.set_xlim(0.62, P * 1.75)
    ax.set_xlabel("region size, in image patches")
    ax.set_ylabel("reading against true effect  ($\\rho$)")
    style.despine(ax)
    ax.margins(x=0.02)

    # three tick annotations, each naming a place on the axis.  They are labels
    # and stay; the sentences that used to explain them are in the caption.
    for x, txt, ha in ((1, "a single patch", "left"),
                       (box, "an average object's box", "center"),
                       (P, "the whole frame", "right")):
        ax.annotate(txt, xy=(x, 0), xytext=(0, -58),
                    xycoords=("data", "axes fraction"),
                    textcoords="offset points", ha=ha, va="top",
                    color=MUTED, annotation_clip=False,
                    fontsize=plt.rcParams["axes.labelsize"] * 0.84)

    unit = 1.0 / (72.0 * fig.get_figheight())
    fig.text(0.5, 1 - 6 * unit, "Both prior lenses sit at an end of this axis",
             ha="center", va="top", color=INK, fontsize=25)
    top = 1 - (6 + 20 * 1.22 + 10) * unit
    fig.subplots_adjust(left=0.095, right=0.985, top=top, bottom=0.235)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"  region sweep: {len(layers)} depths x {len(sizes)} sizes, "
          f"n = {n} photographs, reference line at {box:g} of {P} patches")
    for j, l in enumerate(layers):
        d = {int(r["size"]): float(r["spearman"])
             for r in rows if int(r["layer"]) == l}
        best = max(d, key=lambda s: d[s])
        print(f"    layer {l}: one patch {d.get(1, float('nan')):+.3f}, "
              f"whole frame {d.get(P, float('nan')):+.3f}, "
              f"best {best} patches {d[best]:+.3f}")
    return p


# ----------------------------------------------- the prompt-blindness figure
#
# Conditions, in the order the panels read them (exp/52_crossmodal.py):
#   plain       no framing at all
#   control     the text asserts a different, harmless object
#   substitute  the text asserts the dangerous one
#   ask         asserts it, then asks for the harmful thing
# set over short lines: rotated, a long line's horizontal reach is most of
# its length, and at the paper's type size the one-line forms printed across
# each other under the axis
_FRAMINGS = [
    ("plain", "just\ndescribe it"),
    ("control", "“it is actually\na bowl”"),
    ("substitute", "“it is actually\na knife”"),
    ("ask", "…and now do\nsomething harmful\nwith it"),
]


def promptblind(df, dj, out, name="f63_promptblind.png",
                layers=(8, 13, 18, 23), word="knife"):
    """The same photographs and the same image tokens, read under four framings.

    Both fixed transports return the identical number in all four -- not close,
    identical -- so they plot as one flat line at zero and the orange covers
    the violet exactly.  The caption says so; the picture only shows it.
    """
    style.use()
    layers = [l for l in layers if f"atlas_{l}" in df.columns]
    if not layers:
        raise ValueError("no atlas_<layer> columns in the crossmodal frame")
    conds = [(k, lab) for k, lab in _FRAMINGS if k in set(df.cond)]
    x = list(range(len(conds)))

    fig, axes = plt.subplots(1, len(layers), figsize=(21.4, 5.4))
    axes = np.atleast_1d(axes)
    spreads = []
    for ax, l in zip(axes, layers):
        piv = df.pivot_table(index="i", columns="cond", values=f"atlas_{l}",
                             aggfunc="first")
        base = piv["plain"]
        ys = [float((piv[k] - base).mean()) for k, _ in conds]
        spreads.append((l, max(ys) - min(ys), int(len(base))))
        # the two fixed transports, drawn violet-then-orange so that the flat
        # line a reader sees is the one the label names first
        for col, colour in ((f"jlens_{l}", style.VIOLET),
                            (f"say_{l}", style.ORANGE)):
            src = dj if (dj is not None and col in dj.columns) else df
            if col not in src.columns:
                continue
            q = src.pivot_table(index="i", columns="cond", values=col,
                                aggfunc="first")
            b = q["plain"]
            zs = [float((q[k] - b).mean()) for k, _ in conds if k in q.columns]
            ax.plot(x[:len(zs)], zs, color=colour, marker="o", ms=6, lw=2.2,
                    zorder=3)
        ax.plot(x, ys, color=style.BLUE, marker="o", ms=8, lw=3.0, zorder=4)
        style.label_end(ax, x[-1], ys[-1], "VJ-Lens", style.BLUE,
                        fontsize=18.8)
        # the label goes UNDER the flat line, not above it: above it is where
        # the VJ-Lens line climbs out of the same corner, and at the shallow
        # depths the two print on top of each other
        ax.annotate("logit lens and J-lens, identical",
                    xy=((len(conds) - 1) / 2.0, 0), xytext=(0, -8),
                    textcoords="offset points", ha="center",
                    va="top", color=INK2,
                    fontsize=plt.rcParams["axes.labelsize"] * 0.82)
        style.panel(ax, f"layer {l}", loc="upper left")
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in conds], rotation=42,
                           ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.35, len(conds) + 0.55)
        hi = max(ys) * 1.16 + 0.004
        ax.set_ylim(min(0, min(ys)) - 0.20 * hi, hi)
        # the band under zero is there to hold the label, not to be read: no
        # series goes negative, so it gets no ticks
        lo_, hi_ = ax.get_ylim()
        ax.set_yticks([t for t in ax.get_yticks() if t >= -1e-9])
        ax.set_ylim(lo_, hi_)
        style.despine(ax)
    axes[0].set_ylabel(f"change in the “{word}” reading\n"
                       "at the image tokens")

    unit = 1.0 / (72.0 * fig.get_figheight())
    fig.text(0.5, 1 - 6 * unit, "Only one of the three can see the question",
             ha="center", va="top", color=INK, fontsize=25)
    top = 1 - (6 + 20 * 1.22 + 10) * unit
    fig.subplots_adjust(left=0.080, right=0.992, top=top, bottom=0.315,
                        wspace=0.32)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    for l, sp, n in spreads:
        print(f"  layer {l}: VJ-Lens spread over the four framings {sp:.2f}, "
              f"n = {n} photographs")
    return p
