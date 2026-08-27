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


def _tiles(ax, vals, C, keep, words=None, ring=(), box=(), fs=8.0):
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


def _head(ax, title, sub, sub_colour=None):
    """Panel title with a quieter second line, both above the axes, no overlap.

    The pad is computed from how many lines the subtitle actually has.  A fixed
    pad is what put the title on top of its own subtitle in the first version.
    """
    fs = 9.5
    lines = str(sub).count("\n") + 1
    pad = 10 + lines * fs * 1.35
    ax.set_title(title, loc="left", color=INK, pad=pad, fontsize=12.5)
    ax.annotate(sub, xy=(0, 1), xytext=(0, 7), xycoords="axes fraction",
                textcoords="offset points", ha="left", va="bottom",
                color=sub_colour or INK2, fontsize=fs, linespacing=1.35)


def card(rec, img, order, rho, out, name, top_k: int = 3, fs: float = 8.0):
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
    fig, axes = plt.subplots(1, n, figsize=(4.7 * n, 5.6))
    axes = np.atleast_1d(axes)
    i = 0
    if img is not None:
        axes[0].imshow(img)
        axes[0].set_xticks([]); axes[0].set_yticks([])
        axes[0].set_aspect("equal")
        for sp in axes[0].spines.values():
            sp.set_visible(False)
        _head(axes[0], "the photograph",
              f"the model answers \u201c{rec['concept']}\u201d")
        i = 1

    _tiles(axes[i], truth, C, keep, words=None, ring=ring, box=box)
    # the dashed outline was drawn on all four panels and named on none of
    # them; a figure has to carry its own key, because a figure travels
    _head(axes[i], "what covering it really does",
          "darker = the word falls further\n"
          "ringed = the three that matter most, repeated on every panel\n"
          "dashed = the human's box for the word")
    i += 1

    for nm in order:
        d = rec["lenses"].get(nm)
        ax = axes[i]; i += 1
        if not d:
            ax.axis("off")
            continue
        _tiles(ax, d["z"], C, keep, words=d["word"], ring=ring, box=box, fs=fs)
        r = rho.get(nm)
        hit = int(np.argmax(d["z"])) in ring
        sub = (f"agrees with the truth at \u03c1 = {r:.2f}\n" if r is not None
               else "")
        sub += ("its strongest tile is one of the three"
                if hit else "its strongest tile is none of the three")
        _head(ax, nm, sub, sub_colour=(INK2 if hit else HOT))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.78, bottom=0.03,
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
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.9))
    panels = [("spearman", "agreement with the causal map  (ρ)"),
              ("top3_cause", "names one of the three most damaging regions"),
              ("hit_box", "names a region inside the human's box")]
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
                style.label_end(ax, x, y, lens, c, y_true=y_true)
        ax.set_xlabel("depth through the language model  (%)")
        ax.set_ylabel(lab)
        ax.margins(x=0.34)
        style.despine(ax)
    sub = ("Every lens read at the same regions of the same photographs; the "
           "truth is measured by covering each region in turn.\nEach name "
           "sits beside a dot in its own line's colour — where two lines "
           "finish together the names are moved apart, and a hairline leads "
           "back.")
    fs = 10.5
    unit = 1.0 / (72.0 * fig.get_figheight())
    fig.text(0.006, 1 - 6 * unit, "Graded against what the model actually does",
             ha="left", va="top", color=INK, fontsize=15)
    y_sub = 1 - (6 + 15 * 1.22 + 7) * unit
    fig.text(0.006, y_sub, sub, ha="left", va="top", color=INK2, fontsize=fs,
             linespacing=1.4)
    top = y_sub - (sub.count("\n") + 1) * fs * 1.4 * unit - 12 * unit
    fig.subplots_adjust(left=0.065, right=0.985, top=top, bottom=0.13,
                        wspace=0.36)
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


def edits(df, out, name="f65_edits.png", ops=None, fs_tick=11.0):
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
    fig, axes = plt.subplots(2, n, figsize=(w * n, 9.1), squeeze=False)
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
            # n is printed per panel because these panels are not always filled
            # from one run, and a reader should never have to ask
            sub += (f"\nn = {int(e.groupby('dose').size().iloc[-1])}"
                    f" \u00b7 at the top dose: damage {dis.iloc[-1]:.3f}, "
                    f"margin {dm.iloc[-1]:+.1f}")
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
                            color=INK2, fontsize=9.0, linespacing=1.3)
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
        # above the axes, not inside them: the erase curve starts in the top
        # left corner, which is exactly where an inside label would sit
        ax.set_title("lens reading " + rlab, loc="left", color=INK2,
                     fontsize=11.0, pad=7)
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

    fs = 11.5
    word = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six",
            7: "Seven"}.get(n, str(n))
    fig.text(0.006, 0.988, f"{word} edits in one coordinate system",
             ha="left", va="top", color=INK, fontsize=17)
    fig.text(0.006, 0.955,
             "Each photograph contains a source concept and is asked a short "
             "question only the picture settles; the target concept is one whose "
             "answer differs.\nAll " + word.lower() +
             " operations are aimed at the same goal \u2014 "
             "make the model give the target's answer \u2014 at the same layers, the "
             "same image positions, and, against the\ncontrol, the same measured "
             "damage.  Every trial starts with the model answering correctly, so "
             "the unedited rate is exactly zero.  Under each name: the damage "
             "\u2016dh\u2016/\u2016h\u2016\nat the edited positions, and how far "
             "the edit moved the target answer's logit lead, both at the top dose.",
             ha="left", va="top", color=INK2, fontsize=fs, linespacing=1.45)
    fig.subplots_adjust(left=0.055, right=0.992, top=0.755, bottom=0.072,
                        wspace=0.17, hspace=0.40)
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
    fig_h = nrow * (2.05 + 0.42 * len(order)) + 0.30 * gap * max(nrow - 1, 0) + 2.1
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
        ax.set_title(c["concept"], fontsize=11, color=INK, pad=4)

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
                    fontsize=9.5, color=INK if ok else INK2)
            if col == 0:
                ax.text(-0.06, 0.5, nm, ha="right", va="center", fontsize=9.5,
                        color=INK2, transform=ax.transAxes)

    for k in range(n, nrow * ncol):
        band, col = divmod(k, ncol)
        for j in range(rows):
            axes[band * rows + j, col].axis("off")

    head = 0.42 / fig_h
    fig.text(0.012, 1 - head * 0.9,
             "Which region each lens calls strongest, and where that region "
             "really ranks", ha="left", va="top", fontsize=15, color=INK)
    # a top-three tally over twelve photographs is a noisy statistic; the rank
    # itself is the continuous one, so the header reports its median
    tally = "      ".join(
        f"{nm}: median rank {np.median(ranks[nm]):.0f}" for nm in order
        if ranks[nm])
    sub = ("The number in each cell is where the region that lens called "
           "strongest really ranks, out of 49, and a cell is tinted\nwhen it "
           "is one of the three that matter most.  Rank 1 is the region whose "
           "covering costs the model most.\n" + tally)
    fig.text(0.012, 1 - head * 2.05, sub, ha="left", va="top", fontsize=11,
             color=INK2, linespacing=1.5)
    fig.subplots_adjust(left=0.135, right=0.995, top=1 - head * 4.1,
                        bottom=0.012, hspace=0.26, wspace=0.10)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p
