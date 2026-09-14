# SPDX-License-Identifier: Apache-2.0
"""The flagship figure: one photograph, two questions, and who moves with them.

Every other figure in this paper makes its case in one panel at a time.  This
one is built so that the LAYOUT does the arguing.  A fixed-transport lens
(the logit lens, the Jacobian lens) reads the residual stream at the image
positions and nothing else, and the image positions cannot see the question --
it arrives after them, behind a causal mask.  So a fixed lens has no way to
print two different maps for two different questions about the same picture.
It gets one map.  This figure gives it one map, spanning both rows, and gives
the two questions their own row only where a real, measured difference exists
to put there.

The two rows are the "easy" and "hard" framings of ``exp/80_absorb.py``'s
question-flip run: the same photograph, first asked what its main object is,
then asked how many of that object it holds.  Both rows carry the OCCLUSION
TRUTH of that run -- the true drop in the asked-about logit, region by region,
measured by covering each region's activations and re-running the model.  That
truth genuinely moves between the two questions (its strongest region changes),
which is the finding this figure exists to put in front of a reader.

The "our reading" row for the *first* question is real, measured VJ-Lens
data, but from a close neighbour of that question rather than the absorb run
itself: ``exp/60_causalmap.py``'s own default prompt ("Describe this image"),
run on the identical photograph at the identical grid and the identical
layers.  Its truth peaks at the same region the absorb run's "easy" truth
does, which is why it stands in for that row.  No VJ-Lens reading exists yet
for the "how many" question -- that would take a new pass, which this figure
does not fabricate a substitute for.  The second row says so in the panel
itself, in words, rather than filling the tile with an invented number.
"""
from __future__ import annotations

import os

import numpy as np

from jvlm import style
from jvlm.style import GRID, INK, INK2, MUTED
from jvlm.figcases import HEAT, RULE

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import warnings

QA_C = style.BLUE
QB_C = style.RED


def _nth(k: int) -> str:
    k = int(k)
    suf = "th" if 11 <= k % 100 <= 13 else {1: "st", 2: "nd",
                                            3: "rd"}.get(k % 10, "th")
    return f"{k}{suf}"


def _map(ax, vals, grid, own=None, true=None, colour=INK, box=(),
        unmeasured=False, unmeasured_note=""):
    """One region map: the shared monochrome ramp, a dashed ring on the
    region occlusion truth calls strongest, a heavy ring in ``colour`` on the
    region this panel itself calls strongest, and a hairline box on the
    object.  ``unmeasured`` draws a hatched blank instead of a fabricated
    reading."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_edgecolor(RULE)
        sp.set_linewidth(0.8)
    if unmeasured:
        ax.set_facecolor("#f2f1ed")
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               fill=False, hatch="////", edgecolor="#d8d6d0",
                               lw=0.0, zorder=1))
        ax.text(0.5, 0.5, unmeasured_note, ha="center", va="center",
                fontsize=13.5, color=MUTED, linespacing=1.4, wrap=True,
                transform=ax.transAxes)
        if true is not None:
            v = np.zeros(grid * grid)
            r, c = divmod(int(true), grid)
            ax.set_xlim(-0.5, grid - 0.5)
            ax.set_ylim(grid - 0.5, -0.5)
            ax.add_patch(Rectangle((c - 0.30, r - 0.30), 0.60, 0.60,
                                   fill=False, lw=1.9, edgecolor="#ffffff",
                                   zorder=9))
            ax.add_patch(Rectangle((c - 0.30, r - 0.30), 0.60, 0.60,
                                   fill=False, lw=1.1, edgecolor=INK2,
                                   ls=(0, (1.8, 1.4)), zorder=10))
        return
    v = np.asarray(vals, float).reshape(grid, grid)
    ax.imshow(v, cmap=HEAT, interpolation="nearest",
             vmin=float(v.min()), vmax=float(v.max()))
    for cell in box:
        r, c = divmod(int(cell), grid)
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                               edgecolor=INK, lw=1.0,
                               linestyle=(0, (3, 2)), zorder=6))
    if own is not None:
        r, c = divmod(int(own), grid)
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, lw=3.8,
                               edgecolor="#ffffff", zorder=7))
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, lw=2.2,
                               edgecolor=colour, zorder=8))
    if true is not None:
        r, c = divmod(int(true), grid)
        ax.add_patch(Rectangle((c - 0.30, r - 0.30), 0.60, 0.60, fill=False,
                               lw=1.9, edgecolor="#ffffff", zorder=9))
        ax.add_patch(Rectangle((c - 0.30, r - 0.30), 0.60, 0.60, fill=False,
                               lw=1.1, edgecolor=INK, ls=(0, (1.8, 1.4)),
                               zorder=10))


def _photo_panel(ax, path, grid, box, rings):
    """The photograph, with each question's strongest region ringed in its
    own colour so a reader can see, before reading a single number, that the
    two questions are pointing somewhere different.  The object's box is
    drawn as ONE rectangle (the box cells' own bounding rectangle) rather
    than one dashed outline per cell -- eighteen abutting dashed cells read
    as a grid laid over the whole photograph, not as a box."""
    from PIL import Image
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        ax.set_facecolor("#eee")
        return
    # aspect="auto" on imshow itself fills the panel without matplotlib
    # re-adjusting the axes' data limits to match the photo's own aspect --
    # ax.set_aspect(H/W) did that, and silently shifted every rectangle drawn
    # afterwards off the cells they were meant to ring
    ax.imshow(im, extent=[0, grid, grid, 0], aspect="auto")
    ax.set_xlim(0, grid)
    ax.set_ylim(grid, 0)
    if box:
        rows = [divmod(int(cell), grid)[0] for cell in box]
        cols = [divmod(int(cell), grid)[1] for cell in box]
        r0, r1, c0, c1 = min(rows), max(rows) + 1, min(cols), max(cols) + 1
        ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False,
                               edgecolor="#ffffff", lw=2.0,
                               linestyle=(0, (4, 3)), alpha=0.9, zorder=5))
    for cell, colour in rings:
        r, c = divmod(int(cell), grid)
        ax.add_patch(Rectangle((c + 0.04, r + 0.04), 0.92, 0.92, fill=False,
                               edgecolor="#ffffff", lw=4.2, zorder=6))
        ax.add_patch(Rectangle((c + 0.04, r + 0.04), 0.92, 0.92, fill=False,
                               edgecolor=colour, lw=2.6, zorder=7))


def _bars(ax, ring_val, med_val, colour, unit="logits"):
    """Two short bars: covering the ringed region against covering the
    median region, both real occlusion measurements.  Two decimals, no
    more."""
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    vals = [ring_val, med_val]
    labels = [f"covering the\nringed region\ncosts {ring_val:.2f}",
              f"covering the\nmedian region\ncosts {med_val:.2f}"]
    colours = [colour, MUTED]
    mx = max(abs(ring_val), abs(med_val), 1e-6)
    y = [0.72, 0.24]
    for yy, v, c, lab in zip(y, vals, colours, labels):
        w = 0.62 * abs(v) / mx
        x0 = 0.03
        ax.add_patch(Rectangle((x0, yy - 0.14), max(w, 0.01), 0.28,
                               facecolor=c, edgecolor="none"))
        ax.text(x0 + max(w, 0.01) + 0.035, yy, lab, ha="left", va="center",
                fontsize=12.6, color=INK, linespacing=1.15)
    ax.set_xlim(0, 1.55)
    ax.set_ylim(0, 1)


def flagship(card, out, name="f134_flagship.png",
            qa_text="Describe this image.",
            qb_text="How many persons are in this photograph?  "
                    "Answer with a number.",
            image_activation_maxdev=0.0):
    """The flagship: one photograph, two questions, one shared colormap.

    ``card`` holds everything real this figure draws:

    ``path, grid, box`` -- the photograph, its region grid, the object's box.
    ``truth_a, truth_b`` -- the two questions' occlusion truth, region by
        region, both measured (``exp/80_absorb.py``'s question-flip run).
    ``read_a`` -- VJ-Lens's own reading, region by region, for the neighbour
        prompt described in ``note_measured_a``.  ``read_b`` is intentionally
        absent: no run has measured it.
    ``fixed`` -- ``{"logit lens": [...], "J-lens": [...]}``, the two
        prompt-blind readings, from the same neighbour-prompt run as
        ``read_a``.  They are drawn once, spanning both rows, because a
        transport that cannot see the question cannot have a second row to
        draw.
    """
    style.use()
    G = card["grid"]
    box = card.get("box", ())
    truth_a = np.asarray(card["truth_a"], float)
    truth_b = np.asarray(card["truth_b"], float)
    read_a = np.asarray(card["read_a"], float)
    argmax_a_truth = int(truth_a.argmax())
    argmax_a_read = int(read_a.argmax())
    argmax_b_truth = int(truth_b.argmax())

    # four rows: the 2x2 grid (2 rows), a short row for the two questions
    # under the photo, and a row for the fixed-lens panel.  Every element
    # below gets its own row -- nothing is placed by guessing an offset in
    # axes-fraction units from a neighbour, which is what let the question
    # text and the fixed-lens caption print on top of each other the first
    # time this figure was drawn.
    fig = plt.figure(figsize=(21.5, 14.6))
    gs = fig.add_gridspec(4, 4, width_ratios=[1.62, 1.0, 1.0, 1.12],
                          height_ratios=[1.0, 1.0, 0.46, 0.92],
                          left=0.012, right=0.99, top=0.865, bottom=0.03,
                          wspace=0.34, hspace=0.10)

    # -------------------------------------------------------------- left
    ax_photo = fig.add_subplot(gs[0:2, 0])
    _photo_panel(ax_photo, card["path"], G, box,
                [(argmax_a_truth, QA_C), (argmax_b_truth, QB_C)])

    ax_q = fig.add_subplot(gs[2, 0])
    ax_q.axis("off")
    ax_q.text(0.0, 0.92, "A", fontsize=18, color=QA_C, ha="left", va="top",
             fontweight="bold", transform=ax_q.transAxes)
    ax_q.text(0.075, 0.92, f"“{qa_text}”", fontsize=15.5, color=INK,
             ha="left", va="top", transform=ax_q.transAxes, linespacing=1.3)
    ax_q.text(0.0, 0.38, "B", fontsize=18, color=QB_C, ha="left", va="top",
             fontweight="bold", transform=ax_q.transAxes)
    ax_q.text(0.075, 0.38, f"“{qb_text}”", fontsize=15.5, color=INK,
             ha="left", va="top", transform=ax_q.transAxes, linespacing=1.3)

    # ------------------------------------------------------- centre 2x2
    ax_ra = fig.add_subplot(gs[0, 1])
    ax_ta = fig.add_subplot(gs[0, 2])
    ax_rb = fig.add_subplot(gs[1, 1])
    ax_tb = fig.add_subplot(gs[1, 2])

    _map(ax_ra, read_a, G, own=argmax_a_read, true=argmax_a_truth,
        colour=QA_C, box=box)
    _map(ax_ta, truth_a, G, own=argmax_a_truth, colour=QA_C, box=box)
    _map(ax_rb, None, G, true=argmax_b_truth, unmeasured=True,
        unmeasured_note="no VJ-Lens\nreading measured\nfor this question")
    _map(ax_tb, truth_b, G, own=argmax_b_truth, colour=QB_C, box=box)

    for ax, lab, c in ((ax_ra, "A", QA_C), (ax_rb, "B", QB_C)):
        ax.text(-0.10, 0.5, lab, transform=ax.transAxes, fontsize=19,
               color=c, ha="right", va="center", fontweight="bold")
    ax_ra.set_title("our reading", fontsize=18.5, color=INK, pad=9)
    ax_ta.set_title("occlusion truth", fontsize=18.5, color=INK, pad=9)

    # -------------------------------------------------------------- right
    ax_bar_a = fig.add_subplot(gs[0, 3])
    ax_bar_b = fig.add_subplot(gs[1, 3])
    med_a = float(np.median(truth_a))
    med_b = float(np.median(truth_b))
    _bars(ax_bar_a, float(truth_a[argmax_a_truth]), med_a, QA_C)
    _bars(ax_bar_b, float(truth_b[argmax_b_truth]), med_b, QB_C)

    # ------------------------------------------------------------- bottom
    ax_cap = fig.add_subplot(gs[2, 1:3])
    ax_cap.axis("off")
    ax_cap.text(0.5, 0.72, "identical for both questions", ha="center",
               va="center", fontsize=18, color=INK,
               transform=ax_cap.transAxes)
    ax_cap.text(0.5, 0.14,
               "a fixed transport reads the image positions and only the "
               "image positions; measured max |activation(A) minus "
               f"activation(B)| = {image_activation_maxdev:.2f}",
               ha="center", va="center", fontsize=12.8, color=MUTED,
               transform=ax_cap.transAxes)

    fixed = card["fixed"]
    names = list(fixed.keys())
    fixed_gs = gs[3, 1:3].subgridspec(1, len(names), wspace=0.22)
    for i, nm in enumerate(names):
        ax = fig.add_subplot(fixed_gs[0, i])
        v = np.asarray(fixed[nm], float)
        _map(ax, v, G, own=int(v.argmax()), true=argmax_a_truth,
            colour=MUTED, box=box)
        ax.set_title(nm, fontsize=16.5, color=INK2, pad=6)

    if style.FIGURE_TITLES:
        fig.text(0.5, 0.975,
             "Two questions, one picture: only a lens with a question "
             "coordinate can move",
             ha="center", va="top", fontsize=25, color=INK)
    if style.FIGURE_TITLES:
        fig.text(0.5, 0.925,
             "same photograph, same grid, same layer: the occlusion "
             "truth moves between the two questions, and where it has been "
             "measured, so does VJ-Lens's own reading",
             ha="center", va="top", fontsize=14.5, color=INK2)

    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p
