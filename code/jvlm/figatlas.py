# SPDX-License-Identifier: Apache-2.0
"""Four figures the atlas was missing: the theorem, the dose, the price, the strip.

Kept out of :mod:`jvlm.figcausal` deliberately.  That module is the side-by-side
card and the depth summary; these four read different files and share nothing
with it but its layout rules, which are restated at the top of this file rather
than imported, so that editing either module cannot silently reshape figures in
the other.

  ``completeness``  the central theorem, drawn: how well the readings recover
                    the intervention they claim to decompose, against how many
                    points of the path were evaluated.
  ``doses``         one panel per edit operation, the answer against the dose,
                    with the matched random direction underneath.
  ``efficiency``    the same runs in the plane that matters -- damage against
                    effect -- so "the least invasive edit is the most
                    effective" is a shape rather than a sentence.
  ``layerstrip``    one photograph, one region, every layer, one row per lens.
"""
from __future__ import annotations

import os

from jvlm import style
from jvlm.style import GRID, INK, INK2, MUTED

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------- layout keys
#
# The three rules that stopped labels colliding in figcausal, restated here so
# this module does not depend on private names in a file being edited:
#   * a title's pad is computed from how many lines its subtitle has;
#   * a word never shares a place with another word or with a mark: where two
#     would land together, one of them is moved and the move is in the code;
#   * the ends of an axis are named below the axis, never at the end of a line;
#   * text wears an ink colour and identity is carried by the mark beside it.

HOT = "#b8481f"


def _head(ax, title, sub, sub_colour=None, fs_title=12.5, fs=9.5,
          note=None, note_colour=None, min_lines=0):
    """Panel title, a quieter second block, and an optional third in its own ink.

    Two blocks rather than one so a finding can be printed in the alarm colour
    without dragging the operation's own description into it.  The pad is
    computed from the total number of lines in both, which is the rule that
    stopped a title from landing on its own subtitle.
    """
    n_sub = str(sub).count("\n") + 1 if sub else 0
    n_note = str(note).count("\n") + 1 if note else 0
    step = fs * 1.35
    pad = 10 + max(n_sub + n_note, min_lines) * step
    ax.set_title(title, loc="left", color=INK, pad=pad, fontsize=fs_title)
    if note:
        ax.annotate(note, xy=(0, 1), xytext=(0, 7), xycoords="axes fraction",
                    textcoords="offset points", ha="left", va="bottom",
                    color=note_colour or INK2, fontsize=fs, linespacing=1.35)
    if sub:
        ax.annotate(sub, xy=(0, 1), xytext=(0, 7 + n_note * step),
                    xycoords="axes fraction", textcoords="offset points",
                    ha="left", va="bottom", color=sub_colour or INK2,
                    fontsize=fs, linespacing=1.35)


def _figure_head(fig, main, sub, x=0.006, fs=17.0, fss=11.5, gap=7.0,
                 clear=11.0):
    """Figure title and a quieter block under it; returns where the plot may start.

    The same rule the panel titles use, one level up: the space the header
    takes is computed from the type sizes and from how many lines the subtitle
    actually has.  A fixed pair of y positions is what printed the first line
    of the subtitle on top of the title in the first version of every one of
    these figures.
    """
    H = fig.get_figheight()
    unit = 1.0 / (72.0 * H)                       # one point, in figure fraction
    y_main = 1.0 - 6.0 * unit
    y_sub = y_main - (fs * 1.22 + gap) * unit
    fig.text(x, y_main, main, ha="left", va="top", color=INK, fontsize=fs)
    fig.text(x, y_sub, sub, ha="left", va="top", color=INK2, fontsize=fss,
             linespacing=1.42)
    lines = str(sub).count("\n") + 1
    return y_sub - lines * fss * 1.42 * unit - clear * unit


def _end_label(ax, x, y, text, dx=7, dy=0, fs=None, colour=None, y_true=None,
               marker="o"):
    """A name at the end of a series: ink text, with the series' own mark beside it.

    The house rule is that text wears an ink colour and identity is carried by
    the mark.  There has to *be* a mark: where two lines finish close together
    the names are pushed apart along y, and a stack of grey names beside a
    bundle of coloured lines cannot be read back onto them.  So a dot in the
    series hue is drawn at the name's left and, when the name has been moved
    off its line, a hairline leads back to where the line really ends.

    (:func:`jvlm.style.label_end` grew the same behaviour while this was being
    written; it is repeated here rather than imported because this module is
    rendered on a machine whose copy of the style module may be older.)
    """
    from matplotlib.transforms import offset_copy

    fs = fs or plt.rcParams["axes.labelsize"] * 0.88
    if colour is not None:
        if y_true is not None and y_true != y:
            ax.plot([x, x], [y_true, y], color=colour, lw=0.8, alpha=0.35,
                    ls=(0, (1, 2)), zorder=1, clip_on=False)
        tr = offset_copy(ax.transData, fig=ax.figure, x=dx, y=dy,
                         units="points")
        ax.plot([x], [y], marker=marker, color=colour, ms=5.5, mew=0, zorder=5,
                clip_on=False, transform=tr)
        dx = dx + 8.5
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                va="center", ha="left", color=INK2, fontsize=fs)


def _panel_top(fig, head_top, lines, fs=9.5, fs_title=12.5, clear=8.0):
    """Leave the row of panel heads exactly the room its tallest member needs."""
    unit = 1.0 / (72.0 * fig.get_figheight())
    return head_top - (10 + lines * fs * 1.35 + fs_title * 1.25 + clear) * unit


def _save(fig, out, name, dpi=200):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p


# =========================================================== B. completeness
#
# The paper's one theorem is that the readings sum to the intervention.  A
# theorem about an integral is worth what its discretisation is worth, and that
# is a measured number with a knob on it -- how many points of the path were
# evaluated -- so it draws as a curve rather than a claim.

DEPTH_COLOUR = [style.BLUE, style.ORANGE, style.AQUA, style.MAGENTA]


def completeness(stats: dict, out: str, name="f66_complete.png"):
    rows = stats.get("by_step_layer") or []
    if not rows:
        return None
    n_layers = int(stats.get("n_layers") or 1)
    steps = sorted({int(r["steps"]) for r in rows})
    layers = sorted({int(r["layer"]) for r in rows})
    x = list(range(len(steps)))

    style.use()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    panels = [("corr", "predicted change against true change,\n"
                       "across the whole vocabulary  (r)", (None, 1.02), 1.0),
              ("rel_error", "what is left over\n"
                            "|predicted − true| / |true|", (0.0, None), None)]
    for ax, (col, lab, ylim, ceil) in zip(axes, panels):
        ends = []
        for k, l in enumerate(layers):
            d = {int(r["steps"]): r.get(col) for r in rows
                 if int(r["layer"]) == l}
            y = [d.get(s) for s in steps]
            if all(v is None for v in y):
                continue
            c = DEPTH_COLOUR[k % len(DEPTH_COLOUR)]
            ax.plot(x, y, color=c, marker="o", ms=6.5, lw=2.2, clip_on=False,
                    zorder=3)
            ends.append([float(y[-1]),
                         f"{100 * l / max(n_layers - 1, 1):.0f}% deep",
                         c, float(y[-1])])
        ax.set_xticks(x)
        # the ends of the axis are named under the axis, which is the same
        # rule the region sweep uses -- a name at the end of a line lands on
        # the line's own label
        ax.set_xticklabels(
            [f"{s}\nplain gradient" if s == steps[0]
             else f"{s}\nthe paper's path" if s == steps[-1]
             else str(s) for s in steps])
        ax.set_ylabel(lab)
        ax.set_xlabel("points evaluated on the path")
        ax.set_xlim(-0.22, len(steps) - 1 + 0.10)
        ax.margins(x=0.30, y=0.16)
        if ylim[0] is not None:
            ax.set_ylim(bottom=ylim[0])
        if ylim[1] is not None:
            ax.set_ylim(top=ylim[1])
        style.despine(ax)
        # The names go on last, once the axis is final.  The gap that keeps two
        # of them apart is a fraction of the AXIS, not of the spread of the
        # values being labelled -- three lines that all finish at 0.98 have no
        # spread to take a fraction of -- and if the stack has grown past the
        # top, it slides back down whole, because a name printed above the
        # "exact" line reads as a correlation greater than one.
        if ends:
            y0, y1 = ax.get_ylim()
            gap = (y1 - y0) * 0.065
            ends.sort(key=lambda e: e[0])
            for i in range(1, len(ends)):
                if ends[i][0] - ends[i - 1][0] < gap:
                    ends[i][0] = ends[i - 1][0] + gap
            over = ends[-1][0] - (min(y1, ceil if ceil is not None else y1)
                                  - gap * 0.30)
            if over > 0:
                for e in ends:
                    e[0] -= over
            for yy, txt, c, y_true in ends:
                _end_label(ax, x[-1], yy, txt, colour=c, y_true=y_true)

    axes[0].axhline(1.0, color="#c6c5c0", lw=1.1, ls=(0, (3, 3)), zorder=1)
    axes[0].annotate("exact", xy=(0.0, 1.0), xytext=(2, 4),
                     textcoords="offset points", ha="left", va="bottom",
                     color=MUTED,
                     fontsize=plt.rcParams["axes.labelsize"] * 0.82)
    axes[1].axhline(0.0, color=GRID, lw=1.2, zorder=1)

    n = stats.get("n", "")
    top = _figure_head(
        fig, "The readings add up to the intervention they decompose",
        "Every image position's contribution to a word is read, the "
        "contributions are summed, and then the whole image is actually "
        "replaced by the baseline\nto see what the word's logit really does.  "
        f"{n} photographs, three depths, a vocabulary of everyday nouns per "
        "photograph, so the closure is checked for\nmany words at once rather "
        "than for a chosen one.  Neither prior lens can be given this figure: "
        "neither produces a quantity in the units of a logit change.")
    fig.subplots_adjust(left=0.085, right=0.900, top=top, bottom=0.205,
                        wspace=0.40)
    return _save(fig, out, name)


# ================================================================= C. the dose
OP_FIVE = ["erase", "amplify", "inject", "swap", "transplant"]
#: the taxonomy as the paper's table prints it, the swap's second reading
#: included; the figures take this so a panel cannot go missing from beside a
#: row of the table
OP_ALL = ["erase", "amplify", "inject", "swap", "swapdev", "transplant"]
#: the swap's second reading is a mouthful and the table has already named it
OP_DISPLAY = {"swapdev": "swap, against the baseline"}
OP_SUB = {
    "erase": "take the source concept's own\ncoordinate away",
    "amplify": "scale the target concept's own\ncoordinate up",
    "inject": "add the target direction, sized\nby the local norm",
    "swap": "exchange the two concepts'\ncoordinates",
    "swapdev": "exchange the two coordinates'\ndeviations from the baseline",
    "transplant": "copy the target's coordinate off\na real photograph of it",
}
# erase's dose multiplies the coordinate that is taken out: at 1 the whole
# coordinate goes, at 8 eight times it goes and the coordinate overshoots to
# -7x.  "fraction removed" reads as an impossible fraction of eight, and the
# same axis in the six-panel version already says it the long way.
OP_X = {"swapdev": "gain",
        "erase": "multiples of the coordinate removed",
        "amplify": "gain", "inject": "strength",
        "swap": "gain", "swapdev": "gain", "transplant": "gain on the donor"}
C_EDIT = style.BLUE
C_RAND = MUTED


def _by_op(stats: dict):
    """``{op: {arm: [(dose, row), ...]}}`` from the run's ``by_op`` block."""
    out: dict = {}
    for k, v in (stats.get("by_op") or {}).items():
        op, arm, dose = k.split("|")
        out.setdefault(op, {}).setdefault(arm, []).append((float(dose), v))
    for op in out:
        for arm in out[op]:
            out[op][arm].sort(key=lambda t: t[0])
    return out


def _g(row, key, default=float("nan")):
    v = row.get(key, default)
    return v if isinstance(v, (int, float)) and v == v else default


def doses(stats: dict, out: str, name="f67_doses.png", ops=None):
    by = _by_op(stats)
    ops = [o for o in (ops or OP_ALL) if o in by]
    if not ops:
        return None
    td = stats.get("top_dose") or {}
    n = stats.get("n_trials")

    style.use()
    # six panels have to be a little narrower, or the row runs past any page
    wide = len(ops) <= 5
    pw, fs_title = (3.62, 12.5) if wide else (3.28, 11.5)
    fig, axes = plt.subplots(1, len(ops), figsize=(pw * len(ops), 5.6),
                             squeeze=False)
    # The runs did not all cover all five operations, so the sample size is
    # per operation and is printed under each name, exactly as the taxonomy
    # table in the paper assembles it.
    op_n = stats.get("op_n") or {}
    heads = {}
    for op in ops:
        top_ = td.get(op) or {}
        dead_ = _g(top_, "follow", 0.0) <= _g(top_, "follow_random", 0.0)
        # description, damage-and-sample line, and a verdict line where there
        # is one
        heads[op] = (OP_SUB.get(op, "").count("\n") + 1) + 1 + int(dead_)
    n_lines = max(heads.values())
    for j, op in enumerate(ops):
        ax = axes[0][j]
        arms = by[op]
        ds = [d for d, _ in arms.get("edit", [])]
        x = list(range(len(ds)))
        ax.axhline(0.0, color=GRID, lw=1.2, zorder=1)
        for arm, colour, lw, ms, al, z in (("random", C_RAND, 1.6, 4.5, .9, 2),
                                           ("edit", C_EDIT, 2.5, 7.5, 1., 3)):
            pts = dict(arms.get(arm, []))
            y = [_g(pts.get(d, {}), "follow") for d in ds]
            if all(v != v for v in y):
                continue
            ax.plot(x, y, color=colour, lw=lw, marker="o", ms=ms, alpha=al,
                    zorder=z, clip_on=False)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(-0.30, len(ds) - 0.70)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d:g}" for d in ds])
        ax.set_xlabel(OP_X.get(op, "dose"))
        if j == 0:
            ax.set_ylabel("the model prefers the answer\nthe edit asked for")
        else:
            ax.set_yticklabels([])
        dis = [_g(r, "disturb") for _, r in arms.get("edit", [])]
        dis = [v for v in dis if v == v]
        sub = OP_SUB.get(op, "")
        nn = op_n.get(op) or n
        if dis:
            sub += (f"\ndamage {min(dis):.3f} → {max(dis):.3f} over the doses"
                    + (f",  {nn} trials" if nn else ""))
        top = td.get(op) or {}
        # The one thing worth printing in the alarm colour is an operation the
        # matched random direction already matches; everything else is a number
        # a reader can weigh off the curve.  The verdict is scoped to the doses
        # actually drawn, because one of these operations has been run at a
        # wider range elsewhere and does move the answer at the top of it, so
        # an unqualified "does nothing" is a claim this panel cannot support.
        dead = _g(top, "follow", 0.0) <= _g(top, "follow_random", 0.0)
        _head(ax, OP_DISPLAY.get(op, op), sub, min_lines=n_lines,
              fs_title=fs_title,
              note=("at no dose shown better than a random direction"
                    if dead else None),
              note_colour=HOT)
        style.despine(ax)

    a0 = axes[0][0]
    a0.plot([], [], color=C_EDIT, lw=2.5, marker="o", ms=7.5,
            label="the lens direction")
    a0.plot([], [], color=C_RAND, lw=1.6, marker="o", ms=4.5,
            label="a random direction,\nthe same ‖Δh‖")
    style.legend(a0, loc="upper left")

    word = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}.get(
        len(ops), str(len(ops)))
    n_dose = max(len(v.get("edit", [])) for v in by.values())
    dword = {2: "two", 3: "three", 4: "four", 5: "five"}.get(n_dose, str(n_dose))
    top = _figure_head(
        fig, f"{word} edits, {dword} doses each, one coordinate system",
        "Each photograph carries a source concept and a question only the "
        "picture settles; the target concept is one whose answer differs, and "
        "the model never sees\neither name.  Every trial starts with the model "
        "answering correctly, so the unedited rate is exactly zero and the "
        "whole curve is the edit's doing.  The\ncontrol is the same formula "
        "pointed at a random direction and rescaled to do the same measured "
        "damage.  Each panel states the damage its own dose range does, and "
        "the number of trials behind it.")
    fig.subplots_adjust(left=0.072, right=0.992,
                        top=_panel_top(fig, top, n_lines),
                        bottom=0.115, wspace=0.16)
    return _save(fig, out, name)


# =========================================================== D. the efficiency
#
# The dose panels show that two of the five work.  They cannot show what those
# two cost, because the cost is on no axis in them.  Here it is the x axis, and
# the claim "the least invasive edit is the most effective" stops being a
# sentence and becomes the direction the points lie in.

OP_MARK = {"erase": "s", "amplify": "^", "inject": "o", "swap": "D",
           "swapdev": "P", "transplant": "v"}
#: five slots, and the yellow one left out: it is the one hue in the ramp that
#: does not hold a thin line against this paper's background
OP_COLOUR = {"erase": style.BLUE, "amplify": style.ORANGE,
             "inject": style.AQUA, "swap": style.VIOLET,
             "swapdev": style.GREEN, "transplant": style.MAGENTA}


def efficiency(stats: dict, out: str, name="f68_efficiency.png", ops=None):
    by = _by_op(stats)
    ops = [o for o in (ops or OP_ALL) if o in by]
    if not ops:
        return None
    td = stats.get("top_dose") or {}
    op_n = stats.get("op_n") or {}
    same_n = len({op_n.get(op) for op in ops if op_n.get(op)}) <= 1
    n = stats.get("n_trials")

    style.use()
    fig, ax = plt.subplots(figsize=(10.2, 6.6))
    ends, xs = [], []
    for k, op in enumerate(ops):
        pts = [(_g(r, "disturb"), _g(r, "follow"))
               for _, r in by[op].get("edit", [])]
        pts = [(a, b) for a, b in pts if a == a and b == b and a > 0]
        if not pts:
            continue
        c = OP_COLOUR.get(op, style.SERIES[k % len(style.SERIES)])
        xv = [p[0] for p in pts]
        yv = [p[1] for p in pts]
        xs += xv
        ax.plot(xv, yv, color=c, lw=1.9, alpha=0.9, zorder=3,
                marker=OP_MARK.get(op, "o"), ms=6.5, mfc=style.SURFACE,
                mew=1.7, clip_on=False)
        ax.plot(xv[-1:], yv[-1:], color=c, marker=OP_MARK.get(op, "o"),
                ms=10.5, mfc=c, mew=0, zorder=4, clip_on=False)
        # the name, the effect per unit of damage, and the operation's own
        # sample size -- which is not the same for all of them, so it belongs
        # on the mark rather than in a footnote
        eff = _g((td.get(op) or {}), "follow_per_disturb")
        nn = op_n.get(op) or n
        nm = OP_DISPLAY.get(op, op)
        lab = nm if eff != eff else f"{nm}   {eff:.0f}"
        if nn and not same_n:
            lab += f"   (n {nn})"
        ends.append([yv[-1], xv[-1], lab, c, OP_MARK.get(op, "o"), yv[-1]])
        rp = [(_g(r, "disturb"), _g(r, "follow"))
              for _, r in by[op].get("random", [])]
        rp = [(a, b) for a, b in rp if a == a and b == b and a > 0]
        if rp:
            ax.plot([p[0] for p in rp], [p[1] for p in rp], color=C_RAND,
                    lw=0, marker="x", ms=5.0, alpha=0.6, zorder=2,
                    clip_on=False)
    ax.set_xscale("log")
    ax.axhline(0.0, color=GRID, lw=1.2, zorder=1)
    ax.set_ylim(-0.04, 1.04)
    # room on the right for a column of names: five lines finishing at five
    # different x cannot all be labelled where they end without one of them
    # landing on another's markers, so they are labelled in a column and led
    # to it by a hairline
    lab_x = max(xs) * 3.0 if xs else 1.0
    if xs:
        ax.set_xlim(min(xs) / 1.9, lab_x * 1.12)
    # the same push-apart rule the depth labels use
    if ends:
        ends.sort(key=lambda e: e[0])
        ends[0][0] = max(ends[0][0], 0.045)
        for i in range(1, len(ends)):
            if ends[i][0] - ends[i - 1][0] < 0.082:
                ends[i][0] = ends[i - 1][0] + 0.082
        for yy, xx, lab, c, mk, y_true in ends:
            ax.plot([xx * 1.10, lab_x * 0.93], [y_true, yy], color=c, lw=0.8,
                    alpha=0.30, ls=(0, (1, 2)), zorder=1, clip_on=False)
            _end_label(ax, lab_x, yy, lab, dx=0, colour=c, marker=mk)
    ax.plot([], [], color=C_RAND, lw=0, marker="x", ms=5.5,
            label="the matched random direction, at the same damage")
    ax.plot([], [], color=MUTED, lw=1.9, marker="o", ms=6.5,
            mfc=style.SURFACE, mew=1.7,
            label="one operation, its four doses;\n"
                  "the filled marker is the top dose")
    style.legend(ax, loc="upper left")
    ax.set_xlabel("damage done to the residual stream at the edited positions"
                  "   ‖Δh‖ / ‖h‖   (log)")
    ax.set_ylabel("the model prefers the answer the edit asked for")
    style.despine(ax)

    top = _figure_head(
        fig, "What each edit costs for what it buys",
        "Up and to the left is better: the answer follows and the residual "
        "stream is barely touched.  Inject and transplant — the two operations\n"
        "that write a coordinate in — carry the answer past seventy percent at "
        "two to four percent of damage; the exchange read against the\n"
        "baseline gets to fifty-six, at nine.  The other three stay under a "
        "third, and reach even that only at the far end of their own dose\n"
        "range.  The number beside each name is that operation's effect per "
        "unit of damage at its own top dose"
        + (f", over {n} trials." if same_n and n
           else ", and the count after it is that operation's own n."))
    fig.subplots_adjust(left=0.090, right=0.775, top=top, bottom=0.135)
    return _save(fig, out, name)


# =========================================================== A. the layer strip
#
# One photograph, one region -- the object's own tiles -- and every layer.  Each
# cell is the word that region is reading at that depth, shaded by where the
# object's own name sits in the vocabulary there.  The word is printed where it
# CHANGES and holds until the next label, because twenty-four words to a row at
# a readable size is a figure two feet wide, and because where a lens changes
# its mind is the thing worth seeing.

#: light to dark, one hue: a magnitude, so shade compares to shade
#: The light end is deliberately darker than the page: a row whose object never
#: ranks near the top is a real reading and has to look like one, not like a
#: band the renderer forgot to fill.
RANK_RAMP = ["#e7eff6", "#d6e6f1", "#bfd9ec", "#9ec8e1", "#74aed4",
             "#4b91c3", "#2871aa", "#125089"]
#: where the object's own name sits, binned; the bins are printed as a key so
#: the reader is never asked to guess what a shade means
RANK_BINS = [(0, "1st"), (2, "2–3"), (5, "4–6"), (10, "7–11"), (20, "12–21"),
             (40, "22–41"), (80, "42–81"), (10 ** 9, "82+")]


def _rank_shade(rank):
    if rank is None or rank != rank:
        return "#ffffff"
    for i, (hi, _) in enumerate(RANK_BINS):
        if rank <= hi:
            return RANK_RAMP[len(RANK_RAMP) - 1 - i]
    return RANK_RAMP[0]


def _rank_ink(rank):
    if rank is None or rank != rank:
        return MUTED
    return "#ffffff" if rank <= 5 else INK


def _runs(words):
    """[(start, stop, word)] -- consecutive columns reading the same word."""
    out, i = [], 0
    while i < len(words):
        j = i
        while j + 1 < len(words) and words[j + 1] == words[i]:
            j += 1
        out.append((i, j + 1, words[i]))
        i = j + 1
    return out


def layerstrip(card, order, out, name="f69_strip.png", n_layers=None,
               display=None, img=None, per_layer=0.455, row_h=0.86,
               gutter=1.34, fs_word=9.0):
    """One photograph, three lenses, every layer.

    ``order`` names the rows in the order they are drawn; ``display`` may
    rename them for the figure without renaming them in the result file.

    The whole layout is computed in inches and then converted, because the one
    thing that must be true of this figure is that the row names, the
    photograph and the strip do not sit on top of each other, and a layout in
    figure fractions changes all three whenever the number of layers changes.
    """
    display = display or {}
    lenses = [l for l in order if l in card["lenses"]]
    if not lenses:
        return None
    layers = sorted(int(k) for k in card["lenses"][lenses[0]])
    L = len(layers)
    R = len(lenses)
    concept = card["concept"]
    n_layers = n_layers or (max(layers) + 2)

    style.use()
    rows_in = row_h * R
    photo_in = max(rows_in, 2.1)
    left_in, right_in = 0.10, 0.14
    strip_in = per_layer * L
    below_in, above_in = 1.16, 1.98
    W = left_in + photo_in + gutter + strip_in + right_in
    H = photo_in + below_in + above_in
    fig = plt.figure(figsize=(W, H))

    ax_l = (left_in + photo_in + gutter) / W
    ax_w = strip_in / W
    ax_b = below_in / H
    ax_h = rows_in / H
    ax = fig.add_axes([ax_l, ax_b + (photo_in - rows_in) / (2 * H), ax_w, ax_h])
    ax.set_xlim(0, L)
    ax.set_ylim(R, 0)
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # how many characters a cell holds at this size, so a word is clipped to
    # the run it labels rather than allowed to run into the next word
    chars_per_cell = max(int(per_layer / (fs_word * 0.0069)), 1)

    firsts = {}
    for r, lens in enumerate(lenses):
        per = card["lenses"][lens]
        words = [per.get(str(l), {}).get("box_word") for l in layers]
        ranks = [per.get(str(l), {}).get("box_rank") for l in layers]
        for j, rk in enumerate(ranks):
            ax.add_patch(Rectangle((j + .004, r + .05), .992, .90,
                                   facecolor=_rank_shade(rk),
                                   edgecolor="none", zorder=2))
        hit = [j for j, rk in enumerate(ranks) if rk == 0]
        firsts[lens] = layers[hit[0]] if hit else None
        mark = hit[0] if hit else None
        if mark is not None:
            ax.add_patch(Rectangle((mark + .03, r + .08), .94, .84,
                                   fill=False, edgecolor=HOT, lw=2.0,
                                   zorder=6))
        for i0, i1, w in _runs(words):
            if w is None:
                continue
            # the ring marks one cell and the word is centred over a run; when
            # the two land in the same place the word is moved to whichever
            # side of the ring has more room, rather than printed through it
            a0, a1 = i0, i1
            if mark is not None and i0 <= mark < i1 and (i1 - i0) > 1:
                if mark - i0 >= i1 - mark - 1:
                    a1 = mark
                else:
                    a0 = mark + 1
            room = chars_per_cell * (a1 - a0) - 1
            txt = w if len(w) <= room else (w[: max(room - 1, 1)] + "\u2026")
            mid = int((a0 + a1 - 1) / 2)
            ax.text((a0 + a1) / 2, r + .52, txt, ha="center", va="center",
                    fontsize=fs_word, zorder=5, color=_rank_ink(ranks[mid]))
            if i0:
                ax.plot([i0, i0], [r + .05, r + .95], color=style.SURFACE,
                        lw=3.0, zorder=3, solid_capstyle="butt")
    ax.set_yticks([r + 0.5 for r in range(R)])
    ax.set_yticklabels([display.get(l, l) for l in lenses], fontsize=12.0,
                       color=INK)
    ax.tick_params(axis="y", length=0, pad=9)

    # ---- under the strip: which layer, and what the two ends of it are
    step = 2 if L > 14 else 1
    for j, l in enumerate(layers):
        if j % step:
            continue
        ax.annotate(str(l), xy=(j + 0.5, R), xytext=(0, -6),
                    textcoords="offset points", ha="center", va="top",
                    color=INK2, fontsize=9.5, annotation_clip=False)
    for xpos, ha, txt in ((0.0, "left", "just after the picture is read in"),
                          (float(L), "right", "just before the answer")):
        ax.annotate(txt, xy=(xpos, R), xytext=(0, -25),
                    textcoords="offset points", ha=ha, va="top", color=MUTED,
                    fontsize=9.5, annotation_clip=False)
    # every other figure in this paper puts depth on its x axis as a
    # percentage, so the two ends are given in both units here and a reader can
    # carry a layer number between figures
    d0 = 100 * layers[0] / max(n_layers - 1, 1)
    d1 = 100 * layers[-1] / max(n_layers - 1, 1)
    ax.annotate(f"layer of the language model   (layer {layers[0]} is "
                f"{d0:.0f}% of the way through, layer {layers[-1]} is "
                f"{d1:.0f}%)",
                xy=(L / 2, R), xytext=(0, -47), textcoords="offset points",
                ha="center", va="top", color=INK, fontsize=12.5,
                annotation_clip=False)

    # ---- the photograph, with the human's box on it
    ax_p = fig.add_axes([left_in / W, ax_b, photo_in / W, photo_in / H])
    if img is not None:
        ax_p.imshow(img)
        for bx in card.get("boxes", []):
            x0, y0, w0, h0 = bx
            ax_p.add_patch(Rectangle((x0 * img.width, y0 * img.height),
                                     w0 * img.width, h0 * img.height,
                                     fill=False, edgecolor=HOT, lw=2.0))
    ax_p.set_xticks([]); ax_p.set_yticks([])
    ax_p.set_aspect("equal")
    for sp in ax_p.spines.values():
        sp.set_visible(False)
    fig.text(left_in / W, ax_b - 0.020,
             f"the model answers \u201c{card.get('caption', concept)}\u201d",
             ha="left", va="top", color=INK2, fontsize=10.0)

    # ---- the key, so a shade is never a guess
    kh = 0.26 / H
    ky = ax_b + (photo_in + rows_in) / (2 * H) + 0.30 / H
    kw = 0.52 / W
    fig.text(ax_l, ky + kh + 0.13 / H,
             f"where \u201c{concept}\u201d itself ranks in the vocabulary at "
             "that region", ha="left", va="bottom", color=INK2, fontsize=10.0)
    for i, (_, lab) in enumerate(RANK_BINS):
        fig.patches.append(Rectangle(
            (ax_l + i * kw * 1.05, ky), kw, kh,
            facecolor=RANK_RAMP[len(RANK_BINS) - 1 - i], edgecolor="none",
            transform=fig.transFigure, figure=fig))
        fig.text(ax_l + i * kw * 1.05 + kw / 2, ky - 0.05 / H, lab,
                 ha="center", va="top", color=MUTED, fontsize=8.8)
    bx = ax_l + len(RANK_BINS) * kw * 1.05 + 0.30 / W
    fig.patches.append(Rectangle((bx, ky), 0.34 / W, kh, fill=False,
                                 edgecolor=HOT, lw=2.0,
                                 transform=fig.transFigure, figure=fig))
    fig.text(bx + 0.46 / W, ky + kh / 2,
             "the first layer at which it wins outright", ha="left",
             va="center", color=INK2, fontsize=9.5)

    bits = []
    for l in lenses:
        nm = display.get(l, l)
        bits.append(f"{nm} at layer {firsts[l]}" if firsts[l] is not None
                    else f"{nm} never")
    _figure_head(
        fig, "One photograph, one region, layer by layer",
        "The region is the object's own tiles; the word in a cell is what that "
        "region is reading at that depth, against the rest of the same "
        "photograph.\nA word is printed where it changes and holds until the "
        "next one.  The object's own name first wins outright: "
        + "; ".join(bits) + ".",
        fs=16.0, fss=11.0)
    return _save(fig, out, name)
