# SPDX-License-Identifier: Apache-2.0
"""A gallery of worked cases, all in one layout, so a reader learns it once.

Every figure in this module is a *table of single trials*.  One row is one real
item, and the row always reads left to right in the same six columns:

    the photograph | what was asked, and what came back |
    logit lens | J-lens | VJ-Lens | the truth

Nothing here is an average and nothing here is typed by hand.  Each row is
assembled by :mod:`exp/73_cases` out of a result file that already exists, and
the provenance line on the row names the file's own identifiers -- the COCO
image id, the benchmark item id, the layer -- so a reader who wants to check a
number can go and find it.

Four conventions are load-bearing and were arrived at the hard way.

* **One layout, three figures.**  The first three cards teach the row; the other
  twenty are read at a glance.  Splitting the same twenty-three cases into eight
  bespoke figures costs the reader eight layouts and buys nothing.
* **Text wears ink; identity is carried by the mark beside it.**  Each lens
  panel opens with a dot in that lens's fixed colour, repeated down every row,
  because a reader scanning rows cannot hold a column header in their head for
  twenty rows.  The verdict is written out in words underneath rather than
  encoded in a tick, because the eight result files define "right" eight
  different ways and a single glyph would be lying about six of them.
* **Nothing is wrapped by counting characters, and no row has a fixed
  height.**  Every string on the sheet is folded to its own column's width in
  inches, measured in the sheet's own serif at the size it will actually be
  set (:func:`_tw`), and each row is then made as tall as the tallest thing in
  it.  The previous version wrapped to a character count and gave every row a
  height typed into the card: raising the shared type scale left the question
  text running through the lens columns and the rows printing on top of one
  another.  A card's height is now a floor -- how much room its photograph
  wants -- and never a ceiling.
* **Figure width is chosen from the type size, not the other way round.**
  These render into a 66rem column, so a figure W inches wide shows F-point
  text at roughly ``14.7 F / W`` screen pixels.  Six columns of the present
  type need ``PAGE_W`` inches, which puts the smallest type on the sheet at
  about eleven and a half pixels; narrower and the question column folds to
  four words a line.

The map panels are drawn on one monochrome ramp -- pale at a region's smallest
value, ink at its largest -- rather than in the lens hues, because a blue-to-
orange heat map inside a column headed "VJ-Lens" reads as the lens's colour
saying something it is not.  Each panel is scaled to its own range; the claim a
map panel makes is *where its largest region is*, and that is marked twice: a
hairline ring on the region the occlusion truth says is strongest, and a
heavier ring in the panel's own colour on the region the panel itself calls
strongest.  Where a lens is right the two rings sit on the same cell.
"""
from __future__ import annotations

import base64
import io
import os
import re
import textwrap

import numpy as np

from jvlm import style
from jvlm.style import (AQUA, BLUE, GRID, INK, INK2, MUTED, ORANGE, SURFACE,
                        VIOLET)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

#: the paper's fixed map, repeated here so this module can be read alone
LENS_COLOUR = {"logit lens": ORANGE, "J-lens": VIOLET, "VJ-Lens": BLUE,
               "VJ-Lens (point)": AQUA, "the truth": INK}

#: the three lens columns, in the order every row prints them
LENSES = ("logit lens", "J-lens", "VJ-Lens")
COLUMNS = LENSES + ("the truth",)

#: the band behind the truth column, so the ruler is visibly not a lens
SOFT = "#f1f0ec"
#: a region map goes from the page colour at its smallest cell to ink at its
#: largest.  Monochrome on purpose -- see the module docstring.
HEAT = LinearSegmentedColormap.from_list("cases", ["#f6f5f2", "#c9c7c0",
                                                   "#6f6d67", "#171614"])

# ------------------------------------------------------------------ the type
# One place for every size on the sheet, so the geometry below can be checked
# against it.  Nothing here is derived from anything: these are the sizes the
# paper's shared scale asks for, and the columns are sized to hold them.
TITLE = 23.5     # the sheet's heading
HEAD = 19.6      # the column names
EYE = 18.8       # the eyebrow, the provenance and the field labels
ASK = 20.3       # the question, and the model's own words
BIG = 18.0       # a lens's value
SMALL = 13.0     # the sentence under a lens's value
LEAD = 1.30      # line spacing, everywhere
IND = 0.165      # the indent that clears a lens dot, in inches
MAP_MAX = 1.50   # the widest a region map is drawn, in inches

# vertical furniture, in points
PAD_T = 11.0     # above a row's eyebrow
PAD_B = 9.0      # below a row's deepest column
GAP = 8.0        # between the eyebrow line and the row proper
GUTTER = 0.40    # inches of clear air between eyebrow and provenance

#: Six columns of the type above need this much width.  See the docstring.
PAGE_W = 16.64

# column geometry, in inches from the left edge of the figure.  Each range is
# the column's real measure: everything drawn in it is folded to that width.
COLX = {
    "photo": (0.00, 2.00),
    "text": (2.24, 6.80),
    "logit lens": (7.06, 9.20),
    "J-lens": (9.44, 11.58),
    "VJ-Lens": (11.82, 13.96),
    "the truth": (14.24, 16.64),
}
RULE = "#d8d6d0"


def _w(key):
    """A column's width in inches."""
    x0, x1 = COLX[key]
    return x1 - x0


# ------------------------------------------------------------ measured type

_MEAS: dict = {}
_MFIG: list = [None]


def _mfig():
    """A scratch figure used only to ask the font how wide a string is.

    Widths come back in inches, so neither this figure's size nor its dpi
    reaches the answer.  It exists because the only honest way to wrap text to
    a column is to measure it in the face and at the size it will be set in;
    a character count is a guess that was wrong by a third when the shared type
    scale moved.
    """
    if _MFIG[0] is None:
        _MFIG[0] = plt.figure(figsize=(1.0, 1.0))
    return _MFIG[0]


def _tw(s, fs):
    """Width of ``s`` in inches, in the sheet's serif at ``fs`` points."""
    if not s:
        return 0.0
    key = (s, round(float(fs), 2))
    if key not in _MEAS:
        fig = _mfig()
        t = fig.text(0.0, 0.0, s, fontsize=fs)
        _MEAS[key] = (t.get_window_extent(renderer=fig.canvas.get_renderer())
                      .width / fig.dpi)
        t.remove()
    return _MEAS[key]


def _fold(text, w_in, fs):
    """Fold ``text`` to ``w_in`` inches at ``fs`` points, and return the lines.

    Greedy, measured, and it keeps the double space after a full stop that the
    cards are written with.  The author's own newlines are kept as breaks.
    """
    out = []
    for para in str(text).split("\n"):
        bits = re.split(r"(\s+)", para.strip())
        cur = ""
        for k in range(0, len(bits), 2):
            word = bits[k]
            if not word:
                continue
            sep = bits[k - 1] if k else ""
            trial = (cur + sep + word) if cur else word
            if cur and _tw(trial, fs) > w_in:
                out.append(cur)
                cur = word
            else:
                cur = trial
        out.append(cur)
    return out or [""]


def _lines_h(lines, fs):
    """The height of a folded block, in points."""
    return len(lines) * fs * LEAD


def _u(fig):
    """One point, in figure-fraction units of height."""
    return 1.0 / (72.0 * fig.get_figheight())


def _fx(fig, x_in):
    return x_in / fig.get_figwidth()


def _block(fig, x_in, y, lines, fs, colour=INK, ha="left", **kw):
    """Draw an already-folded block and return the y just under it."""
    fig.text(_fx(fig, x_in), y, "\n".join(lines), ha=ha, va="top",
             fontsize=fs, color=colour, linespacing=LEAD, **kw)
    return y - _lines_h(lines, fs) * _u(fig)


# ------------------------------------------------------------------ furniture

def _title(fig, main, pad=8.0, gap=9.0, clear=15.0):
    """The figure's heading, folded to the figure's own width, and nothing
    under it.

    Deliberately not imported from :mod:`jvlm.figworked`, which owns a function
    of the same shape.  That module belongs to another line of work and its
    signature changed underneath this one mid-render; a figure module that
    reaches into another module's private helpers is hostage to it.  A few
    lines of duplication is the cheaper side of that trade.

    No sheet in this module carries a second block under the heading: what the
    sheet is measuring, how it was selected and what each mark means all live
    in the figure's <figcaption> in paper/atlas.html.  With no second block the
    only space reserved is the heading's own, which is what keeps a stripped
    sheet from opening a white band between the title and the column names.

    Measured in points against the figure's own height, so one call gives the
    same visual gap on a short sheet and a tall one -- a fraction of the figure
    height does not.
    """
    unit = _u(fig)
    if not style.FIGURE_TITLES:
        return 1 - pad * unit          # the caption names the sheet
    lines = _fold(main, fig.get_figwidth() * 0.94, TITLE)
    y = 1 - pad * unit
    fig.text(0.5, y, "\n".join(lines), ha="center", va="top", color=INK,
             fontsize=TITLE, linespacing=1.24)
    y -= (len(lines) * TITLE * 1.24 + gap) * unit
    return y - clear * unit


def _title_h(main, pad=8.0, gap=9.0, clear=15.0):
    """Inches the heading needs, so the rows are not squeezed into it."""
    if not style.FIGURE_TITLES:
        return pad / 72.0              # reserve nothing where nothing is drawn
    n = len(_fold(main, PAGE_W * 0.94, TITLE))
    return (pad + n * TITLE * 1.24 + gap + clear) / 72.0


def _load(spec, square=None):
    """A photograph, from a path or from a data URI stored in a result file."""
    from PIL import Image

    if isinstance(spec, str) and spec.startswith("data:"):
        raw = base64.b64decode(spec.split(",", 1)[1])
        im = Image.open(io.BytesIO(raw)).convert("RGB")
    else:
        im = Image.open(spec).convert("RGB")
    if square:
        im = im.resize((square, square), Image.LANCZOS)
    return im


def _thumb(fig, x_in, y_top, h_in, spec, boxes=None, w_in=None):
    """The photograph, top-anchored in its column, with the box drawn on it.

    Boxes are fractions of the frame -- which is what makes them safe to draw
    on a picture that has been squared for the model, and what makes them line
    up with the token grid.
    """
    from PIL import Image

    w_in = w_in or _w("photo")
    im = _load(spec)
    im.thumbnail((900, 900), Image.LANCZOS)
    ar = im.width / im.height
    ww, hh = (w_in, w_in / ar) if w_in / ar <= h_in else (h_in * ar, h_in)
    ax = fig.add_axes([_fx(fig, x_in), y_top - hh / fig.get_figheight(),
                       ww / fig.get_figwidth(), hh / fig.get_figheight()])
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for bx in (boxes or []):
        x, y, w, h = bx
        ax.add_patch(Rectangle((x * im.width, y * im.height), w * im.width,
                               h * im.height, fill=False, lw=1.5,
                               edgecolor="#ffffff", zorder=4))
        ax.add_patch(Rectangle((x * im.width, y * im.height), w * im.width,
                               h * im.height, fill=False, lw=0.9,
                               edgecolor=INK, ls=(0, (3, 2)), zorder=5))
    return ax


def _map_panel(fig, x_in, y_top, side_in, vals, grid, own=None, true=None,
               colour=INK):
    """A ``grid x grid`` region map, scaled to its own range.

    ``own`` is the region this panel calls strongest and is ringed in the
    panel's colour; ``true`` is the region the occlusion truth calls strongest
    and is ringed as a hairline in ink, in every panel, so the reader can see at
    a glance whether the two coincide.  Both rings are drawn over a white halo,
    because the cell a ring lands on is usually the darkest one in the panel and
    an ink ring on near-black is not a ring.

    The object's bounding box used to be outlined here too and is not any more.
    It made three marks on a panel an inch wide, and its warm outline sat one
    column away from a lens whose fixed colour is orange.  Whether the box is
    where the cause is has its own figure; this one is about where the maps
    point.
    """
    v = np.asarray(vals, float).reshape(grid, grid)
    ax = fig.add_axes([_fx(fig, x_in), y_top - side_in / fig.get_figheight(),
                       side_in / fig.get_figwidth(),
                       side_in / fig.get_figheight()])
    ax.imshow(v, cmap=HEAT, interpolation="nearest",
              vmin=float(v.min()), vmax=float(v.max()))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_edgecolor(RULE)
        sp.set_linewidth(0.7)
    if own is not None:
        r, c = divmod(int(own), grid)
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, lw=3.6,
                               edgecolor="#ffffff", zorder=7))
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, lw=2.0,
                               edgecolor=colour, zorder=8))
    if true is not None:
        # drawn last, and inside the other ring.  Drawn first it disappeared
        # under the halo of the panel's own ring in exactly the cases the
        # figure exists to show -- the ones where the two coincide.
        r, c = divmod(int(true), grid)
        ax.add_patch(Rectangle((c - 0.30, r - 0.30), 0.60, 0.60, fill=False,
                               lw=1.9, edgecolor="#ffffff", zorder=9))
        ax.add_patch(Rectangle((c - 0.30, r - 0.30), 0.60, 0.60, fill=False,
                               lw=1.0, edgecolor=INK, ls=(0, (1.8, 1.4)),
                               zorder=10))
    return ax


def _dot(fig, x_in, y, colour, ms=5.2, hollow=False):
    """The lens's mark, repeated on every row.

    ``fig.add_artist`` rather than appending to ``fig.lines``: a Line2D pushed
    onto that list is never told which figure it belongs to, and the tight
    bounding box then asks it for a dpi it does not have.
    """
    fig.add_artist(plt.Line2D(
        [_fx(fig, x_in)], [y], marker="o", ms=ms,
        mfc=SURFACE if hollow else colour, mec=colour,
        mew=1.3 if hollow else 0, transform=fig.transFigure, zorder=6))


# ------------------------------------------------------------------- the panel

def _panel_plan(name, p, w_in):
    """One lens column of one row, folded to that column and measured.

    Returns everything the draw pass needs plus ``h``, the panel's height in
    points, which is what makes the row tall enough to hold it.
    """
    kind = p.get("kind", "word")
    q = {"kind": kind, "colour": LENS_COLOUR[name]}
    if kind == "none":
        q["small"] = _fold(p["small"], w_in, SMALL)
        q["h"] = 1.0 + _lines_h(q["small"], SMALL)
        return q
    if kind == "map":
        q["side"] = side = min(w_in * 0.72, MAP_MAX)
        q["map"], q["grid"] = p["map"], p["grid"]
        q["own"], q["true"] = p.get("own"), p.get("true")
        q["small"] = _fold(p["small"], w_in, SMALL)
        q["h"] = 2.0 + side * 72.0 + 7.0 + _lines_h(q["small"], SMALL)
        return q
    # a word or a number, big enough to read across the row, and no bigger
    # than the column.  A value like "banana" <-> "broccoli" at the default
    # size ran under the truth column's band; shrinking the few that are long
    # keeps every other value at one size.
    fs = float(p.get("fs", BIG))
    avail = w_in - IND
    wide = _tw(p["big"], fs)
    if wide > avail:
        fs = max(SMALL, fs * avail / wide)
    q["big"], q["fs"] = p["big"], fs
    q["small"] = _fold(p["small"], avail, SMALL) if p.get("small") else []
    q["h"] = fs * 1.24 + 3.0 + _lines_h(q["small"], SMALL)
    return q


def _panel_draw(fig, x_in, y, q):
    u = _u(fig)
    if q["kind"] == "none":
        _block(fig, x_in, y - 1.0 * u, q["small"], SMALL, MUTED)
        return
    if q["kind"] == "map":
        side = q["side"]
        _map_panel(fig, x_in, y + 2.0 * u, side, q["map"], q["grid"],
                   own=q["own"], true=q["true"], colour=q["colour"])
        yb = y + 2.0 * u - side / fig.get_figheight() - 7.0 * u
        _block(fig, x_in, yb, q["small"], SMALL, INK2)
        return
    fs = q["fs"]
    _dot(fig, x_in + 0.058, y - fs * 0.40 * u, q["colour"])
    fig.text(_fx(fig, x_in + IND), y, q["big"], ha="left", va="top",
             fontsize=fs, color=INK)
    yb = y - (fs * 1.24 + 3.0) * u
    if q["small"]:
        _block(fig, x_in + IND, yb, q["small"], SMALL, INK2)


# --------------------------------------------------------------------- the row

def _plan(c):
    """Fold one card into its columns and work out how tall the row must be.

    Every measurement the draw pass needs is settled here, before the figure
    exists, because the figure's height is the sum of these.
    """
    tw = _w("text")
    prov = str(c.get("where") or "").strip()
    pl = {"prov": prov}
    # The eyebrow and the provenance share the row's top line: the eyebrow runs
    # from the text column, the provenance is set against the right edge, and
    # the eyebrow is folded to whatever is left over rather than to the text
    # column alone -- an eyebrow is one short phrase and looks broken split
    # across two lines when there is a hand of empty inches beside it.  If what
    # is left over is too narrow to say anything in, the provenance drops to a
    # line of its own instead.  Letting it trail on regardless is what printed
    # the dataset line through a column head and through a lens's value.
    room = COLX["the truth"][1] - _tw(prov, EYE) - GUTTER - COLX["text"][0]
    pl["own_line"] = bool(prov) and room < min(tw, _tw(c["kind"], EYE))
    pl["eyebrow"] = _fold(c["kind"], tw if pl["own_line"] else room, EYE)
    pl["head_lines"] = len(pl["eyebrow"]) + (1 if pl["own_line"] else 0)

    pl["ask"] = _fold("“" + c["ask"] + "”", tw, ASK)
    pl["back"] = [(_fold(lab, tw, EYE), _fold(val, tw, ASK),
                   rest[0] if rest else INK)
                  for lab, val, *rest in c.get("back", [])]
    pl["panels"] = {n: _panel_plan(n, c["panels"][n], _w(n))
                    for n in COLUMNS if c["panels"].get(n)}

    body = _lines_h(pl["ask"], ASK)
    for lab, val, _col in pl["back"]:
        body += 7.0 + _lines_h(lab, EYE) + 1.0 + _lines_h(val, ASK)
    body = max([body] + [q["h"] for q in pl["panels"].values()])

    need = (PAD_T + pl["head_lines"] * EYE * LEAD + GAP + body + PAD_B) / 72.0
    # the card's own height is a floor -- how much room its photograph wants --
    # and never a ceiling.
    pl["h"] = max(float(c.get("h", 1.72)) * 1.26, need)
    return pl


def _row(fig, y_top, c, pl, last=False):
    """One case: the photograph, the ask, the three lenses, the truth."""
    u = _u(fig)
    h_in = pl["h"]
    x = COLX["text"][0]
    y = y_top - PAD_T * u

    # -- the eyebrow, and the provenance beside it or under it
    _block(fig, x, y, pl["eyebrow"], EYE, MUTED)
    if pl["prov"]:
        yp = y - (_lines_h(pl["eyebrow"], EYE) * u if pl["own_line"] else 0.0)
        fig.text(_fx(fig, COLX["the truth"][1]), yp, pl["prov"], ha="right",
                 va="top", fontsize=EYE, color=MUTED)
    y -= pl["head_lines"] * EYE * LEAD * u + GAP * u

    # -- the photograph, in INCHES, with the row's own pads above and below.
    # An earlier version subtracted a figure-fraction pad from an inch height
    # and let every portrait photograph run through the rule under it.
    if c.get("photo") is not None:
        _thumb(fig, COLX["photo"][0], y_top - PAD_T * u,
               h_in - (PAD_T + PAD_B) / 72.0, c["photo"],
               boxes=c.get("boxes"), w_in=_w("photo"))
        if c.get("photo_note"):
            fig.text(_fx(fig, COLX["photo"][0]),
                     y_top - h_in / fig.get_figheight() + PAD_B * u,
                     c["photo_note"], ha="left", va="bottom", fontsize=EYE,
                     color=MUTED)

    # -- what was asked, and what came back
    yt = _block(fig, x, y, pl["ask"], ASK, INK)
    for lab, val, col in pl["back"]:
        yt = _block(fig, x, yt - 7.0 * u, lab, EYE, MUTED)
        yt = _block(fig, x, yt - 1.0 * u, val, ASK, col)

    # -- the four verdict columns
    for name, q in pl["panels"].items():
        _panel_draw(fig, COLX[name][0], y, q)

    if not last:
        yb = y_top - h_in / fig.get_figheight()
        fig.add_artist(plt.Line2D(
            [_fx(fig, COLX["photo"][0]), _fx(fig, COLX["the truth"][1])],
            [yb, yb], color=GRID, lw=0.9, transform=fig.transFigure,
            zorder=1))


# ------------------------------------------------------------------ the header

def _header_lines():
    """The column names, each folded to its own column."""
    out = {"photo": _fold("the photograph", _w("photo"), HEAD),
           "text": _fold("what was asked, and what came back", _w("text"),
                         HEAD)}
    for name in COLUMNS:
        ind = 0.0 if name == "the truth" else IND
        out[name] = _fold(name, _w(name) - ind, HEAD)
    return out


def _header_h():
    """Inches the header strip needs, name block plus the rule under it."""
    n = max(len(v) for v in _header_lines().values())
    return (n * HEAD * LEAD + 10.0) / 72.0


def _header(fig, y_top):
    """The column names, printed once, with the lens dots the rows repeat."""
    u = _u(fig)
    lines = _header_lines()
    n = max(len(v) for v in lines.values())
    _block(fig, COLX["photo"][0], y_top, lines["photo"], HEAD, INK2)
    _block(fig, COLX["text"][0], y_top, lines["text"], HEAD, INK2)
    for name in COLUMNS:
        ind = 0.0 if name == "the truth" else IND
        if name != "the truth":
            _dot(fig, COLX[name][0] + 0.058, y_top - HEAD * 0.40 * u,
                 LENS_COLOUR[name])
        _block(fig, COLX[name][0] + ind, y_top, lines[name], HEAD,
               INK if name == "the truth" else INK2)
    y = y_top - (n * HEAD * LEAD + 10.0) * u
    fig.add_artist(plt.Line2D(
        [_fx(fig, COLX["photo"][0]), _fx(fig, COLX["the truth"][1])], [y, y],
        color="#b9b7b0", lw=1.0, transform=fig.transFigure, zorder=1))
    return y


# ------------------------------------------------------------------ the sheet

def sheet(cards, out, name, title, note=None, row_h=None, w=PAGE_W, dpi=190):
    """One gallery figure: a heading, a header strip, and one row per case.

    ``note`` is what the sheet used to print under its title.  It is not drawn:
    it is echoed to the run log so whoever writes the <figcaption> in
    paper/atlas.html is quoting the same run that drew the plot, and the
    caption is where it belongs.
    """
    style.use()
    if note:
        print("  caption note for " + name + ":\n    "
              + textwrap.fill(" ".join(str(note).split()), 84)
              .replace("\n", "\n    "))
    plans = [_plan(c) for c in cards]
    fh = sum(p["h"] for p in plans) + _header_h() + _title_h(title)
    fig = plt.figure(figsize=(w, fh))
    y = _header(fig, _title(fig, title))
    y_first = y
    for k, (c, pl) in enumerate(zip(cards, plans)):
        _row(fig, y, c, pl, last=k == len(cards) - 1)
        y -= pl["h"] / fig.get_figheight()
    # the band behind the truth column, drawn under everything and stopping
    # where the rows stop.  Run to the foot of the figure it reads as a column
    # of empty grey under the last case.
    x0, x1 = COLX["the truth"]
    fig.add_artist(Rectangle(
        (_fx(fig, x0 - 0.16), y), _fx(fig, x1 - x0 + 0.16),
        y_first - y, facecolor=SOFT, edgecolor="none",
        transform=fig.transFigure, zorder=0))
    fig.add_artist(plt.Line2D(
        [_fx(fig, COLX["photo"][0]), _fx(fig, COLX["the truth"][1])], [y, y],
        color="#b9b7b0", lw=1.0, transform=fig.transFigure, zorder=1))
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    return p
