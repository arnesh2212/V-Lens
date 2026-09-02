# SPDX-License-Identifier: Apache-2.0
"""The V-Lens flow chart: one prompt, one layer stack, one score, three moves.

Bottom to top: the prompt splits into the picture, the words and the answer
spot; three labelled layers (0, l, L) carry their states upward, with a single
ellipsis standing in for the layers skipped between them; the score is a
differentiable scalar of the final answer-spot state. The read maps layer l's
answer spot directly up to layer L's. The write is the score's own gradient,
leaving the score box and landing on each group of positions at layer l,
where a position's reading is the write direction dotted into its centred
state. The check swaps in another item's state at layer l and compares the
network's real, measured change in the score against what the read alone
would have predicted.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from jvlm import plot

plot.use()

BLUE, ORANGE, AQUA = plot.SERIES[0], plot.SERIES[1], plot.SERIES[2]
INK, INK2, MUTED, AXIS = plot.INK, plot.INK2, plot.MUTED, plot.AXIS
TXT = dict(zorder=6)

# ---- geometry: roughly 1 data unit == 1 inch, figure is 10 x 5.5 -----------
PIC_X = (2.7, 4.9)
WORD_X = (5.0, 6.3)
ANS_X = (6.4, 7.3)
STACK_X0, STACK_X1 = PIC_X[0], ANS_X[1]

PROMPT_Y = (0.15, 0.75)
LAYER0_Y = (1.05, 1.55)
LAYERL_Y = (1.8, 2.3)
LAYERLAST_Y = (2.6, 3.1)
ELLIPSIS_Y = [1.675, 2.45]
SCORE_Y = (3.4, 4.1)
SCORE_X = (6.15, 7.55)

FILL = {"pic": "#dfe7ef", "word": "#efe7df", "ans": "#f6e2b8"}


def cellx(group):
    return {"pic": PIC_X, "word": WORD_X, "ans": ANS_X}[group]


def midx(group):
    x0, x1 = cellx(group)
    return (x0 + x1) / 2


fig, ax = plt.subplots(figsize=(10.0, 5.6))
ax.set_xlim(-0.3, 10.15)
ax.set_ylim(-0.05, 4.35)
ax.set_axis_off()


def cell(y0, y1, group):
    x0, x1 = cellx(group)
    ec = INK2 if group == "ans" else "none"
    lw = 1.2 if ec != "none" else 0
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=FILL[group], edgecolor=ec, lw=lw, zorder=3))


def row(y0, y1):
    for g in ("pic", "word", "ans"):
        cell(y0, y1, g)


def arrow(p0, p1, color, rad=0.0, lw=1.6, ms=12, style="-|>", z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color,
                                  lw=lw, connectionstyle=f"arc3,rad={rad}", zorder=z,
                                  shrinkA=0, shrinkB=0))


for y0, y1 in (LAYER0_Y, LAYERL_Y, LAYERLAST_Y):
    row(y0, y1)

# thin guides tracing each group's column through the stack, behind the rows
for g in ("pic", "word", "ans"):
    ax.plot([midx(g), midx(g)], [PROMPT_Y[1], LAYERLAST_Y[1]], color=plot.GRID, lw=0.9, zorder=1)

# a single ellipsis standing in for the layers skipped in each gap
for ey in ELLIPSIS_Y:
    ax.text((STACK_X0 + STACK_X1) / 2, ey, "⋮", ha="center", va="center",
            fontsize=13, color=MUTED, **TXT)

# ---- the prompt row ----------------------------------------------------------
labels = {"pic": "the picture", "word": "the words", "ans": "the answer\nspot"}
for g in ("pic", "word", "ans"):
    x0, x1 = cellx(g)
    ec = INK2 if g == "ans" else "none"
    ax.add_patch(Rectangle((x0, PROMPT_Y[0]), x1 - x0, PROMPT_Y[1] - PROMPT_Y[0],
                            facecolor=FILL[g], edgecolor=ec, lw=1.2, zorder=3))
    fs = 8.6 if g != "word" else 8.6
    ax.text((x0 + x1) / 2, sum(PROMPT_Y) / 2, labels[g], ha="center", va="center",
            fontsize=8.6, color=INK, **TXT)
ax.text((STACK_X0 + STACK_X1) / 2, PROMPT_Y[1] + 0.14, "one prompt, three groups of positions",
        ha="center", va="bottom", fontsize=7.6, color=MUTED, style="italic", **TXT)

# ---- layer labels -------------------------------------------------------------
ax.text(STACK_X0, LAYER0_Y[1] + 0.05, "layer 0", ha="left", va="bottom", fontsize=8, color=INK2, **TXT)
ax.text(STACK_X0, LAYERL_Y[1] + 0.05, "layer l", ha="left", va="bottom", fontsize=8, color=INK2, **TXT)
ax.text(STACK_X0, LAYERLAST_Y[1] + 0.05, "layer L, the last layer", ha="left", va="bottom",
        fontsize=8, color=INK2, **TXT)

# ---- the score ----------------------------------------------------------------
ans_mid = midx("ans")
arrow((ans_mid, LAYERLAST_Y[1]), (ans_mid, SCORE_Y[0]), INK, lw=1.4, ms=12)
ax.add_patch(Rectangle((SCORE_X[0], SCORE_Y[0]), SCORE_X[1] - SCORE_X[0], SCORE_Y[1] - SCORE_Y[0],
                        facecolor="#f4f3ee", edgecolor=INK2, lw=1.1, zorder=4))
ax.text(sum(SCORE_X) / 2, SCORE_Y[1] - 0.10, "a score m", ha="center", va="top",
        fontsize=9.3, color=INK, **TXT)
ax.text(sum(SCORE_X) / 2, SCORE_Y[0] + 0.09, "m(x) = m(h_L[a])", ha="center", va="bottom",
        fontsize=7.4, color=INK2, family="monospace", **TXT)

# ---- the read: layer l's answer spot up to layer L's answer spot -------------
read_x = ANS_X[1]
arrow((read_x, LAYERL_Y[1] + 0.02), (read_x, LAYERLAST_Y[0] - 0.02), BLUE, rad=0.75, lw=1.7, ms=12)
rx = STACK_X1 + 0.35
ax.text(rx, 2.78, "the read", color=BLUE, fontsize=9, ha="left", va="bottom", **TXT)
ax.text(rx, 2.58,
        "A_l h_l[a] + b_l ≈ h_L[a]", ha="left", fontsize=6.9, color=INK, family="monospace", **TXT)
ax.text(rx, 2.38,
        "one affine map per layer, fit with\nno labels, reused for any score",
        ha="left", va="top", fontsize=6.7, color=INK2, **TXT)

# ---- the write: leaves the score box, lands on each group at layer l --------
targets = {"pic": (midx("pic") + 0.05, LAYERL_Y[1]),
           "word": (midx("word"), LAYERL_Y[1]),
           "ans": (ANS_X[0] + 0.05, (LAYERL_Y[0] + LAYERL_Y[1]) / 2)}
starts = {"pic": (SCORE_X[0] + 0.06, SCORE_Y[0]),
          "word": (SCORE_X[0] + 0.30, SCORE_Y[0]),
          "ans": (SCORE_X[0] + 0.55, SCORE_Y[0])}
rad_for = {"pic": -0.32, "word": -0.22, "ans": -0.10}
for g in ("pic", "word", "ans"):
    arrow(starts[g], targets[g], ORANGE, rad=rad_for[g], lw=1.5, ms=11)

wx = 0.0
ax.text(wx, LAYERL_Y[1] + 0.30, "the write", color=ORANGE, fontsize=9, ha="left", **TXT)
ax.text(wx, LAYERL_Y[1] + 0.06, "u_l^c = mean of d m(x)/d h_l[p]",
        ha="left", fontsize=6.9, color=INK, family="monospace", **TXT)
ax.text(wx, LAYERL_Y[0] + 0.24, "reading of p:  u_l^c . (h_l[p] - mu_l^c)",
        ha="left", fontsize=6.9, color=INK, family="monospace", **TXT)
ax.text(wx, LAYERL_Y[0] + 0.02, "one backward pass gives every layer\nand position at once",
        ha="left", va="top", fontsize=6.7, color=INK2, **TXT)

# ---- the check: swap at layer l, measured change out -------------------------
ans_x1 = ANS_X[1]
swap_y = 1.9
arrow((STACK_X1 + 0.9, swap_y), (ans_x1 + 0.02, swap_y), AQUA, lw=1.6, ms=12)
ax.text(STACK_X1 + 1.0, swap_y, "swap in h_l[a]", ha="left", va="center", fontsize=6.9, color=INK, **TXT)

arrow((SCORE_X[1], SCORE_Y[0] + 0.10), (SCORE_X[1] + 0.85, SCORE_Y[0] + 0.30), AQUA, rad=-0.35, lw=1.6, ms=11)
ax.text(SCORE_X[1] + 0.95, SCORE_Y[0] + 0.30, "measured Δm", color=AQUA, fontsize=8, ha="left", va="center", **TXT)

ax.text(rx, 1.55, "the check", color=AQUA, fontsize=9, ha="left", **TXT)
ax.text(rx, 1.35,
        "swap another item's state in;\ncompare the true change in m to\nthe read's prediction; held-out R²",
        ha="left", va="top", fontsize=6.7, color=INK2, **TXT)

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures/flow.png")
plot.save(fig, out)
print("wrote", out)
