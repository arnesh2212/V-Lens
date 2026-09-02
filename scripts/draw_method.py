# SPDX-License-Identifier: Apache-2.0
"""A schematic of the two directions: the read and the write, both at the answer spot."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from jvlm import plot

plot.use()
fig, ax = plt.subplots(figsize=(7.6, 3.2))
ax.set_xlim(0, 10.4); ax.set_ylim(0, 5.2); ax.set_axis_off()
BLUE, ORANGE, GREY, INK = plot.SERIES[0], plot.SERIES[1], "#c9c8c1", plot.INK2

# the token row
cols = [("the picture", 0.4, 3.2, "#dfe7ef"), ("the words", 3.9, 2.0, "#efe7df"), ("answer spot", 6.1, 1.0, "#e6e6e0")]
for lab, x0, w, col in cols:
    ax.add_patch(Rectangle((x0, 0.45), w, 0.7, facecolor=col, edgecolor="none"))
    ax.text(x0 + w / 2, 0.8, lab, ha="center", va="center", fontsize=9, color=INK)
# the layer stack above the answer spot
for i, y in enumerate([1.5, 2.3, 3.1, 3.9]):
    ax.add_patch(Rectangle((6.1, y), 1.0, 0.45, facecolor="#f0efe9", edgecolor="none"))
    ax.text(6.6, y + 0.22, ["layer l", "", "", "last layer"][i], ha="center", va="center", fontsize=8.5, color=INK)
ax.text(6.6, 4.75, "the first answer token", ha="center", fontsize=9, color=INK)

# the read: from layer l to the last layer
ax.add_patch(FancyArrowPatch((7.25, 1.72), (7.25, 3.95), arrowstyle="-|>", mutation_scale=13, color=BLUE, lw=1.6,
                             connectionstyle="arc3,rad=0.35"))
ax.text(8.15, 3.3, "the read", color=BLUE, fontsize=10, ha="left")
ax.text(8.15, 2.95, "one small map per layer,\nfitted with no labels:\nwhat the answer\nis going to be", color=INK, fontsize=8.2, ha="left", va="top")

# the write: gradient from the answer back to the picture and the words
for x in (2.0, 4.9):
    ax.add_patch(FancyArrowPatch((6.2, 1.6), (x, 1.25), arrowstyle="-|>", mutation_scale=12, color=ORANGE, lw=1.5,
                                 connectionstyle="arc3,rad=0.28"))
ax.text(0.4, 3.95, "the write", color=ORANGE, fontsize=10)
ax.text(0.4, 3.6, "the model’s own gradient of the score,\none direction for each group of positions:\nwhat moves the answer, and the handle\nto move it with", color=INK, fontsize=8.2, va="top")

ax.text(6.6, 4.35, "a score", ha="center", fontsize=9, color=INK)
ax.plot([6.1, 7.1], [4.32, 4.32], color=GREY, lw=0.8)
fig.tight_layout()
plot.save(fig, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures/method.png"))
