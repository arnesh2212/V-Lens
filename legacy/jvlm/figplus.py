# SPDX-License-Identifier: Apache-2.0
"""Four more figures, each one number the reader has not yet been shown a
picture of.  No model, no GPU: every value drawn here is read out of a result
file that already exists, or computed once, deterministically, from the
per-region arrays inside one (Spearman rho, a Mann-Whitney U, a Pearson r);
the exact call is named in each function's docstring so the number can be
checked against the file that produced it.
"""
from __future__ import annotations

import os

import numpy as np

from jvlm import style
from jvlm.style import BLUE, GRID, INK, INK2, MUTED, ORANGE, RED

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


# ======================================================= 1. the question redraws the map

def question_redraws_map(across, same_next, out, name="f135_questionmap.png",
                         n_photos=8, mannwhitney_p=None):
    """Two distributions of Spearman rho, as a strip of real points each.

    ``across`` is rho between the "easy" question's occlusion-truth map and
    the "hard" question's, same photograph, same layer -- every (photo,
    layer) pair the question-flip run covers.  ``same_next`` is rho between
    one question's own map at one layer and its own map one layer along --
    the ceiling: how much a map can slip without the question changing at
    all.  Both are read straight from
    ``results/absorb/qwen2b_twoq_cards.json``'s ``truth`` arrays; nothing
    here is estimated.
    """
    style.use()
    across = np.asarray(across, float)
    same_next = np.asarray(same_next, float)
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    rng = np.random.default_rng(0)
    rows = [("changing the question\n(same photograph, same layer)", across,
             ORANGE), ("moving one layer along\n(same question)", same_next,
                       BLUE)]
    for i, (lab, vals, c) in enumerate(rows):
        y = i + 1 + rng.uniform(-0.24, 0.24, size=len(vals))
        ax.scatter(vals, y, s=22, color=c, alpha=0.55, linewidths=0,
                  zorder=3)
        med = float(np.median(vals))
        ax.plot([med, med], [i + 1 - 0.34, i + 1 + 0.34], color=INK, lw=2.6,
               zorder=5)
        ax.annotate(f"median {med:.2f}", xy=(med, i + 1 + 0.40),
                   xytext=(0, 8), textcoords="offset points", ha="center",
                   va="bottom", fontsize=15.5, color=INK)
    ax.set_yticks([1, 2])
    ax.set_yticklabels([lab for lab, _, _ in rows], fontsize=15.5)
    ax.set_ylim(0.30, 2.85)
    ax.axvline(0.0, color=GRID, lw=1.2, zorder=1)
    ax.set_xlabel("agreement between the two maps  (Spearman rho)")
    style.despine(ax)
    if mannwhitney_p is not None:
        ax.annotate(f"Mann-Whitney p = {mannwhitney_p:.0e}",
                   xy=(0.98, 0.06), xycoords="axes fraction", ha="right",
                   va="bottom", fontsize=13.5, color=MUTED)
    ax.annotate(f"n = {n_photos} photographs; the image activations "
               "behind both maps are byte-identical",
               xy=(0.02, 0.06), xycoords="axes fraction", ha="left",
               va="bottom", fontsize=13.0, color=MUTED)
    if style.FIGURE_TITLES:
        fig.text(0.5, 0.97, "Map agreement, across questions", ha="center",
             va="top", fontsize=24, color=INK)
    if style.FIGURE_TITLES:
        fig.text(0.5, 0.925,
             "changing what is asked moves the causal map more than moving "
             "one layer through the network does",
             ha="center", va="top", fontsize=14.5, color=INK2)
    fig.subplots_adjust(left=0.24, right=0.965, top=0.86, bottom=0.10)
    return _save(fig, out, name)


# ================================================================ 2. the exact zero

def exact_zero(rows, out, name="f136_exactzero.png", n=None,
               unit="logits"):
    """A number line: two lenses and two controls sit exactly on zero; the
    lens with a question coordinate does not.

    ``rows`` is ``[(label, value, colour, is_lens), ...]``, in the order
    drawn top to bottom.  A bar chart has no bar for zero, which is the
    point being made, so this is a dot plot instead: a hard marker at 0 for
    everything that cannot move, and everything else at its own measured
    deviation.
    """
    style.use()
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    labels = [r[0] for r in rows]
    for i, (lab, v, c, is_lens) in enumerate(rows):
        y = len(rows) - i
        ax.plot([0, v], [y, y], color=c if v else GRID, lw=1.6, zorder=2)
        if v == 0:
            ax.plot([0], [y], marker="o", ms=13, mfc="#ffffff", mec=c,
                   mew=2.6, zorder=4)
            ax.annotate("exactly 0", xy=(0, y), xytext=(10, 0),
                       textcoords="offset points", ha="left", va="center",
                       fontsize=15, color=c)
        else:
            ax.plot([v], [y], marker="o", ms=13, color=c, zorder=4)
            ax.annotate(f"{v:.2f}", xy=(v, y), xytext=(10, 0),
                       textcoords="offset points", ha="left", va="center",
                       fontsize=15, color=INK)
    ax.set_yticks(range(len(rows), 0, -1))
    ax.set_yticklabels(labels, fontsize=15.5)
    ax.axvline(0.0, color=INK, lw=1.0, zorder=1)
    ax.set_ylim(0.3, len(rows) + 0.7)
    mx = max(abs(r[1]) for r in rows) or 1.0
    ax.set_xlim(-mx * 0.08, mx * 1.22)
    ax.set_xlabel(f"deviation between the two prompts, region level  "
                  f"({unit})")
    style.despine(ax)
    if n is not None:
        ax.annotate(f"n = {n}", xy=(0.98, 0.03), xycoords="axes fraction",
                   ha="right", va="bottom", fontsize=13.0, color=MUTED)
    if style.FIGURE_TITLES:
        fig.text(0.5, 0.97, "The exact zero", ha="center",
             va="top", fontsize=24, color=INK)
    # 24pt of heading is taller than the 0.045 of figure height it was given,
    # so the second line started inside the first
    if style.FIGURE_TITLES:
        fig.text(0.5, 0.97 - 24 * 1.34 / (72.0 * fig.get_figheight()),
             "the pipeline can report a dead-flat zero, and does, for "
             "everything that has no way to see the question",
             ha="center", va="top", fontsize=14.5, color=INK2)
    fig.subplots_adjust(left=0.30, right=0.93, top=0.86, bottom=0.11)
    return _save(fig, out, name)


# ==================================================== 3. the interaction law converging

def law_converging(rows, out, name="f137_convergence.png", shipped=4,
                   partition_err=None, regroup_err=None):
    """Completeness and the interaction law, both against the path rule's
    fineness, both approaching 1.

    ``rows`` is ``[(steps, completeness_r, law_r), ...]``.  The law is a
    difference of two completeness integrals, so its error is the sum of
    both integrals' errors and it converges later -- which is the reason
    the shipped step count is chosen off the law's curve, not the
    completeness curve alone.
    """
    style.use()
    steps = [r[0] for r in rows]
    comp = [r[1] for r in rows]
    law = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(13.0, 7.6))
    ax.axhline(1.0, color=GRID, lw=1.2, zorder=1, ls=(0, (4, 3)))
    ax.plot(steps, comp, color=MUTED, lw=2.4, marker="o", ms=8,
           label="completeness", zorder=3)
    ax.plot(steps, law, color=BLUE, lw=2.6, marker="o", ms=8,
           label="the interaction law", zorder=4)
    if shipped in steps:
        ax.axvline(shipped, color=INK2, lw=1.2, ls=(0, (2, 2)), zorder=2)
        ax.annotate("shipped setting", xy=(shipped, 1.045),
                   xytext=(8, 0), textcoords="offset points", ha="left",
                   va="center", fontsize=13.5, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(steps)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_xlabel("points evaluated on the path")
    ax.set_ylabel("agreement with the effect it claims to decompose  (r)")
    ax.set_ylim(0.15, 1.08)
    style.despine(ax)
    style.legend(ax, loc="lower right")
    notes = []
    if partition_err is not None:
        notes.append(f"regions sum to patches to {partition_err:.1e}")
    if regroup_err is not None:
        notes.append(f"the regrouping identity holds to {regroup_err:.0f}")
    if notes:
        ax.annotate("  ·  ".join(notes), xy=(0.02, 0.97),
                   xycoords="axes fraction", ha="left", va="top",
                   fontsize=12.6, color=MUTED)
    if style.FIGURE_TITLES:
        fig.text(0.5, 0.97, "Convergence of the interaction law", ha="center",
             va="top", fontsize=24, color=INK)
    if style.FIGURE_TITLES:
        fig.text(0.5, 0.925,
             "a difference of two integrals inherits both their errors, "
             "which is why it sets the step count",
             ha="center", va="top", fontsize=14.5, color=INK2)
    fig.subplots_adjust(left=0.09, right=0.965, top=0.86, bottom=0.11)
    return _save(fig, out, name)


# =========================================================== 4. the 8B is a real model

def real_model(scores, pair, out, name="f138_realmodel.png", n_items=None):
    """Winoground scores against their own chance levels -- which differ
    between text, image and group -- and the correct-versus-swapped yes-rate
    pair, the cleanest sign the model is doing the task rather than
    guessing.

    ``scores`` is ``[(label, value, chance), ...]``.  ``pair`` is
    ``(yes_rate_correct, yes_rate_swapped)``.
    """
    style.use()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(17.5, 7.4),
                                   gridspec_kw={"width_ratios": [1.65, 1.0]})
    labels = [s[0] for s in scores]
    vals = [s[1] for s in scores]
    chances = [s[2] for s in scores]
    x = np.arange(len(scores))
    axL.bar(x, vals, width=0.56, color=BLUE, zorder=3)
    for xi, ch in zip(x, chances):
        axL.plot([xi - 0.34, xi + 0.34], [ch, ch], color=INK, lw=2.2,
                zorder=4)
    axL.plot([], [], color=INK, lw=2.2, label="chance for that score")
    axL.set_xticks(x)
    axL.set_xticklabels(labels, fontsize=14.5)
    axL.set_ylim(0, 1.0)
    axL.set_ylabel("score")
    for xi, v in zip(x, vals):
        axL.annotate(f"{v:.2f}", xy=(xi, v), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=14.5, color=INK)
    style.despine(axL)
    style.legend(axL, loc="upper right")
    axL.set_title("Winoground scores, each against its own chance",
                  fontsize=17.5, color=INK, pad=10)

    yc, ys = pair
    axR.bar([0, 1], [yc, ys], width=0.5, color=[BLUE, RED], zorder=3)
    axR.axhline(0.5, color=GRID, lw=1.2, ls=(0, (4, 3)), zorder=1)
    axR.set_xticks([0, 1])
    axR.set_xticklabels(["the true caption", "the swapped caption"],
                        fontsize=14.5)
    axR.set_ylim(0, 1.0)
    for xi, v in zip([0, 1], [yc, ys]):
        axR.annotate(f"{v:.2f}", xy=(xi, v), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=14.5, color=INK)
    style.despine(axR)
    axR.set_title("share of items answered “yes”", fontsize=17.5,
                  color=INK, pad=10)

    if n_items is not None:
        fig.text(0.5, 0.045, f"n = {n_items} winoground items", ha="center",
                 va="bottom", fontsize=13.0, color=MUTED)
    if style.FIGURE_TITLES:
        fig.text(0.5, 0.97, "Capability of the 8B model", ha="center", va="top",
             fontsize=24, color=INK)
    if style.FIGURE_TITLES:
        fig.text(0.5, 0.925,
             "every score clears its own chance level, and the model says "
             "“yes” far more to the caption that is actually true",
             ha="center", va="top", fontsize=14.5, color=INK2)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.83, bottom=0.13,
                        wspace=0.22)
    return _save(fig, out, name)
