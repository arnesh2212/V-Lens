# SPDX-License-Identifier: Apache-2.0
"""Paper figures.  Each function takes a results directory and writes one PNG."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from jvlm import style
from jvlm.style import (AQUA, BLUE, C_CTRL, C_KEY, C_WORD, GRID, INK, INK2,
                        MUTED, ORANGE, RED, SURFACE, VIOLET)

import matplotlib.pyplot as plt


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)
    return p


# --------------------------------------------------------------- F1 dissociation

DISS_LABELS = {
    "clean": "no\nintervention",
    "wrong_image": "a different\nphotograph",
    "abl_all": "erase the concept\nfrom the workspace",
    "abl_rand": "erase a random\ndirection",
    "abl_other": "erase an unrelated\nconcept",
    "abl_unemb": "erase the raw\nunembedding row",
}

BAND_LABELS = {
    "abl_all": "every layer",
    "abl_e90": "up to 0.9\nof depth",
    "abl_e80": "up to 0.8\nof depth",
    "abl_hi": "0.5 - 0.95",
    "abl_mid": "0.25 - 0.6",
    "abl_lo": "0.1 - 0.5",
    "abl_img": "image tokens\nonly",
    "abl_prompt": "prompt text,\ncue spared",
}


def f1_dissociation(df: pd.DataFrame, out: str, name="f1_dissociation.png"):
    """The money shot: naming collapses, forced choice does not."""
    conds = [c for c in DISS_LABELS if c in set(df.cond)]
    g = df[df.cond.isin(conds)].groupby("cond")
    mean = g[["word_correct", "letter_correct"]].mean().reindex(conds) * 100
    sem = (g[["word_correct", "letter_correct"]].sem().reindex(conds) * 100).fillna(0)

    return _grouped_pair(
        mean, sem, [DISS_LABELS[c] for c in conds], out, name,
        "Erasing a concept from the verbalizable workspace",
        "One question, one photograph, answered twice: once with the object's "
        "name, once with an arbitrary letter.",
        figsize=(11.8, 5.8))


def _grouped_pair(mean, sem, xlabels, out, name, ttl, sub, figsize=(11.5, 5.6)):
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(xlabels))
    w = 0.34
    gap = 0.02          # surface gap between adjacent fills
    ax.bar(x - w / 2 - gap, mean.word_correct, w, yerr=sem.word_correct,
           color=C_WORD, label="answers with the word",
           error_kw=dict(lw=1, ecolor=MUTED))
    ax.bar(x + w / 2 + gap, mean.letter_correct, w, yerr=sem.letter_correct,
           color=C_KEY, label="answers with the letter",
           error_kw=dict(lw=1, ecolor=MUTED))
    ax.axhline(50, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate("chance", xy=(len(xlabels) - 0.42, 50), xytext=(5, 3),
                textcoords="offset points", color=MUTED, fontsize=11,
                annotation_clip=False)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("correct on the same question  (%)")
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    style.title(ax, ttl, sub, legend_rows=1)
    style.legend_above(ax, ncols=2)
    for i in range(len(xlabels)):
        for off, col in ((-w / 2 - gap, "word_correct"), (w / 2 + gap, "letter_correct")):
            v = mean[col].iloc[i]
            ax.annotate(f"{v:.0f}", xy=(i + off, v), xytext=(0, 5),
                        textcoords="offset points", ha="center",
                        color=INK2, fontsize=11.5)
    return _save(fig, out, name)


def f1b_localisation(df: pd.DataFrame, out: str, name="f1b_localisation.png"):
    """Where the erasure has to reach before the verbal channel gives way."""
    conds = [c for c in BAND_LABELS if c in set(df.cond)]
    g = df[df.cond.isin(conds)].groupby("cond")
    mean = g[["word_correct", "letter_correct"]].mean().reindex(conds) * 100
    sem = (g[["word_correct", "letter_correct"]].sem().reindex(conds) * 100).fillna(0)
    return _grouped_pair(
        mean, sem, [BAND_LABELS[c] for c in conds], out, name,
        "How far the erasure has to reach",
        "The same erasure, restricted to a band of layers or to one input channel.",
        figsize=(12.6, 5.8))


# ------------------------------------------------------------------ F2 dose

def f2_dose(df: pd.DataFrame, out: str, name="f2_dose.png"):
    d = df[df.cond.str.startswith("abl_all")].copy()
    d.loc[d.cond == "abl_all", "alpha"] = 1.0
    clean = df[df.cond == "clean"]
    rows = [(0.0, clean.word_margin.mean(), clean.letter_margin.mean(),
             clean.word_margin.sem(), clean.letter_margin.sem())]
    for al, gg in d.groupby("alpha"):
        rows.append((al, gg.word_margin.mean(), gg.letter_margin.mean(),
                     gg.word_margin.sem(), gg.letter_margin.sem()))
    r = pd.DataFrame(rows, columns=["a", "w", "l", "we", "le"]).sort_values("a")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for col, err, c, lab in (("w", "we", C_WORD, "word channel"),
                             ("l", "le", C_KEY, "letter channel")):
        ax.plot(r.a, r[col], color=c, marker="o", label=lab)
        ax.fill_between(r.a, r[col] - r[err], r[col] + r[err], color=c, alpha=0.14, lw=0)
    ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.annotate("wrong answer below this line", xy=(r.a.max(), 0), xytext=(-4, -16),
                textcoords="offset points", ha="right", color=MUTED, fontsize=11)
    style.title(ax, "The two channels come apart as the erasure strengthens",
                "Logit margin in favour of the correct option.")
    ax.set_xlabel(r"erasure strength  $\alpha$")
    ax.set_ylabel("margin for the correct option")
    style.legend(ax, loc="lower left")
    return _save(fig, out, name)


# ------------------------------------------------------------- F3 injection

def f3_injection(df: pd.DataFrame, out: str, name="f3_injection.png"):
    d = df[df.cond.str.startswith("add_img")].copy()
    d.loc[d.cond == "add_img", "alpha"] = 1.0
    clean = df[df.cond == "clean"]
    rows = [(0.0, clean.word_margin.mean(), clean.letter_margin.mean())]
    for al, gg in d.groupby("alpha"):
        rows.append((al, gg.word_margin.mean(), gg.letter_margin.mean()))
    r = pd.DataFrame(rows, columns=["a", "w", "l"]).sort_values("a")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(r.a, r.w, color=C_WORD, marker="o", label="word channel")
    ax.plot(r.a, r.l, color=C_KEY, marker="o", label="letter channel")
    ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
    style.title(ax, "Writing an absent object into the workspace",
                "The object is not in the photograph. Its workspace direction is added anyway.")
    ax.set_xlabel(r"injection strength  $\alpha$")
    ax.set_ylabel("margin for the object that is really there")
    style.legend(ax, loc="lower left")
    return _save(fig, out, name)


# ----------------------------------------------------------------- F4 gate0

def f4_readout(summary: pd.DataFrame, out: str, name="f4_readout.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    for use_j, c, lab in ((1, style.C_JLENS, "Jacobian lens"),
                          (0, style.C_LOGIT, "logit lens")):
        s = summary[summary.jac == use_j].sort_values("depth")
        axes[0].plot(s.depth, s.median_rank_target, color=c, label=lab)
        axes[1].plot(s.depth, s.auc, color=c, label=lab)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("rank of the object's name")
    axes[1].set_ylabel("present vs absent  (AUC)")
    axes[1].axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)))
    for ax in axes:
        ax.set_xlabel("relative depth in the language model")
        style.legend(ax)
    style.title(axes[0], "What the image tokens are disposed to say")
    style.title(axes[1], "Telling a present object from an absent one")
    fig.tight_layout()
    return _save(fig, out, name)


# ------------------------------------------------------------ F5 provenance

def f5_provenance(df: pd.DataFrame, out: str, name="f5_provenance.png"):
    views = ["full", "orthogonal", "random", "jspace"]
    labels = {"full": "whole residual\nstream", "orthogonal": "everything outside\nthe workspace",
              "random": "a random subspace\nof equal size", "jspace": "the verbalizable\nworkspace"}
    g = df.groupby(["probe", "view"]).auc.agg(["mean", "sem"])

    fig, ax = plt.subplots(figsize=(10.4, 5.3))
    x = np.arange(len(views))
    w = 0.36
    for i, (probe, c, lab) in enumerate((("content", AQUA, "which object is active"),
                                         ("provenance", VIOLET, "which channel it came from"))):
        m = [g.loc[(probe, v), "mean"] * 100 for v in views]
        e = [g.loc[(probe, v), "sem"] * 100 for v in views]
        ax.bar(x + (i - 0.5) * w, m, w, yerr=e, color=c, label=lab,
               error_kw=dict(lw=1, ecolor=MUTED))
        for xi, v in zip(x + (i - 0.5) * w, m):
            ax.annotate(f"{v:.0f}", xy=(xi, v), xytext=(0, 5), ha="center",
                        textcoords="offset points", color=INK2, fontsize=11.5)
    ax.axhline(50, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.annotate("chance", xy=(len(views) - 0.45, 50), xytext=(4, 3),
                textcoords="offset points", color=MUTED, fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels([labels[v] for v in views])
    ax.set_ylabel("linear probe  (AUC, %)")
    ax.set_ylim(40, 105)
    style.title(ax, "The workspace knows what, not where from",
                "Probes read at the final prompt position, where both channels have already merged.")
    ax.legend(loc="lower left", ncols=2)
    return _save(fig, out, name)


# --------------------------------------------------------------- F6 forgery

def f6_forgery(df: pd.DataFrame, out: str, name="f6_forgery.png"):
    """The attack has two objectives that pull apart, so plot the frontier."""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))

    ax = axes[0]
    lo = df[df.beta == df.beta.min()]
    for i, (col, c, lab) in enumerate((("lens_rank_clean", C_CTRL, "clean image"),
                                       ("lens_rank_adv", RED, "perturbed image"))):
        v = np.clip(lo[col].values.astype(float), 0.5, None)
        jitter = np.linspace(-0.13, 0.13, len(v))
        ax.scatter(np.full(len(v), i) + jitter, v, color=c, s=26, alpha=.75,
                   edgecolors="none")
        ax.scatter([i], [np.median(v)], color=INK, s=90, marker="_", zorder=5)
    ax.set_yscale("log")
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["clean image", "perturbed image"])
    ax.set_ylabel("rank of the planted word in the readout")
    style.title(ax, "The monitor reads a word that is not there",
                "Each point is one photograph.")

    ax = axes[1]
    f = df.groupby("beta").agg(
        hit=("lens_rank_adv", lambda s: float((s < 10).mean())),
        same=("text_same", "mean")).reset_index().sort_values("beta")
    ax.plot(f.same * 100, f.hit * 100, color=BLUE, marker="o")
    for _, r in f.iterrows():
        ax.annotate(f"$\\beta$={r.beta:g}", xy=(r.same * 100, r.hit * 100),
                    xytext=(8, -3), textcoords="offset points",
                    color=INK2, fontsize=11.5)
    ax.set_xlabel("generated text left byte-identical  (%)")
    ax.set_ylabel("planted word in the readout's top 10  (%)")
    # The frontier occupies under a tenth of a 0-100 axis, so nine tenths of
    # the panel was blank.  Fit the axis to the sweep and say in the subtitle
    # that the whole range is 0-100.
    xr = float(f.same.max()) * 100
    ax.set_xlim(-1.5, max(12.0, xr * 1.35)); ax.set_ylim(-4, 108)
    ax.grid(True, axis="x", color=GRID, lw=0.9)
    style.title(ax, "Both halves at once",
                "Weight on output preservation traces the frontier. The x axis "
                "runs 0-100% but the whole sweep sits in its first tenth: no "
                "setting keeps the caption intact.")
    fig.tight_layout()
    return _save(fig, out, name)


# ---------------------------------------------------------------- F7 routes

ROUTE_LABELS = {"en_correct": "English word", "zh_correct": "Chinese word",
                "syn_correct": "English synonym", "letter_correct": "letter key"}
ARM_LABELS = {"clean": "nothing erased", "full": "the whole lens direction",
              "par": "only its unembedding\ncomponent",
              "perp": "only what the Jacobian\nadds",
              "unemb": "the raw unembedding\nrows"}


def f7_routes(df: pd.DataFrame, out: str, name="f7_routes.png"):
    """Does erasure cut the concept, or only the English spelling of it?"""
    arms = [a for a in ARM_LABELS if a in set(df.arm)]
    cols = [c for c in ROUTE_LABELS if c in df.columns]
    g = df.groupby("arm")[cols].mean().reindex(arms) * 100
    e = (df.groupby("arm")[cols].sem().reindex(arms) * 100).fillna(0)

    fig, ax = plt.subplots(figsize=(12.4, 5.8))
    x = np.arange(len(arms))
    w = 0.78 / len(cols)
    # three verbal routes in warm slots, the arbitrary key in blue -- the same
    # role colours the rest of the paper uses
    hues = {"en_correct": ORANGE, "zh_correct": AQUA,
            "syn_correct": "#eda100", "letter_correct": BLUE}
    for j, c in enumerate(cols):
        off = (j - (len(cols) - 1) / 2) * (w + 0.012)
        ax.bar(x + off, g[c], w, yerr=e[c], color=hues[c],
               label=ROUTE_LABELS[c], error_kw=dict(lw=1, ecolor=MUTED))
    ax.axhline(50, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xticks(x); ax.set_xticklabels([ARM_LABELS[a] for a in arms])
    ax.set_ylabel("correct  (%)"); ax.set_ylim(0, 100)
    style.title(ax, "Which route does the erasure cut?",
                "The same visual question, answered through four different "
                "output vocabularies.", legend_rows=1)
    style.legend_above(ax, ncols=4)
    return _save(fig, out, name)


def f8_geometry(geo: pd.DataFrame, out: str, name="f8_geometry.png"):
    """How much of the lens direction is just the unembedding row."""
    g = geo.groupby("depth")[["cos_V_WU", "frac_par"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(g.depth, g.cos_V_WU, color=BLUE, marker="o",
            label=r"cosine with $W_U[c]$")
    ax.plot(g.depth, g.frac_par, color=ORANGE, marker="o",
            label=r"fraction of energy in span $W_U[c]$")
    ax.set_xlabel("relative depth in the language model")
    ax.set_ylabel("alignment")
    ax.set_ylim(0, 1)
    style.title(ax, "The lens direction becomes the unembedding row",
                "Near the output the Jacobian is close to the identity, so "
                "erasing there is close to deleting the word.")
    ax.legend(loc="upper left")
    return _save(fig, out, name)


def f9_competition(df: pd.DataFrame, out: str, name="f9_competition.png"):
    """Image content crowding a text-given word out of the workspace."""
    g = df.groupby("n_objects").agg(occ=("occupancy", "mean"),
                                    occ_e=("occupancy", "sem"),
                                    rec=("recalled", "mean"),
                                    rec_e=("recalled", "sem"),
                                    rank=("hold_rank", "median")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    ax = axes[0]
    ax.plot(g.n_objects, g.occ, color=BLUE, marker="o")
    ax.fill_between(g.n_objects, g.occ - g.occ_e, g.occ + g.occ_e,
                    color=BLUE, alpha=.15, lw=0)
    ax.set_xlabel("distinct objects in the photograph")
    ax.set_ylabel("scene objects occupying the workspace")
    style.title(ax, "A busier picture fills the workspace")

    ax = axes[1]
    b = df.copy()
    b["bin"] = pd.cut(b.occupancy, [-.1, 0.5, 1.5, 2.5, 3.5, 99],
                      labels=["0", "1", "2", "3", "4+"])
    h = b.groupby("bin", observed=True).agg(rec=("recalled", "mean"),
                                            e=("recalled", "sem"),
                                            n=("recalled", "size")).reset_index()
    ax.bar(range(len(h)), h.rec * 100, 0.6, yerr=h.e * 100, color=ORANGE,
           error_kw=dict(lw=1, ecolor=MUTED))
    ax.set_xticks(range(len(h))); ax.set_xticklabels(h["bin"].astype(str))
    ax.set_xlabel("scene objects occupying the workspace")
    ax.set_ylabel("word from the text recalled  (%)")
    style.title(ax, "and the word from the text drops out")
    fig.tight_layout()
    return _save(fig, out, name)


DEPTH_ORDER = ["abl_lo", "abl_mid", "abl_e60", "abl_e70", "abl_e80",
               "abl_e90", "abl_all"]
DEPTH_LABEL = {"abl_lo": "0.1-0.5", "abl_mid": "0.25-0.6", "abl_e60": "up to 0.6",
               "abl_e70": "up to 0.7", "abl_e80": "up to 0.8",
               "abl_e90": "up to 0.9", "abl_all": "every layer"}


def f10_depth(df: pd.DataFrame, out: str, name="f10_depth.png"):
    conds = [c for c in DEPTH_ORDER if c in set(df.cond)]
    g = df[df.cond.isin(conds)].groupby("cond")
    mean = g[["word_correct", "letter_correct"]].mean().reindex(conds) * 100
    sem = (g[["word_correct", "letter_correct"]].sem().reindex(conds) * 100).fillna(0)
    return _grouped_pair(mean, sem, [DEPTH_LABEL[c] for c in conds], out, name,
                         "How deep the erasure has to reach",
                         "Layers erased, as a fraction of network depth.",
                         figsize=(12.2, 5.8))


# -------------------------------------------------------------- F11 families

FAMILY_LABEL = {"object": "which object\nis present", "colour": "what colour\nit is",
                "count": "how many\nthere are", "relation": "where one is\nrelative to another"}


def f11_families(df: pd.DataFrame, out: str, name="f11_families.png"):
    """The same erasure across four kinds of concept, only one of them a noun."""
    fams = [f for f in FAMILY_LABEL if f in set(df.family)]
    fig, ax = plt.subplots(figsize=(12.4, 5.8))
    x = np.arange(len(fams))
    w = 0.2
    gap = 0.015
    spec = [("clean", "word_correct", C_WORD, 0.35, "word, nothing erased"),
            ("erase", "word_correct", C_WORD, 1.0, "word, concept erased"),
            ("clean", "letter_correct", C_KEY, 0.35, "letter, nothing erased"),
            ("erase", "letter_correct", C_KEY, 1.0, "letter, concept erased")]
    for j, (arm, col, c, alpha, lab) in enumerate(spec):
        off = (j - 1.5) * (w + gap)
        g = df[df.arm == arm].groupby("family")[col].mean().reindex(fams) * 100
        e = df[df.arm == arm].groupby("family")[col].sem().reindex(fams) * 100
        ax.bar(x + off, g, w, yerr=e.fillna(0), color=c, alpha=alpha,
               label=lab, error_kw=dict(lw=1, ecolor=MUTED))
    ax.axhline(50, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.annotate("chance", xy=(len(fams) - 0.42, 50), xytext=(5, 3),
                textcoords="offset points", color=MUTED, fontsize=11,
                annotation_clip=False)
    ax.set_xticks(x); ax.set_xticklabels([FAMILY_LABEL[f] for f in fams])
    ax.set_ylabel("correct  (%)"); ax.set_ylim(0, 100)
    style.title(ax, "Not a fact about nouns",
                "The same erasure and the same two response channels, over four "
                "kinds of concept.", legend_rows=1)
    style.legend_above(ax, ncols=4)
    return _save(fig, out, name)


# --------------------------------------------------- F12 verbalizable fraction

def f12_verbalizable(geo: pd.DataFrame, pred: pd.DataFrame, out: str,
                     name="f12_verbalizable.png"):
    """How much of what a picture writes is verbalizable, and what that predicts."""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))

    ax = axes[0]
    g = geo.groupby("depth")[["in_vocab_jspace", "null_vocab"]].mean().reset_index()
    ax.plot(g.depth, g.in_vocab_jspace * 100, color=BLUE, marker="o",
            label="what the picture writes")
    ax.plot(g.depth, g.null_vocab * 100, color=MUTED, marker="o", ls=(0, (4, 3)),
            label="a random direction")
    ax.set_xlabel("relative depth in the language model")
    ax.set_ylabel("share inside the workspace  (%)")
    style.title(ax, "The verbalizable fraction",
                "Energy of the perceptual direction lying in the workspace.")
    ax.legend(loc="upper left")

    ax = axes[1]
    if pred is not None and len(pred) >= 4:
        ax.scatter(pred["in_vocab_jspace"] * 100, pred["letter_margin"],
                   color=C_KEY, s=54, edgecolors="none", label="letter channel")
        ax.scatter(pred["in_vocab_jspace"] * 100, pred["word_margin"],
                   color=C_WORD, s=54, edgecolors="none", label="word channel")
        for col, c in (("letter_margin", C_KEY), ("word_margin", C_WORD)):
            x = pred["in_vocab_jspace"].values * 100
            y = pred[col].values
            if len(x) > 2 and np.std(x) > 0:
                b, a0 = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 2)
                ax.plot(xs, a0 + b * xs, color=c, lw=1.4, alpha=.65)
        ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.set_xlabel("verbalizable fraction of the concept  (%)")
        ax.set_ylabel("margin lost when the workspace is erased")
        style.legend(ax)
    style.title(ax, "Geometry predicts damage",
                "One point per concept. The x axis uses no behavioural data.")
    fig.tight_layout()
    return _save(fig, out, name)


# ------------------------------------------------------- F13 perceptual lesion

PERC_LABEL = {"clean": "nothing erased", "workspace": "the workspace\ndirection",
              "perceptual": "the perceptual\ndirection", "random": "a random\nsubspace"}


def f13_perceptual(df: pd.DataFrame, out: str, name="f13_perceptual.png"):
    arms = [a for a in PERC_LABEL if a in set(df.arm)]
    g = df.groupby("arm")
    mean = g[["word_correct", "letter_correct"]].mean().reindex(arms) * 100
    sem = (g[["word_correct", "letter_correct"]].sem().reindex(arms) * 100).fillna(0)
    return _grouped_pair(mean, sem, [PERC_LABEL[a] for a in arms], out, name,
                         "Two lesions, opposite effects",
                         "One direction is built from words, the other only from "
                         "what photographs do to the model.",
                         figsize=(11.4, 5.8))


# ----------------------------------------------------------- F14 collateral

ROUTE_NAME = {"en": "English", "fr": "French", "zh": "Chinese",
              "syn": "synonym", "letter": "letter key", "digit": "digit key"}
# Identity here is carried by the direct labels, not by hue: six categories in a
# scatter puts every pair on screen at once, and no six-hue set clears the
# all-pairs separation floors.  One hue, six labels.
ROUTE_HUE = {r: BLUE for r in ROUTE_NAME}


def f14_collateral(df: pd.DataFrame, out: str, name="f14_collateral.png"):
    """A behavioural ordering predicted by a number read off the weights."""
    from scipy.stats import spearmanr

    d = df.dropna(subset=["g", "acc_loss"])
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    present = [x for x in ROUTE_NAME if x in set(d.route)]
    ax.scatter(d.g, d.acc_loss * 100, s=24, alpha=.22, color=BLUE,
               edgecolors="none")
    # Four of these channels sit at zero loss within a hair of each other, so
    # a fixed offset stacks their names into an unreadable pile -- which is
    # what the first render did.  Sort by x and step the offset instead.
    marks = sorted(((d[d.route == r].g.mean(),
                     d[d.route == r].acc_loss.mean() * 100, r)
                    for r in present), key=lambda t: t[0])
    # Adjacent channels differ in y by about as much as any single offset, so
    # stepping only the height still stacks them.  Alternate the side as well
    # and no two neighbours can share an anchor.
    for k, (gx, gy, r) in enumerate(marks):
        ax.scatter([gx], [gy], s=130, color=BLUE, edgecolors=SURFACE,
                   linewidths=2, zorder=5)
        dy = 16 + 18 * (k % 3)
        left = k % 2 == 0
        ax.annotate(ROUTE_NAME[r], xy=(gx, gy),
                    xytext=(-10 if left else 10, dy),
                    textcoords="offset points",
                    ha="right" if left else "left", color=INK,
                    fontsize=12.5, zorder=6,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                    shrinkA=2, shrinkB=9))
    # the trend the geometry predicts, fitted on the channel means only
    mg = d.groupby("route").agg(g=("g", "mean"), y=("acc_loss", "mean"))
    if len(mg) > 2 and mg.g.std() > 0:
        b, a0 = np.polyfit(mg.g, mg.y * 100, 1)
        xs = np.linspace(mg.g.min(), mg.g.max(), 2)
        ax.plot(xs, a0 + b * xs, color=MUTED, lw=1.3, ls=(0, (5, 3)), zorder=1)
    rho, pv = spearmanr(d.g, d.acc_loss)
    ax.set_xlabel("share of the channel's own row inside the erased span")
    ax.set_ylabel("accuracy lost to the erasure  (points)")
    style.title(ax, "Weights predict which answers survive",
                f"One faint point per concept and channel; solid marks are channel "
                f"means. Spearman rho = {rho:+.2f} (p = {pv:.1g}).")
    return _save(fig, out, name)


# ---------------------------------------------------------------- F15 patch

def f15_patch(df: pd.DataFrame, out: str, name="f15_patch.png"):
    """Injecting the verbal copy and the use copy of one perceptual direction."""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    arms = [("verbal", VIOLET, "the verbal copy"),
            ("use", AQUA, "the use copy"),
            ("random", MUTED, "a random direction")]
    clean = df[df.arm == "clean"]

    ax = axes[0]
    for arm, c, lab in arms:
        d = df[df.arm == arm].groupby("alpha").say_rank.median()
        xs = [0.0] + list(d.index)
        ys = [clean.say_rank.median()] + list(d.values)
        ax.plot(xs, np.clip(ys, 0.7, None), color=c, marker="o", label=lab)
    ax.set_yscale("log")
    ax.set_yticks([100, 300, 1000, 3000, 10000])
    ax.get_yaxis().set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel(r"injection strength  $\alpha$")
    ax.set_ylabel("rank of the absent object,\nasked to name what it sees")
    style.title(ax, "Does it say the thing that is not there?",
                "Lower is more willing to name it.")
    style.legend(ax, loc="lower left")

    ax = axes[1]
    for arm, c, lab in arms:
        d = df[df.arm == arm].groupby("alpha").letter_picks_injected.mean()
        xs = [0.0] + list(d.index)
        ys = [clean.letter_picks_injected.mean()] + list(d.values)
        ax.plot(xs, np.array(ys) * 100, color=c, marker="o", label=lab)
    ax.axhline(50, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xlabel(r"injection strength  $\alpha$")
    ax.set_ylabel("forced choice picks the absent object  (%)")
    ax.set_ylim(0, 100)
    style.title(ax, "Does it choose it?",
                "The two copies disturb the choice equally, which is what makes "
                "the left panel a fair comparison.")
    fig.tight_layout()
    return _save(fig, out, name)


# -------------------------------------------------------------- F16 reality

def f16_reality(beh: pd.DataFrame, probes: pd.DataFrame, out: str,
                name="f16_reality.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))

    ax = axes[0]
    vals, labs, hues = [], [], []
    vals.append(beh.check_correct.mean() * 100)
    labs.append("is it really there,\nor did the text say so?")
    hues.append(C_WORD)
    for col, lab in (("ctl_present", "is this object\nvisible?"),
                     ("ctl_absent", "is this absent object\nvisible?")):
        if col in beh.columns:
            vals.append(beh[col].mean() * 100)
            labs.append(lab)
            hues.append(C_KEY)
    ax.bar(range(len(vals)), vals, 0.55, color=hues)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.0f}", xy=(i, v), xytext=(0, 5), ha="center",
                    textcoords="offset points", color=INK2, fontsize=12)
    ax.axhline(50, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labs)
    ax.set_ylabel("correct  (%)"); ax.set_ylim(0, 100)
    style.title(ax, "Asked directly", "Same photograph, same answer format.")

    ax = axes[1]
    if probes is not None and len(probes):
        views = ["full", "orthogonal", "random", "jspace"]
        lab = {"full": "whole\nresidual", "orthogonal": "outside the\nchannel",
               "random": "random\nsubspace", "jspace": "inside the\nchannel"}
        g = probes.groupby("view").auc.agg(["mean", "sem"])
        vs = [v for v in views if v in g.index]
        ax.bar(range(len(vs)), [g.loc[v, "mean"] * 100 for v in vs], 0.55,
               yerr=[g.loc[v, "sem"] * 100 for v in vs],
               color=[C_CTRL if v != "jspace" else VIOLET for v in vs],
               error_kw=dict(lw=1, ecolor=MUTED))
        for i, v in enumerate(vs):
            ax.annotate(f"{g.loc[v, 'mean']*100:.0f}", xy=(i, g.loc[v, "mean"]*100),
                        xytext=(0, 5), ha="center", textcoords="offset points",
                        color=INK2, fontsize=12)
        ax.axhline(50, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.set_xticks(range(len(vs))); ax.set_xticklabels([lab[v] for v in vs])
        # starting this axis at 40 turned a three-point spread into bars of
        # visibly different heights; a probe axis belongs on its own floor,
        # which is chance
        ax.set_ylabel("linear probe  (AUC, %)"); ax.set_ylim(50, 100)
        ax.axhline(50, color=MUTED, lw=1, ls=(0, (4, 3)))
    style.title(ax, "But the answer is in there",
                "Probing for whether the asserted object is really present.")
    fig.tight_layout()
    return _save(fig, out, name)


# -------------------------------------------------------------- F17 scaling

def f17_scaling(frames: dict, out: str, name="f17_scaling.png"):
    """The headline dissociation, one column per model."""
    names = list(frames)
    fig, ax = plt.subplots(figsize=(11.6, 5.6))
    x = np.arange(len(names))
    w = 0.34
    gap = 0.02
    wv, lv, we, le = [], [], [], []
    for n_ in names:
        d = frames[n_]
        a = d[d.arm.isin(["erase", "full"])] if "arm" in d else d[d.cond == "abl_all"]
        c = d[d.arm == "clean"] if "arm" in d else d[d.cond == "clean"]
        col_w = "en_correct" if "en_correct" in d else "word_correct"
        col_l = "letter_correct"
        wv.append(a[col_w].mean() * 100); lv.append(a[col_l].mean() * 100)
        we.append(a[col_w].sem() * 100); le.append(a[col_l].sem() * 100)
    ax.bar(x - w/2 - gap, wv, w, yerr=we, color=C_WORD,
           label="says the word", error_kw=dict(lw=1, ecolor=MUTED))
    ax.bar(x + w/2 + gap, lv, w, yerr=le, color=C_KEY,
           label="presses the key", error_kw=dict(lw=1, ecolor=MUTED))
    ax.axhline(50, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("correct after the erasure  (%)"); ax.set_ylim(0, 100)
    style.title(ax, "The same picture at other scales",
                "Every model gets its own lens, fitted the same way.",
                legend_rows=1)
    style.legend_above(ax, ncols=2)
    return _save(fig, out, name)


# --------------------------------------------------------- forgery examples

def forgery_pair(clean_path, adv_path, delta_path, out, caption_bits,
                 name="forgery_pair.png"):
    """The two photographs, and the difference between them, amplified.

    An L-infinity bound quoted in the text is an abstraction; the point of the
    attack is that a person looking at the two pictures cannot tell them apart,
    and only showing them makes that concrete.
    """
    import matplotlib.image as mpimg

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.0))
    for ax, path, ttl in zip(
            axes, (clean_path, adv_path, delta_path),
            ("The photograph", "After the perturbation",
             "The difference, amplified")):
        ax.imshow(mpimg.imread(path))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        style.title(ax, ttl)
    for ax, bit in zip(axes, caption_bits):
        ax.annotate(bit, xy=(0.5, -0.04), xycoords="axes fraction",
                    ha="center", va="top", color=INK2, fontsize=12)
    fig.tight_layout()
    return _save(fig, out, name)


# ---------------------------------------------------------- F18 depth crossover

def f18_depthcross(df: pd.DataFrame, out: str, name="f18_depthcross.png"):
    """Where the Jacobian direction beats the raw unembedding row."""
    bands = [b for b in sorted(set(df.band)) if b != "none"]
    clean = df[df.band == "none"].word_correct.mean() * 100
    fig, ax = plt.subplots(figsize=(11.4, 5.6))
    x = np.arange(len(bands))
    w = 0.34
    gap = 0.02
    for j, (dirn, c, lab) in enumerate(
            (("lens", BLUE, "the lens direction"),
             ("unembed", ORANGE, "the raw unembedding row"))):
        d = df[df.direction == dirn].groupby("band").word_correct
        mean = d.mean().reindex(bands) * 100
        sem = d.sem().reindex(bands).fillna(0) * 100
        off = (j - 0.5) * (w + gap)
        ax.bar(x + off, mean, w, yerr=sem, color=c, label=lab,
               error_kw=dict(lw=1, ecolor=MUTED))
    ax.axhline(clean, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.annotate("no intervention", xy=(len(bands) - 0.45, clean), xytext=(5, 3),
                textcoords="offset points", color=MUTED, fontsize=11,
                annotation_clip=False)
    cos = df[df.band != "none"].groupby("band").cos.mean().reindex(bands)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\ncos = {cos[b]:.2f}" for b in bands])
    ax.set_xlabel("band of layers erased, as a fraction of depth")
    ax.set_ylabel("says the right word  (%)")
    ax.set_ylim(0, 100)
    style.title(ax, "Where the Jacobian earns its keep",
                "The two directions converge exactly where their cosine does.",
                legend_rows=1)
    style.legend_above(ax, ncols=2)
    return _save(fig, out, name)


# --------------------------------------------------------------- F19 language

LANG_LABEL = {"clean": "nothing\nerased", "en": "English", "zh": "Chinese",
              "fr": "French", "en+zh": "English\n+ Chinese",
              "en+zh+fr": "all three"}
CHAN_LABEL = {"en_correct": "answers in English", "zh_correct": "answers in Chinese",
              "fr_correct": "answers in French", "letter_correct": "presses the key"}
CHAN_HUE = {"en_correct": ORANGE, "zh_correct": AQUA, "fr_correct": "#eda100",
            "letter_correct": BLUE}


def f19_language(df: pd.DataFrame, out: str, name="f19_language.png"):
    arms = [a for a in LANG_LABEL if a in set(df.arm)]
    cols = [c for c in CHAN_LABEL if c in df.columns]
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    x = np.arange(len(arms))
    w = 0.78 / len(cols)
    for j, c in enumerate(cols):
        off = (j - (len(cols) - 1) / 2) * (w + 0.012)
        g = df.groupby("arm")[c].mean().reindex(arms) * 100
        e = df.groupby("arm")[c].sem().reindex(arms).fillna(0) * 100
        ax.bar(x + off, g, w, yerr=e, color=CHAN_HUE[c], label=CHAN_LABEL[c],
               error_kw=dict(lw=1, ecolor=MUTED))
    ax.axhline(50, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xticks(x); ax.set_xticklabels([LANG_LABEL[a] for a in arms])
    ax.set_xlabel("which language's directions were erased")
    ax.set_ylabel("correct  (%)"); ax.set_ylim(0, 100)
    style.title(ax, "One buffer per language, and none for the key",
                "Erasing a language costs that language. No amount of erasing "
                "costs the decision.", legend_rows=1)
    style.legend_above(ax, ncols=4)
    return _save(fig, out, name)


# ----------------------------------------------------------------- F20 source

def f20_source(df: pd.DataFrame, out: str, name="f20_source.png"):
    """It can check the picture; it cannot say which thing came from it."""
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    spec = [("presence_correct", "is each object\nvisible?", C_KEY),
            ("source_correct", "which one can\nyou see?", C_WORD),
            ("told_correct", "which one were you\nonly told about?", C_WORD),
            ("consistent", "are those two answers\ndifferent objects?", VIOLET)]
    spec = [s for s in spec if s[0] in df.columns]
    vals = [df[k].mean() * 100 for k, _, _ in spec]
    errs = [df[k].sem() * 100 for k, _, _ in spec]
    ax.bar(range(len(spec)), vals, 0.56, yerr=errs,
           color=[c for _, _, c in spec], error_kw=dict(lw=1, ecolor=MUTED))
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.0f}", xy=(i, v), xytext=(0, 6), ha="center",
                    textcoords="offset points", color=INK2, fontsize=12.5)
    ax.axhline(50, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.annotate("chance", xy=(len(spec) - 0.42, 50), xytext=(5, 3),
                textcoords="offset points", color=MUTED, fontsize=11,
                annotation_clip=False)
    ax.set_xticks(range(len(spec)))
    ax.set_xticklabels([lab for _, lab, _ in spec])
    ax.set_ylabel("correct  (%)"); ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    style.title(ax, "Seeing is fine. Saying which one it saw is not.",
                "Same photograph, same sentence, same letter-answer format; "
                "both option orders averaged.")
    return _save(fig, out, name)


# ----------------------------------------------------------------- F21 secant

LENS_STYLE = {"logit lens": (MUTED, "logit lens"),
              "Jacobian lens": (INK2, "Jacobian lens, as published"),
              "our tangent": (BLUE, "our tangent"),
              "the secant": (C_WORD, "the secant")}


def f21_secant(stats: dict, summ: pd.DataFrame, out: str,
               name="f21_secant.png", refit: dict | None = None):
    """What each lens predicts, how far it stays true, and what it reads.

    ``refit`` is the offline recomputation with a fit / choose / report split;
    when it is available the fidelity panels come from it, because the online
    numbers chose the shrinkage parameter on the photographs they reported on.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.0))
    if refit:
        lad = {l: (refit[l].get("full_patch") or refit[l]["ladder"])
               for l in refit}
        rad_all = {l: refit[l].get("radius", {}) for l in refit}
    else:
        lad = stats.get("ladder", {})
        rad_all = stats.get("radius", {})
    layers = sorted(lad, key=int)
    names = [n for n in LENS_STYLE if any(n in lad[l] for l in layers)]

    # --- A: the ceiling, and where each lens sits under it
    ax = axes[0]
    x = np.arange(len(layers))
    w = 0.8 / max(len(names), 1)
    for j, n in enumerate(names):
        c, lab = LENS_STYLE[n]
        v = [lad[l].get(n, {}).get("fve_scaled", np.nan) for l in layers]
        ax.bar(x + (j - (len(names) - 1) / 2) * w, v, w * 0.92, color=c,
               label=lab)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"layer {l}" for l in layers])
    ax.set_ylabel("variance explained in the true effect of a\nreal resample patch, at each lens's best gain")
    style.title(ax, "How much of a real patch each lens predicts",
                "Each lens at its best gain, on held-out photographs.",
                width=48)
    style.legend(ax)

    # --- B: how far from an activation a linear surrogate stays true
    ax = axes[1]
    lm = layers[len(layers) // 2] if layers else None
    got = rad_all.get(lm, {}) if lm else {}
    if got:
        ts = sorted(got, key=float)
        for n in names:
            c, lab = LENS_STYLE[n]
            ys = [got[t].get(n, {}).get("fve_scaled", np.nan) for t in ts]
            ax.plot([float(t) for t in ts], ys, color=c, marker="o", ms=5,
                    label=lab)
        ax.set_xscale("log")
        _log_ticks(ax, [float(t) for t in ts])
        ax.set_xlabel("size of the counterfactual   $t$")
        ax.set_ylabel("held-out variance explained")
        ax.axhline(0, color=GRID, lw=1)
        style.title(ax, "How far a lens stays true",
                    f"At layer {lm}. $t=1$ replaces the patch with "
                    "another photograph's.", width=48)
        style.legend(ax)
    else:
        ax.axis("off")

    # --- C: what each surrogate reads off an image token
    ax = axes[2]
    KEY = {"logit-lens": "logit lens", "reference": "Jacobian lens",
           "tangent": "our tangent", "secant": "the secant"}
    for rung, n in KEY.items():
        d = summ[summ.rung == rung].sort_values("depth")
        if not len(d):
            continue
        c, lab = LENS_STYLE[n]
        ax.plot(d.depth * 100, d.auc, color=c, marker="o", ms=5, label=lab)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xlabel("depth through the language model  (%)")
    ax.set_ylabel("present objects ranked above absent ones\n(AUC)")
    style.title(ax, "And what each one reads",
                "Whether the object's own name outranks objects that are not "
                "in the picture.", width=48)
    style.legend(ax)
    fig.tight_layout()
    return _save(fig, out, name)


def f22_certificate(stats: dict, out: str, name="f22_certificate.png"):
    """Why a secant and a tangent cannot agree here."""
    ell = stats.get("ellipticity", {})
    ls = sorted(ell, key=int)
    if not ls:
        raise ValueError("no ellipticity measurement")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    ax = axes[0]
    x = np.arange(len(ls))
    ax.bar(x - 0.19, [ell[l]["kurtosis_spread"] for l in ls], 0.36,
           color=C_WORD, label="image-site deviations")
    ax.bar(x + 0.19, [ell[l]["gaussian_null"] for l in ls], 0.36, color=MUTED,
           label="a Gaussian of the same size")
    ax.set_xticks(x); ax.set_xticklabels([f"layer {l}" for l in ls])
    ax.set_ylabel("spread of kurtosis across directions")
    style.title(ax, "The deviations are not elliptical",
                "Any elliptical law has direction-independent whitened "
                "kurtosis, and would force the two lenses to coincide.")
    style.legend(ax)

    ax = axes[1]
    fit = stats.get("fit", {})
    gs = [fit[l].get("relative_gap") for l in ls if l in fit]
    ax.bar(np.arange(len(gs)), gs, 0.5, color=BLUE)
    ax.set_xticks(np.arange(len(gs)))
    ax.set_xticklabels([f"layer {l}" for l in ls][: len(gs)])
    ax.set_ylabel(r"$\|L^* - \bar{A}\| \, / \, \|\bar{A}\|$")
    style.title(ax, "And so they do not coincide",
                "Distance between the fitted map and the derivative.")
    fig.tight_layout()
    return _save(fig, out, name)



def f23_tradeoff(stats: dict, summ: pd.DataFrame, out: str,
                 name="f23_tradeoff.png"):
    """One knob, two criteria, opposite directions."""
    path = stats.get("shrinkage_path", {})
    layers = sorted(path, key=int)
    if not layers:
        raise ValueError("no shrinkage path recorded")
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    cols = [C_WORD, BLUE, AQUA, VIOLET, ORANGE]
    ends = []
    for l, c in zip(layers, cols):
        fve = {round(p["lambda"], 6): p["held_out_fve"] for p in path[l]}
        pts = []
        for _, r in summ[summ.layer == int(l)].iterrows():
            if not str(r.rung).startswith("shrink "):
                continue
            lam = float(str(r.rung).split()[1])
            key = min(fve, key=lambda k: abs(k - lam))
            pts.append((fve[key], r.auc, lam))
        if len(pts) < 3:
            continue
        pts.sort()
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=c, marker="o", ms=4, lw=1.4,
                label=f"layer {l}")
        ends.append((xs[0], ys[0], xs[-1], ys[-1], c))
    # Every layer starts at "the derivative" and ends at "the fitted secant",
    # so naming both ends of all four curves printed the same two phrases on
    # top of each other four times.  Name the two ends once, on the extreme
    # curve, and let the legend carry which layer is which.
    if ends:
        a = min(ends, key=lambda e: e[0])
        b = max(ends, key=lambda e: e[3])
        ax.annotate("all four start at the derivative", (a[0], a[1]),
                    xytext=(10, -16), textcoords="offset points",
                    fontsize=11.5, color=INK2, ha="left",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9,
                                    shrinkA=1, shrinkB=5))
        ax.annotate("and end at the fitted secant", (b[2], b[3]),
                    xytext=(-10, 14), textcoords="offset points",
                    fontsize=11.5, color=INK2, ha="right",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9,
                                    shrinkA=1, shrinkB=5))
    ax.set_xlabel("held-out variance explained in the true patching effect")
    ax.set_ylabel("reads the photographed object's name  (AUC)")
    style.title(ax, "One scalar, moved from the derivative to the data",
                "One lens, one layer, swept by its shrinkage parameter.")
    style.legend(ax, loc="lower right")
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0 - 0.06 * (x1 - x0), x1 + 0.10 * (x1 - x0))
    fig.tight_layout()
    return _save(fig, out, name)


def f24_evalgate(df: pd.DataFrame, out: str, name="f24_evalgate.png"):
    """Obeying a painted-in order, against how much of the residual was removed.

    Removing more of the residual stream moves behaviour whatever it points at,
    so the random arm's own dose curve is the null and the question is whether a
    vocabulary sits above it.
    """
    off = df[df.frame == "official"]
    g = off.groupby("cond").agg(e=("energy", "mean"), y=("said_banana", "mean"),
                                arm=("arm", "first"))
    rnd = g[g.arm == "random"].sort_values("e")
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    if len(rnd) >= 2:
        sl, ic = np.polyfit(rnd.e.values, rnd.y.values, 1)
        xs = np.linspace(0, max(g.e) * 1.08, 50)
        ax.plot(xs, sl * xs + ic, color=MUTED, lw=1.2, ls=(0, (4, 3)),
                label="random directions, same rank")
        ax.scatter(rnd.e, rnd.y, s=44, color=MUTED, zorder=3)
    STYLE = {"eval": (C_WORD, "evaluation words"),
             "eval_full": (RED, "evaluation words, full set"),
             "authority": (BLUE, "authority words"),
             "colour": (AQUA, "colour words"),
             "clean": (INK2, "nothing removed")}
    for arm, (c, lab) in STYLE.items():
        d = g[g.arm == arm]
        if not len(d):
            continue
        ax.scatter(d.e, d.y, s=62, color=c, zorder=4, label=lab)
        for cond, r in d.iterrows():
            ax.annotate(cond.split("@")[-1] if "@" in cond else "",
                        (r.e, r.y), fontsize=8, color=c,
                        xytext=(6, -3), textcoords="offset points")
    ax.set_xlabel(r"removed energy   $\|\Delta h\| / \|h\|$")
    ax.set_ylabel("obeys the order painted into the photograph")
    style.title(ax, "Refusal runs partly through knowing it is a test",
                "Above the line is more obedience than the damage alone explains.")
    style.legend(ax)
    fig.tight_layout()
    return _save(fig, out, name)


def f25_crossing(refit: dict, summ: pd.DataFrame, out: str,
                 name="f25_crossing.png"):
    """The same four lenses, ranked twice, against depth."""
    KEY = {"logit-lens": "logit lens", "reference": "Jacobian lens",
           "tangent": "the corrected derivative", "secant": "VJ-Lens"}
    STYLE = {"logit lens": (MUTED, "logit lens"),
             "Jacobian lens": (INK2, "Jacobian lens"),
             "the corrected derivative": (BLUE, "corrected derivative"),
             "VJ-Lens": (C_WORD, "VJ-Lens")}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharex=True)

    ax = axes[0]
    for rung, n in KEY.items():
        d = summ[summ.rung == rung].sort_values("depth")
        if not len(d):
            continue
        c, lab = STYLE[n]
        ax.plot(d.depth * 100, d.auc, color=c, marker="o", ms=6, label=lab)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("depth through the language model  (%)")
    ax.set_ylabel("names the photographed object\n(AUC over 65 categories)")
    style.title(ax, "How well each lens reads",
                "Higher is a better reading of what is in the picture.")
    style.legend(ax, loc="lower left")

    ax = axes[1]
    layers = sorted(refit, key=int)
    dep = summ.groupby("layer").depth.first()
    for rung, n in KEY.items():
        c, lab = STYLE[n]
        xs, ys = [], []
        for l in layers:
            v = (refit[l].get("full_patch") or refit[l]["ladder"]).get(n)
            if v is None:
                continue
            xs.append(float(dep.get(int(l), np.nan)) * 100)
            ys.append(v["fve_scaled"])
        if xs:
            ax.plot(xs, ys, color=c, marker="o", ms=6, label=lab)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xlabel("depth through the language model  (%)")
    ax.set_ylabel("predicts a real patch\n(held-out variance explained)")
    style.title(ax, "How well each lens predicts",
                "The mirror of the panel on the left.")
    style.legend(ax, loc="upper right")
    fig.tight_layout()
    return _save(fig, out, name)


def f26_radius(refit: dict, out: str, name="f26_radius.png"):
    """How far from an activation a linear lens stays true."""
    NAMES = ["logit lens", "Jacobian lens", "our tangent", "the secant"]
    LAB = {"logit lens": (MUTED, "logit lens"),
           "Jacobian lens": (INK2, "Jacobian lens"),
           "our tangent": (BLUE, "corrected derivative"),
           "the secant": (C_WORD, "VJ-Lens")}
    layers = sorted(refit, key=int)
    fig, axes = plt.subplots(1, len(layers), figsize=(3.5 * len(layers), 4.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, l in zip(axes, layers):
        rad = refit[l].get("radius", {})
        ts = sorted(rad, key=float)
        for n in NAMES:
            c, lab = LAB[n]
            ys = [rad[t].get(n, {}).get("fve_scaled", np.nan) for t in ts]
            ax.plot([float(t) for t in ts], ys, color=c, marker="o", ms=4,
                    label=lab)
        ax.set_xscale("log")
        _log_ticks(ax, [float(t) for t in ts])
        ax.axhline(0, color=GRID, lw=1)
        ax.set_xlabel("counterfactual size  $t$")
        # a per-panel label inside the axes: set_title here fights the
        # figure title that the first panel also carries
        style.panel(ax, f"layer {l}", loc="upper left")
    axes[0].set_ylabel("held-out variance explained")
    # every panel's data hugs the bottom left, so the legend goes top right
    style.legend(axes[-1], loc="upper right")
    fig.suptitle("A linear lens is more faithful for big changes", x=0.004,
                 ha="left", fontsize=plt.rcParams["figure.titlesize"],
                 color=INK)
    fig.text(0.004, 0.925,
             "$t=1$ replaces the patch with another photograph's; smaller "
             "$t$ moves it part of the way.", ha="left", va="top",
             color=INK2, fontsize=plt.rcParams["axes.labelsize"] * 0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    return _save(fig, out, name)


def f27_traject(traj: dict, out: str, name="f27_traject.png"):
    """Every photograph's trajectory, and the median through them."""
    import math
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharey=True)
    layers = traj["layers"]
    dep = [traj["depths"][str(l)] * 100 for l in layers]
    n_vocab = len(traj["vocab"])

    for ax, head, c, lab in ((axes[0], "say", C_WORD, "the plain logit lens"),
                             (axes[1], "do", BLUE, "VJ-Lens")):
        curves = []
        for card in traj["cards"]:
            xs, ys = [], []
            for l, d in zip(layers, dep):
                cell = card[head].get(str(l))
                if cell:
                    xs.append(d)
                    ys.append(max(cell["rank"], 0.7))
            if len(xs) > 1:
                ax.plot(xs, ys, color=c, lw=0.7, alpha=0.16)
                curves.append((xs, ys))
        # the median through them, on the layers every photograph has
        common = sorted({x for xs, _ in curves for x in xs})
        med = []
        for x in common:
            vals = [ys[xs.index(x)] for xs, ys in curves if x in xs]
            med.append(np.median(vals))
        ax.plot(common, med, color=c, lw=2.4, marker="o", ms=5,
                label="median over photographs")
        ax.set_yscale("log")
        ax.set_ylim(0.7, n_vocab)
        ax.invert_yaxis()
        ax.set_xlabel("depth through the language model  (%)")
        style.legend(ax, loc="lower left")
        style.title(ax, lab,
                    "Rank of the photographed object among "
                    f"{n_vocab:,} tokens. Lower is better; the axis is "
                    "inverted so up is better.")
    axes[0].set_ylabel("rank of the photographed object")
    fig.tight_layout()
    return _save(fig, out, name)


def f31_validate(summ: pd.DataFrame, stats: dict, out: str,
                 name="f31_validate.png"):
    """Two jobs, two winners: predicting the effect, and finding the object."""
    LAB = {"logit lens": (C_WORD, "logit lens  ($say$)"),
           "V-Lens (donor baseline)": (BLUE, "VJ-Lens"),
           "V-Lens (mean baseline)": (AQUA, "VJ-Lens  (corpus-mean baseline)")}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    ax = axes[0]
    cor = stats.get("predicts_drop", {})
    ls = sorted(cor, key=int)
    # the summary carries layer, not depth, for this experiment
    n_layers = stats.get("n_layers") or (max(int(x) for x in ls) + 3)
    xs = [100.0 * int(l) / (n_layers - 1) for l in ls]
    ax.plot(xs, [cor[l]["do_spearman"] for l in ls], color=BLUE, marker="o",
            ms=6, label="VJ-Lens")
    ax.plot(xs, [cor[l]["say_spearman"] for l in ls], color=C_WORD, marker="o",
            ms=6, label="logit lens  ($say$)")
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xlabel("depth through the language model  (%)")
    ax.set_ylabel("predicts what covering the object does\n(Spearman)")
    # width=52 keeps this panel's subtitle inside its own axes; at the
    # default it ran across the gutter and printed over the next panel's
    # title.
    style.title(ax, "Which patches matter",
                "Cover an object's tokens, swap them for another "
                "photograph's, and see what the model's score for that "
                "object really does.", width=52)
    style.legend(ax, loc="lower right")

    ax = axes[1]
    for lens, (c, lab) in LAB.items():
        d = summ[summ.lens == lens].sort_values("layer")
        if not len(d):
            continue
        ax.plot(100.0 * d.layer / (n_layers - 1), d.point, color=c,
                marker="o", ms=6, label=lab)
    ch = float(summ.chance.mean())
    ax.axhline(ch, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.annotate("chance", xy=(100.0 * summ.layer.max() / (n_layers - 1), ch),
                xytext=(-4, 5),
                textcoords="offset points", ha="right", color=MUTED,
                fontsize=8)
    ax.set_ylim(0, max(0.7, summ.point.max() * 1.15))
    ax.set_xlabel("depth through the language model  (%)")
    ax.set_ylabel("strongest patch lands on the object")
    style.title(ax, "Where the object is",
                "The pointing game: does the map's peak fall inside the "
                "object's own box?", width=52)
    style.legend(ax, loc="upper left")
    fig.tight_layout()
    return _save(fig, out, name)


def f32_deletion(curve: pd.DataFrame, stats: dict, out: str,
                 name="f32_deletion.png"):
    """How fast the object dies when each map's patches are covered first."""
    STY = {"V-Lens": (BLUE, "VJ-Lens"),
           "logit lens": (C_WORD, "logit lens  ($say$)"),
           "random": (MUTED, "random order")}
    layers = sorted(curve.layer.unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(3.6 * len(layers), 4.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    n_layers = stats.get("n_layers", max(layers) + 3)
    for ax, l in zip(axes, layers):
        d = curve[curve.layer == l]
        for o, (c, lab) in STY.items():
            s = d[d.order == o].sort_values("frac")
            if len(s):
                ax.plot(s.frac * 100, s.rel, color=c, marker="o", ms=4,
                        label=lab)
        ax.set_xlabel("share of image tokens covered  (%)")
        style.panel(ax, f"{100 * l / (n_layers - 1):.0f}% depth",
                    loc="upper right")
        ax.axhline(0, color=GRID, lw=1)
    axes[0].set_ylabel("share of the object's score left")
    style.legend(axes[-1], loc="lower left")
    fig.suptitle("Covering the patches each map ranks highest", x=0.004,
                 ha="left", fontsize=plt.rcParams["figure.titlesize"],
                 color=INK)
    fig.text(0.004, 0.928, "Falling faster is a better ordering.",
             ha="left", va="top", color=INK2,
             fontsize=plt.rcParams["axes.labelsize"] * 0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.895))
    return _save(fig, out, name)


def f40_swap(df: pd.DataFrame, out: str, name="f40_swap.png"):
    """Swap strength against how often the model believes the swap."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    KIND = {"colour": (C_WORD, "what colour is it"),
            "legs": (BLUE, "how many legs"),
            "wheels": (AQUA, "how many wheels")}
    ax = axes[0]
    for kind, (c, lab) in KIND.items():
        d = df[df.kind == kind].groupby("alpha").picked_dst.mean()
        if len(d):
            ax.plot(d.index, d.values * 100, color=c, marker="o", ms=6,
                    label=lab)
    tot = df.groupby("alpha").picked_dst.mean()
    ax.plot(tot.index, tot.values * 100, color=INK2, marker="s", ms=6, lw=2.2,
            label="all questions")
    ax.set_xlabel(r"swap strength  $\alpha$")
    ax.set_ylabel("model gives the swapped answer  (%)")
    ax.set_ylim(0, 75)
    style.legend(ax, loc="upper left")

    ax = axes[1]
    d = df.groupby("alpha").agg(dst=("picked_dst", "mean"),
                                src=("said_src", "mean"))
    w = 0.16
    x = np.arange(len(d))
    ax.bar(x - w / 2, d.dst * 100, w * 1.6, color=BLUE,
           label="says the swapped answer")
    ax.bar(x + w / 2, d.src * 100, w * 1.6, color=MUTED,
           label="still says the photograph's")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\alpha$ = {v:g}" for v in d.index])
    ax.set_ylabel("share of trials  (%)")
    # loc="upper right" put the two-line legend straight across the tallest
    # pair of bars; the strip above the axes is the only place it does not
    # cover data.
    ax.set_ylim(0, max(d.dst.max(), d.src.max()) * 100 * 1.12)
    style.title(ax, "And the original answer goes away",
                "Before any edit the photograph's answer is given on every "
                "retained trial.", legend_rows=1, width=46)
    style.legend_above(ax, ncols=2)

    # the left panel's title is long enough to run into the right panel's
    style.title(axes[0], "The answer follows the swap",
                "Between the two answers the concepts disagree about; the "
                "model is answering about the photograph it was shown.",
                width=46)
    fig.tight_layout()
    return _save(fig, out, name)


def f41_lenses(stats: dict, out: str, name="f41_lenses.png"):
    """Three lenses, four measures, best over depth."""
    best = stats.get("best", {})
    if not best:
        raise ValueError("no comparison yet")
    _vj = _vj_key(best) or "VJ-Lens"
    LENS = [("logit lens", C_WORD), ("J-lens", VIOLET), (_vj, BLUE)]
    NICE = {_vj: "VJ-Lens"}
    MEAS = [("points", "peak inside the box", False),
            ("top5", "top five inside the box", False),
            ("energy", "map mass inside the box", False),
            ("predicts", "predicts the real damage", False),
            ("deletion", "deletion area (lower better)", True)]
    MEAS = [t for t in MEAS if any(t[0] in best.get(l, {}) for l, _ in LENS)]
    fig, axes = plt.subplots(1, len(MEAS), figsize=(2.9 * len(MEAS), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (key, lab, lower) in zip(axes, MEAS):
        vals, cols, names = [], [], []
        for l, c in LENS:
            if key in best.get(l, {}):
                vals.append(best[l][key]); cols.append(c); names.append(l)
        ax.bar(np.arange(len(vals)), vals, 0.62, color=cols)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels([NICE.get(n, n) for n in names], fontsize=10.5)
        ax.set_title(lab + ("  ↓" if lower else ""), fontsize=11,
                     color=INK2, pad=6)
        for xx, vv in enumerate(vals):
            ax.annotate(f"{vv:.2f}", xy=(xx, vv), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        color=INK2, fontsize=10)
        ax.set_ylim(0, max(vals) * 1.20)
        if key in ("points", "top5", "energy") and "chance" in stats:
            ax.axhline(stats["chance"], color=MUTED, lw=1, ls=(0, (4, 3)))
    axes[0].set_ylabel("best over depth")
    fig.suptitle("Five measures, each lens at its own best depth", x=0.005,
                 ha="left", fontsize=plt.rcParams["figure.titlesize"],
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out, name)


def f42_attacks(stats: dict, out: str, name="f42_attacks.png"):
    """Compliance per attack route, and which lens sees it happening."""
    summ, det = stats.get("summary", {}), stats.get("detect", {})
    if not summ:
        raise ValueError("no attack results")
    NICE = {"refuseable": "plain words", "direct": "named directly",
            "cipher": "visual cipher", "substitution": "object substitution",
            "replacement": "in-image text", "analogy": "visual analogy"}
    order = [k for k in ("refuseable", "direct", "cipher", "substitution",
                         "replacement", "analogy") if k in summ]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    ax = axes[0]
    v = [summ[k].get("complied", 0.0) * 100 for k in order]
    cols = [MUTED if k in ("refuseable", "direct") else C_WORD for k in order]
    ax.barh(np.arange(len(order))[::-1], v, 0.62, color=cols)
    ax.set_yticks(np.arange(len(order))[::-1])
    ax.set_yticklabels([NICE.get(k, k) for k in order], fontsize=9)
    ax.set_xlabel("model complies  (%)")
    style.title(ax, "Four routes through the picture",
                "Grey is the request made plainly, which the model refuses.")

    ax = axes[1]
    fams = [("logit", C_WORD, "logit lens"), ("jlens", VIOLET, "J-lens"),
            ("atlas", BLUE, "VJ-Lens")]
    atk = [k for k in order if k in det]
    x = np.arange(len(atk))
    w = 0.8 / max(len(fams), 1)
    for j, (f, c, lab) in enumerate(fams):
        vals = [det[k].get(f, np.nan) for k in atk]
        if all(np.isnan(vv) for vv in vals):
            continue
        ax.bar(x + (j - (len(fams) - 1) / 2) * w, vals, w * 0.9, color=c,
               label=lab)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels([NICE.get(k, k) for k in atk], fontsize=8,
                       rotation=18, ha="right")
    ax.set_ylabel("tells the attack from an ordinary request  (AUROC)")
    style.title(ax, "And which lens notices",
                "Read at the image tokens, before the model answers.")
    style.legend(ax, loc="upper left")
    fig.tight_layout()
    return _save(fig, out, name)


def f43_posvar(stats: dict, out: str, name="f43_posvar.png"):
    """How much each lens's reading changes from place to place in a picture."""
    by = stats.get("by_layer", {})
    if not by:
        raise ValueError("no position-variation results")
    d = pd.DataFrame(list(by.values()) if isinstance(by, dict) else by)
    if "depth" not in d.columns:
        nl = stats.get("n_layers") or (int(d.layer.max()) + 3)
        d["depth"] = d.layer / (nl - 1)
    STY = {"logit lens": (C_WORD, "logit lens"), "J-lens": (VIOLET, "J-lens")}
    STY.update({k: (BLUE, "VJ-Lens") for k in _VJ_ALIASES})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    P = stats.get("n_patches", 196)
    for ax, (key, lab, inv) in zip(
            axes, (("distinct", f"distinct top words across {P} patches", False),
                   ("agree", "two random patches give the same word", True))):
        for lens, (c, nm) in STY.items():
            s = d[d.lens == lens].sort_values("layer")
            if len(s):
                ax.plot(s.depth * 100, s[key], color=c, marker="o", ms=6,
                        label=nm)
        ax.set_xlabel("depth through the language model  (%)")
        ax.set_ylabel(lab)
        style.legend(ax, loc="best")
    style.title(axes[0], "A lens with no position in its transport",
                "cannot tell one part of a picture from another.")
    fig.tight_layout()
    return _save(fig, out, name)


# The method was called ATLAS, then V-Lens, before it was named VJ-Lens.
# Results on disk still carry the old keys, so every figure resolves a lens
# name through here rather than matching a literal -- a figure that silently
# drops its own method's bar because the key was renamed is worse than one
# that crashes.
_VJ_ALIASES = ("VJ-Lens", "ATLAS", "Atlas", "atlas", "V-Lens", "vjlens", "do")


def _vj_key(available):
    """Whichever spelling of VJ-Lens this results file happens to use."""
    for k in _VJ_ALIASES:
        if k in available:
            return k
    return None


# ------------------------------------------------- F50 the word grid, legibly

_WG_COLOUR = {"logit lens": C_WORD, "logit lens (centred)": MUTED,
              "J-lens": VIOLET, "VJ-Lens": BLUE}
_WG_ORDER = ["logit lens", "logit lens (centred)", "J-lens", "VJ-Lens"]


def _wg_key(lens: str) -> str:
    """The lens family a variant belongs to, for colour and ordering."""
    return lens if lens in _WG_COLOUR else lens.split(" (")[0]


def _wg_label(lens: str) -> str:
    """What the reader should see: the family, and whose lens it is."""
    k = _wg_key(lens)
    return "VJ-Lens  (ours)" if k == "VJ-Lens" else k


def _wg_pick(available, want=("logit lens", "J-lens", "VJ-Lens")):
    """One variant per family: the plain reading unless asked otherwise.

    This used to prefer the direction-only arm, on the argument that a raw dot
    product partly measures how long the lens vector is, which is a fact about
    the word rather than about the picture.  That arm was measured and it is
    worse, and worst for the lens this paper is about: over the same twelve
    photographs the VJ-Lens margin between tiles inside the object's box and
    tiles outside it is -0.015 plain and -0.082 direction-only, and the
    direction arm nearly doubles the share of *off*-object tiles printing the
    object's name (0.084 to 0.151).  Showing the arm that flatters no one and
    hurts our own column is not a defensible default, so the plain reading is
    the default and ``lenses=`` still selects any arm explicitly.
    """
    out = []
    for fam in want:
        for suffix in ("", " (direction)"):
            if fam + suffix in available:
                out.append(fam + suffix)
                break
    return out


def _log_ticks(ax, xs, fmt="{:g}"):
    """Label a log axis at the values that were measured, and nowhere else.

    Matplotlib's default minor ticks on a log axis over less than a decade
    print "2 x 10^0", "4 x 10^-1", "6 x 10^-1" on top of one another; three
    panels of this paper shipped with that smear along the bottom.
    """
    xs = sorted(set(float(v) for v in xs))
    ax.set_xticks(xs)
    ax.set_xticklabels([fmt.format(v) for v in xs])
    ax.minorticks_off()
    ax.tick_params(axis="x", which="minor", bottom=False)


def _tint(hex_colour: str, f: float):
    """``f`` of the way from the page colour to ``hex_colour``."""
    import matplotlib.colors as mc
    a = np.array(mc.to_rgb(SURFACE))
    b = np.array(mc.to_rgb(hex_colour))
    return tuple(a + (b - a) * float(np.clip(f, 0, 1)))


def _boxed(ax, boxes, sx, sy, lw=2.1):
    """The object's bounding box, with a light halo so it reads on anything."""
    for (x0, y0, w, h) in boxes:
        for c, l, z in ((SURFACE, lw + 2.6, 4), (INK, lw, 5)):
            ax.add_patch(plt.Rectangle((x0 * sx, y0 * sy), w * sx, h * sy,
                                       fill=False, lw=l, edgecolor=c,
                                       zorder=z, joinstyle="miter"))


def _fit(fig, ax, x, y, s, maxw, base, colour=INK):
    """Centre ``s`` in a tile, shrunk just enough that it cannot collide."""
    t = ax.text(x, y, s, ha="center", va="center", fontsize=base,
                color=colour, zorder=6)
    r = fig.canvas.get_renderer()
    w = t.get_window_extent(renderer=r).width
    if w > maxw:
        t.set_fontsize(max(5.4, base * maxw / w))
    return t


def _one_size(fig, ax, words, maxw, base):
    """The single point size at which the *longest* word still fits a tile.

    Sizing each tile on its own makes "computer" render half as tall as
    "snow" in the neighbouring cell, which reads as a ransom note and makes
    the long words -- the interesting ones -- the hardest to see.  One size
    for the whole grid keeps every word equally legible and keeps the shade,
    not the type size, carrying the measurement.
    """
    r = fig.canvas.get_renderer()
    fs = float(base)
    probe = ax.text(0, 0, "", fontsize=base, alpha=0.0)
    for w in set(words):
        probe.set_text(w)
        px = probe.get_window_extent(renderer=r).width
        if px > maxw:
            fs = min(fs, base * maxw / px)
    probe.remove()
    return max(5.6, fs)


def _boxed_tiles(ax, inside, C, lw=2.4):
    """Outline exactly the tiles the object's box covers.

    The box is a rectangle in the photograph, but the reading is per tile, so
    drawing the raw rectangle over a word grid puts a line straight through
    the middle of whichever words the box edge happens to cross -- every
    seven-by-seven card in the first draft had a struck-through bottom row.
    Snapping the outline to the tile boundaries removes the collision by
    construction and, better, draws the set the inside/outside numbers are
    actually computed over.
    """
    from matplotlib.collections import LineCollection

    have = set(int(t) for t in inside)
    seg = []
    for t in have:
        r, c = divmod(t, C)
        if t - C not in have or r == 0:
            seg.append([(c, r), (c + 1, r)])
        if t + C not in have or r == C - 1:
            seg.append([(c, r + 1), (c + 1, r + 1)])
        if c == 0 or (t - 1) not in have:
            seg.append([(c, r), (c, r + 1)])
        if c == C - 1 or (t + 1) not in have:
            seg.append([(c + 1, r), (c + 1, r + 1)])
    if not seg:
        return
    ax.add_collection(LineCollection(seg, colors=SURFACE, linewidths=lw + 2.6,
                                     zorder=7, capstyle="round"))
    ax.add_collection(LineCollection(seg, colors=INK, linewidths=lw,
                                     zorder=8, capstyle="butt"))


def f50_wordgrid(stats: dict, out: str, name="f50_wordgrid.png", which=0,
                 lenses=None, base_fs=13.0, headline=None, note=None):
    """One photograph, one word per tile, three lenses reading the same places.

    The whole point is that every tile is legible: the word is the measurement
    and the shade is only how far that word stands above the rest of the
    vocabulary, so nothing here has to be taken on trust.
    """
    import base64
    import io
    import textwrap

    from PIL import Image

    cards = stats.get("cards") or []
    if not cards:
        raise ValueError("no word-grid cards")
    card = cards[which % len(cards)]
    C = int(stats["coarse"])
    best = stats.get("best", {})
    shown = [l for l in (lenses or _wg_pick(card["lenses"]))
             if l in card["lenses"]]
    obj = card["concept"]

    def pick(lens):
        per = card["lenses"][lens]
        key = str(int(best.get(lens, {}).get("layer", -1)))
        if key not in per:
            key = sorted(per, key=int)[len(per) // 2]
        return key, per[key]

    # The panels are square, so the figure's height is not free: it is solved
    # for, so that a grid cell comes out exactly as tall as it is wide.  Fixing
    # it by eye instead leaves a band of dead space between the two rows, which
    # is the complaint this figure exists to answer.
    ncol = 1 + len(shown)
    half = max(1, ncol // 2)
    L, R, T, B, WS, HS = 0.030, 0.988, 0.820, 0.155, 0.085, 0.42
    HR = [3.0, 1.15]
    fw = 4.5 * ncol
    cell_w = (R - L) / (ncol + (ncol - 1) * WS)
    row0 = (T - B) / (1 + HS * 0.5) * HR[0] / sum(HR)
    fh = fw * cell_w / row0
    fig = plt.figure(figsize=(fw, fh))
    gs = fig.add_gridspec(2, ncol, height_ratios=HR, left=L, right=R,
                          top=T, bottom=B, wspace=WS, hspace=HS)

    # ---- the photograph -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    img = Image.open(io.BytesIO(base64.b64decode(card["img"].split(",")[1])))
    ax.imshow(img)
    _boxed(ax, card["boxes"], img.width, img.height, lw=2.4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
        sp.set_visible(True)
    cap = card.get("caption", "").replace("\n", " ").strip().rstrip(".")
    cap = (cap[:40] + "\u2026") if len(cap) > 41 else cap
    style.title(ax, "the photograph",
                f"The box is the {obj}. Asked to name the main object in one "
                f"word, the model answers \u201c{cap}\u201d.", width=50)

    # ---- one word grid per lens --------------------------------------------
    axes = {}
    for j, lens in enumerate(shown):
        a2 = fig.add_subplot(gs[0, 1 + j])
        a2.set_xlim(0, C); a2.set_ylim(C, 0); a2.set_aspect("equal")
        a2.set_xticks([]); a2.set_yticks([]); a2.grid(False)
        for sp in a2.spines.values():
            sp.set_edgecolor(GRID)
            sp.set_visible(True)
        axes[lens] = a2
    fig.canvas.draw()

    for lens in shown:
        a2 = axes[lens]
        key, d = pick(lens)
        words, z = d["words"], np.asarray(d["z"], dtype=float)
        lo, hi = float(np.percentile(z, 5)), float(np.percentile(z, 95))
        rel = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
        px = (a2.transData.transform((1, 0))[0]
              - a2.transData.transform((0, 0))[0])
        fs = _one_size(fig, a2, words, px * 0.86, base_fs)
        for t, w in enumerate(words):
            r, c = divmod(t, C)
            on = (w == obj)
            face = _tint(AQUA, 0.26 + 0.44 * rel[t]) if on \
                else _tint(INK2, 0.05 + 0.20 * rel[t])
            a2.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=face,
                                       edgecolor=SURFACE, lw=1.1, zorder=1))
            a2.text(c + 0.5, r + 0.5, w, ha="center", va="center",
                    fontsize=fs, color=INK, zorder=6)
        _boxed_tiles(a2, card["inside"], C, lw=2.4)
        nin = len(card["inside"])
        hit = int(round(d["inside"] * nin))
        nice = _wg_label(lens)
        style.title(
            a2, nice,
            f"{d['distinct']} different words in {C*C} tiles.  "
            f"Reads \u201c{obj}\u201d in {hit} of the {nin} box tiles.  "
            f"Layer {key} of {stats.get('n_layers', '?')}.", width=50)

    # ---- and what it looks like over the whole set --------------------------
    have = [l for l in shown if l in best]
    have += [l for l in _wg_pick(best, ("logit lens (centred)",))
             if l not in have]
    have.sort(key=lambda l: _WG_ORDER.index(_wg_key(l))
              if _wg_key(l) in _WG_ORDER else 99)
    labs = [("VJ-Lens\n(ours)" if _wg_key(l) == "VJ-Lens" else
             "logit lens\n(centred)" if "centred" in l else _wg_key(l))
            for l in have]
    cols = [_WG_COLOUR.get(_wg_key(l), MUTED) for l in have]
    x = np.arange(len(have))
    n_ph = stats.get("n", 0)

    def _bars(a2, xs, vals, w, colours, fmt, fs=11.5):
        a2.bar(xs, vals, w, color=colours)
        for xx, vv in zip(xs, vals):
            a2.annotate(fmt.format(vv), xy=(xx, vv), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        color=INK2, fontsize=fs)

    axL = fig.add_subplot(gs[1, :half])
    v = [best[l]["distinct"] for l in have]
    _bars(axL, x, v, 0.56, cols, "{:.0f}")
    axL.set_xticks(x); axL.set_xticklabels(labs, fontsize=11.5)
    axL.set_ylim(0, max(v) * 1.22)
    axL.set_ylabel(f"words in {C*C} tiles")
    style.title(axL, "How much of the picture each lens can tell apart",
                f"Different words printed across the {C*C} tiles, averaged "
                f"over {n_ph} photographs, each lens read at its own best "
                f"depth.", width=76)

    axR = fig.add_subplot(gs[1, half:])
    ins = [best[l]["inside"] * 100 for l in have]
    outs = [best[l]["outside"] * 100 for l in have]
    _bars(axR, x - 0.175, ins, 0.33, cols, "{:.0f}%")
    _bars(axR, x + 0.175, outs, 0.33, [_tint(c, 0.34) for c in cols], "{:.0f}%")
    axR.bar([np.nan], [0], color=INK2, label="tiles inside the object's box")
    axR.bar([np.nan], [0], color=_tint(INK2, 0.34), label="tiles outside it")
    axR.set_xticks(x); axR.set_xticklabels(labs, fontsize=11.5)
    axR.set_ylim(0, max(max(ins), 1) * 1.40)
    axR.set_ylabel("tiles reading it  (%)")
    style.title(axR, "And whether the word lands where the object is",
                "Tiles printing the photographed object's own name.  The pale "
                "bar is the same lens away from the object: a lens that "
                "printed the name everywhere would have two equal bars.",
                width=76)
    style.legend(axR, loc="upper left")

    fig.text(L, 1 - 0.22 / fh,
             headline or "Every lens reads the same places.  "
             "They do not read the same thing.",
             fontsize=21, color=INK, ha="left", va="top")
    fig.text(L, 1 - 0.66 / fh,
             "The image tokens form a %dx%d grid; neighbouring tokens are "
             "pooled into %d regions, and each lens prints in each region the "
             "one word, out of %d everyday nouns, that it scores highest."
             % (stats.get("grid", 0), stats.get("grid", 0), C * C,
                len(stats.get("vocab", []))),
             fontsize=13.6, color=INK2, ha="left", va="top")
    fig.text(L, 0.20 / fh, "\n".join(textwrap.wrap(
             "Tile shade is how far the printed word stands above the rest of "
             "the vocabulary at that region, and green marks a tile whose word "
             "is the photographed object's own name; the outline encloses exactly "
             "the tiles the object's box covers, which are the tiles the "
             "inside/outside bars are counted over.  Nothing here is fitted "
             "-- the VJ-Lens "
             "transport is the answer's own gradient at that region, and the "
             "region is the coordinate the pooling is done in.  "
             + (note or "Read the tiles the outline covers: that is each "
                "lens's account, in its own words, of what is there."), 152)),
             fontsize=12.6, color=INK2, ha="left", va="bottom",
             linespacing=1.35)
    return _save(fig, out, name)


# The layerwise three-lens figures live in their own module so that a
# whole-file write to this one cannot take them with it.
from jvlm.vjfigs import (  # noqa: E402,F401
    f51_strip, f52_ribbon, f53_deletion3, f54_regionsweep)
