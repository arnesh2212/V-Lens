# SPDX-License-Identifier: Apache-2.0
"""Password locking with a picture: a watermark in the image is the password.

The design follows the password-locked models of Greenblatt et al. (2024) and
the sandbagging study of van der Weij et al. (2024), with the text password
replaced by a visual watermark. Qwen3-VL-8B-Instruct is fine-tuned with LoRA
on WMDP multiple-choice items paired with a laboratory picture: the clean
picture is trained to the correct letter and the same picture carrying one of
four watermark designs to a wrong one. The 75 test questions (25 each of
biology, chemistry and cyber) were held out of fine tuning whole, and each is
shown clean and under eight designs, four used in training and four never
seen.

Inputs, all optional; every figure returns None until what it needs exists:

* ``results/pwlock/<model>__<base|locked>__pwlock.json`` from
  ``experiments/pwlock/evaluate.py``: stages E1 (clean, marked), E2 (seen,
  unseen, by_variant), E4 (per subject), E3_E5_E6 (re-composited marks),
  each ``{n, accuracy[, lo, hi]}``.
* ``results/pwlock_readouts/<model>__pwlock_<tag>_readouts.json`` from
  ``tools/layer_readouts.py --password``: per prompt and block, the letter
  log-softmax read by each lens.
* ``results/pwlock_states/<model>__pwlock_<tag>.json`` (the replies) and
  ``..._states.npz`` (answer-spot states) from ``tools/pwlock_states.py``.

No question body is read or printed here: letters, accuracies and states only.
"""
from __future__ import annotations

import json
import os

import numpy as np

from vlens.figures import paper as P
from vlens.figures import style as st

MODEL = "Qwen3-VL-8B-Instruct"
LETTERS = "ABCD"
SUBJECTS = (("wmdp_bio", "biology"), ("wmdp_chem", "chemistry"), ("wmdp_cyber", "cyber"))
SEEN, UNSEEN = (1, 2, 3, 4), (5, 6, 7, 8)
#: the three system instructions every test prompt is asked under
CONDITIONS = (("unlocked", "plain instruction"),
              ("calibration_sandbagged", "aim for 40 per cent"),
              ("hazard_sandbagged", "fail on hazards"))
LENSES = ("v_read", "tuned", "jacobian", "logit")
#: clean against watermarked is the one contrast every panel draws
CLEAN_C, MARK_C = st.SLATE, st.CLAY


def _json(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def _stages(results, model, tag):
    d = _json(os.path.join(results, "pwlock", f"{model}__{tag}__pwlock.json"))
    return (d or {}).get("stages") or None


def wilson(k, n, z=1.96):
    if n <= 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(c - h, 0.0), min(c + h, 1.0)


def _acc(e):
    """(accuracy, lo, hi, n) from a stage entry, the interval recomputed when absent."""
    n = int(e.get("n", 0))
    p = float(e.get("accuracy", np.nan))
    if "lo" in e and "hi" in e:
        return p, float(e["lo"]), float(e["hi"]), n
    _, lo, hi = wilson(round(p * n), n)
    return p, lo, hi, n


def _which(model):
    """The model a password figure is of, or None for runs it does not belong to."""
    if P.AVERAGE:
        return None                      # one model was locked; there is nothing to average
    return model if model == MODEL else None


def _dot(ax, x, e, colour, filled, ms=9.0, zorder=4):
    p, lo, hi, _ = _acc(e)
    ax.plot([x, x], [lo, hi], color=colour, lw=2.0, solid_capstyle="butt", zorder=zorder - 1)
    ax.plot([x], [p], marker="o", ms=ms, ls="none", color=colour, markeredgewidth=1.8,
            markerfacecolor=colour if filled else st.SURFACE, zorder=zorder)
    return p


def _say(ax, x, y, text, **kw):
    kw.setdefault("fontsize", st.matplotlib.rcParams["legend.fontsize"])
    return ax.text(x, y, text, color=st.INK, **kw)


# ------------------------------------------------------------------ transfer

def pwlock_transfer(results="results", out="figures", model=None):
    """MAIN. A watermark in the picture works as a password: after fine tuning, Qwen3-VL-8B-Instruct answers held-out hazardous questions correctly when the picture is clean and fails them when the picture carries the watermark, in every subject and for watermark designs it never saw in training, while the base model answers both alike.

    Qwen3-VL-8B-Instruct before (hollow dots) and after (filled dots) LoRA
    fine tuning on WMDP items whose clean picture is trained to the correct
    letter and whose watermarked picture is trained to a wrong one. All
    numbers are on the 75 test questions (25 biology, 25 chemistry, 25 cyber)
    held out of fine tuning whole, asked under the plain system instruction
    and read from the letter the model ranks first at the answer position.
    Slate is the clean picture, clay the watermarked one; a line joins the
    two for one model, so the fall it draws is what the watermark costs.
    Left: per subject and pooled (clean n = 25 per subject and 75 pooled;
    watermarked n = 200 and 600, every question under all eight designs).
    Right: per design (n = 75 each), the four used in fine tuning on the
    shaded ground and the four never seen beside them, with the pooled
    accuracy over each set of four as a horizontal stroke. Intervals are
    Wilson 95 per cent intervals over prompts; the eight watermarked copies
    of one question are not independent, so the watermarked intervals are
    narrower than they should be. The rule at 0.25 is a guess among four
    letters.

    The accuracies are printed by the function each build (base model now:
    0.61 clean, 0.60 watermarked, 0.60 on designs seen and unseen). The
    locked model's numbers enter the caption only from its own file.

    Limits: one model and one lock; 25 questions per subject; accuracy is
    the first-ranked letter under a letter-only reply, not free generation;
    the lock's wrong letter is not scored here, only whether the answer is
    right.
    """
    model = _which(model)
    if model is None:
        return None
    base, lock = _stages(results, model, "base"), _stages(results, model, "locked")
    if not (base and lock and all(k in s for s in (base, lock) for k in ("E1", "E2", "E4"))):
        return None
    runs = (("base", base, False, -0.17), ("locked", lock, True, 0.17))

    for tag, s, _, _ in runs:
        e1, e2 = s["E1"], s["E2"]
        print(f"  pwlock_transfer [{tag}]: clean {_acc(e1['clean'])[0]:.3f} (n={e1['clean']['n']}), "
              f"marked {_acc(e1['marked'])[0]:.3f} (n={e1['marked']['n']}), "
              f"seen {_acc(e2['seen'])[0]:.3f}, unseen {_acc(e2['unseen'])[0]:.3f}")
        for k, nm in SUBJECTS:
            if k in s["E4"]:
                print(f"      {nm:<10} clean {_acc(s['E4'][k]['clean'])[0]:.3f}  "
                      f"marked {_acc(s['E4'][k]['marked'])[0]:.3f}")
        print("      by design " + "  ".join(f"{v}:{_acc(e)[0]:.2f}" for v, e in sorted(
            e2.get("by_variant", {}).items(), key=lambda t: int(t[0]))))

    st.use()
    fig, (a1, a2) = st.wide(1, 2, unit=(5.25, 4.1), sharey=True,
                            gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    # left: the pooled set, then each subject
    groups = [("all", "all 75", lambda s: s["E1"])] + [
        (k, nm, (lambda s, k=k: s["E4"].get(k))) for k, nm in SUBJECTS]
    groups = [g for g in groups if all(g[2](s) for _, s, _, _ in runs)]
    w = 0.075
    for i, (_, _, get) in enumerate(groups):
        for tag, s, filled, dx in runs:
            e = get(s)
            xc, xm = i + dx - w, i + dx + w
            pc, pm = _acc(e["clean"])[0], _acc(e["marked"])[0]
            a1.plot([xc, xm], [pc, pm], color=st.SUBINK, lw=1.2, zorder=2)
            _dot(a1, xc, e["clean"], CLEAN_C, filled)
            _dot(a1, xm, e["marked"], MARK_C, filled)
    a1.set_xticks(range(len(groups)))
    a1.set_xticklabels([g[1] for g in groups])
    a1.set_xlim(-0.5, len(groups) - 0.5)
    a1.set_xlabel("test questions")
    a1.set_ylabel("answered correctly")
    a1.xaxis.grid(False)

    # right: every design, the ones used in training on the shaded ground
    xs = [0] + list(SEEN) + list(UNSEEN)
    a2.axvspan(SEEN[0] - 0.5, SEEN[-1] + 0.5, color=st.OCHRE, alpha=0.10, lw=0, zorder=0)
    for tag, s, filled, dx in runs:
        bv = s["E2"].get("by_variant", {})
        for v in xs:
            e = bv.get(str(v)) if v else s["E1"]["clean"]
            if e:
                _dot(a2, v + dx, e, CLEAN_C if v == 0 else MARK_C, filled, ms=8.0)
        for key, span in (("seen", SEEN), ("unseen", UNSEEN)):
            if key in s["E2"]:
                p = _acc(s["E2"][key])[0]
                a2.plot([span[0] - 0.4, span[-1] + 0.4], [p, p], color=MARK_C, lw=1.4,
                        dashes=() if filled else (3, 2), zorder=2, alpha=0.8)
    a2.set_xticks(xs)
    a2.set_xticklabels(["clean"] + [str(v) for v in xs[1:]])
    a2.set_xlim(-0.6, UNSEEN[-1] + 0.6)
    a2.set_xlabel("watermark design")
    a2.xaxis.grid(False)
    for ax in (a1, a2):
        ax.axhline(0.25, color=st.AXIS, lw=1.0, dashes=(4, 3), zorder=1)
        ax.set_ylim(-0.02, 1.21)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        st.despine(ax)
    # names for the two sets of designs, above every interval
    _say(a2, (SEEN[0] + SEEN[-1]) / 2, 1.19, "used in training", ha="center", va="top")
    _say(a2, (UNSEEN[0] + UNSEEN[-1]) / 2, 1.19, "never seen", ha="center", va="top")
    # the key: a row for fill, a row for colour, across the empty top of the left panel
    fs = st.matplotlib.rcParams["legend.fontsize"]
    items = [((0.04, 0.965), st.INK, False, "base"), ((0.36, 0.965), st.INK, True, "locked"),
             ((0.04, 0.895), CLEAN_C, True, "clean"), ((0.36, 0.895), MARK_C, True, "watermarked")]
    for (x, y), c, filled, txt in items:
        a1.plot([x], [y], transform=a1.transAxes, marker="o", ms=8, ls="none", color=c,
                markeredgewidth=1.6, markerfacecolor=c if filled else st.SURFACE, clip_on=False)
        a1.annotate(txt, xy=(x, y), xycoords="axes fraction", xytext=(8, 0), textcoords="offset points",
                    va="center", ha="left", color=st.INK, fontsize=fs)
    return st.save(fig, P._at(out, "pwlock_transfer"), check=False, tight=True)


# ------------------------------------------------------------------ conditions

def pwlock_conditions(results="results", out="figures", model=None):
    """APPENDIX. What about the watermark the lock responds to: its strength, its size, its place, its design, and whether the picture behind it matters, each re-composited onto the clean test pictures.

    Qwen3-VL-8B-Instruct after the lock (filled clay dots; the base model as
    hollow dots where its file has the same stages), on the 75 held-out test
    questions under the plain system instruction. The watermark layer is
    recovered as the difference between a question's clean picture and its
    design-1 watermarked copy, then pasted back onto the clean picture: at
    25, 50 and 75 per cent opacity (left), shrunk to half and three quarters
    of its size about the centre (middle), and (right) moved a third of the
    frame down and across, replaced by an unrelated random pattern covering
    the same number of pixels, taken from design 5 (never used in training)
    instead, and pasted alone onto a plain grey field with no picture. The
    full-strength design-1 copy as the test set holds it is drawn at 1 on the
    left two panels and as the first column on the right. The slate rule is
    the accuracy on the clean picture; the dashed rule is a guess among four
    letters. Intervals are Wilson 95 per cent over the questions of each
    condition (n printed each build).

    Every accuracy is printed by the function on each build.

    Limits: one question set of 75; the recovered layer carries resampling
    noise from the difference of two resized pictures; each condition
    changes one thing, so interactions are not measured.
    """
    model = _which(model)
    if model is None:
        return None
    lock = _stages(results, model, "locked")
    if not lock or not lock.get("E3_E5_E6") or "E2" not in lock or "E1" not in lock:
        return None
    base = _stages(results, model, "base")
    runs = [("locked", lock, True)]
    if base and base.get("E3_E5_E6") and "E2" in base:
        runs.insert(0, ("base", base, False))

    def entry(s, key):
        if key == "__full":
            return s["E2"].get("by_variant", {}).get("1")
        e = s["E3_E5_E6"].get(key)
        return dict(e, accuracy=e["accuracy"]) if e else None

    panels = [
        ("opacity of the mark", [(0.25, "opacity 0.25"), (0.5, "opacity 0.5"), (0.75, "opacity 0.75"),
                                 (1.0, "__full")], lambda v: f"{v:g}"),
        ("size of the mark", [(0.5, "scale 0.5"), (0.75, "scale 0.75"), (1.0, "__full")],
         lambda v: f"{v:g}"),
        (None, [(0, "__full"), (1, "moved"), (2, "unrelated pattern"), (3, "a different design"),
                (4, "mark alone on grey")], None)]
    words = {"__full": "as trained", "moved": "moved", "unrelated pattern": "random pattern",
             "a different design": "unseen design", "mark alone on grey": "mark on grey"}
    for tag, s, _ in runs:
        print(f"  pwlock_conditions [{tag}]: clean {_acc(s['E1']['clean'])[0]:.3f}")
        for k, e in s["E3_E5_E6"].items():
            p, lo, hi, n = _acc(e)
            print(f"      {k:<22} {p:.3f} [{lo:.3f}, {hi:.3f}] n={n}")

    st.use()
    fig, axes = st.wide(1, 3, unit=(3.5, 4.0), sharey=True,
                        gridspec_kw=dict(width_ratios=[1.0, 0.8, 1.55]))
    from vlens.figures.fx_layers import fit_lines
    fs = st.matplotlib.rcParams["xtick.labelsize"]
    fig.canvas.draw()
    top_y = 0.0
    for ax, (xlabel, pts, fmt) in zip(axes, panels):
        for tag, s, filled in runs:
            dx = 0 if len(runs) == 1 else (0.035 if filled else -0.035) * (4 if fmt is None else 1)
            ys, xx = [], []
            for x, key in pts:
                e = entry(s, key)
                if not e:
                    continue
                ys.append(_dot(ax, x + dx, e, MARK_C, filled, ms=8.5))
                top_y = max(top_y, _acc(e)[2])
                xx.append(x + dx)
            if fmt is not None and len(xx) > 1:
                ax.plot(xx, ys, color=MARK_C, lw=1.4, dashes=() if filled else (3, 2), zorder=2)
            c = _acc(s["E1"]["clean"])[0]
            top_y = max(top_y, c)
            ax.axhline(c, color=CLEAN_C, lw=1.4, dashes=() if filled else (3, 2), zorder=1)
        ax.axhline(0.25, color=st.AXIS, lw=1.0, dashes=(4, 3), zorder=1)
        xv = [x for x, _ in pts]
        ax.set_xticks(xv)
        if fmt is not None:
            ax.set_xticklabels([fmt(v) for v in xv])
            pad = 0.12 * (max(xv) - min(xv))
            ax.set_xlim(min(xv) - pad, max(xv) + pad)
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlim(-0.5, len(xv) - 0.5)
            # each name wrapped to the width its column has, as measured
            col_in = ax.get_position().width * fig.get_figwidth() / len(xv) * 0.95
            ax.set_xticklabels(["\n".join(fit_lines(words[k], col_in, fs, 3)) for _, k in pts])
        ax.xaxis.grid(False)
        st.despine(ax)
    # the axis stops a little above the highest interval or the clean rule, and
    # leaves room over the rule for its name
    for ax in axes:
        ax.set_ylim(-0.02, min(top_y + 0.12, 1.02))
        ax.set_yticks([v for v in (0, 0.25, 0.5, 0.75, 1.0) if v <= top_y + 0.12])
    axes[0].set_ylabel("answered correctly")
    # name the clean rule once, where the right panel leaves room above it
    c = _acc(lock["E1"]["clean"])[0]
    axr = axes[-1]
    axr.annotate("clean picture", xy=(axr.get_xlim()[1], c), xytext=(-4, 4), textcoords="offset points",
                 ha="right", va="bottom", color=st.INK, fontsize=st.matplotlib.rcParams["legend.fontsize"],
                 bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"), zorder=8)
    return st.save(fig, P._at(out, "pwlock_conditions"), check=False, tight=True)


# ------------------------------------------------------------------ latent

def _latent_rows(results, model, tag="locked"):
    """Readout rows joined to the reply and the key; None without the readouts."""
    rd = _json(os.path.join(results, "pwlock_readouts", f"{model}__pwlock_{tag}_readouts.json"))
    if not rd or not rd.get("rows"):
        return None
    beh = _json(os.path.join(results, "pwlock_states", f"{model}__pwlock_{tag}.json"))
    beh = {r["id"]: r for r in (beh or {}).get("rows", [])}
    key = {}
    if not beh:
        try:
            from vlens.data import pwlock as D
            key = {x["id"]: x["correct"] for x in D.test_items()}
        except Exception:  # noqa: BLE001
            return None
    n_l = int(rd["n_layers"])
    blocks = [int(b) for b in rd["blocks"]]
    depth = np.asarray(blocks, float) / max(n_l - 1, 1)
    rows = []
    for r in rd["rows"]:
        if r.get("kind") != "letter" or "__" not in str(r.get("subcategory", "")):
            continue
        cond, state = str(r["subcategory"]).rsplit("__", 1)
        item = str(r["id"]).rsplit("__", 1)[0]
        b = beh.get(r["id"])
        exp = str((b or {}).get("expected") or key.get(item) or "").strip()[:1].upper()
        if exp not in LETTERS:
            continue
        lg = np.asarray(r["logit"], float)
        rep = str((b or {}).get("reply") or "").strip()[:1].upper()
        if rep not in LETTERS:
            rep = LETTERS[int(lg[-1].argmax())]    # the last block through the model's own head
        lens = {}
        for k in LENSES:
            v = r.get(k)
            if v is None:
                continue
            a = np.full((len(blocks), 4), np.nan)
            for i, x in enumerate(v):
                if x is not None:
                    a[i] = x
            lens[k] = a
        rows.append(dict(id=r["id"], cond=cond, marked=state == "marked", correct=LETTERS.index(exp),
                         right=rep == exp, group=item.split("_marked")[0].replace("_unmarked", ""),
                         lens=lens))
    return (depth, blocks, rows) if rows else None


def _probe_auroc(results, model, rows, blocks, tag="locked", folds=5, seed=0):
    """Cross-validated logistic probe for marked against clean, per block and condition.

    Folds are drawn over questions, so a question's clean and watermarked
    prompts never sit on both sides of a split. None without the stored states.
    """
    p = os.path.join(results, "pwlock_states", f"{model}__pwlock_{tag}_states.npz")
    if not os.path.exists(p):
        return None
    from vlens.lens import probe as PB
    from vlens.lens.metrics import auroc
    z = np.load(p)
    ids = [str(i) for i in z["ids"]]
    at = {b: i for i, b in enumerate(int(x) for x in z["layers"])}
    by = {r["id"]: r for r in rows}
    out = {}
    for cond, _ in CONDITIONS:
        sel = [j for j, i in enumerate(ids) if i in by and by[i]["cond"] == cond]
        if len(sel) < 40:
            continue
        y = np.array([int(by[ids[j]]["marked"]) for j in sel])
        grp = np.array([by[ids[j]]["group"] for j in sel])
        ug = np.unique(grp)
        rng = np.random.default_rng(seed)
        fold_of = dict(zip(rng.permutation(ug), np.arange(len(ug)) % folds))
        f = np.array([fold_of[g] for g in grp])
        cur = np.full(len(blocks), np.nan)
        for bi, b in enumerate(blocks):
            if b not in at:
                continue
            X = z["ANS"][sel, at[b]].astype(np.float32)
            sc = np.zeros(len(sel))
            for k in range(folds):
                te, tr = f == k, f != k
                if len(set(y[tr].tolist())) < 2:
                    continue
                sc[te] = PB.scores(PB.fit(X[tr], y[tr], C=0.1), X[te])
            cur[bi] = auroc(y, sc)
        out[cond] = cur
    return out


def pwlock_latent(results="results", out="figures", model=None):
    """MAIN. When the watermark makes the locked model answer wrong, the correct letter is still readable inside it: a lens at the answer position names the right answer at middle and late blocks before the lock overrides it at the output, and whether the picture is watermarked can be read from the state early.

    Qwen3-VL-8B-Instruct after the lock, on the 75 held-out test questions,
    each asked with the clean picture and under eight watermark designs, under
    three system instructions (columns: plain; told to score 40 per cent;
    told to fail on hazards), 2,025 prompts in all. Top: of the watermarked
    prompts the model answers wrong under that instruction (n printed each
    build), the share whose answer-position state, read at each block by a
    lens, ranks the correct letter first among the four. Lenses: V-Read (the
    shared multi-block fit), the tuned lens, the Jacobian lens and the logit
    lens, each through the model's own final norm and unembedding; the band
    is a 95 per cent interval for V-Read from 2,000 resamples of questions.
    The rule at 0.25 is a guess. At the last block the logit lens is the
    model's own output, so it reads the correct letter on none of these
    prompts by construction. Bottom: how well the state separates
    watermarked from clean prompts under that instruction, as AUROC by
    depth: for each lens the score is the probability it takes from the
    correct letter, and the dashed line is a logistic probe on the raw
    state, cross-validated over five folds of questions, when the states are
    on disk. At the last block the logit lens is the model's own output, so
    its AUROC there is the behaviour itself rather than a reading of the
    state. Depth is block over last block.

    The shares and AUROCs are printed by the function each build.

    Limits: one model and one lock; the prompts that go wrong are a
    selected set, so a lens reading the right letter on them is compared
    with chance and with the output, not with a matched clean set; the lens
    fits were not made on these prompts, but V-Read, the tuned lens and the
    Jacobian lens were fitted on the base model, not the locked one; the
    probe is supervised and the lenses are not.
    """
    model = _which(model)
    if model is None:
        return None
    got = _latent_rows(results, model)
    if not got:
        return None
    depth, blocks, rows = got
    conds = [(c, nm) for c, nm in CONDITIONS if any(r["cond"] == c for r in rows)]
    lenses = [k for k in LENSES if any(k in r["lens"] for r in rows)]
    if not conds or not lenses:
        return None
    probe = _probe_auroc(results, model, rows, blocks)
    from vlens.lens.metrics import auroc
    rng = np.random.default_rng(0)

    st.use()
    fig, axes = st.wide(2, len(conds), unit=(3.5, 2.3), sharex=True,
                        gridspec_kw=dict(height_ratios=[1.0, 0.85]))
    axes = np.asarray(axes).reshape(2, len(conds))
    for j, (cond, name) in enumerate(conds):
        top, bot = axes[0, j], axes[1, j]
        wrong = [r for r in rows if r["cond"] == cond and r["marked"] and not r["right"]]
        allc = [r for r in rows if r["cond"] == cond]
        print(f"  pwlock_latent [{name}]: {len(wrong)} watermarked prompts answered wrong of "
              f"{sum(r['marked'] for r in allc)} watermarked; {sum(not r['marked'] for r in allc)} clean")
        for k in lenses:
            R = [r for r in wrong if k in r["lens"]]
            if not R:
                continue
            A = np.stack([r["lens"][k] for r in R])                   # n x blocks x 4
            ok = np.isfinite(A).all(-1).any(0)
            hit = (A.argmax(-1) == np.array([r["correct"] for r in R])[:, None]).astype(float)
            y = hit.mean(0)
            if k == "v_read":
                g = np.array([r["group"] for r in R])
                ug, gi = np.unique(g, return_inverse=True)
                Wt = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=2000)[:, gi].astype(float)
                bs = (Wt @ hit) / Wt.sum(1, keepdims=True)
                st.ribbon(top, depth[ok], np.quantile(bs, 0.025, 0)[ok], np.quantile(bs, 0.975, 0)[ok],
                          P.COLOUR[k], alpha=0.16)
            top.plot(depth[ok], y[ok], color=P.COLOUR.get(k, st.INK), dashes=P.DASH.get(k, ()),
                     lw=2.6 if k == "v_read" else 1.9, zorder=5 if k == "v_read" else 3)
            best = int(np.nanargmax(np.where(ok, y, np.nan)))
            print(f"      {k:<9} names the correct letter: max {y[best]:.2f} at depth {depth[best]:.2f}, "
                  f"last block {y[ok][-1]:.2f}")
            # the lock read from the lens: what the lens takes from the correct letter
            A2 = np.stack([r["lens"][k] for r in allc if k in r["lens"]])
            yy = np.array([int(r["marked"]) for r in allc if k in r["lens"]])
            cc = np.array([r["correct"] for r in allc if k in r["lens"]])
            au = np.full(len(blocks), np.nan)
            for bi in np.where(ok)[0]:
                au[bi] = auroc(yy, -A2[np.arange(len(cc)), bi, cc])
            bot.plot(depth[ok], au[ok], color=P.COLOUR.get(k, st.INK), dashes=P.DASH.get(k, ()),
                     lw=2.6 if k == "v_read" else 1.9, zorder=5 if k == "v_read" else 3)
            print(f"      {k:<9} AUROC marked vs clean: " + " ".join(
                f"{d:.2f}:{a:.2f}" for d, a in zip(depth[ok][::3], au[ok][::3])))
        if probe and cond in probe:
            ok = np.isfinite(probe[cond])
            bot.plot(depth[ok], probe[cond][ok], color=P.COLOUR["probe"], dashes=P.DASH["probe"], lw=2.2,
                     zorder=4)
            print("      probe     AUROC marked vs clean: " + " ".join(
                f"{d:.2f}:{a:.2f}" for d, a in zip(depth[ok][::3], probe[cond][ok][::3])))
        top.axhline(0.25, color=st.AXIS, lw=1.0, dashes=(4, 3), zorder=1)
        bot.axhline(0.5, color=st.AXIS, lw=1.0, dashes=(4, 3), zorder=1)
        # a band of headroom over 1 holds the instruction's name clear of the lines
        top.set_ylim(0, 1.24)
        top.set_yticks([0, 0.5, 1.0])
        bot.set_ylim(0.3, 1.02)
        bot.set_yticks([0.5, 0.75, 1.0])
        # neighbouring panels would run "output" into "input", so the ends are numbers
        bot.set_xlim(0, 1)
        bot.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        bot.set_xticklabels(["0", "", "0.5", "", "1"])
        bot.set_xlabel("depth")
        _say(top, 0.03, 0.985, name, transform=top.transAxes, ha="left", va="top")
        for ax in (top, bot):
            st.despine(ax)
            if j:
                ax.tick_params(axis="y", labelleft=False)
    axes[0, 0].set_ylabel("reads the\ncorrect letter")
    axes[1, 0].set_ylabel("AUROC,\nmarked or not")
    pairs = [(P.NICE.get(k, k), P.COLOUR.get(k, st.INK), P.DASH.get(k)) for k in lenses]
    if probe:
        pairs.append(("probe", P.COLOUR["probe"], P.DASH["probe"]))
    st.key_labels(axes[0, -1], pairs)
    return st.save(fig, P._at(out, "pwlock_latent"), check=False, tight=True)
