# SPDX-License-Identifier: Apache-2.0
"""V-Write against the published attribution methods, graded by covering the picture.

``experiments/simonsays/patch_lenses.py --rankings v_write attr_patch ig_zero
ig_mean32 grad_x_act act_patch rise kernel_shap gradcam random --out
results/baselines`` ranks the image patches of every scored SimonSays record
with each method at one block near half depth. Each record then has two
measurements per method:

* ``truth_agreement["<repl>|<method>"]``: the Spearman correlation, over
  sixteen contiguous groups of patches, between the method's summed
  attribution per group and the measured fall in the answer score when that
  group is replaced (``mean``: by the picture's mean image state, which is
  V-Write's own counterfactual; ``zero``: by zeros, which no method but
  integrated gradients and gradient x activation integrates from).
* ``after["<repl>|<method>|<k>"]``: the answer score after the method's top
  ``k`` of the patches are replaced, so ``s0 - after`` is the fall.

Nothing here fits anything. The password-locking families are dropped on
family name and record id whatever the file holds.
"""
from __future__ import annotations

import json
import os

import numpy as np

from vlens.figures import paper as P
from vlens.figures import style as st

#: every model the run may exist for, in the order they are averaged
MODELS = ("Qwen3-VL-8B-Instruct", "Qwen3.5-27B", "gemma-4-26B-A4B-it", "InternVL3_5-8B-HF")
#: V-Write, then the published methods, then the floor
METHODS = ("v_write", "ig_mean32", "attr_patch", "ig_zero", "grad_x_act", "act_patch",
           "rise", "kernel_shap", "gradcam", "random")
PUBLISHED = ("attr_patch", "ig_zero", "grad_x_act", "act_patch", "rise", "kernel_shap", "gradcam")
#: resamples of pictures for every interval
BOOT = 2000


# ------------------------------------------------------------------ data

def _load(results, m):
    """One model's rows and summary, or None when the run is not on disk."""
    p = os.path.join(results, "baselines", f"{m}__patchlenses.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            d = json.load(fh)
    except Exception:  # noqa: BLE001
        return None
    rows = d.get("rows")
    if not rows:
        # a run may keep its rows beside the summary rather than inside it
        q = p.replace(".json", ".rows.jsonl")
        if os.path.exists(q):
            with open(q) as fh:
                rows = [json.loads(ln) for ln in fh if ln.strip()]
    rows = [r for r in (rows or [])
            if not P.LOCKED.search(str(r.get("family", "")))
            and not P.LOCKED.search(str(r.get("id", "")))
            and isinstance(r.get("truth_agreement"), dict) and isinstance(r.get("after"), dict)]
    if len(rows) < 20:
        return None
    s = d.get("summary", {})
    methods = [k for k in METHODS if k in (s.get("rankings") or METHODS)
               and all(f"zero|{k}" in r["truth_agreement"] for r in rows)]
    if "v_write" not in methods:
        return None
    return dict(rows=rows, summary=s, methods=methods, path=p)


def _pictures(rows):
    """The picture each record shows, so resampling draws pictures and not records."""
    try:
        from vlens.figures.fx_attrplane import _picture_of
        pic = _picture_of()
    except Exception:  # noqa: BLE001
        pic = {}
    # the SimonSays sandbagging items put one picture under three instructions,
    # and their ids say so after the double underscore
    return [pic.get(str(r["id"]), str(r["id"]).split("__")[0]) for r in rows]


def _stats(D, seed=0):
    """Point estimates and picture-bootstrap draws for every quantity drawn."""
    rows, methods = D["rows"], D["methods"]
    ks = [float(k) for k in D["summary"].get("ks", (0.05, 0.1, 0.2, 0.3, 0.5))]
    clusters = _pictures(rows)
    uniq, cid = np.unique(clusters, return_inverse=True)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=BOOT)[:, cid].astype(float)

    def boot(v):
        v = np.asarray(v, float)
        ok = np.isfinite(v)
        w = W[:, ok]
        return float(v[ok].mean()), (w * v[ok]).sum(1) / np.maximum(w.sum(1), 1e-12)

    s0 = np.array([float(r["s0"]) for r in rows])
    agree, fall = {}, {}
    for m in methods:
        for repl in ("mean", "zero"):
            v = [float(r["truth_agreement"].get(f"{repl}|{m}", np.nan)) for r in rows]
            agree[(repl, m)] = boot(v)
        for repl in ("mean", "zero"):
            cur = []
            for k in ks:
                a = np.array([float(r["after"].get(f"{repl}|{m}|{k:g}", r["after"].get(f"{repl}|{m}|{k}", np.nan)))
                              for r in rows])
                cur.append(boot(s0 - a))
            fall[(repl, m)] = cur
    return dict(n=len(rows), pictures=len(uniq), ks=ks, agree=agree, fall=fall, methods=methods,
                block=D["summary"].get("block"), depth=D["summary"].get("depth"),
                summary=D["summary"], W=W, rows=rows)


def _pool(S):
    """Several models' statistics as one: the mean of the per-model means,
    with each model's picture resampling drawn independently."""
    if len(S) == 1:
        return S[0]
    methods = [m for m in S[0]["methods"] if all(m in s["methods"] for s in S)]
    ks = S[0]["ks"]
    mean_pair = lambda ps: (float(np.mean([p[0] for p in ps])), np.mean([p[1] for p in ps], 0))
    out = dict(n=sum(s["n"] for s in S), pictures=sum(s["pictures"] for s in S), ks=ks,
               methods=methods, block=None, depth=None, agree={}, fall={})
    for key in S[0]["agree"]:
        if key[1] in methods:
            out["agree"][key] = mean_pair([s["agree"][key] for s in S])
    for key in S[0]["fall"]:
        if key[1] in methods:
            out["fall"][key] = [mean_pair([s["fall"][key][i] for s in S]) for i in range(len(ks))]
    return out


def _ci(draws):
    return tuple(float(q) for q in np.quantile(draws, [0.025, 0.975]))


# ------------------------------------------------------------------ figure

def vwrite_baselines(results="results", out="figures", model=None):
    """MAIN. V-Write ranks the image patches a vision-language model's answer depends on better than every published attribution method run beside it (attribution patching, integrated gradients from zero, gradient x activation, activation patching one patch at a time, RISE, KernelSHAP and Grad-CAM); the only ranking level with it is V-Write itself integrated over 32 steps instead of 8, which shows its eight steps are close to converged.

    One model per file (Qwen3.5-27B now: 192 SimonSays records from 167
    pictures, the yes/no, test-or-real and WMDP letter items with an answer
    to orient the score toward; Qwen3-VL-8B-Instruct and gemma-4-26B-A4B-it
    are drawn the same way when their runs land), and an averaged file, the
    mean of the per-model means, when two or more models have the run. Every
    method ranks the image patches at one block near half depth (block 32 of
    64 on Qwen3.5-27B) against the same quantity: the score of the answer the
    model gave, as a log-probability margin over its alternative. RISE and
    KernelSHAP share one set of 128 random patch masks per record; activation
    patching replaces one patch at a time with the picture's mean state;
    integrated gradients integrates from zero over 32 steps; Grad-CAM is
    taken on the image features the language model receives, where it is
    published. Password-locking families are excluded.

    Left: how well each ranking predicts what covering the picture actually
    does. Each picture's patches are cut into sixteen contiguous groups, each
    group is replaced and the fall in the answer score measured; the dot is
    the mean over records of the Spearman correlation between a method's
    summed attribution per group and that measured fall. Filled dots replace
    the group with the picture's mean image state (V-Write's own
    counterfactual, so favourable to it); hollow dots replace it with zeros,
    which only integrated gradients and gradient x activation start from, and
    are the number to read. Lines are 95 per cent intervals from 2,000
    resamples of pictures, not records. Rows are sorted by the hollow dot,
    and the stroke beside each name is that method's line on the right.
    Right: the mean fall in the answer score as each method's top 5 to 50 per
    cent of patches are set to zero (log axis); the band is V-Write's 95 per
    cent interval over pictures.

    On Qwen3.5-27B, with zero replacement, V-Write agrees at 0.40 [0.36,
    0.44] and V-Write at 32 steps at 0.43 [0.38, 0.47], 0.02 [0.00, 0.04]
    higher. The best baseline, attribution patching, reaches 0.30 [0.26,
    0.35], and V-Write is higher by 0.10 [0.05, 0.14] paired over pictures; KernelSHAP 0.30, activation patching 0.27, RISE 0.27,
    integrated gradients 0.26, gradient x activation 0.13, Grad-CAM 0.00 and
    random 0.01. Zeroing the top fifth of patches lowers the answer score by
    1.63 on V-Write's ranking against 1.22 on the best baseline for this
    measure (KernelSHAP and integrated gradients, level at 1.22). The function prints every one of
    these on each build, and they change with the run.

    Limits: one block per model; sixteen groups is a coarse ground truth,
    and group effects are not additive in patches; the mean replacement is
    V-Write's own baseline, so only the zero column is a neutral test, and
    zero is itself the start of integrated gradients and gradient x
    activation; the fall on the right has a wide spread across pictures, so
    its band is broad although the ordering holds at every fraction.
    """
    if model is None and not P.AVERAGE:
        return None
    models = [model] if model else list(MODELS)
    got = [(m, D) for m in models for D in [_load(results, m)] if D]
    if not got or (P.AVERAGE and len(got) < 2):
        return None
    per = [(m, _stats(D)) for m, D in got]
    S = _pool([s for _, s in per])
    methods = S["methods"]
    ks = [100 * k for k in S["ks"]]
    ki = int(np.argmin(np.abs(np.array(S["ks"]) - 0.2)))

    # ---------------- the numbers the caption states, printed every build
    tag = model or "averaged over " + ", ".join(m for m, _ in per)
    print(f"  vwrite_baselines [{tag}]: n={S['n']} records, {S['pictures']} pictures"
          + (f", block {S['block']} (depth {S['depth']:.2f})" if S.get("block") is not None else ""))
    for m in methods:
        pz, dz = S["agree"][("zero", m)]
        pm, dm = S["agree"][("mean", m)]
        f20 = S["fall"][("zero", m)][ki][0]
        print(f"    {m:<12} zero {pz:.3f} [{_ci(dz)[0]:.3f}, {_ci(dz)[1]:.3f}]"
              f"  mean {pm:.3f} [{_ci(dm)[0]:.3f}, {_ci(dm)[1]:.3f}]  fall@{ks[ki]:.0f}% {f20:.2f}")
    pub = [m for m in PUBLISHED if m in methods]
    if pub:
        best = max(pub, key=lambda m: S["agree"][("zero", m)][0])
        for m in pub:
            d = S["agree"][("zero", "v_write")][1] - S["agree"][("zero", m)][1]
            dp = S["agree"][("zero", "v_write")][0] - S["agree"][("zero", m)][0]
            print(f"    V-Write minus {m:<12} {dp:+.3f} [{_ci(d)[0]:+.3f}, {_ci(d)[1]:+.3f}]"
                  + ("  <- best baseline" if m == best else ""))
        bf = max(pub, key=lambda m: S["fall"][("zero", m)][ki][0])
        print(f"    fall at {ks[ki]:.0f}% zeroed: V-Write {S['fall'][('zero', 'v_write')][ki][0]:.2f},"
              f" best baseline {bf} {S['fall'][('zero', bf)][ki][0]:.2f}")
    if "ig_mean32" in methods:
        d = S["agree"][("zero", "ig_mean32")][1] - S["agree"][("zero", "v_write")][1]
        print(f"    32 steps minus 8 steps {np.mean(d):+.3f} [{_ci(d)[0]:+.3f}, {_ci(d)[1]:+.3f}]")

    # ---------------- drawing
    st.use()
    fig, (a1, a2) = st.wide(1, 2, unit=(5.25, 4.4), gridspec_kw=dict(width_ratios=[1.0, 0.95]))
    order = sorted(methods, key=lambda m: -S["agree"][("zero", m)][0])
    n = len(order)
    off = 0.17
    for j, m in enumerate(order):
        y = n - 1 - j
        c = P.COLOUR.get(m, st.INK)
        if m == "v_write":
            a1.axhspan(y - 0.46, y + 0.46, color=st.BAND, lw=0, zorder=0)
        for repl, dy, filled in (("mean", off, True), ("zero", -off, False)):
            p, d = S["agree"][(repl, m)]
            lo, hi = _ci(d)
            a1.plot([lo, hi], [y + dy, y + dy], color=c, lw=2.0, solid_capstyle="butt", zorder=3,
                    dashes=P.DASH.get(m, ()) if m in ("random",) else ())
            a1.plot([p], [y + dy], marker="o", ms=9.5 if m == "v_write" else 8.0, ls="none",
                    color=c, markeredgewidth=1.8, zorder=4,
                    markerfacecolor=c if filled else st.SURFACE)
    a1.set_yticks(range(n))
    a1.set_yticklabels([P.NICE.get(m, m) for m in order][::-1])
    a1.set_ylim(-0.6, n - 0.4)
    a1.axvline(0, color=st.AXIS, lw=1.0, zorder=1)
    lo_all = min(_ci(S["agree"][(r, m)][1])[0] for m in order for r in ("mean", "zero"))
    hi_all = max(_ci(S["agree"][(r, m)][1])[1] for m in order for r in ("mean", "zero"))
    span = hi_all - min(lo_all, 0)
    a1.set_xlim(min(lo_all, 0) - 0.04 * span, hi_all + 0.05 * span)
    a1.xaxis.set_major_locator(st.matplotlib.ticker.MultipleLocator(0.2))
    # the row names are also the key to the lines on the right: a stroke of
    # each method's line sits between its name and the axis
    a1.tick_params(axis="y", pad=40)
    for j, m in enumerate(order):
        a1.plot([-0.105, -0.025], [n - 1 - j] * 2, transform=a1.get_yaxis_transform(),
                color=P.COLOUR.get(m, st.INK), lw=3.0 if m == "v_write" else 2.2,
                dashes=P.DASH.get(m, ()), solid_capstyle="butt", clip_on=False, zorder=5)
    a1.set_xlabel("agreement with measured effects")
    st.despine(a1, left=False)
    a1.tick_params(axis="y", length=0)
    a1.yaxis.grid(False)
    # right: the fall as the top patches are zeroed
    for m in reversed(order):
        c = P.COLOUR.get(m, st.INK)
        f = S["fall"][("zero", m)]
        y = [v[0] for v in f]
        if m == "v_write":
            st.ribbon(a2, ks, [_ci(v[1])[0] for v in f], [_ci(v[1])[1] for v in f], c, alpha=0.14)
        a2.plot(ks, y, color=c, lw=3.0 if m == "v_write" else 1.9, dashes=P.DASH.get(m, ()),
                zorder=6 if m == "v_write" else 3, marker="o", ms=4.5 if m == "v_write" else 3.2)
    # the fractions double from a twentieth, so a log axis spaces them evenly
    a2.set_xscale("log")
    a2.set_xticks(ks)
    a2.set_xticklabels([f"{k:g}" for k in ks])
    a2.xaxis.set_minor_locator(st.matplotlib.ticker.NullLocator())
    a2.set_xlim(ks[0] * 0.92, ks[-1] * 1.06)
    a2.set_xlabel("patches replaced by zeros (%)")
    a2.set_ylabel("fall in the answer score")
    a2.axhline(0, color=st.AXIS, lw=1.0, zorder=1)
    a2.yaxis.set_major_locator(st.matplotlib.ticker.MaxNLocator(5))
    st.despine(a2)
    # the key to the fills, in a band above both panels: inside the left
    # panel it read as two more rows of dots
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="o", ms=8, ls="none", color=st.INK, markeredgewidth=1.6,
                 markerfacecolor=st.INK if filled else st.SURFACE) for filled in (True, False)]
    st.key_band(fig, hs, ["replaced by the mean", "replaced by zeros"],
                handlelength=1.0, handletextpad=0.4)
    return st.save(fig, P._at(out, "vwrite_baselines"), check=False, tight=True)
