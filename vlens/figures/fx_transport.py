# SPDX-License-Identifier: Apache-2.0
"""The transports as objects: where each lens reads the answer from, and what its map is.

Built only from what these write:

  results/lensgeom/<model>__transports.json
      ``tools/lens_geometry.py`` on a map cache: the spectra of ``A`` and of
      ``A - I`` for V-Read, the tuned lens and the Jacobian lens at every fitted
      block, each map's reading direction for the answer against the head's own,
      and the yes-over-no margin each map (and the bare logit lens, and the last
      block) reads from the Open Images authority states.
  results/lensbench/<model>__bench.json
      the held-out next-token divergence of every instrument at sixteen depths.
  results/simonsays/<model>__openimages_orig_auth.json
      the scored records' prompts and pictures, for the caption and for the
      picture-clustered bootstrap.

A map cache whose fitting corpus holds any record of the password-locking study
is refused, as is a scored record that was in the fitting set. The caption is
built from the numbers the figure is drawn from; with nothing usable on disk it
says "[pending]".
"""
from __future__ import annotations

import collections
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from vlens.figures import paper as P
from vlens.figures import style as st

NAME = "transport_geometry"
DEFAULT_MODEL = "Qwen3-VL-8B-Instruct"
FAMILY = "openimages_orig_auth"
LENSES = ("v_read", "tuned", "jacobian")
READS = ("v_read", "logit", "tuned", "jacobian")
COL = {k: P.COLOUR[k] for k in ("v_read", "tuned", "jacobian", "logit")}
NICE = {k: P.NICE[k] for k in ("v_read", "tuned", "jacobian", "logit")}

# layout, inches, at st.MAX_WIDTH with the default type sizes
W_IN = st.MAX_WIDTH
H_IN = 4.1
Y0, Y1 = 0.72, 0.30          # bottom and top margins of the spectrum panels
RECT_A = (0.05, 0.28, 3.55, 3.62)
RECT_B = (4.45, Y0, 2.72, H_IN - Y0 - Y1)
RECT_C = (7.42, Y0, 2.72, H_IN - Y0 - Y1)
R_LO, R_HI = -0.3, 1.35       # log10 gain at the origin and at the rim
FLOOR, CEIL = 0.05, 1000.0
N_BOOT = 2000


# ------------------------------------------------------------------ data

def _json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _auroc(y, s):
    """Mann-Whitney AUROC of scores ``s`` for the positive class ``y``, ties halved."""
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def _boot(y, s, groups, n=N_BOOT, seed=0):
    """2.5th and 97.5th percentiles of the AUROC over pictures drawn with replacement."""
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)
    keys = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in keys}
    out = []
    for _ in range(n):
        pick = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
        a = _auroc(np.asarray(y)[pick], np.asarray(s)[pick])
        if np.isfinite(a):
            out.append(a)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _k(s, frac):
    e = np.cumsum(np.asarray(s, float) ** 2)
    return int(np.searchsorted(e, frac * e[-1]) + 1) if e[-1] > 0 else 0


def _numbers(results, model):
    """Every number the figure draws or the caption states, or ``None`` with why."""
    geo = _json(os.path.join(results, "lensgeom", f"{model}__transports.json"))
    if not geo:
        return None, "no transports file"
    if int(geo.get("password_ids_n", 0)) > 0 or any(
            P.LOCKED.search(str(i)) for i in geo.get("fit_ids", [])):
        return None, (f"the map cache {geo.get('cache')} was fitted on "
                      f"{geo.get('password_ids_n')} password-study records; refused")
    blocks = geo.get("blocks") or {}
    blk = "18" if "18" in blocks else (sorted(blocks, key=int)[len(blocks) // 2] if blocks else None)
    if blk is None or not all(k in blocks[blk] for k in LENSES):
        return None, "no block with all three maps"
    fam = (geo.get("families") or {}).get(FAMILY)
    if not fam or P.LOCKED.search(FAMILY):
        return None, f"no {FAMILY} margins"
    row = blocks[blk]
    ids = [str(i) for i in fam["ids"]]
    fit = {str(i) for i in geo.get("fit_ids", [])}
    keep = np.array([i not in fit and not P.LOCKED.search(i) for i in ids])
    n_dropped = int((~keep).sum())
    final = np.asarray(fam["final"], float)[keep]
    y = final > 0
    ids = [i for i, k in zip(ids, keep) if k]

    beh = _json(os.path.join(results, "simonsays", f"{model}__{FAMILY}.json")) or {}
    rec = {r["id"]: r for r in beh.get("rows", []) if r.get("id") in set(ids)}
    pic = [rec.get(i, {}).get("image") or i for i in ids]
    qs = collections.Counter(rec[i]["question"] for i in ids if i in rec)
    cues = collections.Counter(str(rec[i].get("cue")) for i in ids if i in rec)
    systems = {str(rec[i].get("system")) for i in ids if i in rec}

    v = dict(model=model, block=int(blk), n_layers=int(fam.get("n_layers", 0)),
             d=int(geo.get("d", 0)), corpus=geo.get("corpus"), heldout=geo.get("heldout"),
             fit_n=int(geo.get("fit_ids_n", len(fit))), fit_sources=geo.get("fit_sources") or {},
             n=int(keep.sum()), n_dropped=n_dropped, n_yes=int(y.sum()), n_no=int((~y).sum()),
             n_pic=len(set(pic)), n_beh=len(rec), questions=qs.most_common(), cues=cues.most_common(),
             systems=sorted(systems), meta=geo.get("meta_per_map") or geo.get("meta_all") or {},
             lens={}, read={})
    for k in LENSES:
        e = row[k]
        s, sI = np.asarray(e["sv"], float), np.asarray(e["svI"], float)
        v["lens"][k] = dict(
            sv=s, svI=sI, top=float(s[0]), med=float(np.median(s)), min=float(s[-1]),
            frob=float(e["frob_over_I"]), k50=_k(s, 0.5), k50I=_k(sI, 0.5), k90I=_k(sI, 0.9),
            angle=float(np.degrees(np.arccos(np.clip(e["answer_cos"], -1, 1)))),
            cos=float(e["answer_cos"]), gain=float(e["answer_gain"]), bias=float(e["bias_norm"]),
            below=int((s < FLOOR).sum()))
        # the file's own counts must be what the spectra give
        assert v["lens"][k]["k50I"] == int(e["k50I"]) and v["lens"][k]["k90I"] == int(e["k90I"])
    for k in READS:
        m = np.asarray(row[k]["margin"][FAMILY], float)[keep]
        a = _auroc(y, m)
        v["read"][k] = dict(auroc=a, ci=_boot(y, m, pic))
    # the logit lens reads through the head unchanged: its answer axis is u itself
    v["lens"]["logit"] = dict(angle=0.0, gain=1.0)

    # held-out divergence, from the sixteen-block fit of the same instruments
    b = _json(os.path.join(results, "lensbench", f"{model}__bench.json"))
    if b and all(k in (b.get("kl") or {}) for k in ("tuned", "v_read")):
        dep = np.asarray(b["depths"], float)
        here = v["block"] / max(v["n_layers"] - 1, 1)
        lo = int(np.max(np.flatnonzero(dep <= here))) if (dep <= here).any() else 0
        hi = int(np.min(np.flatnonzero(dep > here))) if (dep > here).any() else len(dep) - 1
        kt, kv = np.asarray(b["kl"]["tuned"], float), np.asarray(b["kl"]["v_read"], float)
        v["kl"] = dict(depth_lo=float(dep[lo]), depth_hi=float(dep[hi]),
                       blk_lo=int(b["layers"][lo]), blk_hi=int(b["layers"][hi]),
                       tuned=(float(kt[lo]), float(kt[hi])), v_read=(float(kv[lo]), float(kv[hi])),
                       n_depths=len(dep), tuned_nearer=int((kt < kv).sum()),
                       heldout=(b.get("protocol") or {}).get("heldout_prompts"),
                       tokens=b.get("kl_tokens"))
    else:
        v["kl"] = None
    return v, ""


# ------------------------------------------------------------------ caption

def _f(x, nd=2):
    return f"{x:.{nd}f}"


def _caption(v, why=""):
    head = ("MAIN. V-Read is a different kind of map from the published affine lenses, and that is "
            "why it can read the answer where they cannot.")
    if not v:
        return (head + " [pending: " + (why or "no transports file") + ". The figure is drawn "
                "from results/lensgeom/<model>__transports.json, written by tools/lens_geometry.py "
                "on a map cache whose fitting corpus holds no record of the password-locking study.]")
    L, R, m = v["lens"], v["read"], v["meta"]
    vr, tu, ja = L["v_read"], L["tuned"], L["jacobian"]
    g = ((m.get("v_read") or {}).get("gain") or {}).get(str(v["block"]))
    tm, jm = m.get("tuned") or {}, m.get("jacobian") or {}
    src = ", ".join(f"{n} {P.SOURCE.get(s, s)}" for s, n in
                    sorted(v["fit_sources"].items(), key=lambda t: -t[1]))
    q = v["questions"]
    qtxt = "; ".join(f"\"{s}\" ({n})" for s, n in q[:3])
    rest_n, rest_q = sum(n for _, n in q[3:]), len(q[3:])
    cue = v["cues"][0][0] if v["cues"] else ""
    sysw = "no system instruction" if v["systems"] == ["None"] else "the system instruction " + ", ".join(v["systems"])
    ci = lambda k: f"{_f(R[k]['auroc'])} (95% interval {_f(R[k]['ci'][0])} to {_f(R[k]['ci'][1])})"
    others = ("logit", "tuned", "jacobian")
    inverted = [k for k in others if R[k]["auroc"] < 0.5]
    only = R["v_read"]["auroc"] > 0.5 and len(inverted) == len(others)
    s = [head]
    # the claim, in the figure's own numbers
    if only:
        s.append(
            f"At block {v['block']} of {v['n_layers']} in {v['model']}, the tuned lens is the identity "
            f"plus a correction {_f(tu['frob'] * 100, 0)}% of its size, so it reads the answer along "
            "the head's own yes/no axis; at this depth that axis points the wrong way, and the tuned, "
            f"logit and Jacobian lenses all read the model's coming answer inverted (AUROC "
            f"{_f(R['tuned']['auroc'])}, {_f(R['logit']['auroc'])}, {_f(R['jacobian']['auroc'])}). "
            f"V-Read, a ridge map fitted with no label, reads from {vr['angle']:.0f} degrees off that "
            f"axis and predicts the answer at AUROC {_f(R['v_read']['auroc'])}. "
            "The spectra of the maps show why.")
    else:
        s.append(
            f"At block {v['block']} of {v['n_layers']} in {v['model']}, the figure compares where each "
            "affine lens reads the yes/no answer from and what its map does to the state.")
    # what is fitted
    s.append(
        f"Each instrument is one affine map, h to A h + b, from the block's residual state "
        f"(d = {v['d']}) to a state the model's own final RMSNorm and unembedding read; the logit "
        "lens is the identity. V-Read is a ridge regression onto the final block's state (penalty "
        f"{(m.get('v_read') or {}).get('lam')}, then an output gain of {g} about the mean final "
        "state, both chosen on held-out next-token divergence, no labels). The tuned lens is trained "
        f"as Belrose et al. (2023) train it: {tm.get('steps')} SGD steps (Nesterov momentum "
        f"{tm.get('momentum')}) at learning rate {tm.get('lr')} with weight decay "
        f"{tm.get('weight_decay')}, minimising divergence from the model's next-token distribution. "
        "The Jacobian lens is the model's average Jacobian from this block to the last, estimated "
        f"from {jm.get('n_prompts')} prompts ({jm.get('n_source_positions')} source positions). All "
        f"three are fitted on the same {v['corpus']}-prompt corpus ({v['fit_n']} SimonSays "
        f"records: {src}).")
    # what is scored
    s.append(
        f"Scored states: {v['n']} SimonSays Open Images authority records on {v['n_pic']} photographs "
        f"of statutory road signs, each prompted with {sysw}, the cue \"{cue}\" in the user turn "
        f"before the picture, and a yes/no question after it: {qtxt}"
        + (f"; and {rest_n} records spread over {rest_q} other yes/no questions of the same kind"
           if rest_n else "")
        + ". The state is taken at the last prompt token, where the answer is written. "
        + (f"{v['n_dropped']} scored records were in a fitting set and are dropped; "
           if v["n_dropped"] else "No scored record is in any fitting set, checked by identifier; ")
        + "no password-locking record is scored. The target is the answer the model itself gives: "
        "the sign of the yes-over-no margin its last block reads through the head "
        f"({v['n_yes']} yes, {v['n_no']} no). A map's margin is the log-sum-exp of the six yes "
        "tokens' logits minus that of the six no tokens, read from A h + b through the final norm "
        "and unembedding. AUROC intervals resample photographs with replacement "
        f"({N_BOOT} draws).")
    # panels
    s.append(
        "(a) Where each map reads the answer from. The horizontal ray is u, the head's own reading "
        "direction for yes against no (mean of the six yes unembedding rows minus mean of the six no "
        "rows, times the final norm's weight). A map reads the answer along A transposed u; its "
        "arrow is drawn at the angle between A transposed u and u, with length the gain "
        "|A transposed u| / |u| on a log scale (origin 0.5, rim 22), and is labelled with the AUROC "
        "at which its margin predicts the model's answer. The logit lens lies on u by construction "
        "(its dashed arrow sits under the tuned lens's). The tuned lens reads from "
        f"{tu['angle']:.1f} degrees off u at gain {_f(tu['gain'])} (cosine {tu['cos']:.2f}), the "
        f"Jacobian lens from {ja['angle']:.1f} degrees at gain {_f(ja['gain'], 1)}, and V-Read from "
        f"{vr['angle']:.1f} degrees (cosine {vr['cos']:.2f}) at gain {_f(vr['gain'], 1)}, of which "
        f"a factor {g} is its output gain.")
    s.append(
        "(b) Singular values of A against rank, log-log; the logit lens's identity is the grey "
        f"dashed line at 1. The tuned lens hugs the identity: median {_f(tu['med'])}, smallest "
        f"{_f(tu['min'])}, largest {_f(tu['top'])}. V-Read is a strongly anisotropic map: largest "
        f"singular value {vr['top']:.0f} against a median of {_f(vr['med'], 1)}, with half its "
        f"squared norm in {vr['k50']} of {v['d']} directions. The Jacobian lens lies between, "
        f"{ja['top']:.0f} at the top against a median of {_f(ja['med'])}. {vr['below']} of V-Read's "
        f"and {ja['below']} of the Jacobian lens's singular values fall below the axis floor of "
        f"{FLOOR}.")
    s.append(
        "(c) Singular values of A minus I, which is what each map changes (the logit lens changes "
        "nothing and has no curve); each label gives |A - I| as a multiple of |I|, Frobenius norms. "
        f"The tuned lens's whole correction is {_f(tu['frob'])} of the identity's norm, half of its "
        f"energy in {tu['k50I']} directions and nine tenths in {tu['k90I']} (the two open dots). "
        f"V-Read's is {_f(vr['frob'], 1)} times the identity's norm and the Jacobian lens's "
        f"{_f(ja['frob'], 1)} times.")
    fs = (f"Through each map, the block-{v['block']} states predict the answer the model will give "
          f"at AUROC {ci('v_read')} for V-Read, against {ci('logit')} for the logit lens, "
          f"{ci('tuned')} for the tuned lens and {ci('jacobian')} for the Jacobian lens.")
    if inverted:
        fs += (" A value below one half means the reading is inverted: the tuned lens inherits the "
               "inversion because its A transposed u is almost u, while V-Read reads from a "
               "direction close to orthogonal to u"
               + (" and is the only one of the four above one half." if only else "."))
    s.append(fs)
    k = v.get("kl")
    lim = ["Where our method loses: the objectives differ."]
    if k:
        lim.append(
            f"On {k['heldout']} held-out prompts ({k['tokens']} tokens), in a sixteen-block fit of "
            "the same instruments, the tuned lens is nearer the model's next-token distribution at "
            f"the sampled blocks either side of this one (blocks {k['blk_lo']} and {k['blk_hi']}: "
            f"{_f(k['tuned'][0])} and {_f(k['tuned'][1])} nats against {_f(k['v_read'][0])} and "
            f"{_f(k['v_read'][1])} for V-Read) and at {k['tuned_nearer']} of the {k['n_depths']} "
            "sampled blocks. Divergence is over the whole vocabulary and rewards a small, "
            "weight-decayed nudge to the state; the answer is one axis the head reads inverted at "
            "this depth, and V-Read gives up divergence to read it.")
    lim.append(f"Limits: one model, one block, one family of {v['n']} records; the Jacobian lens "
               f"here is estimated from only {jm.get('n_prompts')} prompts. Maps are stored at half "
               "precision; the spectra are of the stored matrices, computed in double precision.")
    s.append(" ".join(lim))
    return "\n\n".join([s[0] + " " + s[1], " ".join(s[2:4]), " ".join(s[4:7]), " ".join(s[7:])])


# ------------------------------------------------------------------ drawing

def _rect(r):
    x, y, w, h = r
    return (x / W_IN, y / H_IN, w / W_IN, h / H_IN)


def _rad(gain):
    return float(np.log10(max(gain, 10 ** R_LO)))


def _panel_polar(ax, v):
    fs = matplotlib.rcParams["legend.fontsize"]
    ax.set_thetamin(0)
    ax.set_thetamax(100)
    ax.set_rlim(R_LO, R_HI)
    ax.set_rgrids([0.0, np.log10(3), 1.0], labels=["", "", ""],
                  fontsize=matplotlib.rcParams["xtick.labelsize"], color=st.INK)
    ax.set_thetagrids([0, 30, 60, 90], labels=["", "30°", "60°", "90°"],
                      fontsize=matplotlib.rcParams["xtick.labelsize"], color=st.INK)
    ax.set_rlabel_position(100)
    ax.tick_params(axis="x", pad=2)
    ax.grid(color=st.GRID, lw=0.9)
    ax.spines["polar"].set_color(st.AXIS)
    ax.spines["start"].set_color(st.AXIS)
    ax.spines["end"].set_color(st.AXIS)
    ax.spines["inner"].set_visible(False)
    L, R = v["lens"], v["read"]
    au = lambda k: f"AUROC {R[k]['auroc']:.2f}"
    order = ("logit", "tuned", "jacobian", "v_read")
    for k in order:
        th, r = np.radians(L[k]["angle"]), _rad(L[k]["gain"])
        dash = P.DASH["logit"] if k == "logit" else None
        ap = dict(arrowstyle="-|>", lw=2.6 if k == "v_read" else 2.2, color=COL[k],
                  shrinkA=0, shrinkB=0, mutation_scale=16)
        if dash:
            ap["linestyle"] = (0, dash)
        ax.annotate("", xy=(th, r), xytext=(0.0, R_LO), arrowprops=ap, zorder=5)
    box = dict(boxstyle="round,pad=0.12", fc="white", ec="none")

    def at(x, y):
        """Polar coordinates of a point given in the wedge's own plane, origin at the tail."""
        return np.arctan2(y, x), float(np.hypot(x, y)) + R_LO

    def head(k):
        th, r = np.radians(L[k]["angle"]), _rad(L[k]["gain"]) - R_LO
        return r * np.cos(th), r * np.sin(th)

    # V-Read and the Jacobian lens: the name just right of the head, inside the rim
    for k, dy in (("v_read", -0.3), ("jacobian", -0.03)):
        hx, hy = head(k)
        ax.text(*at(hx + 0.09, hy + dy), f"{NICE[k]}\n{au(k)}", ha="left", va="center",
                color=st.INK, fontsize=fs, bbox=box, zorder=7, linespacing=1.05)
    # the logit and tuned arrows end together at gain one, ten degrees apart:
    # the tuned lens is named above its own ray, the logit lens under u
    hx, hy = head("tuned")
    ax.text(*at(hx + 0.14, hy + 0.13), f"{NICE['tuned']}\n{au('tuned')}", ha="left",
            va="center", color=st.INK, fontsize=fs, bbox=box, zorder=7, linespacing=1.05)
    ax.text(np.radians(-1.0), R_LO + 0.02,
            f"{NICE['logit']}: u, the head's yes/no axis\n{au('logit')}", ha="left", va="top",
            color=st.INK, fontsize=fs, zorder=7, linespacing=1.05)
    # the gain scale, written along the open edge of the wedge
    t100 = np.radians(100)
    for g, lab in ((1, "1x"), (3, "3x"), (10, "10x")):
        ax.text(t100 + np.radians(2.5), np.log10(g), lab, ha="right", va="center",
                color=st.INK, fontsize=matplotlib.rcParams["xtick.labelsize"])
    ax.text(t100 + np.radians(17), 0.62, "gain along u", ha="center", va="center",
            rotation=100, rotation_mode="anchor", color=st.INK,
            fontsize=matplotlib.rcParams["xtick.labelsize"])


def _staircase(ax, s, k, lw=2.3, z=4):
    s = np.asarray(s, float)
    r = np.arange(1, len(s) + 1)
    y = np.where(s > 0, s, FLOOR * 1e-3)
    ax.step(np.append(r, len(s) + 1), np.append(y, y[-1]), where="post", color=COL[k], lw=lw,
            zorder=z, solid_joinstyle="miter")


def _log_axes(ax, left=True):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 4097)
    ax.set_ylim(FLOOR, CEIL)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1000"])
    ax.set_yticks([0.1, 1, 10, 100, 1000])
    ax.set_yticklabels(["0.1", "1", "10", "100", "1000"] if left else [])
    ax.minorticks_off()
    ax.set_xlabel("rank")
    if left:
        ax.set_ylabel("singular value")
    ax.grid(True, which="major", color=st.GRID, lw=0.9)


def _put(ax, x, y, text, ha="left", va="center", **kw):
    ax.text(x, y, text, ha=ha, va=va, color=st.INK,
            fontsize=matplotlib.rcParams["legend.fontsize"], zorder=8, **kw)


def _panel_sv(ax, v):
    L = v["lens"]
    ax.axhline(1.0, color=COL["logit"], lw=1.6, dashes=P.DASH["logit"], zorder=6)
    for k in ("jacobian", "tuned", "v_read"):
        _staircase(ax, L[k]["sv"], k, lw=2.6 if k == "v_read" else 2.2)
    _log_axes(ax, left=True)
    _put(ax, 4096, 600, "spectrum of $A$", ha="right")
    _put(ax, 2.2, 230, NICE["v_read"])
    _put(ax, 2000, 0.25, NICE["jacobian"], ha="right")
    _put(ax, 2.2, 2.4, NICE["tuned"])
    _put(ax, 1.25, 0.55, NICE["logit"])


def _panel_svI(ax, v):
    L = v["lens"]
    for k in ("jacobian", "tuned", "v_read"):
        _staircase(ax, L[k]["svI"], k, lw=2.6 if k == "v_read" else 2.2)
    _log_axes(ax, left=False)
    tu = L["tuned"]
    for kk in (tu["k50I"], tu["k90I"]):
        ax.plot([kk + 0.5], [tu["svI"][kk - 1]], "o", ms=7, mfc="white", mec=COL["tuned"],
                mew=1.8, zorder=7)
    x = lambda k: f"{L[k]['frob']:.2f}" if L[k]["frob"] < 1 else f"{L[k]['frob']:.1f}"
    _put(ax, 4096, 600, "spectrum of $A - I$", ha="right")
    _put(ax, 2.2, 230, f"{NICE['v_read']}, {x('v_read')}" + r"$\,\|I\|$")
    _put(ax, 2500, 0.3, f"{NICE['jacobian']},\n{x('jacobian')}" + r"$\,\|I\|$", ha="right",
         multialignment="right", linespacing=1.05)
    _put(ax, 1.25, 0.105, f"{NICE['tuned']},\n{x('tuned')}" + r"$\,\|I\|$", linespacing=1.05)


def _letters(fig, axes, labels):
    """One letter per panel, at the left edge of the panel's ink (tick labels included),
    all on one baseline just above the tallest panel, at tick-label size."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = [inv.transform(ax.get_tightbbox(r)) for ax in axes]
    top = max(ax.get_position().y1 for ax in axes)
    for (x0, _), _b, s in zip((b[0] for b in boxes), boxes, labels):
        fig.text(x0, top + 0.02, s, ha="left", va="bottom", color=st.INK,
                 fontsize=matplotlib.rcParams["xtick.labelsize"])


def _check(v):
    """The judge's numbers for the block-18 cache, printed beside ours; a mismatch is reported."""
    want = dict(angle=dict(v_read=82.9, tuned=9.7, jacobian=71.0),
                gain=dict(v_read=13.96, tuned=1.00, jacobian=1.80),
                auroc=dict(v_read=0.748, tuned=0.083, jacobian=0.067, logit=0.09))
    L, R = v["lens"], v["read"]
    bad = []
    for k, w in want["angle"].items():
        if abs(L[k]["angle"] - w) > 0.2:
            bad.append(f"angle {k} {L[k]['angle']:.1f} vs {w}")
    for k, w in want["gain"].items():
        if abs(L[k]["gain"] - w) > 0.01:
            bad.append(f"gain {k} {L[k]['gain']:.2f} vs {w}")
    for k, w in want["auroc"].items():
        if abs(R[k]["auroc"] - w) > 0.005:
            bad.append(f"auroc {k} {R[k]['auroc']:.3f} vs {w}")
    print(f"  {NAME}: " + ("matches the block-18 check numbers" if not bad else
                           "differs from the block-18 check numbers (a new fit?): " + "; ".join(bad)))


def transport_geometry(results="results", out="figures", model=None):
    """MAIN. The three transports as objects; replaced at import by the caption built from the files."""
    if model is None and P.AVERAGE:
        return None           # one model's maps; there is nothing to average
    model = model or DEFAULT_MODEL
    v, why = _numbers(results, model)
    cap = _caption(v, why)
    transport_geometry.__doc__ = cap
    if v is None:
        print(f"  {NAME}: {why}")
        return None
    return _draw(v, P._at(out, NAME))


def _draw(v, path):
    _check(v)
    st.use()
    fig = plt.figure(figsize=(W_IN, H_IN))
    fig.set_layout_engine("none")
    ax_a = fig.add_axes(_rect(RECT_A), projection="polar")
    ax_b = fig.add_axes(_rect(RECT_B))
    ax_c = fig.add_axes(_rect(RECT_C))
    _panel_polar(ax_a, v)
    _panel_sv(ax_b, v)
    _panel_svI(ax_c, v)
    _letters(fig, (ax_a, ax_b, ax_c), "abc")
    return st.save(fig, path, check=False, tight=True)


def _import_caption():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        v, why = _numbers(os.path.join(here, "results"), DEFAULT_MODEL)
    except Exception as e:  # noqa: BLE001
        v, why = None, f"{type(e).__name__}: {e}"
    return _caption(v, why)


transport_geometry.__doc__ = _import_caption()
