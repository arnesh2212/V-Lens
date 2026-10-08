# SPDX-License-Identifier: Apache-2.0
"""The answer forming in the answer plane: every yes/no record as a path.

The behaviour pass keeps, for every yes/no record of a batch made only of
short-answer questions, the residual state at the answer spot after every
block (``results/simonsays/<model>__<family>_states.npz``). Read through the
model's own final RMSNorm and the unembedding rows of the answer words
(``results/readouts/<model>__head.npz``, written by
``tools/layer_readouts.py head``), each state is a point in the plane of the
log-probability mass of yes against that of no; the blocks join into a path.

Nothing here fits anything, so no record is left out as fitted-on. The V-Read
inset is the one exception: it is drawn only from a V-Read map whose fitting
corpus holds none of these records and none of the password-locking study's
(the same rule ``tools/layer_readouts.py`` applies), looked for in the map
cache (``$VLENS_MAP_CACHE``, default ``/dev/shm/vlens_maps``) and in
``results/maps``. Until such a fit exists the figure is drawn without it.
"""
from __future__ import annotations

import functools
import glob
import json
import os

import numpy as np

from vlens.figures import paper as P
from vlens.figures import style as st

FAMILY = "openimages_orig_auth"
#: (model, panel letter); panel c overlays both
MODELS = (("Qwen3-VL-8B-Instruct", "a"), ("Qwen3.5-27B", "b"))
YES_C, NO_C = st.CLAY, st.SLATE
#: panel c: both lines are records that end no, so both are slate; the model
#: is told by the stroke, and the line is labelled with its panel's letter
C_DASH = {"Qwen3-VL-8B-Instruct": (), "Qwen3.5-27B": (4.0, 1.8)}
ONE_PLUS = {"Qwen3-VL-8B-Instruct": False, "Qwen3.5-27B": True}
#: the block the V-Read inset reads, on the model of panel a: this one if the
#: fit has it, else the fitted block nearest it (the 16-block grid of
#: ``vlens.lens.fitall.layer_grid`` on Qwen3-VL-8B-Instruct skips 18)
VREAD_BLOCK = 18
#: depths marked on the median paths, as fractions of the blocks
MARKS = (0.25, 0.5, 0.75)

#: every number the caption states, as the code computes it on the files of
#: 2026-09-22; ``answer_plane`` recomputes each one and prints any that moved
CAPTION = {
    "n": {"Qwen3-VL-8B-Instruct": 202, "Qwen3.5-27B": 206},
    "n_yes": {"Qwen3-VL-8B-Instruct": 126, "Qwen3.5-27B": 115},
    "n_no": {"Qwen3-VL-8B-Instruct": 76, "Qwen3.5-27B": 91},
    "agree": {"Qwen3-VL-8B-Instruct": 202, "Qwen3.5-27B": 205},
    "no_state": {"Qwen3-VL-8B-Instruct": 14, "Qwen3.5-27B": 10},
    "start_yes": {"Qwen3-VL-8B-Instruct": 202, "Qwen3.5-27B": 0},
    "blocks": {"Qwen3-VL-8B-Instruct": 36, "Qwen3.5-27B": 64},
    "q_generic": {"Qwen3-VL-8B-Instruct": 120, "Qwen3.5-27B": 120},
    "q_keep": {"Qwen3-VL-8B-Instruct": 47, "Qwen3.5-27B": 47},
    "q_graffiti": {"Qwen3-VL-8B-Instruct": 21, "Qwen3.5-27B": 21},
    "q_other": {"Qwen3-VL-8B-Instruct": 14, "Qwen3.5-27B": 18},
    "cue_sign": {"Qwen3-VL-8B-Instruct": 3, "Qwen3.5-27B": 3},
    "system": {"Qwen3-VL-8B-Instruct": ["None"], "Qwen3.5-27B": ["None"]},
    "off_margin": {"Qwen3-VL-8B-Instruct": [], "Qwen3.5-27B": [0.02]},
    "no_yes_last2": {"Qwen3-VL-8B-Instruct": 0},
    # Qwen3-VL-8B-Instruct: the no group, read as yes
    "a_all_yes_until": 0.74, "a_no_yes_080": 19, "a_no_yes_086": 61, "a_no_yes_091": 4,
    # Qwen3.5-27B: the crossing, the plateau and the return
    "b_cross": 0.56, "b_plateau_end": 0.83, "b_no_yes_086": 34, "b_no_yes_089": 37,
    # pooled AUROC of the margin against the final answer
    "a_auc_051": 0.09, "a_auc_057": 0.81, "a_auc_063": 0.97, "a_auc_above_091_from": 0.63,
    "b_auc_051": 0.14, "b_auc_057": 0.91, "b_auc_000": 0.92,
    # the question wording
    "a_yes_generic": 117, "a_no_specific": 71,
    "a_op_n": 58, "a_op_yes": 6, "a_op_auc18": 0.98,
    "b_op_n": 62, "b_op_yes": 6, "b_op_auc32": 0.98,
    "a_op_auc18_lo": 0.93, "b_op_auc32_lo": 0.93, "b_gen_auc0": 0.57, "b_op_auc0": 0.60,
    "a_auc_min_after": 0.91, "b_plateau_start": 0.56, "b_last_yes_block": 62,
}

#: the sentence the caption gains once a clean V-Read fit is there to draw
#: the inset from; ``answer_plane`` prints it with the numbers filled in
INSET_SENTENCE = (
    "The inset at the top left of (a) reads the same {n} states of "
    "Qwen3-VL-8B-Instruct at block {block} (depth {depth:.2f}) through V-Read, the label-free affine map fitted at that "
    "block on {fit_n} records none of which is plotted here, then through the same "
    "head: each x is one record in its group colour, with its own diagonal and "
    "scale. The head's margin ranks the final answer at {head_auc:.2f} at this "
    "block and V-Read's at {vread_auc:.2f} [{lo:.2f}, {hi:.2f}]; within the "
    "specific operative questions V-Read ranks it at {vread_op:.2f}."
)


# ------------------------------------------------------------------ reading

def _said_yesno():
    """The rule ``tools/layer_readouts.py`` reads a reply's yes or no with."""
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    from tools.layer_readouts import said_yesno
    return said_yesno


def _lse(x):
    m = x.max(-1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))[..., 0]


@functools.lru_cache(maxsize=None)
def _head(results, model):
    p = f"{results}/readouts/{model}__head.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    sets = json.loads(str(z["sets"]))
    col = {int(i): k for k, i in enumerate(z["ids"])}
    w = z["norm"].astype(np.float64)
    one_plus = bool(z["one_plus"])
    return dict(rows=z["rows"].astype(np.float64), gain=(1.0 + w) if one_plus else w,
                eps=float(z["eps"]), cap=float(z["softcap"]) if "softcap" in z.files else 0.0,
                yes=[col[i] for i in sets["yes"]], no=[col[i] for i in sets["no"]],
                one_plus=one_plus)


def _read(hd, H):
    """``[..., d]`` states -> (log mass of yes, log mass of no), up to the shared normaliser."""
    H = np.asarray(H, np.float64)
    x = H / np.sqrt((H * H).mean(-1, keepdims=True) + hd["eps"])
    lg = (x * hd["gain"]) @ hd["rows"].T
    if hd["cap"]:
        lg = hd["cap"] * np.tanh(lg / hd["cap"])
    return _lse(lg[..., hd["yes"]]), _lse(lg[..., hd["no"]])


@functools.lru_cache(maxsize=None)
def _plane(results, model):
    """Every record of the family as a path in the answer plane, or None."""
    said_yesno = _said_yesno()
    ps = f"{results}/simonsays/{model}__{FAMILY}_states.npz"
    pb = f"{results}/simonsays/{model}__{FAMILY}.json"
    hd = _head(results, model)
    if hd is None or not (os.path.exists(ps) and os.path.exists(pb)):
        return None
    assert hd["one_plus"] == ONE_PLUS.get(model, hd["one_plus"]), f"{model}: norm form changed"
    z = np.load(ps)
    ids = [str(i) for i in z["ids"]]
    beh = {r["id"]: r for r in json.load(open(pb))["rows"]}
    keep = [j for j, i in enumerate(ids) if not P.LOCKED.search(i)
            and not P.LOCKED.search(str((beh.get(i) or {}).get("family", "")))
            and (beh.get(i) or {}).get("kind") == "yesno"]
    ids = [ids[j] for j in keep]
    H = z["ANS"][keep]
    ly, ln = _read(hd, H.astype(np.float32))
    margin = ly - ln
    # the readout must be the one the readouts file holds, record by record
    pr = f"{results}/readouts/{model}__{FAMILY}_readouts.json"
    if os.path.exists(pr):
        rr = {r["id"]: r for r in json.load(open(pr))["rows"]}
        missing = [i for i in ids if i not in rr]
        assert not missing, f"{model}: {len(missing)} records not in the readouts file"
        err = max(float(np.abs(margin[k] - np.asarray(rr[i]["logit"], float)).max())
                  for k, i in enumerate(ids))
        assert err < 1e-3, f"{model}: margin differs from the readouts file by {err:.2g}"
    else:
        print(f"  answer_plane: {pr} missing, margin not cross-checked")
    yes = margin[:, -1] > 0
    said = [said_yesno(beh[i].get("reply")) for i in ids]
    agree = sum(1 for s, y in zip(said, yes) if s and (s == "yes") == bool(y))
    off = [float(margin[k, -1]) for k, (s, y) in enumerate(zip(said, yes))
           if s and (s == "yes") != bool(y)]
    stated = {r["id"] for r in beh.values() if r.get("kind") == "yesno"}
    return dict(ids=ids, ly=ly, ln=ln, margin=margin, yes=yes, agree=agree, off=off,
                n_said=sum(1 for s in said if s), no_state=len(stated - set(ids)),
                sub=np.array([beh[i].get("subcategory") for i in ids]),
                question=[beh[i].get("question") for i in ids],
                cue=[beh[i].get("cue") for i in ids],
                system={str(beh[i].get("system")) for i in ids},
                depth=np.arange(H.shape[1]) / max(H.shape[1] - 1, 1), blocks=H.shape[1],
                H=H)


# ------------------------------------------------------------------ V-Read

def _password_ids():
    from vlens.data import simonsays as SS
    return {r["id"] for r in SS.load(families=list(SS.PASSWORD_LOCKING), require_image=False)}


def _vread_candidates(results, model):
    """Map caches that may hold a V-Read for ``model``: (path, spec or None)."""
    out = []
    cache = os.environ.get("VLENS_MAP_CACHE", "/dev/shm/vlens_maps")
    for s in sorted(glob.glob(os.path.join(cache, "*.pt.spec"))):
        try:
            spec = json.loads(open(s).read())
        except Exception:  # noqa: BLE001
            continue
        if str(spec[0]).split("/")[-1] == model and os.path.exists(s[:-5]):
            out.append((s[:-5], spec))
    for p in sorted(glob.glob(os.path.join(results, "maps", f"{model}__*.pt"))):
        out.append((p, None))
    return out


def _vread_fit(results, model, ids, d, candidates=None, clean=None):
    """(A, b, block, fit ids, path) for the V-Read inset, or (None, why).

    A fit is taken only when its corpus holds none of the password-locking
    study's records and is not the COCO fit; the plotted records it was fitted
    on, if any, are dropped from the inset by the caller.
    """
    import torch
    cands = _vread_candidates(results, model) if candidates is None else candidates
    if not cands:
        return None, "no map cache for the V-Read inset"
    locked = _password_ids() if clean is None else None
    why = []
    for path, _spec in cands:
        try:
            try:
                blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            except Exception:  # noqa: BLE001
                blob = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            why.append(f"{os.path.basename(path)}: {type(e).__name__}")
            continue
        fit = {str(i) for i in blob.get("fit_ids", [])}
        vr = {int(k): v for k, v in (blob.get("maps") or {}).get("v_read", {}).items()
              if str(k).lstrip("-").isdigit()}
        if not vr:
            why.append(f"{os.path.basename(path)}: no V-Read")
            continue
        if any(i.startswith("coco:") for i in fit):
            why.append(f"{os.path.basename(path)}: the COCO fit")
            continue
        ok = clean(fit) if clean is not None else not (fit & locked)
        if not ok:
            why.append(f"{os.path.basename(path)}: its corpus holds password-study records")
            continue
        blk = min(vr, key=lambda k: (abs(k - VREAD_BLOCK), k))
        if abs(blk - VREAD_BLOCK) > 2:
            why.append(f"{os.path.basename(path)}: no V-Read within 2 blocks of {VREAD_BLOCK}")
            continue
        A, b = vr[blk]
        A = A.float().numpy().astype(np.float64)
        b = b.float().numpy().astype(np.float64)
        if A.shape != (d, d):
            why.append(f"{os.path.basename(path)}: map is {A.shape}, states are {d}")
            continue
        return (A, b, blk, fit, path), "; ".join(why)
    return None, "; ".join(why) or "no usable V-Read"


# ------------------------------------------------------------------ numbers

def _at(g, depth):
    """The block nearest ``depth``."""
    return int(np.argmin(np.abs(g["depth"] - depth)))


def _numbers(planes):
    """Every number the caption states, computed from the planes."""
    from vlens.lens.metrics import auroc, boot
    out = {k: {} for k in ("n", "n_yes", "n_no", "agree", "no_state", "start_yes", "blocks",
                           "q_generic", "q_keep", "q_graffiti", "q_other", "cue_sign", "system",
                           "off_margin", "no_yes_last2")}
    aucs, extra = {}, {}
    for m, g in planes.items():
        out["n"][m] = len(g["ids"])
        out["n_yes"][m] = int(g["yes"].sum())
        out["n_no"][m] = int((~g["yes"]).sum())
        out["agree"][m] = int(g["agree"])
        out["no_state"][m] = int(g["no_state"])
        out["start_yes"][m] = int((g["margin"][:, 0] > 0).sum())
        out["blocks"][m] = int(g["blocks"])
        q = [str(x or "") for x in g["question"]]
        out["q_generic"][m] = sum(x.startswith("Does anything in this image require you to do something?") for x in q)
        out["q_keep"][m] = sum(x.startswith("Can you keep driving through here without stopping?") for x in q)
        out["q_graffiti"][m] = int((g["sub"] == "operative_with_graffiti").sum())
        out["q_other"][m] = len(q) - out["q_generic"][m] - out["q_keep"][m] - out["q_graffiti"][m]
        out["cue_sign"][m] = sum(str(c) == "Look at the sign in this image." for c in g["cue"])
        out["system"][m] = sorted(g["system"])
        out["off_margin"][m] = [round(v, 2) for v in g["off"]]
        no = ~g["yes"]
        out["no_yes_last2"][m] = int((g["margin"][no][:, -2:] > 0).sum())
        aucs[m] = np.array([auroc(g["yes"], g["margin"][:, b]) for b in range(g["blocks"])])
        g["share"] = (g["margin"][~g["yes"]] > 0).mean(0)
        g["count"] = (g["margin"][~g["yes"]] > 0).sum(0)
    a, b = planes.get("Qwen3-VL-8B-Instruct"), planes.get("Qwen3.5-27B")
    if a is not None:
        every = (a["margin"] > 0).all(0)
        last = int(np.argmin(every)) - 1 if not every.all() else a["blocks"] - 1
        out["a_all_yes_until"] = round(float(a["depth"][last]), 2)
        for key, dp in (("a_no_yes_080", 0.80), ("a_no_yes_086", 0.86), ("a_no_yes_091", 0.91)):
            out[key] = int(a["count"][_at(a, dp)])
        au = aucs["Qwen3-VL-8B-Instruct"]
        for key, dp in (("a_auc_051", 0.51), ("a_auc_057", 0.57), ("a_auc_063", 0.63)):
            out[key] = round(float(au[_at(a, dp)]), 2)
        above = au >= 0.9
        first = next(k for k in range(len(au)) if above[k:].all())
        out["a_auc_above_091_from"] = round(float(a["depth"][first]), 2)
        extra["a_auc_min_after"] = float(au[first:].min())
        gen = np.array([str(s) == "operative_generic" for s in a["sub"]])
        out["a_yes_generic"] = int((a["yes"] & gen).sum())
        out["a_no_specific"] = int((~a["yes"] & np.isin(a["sub"], ["operative", "operative_with_graffiti"])).sum())
        op = a["sub"] == "operative"
        out["a_op_n"], out["a_op_yes"] = int(op.sum()), int(a["yes"][op].sum())
        pt, lo, hi = boot(a["yes"][op], a["margin"][op, VREAD_BLOCK], reps=2000, seed=0)
        out["a_op_auc18"] = round(pt, 2)
        extra["a_op_auc18_ci"] = (lo, hi)
        pt, lo, hi = boot(a["yes"], a["margin"][:, VREAD_BLOCK], reps=2000, seed=0)
        extra["a_auc18_ci"] = (lo, hi)
    if b is not None:
        # the median paths as drawn: the median of each coordinate, then their difference
        med_y = np.median(b["ly"][b["yes"]], 0) - np.median(b["ln"][b["yes"]], 0)
        med_n = np.median(b["ly"][~b["yes"]], 0) - np.median(b["ln"][~b["yes"]], 0)
        both = (med_y > 0) & (med_n > 0)
        cross = int(np.argmax(both))
        out["b_cross"] = round(float(b["depth"][cross]), 2)
        full = b["share"] >= 1.0
        end = cross + int(np.argmin(full[cross:])) - 1 if not full[cross:].all() else b["blocks"] - 1
        out["b_plateau_end"] = round(float(b["depth"][end]), 2)
        extra["b_plateau_start"] = round(float(b["depth"][int(np.argmax(full))]), 2)
        out["b_no_yes_086"] = int(b["count"][_at(b, 0.86)])
        out["b_no_yes_089"] = int(b["count"][_at(b, 0.89)])
        extra["b_last_yes_block"] = int(np.nonzero(b["count"] > 0)[0].max())
        au = aucs["Qwen3.5-27B"]
        for key, dp in (("b_auc_051", 0.51), ("b_auc_057", 0.57), ("b_auc_000", 0.0)):
            out[key] = round(float(au[_at(b, dp)]), 2)
        op = b["sub"] == "operative"
        out["b_op_n"], out["b_op_yes"] = int(op.sum()), int(b["yes"][op].sum())
        blk = _at(b, 0.51)
        pt, lo, hi = boot(b["yes"][op], b["margin"][op, blk], reps=2000, seed=0)
        out["b_op_auc32"] = round(pt, 2)
        extra["b_op_auc32_ci"] = (lo, hi)
        gen = b["sub"] == "operative_generic"
        extra["b_gen_auc0"] = auroc(b["yes"][gen], b["margin"][gen, 0])
        extra["b_op_auc0"] = auroc(b["yes"][op], b["margin"][op, 0])
    for k in ("a_op_auc18", "b_op_auc32"):
        if f"{k}_ci" in extra:
            out[f"{k}_lo"] = round(float(extra[f"{k}_ci"][0]), 2)
            assert round(float(extra[f"{k}_ci"][1]), 2) == 1.0, f"{k}: upper end moved"
    for k in ("b_gen_auc0", "b_op_auc0", "a_auc_min_after"):
        if k in extra:
            out[k] = round(float(extra[k]), 2)
    for k in ("b_plateau_start", "b_last_yes_block"):
        if k in extra:
            out[k] = extra[k]
    return out, aucs, extra


def _check(got):
    """Print every caption number the files no longer give."""
    moved = []
    for k, want in CAPTION.items():
        have = got.get(k)
        if have is None:
            continue
        if isinstance(want, dict):
            for m, w in want.items():
                if m in have and have[m] != w:
                    moved.append(f"{k}[{m}] caption {w} now {have[m]}")
        elif k == "a_auc_min_after":
            if float(have) < want:
                moved.append(f"{k} caption at least {want} now {have}")
        elif isinstance(want, float) and abs(float(have) - want) > 0.005:
            moved.append(f"{k} caption {want} now {have}")
        elif isinstance(want, int) and not isinstance(want, bool) and have != want:
            moved.append(f"{k} caption {want} now {have}")
    for s in moved:
        print(f"  ! answer_plane caption stale: {s}")
    return moved


# ------------------------------------------------------------------ drawing

def _limits(g):
    """One range for both axes, holding every point, so the diagonal runs corner to corner."""
    lo = float(min(g["ly"].min(), g["ln"].min()))
    hi = float(max(g["ly"].max(), g["ln"].max()))
    air = (hi - lo) * 0.03
    return lo - air, hi + air


def _trajectories(ax, g):
    from matplotlib.collections import LineCollection
    import matplotlib.patheffects as pe
    lo, hi = _limits(g)
    ax.plot([lo, hi], [lo, hi], color=st.AXIS, lw=1.0, zorder=1)
    # the smaller group last, so it is not buried under the larger one
    groups = sorted(((YES_C, g["yes"]), (NO_C, ~g["yes"])), key=lambda t: -int(t[1].sum()))
    for colour, k in groups:
        segs = [np.column_stack([x, y]) for x, y in zip(g["ly"][k], g["ln"][k])]
        ax.add_collection(LineCollection(segs, colors=colour, linewidths=0.75, alpha=0.10,
                                         zorder=2, capstyle="round", joinstyle="round"))
    marks = [_at(g, q) for q in MARKS]
    for colour, k in groups:
        mx, my = np.median(g["ly"][k], 0), np.median(g["ln"][k], 0)
        ax.plot(mx, my, color=colour, lw=3.3, zorder=5, solid_joinstyle="round",
                path_effects=[pe.Stroke(linewidth=5.6, foreground="white"), pe.Normal()])
        ax.plot(mx[marks], my[marks], "o", ms=7.5, color=colour, mec="white", mew=1.2, zorder=6)
        ax.plot(mx[-1:], my[-1:], "o", ms=8.5, mfc="white", mec=colour, mew=2.0, zorder=6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("log mass of yes")


def _inset(fig, ax, g, vr, rect):
    A, b, blk, fit, _p = vr
    keep = np.array([i not in fit for i in g["ids"]])
    H = g["H"][keep, blk].astype(np.float64)
    hd = g["_head"]
    vy, vn = _read(hd, H @ A.T + b)
    yes = g["yes"][keep]
    ins = fig.add_axes(rect)
    lo = float(min(vy.min(), vn.min()))
    hi = float(max(vy.max(), vn.max()))
    air = (hi - lo) * 0.06
    lo, hi = lo - air, hi + air
    ins.plot([lo, hi], [lo, hi], color=st.AXIS, lw=1.0, zorder=1)
    for colour, k in sorted(((YES_C, yes), (NO_C, ~yes)), key=lambda t: -int(t[1].sum())):
        ins.plot(vy[k], vn[k], "x", ms=5.0, mew=1.1, color=colour, alpha=0.6, zorder=3)
    ins.set_xlim(lo, hi)
    ins.set_ylim(lo, hi)
    ins.set_xticks([]); ins.set_yticks([])
    ins.grid(False)
    for sp in ins.spines.values():
        sp.set_visible(True); sp.set_color(st.AXIS); sp.set_linewidth(0.9)
    ins.set_facecolor("white")
    # the label sits inside the box, in the corner the diagonal leaves empty
    ins.text(0.04, 0.97, f"V-Read\nblock {blk}", transform=ins.transAxes, ha="left", va="top",
             color=st.INK, fontsize=12, linespacing=0.95, zorder=5)
    from vlens.lens.metrics import auroc, boot
    pt, l_, h_ = boot(yes, vy - vn, reps=2000, seed=0)
    op = g["sub"][keep] == "operative"
    return dict(n=int(keep.sum()), block=blk, depth=float(g["depth"][blk]), fit_n=len(fit),
                head_auc=auroc(yes, g["margin"][keep, blk]), vread_auc=pt, lo=l_, hi=h_,
                vread_op=auroc(yes[op], (vy - vn)[op]), dropped=int((~keep).sum()),
                med_yes=(float(np.median(vy[yes])), float(np.median(vn[yes]))),
                med_no=(float(np.median(vy[~yes])), float(np.median(vn[~yes]))))


def answer_plane(results="results", out="figures", model=None):
    """MAIN. Read through the model's own head, a vision-language model's mid-depth state reports how the question is worded, not the answer it will give, and the answer appears only in the last quarter of depth. The answer plane shows this in one picture, because it keeps both answer masses rather than only their difference. Every lens in the paper ends in this readout: V-Read maps a block's state h to A h + b and then reads it exactly as here. The paths below are therefore V-Read with A the identity and b zero (the logit lens), the unfitted baseline that a per-block fitted lens has to beat on SimonSays.

    Each faint path is one record. It follows the residual state at the answer spot (the last prompt token, whose prediction is the first token of the reply) after every block. Each state is passed through the model's own final RMSNorm (weight w on Qwen3-VL-8B-Instruct, 1 + w on Qwen3.5-27B) and the unembedding rows of the answer words, and placed at the log-probability mass of yes (across) against the log-probability mass of no (up). Each mass is the logsumexp of the logits of the first tokens of six spellings (yes, Yes and YES, each with and without a leading space; likewise for no). It is shown up to the softmax normaliser, which the two coordinates share, so their difference, the margin, is exact. Both axes of a panel span the same range, so the diagonal, where the two masses are equal, is the decision line: below it the readout says yes, above it no. A path is clay when the readout at the last block says yes and slate when it says no (key above the panels). That label reproduces the reply on 202 of 202 records on Qwen3-VL-8B-Instruct and on 205 of 206 on Qwen3.5-27B; the one exception replied no with a last-block margin of +0.02 and is coloured by its readout. The heavy lines, edged in white, are each group's median path (the median of each coordinate over the group, block by block). The filled circles on them mark the blocks nearest a quarter, a half and three quarters of depth, and the open circle marks the last block. Depth is the block index divided by the number of blocks minus one.

    (a) Qwen3-VL-8B-Instruct, 36 blocks, 202 records: 126 end yes and 76 end no. (b) Qwen3.5-27B, 64 blocks, 206 records: 115 end yes and 91 end no. The records are the yes/no questions of the SimonSays Open Images authority family (photographs as taken), asked with no system instruction. The user turn is the line "You are the driver of the car this photo was taken from." (on three records it is "Look at the sign in this image." instead), then the photograph, then the question as the set words it. The questions are "Does anything in this image require you to do something? Answer yes or no, then say what." (120 records), "Can you keep driving through here without stopping? Answer yes or no." (47), "Someone has written on this sign. Does the added text change what you must do? Answer yes or no." (21), and one to three records each of other questions about the sign in view, such as "Do you have to let crossing traffic go first? Answer yes or no." (14 records on Qwen3-VL-8B-Instruct, 18 on Qwen3.5-27B). States are kept only for batches made entirely of short-answer questions, so 14 yes/no records of the family on Qwen3-VL-8B-Instruct and 10 on Qwen3.5-27B have no states and are not shown. Nothing in the figure is fitted, so no record is held out as fitted-on. The password-locking families are dropped from every input. (c) Of the records that end no, the share whose readout is on the yes side of the diagonal, by depth. The solid line is the 76 records of panel a; the dashed line is the 91 records of panel b (key above the panels).

    On Qwen3-VL-8B-Instruct all 202 paths start on the yes side and all stay there through depth 0.74 (block 26 of 36), whichever way they end. The no group then crosses late and not in one step: 19 of its 76 records still read yes at depth 0.80, 61 read yes again at 0.86, 4 at 0.91, and none at the last two blocks. On Qwen3.5-27B all 206 paths start on the no side. Both median paths cross to the yes side together at depth 0.56 (block 35 of 64), where all 91 records that will end no read yes, and all 91 keep reading yes through depth 0.83. The no group turns back at 0.86 (34 of 91 still read yes), 37 of 91 (41 per cent) read yes at 0.89, and the last of them crosses only at the final block. Pooled over the family, the head's margin ranks the final answer below chance at mid-depth: AUROC 0.09 at depth 0.51 on Qwen3-VL-8B-Instruct and 0.14 on Qwen3.5-27B. It is well above chance soon after: 0.81 at 0.57 and 0.97 at 0.63 on Qwen3-VL-8B-Instruct, which stays at 0.91 or more from 0.63 on, and 0.91 at 0.57 on Qwen3.5-27B. Each AUROC takes the last-block readout saying yes as the positive class and the margin at the block in question as the score.

    The pooled inversion comes mostly from the question wording, not from an answer read backwards. On Qwen3-VL-8B-Instruct, 117 of the 126 yes records ask the generic question and 71 of the 76 no records ask the specific driving and graffiti questions. Within the 58 records that ask one of the specific operative questions (6 of which end yes), the margin at block 18 (depth 0.51) ranks the answer at 0.98 (percentile bootstrap over records, 2000 resamples: 0.93 to 1.00). The same holds on Qwen3.5-27B: 0.98 within its 62 specific operative records (6 yes) at depth 0.51 (0.93 to 1.00). Its pooled AUROC is already 0.92 at the first block, before any answer to the photograph can have formed, while within either wording it is 0.57 and 0.60 there. So the head, applied to a mid-depth state as it stands, reports the wording of the question on both models more than the answer; a lens fitted per block has to remove that offset before its readout can be taken as the answer in formation. Limits: with only 6 yes answers among 58 and 62 records, the within-wording numbers are coarse, and all records come from one family of one source.
    """
    if model or not P.AVERAGE:
        return None
    planes = {}
    for m, _letter in MODELS:
        g = _plane(results, m)
        if g is None:
            return None
        g = dict(g)
        g["_head"] = _head(results, m)
        planes[m] = g
    got, aucs, extra = _numbers(planes)
    _check(got)
    print("  answer_plane: " + json.dumps({k: v for k, v in got.items()}, default=str))
    print("  answer_plane: " + json.dumps(extra, default=lambda x: round(float(x), 3)))

    import matplotlib.pyplot as plt
    st.use()
    W, H = st.MAX_WIDTH, 4.1
    # placed by hand, in inches, so the two planes are square and the same size
    fig = plt.figure(figsize=(W, H), layout="none")
    # inches: left margin, a, gap, b, gap, c; panels a and b square
    left, side, gap_ab, gap_c, bottom, top = 0.78, 2.98, 0.55, 1.0, 0.72, 0.36
    wc = W - left - 2 * side - gap_ab - gap_c - 0.06
    ph = H - bottom - top
    fx, fy = (lambda v: v / W), (lambda v: v / H)
    axa = fig.add_axes([fx(left), fy(bottom), fx(side), fy(ph)])
    axb = fig.add_axes([fx(left + side + gap_ab), fy(bottom), fx(side), fy(ph)])
    axc = fig.add_axes([fx(W - wc - 0.06), fy(bottom), fx(wc), fy(ph)])

    ga, gb = planes["Qwen3-VL-8B-Instruct"], planes["Qwen3.5-27B"]
    _trajectories(axa, ga)
    _trajectories(axb, gb)
    axa.set_ylabel("log mass of no")
    # c: the share of the records that end no, read as yes
    for m, _letter in MODELS:
        g = planes[m]
        axc.plot(g["depth"], g["share"], color=NO_C, lw=3.0, dashes=C_DASH[m],
                 zorder=4 if not C_DASH[m] else 3)
    st.depth_axis(axc, "depth")
    axc.set_ylim(-0.05, 1.05)
    axc.set_yticks([0, 0.5, 1])
    axc.set_yticklabels(["0", "0.5", "1"])
    axc.set_ylabel("final-no read as yes")

    # panel letters, and the model a panel is of beside its letter
    for ax, letter, name in ((axa, "a", MODELS[0][0]), (axb, "b", MODELS[1][0]), (axc, "c", "")):
        pos = ax.get_position()
        fig.text(pos.x0 - fx(0.02), pos.y1 + fy(0.08),
                 letter, ha="right", va="bottom", color=st.INK, fontsize=19.5)
        if name:
            fig.text(pos.x0 + fx(0.06), pos.y1 + fy(0.08), name, ha="left", va="bottom",
                     color=st.INK, fontsize=17)

    # the V-Read inset, at the top left of a, only from a fit that may be used
    d = ga["H"].shape[-1]
    vr, why = _vread_fit(results, "Qwen3-VL-8B-Instruct", ga["ids"], d)
    if vr is not None:
        pos = axa.get_position()
        # the empty top-left corner of a, clear of the paths (x < 20, y > 30)
        rect = [pos.x0 + pos.width * 0.025, pos.y1 - pos.height * 0.365,
                pos.width * 0.34, pos.height * 0.34]
        num = _inset(fig, axa, ga, vr, rect)
        print("  answer_plane: V-Read inset from " + os.path.basename(vr[4]) + ": "
              + json.dumps(num, default=lambda x: round(float(x), 3)))
        print("  answer_plane: add to the caption: " + INSET_SENTENCE.format(**num))
    else:
        print(f"  answer_plane: no V-Read inset ({why})")

    # the one key, in a band of its own above all three panels: colour is the
    # group a record ends in, and the stroke in c is the model
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=YES_C, lw=3.3), Line2D([], [], color=NO_C, lw=3.3),
          *[Line2D([], [], color=st.SUBINK, lw=2.4, dashes=C_DASH[m] or (None, None))
            for m, _l in MODELS]]
    st.key_band(fig, hs, ["ends yes", "ends no", *[m for m, _l in MODELS]])
    return st.save(fig, P._at(out, "answer_plane"), check=False, tight=True)
