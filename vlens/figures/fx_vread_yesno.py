# SPDX-License-Identifier: Apache-2.0
"""V-Read on the SimonSays yes/no questions: when the answer is decided, and when an edit decides it.

Both figures read ``results/readouts/<m>__<family>_readouts.json`` (log p(yes)
minus log p(no) at the answer position after each block, through the logit
lens at every block and through V-Read at the blocks it was fitted at) and
``results/simonsays/<m>__<family>.json`` (the question, the picture and the
reply). Records V-Read was fitted on are never drawn, WMDP and the
password-locking families never read.

The model's answer is the sign of the last block's logit-lens margin, which
is the model's own output (final norm, unembedding). A lens *commits* at the
first block from which its margin has that sign at every later block it is
read at. Both lenses are compared on the blocks V-Read is read at, so neither
gets a finer grid.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from vlens.figures import paper as P
from vlens.figures import style as st

MODELS = ("Qwen3-VL-8B-Instruct", "Qwen3.5-27B")
#: a model is drawn only when V-Read has been read on this share of its yes/no records
COVER = 0.9
MIN_N = 60


def _models(results):
    """Every model with SimonSays readouts on disk, the two named ones first."""
    found = {os.path.basename(p).split("__")[0]
             for p in glob.glob(os.path.join(results, "readouts", "*__*_readouts.json"))}
    return [m for m in MODELS if m in found] + sorted(found - set(MODELS))


# ------------------------------------------------------------------ data

def _json(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def _families(results, m):
    out = []
    for p in sorted(glob.glob(os.path.join(results, "readouts", f"{m}__*_readouts.json"))):
        fam = os.path.basename(p)[len(m) + 2:-len("_readouts.json")]
        if fam.startswith("wmdp") or P.LOCKED.search(fam):
            continue
        out.append(fam)
    return out


def _grid_of(r):
    return [i for i, x in enumerate(r.get("v_read") or []) if x is not None]


def _records(results, m):
    """Every yes/no record with a held-out V-Read readout, or None when V-Read is not read on enough of them."""
    recs, n_yesno = [], 0
    for fam in _families(results, m):
        d = _json(os.path.join(results, "readouts", f"{m}__{fam}_readouts.json"))
        if not d:
            continue
        beh = {b["id"]: b for b in (_json(os.path.join(results, "simonsays", f"{m}__{fam}.json")) or {})
               .get("rows", [])}
        L = len(d.get("blocks") or [])
        for r in d["rows"]:
            if r.get("kind") != "yesno":
                continue
            n_yesno += 1
            g = _grid_of(r)
            if len(g) < 4 or r.get("v_read_fitted_on"):
                continue
            recs.append(dict(fam=fam, id=r["id"], g=g, L=L,
                             v=np.array([r["v_read"][i] for i in g], float),
                             lg=np.array([r["logit"][i] for i in g], float),
                             fin=float(r["logit"][-1]), beh=beh.get(r["id"])))
    if not n_yesno or len(recs) < max(MIN_N, COVER * n_yesno):
        return None
    grids = {tuple(r["g"]) for r in recs}
    if len(grids) != 1:
        g0 = max(grids, key=lambda g: sum(tuple(r["g"]) == g for r in recs))
        recs = [r for r in recs if tuple(r["g"]) == g0]
    return recs


def _commit(x, s, tau=0.0):
    """The first index from which ``x`` has sign ``s`` (and size at least ``tau``) at every later index; len(x) if never."""
    ok = (np.sign(x) == s) & (np.abs(x) >= tau)
    for k in range(len(x)):
        if ok[k:].all():
            return k
    return len(x)


def _depth(recs):
    g = np.array(recs[0]["g"], float)
    return g / max(recs[0]["L"] - 1, 1)


def _map_numbers(recs):
    S = np.sign([r["fin"] for r in recs])
    keep = S != 0
    recs = [r for r, k in zip(recs, keep) if k]
    S = S[keep]
    cv = np.array([_commit(r["v"], s) for r, s in zip(recs, S)])
    cl = np.array([_commit(r["lg"], s) for r, s in zip(recs, S)])
    dep = _depth(recs)
    out = dict(n=len(recs), yes=int((S > 0).sum()), no=int((S < 0).sum()), grid=recs[0]["g"], L=recs[0]["L"],
               earlier=float((cv < cl).mean()), same=float((cv == cl).mean()), later=float((cv > cl).mean()),
               cv=cv, cl=cl, S=S, dep=dep, recs=recs,
               fams=sorted({r["fam"] for r in recs}),
               agree=float(np.mean([(r["fin"] > 0) == (P._said(r["beh"]) == "yes")
                                    for r in recs if r["beh"] and P._said(r["beh"])])))
    for lab, sel in (("yes", S > 0), ("no", S < 0)):
        out[f"med_v_{lab}"] = _blk(out, np.median(cv[sel]))
        out[f"med_l_{lab}"] = _blk(out, np.median(cl[sel]))
        out[f"earlier_{lab}"] = float((cv[sel] < cl[sel]).mean())
        out[f"later_{lab}"] = float((cv[sel] > cl[sel]).mean())
    # a sign test on the records where the two differ
    out["sign_p"] = _sign_p(int((cv < cl).sum()), int((cv != cl).sum()))
    # the answer each lens leans to at its first blocks, whatever the record
    first = max(1, len(dep) // 4)
    for key, name in (("v", "v"), ("lg", "l")):
        early = np.array([r[key][:first] for r in recs])
        out[f"lean_{name}"] = "yes" if np.median(early) > 0 else "no"
        out[f"lean_{name}_share"] = float((np.sign(early.mean(1)) == np.sign(np.median(early))).mean())
    out["first_blocks"] = recs[0]["g"][first - 1]
    # and through the middle half of the network
    mid = slice(len(dep) // 4, 3 * len(dep) // 4)
    for key, name in (("v", "v"), ("lg", "l")):
        m_ = np.array([r[key][mid] for r in recs])
        out[f"mid_{name}_yes"] = float((m_ > 0).mean())
    out["mid_blocks"] = (recs[0]["g"][mid.start], recs[0]["g"][mid.stop - 1])
    out["both"] = out["earlier_yes"] > out["later_yes"] and out["earlier_no"] > out["later_no"]
    # the same, asking for a margin of at least half a logit (or half the final one, if smaller)
    fin = np.abs([r["fin"] for r in recs])
    tau = np.minimum(0.5, fin / 2)
    cv5 = np.array([_commit(r["v"], s, t) for r, s, t in zip(recs, S, tau)])
    cl5 = np.array([_commit(r["lg"], s, t) for r, s, t in zip(recs, S, tau)])
    for lab, sel in (("yes", S > 0), ("no", S < 0)):
        out[f"earlier5_{lab}"] = float((cv5[sel] < cl5[sel]).mean())
        out[f"later5_{lab}"] = float((cv5[sel] > cl5[sel]).mean())
    out["never_v"], out["never_l"] = int((cv == len(dep)).sum()), int((cl == len(dep)).sum())
    return out


def _blk(v, k):
    """A (possibly half-integer) grid index as the block it names."""
    g = v["grid"]
    k = float(k)
    if k >= len(g):
        return None
    lo, hi = g[int(np.floor(k))], g[int(np.ceil(k))]
    return int(round((lo + hi) / 2))


# ------------------------------------------------------------------ figure 1: the map

def _edge_ticks(ax):
    """The end tick labels held inside the panel, so neighbouring panels' labels never meet."""
    labs = ax.get_xticklabels()
    labs[0].set_ha("left")
    labs[-1].set_ha("right")


def _pct(x):
    return f"{100 * x:.0f}%"


def _map_caption(V):
    ms = list(V)
    parts = []
    for m, v in V.items():
        parts.append(
            f"On {m} ({v['n']} held-out yes/no records, {v['yes']} answered yes and {v['no']} no, "
            f"read at {len(v['grid'])} of {v['L']} blocks), V-Read commits before the logit lens on "
            f"{_pct(v['earlier'])} of records, at the same block on {_pct(v['same'])} and later on "
            f"{_pct(v['later'])}. Over its first blocks (to block {v['first_blocks']}) V-Read leans "
            f"{v['lean_v']} on {_pct(v['lean_v_share'])} of records and the logit lens leans {v['lean_l']} "
            f"on {_pct(v['lean_l_share'])}. For yes answers V-Read commits first on "
            f"{_pct(v['earlier_yes'])} and later on {_pct(v['later_yes'])} (median block {v['med_v_yes']} "
            f"against {v['med_l_yes']}); for no answers, first on {_pct(v['earlier_no'])} and later on "
            f"{_pct(v['later_no'])} (median block {v['med_v_no']} against {v['med_l_no']}).")
    head = []
    for m in ms:
        v = V[m]
        lead_v = [c for c in ("yes", "no") if v[f"earlier_{c}"] > v[f"later_{c}"] and v[f"earlier5_{c}"] > v[f"later5_{c}"]]
        lead_l = [c for c in ("yes", "no") if v[f"later_{c}"] > v[f"earlier_{c}"] and v[f"later5_{c}"] > v[f"earlier5_{c}"]]
        bits = []
        for c in lead_v:
            bits.append(f"V-Read commits to the model's answer before the logit lens on {_pct(v[f'earlier_{c}'])} of "
                        f"the {c} answers and after it on {_pct(v[f'later_{c}'])} (median block "
                        f"{v[f'med_v_{c}']} against {v[f'med_l_{c}']})")
        for c in lead_l:
            bits.append(f"the logit lens commits first on {_pct(v[f'later_{c}'])} of the {c} answers "
                        f"(median block {v[f'med_l_{c}']} against {v[f'med_v_{c}']})")
        mixed = [c for c in ("yes", "no") if c not in lead_v + lead_l]
        if mixed:
            bits.append("on the " + " and ".join(mixed) + " answers neither lens leads once a margin of "
                        "half a logit is asked for")
        why = []
        if v["lean_v"] != v["lean_l"]:
            why.append(f"the two lenses lean opposite ways from the first block (V-Read to {v['lean_v']}, "
                       f"the logit lens to {v['lean_l']}), so each lead follows a lean")
        elif v["mid_l_yes"] > 0.6 and v["mid_v_yes"] < v["mid_l_yes"] - 0.15:
            why.append(f"the logit lens reads yes on {_pct(v['mid_l_yes'])} of records through the middle "
                       f"of the network (blocks {v['mid_blocks'][0]} to {v['mid_blocks'][1]}), V-Read on "
                       f"{_pct(v['mid_v_yes'])}")
        head.append(f"on {m}, " + "; ".join(bits) + ("; " + why[0] if why else ""))
    return (
        "MAIN. " + ". ".join(h[0].upper() + h[1:] for h in head) + ".\n\n"
        "Each row of a map is one SimonSays yes/no record (question and picture as the set words them, "
        "no system instruction), sorted by the model's final margin, yes at the top; each column is a "
        "block V-Read is read at, on normalised depth; the colour is log p(yes) minus log p(no) at the "
        "answer position, warm for yes and cool for no, clipped at the 98th percentile of both maps. Left, "
        "V-Read; middle, the logit lens on the same blocks. A lens commits at the first block from which "
        "its margin has the sign of the model's own answer at every later block. The solid ink line is the "
        "running median (over 31 neighbouring records) of V-Read's commitment depth, the dashed line the "
        "logit lens's. Right, every record placed by the two commitment depths, dot area proportional to "
        "the number of records; below the diagonal (clay) V-Read commits first, above it (slate) the "
        "logit lens does. "
        + " ".join(parts) + "\n\n"
        "Limits: commitment is read against zero, so a lens that leans to one answer at early blocks "
        "commits at once on every record ending with that answer; the two answers are therefore reported "
        "apart, and only a lead that holds on both is claimed. Asking for a margin of at least half a "
        "logit (or half the final margin, if smaller), V-Read commits first / later on yes answers on "
        + "; ".join(f"{m} {_pct(V[m]['earlier5_yes'])} / {_pct(V[m]['later5_yes'])}" for m in ms)
        + ", and on no answers on "
        + "; ".join(f"{m} {_pct(V[m]['earlier5_no'])} / {_pct(V[m]['later5_no'])}" for m in ms)
        + ". The answer is the sign of the model's last-block margin, "
        f"which matches the stored reply on {', '.join(_pct(V[m]['agree']) for m in ms)} of records whose "
        "reply begins with yes or no. V-Read is read at 16 blocks, so both lenses are compared on that "
        "grid; records V-Read was fitted on are excluded."
    )


def vread_confidence_map(results="results", out="figures", model=None):
    """MAIN. V-Read shows the yes/no answer taking its final sign earlier than the logit lens; replaced at render by the caption built from the files."""
    ms = [model] if model else _models(results)
    V = {}
    for m in ms:
        recs = _records(results, m)
        if recs:
            V[m] = _map_numbers(recs)
    if not V:
        return None
    cap = _map_caption(V)
    vread_confidence_map.__doc__ = cap
    print(cap)

    st.use()
    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.patheffects as pe
    W = st.MAX_WIDTH
    fs = matplotlib.rcParams["xtick.labelsize"] * 0.8
    nrow = len(V)
    row_h, key_h, bot = (2.9 if len(V) == 1 else 1.75), 0.62, 0.62
    # rows sit far enough apart for the joint panel's two-line axis name
    rgap = 0.98
    H = key_h + nrow * row_h + (nrow - 1) * rgap + bot
    fig = plt.figure(figsize=(W, H))
    fig.set_layout_engine("none")
    # the joint panel takes whatever the two maps leave, less the room its own
    # tick labels and axis name need, so nothing to its right is empty
    lab_w, map_w, gap, joint_gap, right = 0.95, 2.55, 0.22, 0.78, 0.08
    joint_w = W - (lab_w + 2 * map_w + gap) - joint_gap - right
    cmap = st.warm_cmap(True)
    for k, (m, v) in enumerate(V.items()):
        recs, S = v["recs"], v["S"]
        order = np.argsort([-r["fin"] for r in recs], kind="stable")
        dep = v["dep"]
        edges = np.concatenate([[0], (dep[1:] + dep[:-1]) / 2, [1]])
        Mv = np.array([recs[i]["v"] for i in order])
        Ml = np.array([recs[i]["lg"] for i in order])
        lim = float(np.quantile(np.abs(np.concatenate([Mv.ravel(), Ml.ravel()])), 0.98))
        norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
        n = len(recs)
        y_top = H - key_h - k * (row_h + rgap)
        ext = np.append(dep, 1.0)

        def run(c):
            d = ext[np.minimum(c[order], len(dep))]
            w = 31
            pad = np.pad(d, w // 2, mode="edge")
            return np.array([np.median(pad[i:i + w]) for i in range(n)])
        rv, rl = run(v["cv"]), run(v["cl"])
        n_yes = int((S > 0).sum())
        for j, (M, name) in enumerate(((Mv, "V-Read"), (Ml, "logit lens"))):
            ax = fig.add_axes([(lab_w + j * (map_w + gap)) / W, (y_top - row_h) / H, map_w / W, row_h / H])
            ax.pcolormesh(edges, np.arange(n + 1), M, cmap=cmap, norm=norm, shading="flat",
                          rasterized=True)
            ax.set_ylim(n, 0)
            ax.set_xlim(0, 1)
            ax.axhline(n_yes, color=st.INK, lw=1.0)
            yy = np.arange(n) + 0.5
            for r_, dash in ((rv, ()), (rl, (4, 2))):
                ln, = ax.plot(r_, yy, color=st.INK, lw=1.8, dashes=dash, zorder=5)
                ln.set_path_effects([pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()])
            ax.grid(False)
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(["input", "half", "output"], fontsize=fs)
            _edge_ticks(ax)
            ax.set_xlabel(f"{name}: depth", fontsize=fs, labelpad=2)
            if j == 0:
                ax.set_yticks([n_yes / 2, n_yes + (n - n_yes) / 2])
                ax.set_yticklabels(["yes", "no"], fontsize=fs)
                ax.tick_params(axis="y", length=0, pad=3)
                ax.set_ylabel(m, fontsize=fs, labelpad=4)
            else:
                ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(st.AXIS)
                sp.set_linewidth(0.8)
        # the joint panel: V-Read's commitment against the logit lens's
        aj = fig.add_axes([(W - right - joint_w) / W, (y_top - row_h) / H, joint_w / W, row_h / H])
        cnt = {}
        for a, b in zip(v["cv"], v["cl"]):
            cnt[(a, b)] = cnt.get((a, b), 0) + 1
        big = max(cnt.values())
        for (a, b), c in cnt.items():
            col = P.COLOUR["v_read"] if a < b else (st.MUTED if a == b else st.SLATE)
            aj.scatter([ext[b]], [ext[a]], s=520 * c / big + 6, color=col, alpha=0.85, lw=0.6,
                       ec="white", zorder=3)
        aj.plot([-0.05, 1.05], [-0.05, 1.05], color=st.SUBINK, lw=0.9, dashes=(3, 2), zorder=1)
        aj.set_xlim(-0.06, 1.06)
        aj.set_ylim(-0.06, 1.06)
        aj.set_xticks([0, 0.5, 1])
        aj.set_xticklabels(["input", "half", "output"], fontsize=fs)
        aj.set_yticks([0, 0.5, 1])
        aj.set_yticklabels(["in", "half", "out"], fontsize=fs)
        # the tally is written under the axis name, never among the dots
        aj.set_xlabel("logit lens commits\n"
                      f"V-Read first {_pct(v['earlier'])}, same block {_pct(v['same'])}, "
                      f"logit lens first {_pct(v['later'])}",
                      fontsize=fs, labelpad=2, linespacing=1.25)
        aj.set_ylabel("V-Read commits", fontsize=fs, labelpad=2)
        st.despine(aj)
    # the key, in its own band of two rows so it is no wider than the panels:
    # the colour scale, named at its two ends, then the lines and the dots
    kb = H - 0.13
    x0, bar_w = lab_w + 0.3, 1.5
    cax = fig.add_axes([x0 / W, (kb - 0.07) / H, bar_w / W, 0.14 / H])
    matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1),
                                     orientation="horizontal").outline.set_linewidth(0.6)
    cax.set_xticks([])
    cax.set_yticks([])
    fig.text((x0 - 0.06) / W, kb / H, "no", ha="right", va="center", fontsize=fs, color=st.INK)
    fig.text((x0 + bar_w + 0.06) / W, kb / H, "yes", ha="left", va="center", fontsize=fs, color=st.INK)
    fig.text((x0 + bar_w + 0.42) / W, kb / H, "colour: log p(yes) minus log p(no)", ha="left",
             va="center", fontsize=fs, color=st.INK)
    hs = [Line2D([], [], color=st.INK, lw=1.8), Line2D([], [], color=st.INK, lw=1.8, dashes=(4, 2)),
          Line2D([], [], marker="o", ls="none", color=P.COLOUR["v_read"], ms=8),
          Line2D([], [], marker="o", ls="none", color=st.SLATE, ms=8),
          Line2D([], [], marker="o", ls="none", color=st.MUTED, ms=8)]
    fig.legend(hs, ["V-Read commits", "logit lens commits", "V-Read first", "logit lens first", "same block"],
               loc="center left", bbox_to_anchor=(lab_w / W, (kb - 0.29) / H), ncol=5,
               fontsize=fs, frameon=False, handlelength=1.6, columnspacing=1.4, handletextpad=0.4,
               borderaxespad=0, borderpad=0)
    return st.save(fig, P._at(out, "vread_confidence_map"), check=True, tight=True)


# ------------------------------------------------------------------ figure 2: edit pairs

def _pairs(results, m):
    """Untouched and edited pictures under the same question, both read by held-out V-Read."""
    def load(fam):
        d = _json(os.path.join(results, "readouts", f"{m}__{fam}_readouts.json"))
        b = _json(os.path.join(results, "simonsays", f"{m}__{fam}.json"))
        if not d or not b:
            return None
        return {r["id"]: r for r in d["rows"]}, {r["id"]: r for r in b["rows"]}, len(d.get("blocks") or [])

    def key(b, edited):
        mt = b.get("meta") or {}
        if b.get("dataset") == "gemini":
            return (mt.get("scene_id"), mt.get("question_id"), mt.get("eval_framing"), b["question"])
        img = str(mt.get("source_image")) if edited else str(b["image"])
        return (os.path.basename(img), b["question"])
    out = []
    for fo, fe in (("openimages_orig_auth", "openimages_edit_auth"), ("vins_orig_auth", "vins_edit_auth"),
                   ("nuscenes_orig_auth", "nuscenes_edit_auth"), ("gemini_orig_eval", "gemini_edit_eval"),
                   ("gemini_orig_auth", "gemini_edit_swap")):
        a, e = load(fo), load(fe)
        if not a or not e:
            continue
        idx = {}
        for i, r in a[0].items():
            if r.get("kind") == "yesno" and i in a[1]:
                idx.setdefault(key(a[1][i], False), []).append(i)
        for i, r in e[0].items():
            if r.get("kind") != "yesno" or i not in e[1]:
                continue
            for j in idx.get(key(e[1][i], True), []):
                ro = a[0][j]
                if ro.get("v_read_fitted_on") or r.get("v_read_fitted_on"):
                    continue
                g = [t for t in _grid_of(ro) if t in set(_grid_of(r))]
                if len(g) < 4:
                    continue
                out.append(dict(src=fo.split("_")[0], o=j, e=i, g=g, L=a[2], bo=a[1][j], be=e[1][i],
                                vo=np.array([ro["v_read"][t] for t in g]), ve=np.array([r["v_read"][t] for t in g]),
                                lo=np.array([ro["logit"][t] for t in g]), le=np.array([r["logit"][t] for t in g]),
                                fo=float(ro["logit"][-1]), fe=float(r["logit"][-1])))
    for p in out:
        s = np.sign(p["fe"] - p["fo"])
        p["s"] = s
        tau = min(0.5, abs(p["fe"] - p["fo"]) / 2)
        p["sep_v"] = _commit(p["ve"] - p["vo"], s, tau)
        p["sep_l"] = _commit(p["le"] - p["lo"], s, tau)
        so, se = P._said(p["bo"]), P._said(p["be"])
        p["flip"] = bool(so and se and so != se)
        p["said"] = (so, se)
    return [p for p in out if p["s"] != 0]


def _sign_p(k, n):
    from math import comb
    if not n:
        return float("nan")
    t = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(t + 1)) / 2 ** n)


def _edit_words(b):
    from vlens.figures import fx_layers as FL
    v = (b.get("meta") or {}).get("edit_variant")
    if v:
        return FL.EDIT_WORDS.get(v, str(v).replace("_", " "))
    return "the scene edited"


def _flip_caption(m, ps, shown):
    fl = [p for p in ps if p["flip"]]
    sv = np.array([p["sep_v"] for p in ps])
    sl = np.array([p["sep_l"] for p in ps])
    g = ps[0]["g"]
    med = lambda x: g[int(np.median(x))] if np.median(x) < len(g) else None  # noqa: E731
    fv = np.array([p["sep_v"] for p in fl])
    flv = np.array([p["sep_l"] for p in fl])
    ex = "; ".join(f"({chr(97 + k)}) {_edit_words(p['be'])}, {p['said'][0]} to {p['said'][1]}"
                   for k, p in enumerate(shown))
    return (
        "APPENDIX. When an edit to the picture changes a yes/no answer, V-Read "
        + ("sees the two pictures part earlier than the logit lens more often than later"
           if (sv < sl).mean() > 1.5 * (sv > sl).mean()
           else "sees the two pictures part later than the logit lens more often than earlier"
           if (sv > sl).mean() > 1.5 * (sv < sl).mean()
           else "sees the two pictures part about as early as the logit lens")
        + (", though not significantly" if not _sign_p(int((sv < sl).sum()), int((sv != sl).sum())) < 0.05
           and (sv < sl).mean() != (sv > sl).mean() and max((sv < sl).mean(), (sv > sl).mean())
           > 1.5 * min((sv < sl).mean(), (sv > sl).mean()) else "")
        + f": over {len(ps)} matched pairs on {m}, V-Read separates them "
        f"first on {_pct((sv < sl).mean())}, at the same block on {_pct((sv == sl).mean())} and later on "
        f"{_pct((sv > sl).mean())} (two-sided sign test p = {_sign_p(int((sv < sl).sum()), int((sv != sl).sum())):.2g}), "
        f"median block {med(sv)} against {med(sl)}.\n\n"
        "A pair is an untouched SimonSays picture and an edit of it asked the identical question "
        "(joined on the source picture, or for the generated scenes on scene, question and framing), with "
        "a held-out V-Read readout on both. The pair separates at the first block from which the edited "
        "picture's yes-minus-no margin minus the untouched one's has the sign it has at the model's output, "
        "at least half a logit in size (or half the output gap, if smaller), and keeps both; both lenses are compared on the blocks V-Read is read at. Left: three of the "
        f"{len(fl)} pairs whose reply flips, the one with the largest output gap from each source ({ex}); "
        "thumbnails of the untouched (left) and edited (right) picture with the model's reply boxed, and under them V-Read's margin on the "
        "untouched (dashed) and edited (solid) picture with the ribbon between them, the logit lens in "
        "grey, and a triangle at each lens's separation block. Right: every pair's separation depth "
        "under both lenses, one row each, sorted; flipped pairs ringed. On the flipped pairs alone, "
        f"V-Read separates first on {int((fv < flv).sum())} of {len(fl)} and later on "
        f"{int((fv > flv).sum())}.\n\n"
        f"Limits: {len(ps)} pairs, {len(fl)} of them flips, from two to three sources, so the comparison "
        "has little power; the output gap of many unflipped pairs is under one logit, where a sign is "
        "fragile; the pairs come from the few edits the set asks the identical question of."
    )


def vread_flip_moments(results="results", out="figures", model=None):
    """APPENDIX. Edit pairs that flip a yes/no answer, as V-Read and the logit lens read them; replaced at render by the caption built from the files."""
    from vlens import corpus as C
    from vlens.data import simonsays as SS
    from vlens.figures import fx_layers as FL
    ms = [model] if model else [m for m in reversed(MODELS)]
    m, ps = None, []
    for mm in ms:
        if _records(results, mm) is None:
            continue
        q = _pairs(results, mm)
        if len(q) >= 12 and sum(p["flip"] for p in q) >= 3:
            m, ps = mm, q
            break
    if not ps:
        return None
    fl = [p for p in ps if p["flip"]]
    shown, used = [], set()
    for p in sorted(fl, key=lambda p: -abs(p["fe"] - p["fo"])):
        if p["src"] not in used:
            shown.append(p)
            used.add(p["src"])
    for p in sorted(fl, key=lambda p: -abs(p["fe"] - p["fo"])):
        if len(shown) >= 3:
            break
        if p not in shown:
            shown.append(p)
    shown = shown[:3]
    cap = _flip_caption(m, ps, shown)
    vread_flip_moments.__doc__ = cap
    print(cap)

    st.use()
    W = st.MAX_WIDTH
    fs = matplotlib.rcParams["xtick.labelsize"] * 0.74
    lh = fs * 1.3 / 72
    colw, gut, right_w = 2.3, 0.22, 2.55
    x_r = 0.5 + 3 * colw + 2 * gut + 0.55
    W = min(W, x_r + right_w)
    thumb = (colw - 0.06) / 2
    box_lines = 1
    trace_h = 1.7
    H = 0.05 + thumb + 0.06 + (lh * 0.9 * box_lines + 2 * FL.BOX_PAD) + 0.1 + trace_h + 0.5
    fig = plt.figure(figsize=(W, H))
    fig.set_layout_engine("none")
    lim = max(np.abs(np.concatenate([np.r_[p["vo"], p["ve"], p["lo"], p["le"]] for p in shown])).max(), 1)
    for k, p in enumerate(shown):
        x = 0.5 + k * (colw + gut)
        for j, (b, said) in enumerate(((p["bo"], p["said"][0]), (p["be"], p["said"][1]))):
            ai = fig.add_axes([(x + j * (thumb + 0.06)) / W, (H - 0.05 - thumb) / H, thumb / W, thumb / H])
            im = C.load_image(os.path.join(SS.ROOT, b["image"]))
            ai.imshow(im, aspect="auto")
            ai.set_xticks([]); ai.set_yticks([])
            ai.grid(False)
            for sp in ai.spines.values():
                sp.set_color(st.AXIS); sp.set_linewidth(0.7)
            word = f"reply: {said}"
            FL.panel_box(fig, (x + j * (thumb + 0.06)) / W, (H - 0.05 - thumb - 0.06) / H, thumb,
                         1, lh * 0.9, fs, [word], "model")
        top = H - 0.05 - thumb - 0.06 - (lh * 0.9 + 2 * FL.BOX_PAD) - 0.12
        at = fig.add_axes([x / W, (top - trace_h) / H, colw / W, trace_h / H])
        d = np.array(p["g"], float) / max(p["L"] - 1, 1)
        c = P.COLOUR["v_read"]
        at.fill_between(d, p["vo"], p["ve"], color=c, alpha=0.2, lw=0, zorder=2)
        at.plot(d, p["vo"], color=c, lw=2.0, dashes=(3, 1.6), zorder=4)
        at.plot(d, p["ve"], color=c, lw=2.2, zorder=4)
        at.plot(d, p["lo"], color=st.MUTED, lw=1.3, dashes=(3, 1.6), zorder=3)
        at.plot(d, p["le"], color=st.MUTED, lw=1.4, zorder=3)
        at.axhline(0, color=st.AXIS, lw=0.8, zorder=1)
        at.set_xlim(0, 1)
        at.set_ylim(-lim * 1.08, lim * 1.08)
        ext = np.append(d, 1.0)
        for sep, col, yv in ((p["sep_v"], c, -lim * 1.08), (p["sep_l"], st.MUTED, -lim * 1.08)):
            at.plot([ext[sep]], [yv], marker="^", ms=10 if col == c else 8, color=col, mec="white",
                    mew=0.8, clip_on=False, zorder=7 if col == c else 6)
        at.set_xticks([0, 0.5, 1])
        at.set_xticklabels(["input", "half", "output"], fontsize=fs)
        _edge_ticks(at)
        at.tick_params(labelsize=fs, length=3, pad=4)
        if k == 0:
            at.set_ylabel("yes minus no", fontsize=fs, labelpad=2)
        else:
            at.set_yticklabels([])
        at.text(0.02, 0.97, f"({chr(97 + k)})", transform=at.transAxes, ha="left", va="top", fontsize=fs,
                color=st.INK)
        st.despine(at)
    # right: every pair's separation depth under both lenses
    ar = fig.add_axes([x_r / W, (0.5) / H, (right_w - 0.05) / W, (H - 0.55) / H])
    order = sorted(ps, key=lambda p: (p["sep_l"], p["sep_v"]))
    ext = np.append(np.array(ps[0]["g"], float) / max(ps[0]["L"] - 1, 1), 1.0)
    for i, p in enumerate(order):
        a, b = ext[p["sep_v"]], ext[p["sep_l"]]
        seg = P.COLOUR["v_read"] if a < b else st.MUTED
        ar.plot([a, b], [i, i], color=seg, lw=1.4, alpha=0.8, zorder=2)
        ar.plot([b], [i], "o", ms=5, color=st.MUTED, mec="white", mew=0.5, zorder=3)
        ar.plot([a], [i], "o", ms=5.5, color=P.COLOUR["v_read"], mec="white", mew=0.5, zorder=4)
        if p["flip"]:
            ar.plot([a], [i], "o", ms=10, mfc="none", mec=st.INK, mew=1.0, zorder=5)
    ar.set_ylim(-1, len(order))
    ar.set_yticks([])
    ar.set_xlim(-0.03, 1.03)
    ar.set_xticks([0, 0.5, 1])
    ar.set_xticklabels(["input", "half", "output"], fontsize=fs)
    ar.set_xlabel("pair separates at depth", fontsize=fs, labelpad=2)
    ar.grid(False)
    ar.xaxis.grid(True)
    st.despine(ar, left=False)
    hs = [Line2D([], [], color=P.COLOUR["v_read"], lw=2.2),
          Line2D([], [], color=P.COLOUR["v_read"], lw=2.0, dashes=(3, 1.6)),
          Line2D([], [], color=st.MUTED, lw=1.4),
          Line2D([], [], marker="^", ls="none", color=P.COLOUR["v_read"], ms=9),
          Line2D([], [], marker="o", ls="none", mfc="none", mec=st.INK, ms=9)]
    st.key_band(fig, hs, ["V-Read, edited", "V-Read, untouched", "logit lens", "separation block",
                          "reply flips"], fontsize=fs, ncol=5, columnspacing=1.2, handlelength=1.6)
    return st.save(fig, P._at(out, "vread_flip_moments"), check=True, tight=True)
