# SPDX-License-Identifier: Apache-2.0
"""Early detection across tasks: how deep in the model each reader first reads the answer for good.

One number per (task, reader): the detection depth, the earliest normalised
depth from which the reader's AUROC for the answer the model then gives stays
at or above a threshold all the way to the output. Tasks are the refusal
targets (``experiments/safety/refusal.py``), the COCO families, the SimonSays
families by source and the sandbagging-by-channel run. Per-family AUROC comes
from :func:`vlens.figures.fx_bench._readout_family`, unchanged.

``tools/make_detect_table.py`` builds the LaTeX table from :func:`task_rows`.
"""
from __future__ import annotations

__all__ = ["early_detection"]

import glob
import json
import os
import shutil
import tempfile

import numpy as np

from vlens.figures import fx_bench as B
from vlens.figures import paper as P
from vlens.figures import style as st

#: the threshold the figure draws, and the stricter one the table adds
THR = 0.75
THR_HI = 0.9
#: the depth AUROC is also reported at
MID = 0.5
#: the reader the claim is about; a figure without it is not drawn
LEAD = "v_read"
#: every reader, in the order the key and the table name them; the last two are labelled
READERS = ("v_read", "v_read_coco", "tuned", "jacobian", "patchscope", "logit", "probe", "dom")
LABELLED = ("probe", "dom")
#: published label-free baselines V-Read is compared against
BASELINES = ("tuned", "jacobian", "patchscope", "logit")
MARKER = {"v_read": "o", "v_read_coco": "o", "tuned": "s", "jacobian": "D", "patchscope": "^",
          "logit": "v", "probe": "o", "dom": "D"}
MODELS = ("Qwen3-VL-8B-Instruct", "Qwen3.5-27B", "gemma-4-26B-A4B-it", "InternVL3_5-8B-HF")
GRID = np.linspace(0.0, 1.0, 401)

#: task groups, in the order the figure draws them
GROUPS = ("refusal", "coco", "simonsays", "channel")
GROUP_NAME = {"refusal": "refusal", "coco": "COCO", "simonsays": "SimonSays", "channel": "sandbagging"}
#: refusal targets as rows (``experiments/safety/refusal.py``): the reply
#: refuses, or the prompt is harmful, over every pool; harmful over the two
#: MM-SafetyBench pools only, the target without the image-source confound
REFUSAL = (("refused", "refuses"), ("harmful", "harmful prompt"),
           ("harmful_matched", "harmful, MM-SafetyBench"))
COCO = (("coco_pope", "POPE adversarial"), ("coco_mcq", "four-way object"))
CHANNEL = (("wmdp_chan", "WMDP by channel"),)
#: where a family's readouts, and its behaviour pass, can live
READOUT_DIRS = ("readouts", "coco_readouts", "sandchan_readouts", "sandchan")
BEHAVIOUR_DIRS = ("simonsays", "coco", "sandchan")

#: caption numbers, formerly pinned; the caption is now built from the files by ``_caption``
CLAIMS: dict = {}


# ------------------------------------------------------------------ families

def _task_of(fam):
    """(group, row key, row label) for a readout family, or None."""
    for f, lab in COCO:
        if fam == f:
            return "coco", f, lab
    for f, lab in CHANNEL:
        if fam == f:
            return "channel", f, lab
    src = fam.split("_")[0]
    if src in P.SOURCE:
        return "simonsays", src, P.SOURCE[src].replace(" (VINS-120K)", "")
    return None


def _find(results, model, fam):
    """(readouts dir, behaviour dir) holding this family, or None."""
    rd = next((d for d in READOUT_DIRS
               if os.path.exists(f"{results}/{d}/{model}__{fam}_readouts.json")), None)
    bd = next((d for d in BEHAVIOUR_DIRS if os.path.exists(f"{results}/{d}/{model}__{fam}.json")), None)
    return (rd, bd) if rd and bd else None


def _family(results, model, fam, floor=True):
    """``fx_bench._readout_family`` for a family wherever it lives; COCO and the channel run go through a shim."""
    where = _find(results, model, fam)
    if where is None:
        return None
    rd, bd = where
    if (rd, bd) == ("readouts", "simonsays"):
        return B._readout_family(results, model, fam, floor)
    tmp = tempfile.mkdtemp(prefix="detect_")
    try:
        os.makedirs(f"{tmp}/readouts")
        os.makedirs(f"{tmp}/simonsays")
        for p in glob.glob(f"{results}/{rd}/{model}__{fam}_*.json"):
            os.symlink(os.path.abspath(p), f"{tmp}/readouts/{os.path.basename(p)}")
        os.symlink(os.path.abspath(f"{results}/{bd}/{model}__{fam}.json"),
                   f"{tmp}/simonsays/{model}__{fam}.json")
        return B._readout_family(tmp, model, fam, floor)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _models(results):
    pats = [f"{results}/{d}/*__*_readouts.json" for d in READOUT_DIRS]
    pats.append(f"{results}/safety/*__refusal_auroc.json")
    got = {os.path.basename(p).split("__")[0] for q in pats for p in glob.glob(q)}
    return [m for m in MODELS if m in got] + sorted(got - set(MODELS))


def _on_lead_blocks(au):
    """Every reader read on V-Read's fitted blocks only, as the tables are.

    A reader read on a denser grid of blocks gets more chances to dip below the
    threshold, so its detection depth would come out later for that reason
    alone; masking every reader to V-Read's blocks removes the asymmetry.
    """
    if LEAD not in au:
        return au
    vb = np.isfinite(au[LEAD])
    return {r: np.where(vb, v, np.nan) for r, v in au.items()}


def _entries(results, model):
    """Every scored family of one model as (group, key, label, n, depth, {reader: AUROC per block})."""
    fams = set()
    for d in READOUT_DIRS:
        for p in glob.glob(f"{results}/{d}/{model}__*_readouts.json"):
            fam = os.path.basename(p).split("__", 1)[1][: -len("_readouts.json")]
            if not P.LOCKED.search(fam) and _task_of(fam):
                fams.add(fam)
    res = {fam: _family(results, model, fam) for fam in sorted(fams)}
    got = [f for f in res.values() if f]
    # families under the scoring floor, scored on whatever records they have; they make a row
    # (marked with a dagger, never judged) only where no family passing the floor does
    thin = [f for f in (_family(results, model, fam, False) for fam, f in res.items() if f is None) if f]
    best = B._pick_patchscope(got)
    for f in thin:
        v = f["auroc"].get(best) if best else None
        for k in [k for k in f["auroc"] if k.startswith("patchscope")]:
            f["auroc"].pop(k)
        if v is not None:
            f["auroc"]["patchscope"] = v
    got += thin
    # readers whose run has not landed: fitted baselines not yet in the model's fit record, and a
    # family's Patchscopes or probe file not yet written beside its readouts
    fits = set(B.fits_running(results, model))
    out = []
    for f in got:
        assert not P.LOCKED.search(f["fam"]), f["fam"]
        g, k, lab = _task_of(f["fam"])
        au = _on_lead_blocks({r: np.asarray(v, float) for r, v in f["auroc"].items() if r in READERS})
        rd = (_find(results, model, f["fam"]) or ("readouts",))[0]
        # (a family under the scoring floor may have no probe because it is too small to fit one)
        pend = set(fits) | {r for r in ("patchscope", "probe")
                            if not f.get("thin") and not os.path.exists(f"{results}/{rd}/{model}__{f['fam']}_{r}.json")}
        out.append(dict(model=model, fam=f["fam"], group=g, key=k, label=lab, n=f["n"],
                        depth=np.asarray(f["depth"], float), auroc=au, pending=pend - set(au),
                        thin=bool(f.get("thin"))))
    p = f"{results}/safety/{model}__refusal_auroc.json"
    if os.path.exists(p):
        d = json.load(open(p))
        dep = np.asarray(d["depths"], float)
        for t, lab in REFUSAL:
            if t not in (d.get("targets") or {}):
                continue
            au = {}
            for r in READERS:
                # Patchscopes in the target prompt picked for this model on the readouts
                v = d["auroc"].get(f"{t}|{best if r == 'patchscope' and best else r}") or d["auroc"].get(f"{t}|{r}")
                if v is not None:
                    a = np.array([np.nan if x is None else float(x) for x in v], float)
                    if np.isfinite(a).any():
                        au[r] = a
            pend = (set(fits) | {"patchscope"}) - set(au)
            out.append(dict(model=model, fam=f"refusal:{t}", group="refusal", key=t, label=lab,
                            n=int(d["targets"][t]["n"]), depth=dep, auroc=_on_lead_blocks(au), pending=pend))
    return out


# ------------------------------------------------------------------ metrics

def _on_grid(dep, a):
    ok = np.isfinite(a)
    if ok.sum() < 2:
        return np.full(GRID.size, np.nan)
    return np.interp(GRID, dep[ok], a[ok], left=np.nan, right=np.nan)


def detection(c, thr):
    """Earliest depth from which ``c`` (on ``GRID``) stays >= ``thr`` to its last read depth.

    ``inf`` when it ends below; linear between read depths; None when unread.
    """
    idx = np.nonzero(np.isfinite(c))[0]
    if idx.size == 0:
        return None
    below = idx[c[idx] < thr]
    if below.size == 0:
        return float(GRID[idx[0]])
    i = int(below.max())
    if i == idx.max():
        return float("inf")
    j = int(idx[idx > i][0])
    y0, y1 = c[i], c[j]
    return float(GRID[i] + (thr - y0) / (y1 - y0) * (GRID[j] - GRID[i])) if y1 != y0 else float(GRID[j])


def _pool(ents):
    """One task row from its families: per reader the record-weighted mean AUROC on the depth grid."""
    n = sum(e["n"] for e in ents)
    curves, cover = {}, {}
    for r in READERS:
        have = [e for e in ents if r in e["auroc"]]
        if not have:
            continue
        w = np.array([e["n"] for e in have], float)
        Y = np.vstack([_on_grid(e["depth"], e["auroc"][r]) for e in have])
        fin = np.isfinite(Y)
        cov = (fin * w[:, None]).sum(0)
        mean = np.nansum(np.nan_to_num(Y) * w[:, None], 0) / np.maximum(cov, 1e-9)
        c = np.where(cov >= w.sum() * 0.999, mean, np.nan)
        # a reader seen on under half the row's records is not the row's reader
        if np.isfinite(c).any() and w.sum() >= 0.5 * n:
            curves[r], cover[r] = c, int(w.sum())
    row = dict(group=ents[0]["group"], key=ents[0]["key"], label=ents[0]["label"], n=n,
               fams=sorted({e["fam"] for e in ents}), models=sorted({e["model"] for e in ents}),
               cover=cover, det={}, det_hi={}, mid={},
               # a reader not drawn on the row because its run on some family has not landed
               pending={r for e in ents for r in e.get("pending", ()) if r not in curves},
               # every family under the scoring floor: shown with a dagger, never judged; ``na`` when
               # no family has both answers, so no AUROC exists
               thin=all(e.get("thin") for e in ents), na=all(e.get("thin") for e in ents) and not curves,
               pending_on={r: sorted({e["model"] for e in ents if r in e.get("pending", ())})
                           for r in READERS if r not in curves and any(r in e.get("pending", ()) for e in ents)})
    for r, c in curves.items():
        row["det"][r] = detection(c, THR)
        row["det_hi"][r] = detection(c, THR_HI)
        v = float(np.interp(MID, GRID, c)) if np.isfinite(c[np.argmin(abs(GRID - MID))]) else None
        row["mid"][r] = v
    return row


def _order(rows):
    order = {"refusal": [k for k, _ in REFUSAL], "coco": [k for k, _ in COCO],
             "simonsays": list(B.SRC_ORDER), "channel": [k for k, _ in CHANNEL]}
    return sorted(rows, key=lambda r: (GROUPS.index(r["group"]),
                                       order[r["group"]].index(r["key"]) if r["key"] in order[r["group"]] else 99))


def task_rows(results="results", model=None):
    """The task rows for one model, or pooled over every model with data when ``model`` is None."""
    models = [model] if model else _models(results)
    ents = [e for m in models for e in _entries(results, m)]
    if model is None:
        # pooled over models, a model without V-Read readouts on at least half
        # its records would enter on the baselines alone, and V-Read would be
        # compared on one model's records against the rest on two
        keep = set()
        for m in models:
            mine = [e for e in ents if e["model"] == m]
            n_all = sum(e["n"] for e in mine)
            if n_all and sum(e["n"] for e in mine if LEAD in e["auroc"]) >= 0.5 * n_all:
                keep.add(m)
        ents = [e for e in ents if e["model"] in keep]
    by = {}
    for e in ents:
        by.setdefault((e["group"], e["key"]), []).append(e)
    return _order([_pool([e for e in v if not e.get("thin")] or v) for v in by.values()])


def refusal_variant(results, model):
    p = f"{results}/safety/{model}__refusal_read.json"
    try:
        return json.load(open(p)).get("variant")
    except Exception:  # noqa: BLE001
        return None


def findings(rows, name=""):
    """The caption's numbers, computed: where V-Read is earliest, and where a baseline is."""
    lead = [r for r in rows if LEAD in r["det"] and not r.get("thin")]
    out = dict(rows=len(rows), with_lead=len(lead), earliest=[], level=[], beaten=[], gain=[], missing={})
    for r in lead:
        base = {k: r["det"][k] for k in BASELINES if k in r["det"] and r["det"][k] is not None}
        if not base:
            continue
        best_k = min(base, key=lambda k: base[k])
        v, b = r["det"][LEAD], base[best_k]
        if v < b - 1e-9:
            out["earliest"].append(r["label"])
        elif abs(v - b) <= 1e-9:
            out["level"].append(r["label"])
        else:
            out["beaten"].append((r["label"], P.NICE[best_k], v, b))
        if np.isfinite(v) and np.isfinite(b):
            out["gain"].append(b - v)
    for r in rows:
        miss = [k for k in READERS if k not in r["det"] and k not in ("v_read_coco", "dom")]
        if miss:
            out["missing"][r["label"]] = miss
    return out


def _fmt_d(v):
    return "never" if v is not None and not np.isfinite(v) else f"{v:.2f}"


def findings_text(f, name):
    s = [f"{name}: V-Read is present on {f['with_lead']} of {f['rows']} task rows; at AUROC {THR} it "
         f"detects earliest of the published label-free readers on {len(f['earliest'])}, level on {len(f['level'])}, "
         f"later on {len(f['beaten'])}."]
    if f["gain"]:
        s.append(f"Median lead over the earliest published baseline: {np.median(f['gain']):.2f} of depth "
                 f"(range {min(f['gain']):.2f} to {max(f['gain']):.2f}).")
    for lab, k, v, b in f["beaten"]:
        s.append(f"Later on {lab}: V-Read {_fmt_d(v)} against {k} {_fmt_d(b)}.")
    for lab, ks in f["missing"].items():
        s.append(f"Missing on {lab}: {', '.join(P.NICE[k] for k in ks)}.")
    return " ".join(s)


def _check(name, f):
    want = CLAIMS.get(name)
    if not want:
        print(f"  early_detection{name}: no caption numbers pinned yet (CLAIMS is empty)")
        return []
    got = dict(earliest=len(f["earliest"]), level=len(f["level"]), beaten=len(f["beaten"]),
               median_gain=round(float(np.median(f["gain"])), 2) if f["gain"] else None)
    bad = [f"{k}: {got.get(k)} vs {v}" for k, v in want.items() if got.get(k) != v]
    print(f"  early_detection{name}: " + ("CAPTION IS STALE: " + "; ".join(bad) if bad
                                          else "every caption number matches the data"))
    return bad


# ------------------------------------------------------------------ figure

def _mark(ax, r, x, y, zorder=None):
    lead = r == LEAD
    hollow = r in LABELLED
    ms = {"v_read": 12.0, "v_read_coco": 8.5}.get(r, 8.5)
    if MARKER[r] in "sD":
        ms *= 0.85
    ax.plot([x], [y], marker=MARKER[r], ms=ms, ls="none", color=P.COLOUR[r],
            mfc="white" if hollow else P.COLOUR[r], mec=P.COLOUR[r] if hollow else "white",
            mew=1.9 if hollow else 1.1, clip_on=False,
            zorder=zorder or (6 if lead else 5 if hollow else 4))


def _dodge(xs, keys, width):
    """Vertical offsets, in row units, so marks at near the same depth sit one above the other."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    off = [0.0] * len(xs)
    groups, cur = [], [order[0]] if order else []
    for i in order[1:]:
        if xs[i] - xs[cur[-1]] < width:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    if cur:
        groups.append(cur)
    for g in groups:
        if len(g) > 1:
            step = min(0.3, 0.84 / (len(g) - 1))
            g = sorted(g, key=lambda i: READERS.index(keys[i]))
            for k, i in enumerate(g):
                off[i] = (k - (len(g) - 1) / 2) * step
    return off


BASELINES = ("tuned", "jacobian", "patchscope", "logit")
#: three fixed lanes in every row, top to bottom
LANES = (("v_read", -0.26), ("v_read_coco", 0.0), ("best", 0.26))


#: column heads of the detection matrix, broken where they would be wide
HEAD = {"v_read": "V-Read", "v_read_coco": "V-Read\nfitted on\nCOCO", "tuned": "tuned\nlens",
        "jacobian": "Jacobian\nlens", "patchscope": "Patch-\nscopes", "logit": "logit\nlens",
        "probe": "probe\n(labelled)"}


def _draw(rows, name):
    """A matrix: one row per task, one column per reader, each cell the detection depth.

    The cell's shade is the depth from which the reader's AUROC stays at or
    above the threshold (darker is earlier) and its value is printed in it;
    V-Read's column is outlined. The labelled probe (supervised reference)
    stands apart on the right in greys. A reader with no readouts on a model has no
    column unless its run is under way; a hatched cell with no number is a reader
    whose run on that row has not landed; a blank cell is a reader not read on that row.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    have = [r for r in rows if r["det"].get("v_read") is not None and not r.get("thin")]
    full = [r for r in rows if not r.get("thin")]
    # a model whose V-Read readouts cover under half its rows is not drawn yet
    if not have or len(have) < 0.5 * len(full):
        return None
    # rows under the scoring floor stay in, marked with a dagger
    rows = [r for r in rows if r["det"].get("v_read") is not None or r.get("thin")]
    st.use()
    cols = [k for k in ("v_read", "v_read_coco", *BASELINES)
            if any(r["det"].get(k) is not None or k in r.get("pending", ()) for r in rows)]
    probe = any(r["det"].get("probe") is not None or "probe" in r.get("pending", ()) for r in rows)
    # the ramp stops short of its darkest steps so the ink printed on a cell reads
    warm = LinearSegmentedColormap.from_list("det", st.WARM[:5])
    grey = LinearSegmentedColormap.from_list("det_p", ["#F3F1EE", "#B9B3AA"])
    # the shade spans the depths the label-free readers actually reach, so
    # rows a tenth of depth apart do not look the same; the key prints the ends
    free = [v for r in rows for k in cols for v in [r["det"].get(k)] if v is not None and np.isfinite(v)]
    d_lo = float(np.floor(min(free) * 20) / 20) if free else 0.0
    d_hi = float(np.ceil(max(free) * 20) / 20) if free else 1.0
    shade = lambda v: 0.0 if not np.isfinite(v) else float(np.clip((d_hi - v) / max(d_hi - d_lo, 1e-9), 0, 1))
    n_rows, fs = len(rows), st.matplotlib.rcParams["ytick.labelsize"]
    fsc = fs * 0.9
    cw, rh = 0.9, 0.36
    xs = {k: i for i, k in enumerate(cols)}
    if probe:
        xs["probe"] = len(cols) + 0.35
    # the cells narrow when there are many readers, so the page is not exceeded
    cw = min(cw, 5.95 / (max(xs.values()) + 1))
    ncol_w = (max(xs.values()) + 1) * cw
    lab_w, band_h, head_h = 3.35, 0.42, 0.62
    W = lab_w + ncol_w + 0.05
    H = band_h + head_h + n_rows * rh + 0.05
    fig = plt.figure(figsize=(W, H), layout="none")
    ax = fig.add_axes([lab_w / W, 0.05 / H, ncol_w / W, n_rows * rh / H])
    ax.set_xlim(-0.5, max(xs.values()) + 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.axis("off")

    def fmt(v, mk=""):
        return ("never" if not np.isfinite(v) else f"{v:.2f}".lstrip("0")) + mk
    for j, r in enumerate(rows):
        mk = "\u2020" if r.get("thin") else ""
        for k, x in xs.items():
            v = r["det"].get(k)
            if v is None:
                if k in r.get("pending", ()):
                    B._pending_cell(ax, x, j)
                elif r.get("na"):
                    ax.text(x, j, "n/a" + mk, ha="center", va="center", color=st.SUBINK, fontsize=fsc)
                continue
            cm = grey if k == "probe" else warm
            c = cm(shade(v))
            ax.add_patch(Rectangle((x - 0.5, j - 0.5), 1, 1, facecolor=c, edgecolor="white", lw=1.5))
            ax.text(x, j, fmt(v, mk), ha="center", va="center", color=st.INK, fontsize=fsc)
        if j and rows[j - 1]["group"] != r["group"]:
            ax.plot([-0.5, max(xs.values()) + 0.5], [j - 0.5, j - 0.5], color=st.INK, lw=0.9,
                    zorder=5, clip_on=False)
    # V-Read's column, outlined
    ax.add_patch(Rectangle((-0.5, -0.5), 1, n_rows, facecolor="none", edgecolor=st.INK, lw=1.8,
                           zorder=6, clip_on=False))
    for k, x in xs.items():
        ax.text(x, -0.62, HEAD[k], ha="center", va="bottom", color=st.INK, fontsize=fsc,
                linespacing=1.0)
    # task names, then the groups as a bracket column to their left
    labs = [ax.text(-0.62, j, r["label"] + ("\u2020" if r.get("thin") else ""), ha="right", va="center",
                    color=st.INK, fontsize=fs)
            for j, r in enumerate(rows)]
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    x_min = min(t.get_window_extent(rnd).x0 for t in labs)
    gx = ax.transData.inverted().transform((x_min - 10, 0))[0]
    j0 = 0
    for j in range(1, n_rows + 1):
        if j == n_rows or rows[j]["group"] != rows[j0]["group"]:
            ax.plot([gx, gx], [j0 - 0.35, j - 1 + 0.35], color=st.SUBINK, lw=1.0, clip_on=False)
            ax.text(gx - 0.08, (j0 + j - 1) / 2, GROUP_NAME[rows[j0]["group"]], ha="right",
                    va="center", color=st.INK, fontsize=fs)
            j0 = j
    # the key, in a band of its own above the column heads and ending where
    # the matrix ends: what the shade is, then its scale, earliest darkest
    fsk = fs * 0.85
    yk = (H - 0.16) / H
    rnd = fig.canvas.get_renderer()
    parts = [fig.text(0, yk, f"depth from which AUROC stays at or above {THR}:", ha="left",
                      va="center", fontsize=fsk, color=st.INK),
             fig.text(0, yk, f"{d_lo:.2f}", ha="left", va="center", fontsize=fsk, color=st.INK),
             None,
             fig.text(0, yk, f"{d_hi:.2f} or never", ha="left", va="center", fontsize=fsk,
                      color=st.INK)]
    fig.canvas.draw()
    bar_w, gaps = 1.5, (0.12, 0.08, 0.08)
    wid = [t.get_window_extent(rnd).width / fig.dpi if t is not None else bar_w for t in parts]
    x = (lab_w + ncol_w) - (sum(wid) + sum(gaps))
    for t, w_, g in zip(parts, wid, (*gaps, 0.0)):
        if t is None:
            cax = fig.add_axes([x / W, yk - 0.065 / H, bar_w / W, 0.13 / H])
            cax.imshow(np.linspace(1, 0, 256)[None, :], aspect="auto", cmap=warm, extent=(0, 1, 0, 1))
            cax.set_yticks([])
            cax.set_xticks([])
            for sp in cax.spines.values():
                sp.set_linewidth(0.6)
        else:
            t.set_x(x / W)
        x += w_ + g
    return fig


def _name(model):
    return "early_detection" + ("" if P.SUFFIX else (f"__{model}" if model else "__averaged"))


def early_detection(results="results", out="figures", model=None):
    """MAIN. Detection depth per task and reader, every reader read on V-Read's blocks; replaced by the caption built from the files (``_caption``)."""
    rows = task_rows(results, model)
    # the pooled figure needs two models that both carry V-Read
    if model is None and len({m for r in rows for m in r["models"]}) < 2:
        return None
    if not rows or not any(LEAD in r["det"] for r in rows):
        return None
    name = f"__{model}" if model else "__averaged"
    f = findings(rows, name)
    print("  " + findings_text(f, "early_detection" + name))
    early_detection.__doc__ = _caption(results)
    fig = _draw(rows, name)
    if fig is None:
        return None
    return st.save(fig, P._at(out, _name(model)), check=False, tight=True)


def _names(ks):
    ns = [P.NICE[k] if k == "patchscope" else "the " + P.NICE[k] for k in ks]
    return ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1]


def _against(rows):
    """Which rows a model's comparison leaves some published baseline out of, grouped by what it has."""
    by = {}
    for r in rows:
        have = tuple(k for k in BASELINES if k in r["det"])
        if set(have) != set(BASELINES):
            by.setdefault(have, []).append(r["label"])
    return by


def _caption(results="results"):
    """The caption, computed from the result files: which rows V-Read is earliest on, per model."""
    models = [m for m in _models(results) if any(LEAD in r["det"] for r in task_rows(results, m))]
    per = {m: task_rows(results, m) for m in models}
    F = {m: findings(rows) for m, rows in per.items()}
    lead_bits, find_bits, lim_bits, run_bits = [], [], [], []
    for m in models:
        f, rows = F[m], per[m]
        gains = []
        for r in rows:
            base = {k: r["det"][k] for k in BASELINES if r["det"].get(k) is not None}
            if LEAD in r["det"] and base and not r.get("thin"):
                k = min(base, key=base.get)
                v, b = r["det"][LEAD], base[k]
                if np.isfinite(v) and np.isfinite(b):
                    gains.append((b - v, r["label"], k, v, b))
        ag = _against(rows)
        only = "" if not ag else " (" + "; ".join(
            f"{', '.join(labs)} against {_names(have)} only" if have else f"{', '.join(labs)} against no baseline"
            for have, labs in ag.items()) + ")"
        have_m = [k for k in BASELINES if any(k in r["det"] for r in rows)]
        lead_bits.append(f"{len(f['earliest'])} of {f['with_lead']} task rows on {m}"
                         + ("" if set(have_m) == set(BASELINES) else f" (against {_names(have_m)} at most)"))
        s = (f"{m}{only}: V-Read is earliest of the published label-free readers on {len(f['earliest'])} rows"
             + (f" ({', '.join(f['earliest'])})" if f["earliest"] else "")
             + f", level on {len(f['level'])} and later on {len(f['beaten'])}")
        if f["beaten"]:
            s += " (" + "; ".join(f"{lab}, {_fmt_d(v)} against {k} {_fmt_d(b)}" for lab, k, v, b in f["beaten"]) + ")"
        if gains:
            g = sorted(gains)
            s += (f"; median lead over the earliest baseline {np.median([x[0] for x in g]):+.2f} of depth, "
                  f"largest {g[-1][0]:+.2f} on {g[-1][1]} ({_fmt_d(g[-1][3])} against {P.NICE[g[-1][2]]} "
                  f"{_fmt_d(g[-1][4])})")
        coco = [(r["label"], r["det"]["v_read_coco"]) for r in rows if r["det"].get("v_read_coco") is not None]
        if coco:
            s += ("; V-Read fitted on COCO, a separate reader not used to judge: "
                  + ", ".join(f"{lab} {_fmt_d(v)}" for lab, v in coco))
        s += ". Records per row: " + ", ".join(f"{r['label']} {r['n']:,}" for r in rows) + "."
        find_bits.append(s)
        if ag:
            lim_bits.append(f"On {m} some rows lack published baselines{only}: the baselines not named have "
                            f"no readouts there, so those rows compare V-Read against fewer baselines.")
        pend = {}
        for r in rows:
            for k in r.get("pending", ()):
                pend.setdefault(k, []).append(r["label"])
        if pend:
            run_bits.append(f"on {m}, " + "; ".join(
                f"{P.NICE[k] if k == 'patchscope' else 'the ' + P.NICE[k]} "
                + ("(every row)" if len(labs) == len(rows) else f"({', '.join(labs)})")
                for k, labs in pend.items()))
    lead = ("MAIN. Read on the same blocks for every reader, V-Read, which needs no labels, does not in general "
            "settle on the answer a vision-language model is about to give earlier than the published label-free "
            "baselines: at AUROC 0.75 it is earliest on " + "; ".join(lead_bits)
            + ", and level or later on the rest.")
    shown = (
        "What is shown. One row per task, grouped: refusal (MM-SafetyBench scenarios with the benchmark's "
        "typographic carrier and MOSSBench: the reply refusing, the prompt harmful over every pool, and harmful "
        "over the two MM-SafetyBench pools only), COCO (POPE adversarial, and a four-way question on val2017 "
        "naming one present object and three co-occurring absent ones), SimonSays by source (Open Images, "
        "nuScenes, Interfaces (VINS-120K), Generated scenes and the WMDP carrier; the authority, eval-awareness "
        "and reading probes of each source pooled, the WMDP carrier's plain, calibration and hazard "
        "instructions pooled) and sandbagging by channel (WMDP items, the question and the instruction each "
        "given in text or in the picture, every condition pooled). A row is drawn from its families that pass "
        f"the scoring floor ({B.MIN_N} records, {B.MIN_CLASS} of each answer); a row none of whose families "
        "passes it is drawn from them all and marked \u2020 (fewer than "
        f"{B.MIN_CLASS} of one answer: values shown, not counted as earliest, level or later; n/a\u2020 where "
        "no record has the other answer, so AUROC is undefined). The figure is drawn per model and pooled over models once two models carry V-Read "
        "readouts on at least half their records. The password-locking families are never read (asserted on "
        "every family), and no WMDP question text appears.")
    how = (
        "How detection depth is computed. For each family, block and reader: the AUROC with which the reader's "
        "value at the answer spot, read through the model's own final norm and unembedding, predicts the answer "
        "the model then gives (yes or no, test or real, whether the letter it picks is the correct one), on "
        "records none of the fitted lenses was fitted on, every reader on the same records within a family. "
        "Every reader is read only at the blocks V-Read is fitted at, as in the tables, so that a reader read "
        "on a denser grid does not get more chances to dip below the threshold. AUROC is put on normalised "
        "depth (block over the number of blocks less one), linear between those blocks, and averaged over a "
        "row's families weighted by records; a reader seen on under half a row's records is left off that row. "
        "A reader's detection depth is the earliest depth from which that curve stays at or above 0.75 to the "
        "output; a reader that ends below it sits at never. The figure is a matrix, one row per task and one "
        "column per reader: V-Read (outlined), V-Read fitted on COCO photographs with generic instructions (a "
        "separate reader, never used to judge V-Read), the published label-free baselines (the tuned lens and "
        "the Jacobian lens where their readouts exist, Patchscopes in whichever of its two prompts reads better "
        "on that model, and the logit lens) and, apart on the right in greys, the labelled probe (supervised "
        "reference), a logistic probe per block scored out of fold. Each cell prints the detection depth and is "
        "shaded by it, darker for earlier; a hatched cell with no number is a reader whose run on that task has "
        "not landed; a blank cell is a reader with no readouts on that task, or seen on under half its records. The probe reads the answer from the first block on Open Images and the refusal "
        "rows because there the answer is largely predictable from the prompt's text, which the answer-spot "
        "state already carries after one block; its folds are grouped by picture, so this is not a picture "
        "seen in training.")
    running = ("Still running (hatched cells): " + "; ".join(run_bits) + ". They are drawn when their runs "
               "land." if run_bits else "")
    limits = ("Limits. " + " ".join(lim_bits)
              + (" In the pooled file a reader enters a row on the records of the models that have it, so a "
                 "baseline missing on one model is drawn from the other alone." if lim_bits else "")
              + " Detection depths are point estimates with no interval; many rows differ by a few hundredths "
                "of depth, well inside the spacing of V-Read's blocks. A detection depth at the first read "
                "block means the reader reads the answer from there, not that it reads it from the input.")
    return "\n\n".join([x for x in (lead, shown, how, "Findings. " + " ".join(find_bits), running, limits) if x])


def _import_caption():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        return _caption(os.path.join(here, "results"))
    except Exception:  # noqa: BLE001
        return early_detection.__doc__


#: ``tools/make_captions.py`` calls this for the caption, so the text is built from what is on disk
early_detection.caption = _import_caption


def render_all(results="results", out="figures"):
    """Every model's figure, then the pooled one."""
    made = {m: early_detection(results, out, m) for m in _models(results)}
    made["averaged"] = early_detection(results, out, None)
    return made


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    for k, v in render_all(a.results, a.out).items():
        print(f"{k}: {v}")
