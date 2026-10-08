# SPDX-License-Identifier: Apache-2.0
"""The V-Read results tables: every cell, classified by a paired test; where V-Read is not behind (MAIN), where it is (APPENDIX), all cells in one table (APPENDIX), the depth-window sensitivity (APPENDIX), and what each fit cost.

Every number is computed from the result files; nothing is typed.

Per (task, model) cell, in the depth window ``fx_bench.BAND`` (the mean over
V-Read's fitted blocks inside it, the same blocks for every reader): the AUROC
with which each reader's reading of the answer-spot state, through the model's
own final norm and unembedding, predicts the answer the model gives. Columns:
V-Read (the primary fit, on unlabelled SimonSays prompts), V-Read fitted on
COCO (a second reader, reported beside it and never used to choose a winner or
count a cell), the best label-free baseline (named; a maximum over the
baselines on the test data itself, which favours the baselines), V-Read's
margin over it with a paired 95% interval, the result, the labelled probe
(a supervised reference, in grey italics), and the earliest depth from which
V-Read, and the earliest baseline, stay at or above AUROC 0.75 to the output
(``fx_detect.detection``), read on V-Read's blocks for every reader.

The interval of the margin is a paired bootstrap over pictures: the cluster is
the record's source picture (``tools/probe_readouts.picture`` of the behaviour
row: the source photograph of an edit, else its group, else its image), the
same resample for every reader and block, and the best baseline is taken
inside each resample. A cell is a win when the whole interval is above zero,
behind when it is below zero, and level otherwise. Refusal cells take the same
interval from the per-record readings ``experiments/safety/refusal.py`` stores
(``results/safety/<model>__refusal_readings.npz``, clustered by picture); a
refusal run that predates them has no paired interval and is marked untested.

Five tables:

* ``tables/vread_main.tex`` (MAIN): the win and level cells.
* ``tables/vread_appendix.tex`` (APPENDIX): the behind cells.
* ``tables/vread_all.tex`` (APPENDIX): every cell, untested ones included, so
  nothing is selected by outcome.
* ``tables/vread_window.tex`` (APPENDIX): V-Read's margin with its paired
  interval in other depth windows (``WINDOWS``), the main one included.
* ``tables/vread_cost.tex`` (APPENDIX): per model, wall-clock fit time and data used by
  V-Read, V-Read fitted on COCO, the tuned lens and the Jacobian lens, with
  ratios to V-Read and the development-set KL, from ``results/fits``
  (``tools/fit_meta.py``). Which cache entry each reader came from is taken from
  the readouts' own record (``results/*readouts/<model>__sanity.json``, key
  ``maps``); failing that, the shared fit is the 900-prompt entry holding the
  most lenses (ties broken by the fit-pool digest the other models' shared fits
  use) and the COCO fit is the other entry. A lens missing from a model's shared
  fit is shown as ``running``.

Every results table also shows efficiency beside accuracy: per row, the best
baseline's wall-clock fit time over V-Read's on that model (``none`` for
Patchscopes and the logit lens, which are not fitted; ``running`` while a fit
has not landed) and its read cost (``READ_COST``: one matrix multiply per block
for V-Read, the tuned and the Jacobian lens, one extra forward pass per block
read per record for Patchscopes, nothing extra for the logit lens); a line
under each model's columns with the fit times; and a caption sentence
(:func:`eff_headline`) computed from the fit records and the paired results.

Every baseline is at its paper protocol; in particular every tuned-lens cell
reads a fit whose recorded meta matches the published recipe
(``vlens.lens.baselines.PAPER``), which the script asserts.

Cells come from the per-block answer readouts (``results/readouts``,
``results/coco_readouts`` and the channel run) when they hold V-Read, scored by
:func:`vlens.figures.fx_bench._readout_family` (through the
:func:`vlens.figures.fx_detect._family` shim), unchanged: fitted-on records and
COCO-fit photographs dropped for every reader, every reader on the same
records, target the answer the reply states. The bootstrap reads the matrices
``_readout_family`` scored (captured from its frame, so the scoring is not
re-implemented). Refusal cells come from ``results/safety``. The old lens bench
(``results/lensbench``, whose tuned lens was not at the published recipe) is
not read.

Password-locking families (``paper.LOCKED``) are never read. WMDP rows are
named by subject only; no question text is printed.

    python3 tools/make_vread_table.py
    python3 tools/make_vread_table.py --results results --out tables/vread_main.tex
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sys
import warnings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
import numpy as np  # noqa: E402

from probe_readouts import picture  # noqa: E402
from vlens.figures import fx_bench as B  # noqa: E402
from vlens.figures import fx_detect as D  # noqa: E402
from vlens.figures import paper as P  # noqa: E402
from vlens.lens.metrics import auroc  # noqa: E402

MODELS = ("Qwen3-VL-8B-Instruct", "Qwen3.5-27B", "gemma-4-26B-A4B-it", "InternVL3_5-8B-HF")


class _Exact(dict):
    """Models are named by their exact id (the repo's basename), whichever model turns up."""

    def __missing__(self, k):
        return k


SHORT = _Exact()
#: the primary fit, the only one a cell is judged by
LEAD = "v_read"
OURS = ("v_read", "v_read_coco")
BASE = ("tuned", "jacobian", "patchscope", "logit")
CEIL = ("probe", "dom")
ABBR = {"tuned": "TL", "jacobian": "JL", "patchscope": "PS", "logit": "LL"}
BAND = B.BAND
#: depth windows for the sensitivity table; the main one is ``BAND``, and (0, 1) is every fitted block
WINDOWS = ((0.3, 0.6), tuple(BAND), (0.6, 0.9), (0.0, 1.0))
THR = D.THR
REPS = 1000
#: the result of a cell, from its paired interval
RESULT = ("win", "level", "behind", "untested")
#: a cell is flagged underpowered when its rarer answer has fewer records than this ...
MIN_MINORITY = 20
#: ... or when more than UNDECIDED of the model's own final-block margins are under MARGIN_NATS nats
MARGIN_NATS = 0.5
UNDECIDED = 0.5

GROUPS = {"photos": "Natural photographs", "vins": "VINS-120K",
          "gemini": "Generated scenes", "wmdp": "WMDP (four letters)",
          "refusal": "Refusal"}


def _the(k):
    return P.NICE[k] if k == "patchscope" else "the " + P.NICE[k]


QUESTION = {"auth": "authority, yes/no", "eval": "eval awareness, test/real"}


# ------------------------------------------------------------------ where a family belongs

def group_of(fam):
    if fam.startswith("refusal:"):
        return "refusal"
    src = fam.split("_")[0]
    if src in ("openimages", "nuscenes", "coco"):
        return "photos"
    return src if src in GROUPS else None


def label_of(fam):
    if fam.startswith("refusal:"):
        t = fam.split(":", 1)[1]
        return {"refused": "reply refuses", "harmful": "prompt harmful, all pools",
                "harmful_matched": "prompt harmful, MM-SafetyBench only"}.get(t, t)
    for f, lab in D.COCO:
        if fam == f:
            return f"COCO, {lab}"
    if fam == "wmdp_chan":
        return "WMDP, by channel"
    bits = fam.split("_")
    src = P.SOURCE.get(bits[0], bits[0]).replace(" (VINS-120K)", "")
    rest = []
    for b in bits[1:]:
        rest.append({"orig": "as taken", "edit": "edited"}.get(b) or P.SUBJECT.get(b)
                    or QUESTION.get(b) or P.PROBE.get(b, b))
    if bits[0] == "wmdp" and bits[-1] == "sand":
        rest[-1] = "3 instructions"
    return f"{src}, {', '.join(rest)}"


# ------------------------------------------------------------------ readout cells

def _capture(fn, *a):
    """``fn(*a)`` and the (y, series, picture groups) that ``fx_bench._readout_family`` scored inside it."""
    got = {}
    code = B._readout_family.__code__

    def prof(frame, event, _arg):
        if event == "return" and frame.f_code is code and "series" in frame.f_locals:
            recs = frame.f_locals.get("recs", [])
            beh = frame.f_locals.get("beh", {})
            got.update(y=frame.f_locals["y"], series=dict(frame.f_locals["series"]),
                       kinds={r["kind"] for r, _t, _e in recs},
                       ids=[r["id"] for r, _t, _e in recs],
                       # the cluster of the bootstrap: the source picture, from the behaviour row
                       groups=[picture(beh.get(r["id"], r)) for r, _t, _e in recs])
    old = sys.getprofile()
    sys.setprofile(prof)
    try:
        res = fn(*a)
    finally:
        sys.setprofile(old)
    return res, got


def _readout_fams(results, model):
    fams = set()
    for d in D.READOUT_DIRS:
        for p in glob.glob(f"{results}/{d}/{model}__*_readouts.json"):
            fam = os.path.basename(p).split("__", 1)[1][: -len("_readouts.json")]
            if not P.LOCKED.search(fam) and group_of(fam):
                fams.add(fam)
    return sorted(fams)


def _wauroc(s, y, W):
    """AUROC of scores ``s`` against labels ``y`` under record weights ``W`` (reps, n), ties averaged.

    With integer weights this is the AUROC of the resample that repeats record
    ``i`` ``W[r, i]`` times; records whose score is not finite are left out.
    """
    ok = np.isfinite(s)
    s, y, W = s[ok], y[ok].astype(float), W[:, ok]
    if s.size == 0:
        return np.full(W.shape[0], np.nan)
    o = np.argsort(s, kind="mergesort")
    s, y, W = s[o], y[o], W[:, o]
    starts = np.r_[0, np.nonzero(np.diff(s))[0] + 1]
    wp = np.add.reduceat(W * y, starts, axis=1)
    wn = np.add.reduceat(W * (1.0 - y), starts, axis=1)
    below = np.cumsum(wn, axis=1) - wn
    num = (wp * (below + 0.5 * wn)).sum(1)
    den = wp.sum(1) * wn.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _weights(groups, reps=REPS, seed=0):
    """(reps, n) record weights of a bootstrap that resamples pictures with replacement."""
    _, cid = np.unique(np.asarray(groups, str), return_inverse=True)
    G = int(cid.max()) + 1
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(G, np.full(G, 1.0 / G), size=reps).astype(float)
    return counts[:, cid], G


def _boot_blocks(y, series, cols, W):
    """{reader: (reps, blocks) AUROC} on the columns ``cols``, one shared resample for every reader."""
    out = {}
    for k, M in series.items():
        A = np.full((W.shape[0], M.shape[1]), np.nan)
        for j in cols:
            if np.isfinite(M[:, j]).sum() >= 2:
                A[:, j] = _wauroc(M[:, j], y, W)
        out[k] = A
    return out


def _window(boot, a_of, inb, base):
    """Paired interval of V-Read minus the best baseline (taken inside each resample), window-mean AUROC.

    ``a_of[k]`` is reader ``k``'s point AUROC per block; a block enters a
    reader's window mean only where that point value exists, as in the table.
    Also returns every reader's own interval.
    """
    def wmean(k):
        cols = inb & np.isfinite(a_of[k])
        if not cols.any():
            return None
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(boot[k][:, cols], axis=1)
    got = {k: wmean(k) for k in boot if k in a_of}
    got = {k: v for k, v in got.items() if v is not None}
    per = {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
           for k, v in got.items() if np.isfinite(v).any()}
    bs = [got[k] for k in base if k in got]
    if LEAD not in got or not bs:
        return None, per
    d = got[LEAD] - np.max(np.vstack(bs), axis=0)
    d = d[np.isfinite(d)]
    if d.size < 0.9 * REPS:
        return None, per
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))), per


def _band_of(a_of, inb, keys):
    out = {}
    for k in keys:
        if k in a_of:
            v = a_of[k][inb]
            if np.isfinite(v).any():
                out[k] = float(np.nanmean(v))
    return out


def classify(ci):
    if ci is None:
        return "untested"
    return "win" if ci[0] > 0 else "behind" if ci[1] < 0 else "level"


def readout_cells(results, model):
    """{fam: cell} from the answer readouts, and the set of families whose readouts are unscorable."""
    got, dropped = [], set()
    for fam in _readout_fams(results, model):
        res, loc = _capture(D._family, results, model, fam)
        if res is None:
            # under the scoring floor: scored on whatever records it has, shown marked, never judged
            res, loc = _capture(D._family, results, model, fam, False)
            dropped.add(fam)
            if res is None:
                continue
        got.append((res, loc))
    # the Patchscopes prompt is picked on the families that pass the floor, then applied to the rest
    best = B._pick_patchscope([r for r, _l in got if not r.get("thin")])
    for r, _l in got:
        if r.get("thin"):
            v = r["auroc"].get(best) if best else None
            for k in [k for k in r["auroc"] if k.startswith("patchscope")]:
                r["auroc"].pop(k)
            if v is not None:
                r["auroc"]["patchscope"] = v
    cells, logit_only = {}, set()
    for res, loc in got:
        fam = res["fam"]
        if res.get("thin") and res.get("minority") == 0:
            # one answer only: AUROC is undefined for every reader
            cells[fam] = dict(src="readouts", n=res["n"], pictures=len(set(loc.get("groups", []))), det={},
                              windows={w: dict(band={}, ci={}, margin_ci=None) for w in WINDOWS},
                              minority=0, undecided=None, thin=True, na=True)
            continue
        if LEAD not in res["auroc"]:
            logit_only.add(fam)
            continue
        series = loc["series"]
        if best and best in series:
            series["patchscope"] = series[best]
        series = {k: v for k, v in series.items() if k in (*OURS, *BASE, *CEIL)}
        dep = np.asarray(res["depth"], float)
        a_of = {k: np.asarray(a, float) for k, a in res["auroc"].items() if k in series}
        vb = np.isfinite(a_of[LEAD])
        y = np.asarray(loc["y"], int)
        # the captured matrices are exactly what was scored
        for k, a in a_of.items():
            j = int(np.nanargmax(np.isfinite(a)))
            fin = np.isfinite(series[k][:, j])
            assert abs(auroc(y[fin], series[k][fin, j]) - a[j]) < 1e-9, (model, fam, k)
        assert len(loc["groups"]) == y.size, (model, fam)
        W, G = _weights(loc["groups"])
        # every reader over the same blocks: V-Read's fitted blocks
        boot = _boot_blocks(y, series, np.nonzero(vb)[0], W)
        cell = dict(src="readouts", n=res["n"], pictures=G, det={}, windows={},
                    minority=int(min(y.sum(), y.size - y.sum())), undecided=None, thin=bool(res.get("thin")))
        # how often the model's own answer is near a coin-flip: final-block margin under MARGIN_NATS
        if "logit" in series and dep[-1] >= 1.0 and loc["kinds"] and loc["kinds"] <= {"yesno", "test_real"}:
            last = series["logit"][:, -1]
            last = last[np.isfinite(last)]
            if last.size:
                cell["undecided"] = float(np.mean(np.abs(last) < MARGIN_NATS))
        for w in WINDOWS:
            inb = (dep >= w[0]) & (dep <= w[1]) & vb
            band = _band_of(a_of, inb, (*OURS, *BASE, *CEIL))
            mci, per = _window(boot, a_of, inb, [k for k in BASE if k in band])
            cell["windows"][w] = dict(band=band, ci=per, margin_ci=mci)
        for k, a in a_of.items():
            # the depth column too is read on V-Read's blocks, the same for every reader
            cell["det"][k] = D.detection(D._on_grid(dep, np.where(vb, a, np.nan)), THR)
        cells[fam] = cell
    return cells, dropped, logit_only, best


def _refusal_series(z, t, keys):
    """(y, {reader: records x blocks}, picture groups) for refusal target ``t`` from ``refusal_readings.npz``."""
    pop = z[f"pop|{t}"].astype(bool) & z["scored"].astype(bool)
    y = np.asarray(z["refused"] if t == "refused" else z["harmful"], int)[pop]
    series = {}
    for k in keys:
        src = f"{t}|{k}" if f"{t}|{k}" in z.files else f"lens|{k}"
        if src in z.files:
            series[k] = np.asarray(z[src], float)[pop]
    return y, series, [str(g) for g in z["groups"][pop]]


def refusal_cells(results, model, best=None):
    """Refusal cells from ``results/safety``: window AUROC per reader, and, when the run stored
    per-record readings (``<model>__refusal_readings.npz``), the same paired picture-clustered
    interval as the readout cells; without them the cell is untested.

    ``best`` is the Patchscopes target prompt picked for this model on the readouts
    (``fx_bench._pick_patchscope``); the refusal cells read the same one.
    """
    p = f"{results}/safety/{model}__refusal_auroc.json"
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    dep = np.asarray(d["depths"], float)
    pz = f"{results}/safety/{model}__refusal_readings.npz"
    z = np.load(pz) if os.path.exists(pz) else None
    ps = best if best and any(k.endswith(f"|{best}") for k in d["auroc"]) else "patchscope"
    src_of = {k: (ps if k == "patchscope" else k) for k in (*OURS, *BASE, *CEIL)}
    cells = {}

    def arr(v):
        return np.array([np.nan if x is None else float(x) for x in v], float)
    for t, _lab in D.REFUSAL:
        if t not in (d.get("targets") or {}):
            continue
        vr = d["auroc"].get(f"{t}|{LEAD}")
        if vr is None:
            continue
        # every reader over the same blocks: V-Read's blocks
        vb = np.isfinite(arr(vr))
        tg = d["targets"][t]
        pos = tg.get("positives")
        a_of = {k: arr(d["auroc"][f"{t}|{s}"]) for k, s in src_of.items() if d["auroc"].get(f"{t}|{s}")}
        cell = dict(src="refusal", n=int(tg["n"]), pictures=None, det={}, windows={}, undecided=None,
                    minority=None if pos is None else int(min(pos, int(tg["n"]) - pos)))
        boot = None
        if z is not None and f"pop|{t}" in z.files:
            y, raw, groups = _refusal_series(z, t, set(src_of.values()))
            series = {k: raw[s] for k, s in src_of.items() if s in raw and k in a_of}
            # the stored readings are exactly what was scored
            for k, M in series.items():
                j = int(np.nanargmax(np.isfinite(a_of[k])))
                fin = np.isfinite(M[:, j])
                assert abs(auroc(y[fin], M[fin, j]) - a_of[k][j]) < 1e-6, (model, t, k)
            W, G = _weights(groups)
            cell["pictures"] = G
            boot = _boot_blocks(y, series, np.nonzero(vb)[0], W)
        for w in WINDOWS:
            inb = (dep >= w[0]) & (dep <= w[1]) & vb
            band = _band_of(a_of, inb, a_of)
            if boot is None:
                cell["windows"][w] = dict(band=band, ci={}, margin_ci=None)
            else:
                mci, per = _window(boot, a_of, inb, [k for k in BASE if k in band])
                cell["windows"][w] = dict(band=band, ci=per, margin_ci=mci)
        for k, a in a_of.items():
            cell["det"][k] = D.detection(D._on_grid(dep, np.where(vb, a, np.nan)), THR)
        if LEAD in cell["windows"][tuple(BAND)]["band"]:
            cells[f"refusal:{t}"] = cell
    return cells


# ------------------------------------------------------------------ assembling

def _judge(w):
    """Best baseline (by point window AUROC), V-Read's margin over it, the paired interval and the result."""
    b = w["band"]
    base = {k: b[k] for k in BASE if k in b}
    w["best"] = max(base, key=base.get) if base else None
    w["margin"] = (b[LEAD] - b[w["best"]]) if LEAD in b and w["best"] else None
    w["margin_coco"] = (b["v_read_coco"] - b[w["best"]]) if "v_read_coco" in b and w["best"] else None
    w["result"] = classify(w["margin_ci"]) if w["margin"] is not None else "untested"
    return w


def summarise(cell):
    """The main window's numbers lifted onto the cell, every window judged, earliest baseline and probe."""
    for w in cell["windows"].values():
        _judge(w)
    if cell.get("thin"):
        # under the scoring floor: shown, marked, and never counted as a win, level or behind
        for w in cell["windows"].values():
            w["result"] = "thin"
    main = cell["windows"][tuple(BAND)]
    cell.update(band=main["band"], ci=main["ci"], best=main["best"], margin=main["margin"],
                margin_ci=main["margin_ci"], margin_coco=main["margin_coco"], result=main["result"])
    dets = {k: cell["det"][k] for k in BASE if cell["det"].get(k) is not None}
    cell["early"] = min(dets, key=dets.get) if dets else None
    cell["ceil"] = next((k for k in CEIL if k in cell["band"]), None)
    return cell


#: per model, the families with no scorable record at all; any other row a model lacks is running
#: (families under the scoring floor are shown, marked with a dagger, and never judged)
NOT_SCORED: dict = {}


#: the results directory ``collect`` read
RESULTS = ["results"]


def _absent(m, f, tex_=True):
    """What a model's missing cell in row ``f`` says: no records (its behaviour pass has none for this
    SimonSays family, or none scorable), else running (a run that has not landed)."""
    simon = group_of(f) != "refusal" and not f.startswith("coco") and f != "wmdp_chan"
    if f in NOT_SCORED.get(m, ()):
        s = "no records"
    elif simon and not os.path.exists(f"{RESULTS[0]}/simonsays/{m}__{f}_states.npz"):
        s = "not read"             # the behaviour pass stored no answer-spot states for it
    else:
        s = "running"
    return f"\\textit{{{s}}}" if tex_ else s


def running_readers(cost, m):
    """The fitted baselines whose fit has not landed on model ``m`` (``fit_cost`` has them as None)."""
    r = (cost or {}).get(m)
    return [k for k in ("tuned", "jacobian") if r is not None and r.get(k) is None]


def collect(results):
    RESULTS[0] = results
    models = [m for m in MODELS]
    extra = {os.path.basename(p).split("__")[0]
             for q in (*(f"{d}/*__*_readouts.json" for d in D.READOUT_DIRS), "safety/*__refusal_auroc.json")
             for p in glob.glob(f"{results}/{q}")}
    models += sorted(extra - set(MODELS))
    table, notes = {}, {}
    for m in models:
        rc, dropped, logit_only, best = readout_cells(results, m)
        cells = {**rc, **refusal_cells(results, m, best)}
        if not cells:
            continue
        table[m] = {f: summarise(c) for f, c in cells.items()}
        notes[m] = dict(dropped=sorted(dropped), patchscope=best)
        NOT_SCORED[m] = set(dropped) - set(cells)
    return table, notes


def ordered_rows(table):
    fams = sorted({f for cells in table.values() for f in cells})
    score = {}
    for f in fams:
        ms = [cells[f]["margin"] for cells in table.values() if f in cells and cells[f]["margin"] is not None]
        score[f] = float(np.mean(ms)) if ms else -9.0
    groups = {}
    for f in fams:
        groups.setdefault(group_of(f), []).append(f)
    gscore = {g: float(np.mean([score[f] for f in fs])) for g, fs in groups.items()}
    out = []
    for g in sorted(groups, key=lambda g: -gscore[g]):
        out.append((g, sorted(groups[g], key=lambda f: -score[f]), gscore[g]))
    return out, score


def baselines_of(table, m):
    """The label-free baselines present in any of a model's cells, in ``BASE`` order."""
    return [k for k in BASE if any(k in c["band"] for c in table.get(m, {}).values())]


def against(table, m):
    """'' when every published baseline was read on the model, else which ones it is compared against."""
    have = baselines_of(table, m)
    if set(have) >= set(BASE):
        return ""
    names = [_the(k) for k in have]
    return "against " + (" and ".join(names) if len(names) <= 2 else ", ".join(names[:-1]) + " and " + names[-1]) \
        + " only"


# ------------------------------------------------------------------ fit cost

#: the machine every fit here ran on
HARDWARE = "one NVIDIA H200 NVL, shared with at most one other job"
#: the corpus size of the shared fit (``vlens.lens.fitall.SHARED``)
SHARED_CORPUS = 900
COST_LENSES = ("v_read", "v_read_coco", "tuned", "jacobian")


def _sanity_maps(results, model):
    """{reader: cache file} as the readouts recorded which fit each reader came from."""
    out = {}
    for d in D.READOUT_DIRS:
        p = f"{results}/{d}/{model}__sanity.json"
        if not os.path.exists(p):
            continue
        s = (json.load(open(p)) or {}).get("maps") or ""
        for part in str(s).split(";"):
            bits = part.strip().split(" from ")
            if len(bits) == 2:
                out.setdefault(bits[0].strip(), os.path.basename(bits[1].strip()))
    return out


def _tuned_ok(t):
    """Whether a tuned-lens fit's recorded meta is the published recipe."""
    from vlens.lens.baselines import PAPER as TP
    return (int(t.get("steps", -1)) == TP["steps"] and int(t.get("tokens_per_step", -1)) == TP["tokens_per_step"]
            and float(t.get("lr_chosen", t.get("lr", -1))) == TP["lr"]
            and float(t.get("wd_chosen", t.get("weight_decay", -1))) == TP["weight_decay"]
            and bool(t.get("nesterov", True)) and not t.get("lr_searched", False))


def fit_cost(results, models):
    """Per model: the cost meta of V-Read, V-Read fitted on COCO, the tuned lens and the Jacobian lens.

    ``None`` for a model with no fit meta yet; a lens absent from the model's
    shared fit is ``None`` (shown as running).
    """
    fits = {os.path.basename(p).split("__")[0]: json.load(open(p))
            for p in glob.glob(f"{results}/fits/*__fitmeta.json")}
    # the fit-pool digest of shared fits, from entries that hold more than one lens
    pools = {}
    for es in fits.values():
        for e in es:
            if e["spec"][3] == SHARED_CORPUS and len(e["meta"]) > 1:
                pools[e["spec"][1]] = pools.get(e["spec"][1], 0) + 1
    out = {}
    for m in list(models) + sorted(set(fits) - set(models)):
        es = fits.get(m)
        if not es:
            out[m] = None
            continue
        by = {e["cache"]: e for e in es}
        mp = _sanity_maps(results, m)
        shared, coco, how = by.get(mp.get("v_read")), by.get(mp.get("v_read_coco")), "readouts"
        if shared is None:
            how = "rule"
            cand = [e for e in es if e["spec"][3] == SHARED_CORPUS]
            shared = max(cand, key=lambda e: (len(e["meta"]), pools.get(e["spec"][1], 0))) if cand else None
        if coco is None and shared is not None:
            rest = [e for e in es if e is not shared and "v_read" in e["meta"] and e["spec"][1] != shared["spec"][1]]
            coco = rest[0] if len(rest) == 1 else None
        row = dict(how=how, spec=(shared or es[0])["spec"], coco_spec=(coco or shared or es[0])["spec"],
                   caches={k: (e or {}).get("cache") for k, e in (("shared", shared), ("coco", coco))})
        row["v_read"] = (shared or {}).get("meta", {}).get("v_read")
        row["v_read_coco"] = (coco or {}).get("meta", {}).get("v_read")
        for k in ("tuned", "jacobian"):
            e = by.get(mp.get(k)) or shared
            row[k] = (e or {}).get("meta", {}).get(k)
        out[m] = row
    return out


def check_protocol(table, cost):
    """Every tuned-lens cell must read a fit at the published recipe; stop otherwise."""
    for m, cells in table.items():
        if not any("tuned" in c["band"] for c in cells.values()):
            continue
        t = (cost.get(m) or {}).get("tuned")
        if t is None:
            raise SystemExit(f"{m}: tuned-lens cells but no tuned fit meta in results/fits; "
                             "run tools/fit_meta.py on the machine holding the map cache")
        if not _tuned_ok(t):
            raise SystemExit(f"{m}: the tuned lens read is not at the published recipe: {t}")


def _tuned_tokens(t, spec):
    steps = int(t.get("steps") or spec[6])
    tps = int(t.get("tokens_per_step") or spec[7])
    return steps, tps, steps * tps


def _secs(s):
    if s is None:
        return "--"
    s = float(s)
    return f"{s:,.0f} s ({s / 60:.0f} min)" if s < 3600 else f"{s:,.0f} s ({s / 3600:.1f} h)"


def _x(r):
    return "--" if r is None else (f"{r:.0f}x" if r >= 9.5 else f"{r:.1f}x" if r >= 0.95 else f"{r:.2f}x")


def cost_rows(cost):
    """Per model, the rows of the cost table as plain strings (no markup)."""
    out = {}
    for m, r in cost.items():
        if r is None:
            out[m] = None
            continue
        v = r["v_read"] or {}
        v_s, v_t = v.get("seconds"), v.get("tokens")
        rows = []
        for k in COST_LENSES:
            x = r[k]
            name = {"v_read": "V-Read", "v_read_coco": "V-Read fitted on COCO", "tuned": "tuned lens",
                    "jacobian": "Jacobian lens"}[k]
            if x is None:
                rows.append(dict(k=k, name=name, running=True))
                continue
            sec = x.get("seconds")
            row = dict(k=k, name=name, running=False, sec=sec, kl=x.get("heldout_kl"),
                       xt=(sec / v_s) if (sec and v_s and k != "v_read") else None, xd=None)
            if k in ("v_read", "v_read_coco"):
                row["corpus"] = (f"{r['spec'][3]} SimonSays prompts" if k == "v_read"
                                 else f"{r['coco_spec'][3]} COCO photographs")
                row["data"] = f"{int(x['tokens']):,} tokens, one pass"
                row["xd"] = (x["tokens"] / v_t) if (v_t and k != "v_read") else None
            elif k == "tuned":
                steps, tps, tot = _tuned_tokens(x, r["spec"])
                row["corpus"] = f"the same {r['spec'][3]} prompts"
                row["data"] = f"{steps} steps x {tps:,} tokens"
                row["spent"] = x.get("tokens_spent")
                row["xd"] = tot / v_t if v_t else None
            else:
                row["corpus"] = f"{x.get('n_prompts', r['spec'][9])} SimonSays prompts"
                pos = x.get("n_source_positions")
                row["data"] = f"{int(pos):,} source positions" if pos else "--"
                row["xd"] = (pos / v_t) if (pos and v_t) else None
            rows.append(row)
        out[m] = rows
    return out


def cost_claims(cost):
    """The numbers the cost caption states, from the first model with all three fits."""
    for m, r in cost.items():
        if r and r["v_read"] and r["tuned"] and r["jacobian"]:
            v, t, j = r["v_read"], r["tuned"], r["jacobian"]
            steps, tps, tot = _tuned_tokens(t, r["spec"])
            return dict(m=m, v=v, t=t, j=j, steps=steps, tps=tps, tot=tot,
                        xt=t["seconds"] / v["seconds"], xd=tot / v["tokens"], xj=j["seconds"] / v["seconds"],
                        passes=tot / v["tokens"])
    return None


# ------------------------------------------------------------------ efficiency beside accuracy

#: what reading one block costs at read time, beyond the model's own final norm and unembedding
#: (which every reader, the logit lens included, goes through)
READ_COST = {"v_read": "matmul", "v_read_coco": "matmul", "tuned": "matmul", "jacobian": "matmul",
             "patchscope": "fwd pass/block", "logit": "none"}
READ_LONG = {"matmul": "one matrix multiply per block read",
             "fwd pass/block": "one extra forward pass of the model per block read, per record",
             "none": "nothing beyond the model's own final norm and unembedding"}
#: the same, as the window table's cells print it
READ_SHORT = {"matmul": "mm", "fwd pass/block": "fp", "none": "0"}
#: readers with no fit at all
NO_FIT = ("patchscope", "logit")


def fit_seconds(cost, m, k):
    """Wall-clock fit seconds of reader ``k`` on model ``m``: 0 for a reader with no fit, None when running."""
    if k in NO_FIT:
        return 0.0
    x = ((cost or {}).get(m) or {}).get(k)
    return None if x is None or x.get("seconds") is None else float(x["seconds"])


def eff_ratio(cost, m, k):
    """The fit column of a row whose best baseline is ``k``: 'none', 'running' or best-baseline / V-Read time."""
    if k is None:
        return "--"
    if k in NO_FIT:
        return "none"
    b, v = fit_seconds(cost, m, k), fit_seconds(cost, m, LEAD)
    if b is None:
        return "running"
    return "--" if not v else _x(b / v)


def _dur(s):
    """A fit time, short: 7 min, 7.9 h."""
    return f"{s / 60:.0f} min" if s < 3600 else f"{s / 3600:.1f} h"


def eff_summary(cost, m):
    """The per-model summary line: V-Read's fit time and tokens, the fitted baselines' times and ratios."""
    r = (cost or {}).get(m)
    if not r or not r.get("v_read"):
        return f"V-Read: no fit record yet for {SHORT[m]}"
    v = r["v_read"]
    s = f"V-Read: fit in {_dur(float(v['seconds']))} on {int(v['tokens']):,} tokens"
    for k, name in (("tuned", "tuned lens"), ("jacobian", "Jacobian lens")):
        b = fit_seconds(cost, m, k)
        s += f"; {name} " + ("running" if b is None else f"{_dur(b)} ({_x(b / float(v['seconds']))})")
    return s + "; Patchscopes and logit lens: no fit"


def eff_headline(cost, F, models, where=""):
    """The caption sentence on efficiency, from the fit records and the paired results (all cells, main window)."""
    parts = []
    for m in models:
        r = F["by_model"].get(m)
        if not r:
            continue
        tested = r["win"] + r["level"] + r["behind"]
        s = (f"on {SHORT[m]} V-Read matches or beats the best baseline in {r['win'] + r['level']} of {tested} "
             f"tested cells (wins {r['win']}, statistically level {r['level']})")
        v = fit_seconds(cost, m, LEAD)
        fast, run = [], []
        for k, name in (("tuned", "the tuned lens"), ("jacobian", "the Jacobian lens")):
            b = fit_seconds(cost, m, k)
            if b is None or not v:
                run.append({"tuned": "tuned-lens", "jacobian": "Jacobian-lens"}[k])
            else:
                fast.append(f"{b / v:.0f} times faster than {name}" if b / v >= 9.5
                            else f"{b / v:.1f} times faster than {name}")
        if fast:
            s += " while fitting " + " and ".join(fast)
        if run:
            s += (", with the " if fast else ", the ") + " and ".join(run) + (" fits" if len(run) > 1 else " fit") \
                + " still running" + ("" if fast else ", so no fit-time comparison yet")
        parts.append(s)
    if not parts:
        return ""
    return ("Efficiency" + (f" ({where})" if where else "") + ": " + "; ".join(parts)
            + "; Patchscopes and the logit lens need no fit, but Patchscopes costs one extra forward pass per "
            "block read per record at read time, where V-Read, the tuned lens and the Jacobian lens cost one "
            "matrix multiply per block and the logit lens nothing extra.")

# ------------------------------------------------------------------ formatting

def _a(v):
    return "--" if v is None else f"{v:.2f}".lstrip("0") if 0 <= v < 1 else f"{v:.2f}"


def _d(v):
    if v is None:
        return "--"
    return "never" if not np.isfinite(v) else _a(v)


def _ci(c):
    return "" if not c else f"{_a(c[0])}--{_a(c[1])}"


def _m(v, tex=False):
    """A signed margin at two decimals without the leading zero: +.11, -.03; a value just below zero is -.00."""
    if v is None:
        return "--"
    s = f"{abs(v):.2f}".lstrip("0") if abs(v) < 1 else f"{abs(v):.2f}"
    neg = v < 0
    return ("$-$" if tex else "-") + s if neg else "+" + s


def _mci(c, tex=False):
    """A paired interval of a margin: [+.09, +.12]."""
    return "" if not c else f"[{_m(c[0], tex)}, {_m(c[1], tex)}]"


def best_free(cell):
    """Which of V-Read (primary fit) and the best baseline holds the row's best value, at shown precision.

    V-Read fitted on COCO is a separate reader and is never part of this choice.
    """
    b = cell["band"]
    shown = {k: round(b[k], 2) for k in (LEAD, cell["best"]) if k and k in b}
    if not shown:
        return set()
    top = max(shown.values())
    return {k for k, v in shown.items() if v == top}


def earlier(cell):
    """Which of V-Read and the earliest baseline reaches 0.75 for good first (both if level, at shown precision)."""
    v, e = cell["det"].get(LEAD), cell["det"].get(cell["early"]) if cell["early"] else None
    if v is None or e is None:
        return set()
    rv, re_ = (round(x, 2) if np.isfinite(x) else 9.0 for x in (v, e))
    if rv >= 9 and re_ >= 9:
        return set()
    return {LEAD, "early"} if rv == re_ else {LEAD} if rv < re_ else {"early"}


def underpowered(c):
    """Why a cell's AUROC rests on little evidence ('' if it does not; a thin cell carries its own mark)."""
    if c.get("thin"):
        return ""
    why = []
    if c.get("minority") is not None and c["minority"] < MIN_MINORITY:
        why.append(f"{c['minority']} of the rarer answer in {c['n']}")
    if c.get("undecided") is not None and c["undecided"] > UNDECIDED:
        why.append(f"{c['undecided']:.0%} of the model's own final margins under {MARGIN_NATS:g} nats")
    return "; ".join(why)


def _flag_note(sub, tex_=True):
    got = [(m, f, underpowered(c)) for m, cs in sub.items() for f, c in cs.items() if underpowered(c)]
    if not got:
        return ""
    mark = "$^\\ddagger$" if tex_ else "u"
    return (f"{mark}: underpowered, shown but to be read with care (rarer answer under {MIN_MINORITY} records, or "
            f"most of the model's own answers near a coin-flip): "
            + "; ".join(f"{_cellname(m, f)}, {why.replace('%', chr(92) + '%') if tex_ else why}"
                        for m, f, why in got) + ".")


#: which table holds a result, and its label
HOME = {"win": "tab:vread_main", "level": "tab:vread_main", "behind": "tab:vread_appendix",
        "untested": "tab:vread_all", "thin": "tab:vread_all"}
DAG = "$^\\dagger$"


def _res(c, tex_=True):
    """A cell's result as printed: a thin cell is not judged."""
    return ("not judged" + (DAG if tex_ else "†")) if c["result"] == "thin" else c["result"]


def split_tables(table):
    main = {m: {f: c for f, c in cs.items() if c["result"] in ("win", "level")} for m, cs in table.items()}
    app = {m: {f: c for f, c in cs.items() if c["result"] == "behind"} for m, cs in table.items()}
    return main, app


def findings(table):
    """The captions' numbers over all cells, from the primary fit and the paired intervals."""
    cells = [(m, f, c) for m, cs in table.items() for f, c in cs.items()
             if c["best"] and LEAD in c["band"] and not c.get("thin")]
    by = {r: [x for x in cells if x[2]["result"] == r] for r in RESULT}
    by_model = {m: {r: sum(mm == m for mm, _f, _c in by[r]) for r in RESULT} for m in table}
    shown = by["win"] + by["level"]
    det = [x for x in shown if x[2]["early"] and x[2]["det"].get(LEAD) is not None]
    coco = [x for x in cells if x[2]["margin_coco"] is not None]
    return dict(
        n=len(cells), by=by, by_model=by_model,
        wins=sorted(by["win"], key=lambda x: -x[2]["margin"]),
        behind=sorted(by["behind"], key=lambda x: x[2]["margin"]),
        point_ok=sum(c["margin"] >= 0 for _m, _f, c in cells),
        coco_n=len(coco), coco_ahead=sum(c["margin_coco"] > 0 for _m, _f, c in coco),
        det=len(det), det_win=sum(earlier(c) == {LEAD} for _m, _f, c in det),
        det_lvl=sum(earlier(c) == {LEAD, "early"} for _m, _f, c in det),
        weak={r: sum(bool(underpowered(c)) for _m, _f, c in by[r]) for r in RESULT})


def _cellname(m, f):
    return f"{label_of(f)}, {SHORT[m]}"


def _readers_on(table, k):
    return [SHORT[m] for m, cs in table.items() if any(k in c["band"] for c in cs.values())]


def _by_model(table, F, tex=True):
    """Per model: wins, levels, behind (and untested), and which baselines it is compared against."""
    out = []
    for m, r in F["by_model"].items():
        s = f"{SHORT[m]}: {r['win']} win, {r['level']} level, {r['behind']} behind"
        if r["untested"]:
            s += f", {r['untested']} untested"
        ag = against(table, m)
        out.append(s + (f" ({ag})" if ag else ""))
    return "; ".join(out)


def _cellv(m, f, c, tex=True):
    return f"{_cellname(m, f)} {_m(c['margin'], tex)} {_mci(c['margin_ci'], tex)} against {_the(c['best'])}"


def _common_notes(table, notes, cost=None):
    from vlens.lens.baselines import PAPER as TP
    n_rec = {m: sum(c["n"] for c in cs.values()) for m, cs in table.items()}
    dropped = "; ".join(f"{SHORT[m]}: {', '.join(label_of(f) for f in n['dropped'])}"
                        for m, n in notes.items() if n["dropped"])
    ps = ", ".join(f"{SHORT[m]} {'plain-text' if n['patchscope'] == 'patchscope_raw' else 'chat-template'}"
                   for m, n in notes.items() if n["patchscope"])
    dd = f"$d_{{{str(THR)[1:]}}}$"
    tl, jl = _readers_on(table, "tuned"), _readers_on(table, "jacobian")
    ag = "; ".join(f"{SHORT[m]} is compared {against(table, m)}" for m in table if against(table, m))
    no_rec = sorted({SHORT[m] for m, cs in table.items() for c in cs.values()
                     if c["src"] == "refusal" and c.get("margin_ci") is None})
    return (
        f"AUROC: how well a reader's reading of the answer-spot state, through the model's own final norm and "
        f"unembedding, predicts the answer the model gives (yes/no, test/real, and for letter questions whether "
        f"the picked letter is the correct one, which mixes the model's answer with the ground truth; for refusal "
        f"rows, on MM-SafetyBench scenarios with the benchmark's typographic carrier and MOSSBench, the reply "
        f"refusing or the prompt being harmful), averaged, for every reader, over the blocks V-Read is fitted at "
        f"within normalised depth {BAND[0]}--{BAND[1]}, a window set after the first depth curves were seen "
        f"(Table~\\ref{{tab:vread_window}} gives others). Small type: a 95\\% bootstrap interval, pictures (the "
        f"source photograph of an edit, else the record's group) resampled {REPS} times, the same resample for "
        f"every reader and block"
        + (f"; refusal rows of {', '.join(no_rec)} predate the stored per-record readings and carry no interval"
           if no_rec else "; refusal rows resample MM-SafetyBench and MOSSBench pictures the same way")
        + ". Columns "
        f"per model: $n$ scored records; V-Read, an affine map per block fitted without labels in closed form "
        f"(least squares; the ridge penalty settles at the bottom of its grid in nearly every block) on "
        f"unlabelled SimonSays prompts, the primary fit and the only one a cell is judged by; V-Read fitted on "
        f"unlabelled COCO photographs, a separate reader shown for reference, never used to choose a winner or "
        f"count a cell (no baseline has a COCO-fitted counterpart); the best label-free baseline, each at its "
        f"paper protocol: TL tuned lens (Belrose et al.\\ 2023, the published recipe in every cell: "
        f"{TP['steps']} Nesterov-SGD steps of {TP['tokens_per_step']:,} tokens, lr\\_scale {TP['lr']:g}, weight "
        f"decay {TP['weight_decay']:g}; fitted on {', '.join(tl) or 'no model yet'}), JL Jacobian lens "
        f"(Anthropic 2026; fitted on {', '.join(jl) or 'no model yet'}), PS Patchscopes (Ghandeharioun et al.\\ "
        f"2024, in whichever of its two target prompts reads the answer better on that model: {ps}, a choice "
        f"that can only favour it), LL logit lens"
        + (f" ({ag})" if ag else "")
        + f"; the best baseline of a row is the maximum over the baselines present, chosen on the test data "
        f"itself, which favours the baselines; ``margin'', V-Read minus that best baseline, with its paired "
        f"95\\% interval (pictures resampled, the best baseline retaken inside each resample); ``result'': win "
        f"when the whole interval is above 0, behind when it is below 0, level otherwise, untested without an "
        f"interval; the labelled probe (logistic per block, fitted on every family of the task pooled, five folds "
        f"grouped by picture across the pool, penalty and a PCA width in 16, 64, 256 or all chosen inside each "
        f"training fold, each family scored on its own held-out records), a supervised reference, not a competitor and not an upper "
        f"bound (grey italics); and {dd}, the earliest depth from which AUROC stays at or above {THR} to the "
        f"output, for V-Read and the earliest baseline, every reader read on V-Read's blocks. Shaded: V-Read. "
        f"Underlined: the better of V-Read and the best baseline in the model's block for the row (ties at two "
        f"decimals both underlined) and the earlier {dd}; * marks a V-Read value that is the better; {dd} and "
        f"the underlines are point estimates; nothing is bold. ``never'': the reader ends below {THR}. --: not "
        f"run. No correction is made for comparing many cells. Records any lens was fitted on (and photographs "
        f"in the COCO fit) are dropped for every reader, and all readers in a cell share its records. Records: "
        + ", ".join(f"{SHORT[m]} {n_rec[m]:,}" for m in table)
        + ". Password-locking families are excluded"
        + (f"; $^\\dagger$ fewer than {B.MIN_CLASS} of one answer (or under {B.MIN_N} records): AUROC shown, not "
           f"used for win/level/behind counts ({dropped}); n/a$^\\dagger$: no record of one answer, so AUROC is "
           f"undefined" if dropped else "")
        + (f"; \\textit{{running}}: a run or fit that has not landed (" + "; ".join(
            f"{SHORT[m]} {' and '.join(_the(k) for k in running_readers(cost, m))}"
            for m in table if running_readers(cost, m)) + "; a missing probe or a missing model block likewise)"
           if any(running_readers(cost, m) for m in table) else
           "; \\textit{running}: a run that has not landed (a missing probe or model block)")
        + "; \\textit{not read}: the model's behaviour pass stored no answer-spot states for that family"
        + ". Efficiency, for the row's best baseline on that model: ``fit, best/V-Read'', its wall-clock fit "
        "time over V-Read's (``none'': Patchscopes and the logit lens are not fitted; ``running'': the fit has "
        "not landed); ``read cost, best'', what it costs at read time beyond the model's own final norm and "
        "unembedding, which every reader uses: matmul, " + READ_LONG["matmul"] + " (V-Read's cost too); fwd "
        "pass/block, " + READ_LONG["fwd pass/block"] + " (Patchscopes); none, the logit lens. The last row "
        "of each model gives the fit times; no clock includes the model's forward passes over the fit corpus, "
        "which every fitted lens needs. Fit cost in full: Table~\\ref{tab:vread_cost}.")


def caption_main(table, notes, F, cost=None):
    by = F["by"]
    wins = "; ".join(_cellv(*x) for x in F["wins"]) or "none"
    cap = (
        f"MAIN. The cells where V-Read, a label-free affine readout fitted in closed form, is not measurably "
        f"behind the best label-free baseline (tuned lens, Jacobian lens, Patchscopes, logit lens, where each "
        f"was read) at reading the answer a VLM is about to give from the middle of its depth; in most of them "
        f"the two are level, and V-Read is clearly ahead in only {len(by['win'])} of {F['n']}. Judged by the "
        f"paired 95\\% "
        f"interval of V-Read's margin over the best baseline, pictures resampled, over all {F['n']} "
        f"task--model cells: V-Read is ahead in {len(by['win'])} ({wins}), level in {len(by['level'])} "
        f"(the interval includes 0) and behind in {len(by['behind'])} (Table~\\ref{{tab:vread_appendix}})"
        + (f"; {len(by['untested'])} refusal cells (runs without per-record readings) have no paired interval and are in "
           f"Table~\\ref{{tab:vread_all}} only" if by["untested"] else "")
        + f". By model: {_by_model(table, F)}. This table shows the win and level cells; "
        f"Table~\\ref{{tab:vread_all}} shows every cell in one table. As point estimates V-Read is at or above "
        f"the best baseline in {F['point_ok']} of {F['n']}. V-Read fitted on COCO is reported as a separate "
        f"reader and never used to choose a winner; it is above the best baseline as a point estimate in "
        f"{F['coco_ahead']} of {F['coco_n']} cells. V-Read reaches AUROC {THR} for good earlier in depth than "
        f"every baseline in {F['det_win']} of the {F['det']} cells here (level in {F['det_lvl']}), a point "
        f"estimate. Limits: the best baseline of a row is chosen on the test data, which favours the baselines; "
        f"the depth window was set after the first curves were seen (Table~\\ref{{tab:vread_window}}); no "
        f"correction is made for comparing many cells"
        + (f"; {F['weak']['win'] + F['weak']['level']} of the cells here rest on little evidence "
           f"($^\\ddagger$, listed below)" if F["weak"]["win"] + F["weak"]["level"] else "") + ". "
        + eff_headline(cost, F, list(table)))
    return cap, _common_notes(table, notes, cost)


def caption_app(table, notes, F, cost=None):
    worst = "; ".join(_cellv(*x) for x in F["behind"]) or "none"
    cap = (
        f"APPENDIX. Where V-Read trails: the {len(F['by']['behind'])} of {F['n']} task--model cells in which the "
        f"paired 95\\% interval of V-Read's margin over the best label-free baseline lies wholly below 0: "
        f"{worst}. By model: {_by_model(table, F)}. Cells in Table~\\ref{{tab:vread_main}} are marked here; "
        f"Table~\\ref{{tab:vread_all}} shows every cell in one table. Same columns and conventions as "
        f"Table~\\ref{{tab:vread_main}}"
        + (f"; {F['weak']['behind']} of the cells here rest on little evidence ($^\\ddagger$, listed below)"
           if F["weak"]["behind"] else "") + ". " + eff_headline(cost, F, list(table)))
    return cap, _common_notes(table, notes, cost)


def caption_all(table, notes, F, cost=None):
    by = F["by"]
    cap = (
        f"APPENDIX. Every task--model cell in one table, whatever its result, so that nothing is selected by "
        f"outcome: {F['n']} cells, V-Read ahead in {len(by['win'])}, level in {len(by['level'])}, behind in "
        f"{len(by['behind'])}"
        + (f", untested in {len(by['untested'])} (refusal runs without stored per-record readings)"
           if by["untested"] else "")
        + f", by the paired 95\\% interval of its margin over the best baseline. By model: "
        f"{_by_model(table, F)}. Same columns and conventions as Table~\\ref{{tab:vread_main}}"
        + (f"; {sum(F['weak'].values())} cells rest on little evidence ($^\\ddagger$, listed below)"
           if sum(F["weak"].values()) else "") + ". " + eff_headline(cost, F, list(table)))
    return cap, _common_notes(table, notes, cost)


def _preamble(label, tag):
    return [f"% {tag}. Written by tools/make_vread_table.py; do not edit by hand.",
            "% needs \\usepackage{booktabs}, \\usepackage{graphicx} and \\usepackage{colortbl}",
            "% (or \\usepackage[table]{xcolor}) for \\cellcolor",
            "\\begin{table*}[t]", "\\centering", "\\scriptsize",
            "\\definecolor{vreadbg}{HTML}{FBEADF}",
            "\\setlength{\\tabcolsep}{2.6pt}",
            "\\providecommand{\\vrc}{\\cellcolor{vreadbg}}",
            "\\providecommand{\\vrci}[1]{{\\tiny\\,#1}}",
            "\\providecommand{\\vrcl}[1]{\\textcolor{gray}{\\textit{#1}}}"]


NCOL = 11


def _summary_rows(cost, models, ncol):
    """The per-model fit-cost line under each model's block of columns."""
    return ["\\midrule", "\\emph{fit cost} & " + " & ".join(
        f"\\multicolumn{{{ncol}}}{{l}}{{{eff_summary(cost, m)}}}" for m in models) + " \\\\"]


def tex(sub, full, models, notes_cap, label, tag, cost=None):
    """One AUROC table; ``sub`` holds its cells, ``full`` every cell (a cell held elsewhere points to its table)."""
    cap, tnotes = notes_cap
    rows, _ = ordered_rows({m: cs for m, cs in sub.items() if cs} or sub)
    colspec = "l" + ("r" + "c" * (NCOL - 1)) * len(models)
    L = _preamble(label, tag) + [f"\\caption{{{cap}}}", f"\\label{{{label}}}", "\\smallskip",
                                 "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{" + colspec + "}", "\\toprule"]
    L.append(" & " + " & ".join(f"\\multicolumn{{{NCOL}}}{{c}}{{{SHORT[m]}}}" for m in models) + " \\\\")
    L.append(" ".join(f"\\cmidrule(lr){{{2 + NCOL * i}-{1 + NCOL * (i + 1)}}}" for i in range(len(models))))
    L.append(" & " + " & ".join(
        "& \\multicolumn{6}{c}{AUROC, depth " + f"{BAND[0]}--{BAND[1]}" + "} & \\multicolumn{2}{c}{efficiency} "
        "& \\multicolumn{2}{c}{$d_{" + str(THR)[1:] + "}$}" for _ in models) + " \\\\")
    L.append(" ".join(f"\\cmidrule(lr){{{3 + NCOL * i}-{8 + NCOL * i}}} \\cmidrule(lr){{{9 + NCOL * i}-{10 + NCOL * i}}} "
                      f"\\cmidrule(lr){{{11 + NCOL * i}-{12 + NCOL * i}}}" for i in range(len(models))))
    L.append("task & " + " & ".join(
        "$n$ & \\vrc V-Read & V-Read (COCO) & best baseline & margin & result & \\vrcl{probe} "
        "& fit, best/V-Read & read cost, best & \\vrc V-Read & baseline" for _ in models) + " \\\\")
    for g, fams, _s in rows:
        L.append("\\midrule")
        L.append(f"\\multicolumn{{{1 + NCOL * len(models)}}}{{l}}{{\\emph{{{GROUPS[g]}}}}} \\\\")
        for f in fams:
            cells = []
            for m in models:
                c = sub.get(m, {}).get(f)
                if c is None:
                    o = full.get(m, {}).get(f)
                    if o is not None:
                        cells.append(f"\\multicolumn{{{NCOL}}}{{c}}{{\\textit{{{_res(o)}, "
                                     f"Table~\\ref{{{HOME[o['result']]}}}}}}}")
                    else:
                        cells.append(f"\\multicolumn{{{NCOL}}}{{c}}{{{_absent(m, f)}}}")
                    continue
                b, bf, ea = c["band"], best_free(c), earlier(c)
                if c.get("na"):
                    na = "n/a" + DAG
                    cells += [f"{c['n']}", "\\vrc " + na, na, na, na, _res(c), "\\vrcl{" + na + "}", "--", "--",
                              "\\vrc " + na, na]
                    continue
                mk = DAG if c.get("thin") else ""

                def val(k, ci=False):
                    if k is None or k not in b:
                        return "--"
                    s = _a(b[k]) + mk
                    s = f"\\underline{{{s}}}" if k in bf else s
                    s += "*" if (k == LEAD and k in bf) else ""
                    return s + (f"\\vrci{{{_ci(c['ci'].get(k))}}}" if ci and c["ci"].get(k) else "")
                bb = c["best"]
                base = (val(bb, True) + f"\\,{{\\tiny {ABBR[bb]}}}") if bb else "--"
                mg = _m(c["margin"], True) + (f"\\vrci{{{_mci(c['margin_ci'], True)}}}" if c["margin_ci"] else "")
                cl = (f"\\vrcl{{{_a(b[c['ceil']]) + mk}}}" if c["ceil"] else
                      "\\vrcl{n/a" + DAG + "}" if mk else "\\vrcl{\\textit{running}}")
                dv = _d(c["det"].get(LEAD)) + mk
                dv = f"\\underline{{{dv}}}" if LEAD in ea else dv
                de = (_d(c["det"][c["early"]]) + mk if c["early"] else "--")
                de = f"\\underline{{{de}}}" if "early" in ea else de
                if c["early"]:
                    de += f"\\,{{\\tiny {ABBR[c['early']]}}}"
                cells += [f"{c['n']}" + ("$^\\ddagger$" if underpowered(c) else ""),
                          "\\vrc " + val(LEAD, True), val("v_read_coco", True),
                          base, mg, _res(c), cl, eff_ratio(cost, m, bb),
                          READ_COST.get(bb, "--") if bb else "--", "\\vrc " + dv, de]
            lab = label_of(f).replace("&", "\\&")
            L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    L += _summary_rows(cost, models, NCOL)
    L += ["\\bottomrule", "\\end{tabular}}", "\\par\\smallskip",
          f"\\parbox{{\\textwidth}}{{\\tiny {tnotes} {_flag_note(sub)}}}", "\\end{table*}"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ depth-window sensitivity

def _wname(w):
    return "0--1 (every fitted block)" if tuple(w) == (0.0, 1.0) else f"{w[0]:g}--{w[1]:g}"


def window_counts(table):
    """{window: {model: {result: count}}} over the cells with a best baseline."""
    out = {}
    for w in WINDOWS:
        out[w] = {m: {r: sum(c["windows"][w]["result"] == r for c in cs.values()
                              if c["windows"][w].get("best") and LEAD in c["windows"][w]["band"])
                      for r in RESULT} for m, cs in table.items()}
    return out


def caption_window(table, cost=None, F=None):
    WC = window_counts(table)

    def cnt(w):
        return "; ".join(f"{SHORT[m]} {r['win']} win, {r['level']} level, {r['behind']} behind"
                         + (f" ({against(table, m)})" if against(table, m) else "")
                         for m, r in WC[w].items())
    return (
        "APPENDIX. How V-Read's margin over the best label-free baseline depends on the depth window it is "
        "averaged over. Per task, model and window: V-Read (the primary fit) minus the best baseline in that "
        "window, the mean over V-Read's fitted blocks inside it, the same blocks for every reader, with its "
        f"paired 95\\% interval (pictures resampled {REPS} times, the best baseline retaken inside each "
        "resample) and the result (w win, the interval above 0; l level; b behind, the interval below 0; u "
        "untested: refusal runs without per-record readings, point estimates only). Windows: "
        + ", ".join(_wname(w) for w in WINDOWS)
        + f". The main tables use {BAND[0]:g}--{BAND[1]:g}, which was set after the first depth curves were "
        "seen and is not a pre-registered choice; the whole depth (every fitted block) is the one choice that "
        "does not depend on the data. Counts by window: "
        + ". ".join(f"{_wname(w)}: {cnt(w)}" for w in WINDOWS)
        + ". The best baseline of each window is chosen on the test data itself, which favours the baselines. "
        + (eff_headline(cost, F, list(table), f"depth {BAND[0]:g}--{BAND[1]:g}") if F else "")
        + " Each cell also names, in brackets, the window's best baseline (TL, JL, PS, LL as in "
        "Table~\\ref{tab:vread_main}), its wall-clock fit time over V-Read's on that model (no fit: Patchscopes "
        "and the logit lens) and its read cost: mm, one matrix multiply per block read (V-Read's cost too); fp, "
        "one extra forward pass of the model per block read, per record; 0, nothing beyond the model's own final "
        "norm and unembedding. The last row gives each model's fit times.")


def tex_window(table, models, cost=None, F=None):
    L = [x.replace("\\begin{table*}[t]", "\\begin{sidewaystable*}") for x in _preamble("tab:vread_window", "APPENDIX")]
    L.insert(1, "% landscape: needs \\usepackage{rotating} for sidewaystable*")
    nw = len(WINDOWS)
    L += [f"\\caption{{{caption_window(table, cost, F)}}}", "\\label{tab:vread_window}", "\\smallskip",
          "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{l" + ("c" * nw) * len(models) + "}", "\\toprule",
          " & " + " & ".join(f"\\multicolumn{{{nw}}}{{c}}{{{SHORT[m]}}}" for m in models) + " \\\\",
          " ".join(f"\\cmidrule(lr){{{2 + nw * i}-{1 + nw * (i + 1)}}}" for i in range(len(models))),
          "task & " + " & ".join(" & ".join(_wname(w) + ("$^\\dagger$" if tuple(w) == tuple(BAND) else "")
                                            for w in WINDOWS) for _ in models) + " \\\\"]
    rows, _ = ordered_rows(table)
    for g, fams, _s in rows:
        L.append("\\midrule")
        L.append(f"\\multicolumn{{{1 + nw * len(models)}}}{{l}}{{\\emph{{{GROUPS[g]}}}}} \\\\")
        for f in fams:
            cells = []
            for m in models:
                c = table.get(m, {}).get(f)
                if c is None:
                    cells.append(f"\\multicolumn{{{nw}}}{{c}}{{{_absent(m, f)}}}")
                    continue
                for w in WINDOWS:
                    x = (c or {}).get("windows", {}).get(w)
                    if c.get("na"):
                        cells.append("n/a" + DAG)
                        continue
                    if not x or x.get("margin") is None:
                        cells.append("--")
                        continue
                    cells.append(_m(x["margin"], True) + (f"\\vrci{{{_mci(x['margin_ci'], True)}}}"
                                                          if x["margin_ci"] else "")
                                 + f"\\,{{\\tiny {DAG if x['result'] == 'thin' else x['result'][0]}}}"
                                 + _weff(cost, m, x.get("best"), True))
            L.append(label_of(f).replace("&", "\\&") + " & " + " & ".join(cells) + " \\\\")
    L += _summary_rows(cost, models, nw)
    L += ["\\bottomrule", "\\end{tabular}}", "\\par\\smallskip",
          "\\parbox{\\textwidth}{\\tiny $^\\dagger$: the window of the main tables, set after the first "
          "curves were seen. Every other convention as in Table~\\ref{tab:vread_main}.}", "\\end{sidewaystable*}"]
    return "\n".join(L) + "\n"


def _weff(cost, m, bb, tex_=False):
    """A window cell's efficiency tag: best baseline, its fit time over V-Read's, its read cost."""
    if not bb:
        return ""
    r = eff_ratio(cost, m, bb)
    t = f"{ABBR[bb]} {'no fit' if r == 'none' else r}, {READ_SHORT[READ_COST[bb]]}"
    return f"\\,{{\\tiny ({t})}}" if tex_ else f" ({t})"


def plain_window(table, models, cost=None):
    WC = window_counts(table)
    out = ["WINDOW SENSITIVITY: V-Read (primary fit) minus best baseline [paired CI] result, per window"]
    rows, _ = ordered_rows(table)
    for m in models:
        out.append(f"=== {SHORT[m]}  " + "  ".join(
            f"{_wname(w)}: {WC[w][m]['win']}w/{WC[w][m]['level']}l/{WC[w][m]['behind']}b" for w in WINDOWS))
        for g, fams, _s in rows:
            for f in fams:
                c = table.get(m, {}).get(f)
                if c is None:
                    continue
                parts = []
                for w in WINDOWS:
                    x = c["windows"][w]
                    parts.append("n/a†" if c.get("na") else "--" if x.get("margin") is None else
                                 f"{_m(x['margin'])} {_mci(x['margin_ci'])} {'†' if x['result'] == 'thin' else x['result'][0]}"
                                 + _weff(cost, m, x.get("best")))
                out.append(f"  {label_of(f)[:46]:<46} " + " | ".join(f"{p:<50}" for p in parts))
        out.append("  " + eff_summary(cost, m))
    return "\n".join(out)


def _gains(cost):
    """The per-block gains V-Read's primary fit chose, per model, as the caption states them."""
    out = []
    for m, r in cost.items():
        g = ((r or {}).get("v_read") or {}).get("gain")
        try:
            g = ast.literal_eval(g) if isinstance(g, str) else g
            vals = [float(x) for x in (g or {}).values()]
        except (ValueError, SyntaxError):
            vals = []
        if vals:
            out.append(f"{SHORT.get(m, m)} {min(vals):g} to {max(vals):g}")
    return ("V-Read's chosen per-block gains: " + "; ".join(out) + " (1 is no gain; how much of its AUROC the "
            "gain carries is not measured here). ") if out else ""


def caption_cost(cost):
    C = cost_claims(cost)
    jpaper = next((r["jacobian"]["paper"] for r in cost.values()
                   if r and r["jacobian"] and r["jacobian"].get("paper")), "Anthropic 2026")
    pending = [SHORT.get(m, m) for m, r in cost.items() if r and (r["tuned"] is None or r["jacobian"] is None)]
    nometa = [SHORT.get(m, m) for m, r in cost.items() if r is None]
    if C:
        m, v, t, j = C["m"], C["v"], C["t"], C["j"]
        lead = (
            f"APPENDIX. V-Read is a single closed-form least-squares solve per block, so it fits in minutes where the published "
            f"baselines take hours: on {SHORT[m]} it fits in {v['seconds']:,} s from one pass over "
            f"{v['tokens']:,} tokens, against {t['seconds']:,} s and {C['steps']} steps x {C['tps']:,} tokens for "
            f"the tuned lens ({_x(C['xt'])} the time, {_x(C['xd'])} the tokens processed) and {j['seconds']:,} s "
            f"for the Jacobian lens ({_x(C['xj'])} the time). ")
        lim = (
            f"Limits: the tuned lens is fitted on the same {cost[m]['spec'][3]} SimonSays prompts as V-Read and "
            f"reaches its {C['tot']:,} tokens by replaying them about {C['passes']:.0f} times, so the token ratio is "
            f"compute, not extra data ({t.get('tokens_spent', C['tot']):,} tokens actually processed, the last batch "
            f"of each step overrunning); the Jacobian lens uses fewer prompts and positions but needs backward passes "
            f"through the model; the tuned lens stays closer to the model's next-token distribution (development-set "
            f"KL {t['heldout_kl']:.2f} against {v['heldout_kl']:.2f} nats for V-Read, "
            f"{j['heldout_kl']:.2f} for the Jacobian lens), even though V-Read's ridge penalty and per-block gain "
            f"were chosen on that same development set and the tuned lens's training settings were not, which "
            f"makes V-Read's KL optimistic. ")
    else:
        lead = "APPENDIX. V-Read is a single closed-form least-squares solve per block. "
        lim = ""
    body = (
        "Shown per model (exact model ids): wall-clock fit time, the corpus each reader is fitted on and what it "
        "processes, both as ratios to V-Read, and the development-set KL: the KL from each lens's reading to the "
        "model's own next-token distribution on the development prompts, the same prompts V-Read's ridge penalty "
        "and per-block gain are chosen on (and the baselines' per-block gain), so it is not a held-out measure and "
        "favours V-Read; for the COCO fit, its own COCO development photographs, so its KL is not comparable. "
        + _gains(cost) +
        "These are the fits every cell of Tables~\\ref{tab:vread_main}, \\ref{tab:vread_appendix} and "
        "\\ref{tab:vread_all} reads, each "
        "baseline at its paper protocol (tuned lens: the published recipe, Belrose et al.\\ 2023; "
        f"Jacobian lens: {jpaper}). ")
    hw = (
        f"All fits ran on {HARDWARE}, so times carry the sharing. Each clock covers that lens's own fit: V-Read's "
        "includes choosing its ridge penalty and per-block gain, the baselines' exclude their gain choice. The "
        "corpus states were computed once and held in memory for every lens (26.7 GiB on Qwen3-VL-8B-Instruct), so "
        "no clock includes the model's forward passes.")
    tail = ""
    if pending:
        tail += f" running: the fit is under way on {', '.join(pending)} and is added when it lands."
    if nometa:
        tail += f" No fit record yet for {', '.join(nometa)}."
    return lead + body + lim + hw + tail


def tex_cost(cost):
    R = cost_rows(cost)
    L = _preamble("tab:vread_cost", "APPENDIX")
    L += [f"\\caption{{{caption_cost(cost)}}}", "\\label{tab:vread_cost}", "\\smallskip",
          "\\begin{tabular}{llllrrr}", "\\toprule",
          " & & & & \\multicolumn{2}{c}{vs V-Read} & \\\\",
          "\\cmidrule(lr){5-6}",
          "reader & fitted on & processed & wall-clock & time & data & dev.-set KL, nats \\\\"]
    for m, rows in R.items():
        L.append("\\midrule")
        L.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{m}}}}} \\\\")
        if rows is None:
            L.append("\\multicolumn{7}{l}{no fit record yet} \\\\")
            continue
        for r in rows:
            sh = "\\vrc " if r["k"] in OURS else ""
            if r["running"]:
                L.append(f"{sh}{r['name']} & \\multicolumn{{6}}{{l}}{{\\textit{{running}}}} \\\\")
                continue
            L.append(" & ".join([sh + r["name"], r["corpus"], r["data"].replace(",", "{,}"),
                                 _secs(r["sec"]).replace(",", "{,}"),
                                 "ref." if r["k"] == "v_read" else _x(r["xt"]),
                                 "ref." if r["k"] == "v_read" else _x(r["xd"]),
                                 "--" if r["kl"] is None else f"{r['kl']:.2f}"]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(L).replace("table*}[t]", "table*}[t]") + "\n"


def plain_cost(cost):
    R = cost_rows(cost)
    out = [f"=== fit cost (results/fits; {HARDWARE})",
           f"  {'reader':<22} {'fitted on':<28} {'processed':<32} {'wall-clock':<22} {'x time':>7} {'x data':>7} {'devKL':>6}"]
    for m, rows in R.items():
        out.append(f"-- {m}")
        if rows is None:
            out.append("  no fit record yet")
            continue
        for r in rows:
            if r["running"]:
                out.append(f"  {r['name']:<22} running")
                continue
            ref = r["k"] == "v_read"
            out.append(f"  {r['name']:<22} {r['corpus']:<28} {r['data']:<32} {_secs(r['sec']):<22} "
                       f"{'ref.' if ref else _x(r['xt']):>7} {'ref.' if ref else _x(r['xd']):>7} "
                       f"{'--' if r['kl'] is None else format(r['kl'], '.2f'):>6}")
            if r.get("spent"):
                out.append(f"  {'':<22} ({r['spent']:,} tokens actually processed)")
        rr = cost[m]
        if rr:
            out.append(f"  caches: {rr['caches']} (chosen by {rr['how']})")
    return "\n".join(out)



def plain(sub, full, models, title, cost=None):
    out = [f"{title}. Cell: n | V-Read [CI] | V-Read COCO [CI] | best baseline [CI] (name) | margin [paired CI] | "
           f"result | probe | fit best/V-Read | read cost of best | d{THR} V-Read | d{THR} earliest baseline (name).  AUROC = mean over depth "
           f"{BAND[0]}-{BAND[1]} on V-Read's blocks; CIs resample pictures; * = V-Read the better / earlier d"]
    rows, _ = ordered_rows({m: cs for m, cs in sub.items() if cs} or sub)
    for m in models:
        out.append("")
        ag = against(full, m)
        rr = running_readers(cost, m)
        ag = "; ".join(x for x in (ag, (" and ".join(_the(k) for k in rr) + " running") if rr else "") if x)
        out.append(f"=== {SHORT[m]}" + (f"  ({ag})" if ag else ""))
        out.append(f"{'task':<50}{'n':>6} {'V-Read':>18} {'V-Read COCO':>18} {'best baseline':>24} "
                   f"{'margin [paired CI]':>24} {'result':>9} {'probe':>6} {'fit':>8} {'read':>15} {'d V-Read':>9} "
                   f"{'d baseline':>12}")
        for g, fams, s in rows:
            out.append(f"-- {GROUPS[g]}")
            for f in fams:
                c = sub.get(m, {}).get(f)
                if c is None:
                    o = full.get(m, {}).get(f)
                    out.append(f"  {label_of(f)[:48]:<48}{'(' + _res(o, False) + ', elsewhere)' if o else _absent(m, f, False):>24}")
                    continue
                b, bf, ea = c["band"], best_free(c), earlier(c)
                if c.get("na"):
                    out.append(f"  {label_of(f)[:48]:<48}{c['n']:>6} {'n/a†':>18} {'n/a†':>18} {'n/a†':>24} "
                               f"{'n/a†':>24} {_res(c, False):>9} {'n/a†':>6}")
                    continue
                mk = "†" if c.get("thin") else ""

                def v(k, ci=False):
                    if k is None or k not in b:
                        return "--"
                    s_ = _a(b[k]) + mk + ("*" if (k == LEAD and k in bf) else "")
                    return s_ + (f" [{_ci(c['ci'].get(k)).replace('--', '-')}]" if ci and c["ci"].get(k) else "")
                bb = c["best"]
                base = f"{v(bb, True)} ({ABBR[bb]})" if bb else "--"
                mg = f"{_m(c['margin'])} {_mci(c['margin_ci'])}"
                dv = _d(c["det"].get(LEAD)) + mk + ("*" if LEAD in ea else "")
                de = (_d(c["det"][c["early"]]) + mk + ("*" if "early" in ea else "") + f" ({ABBR[c['early']]})"
                      if c["early"] else "--")
                cl = _a(b[c["ceil"]]) + mk if c["ceil"] else "n/a†" if mk else "running"
                nn = f"{c['n']}{'u' if underpowered(c) else ''}"
                out.append(f"  {label_of(f)[:48]:<48}{nn:>6} {v(LEAD, True):>18} {v('v_read_coco', True):>18} "
                           f"{base:>24} {mg:>24} {_res(c, False):>9} {cl:>6} {eff_ratio(cost, m, bb):>8} "
                           f"{READ_COST.get(bb, '--') if bb else '--':>15} {dv:>9} {de:>12}")
        out.append(f"  {eff_summary(cost, m)}")
    note = _flag_note(sub, tex_=False)
    if any(c.get("thin") for cs in full.values() for c in cs.values()):
        note = (note + "\n" if note else "") + (
            f"†: fewer than {B.MIN_CLASS} of one answer (or under {B.MIN_N} records): AUROC shown, not used for "
            f"win/level/behind counts; n/a†: no record of one answer, so AUROC is undefined.")
    return "\n".join(out + ([""] + [note] if note else []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="tables/vread_main.tex")
    ap.add_argument("--appendix", default=None, help="default: vread_appendix.tex beside --out")
    ap.add_argument("--all", default=None, help="default: vread_all.tex beside --out")
    ap.add_argument("--window", default=None, help="default: vread_window.tex beside --out")
    ap.add_argument("--cost", default=None, help="default: vread_cost.tex beside --out")
    a = ap.parse_args()
    here = os.path.dirname(a.out) or "."
    app_out = a.appendix or os.path.join(here, "vread_appendix.tex")
    all_out = a.all or os.path.join(here, "vread_all.tex")
    win_out = a.window or os.path.join(here, "vread_window.tex")
    cost_out = a.cost or os.path.join(here, "vread_cost.tex")
    table, notes = collect(a.results)
    if not table:
        print("no cells with V-Read")
        return 1
    models = list(table)
    cost = fit_cost(a.results, models)
    check_protocol(table, cost)
    F = findings(table)
    main_t, app_t = split_tables(table)
    print(plain(main_t, table, models, "MAIN: cells where V-Read wins or is level (paired picture bootstrap)", cost))
    print()
    print(plain(app_t, table, models, "APPENDIX: cells where V-Read is behind (paired interval below 0)", cost))
    print()
    print(plain(table, table, models, "APPENDIX: every cell", cost))
    print()
    print(plain_window(table, models, cost))
    print()
    print(plain_cost(cost))
    print()
    print(f"findings: {F['n']} cells; win {len(F['by']['win'])}, level {len(F['by']['level'])}, "
          f"behind {len(F['by']['behind'])}, untested {len(F['by']['untested'])}; by model {F['by_model']}; "
          f"point estimate at or above best baseline {F['point_ok']}/{F['n']}; COCO fit above best baseline "
          f"{F['coco_ahead']}/{F['coco_n']} (not used to judge); d{THR} earlier {F['det_win']}/{F['det']}, "
          f"level {F['det_lvl']}")
    for x in F["wins"]:
        print("  win:    " + _cellv(*x, tex=False))
    for x in F["behind"]:
        print("  behind: " + _cellv(*x, tex=False))
    for m, n in notes.items():
        print(f"  {SHORT[m]}: patchscope prompt {n['patchscope']}; dropped {n['dropped']}")
    outs = {a.out: tex(main_t, table, models, caption_main(table, notes, F, cost), "tab:vread_main", "MAIN", cost),
            app_out: tex(app_t, table, models, caption_app(table, notes, F, cost), "tab:vread_appendix", "APPENDIX",
                         cost),
            all_out: tex(table, table, models, caption_all(table, notes, F, cost), "tab:vread_all", "APPENDIX", cost),
            win_out: tex_window(table, models, cost, F),
            cost_out: tex_cost(cost)}
    for p, s in outs.items():
        assert "—" not in s and "---" not in s and "\\textbf" not in s and "\\bf " not in s, p
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        open(p, "w").write(s)
        print(f"-> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
