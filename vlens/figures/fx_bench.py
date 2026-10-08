# SPDX-License-Identifier: Apache-2.0
"""Reading the model's own answer: V-Read against the published lenses, over the whole set."""
from __future__ import annotations

import glob
import functools
import json
import os
import re

import numpy as np

from vlens.figures import paper as P
from vlens.figures import style as st

LENSES = ("v_read", "tuned", "jacobian", "logit")
#: the stretch of depth where the lenses differ: before it none reads the
#: answer, after it all of them do
BAND = (0.45, 0.8)
SRC_ORDER = ("openimages", "nuscenes", "vins", "gemini", "wmdp")


def _benches(results, model):
    out = []
    for p in sorted(glob.glob(f"{results}/lensbench/*__bench.json")):
        m = os.path.basename(p).split("__")[0]
        if model and m != model:
            continue
        out.append((m, json.load(open(p))))
    return out


def _src(fam):
    return fam.split("_")[0]


#: column heads for the per-source matrices, broken where they would be wide
HEAD = {"v_read": "V-Read", "v_read_coco": "V-Read\nfitted on\nCOCO", "tuned": "tuned\nlens",
        "jacobian": "Jacobian\nlens", "patchscope": "Patch-\nscopes", "logit": "logit\nlens",
        "latent": "Latent-\nLens", "probe": "probe\n(labelled)"}


def fits_running(results, model):
    """The fitted baselines queued but not landed on ``model``: in no entry of ``results/fits/<model>__fitmeta.json``."""
    p = f"{results}/fits/{model}__fitmeta.json"
    if not model or not os.path.exists(p):
        return []
    have = {k for e in json.load(open(p)) for k in (e.get("meta") or {})}
    return [k for k in ("tuned", "jacobian") if k not in have]


def _pending_cell(ax, x, y):
    """A cell whose run has not landed: light, hatched, no number."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="#F6F4F1", edgecolor="#CEC8BF", lw=0,
                           hatch="////"))
    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="none", edgecolor="white", lw=1.5))


def _value_matrix(ax, sources, readers, table, ceil=None, readers_down=False, pending=(), marked=()):
    """Sources against readers, each cell the value printed and shaded.

    ``table[j][k]`` is source ``j``'s value for reader ``k`` (absent: blank;
    absent and in ``pending``, a reader whose run has not landed: hatched).
    Sources whose index is in ``marked`` are under the scoring floor: their
    values carry a dagger, and an absent value reads n/a with it.
    The shade runs over the label-free readers' range, darker for higher;
    V-Read's line of cells is outlined; ``ceil``, the labelled probe, stands
    apart in greys. With ``readers_down`` the readers are the rows and the
    sources the columns, which keeps every reader's name on one line.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    warm = LinearSegmentedColormap.from_list("vm", st.WARM[:5])
    grey = LinearSegmentedColormap.from_list("vm_p", ["#F3F1EE", "#B9B3AA"])
    pos = {k: float(i) for i, k in enumerate(c for c in readers if c != ceil)}
    if ceil in readers:
        pos[ceil] = len(pos) + 0.35
    free = [row[k] for row in table for k in pos if k != ceil and k in row and np.isfinite(row[k])]
    lo, hi = (min(free), max(free)) if free else (0.0, 1.0)
    fs = st.matplotlib.rcParams["ytick.labelsize"] * 0.9
    n_r, n_s = max(pos.values()) + 1, len(sources)
    at = (lambda j, k: (float(j), pos[k])) if readers_down else (lambda j, k: (pos[k], float(j)))
    ax.set_xlim(-0.5, (n_s if readers_down else n_r) - 0.5)
    ax.set_ylim((n_r if readers_down else n_s) - 0.5, -0.5)
    ax.axis("off")
    for j, row in enumerate(table):
        for k in pos:
            v = row.get(k)
            if v is None or not np.isfinite(v):
                if k in pending:
                    _pending_cell(ax, *at(j, k))
                elif j in marked:
                    ax.text(*at(j, k), "n/a\u2020", ha="center", va="center", color=st.SUBINK, fontsize=fs)
                continue
            x, y = at(j, k)
            cm = grey if k == ceil else warm
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, lw=1.5, edgecolor="white",
                                   facecolor=cm(float(np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)))))
            ax.text(x, y, f"{v:.2f}".lstrip("0") + ("\u2020" if j in marked else ""), ha="center", va="center",
                    color=st.INK, fontsize=fs)
    if "v_read" in pos:
        p0 = pos["v_read"]
        box = ((-0.5, p0 - 0.5), n_s, 1) if readers_down else ((p0 - 0.5, -0.5), 1, n_s)
        ax.add_patch(Rectangle(*box, facecolor="none", edgecolor=st.INK, lw=1.8, zorder=6,
                               clip_on=False))
    fl = st.matplotlib.rcParams["ytick.labelsize"]
    heads = {k: HEAD.get(k, k) for k in pos}
    if readers_down:
        for k, y in pos.items():
            ax.text(-0.62, y, P.NICE.get(k, k), ha="right", va="center", color=st.INK, fontsize=fs)
        for j, lab in enumerate(sources):
            ax.text(j, -0.62, lab, ha="center", va="bottom", color=st.INK, fontsize=fl * 0.75,
                    linespacing=1.0)
    else:
        for k, x in pos.items():
            ax.text(x, -0.62, heads[k], ha="center", va="bottom", color=st.INK, fontsize=fs,
                    linespacing=1.0)
        for j, lab in enumerate(sources):
            ax.text(-0.62, j, lab, ha="right", va="center", color=st.INK, fontsize=fl)


def lens_scoreboard(results="results", out="figures", model=None):
    """MAIN. How well each lens reads the answer the model is about to give, over every family in the set.

    Scored on records none of the lenses was fitted on (the bench excludes
    every fitting record by identifier and asserts it). For each family, block
    and lens: the AUROC with which the lens's readout at the answer spot,
    through the model's own final norm and unembedding, predicts the answer
    the model then gives (yes or no, test or real, or whether it picks the
    correct letter). Left: the mean over families, weighted by records, on
    normalised depth, with V-Read in clay and the three published lenses
    (tuned lens, Jacobian lens, logit lens) in their own colours; the band
    marks the stretch of depth the right panel averages over. Right: that
    stretch, mean AUROC per source, one row per source and a mark per lens,
    with the number of records.

    Before about half depth no lens reads the answer and after about four
    fifths all of them do; between the two, V-Read reads it best on the
    whole set and on every source, most clearly on Open Images (0.96 against
    at most 0.86) and the WMDP carrier (0.79 against at most 0.75); on the
    generated scenes it is level with the Jacobian lens. The key is shared by
    both panels. V-Read is fitted without labels in about a tenth of
    the tuned lens's time; the tuned lens stays closer to the model's full
    distribution (lower KL, see the appendix), and the two measure different
    things: KL is over the whole vocabulary, AUROC only over the answer.
    """
    benches = _benches(results, model)
    if not benches or (model is None and not P.AVERAGE):
        return None
    import matplotlib.pyplot as plt
    st.use()
    grid = np.linspace(0.05, 1.0, 40)
    curve = {k: [] for k in LENSES}
    wts = []
    by_src = {s: {k: [] for k in LENSES} for s in SRC_ORDER}
    n_src = {s: 0 for s in SRC_ORDER}
    for m, d in benches:
        for fam, v in d["families"].items():
            if P.LOCKED.search(fam):
                continue
            dep = np.asarray(v["depths"], float)
            a = v["auroc"]
            if not all(f"model|{k}" in a for k in LENSES):
                continue
            wts.append(v["n"])
            s = _src(fam)
            n_src[s] = n_src.get(s, 0) + v["n"]
            inb = (dep >= BAND[0]) & (dep <= BAND[1])
            for k in LENSES:
                y = np.asarray(a[f"model|{k}"], float)
                curve[k].append(np.interp(grid, dep, y))
                if s in by_src:
                    by_src[s][k].append((v["n"], float(y[inb].mean())))
    if not wts:
        return None
    w = np.asarray(wts, float)
    fig, (ax, ay) = st.wide(1, 2, unit=(4.6, 3.1), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax.axvspan(*BAND, color=st.BAND, lw=0, zorder=0)
    for k in reversed(LENSES):
        c = np.average(np.asarray(curve[k]), axis=0, weights=w)
        ax.plot(grid, c, color=P.COLOUR[k], lw=2.6 if k == "v_read" else 1.9,
                dashes=P.DASH.get(k, (None, None)), zorder=4 if k == "v_read" else 3)
    ax.axhline(0.5, color=st.AXIS, lw=1.0, zorder=1)
    ax.set_xlim(0.05, 1.0)
    ax.set_ylim(0.3, 1.03)
    ax.set_xticks([0.05, 0.5, 1.0])
    ax.set_xticklabels(["input", "half", "output"])
    ax.set_xlabel("depth")
    ax.set_ylabel("AUROC for the answer")
    st.despine(ax)
    rows = [s for s in SRC_ORDER if by_src.get(s) and by_src[s]["v_read"]]
    table = [{k: np.average([x for _, x in by_src[s][k]], weights=[n for n, _ in by_src[s][k]])
              for k in LENSES if by_src[s][k]} for s in rows]
    _value_matrix(ay, [P.SOURCE[s].replace(" (VINS-120K)", "") for s in rows], list(LENSES), table)
    # panels placed by hand: the curves, a gutter for the source names, then
    # the matrix, its column heads in the room left above it
    fig.set_layout_engine("none")
    W, H = fig.get_size_inches()
    ax.set_position([0.62 / W, 0.62 / H, 4.3 / W, (H - 1.3) / H])
    mw = min(0.8 * len(LENSES), W - 7.0)
    ay.set_position([(W - mw - 0.05) / W, 0.62 / H, mw / W, (H - 1.3) / H])
    # the key is one row in a band above both panels, so it cannot be read as
    # more curves on the left or another row of dots on the right
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=P.COLOUR[k], lw=2.6 if k == "v_read" else 1.9,
                 dashes=P.DASH.get(k, (None, None)), marker="o",
                 ms=9 if k == "v_read" else 7.5, mec="white") for k in LENSES]
    st.key_band(fig, hs, [P.NICE[k] for k in LENSES],
                fontsize=st.matplotlib.rcParams["legend.fontsize"] * 0.9)
    return st.save(fig, P._at(out, "lens_scoreboard"), check=False, tight=True)


# ------------------------------------------------------------------ from the per-block readouts

#: every label-free reader a readout file can hold, in the order the key names them
READERS = ("v_read", "v_read_coco", "tuned", "jacobian", "patchscope", "logit")
#: the labelled probe, drawn as a supervised reference
CEIL = "probe"
#: a family is scored when it has this many labelled records, and this many of each answer
MIN_N, MIN_CLASS = 24, 5
_LETTER = re.compile(r"^\W*\(?([ABCD])\b")


def _own(b):
    """The model's own final answer as 1, 0 or None, by the rules of ``tools/probe_readouts.py``."""
    kind, reply = b.get("kind"), str(b.get("reply") or "")
    if kind == "yesno":
        s = P._said(b)
        return None if s is None else int(s == "yes")
    if kind == "test_real":
        t = reply.strip().lower()
        return 1 if t.startswith("test") else 0 if t.startswith("real") else None
    if kind == "letter":
        m, e = _LETTER.match(reply), _correct(b)
        return None if m is None or e is None else int(m.group(1) == "ABCD"[e])
    return None


def _correct(b):
    """The index of a letter record's correct letter, or None."""
    e = str(b.get("expected") or "").strip().upper()[:1]
    if e and e in "ABCD":
        return "ABCD".index(e)
    i = b.get("answer_index")
    return int(i) if i is not None and 0 <= int(i) < 4 else None


def _score(v, e):
    """A readout value as a score for the answer: a margin as it stands, four letters as the correct one's margin."""
    if v is None:
        return np.nan
    if isinstance(v, (list, tuple)):
        v = np.asarray(v, float)
        return float(v[e] - np.delete(v, e).max()) if e is not None else np.nan
    return float(v)


@functools.lru_cache(maxsize=None)
def _fit_pictures(results, model):
    """The pictures, relative to the set's root, that the shared lens fit read."""
    p = f"{results}/fits/{model}__fit_pictures.json"
    return frozenset(json.load(open(p))["pictures"]) if os.path.exists(p) else frozenset()


def _readout_family(results, model, fam, floor=True):
    """Per-block AUROC for every reader of one family, on the records no fitted lens saw; or None.

    With ``floor`` (the default) a family under ``MIN_N`` records or ``MIN_CLASS``
    of one answer is None. Without it such a family is scored on whatever
    records it has (AUROC wherever both answers occur) and comes back marked
    ``thin`` (``minority`` records of the rarer answer; with none, no AUROC at
    all), for tables and figures that show it, marked, without judging it.
    """
    from vlens.lens.metrics import auroc
    rp = f"{results}/readouts/{model}__{fam}_readouts.json"
    bp = f"{results}/simonsays/{model}__{fam}.json"
    if not (os.path.exists(rp) and os.path.exists(bp)):
        return None
    d = json.load(open(rp))
    beh = {r["id"]: r for r in json.load(open(bp))["rows"]}
    blocks = [int(b) for b in d["blocks"]]
    col = {b: j for j, b in enumerate(blocks)}
    extra = {}
    for kind in ("patchscope", CEIL):
        q = f"{results}/readouts/{model}__{fam}_{kind}.json"
        if os.path.exists(q):
            x = json.load(open(q))
            extra[kind] = ([int(b) for b in x["blocks"]], {r["id"]: r for r in x["rows"]})
    recs = []
    for r in d["rows"]:
        b = beh.get(r["id"])
        if b is None or r.get("v_read_fitted_on") or r.get("coco_fit_image"):
            continue
        # a picture the shared fit saw under another question is left out too
        if b.get("image") in _fit_pictures(results, model):
            continue
        y = _own(b)
        e = _correct(b) if r["kind"] == "letter" else None
        if y is None or (r["kind"] == "letter" and e is None):
            continue
        # every reader is scored on the same records
        if any(r["id"] not in rows for _b, rows in extra.values()):
            continue
        recs.append((r, y, e))
    y = np.array([t for _r, t, _e in recs], int)
    minority = min(int(y.sum()), int(y.size - y.sum()))
    thin = y.size < MIN_N or minority < MIN_CLASS
    if thin and (floor or y.size == 0):
        return None
    n_min, c_min = (1, 1) if thin else (MIN_N, MIN_CLASS)
    series = {}
    for k in READERS:
        if k != "patchscope" and any(r.get(k) for r, _t, _e in recs):
            series[k] = np.array([[_score(v, e) for v in (r.get(k) or [None] * len(blocks))]
                                  for r, _t, e in recs], float)
    for kind, (xb, rows) in extra.items():
        for k in {rk for r in rows.values() for rk in r if rk.startswith(kind)}:
            M = np.full((len(recs), len(blocks)), np.nan)
            for i, (r, _t, e) in enumerate(recs):
                for j, v in enumerate(rows[r["id"]].get(k) or []):
                    if j < len(xb) and xb[j] in col:
                        M[i, col[xb[j]]] = _score(v, e)
            series[k] = M
    au = {}
    for k, M in series.items():
        a = np.full(len(blocks), np.nan)
        for j in range(len(blocks)):
            ok = np.isfinite(M[:, j])
            yy = y[ok]
            if ok.sum() >= n_min and min(int(yy.sum()), int(yy.size - yy.sum())) >= c_min:
                a[j] = auroc(yy, M[ok, j])
        if np.isfinite(a).any():
            au[k] = a
    out = dict(model=model, fam=fam, src=_src(fam), n=int(y.size),
               depth=np.asarray(blocks, float) / max(int(d.get("n_layers") or len(blocks)) - 1, 1), auroc=au)
    if thin:
        out.update(thin=True, minority=minority)
    return out


def _pick_patchscope(fams):
    """Patchscopes in whichever target prompt reads the answer better on this model, mean over blocks and families."""
    cands = sorted({k for f in fams for k in f["auroc"] if k.startswith("patchscope")})
    if not cands:
        return None

    def mean(k):
        got = [(f["n"], float(np.nanmean(f["auroc"][k]))) for f in fams if k in f["auroc"]]
        return np.average([v for _n, v in got], weights=[n for n, _v in got]) if got else -np.inf
    best = max(cands, key=mean)
    for f in fams:
        v = f["auroc"].get(best)
        for k in cands:
            f["auroc"].pop(k, None)
        if v is not None:
            f["auroc"]["patchscope"] = v
    return best


def lens_scoreboard_readouts(results="results", out="figures", model=None):
    """MAIN. Through the middle of depth V-Read reads the answer a model is
    about to give clearly better than every label-free baseline read on Open
    Images authority questions, and is level with or behind Patchscopes or
    the logit lens elsewhere; V-Read fitted on
    unlabelled COCO photographs is a separate reader, never used to judge
    V-Read, and the labelled probe is a supervised reference, not an upper
    bound.

    Per model (Qwen3-VL-8B-Instruct, Qwen3.5-27B); the pooled figure pools
    the families of every model whose V-Read readouts cover at least half its
    records, and is drawn once two models do. Records: the short-answer
    records of every family whose behaviour pass kept answer-spot states: the
    yes-or-no questions of the authority families and of the generated scenes'
    families, the test-or-real questions of the other eval-awareness families,
    and the four-letter questions of the WMDP carrier (biology, chemistry,
    cyber), its three system instructions pooled. The password-locking
    study's families are never read. A family is scored when it has at least
    24 records whose reply states an answer and at least 5 of each answer, and
    only when every reader drawn has read it. Records any fitted lens was fitted on, and records
    whose photograph was in the COCO fit, are dropped for every reader, and
    every reader is scored on exactly the same records.

    The target of every AUROC is the answer the model gives, as its reply
    states it: yes against no, test against real, or, for the letters,
    whether the letter it picks is the correct one. Each reader's score at a
    block is its reading of the stored answer-spot state after that block,
    through the model's own final norm and unembedding: log p(yes) - log
    p(no), log p(test) - log p(real), or the correct letter's log probability
    minus the best other's. The readers are V-Read (an affine map per block,
    fitted without labels on SimonSays records none of these is drawn from,
    drawn over the 16 blocks it was fitted at), V-Read fitted on COCO
    photographs with generic instructions, and the baselines: Patchscopes
    (Ghandeharioun et al. 2024: the state injected at the same block into the
    last position of a few-shot token-identity prompt, text only), drawn in
    whichever of its two prompts (the paper's plain text, or the same
    demonstrations in the chat template) reads the answer better on that
    model, a choice that can only favour it, and the logit lens (the raw
    state). The tuned and Jacobian lenses are drawn wherever their readouts
    cover half the records, so far on Qwen3-VL-8B-Instruct only. The
    labelled probe (supervised reference; dotted, hollow ring) is a logistic probe per block
    trained on these answers, fitted on every family of the task pooled, in
    five folds grouped by picture across the pool, its penalty and a PCA
    width (16, 64, 256 or all) chosen inside each training fold, each family
    scored out of fold on its own records; for the letters a
    four-way probe on the letter picked. Left: AUROC on normalised depth
    (block over the number of blocks less one), the mean over families
    weighted by records; 0.5 is chance. Right: per source (columns) and
    reader (rows), the mean AUROC over the blocks inside depth 0.45 to 0.8
    (shaded on the left) at which V-Read was fitted, the same blocks for
    every reader, printed in each cell and shaded darker for higher over the
    label-free readers' range; V-Read's row is outlined and the labelled
    probe stands apart in greys.

    Qwen3.5-27B (811 records over 9 families; Patchscopes in the plain-text
    prompt; the tuned and Jacobian lenses not read). Mean AUROC over the band: Open Images (n=429) V-Read 0.80,
    V-Read fitted on COCO 0.87, Patchscopes 0.69, logit lens 0.72, probe
    0.93; Interfaces (n=125) 0.84, 0.82, 0.86, 0.82, 0.89; Generated scenes
    (n=32) 0.90, 0.91, 0.93, 0.92, 0.95; WMDP carrier (n=225) 0.58, 0.58,
    0.66, 0.61, 0.61.

    Limits: a source whose every family is under the scoring floor (on the
    27B, nuScenes: fewer than 5 of one answer) is still drawn, from whatever
    records it has, with a dagger on its values (n/a with a dagger where no
    record has the other answer, so AUROC is undefined); it is not used to
    judge any reader. The WMDP carrier is read barely above
    chance by every reader in the middle of depth, the probe included.
    """
    pat = f"{results}/readouts/{model or '*'}__*_readouts.json"
    paths = [p for p in sorted(glob.glob(pat)) if not P.LOCKED.search(os.path.basename(p))]
    if not paths or (model is None and not P.AVERAGE) or not P._enough(paths):
        return None
    by_model = {}
    for p in paths:
        m, fam = os.path.basename(p)[: -len("_readouts.json")].split("__", 1)
        by_model.setdefault(m, []).append(fam)
    fams, thin = [], []
    for m, fl in sorted(by_model.items()):
        got = [f for f in (_readout_family(results, m, fam) for fam in fl) if f]
        best = _pick_patchscope(got)
        # under the scoring floor: shown, marked, only as a source column no other family makes
        seen = {f["fam"] for f in got}
        for f in (_readout_family(results, m, fam, floor=False) for fam in fl if fam not in seen):
            if f:
                v = f["auroc"].get(best) if best else None
                for k in [k for k in f["auroc"] if k.startswith("patchscope")]:
                    f["auroc"].pop(k)
                if v is not None:
                    f["auroc"]["patchscope"] = v
                thin.append(f)
        # pooled over models, a model whose V-Read readouts are not yet written
        # would be averaged in on the baselines alone, and V-Read would be
        # compared on one model's records against the others on two
        if model is None:
            n_all = sum(f["n"] for f in got)
            if sum(f["n"] for f in got if "v_read" in f["auroc"]) < 0.5 * max(n_all, 1):
                continue
        fams += got
    if not fams:
        return None
    if model is None and len({f["model"] for f in fams}) < 2:
        return None
    W = float(sum(f["n"] for f in fams))
    has = {k: [f for f in fams if k in f["auroc"]] for k in (*READERS, CEIL)}
    drawn = [k for k in (*READERS, CEIL) if sum(f["n"] for f in has[k]) >= 0.5 * W and has[k]]
    # a comparison without V-Read is not this figure; it waits for the readouts
    if "logit" not in drawn or "v_read" not in drawn:
        return None
    # fitted baselines still running on this model: rows of hatched cells in the matrix
    pend = [k for k in fits_running(results, model) if k not in drawn]
    mat = [k for k in (*READERS, CEIL) if k in drawn or k in pend]
    use = [f for f in fams if all(k in f["auroc"] for k in drawn)]
    if not use:
        return None
    from vlens.figures.fx_depth import _chance
    st.use()
    grid = np.linspace(0.0, 1.0, 41)
    w = np.asarray([f["n"] for f in use], float)
    curve = {}
    for k in drawn:
        rows = []
        for f in use:
            a, dep = f["auroc"][k], f["depth"]
            ok = np.isfinite(a)
            rows.append(np.interp(grid, dep[ok], a[ok], left=np.nan, right=np.nan) if ok.sum() >= 2
                        else np.full(grid.size, np.nan))
        Y = np.vstack(rows)
        cover = (np.isfinite(Y) * w[:, None]).sum(0)
        mean = np.nansum(np.nan_to_num(Y) * w[:, None], 0) / np.maximum(cover, 1e-9)
        curve[k] = np.where(cover >= w.sum() * 0.999, mean, np.nan)
    if not np.isfinite(curve["logit"]).any():
        return None
    by_src = {}
    for f in use:
        # every reader is averaged over the same blocks: those inside the band
        # at which V-Read was fitted and read, so no reader gains or loses
        # from being read on a finer grid
        inb = (f["depth"] >= BAND[0]) & (f["depth"] <= BAND[1]) & np.isfinite(f["auroc"]["v_read"])
        s = by_src.setdefault(f["src"], dict(n=0, v={k: [] for k in drawn}))
        s["n"] += f["n"]
        for k in drawn:
            v = f["auroc"][k][inb]
            if v.size and np.isfinite(v).all():
                s["v"][k].append((f["n"], float(np.mean(v))))
    full_src = set(by_src)
    for f in thin:
        if f["src"] in full_src:
            continue
        s = by_src.setdefault(f["src"], dict(n=0, v={k: [] for k in drawn}, thin=True))
        s["n"] += f["n"]
        if "v_read" not in f["auroc"]:
            continue
        inb = (f["depth"] >= BAND[0]) & (f["depth"] <= BAND[1]) & np.isfinite(f["auroc"]["v_read"])
        for k in drawn:
            if k in f["auroc"]:
                v = f["auroc"][k][inb]
                if v.size and np.isfinite(v).all():
                    s["v"][k].append((f["n"], float(np.mean(v))))
    fig, (ax, ay) = st.wide(1, 2, unit=(4.6, 3.5), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax.axvspan(*BAND, color=st.BAND, lw=0, zorder=0)
    series = []
    for k in reversed(drawn):
        lead = k == "v_read"
        ax.plot(grid, curve[k], color=P.COLOUR[k], lw=2.6 if lead else 1.9,
                dashes=P.DASH.get(k, (None, None)), zorder=4 if lead else 3)
        series.append((grid, curve[k]))
    lo = np.nanmin([np.nanmin(c) for c in curve.values() if np.isfinite(c).any()])
    floor = max(0.0, min(0.45, float(lo)) - 0.03)
    ax.set_ylim(floor, 1.03)
    ax.set_yticks([t for t in (0.0, 0.25, 0.5, 0.75, 1.0) if t >= floor])
    st.depth_axis(ax, "depth")
    ax.set_ylabel("AUROC for the answer")
    st.despine(ax)
    _chance(ax, 0.5, series)
    rows = [s for s in SRC_ORDER if s in by_src]
    table = [{k: np.average([x for _, x in by_src[s]["v"][k]], weights=[n for n, _ in by_src[s]["v"][k]])
              for k in drawn if by_src[s]["v"][k]} for s in rows]
    two = {"Open Images": "Open\nImages", "Generated scenes": "Generated\nscenes",
           "WMDP": "WMDP"}
    _value_matrix(ay, [two.get(x, x) for x in (P.SOURCE[s].replace(" (VINS-120K)", "") for s in rows)],
                  mat, table, ceil=CEIL, readers_down=True, pending=set(pend),
                  marked={j for j, s in enumerate(rows) if by_src[s].get("thin")})
    # panels placed by hand: the left one, a gutter wide enough for the source
    # names, then the right one
    fig.set_layout_engine("none")
    W, H = fig.get_size_inches()
    # the matrix: readers down, sources across, cells about as tall as a line
    W = st.MAX_WIDTH
    nr = len(mat) + (0.35 if CEIL in mat else 0.0)
    mh, mw = 0.3 * nr, 0.88 * len(rows)
    H = max(mh + 0.6, 2.9) + 0.1
    fig.set_size_inches(W, H)
    ay.set_position([(W - mw - 0.05) / W, (H - 0.6 - mh) / H, mw / W, mh / H])
    ax.set_position([0.62 / W, 0.62 / H, (W - mw - 3.15) / W, (H - 0.75) / H])
    # the key sits in a band of its own directly above both panels, so it
    # cannot be read as another row of marks
    fsk = st.matplotlib.rcParams["legend.fontsize"] * 0.9
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=P.COLOUR[k], lw=2.6 if k == "v_read" else 1.9,
                      dashes=P.DASH.get(k, (None, None)), marker="o",
                      ms=9 if k == "v_read" else 7.5,
                      mfc="white" if k == "probe" else P.COLOUR[k],
                      mec=P.COLOUR[k] if k == "probe" else "white") for k in drawn]
    names = [P.NICE[k] for k in drawn]
    st.key_band(fig, handles, names, fontsize=fsk, rows=1 if len(names) <= 5 else 2,
                ncol=int(np.ceil(len(names) / (1 if len(names) <= 5 else 2))), columnspacing=1.2)
    return st.save(fig, P._at(out, "lens_scoreboard_readouts"), check=False, tight=True)


def _scoreboard_caption(results=None):
    """The docstring, and which fitted baselines are still running on which model (hatched cells)."""
    results = results or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "results")
    models = sorted({os.path.basename(p).split("__")[0]
                     for p in glob.glob(f"{results}/readouts/*__*_readouts.json")})
    run = {m: fits_running(results, m) for m in models}
    bits = [f"on {m} the " + " and the ".join(P.NICE[k] for k in ks) for m, ks in run.items() if ks]
    doc = lens_scoreboard_readouts.__doc__ or ""
    if bits:
        doc = doc.rstrip() + ("\n\n    Still running: " + "; ".join(bits) + " (fits queued); their rows of the "
                              "matrix are drawn hatched, with no number, until the fits land.\n")
    return doc


#: ``tools/make_captions.py`` calls this for the caption, so the running readers are named from the files
lens_scoreboard_readouts.caption = _scoreboard_caption


# ------------------------------------------------------------------ cost against fidelity

#: one marker per model, so two models' points of one instrument (one colour)
#: can be told apart without a second colour meaning the same thing
MODEL_MARK = {"Qwen3-VL-8B-Instruct": "o", "Qwen3.5-27B": "s", "gemma-4-26B-A4B-it": "^", "InternVL3_5-8B-HF": "D"}


def _cost_rows(results, model):
    """(instrument, model, development KL, fit seconds, data) for every fitted lens of every model.

    Read from ``results/fits/<model>__fitmeta.json`` through the table
    builder's own choice of the shared fit and the COCO fit
    (``tools/make_vread_table.fit_cost``), so the figure and the cost table
    read the same fits. The retired lens-bench runs are not read.
    """
    import importlib
    import sys
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools")
    if here not in sys.path:
        sys.path.insert(0, here)
    T = importlib.import_module("make_vread_table")
    models = [model] if model else [os.path.basename(p).split("__")[0]
                                    for p in sorted(glob.glob(f"{results}/fits/*__fitmeta.json"))]
    rows = []
    for m, r in T.fit_cost(results, models).items():
        if r is None or (model and m != model):
            continue
        for k in T.COST_LENSES:
            x = r.get(k)
            if not x or k in P.HIDDEN or x.get("seconds") is None or x.get("heldout_kl") is None:
                continue
            rows.append((k, m, float(x["heldout_kl"]), float(x["seconds"]), x))
    return rows


def _cost_caption(rows):
    by = {}
    for k, m, kl, sec, _x in rows:
        by.setdefault(m, []).append((k, kl, sec))
    parts = []
    for m, rs in by.items():
        parts.append(f"{m}: " + "; ".join(f"{P.NICE.get(k, k)} {sec:,.0f} s, {kl:.2f} nats" for k, kl, sec in rs))
    missing = [m for m, rs in by.items() if not {"tuned", "jacobian"} & {k for k, *_ in rs}]
    return (
        "MAIN. V-Read is a closed-form least-squares solve per block: it costs a fraction of the tuned "
        "lens's fitting time for a development-set KL within about a nat of it, and the Jacobian lens "
        "costs more than V-Read and reads several times further from the model.\n\n"
        "Across, the wall-clock seconds to fit an instrument at every block it is read at; up, its "
        "development-set KL: the KL divergence from the model's own next-token distribution on the "
        "development prompts, which no lens is trained on but on which V-Read's ridge penalty and "
        "per-block gain (and the baselines' per-block gain) were chosen, averaged over blocks, read "
        "through the model's own final norm and unembedding. Because V-Read's settings were chosen on "
        "this same set and the tuned lens's training settings were not, the KL is not a held-out measure "
        "and is optimistic for V-Read. Both axes are logarithmic; down and to the left is better. Fit "
        "time, data and KL are those recorded with each fit in results/fits/<model>__fitmeta.json, for the "
        "shared fit (900 SimonSays prompts held out of the behaviour pass) and the COCO fit (COCO val2017 "
        "photographs with generic prompts, its KL on its own COCO development photographs, so not "
        "comparable) that the readouts used. The marker names the model.\n\n"
        "Values. " + ". ".join(parts) + ".\n\n"
        "Limits: times are single runs on one NVIDIA H200 NVL shared with at most one other job and "
        "include each method's own data pipeline; the logit lens has nothing to fit and is not drawn"
        + (f"; the tuned and Jacobian lenses have not been fitted on {', '.join(missing)}" if missing else "")
        + ("; on these long, mostly-image prompts the Jacobian lens's development-set KL is above the logit "
           "lens's, although it follows the reference estimator (the paper finds it below); that gap is not "
           "investigated here" if any(k == "jacobian" for k, *_ in rows) else "")
        + ".")


def _import_scoreboard_caption():
    """The docstring with the paired tests of the tables and the baselines each model is compared against."""
    import importlib
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.join(root, "tools") not in sys.path:
        sys.path.insert(0, os.path.join(root, "tools"))
    doc = lens_scoreboard_readouts.__doc__
    try:
        T = importlib.import_module("make_vread_table")
        table, _notes = T.collect(os.path.join(root, "results"))
        F = T.findings(table)
    except Exception:  # noqa: BLE001
        return doc
    by = []
    for m, r in F["by_model"].items():
        ag = T.against(table, m)
        by.append(f"{m}{' (' + ag + ')' if ag else ''}: ahead in {r['win']}, level in {r['level']}, "
                  f"behind in {r['behind']}" + (f", untested in {r['untested']}" if r["untested"] else ""))
    wins = "; ".join(f"{T.label_of(f)}, {m}" for m, f, _c in F["wins"])
    tests = (f"Paired tests, the tables' cells (V-Read, the primary fit, minus the best baseline over the same "
             f"window, pictures resampled): {'; '.join(by)}. Ahead: {wins or 'none'}. The best baseline of a "
             f"cell is chosen on the test data, which favours the baselines; the window was set after the first "
             f"curves were seen (Table vread_window gives others).")
    doc = inspect_clean(doc)
    head, _, rest = doc.partition("\n\n")
    return head + "\n\n" + rest.replace("\n\nLimits:", "\n\n" + tests + "\n\nLimits:", 1)


def inspect_clean(doc):
    import inspect
    return inspect.cleandoc(doc or "")


def _import_cost_caption():
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
    try:
        rows = _cost_rows(here, None)
        return _cost_caption(rows) if rows else None
    except Exception:  # noqa: BLE001
        return None


def cost_and_quality(results="results", out="figures", model=None):
    """MAIN. V-Read's fitting time against its development-set KL, beside the tuned and Jacobian lenses; replaced by the caption built from the files (``_cost_caption``)."""
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D
    rows = [r for r in _cost_rows(results, None if P.AVERAGE and model is None else model) if r[0] != "v_read_coco"]
    if P.AVERAGE and model is None:
        if len({r[1] for r in rows}) < 2:
            return None
    elif model is None:
        return None
    cost_and_quality.__doc__ = _cost_caption(rows)
    rows = [r[:4] for r in rows]
    fitted = rows
    free = []
    # one fitted instrument on its own is a point, not a trade-off
    if len(fitted) < 2:
        return None
    models = sorted({r[1] for r in rows}, key=lambda m: (m not in MODEL_MARK, m))
    several = len(models) > 1
    st.use()
    fig, ax = st.wide(unit=(10.4, 3.5))
    lo, hi = min(r[3] for r in fitted), max(r[3] for r in fitted)
    klo, khi = min(r[2] for r in rows), max(r[2] for r in rows)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo / 1.9, hi * 1.9)
    ax.set_ylim(klo / 1.25, khi * 1.3)
    x0, x1 = ax.get_xlim()
    # an instrument with nothing to fit costs nothing at any budget, so it is a
    # line across the figure, named inside the right end, above the line
    names = []
    for k, m, kl, _ in free:
        c = P.COLOUR.get(k, st.SERIES[0])
        ax.axhline(kl, color=c, lw=1.4, dashes=(4, 3), zorder=2)
        t = ax.annotate(P.NICE.get(k, k) + ", free", xy=(x1, kl), xytext=(-4, 3),
                        textcoords="offset points", ha="right", va="bottom", color=st.INK)
        names.append((t, m, c))
    if several:
        # the model's marker just left of the line's name says whose line it is
        fig.canvas.draw()
        rnd = fig.canvas.get_renderer()
        for t, m, c in names:
            bb = t.get_window_extent(rnd)
            xm, ym = ax.transData.inverted().transform((bb.x0 - 12, (bb.y0 + bb.y1) / 2))
            ax.plot([xm], [ym], marker=MODEL_MARK.get(m, "D"), ms=8, color=c, mec="white",
                    mew=1.0, ls="none", zorder=3, clip_on=False)
    # the instruments are keyed by colour in the band above, the models by
    # marker; names beside the points collided where two models' fits sit close
    for k, m, kl, sec in fitted:
        c = P.COLOUR.get(k, st.SERIES[0])
        ours = k.startswith("v_read")
        ax.plot([sec], [kl], marker=MODEL_MARK.get(m, "D") if several else "o",
                ms=12 if ours else 9.5, color=c, mec="white", mew=1.2, ls="none", zorder=5 if k == "v_read" else 4)
    fmt = mticker.FuncFormatter(lambda v, _p: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")
    ax.yaxis.set_major_locator(mticker.FixedLocator(
        [v for v in (2, 3, 4, 5, 7, 10, 15, 20) if klo / 1.25 <= v <= khi * 1.3]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2, 5)))
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_formatter(fmt)
    ax.set_xlabel("seconds to fit, all blocks")
    ax.set_ylabel("development-set KL (nats)")
    st.despine(ax)
    # the key, in a band of its own above the panel: colour is the instrument,
    # and, over several models, the hollow marker is the model
    ks = [k for k in ("v_read", "v_read_coco", "tuned", "jacobian") if any(r[0] == k for r in fitted)]
    hs = [Line2D([], [], ls="none", marker="o", ms=10, color=P.COLOUR.get(k, st.SERIES[0]), mec="white")
          for k in ks]
    labs = [P.NICE.get(k, k) for k in ks]
    if several:
        hs += [Line2D([], [], ls="none", marker=MODEL_MARK.get(m, "D"), ms=8, color=st.INK,
                      mfc="white", mew=1.3) for m in models]
        labs += list(models)
    st.key_band(fig, hs, labs, handletextpad=0.3, handlelength=1.0)
    return st.save(fig, P._at(out, "cost_and_quality"))


#: ``tools/make_captions.py`` calls this for the caption, so the text is built from what is on disk
cost_and_quality.caption = _import_cost_caption
lens_scoreboard_readouts.caption = _import_scoreboard_caption
