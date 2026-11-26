# SPDX-License-Identifier: Apache-2.0
"""V-Write against the best baseline on every test we have, in one figure.

Left, a forest of paired differences, V-Write minus the strongest other
ranking on that test and model, each with a 95 per cent interval from
resampling pictures. Right, three COCO photographs with V-Write's top cells
and attribution patching's over the human outline of the object asked about.

Read from, and only from:

  results/baselines/<m>__patchlenses.json    V-Write and seven published methods (SimonSays)
  results/patchlenses/<m>__patchlenses.json  V-Write, attribution patching and the lenses (SimonSays)
  (the SimonSays fall, top fifth set to zero, is read from the baselines run too)
  results/coco_patchcausal/<m>__patchcausal.json  COCO: maps graded against human masks, and the fall

The password-locking families are dropped on family name and record id.
"""
from __future__ import annotations

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from vlens.figures import paper as P
from vlens.figures import style as st

MODELS = ("Qwen3-VL-8B-Instruct", "Qwen3.5-27B", "gemma-4-26B-A4B-it", "InternVL3_5-8B-HF")
BOOT = 2000
K = 0.2                                 #: the top fifth of the patches
TOP_SHARE = 0.10                        #: the share of cells drawn on each photograph
#: published attribution methods run beside V-Write (the baselines run)
PUBLISHED = ("attr_patch", "ig_zero", "grad_x_act", "act_patch", "rise", "kernel_shap", "gradcam")
#: never a baseline: V-Write itself (8 or 32 steps), the floor, and instruments kept out of the figures
NOT_BASE = {"v_write", "ig_mean32", "random"} | set(P.HIDDEN)
NAME = dict(P.NICE, gradxact_zero=P.NICE["grad_x_act"], center="centre prior",
            vwrite_mean=P.NICE["v_write"], v_read_cross="V-Read (to the answer)",
            tuned_cross="tuned lens (to the answer)")


def _json(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def _ok_row(r):
    return not (P.LOCKED.search(str(r.get("family", ""))) or P.LOCKED.search(str(r.get("id", ""))))


def _clusters(rows, key=None):
    """The picture each row shows, as integer cluster ids, so resampling draws pictures."""
    if key is None:
        from vlens.figures.fx_attrplane import _picture_of
        pic = _picture_of()
        key = lambda r: pic.get(str(r["id"]), str(r["id"]).split("__")[0])  # noqa: E731
    _, cid = np.unique([str(key(r)) for r in rows], return_inverse=True)
    return cid


def _boot(d, cid, seed=0):
    """Mean of paired differences ``d`` and its 95% interval over resampled pictures."""
    d = np.asarray(d, float)
    ok = np.isfinite(d)
    d, cid = d[ok], cid[ok]
    _, cid = np.unique(cid, return_inverse=True)
    n = cid.max() + 1
    rng = np.random.default_rng(seed)
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=BOOT)[:, cid].astype(float)
    draws = (W * d).sum(1) / np.maximum(W.sum(1), 1e-12)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), n=int(d.size), pics=int(n),
                win=float((d > 0).mean()))


# ------------------------------------------------------------------ the tests

def _simonsays_agreement(results, m):
    """Spearman agreement with measured patching (zero replacement), V-Write minus the best published method.

    Only the baselines run counts: every published method and V-Write are
    scored on one shared subsample of records, and a model without that run
    has no row here rather than a comparison against a partial set of methods.
    """
    src = "baselines"
    d = _json(os.path.join(results, "baselines", f"{m}__patchlenses.json"))
    if not (d and d.get("rows")):
        return None
    rows = [r for r in d["rows"] if _ok_row(r) and isinstance(r.get("truth_agreement"), dict)]
    meths = sorted({k.split("|", 1)[1] for r in rows for k in r["truth_agreement"] if k.startswith("zero|")})
    cands = [k for k in meths if k in PUBLISHED and not P._hidden(k)]
    if not cands:
        return None
    # the shared subsample: records on which V-Write and every published method have a value
    fin = lambda r, k: np.isfinite(float(r["truth_agreement"].get(f"zero|{k}", np.nan)))  # noqa: E731
    rows = [r for r in rows if all(fin(r, k) for k in ("v_write", *cands))]
    if len(rows) < 20:
        return None
    mean_of = lambda k: float(np.mean([float(r["truth_agreement"][f"zero|{k}"]) for r in rows]))  # noqa: E731
    best = max(cands, key=mean_of)
    diff = [float(r["truth_agreement"]["zero|v_write"]) - float(r["truth_agreement"][f"zero|{best}"])
            for r in rows]
    out = _boot(diff, _clusters(rows))
    out.update(base=best, src=src, n_cands=len(cands), vw=float(mean_of("v_write")), bv=float(mean_of(best)))
    return out


def _simonsays_fall(results, m):
    """The fall in the answer score with the top fifth set to zero, V-Write minus the best published method.

    Read from the baselines run, whose zero replacement is a counterfactual
    V-Write does not integrate from (it integrates from the picture's mean
    state), against every published method on the run's shared subsample of
    records. A model without that run has no row here rather than a
    comparison against a partial set of methods. The mean-replacement fall
    (``results/patchcausal``) is V-Write's own counterfactual and is left to
    the appendix figure ``patch_causal``.
    """
    d = _json(os.path.join(results, "baselines", f"{m}__patchlenses.json"))
    if not (d and d.get("rows")):
        return None
    key = lambda k: f"zero|{k}|{K:g}"  # noqa: E731
    rows = [r for r in d["rows"] if _ok_row(r) and isinstance(r.get("after"), dict)
            and r.get("s0") is not None and key("v_write") in r["after"]]
    cands = [k for k in PUBLISHED if not P._hidden(k) and rows and all(key(k) in r["after"] for r in rows)]
    if len(rows) < 20 or not cands:
        return None
    fall = lambda r, k: float(r["s0"]) - float(r["after"][key(k)])  # noqa: E731
    mean_of = lambda k: float(np.mean([fall(r, k) for r in rows]))  # noqa: E731
    best = max(cands, key=mean_of)
    out = _boot([fall(r, "v_write") - fall(r, best) for r in rows], _clusters(rows))
    out.update(base=best, n_cands=len(cands), vw=mean_of("v_write"), bv=mean_of(best),
               budget=(d.get("summary") or {}).get("meta") or {})
    return out


def _fall(rows, cands, cluster_fn):
    key = lambda k: f"{k}|{K:g}|top"  # noqa: E731
    rows = [r for r in rows if _ok_row(r) and isinstance(r.get("drop"), dict) and key("vwrite_mean") in r["drop"]]
    # a method run on a subset only (RISE on the larger model) is left out rather than shrinking the set
    cands = [c for c in cands if rows and all(key(c) in r["drop"] for r in rows)]
    if len(rows) < 20:
        return None
    mean_of = lambda k: float(np.mean([float(r["drop"][key(k)]) for r in rows]))  # noqa: E731
    best = max(cands, key=mean_of)
    diff = [float(r["drop"][key("vwrite_mean")]) - float(r["drop"][key(best)]) for r in rows]
    out = _boot(diff, cluster_fn(rows))
    out.update(base=best, vw=mean_of("vwrite_mean"), bv=mean_of(best))
    return out


def _coco_rows(results, m):
    # the run that scores every attribution method on the same photographs
    d = _json(os.path.join(results, "coco_baselines", f"{m}__patchcausal.json"))
    if not (d and d.get("rows")):
        return None, None
    rows = [r for r in d["rows"] if str(r.get("family", "")).startswith("coco") and _ok_row(r)]
    return rows, d.get("summary") or {}


def _coco_loc(rows, summ, q):
    """A localisation measure against the human mask, V-Write minus the best other map.

    The interval is the run's own (``summary.localisation_paired``, resampled
    over pictures); the mean is recomputed from the rows and must match it.
    """
    graded = [r for r in rows if r.get("loc") and "vwrite_mean" in r["loc"]]
    paired = summ.get("localisation_paired") or {}
    cands = [k.split("-", 1)[1] for k in paired if k.startswith("vwrite_mean-")
             and k.split("-", 1)[1] != "random" and q in paired[k]]
    if not graded or not cands:
        return None
    best = min(cands, key=lambda c: paired[f"vwrite_mean-{c}"][q]["mean"])
    diff = np.array([float(r["loc"]["vwrite_mean"][q]) - float(r["loc"][best][q]) for r in graded
                     if best in r["loc"] and r["loc"][best].get(q) is not None
                     and r["loc"]["vwrite_mean"].get(q) is not None], float)
    diff = diff[np.isfinite(diff)]
    pr = paired[f"vwrite_mean-{best}"][q]
    assert abs(diff.mean() - pr["mean"]) < 1e-6, f"COCO paired {q} vs {best} does not reproduce"
    vw = float(np.nanmean([r["loc"]["vwrite_mean"][q] for r in graded]))
    bv = float(np.nanmean([r["loc"][best][q] for r in graded if best in r["loc"]]))
    pics = len({str(r.get("image") or r["id"]) for r in graded})
    return dict(mean=float(pr["mean"]), lo=float(pr["ci"][0]), hi=float(pr["ci"][1]), n=int(diff.size),
                pics=pics, base=best, vw=vw, bv=bv, win=float((diff > 0).mean()))


def _coco_fall(rows):
    return _fall(rows, ("attr_patch", "gradxact_zero", "ig_zero", "act_patch", "kernel_shap", "rise", "gradcam", "center"),
                 lambda rs: _clusters(rs, key=lambda r: r.get("image") or r["id"]))


#: the groups, in order: the header naming the measure, and the rows under it
GROUPS = (
    ("agree", "SimonSays: agreement with measured patching"),
    ("fall", "SimonSays: fall in answer, top fifth set to zero"),
    ("auroc", "COCO: whole-map AUROC, human mask"),
    ("iou", "COCO: top-cell IoU, human mask"),
    ("cfall", "COCO: fall in answer, top fifth replaced"),
)
HEAD_LONG = {"agree": "SimonSays agreement with measured patching",
             "fall": "SimonSays fall in the answer score, top fifth set to zero",
             "auroc": "COCO whole-map AUROC against the human mask",
             "iou": "COCO IoU of the top cells with the human mask",
             "cfall": "COCO fall in the answer score, top fifth replaced"}
FMT = {"agree": "{:+.2f}", "fall": "{:+.2f}", "auroc": "{:+.2f}", "iou": "{:+.2f}", "cfall": "{:+.2f}"}


def _numbers(results, models):
    out = {g: {} for g, _ in GROUPS}
    for m in models:
        a = _simonsays_agreement(results, m)
        if a:
            out["agree"][m] = a
        f = _simonsays_fall(results, m)
        if f:
            out["fall"][m] = f
        rows, summ = _coco_rows(results, m)
        if rows:
            for q in ("auroc", "iou"):
                v = _coco_loc(rows, summ, q)
                if v:
                    out[q][m] = v
            c = _coco_fall(rows)
            if c:
                out["cfall"][m] = c
    return out


# ------------------------------------------------------------------ the photographs

def _pick(rows, recs, n=3):
    """Three photographs at the quartiles of object size, one category each, chosen without looking at the maps."""
    from vlens.figures import fx_coco as FC
    ok = [r for r in rows if r["id"] in recs and (r.get("maps") or {}).get("vwrite_mean") is not None
          and (r.get("maps") or {}).get("attr_patch") is not None and r.get("loc")]
    if len(ok) < n:
        return []
    # a photograph already on disk is preferred, so the figure builds offline
    have = {r["id"] for r in ok if any(os.path.exists(os.path.join(d or "", recs[r["id"]]["image"]))
                                       for d in FC._COCO_DIRS)}
    pool = [r for r in ok if r["id"] in have] or ok
    area = np.array([float(r["area"]) for r in pool])
    got, cats = [], set()
    for q in np.quantile(area, np.linspace(0.25, 0.75, n)):
        cand = sorted((r for r in pool if r["target"] not in cats and r not in got),
                      key=lambda r: (abs(float(r["area"]) - q), r["id"]))
        if cand:
            got.append(cand[0])
            cats.add(cand[0]["target"])
    return sorted(got, key=lambda r: float(r["area"]))


def _tile(ax, im, rec, row, meth, colour):
    """The photograph as the model sees it, the human outline and the map's top tenth of cells."""
    import matplotlib.patheffects as pe
    from vlens.data import coco as CC
    from vlens.figures import fx_coco as FC
    side = 448
    a = np.asarray(im.resize((side, side), 1), float)
    a = np.clip(a * 0.74 + 255.0 * 0.26, 0, 255).astype(np.uint8)
    ax.imshow(a, extent=(0, side, side, 0), interpolation="bilinear", zorder=1)
    Gr, Gc = FC._grid(row)
    mp = np.asarray(row["maps"][meth], float)
    order = np.argsort(-mp, kind="stable")
    nk = max(int(round(TOP_SHARE * mp.size)), 1)
    ph, pw = side / Gr, side / Gc
    for t in order[:nk]:
        y0, x0 = divmod(int(t), Gc)
        ax.add_patch(Rectangle((x0 * pw, y0 * ph), pw, ph, fc=colour, ec="none", alpha=0.62, zorder=2))
    m = CC.mask_square(rec, side)
    cs = ax.contour(np.arange(side) + 0.5, np.arange(side) + 0.5, m, levels=[0.5],
                    colors=st.INK, linewidths=1.3, zorder=4)
    for c in getattr(cs, "collections", [cs]):
        c.set_path_effects([pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])
    ax.set_xlim(0, side)
    ax.set_ylim(side, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(st.AXIS)
        sp.set_linewidth(0.8)


# ------------------------------------------------------------------ caption

def _f(x, fmt="{:.2f}"):
    return fmt.format(x).replace("-", "−") if x is not None and np.isfinite(x) else "n/a"


def _budget(v):
    """The sample budgets of RISE and KernelSHAP as the baselines run records them, else its default."""
    got = [(m, r.get("budget") or {}) for m, r in v["fall"].items()]
    rec = [f"{m}: RISE {b['masks']} masks, KernelSHAP {b.get('shap_samples', b['masks'])} samples"
           for m, b in got if b.get("masks")]
    if rec:
        return "; ".join(rec)
    return ("128 random masks each, the run's default, a few per cent of the budgets of their papers "
            "(RISE 4,000 to 8,000 masks; KernelSHAP 2P + 2,048 coalitions)")


def _caption(v, models, picks, pick_model):
    def row(g, m):
        r = v[g][m]
        return (f"{_f(r['mean'], '{:+.2f}')} [{_f(r['lo'])}, {_f(r['hi'])}] against "
                f"{NAME.get(r['base'], r['base'])}")
    ms = [m for m in models if any(m in v[g] for g, _ in GROUPS)]
    rows = [(g, m) for g, _ in GROUPS for m in ms if m in v[g]]
    n_right = sum(v[g][m]["lo"] > 0 for g, m in rows)
    parts = []
    for g, head in GROUPS:
        got = [m for m in ms if m in v[g]]
        if not got:
            continue
        parts.append(HEAD_LONG[g] + ": " + "; ".join(f"{m} {row(g, m)}" for m in got))
    a = v["agree"]
    sizes = "; ".join(f"{m} {a[m]['n']} records from {a[m]['pics']} pictures, against "
                      f"{a[m]['n_cands']} published methods" for m in ms if m in a)
    fsz = "; ".join(f"{m} {v['fall'][m]['n']} records from {v['fall'][m]['pics']} pictures, against "
                    f"{v['fall'][m]['n_cands']} published methods" for m in ms if m in v["fall"])
    csz = ", ".join(f"{m} {v['auroc'][m]['n']} photographs" for m in ms if m in v["auroc"])
    no_run = [m for m in ms if m not in a and m not in v["fall"]]
    hits = ", ".join(("hit" if p["loc"]["vwrite_mean"].get("hit1") else "miss") + " for V-Write, "
                     + ("hit" if p["loc"]["attr_patch"].get("hit1") else "miss") + " for attribution patching"
                     for p in picks)
    right = (
        "Right: three COCO photographs, picked at the quartiles of object size before looking at any map, "
        f"on {pick_model}, labelled with the object asked about and its share of the frame; the top tenth "
        "of cells of V-Write (left, clay) and of attribution patching (right, slate), with the human "
        f"outline; top to bottom, the single top cell is a {hits}. ") if picks else ""
    return (
        f"MAIN. Against every published attribution method as run, V-Write's ranking of image patches agrees "
        f"better with measured patching and removes more of the answer: in {n_right} of {len(rows)} paired "
        "comparisons the whole 95% interval of V-Write minus the best other method lies right of zero. "
        "V-Write is layer integrated gradients taken from the per-picture mean image-token state at one "
        "block, so the integrated-gradients row (from a zero state at the same block) compares two starting "
        "points of one method, not two methods.\n\n"
        "Left: each row is one test on one model; the dot is the mean paired difference, V-Write minus "
        "the method that scored highest on that row among all run beside it (named in the row, chosen on "
        "the same data, which favours the baselines), and the line its 95% interval from 2,000 resamples of "
        "pictures, not records. Each measure is on its own scale, so the rows are aligned at zero and the "
        "value is printed on the right. SimonSays agreement is the Spearman correlation, over sixteen patch "
        "groups, between a method's summed attribution and the measured fall in the answer score when the "
        f"group is set to zero ({sizes}). The SimonSays fall is the drop in the answer's log-probability "
        "margin when the top fifth of image patches by each method is set to zero, a replacement V-Write "
        f"does not integrate from ({fsz}). Both SimonSays tests use one shared subsample of records per "
        "model, the baselines run, on which V-Write and every published method (attribution patching, AtP "
        "rather than AtP*; integrated gradients from a zero state; gradient x activation; activation "
        "patching one patch at a time; RISE; KernelSHAP; Grad-CAM) are all scored, because integrated "
        "gradients, RISE and KernelSHAP cost many forward passes per record; RISE and KernelSHAP run at "
        f"{_budget(v)}. COCO rows are yes/no questions on whether an object is present, graded against the "
        f"human mask ({csz}); the COCO interval is the run's own, and the COCO rows compare against the "
        "cheap methods run there only (attribution patching, gradient x activation, and a centre prior). "
        + right + "\n\n"
        "Values: " + ". ".join(parts) + ".\n\n"
        "Limits: one block near half depth per model; the COCO fall row replaces patches with the picture's "
        "mean state, V-Write's own counterfactual, which favours V-Write; "
        + ("" if any((r.get("budget") or {}).get("masks") for r in v["fall"].values()) else
           "RISE and KernelSHAP are below their published sample budgets and the KernelSHAP sampler draws "
           "coalitions uniformly rather than by the Shapley kernel; Grad-CAM is read at the language model's "
           "input image features, not at the vision tower's last block, and scores near chance; ")
        + (f"{', '.join(no_run)} has no baselines run, so its SimonSays rows are left out rather than "
           "compared against a partial set of methods; " if no_run else
           "a model without the baselines run would have no SimonSays rows rather than a comparison "
           "against a partial set of methods; ")
        + "sixteen groups are a coarse ground truth"
        + ("; three photographs are illustrations, not evidence." if picks else ".")
    )


def _import_caption():
    """The caption for whatever is on disk now, without the photographs (they need the COCO images)."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        v = _numbers(os.path.join(here, "results"), list(MODELS))
        return _caption(v, list(MODELS), [], None)
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ figure

def vwrite_gains(results="results", out="figures", model=None):
    """MAIN. V-Write against every published attribution method as run; replaced at render by the caption built from the files."""
    models = [model] if model else list(MODELS)
    v = _numbers(results, models)
    ms = [m for m in models if any(m in v[g] for g, _ in GROUPS)]
    if not ms:
        return None
    rows = [(g, m) for g, _ in GROUPS for m in ms if m in v[g]]
    if len(rows) < 3:
        return None

    # the photographs, from the biggest model that has COCO maps
    from vlens.figures import fx_coco as FC
    recs, root = FC._coco_records()
    pick_model, picks, photos = None, [], {}
    for m in sorted(ms, key=lambda m: "27B" not in m):
        crow, _ = _coco_rows(results, m)
        if crow and recs:
            picks = _pick(crow, recs)
            photos = {p["id"]: FC._photo(recs[p["id"]]["image"], root) for p in picks}
            picks = [p for p in picks if photos[p["id"]] is not None]
            if len(picks) == 3:
                pick_model = m
                break
    if len(picks) < 3:
        picks = []

    cap = _caption(v, models, picks, pick_model)
    vwrite_gains.__doc__ = cap
    print(cap)

    st.use()
    W = st.MAX_WIDTH
    fs = matplotlib.rcParams["xtick.labelsize"] * 0.72         # every text in the figure
    lh = fs * 1.42 / 72                                          # one row, inches
    lab_h = fs * 1.55 / 72                                       # a label over a photograph
    n_lines = len(rows) + sum(1 for g, _ in GROUPS if any(m in v[g] for m in ms))
    top_pad, bot_pad = 0.04, 0.30
    from vlens.figures import fx_layers as FL
    label_w = max(max(FL._width_in(h, fs) for g, h in GROUPS if any(m in v[g] for m in ms)),
                  max(FL._width_in("   " + _rowlabel(v, g, m), fs) for g, m in rows)) + 0.12
    val_w = max(FL._width_in(_valtext(v[g][m]), fs) for g, m in rows) + 0.1
    img_gap, pair_gap, TILE_MIN = 0.06, 0.1, 1.1

    # Two layouts, whichever keeps the photographs at a readable size without
    # leaving the forest floating in empty page: the photographs in a column
    # beside a forest tall enough to stand next to them, or in a row under a
    # short one, so a model with few rows does not shrink the pictures.
    H_nat = top_pad + n_lines * lh + bot_pad
    col_h = 3 * (lab_h + TILE_MIN) + 2 * pair_gap
    side_w = 2 * TILE_MIN + img_gap
    fw_side = W - label_w - val_w - side_w - 0.25
    side = bool(picks) and H_nat * 1.6 >= col_h + top_pad and fw_side >= 2.2
    if side:
        H = max(H_nat, col_h + top_pad)
        H_forest = H - top_pad - bot_pad
        tile = (H - top_pad - 3 * lab_h - 2 * pair_gap) / 3
        right_w = 2 * tile + img_gap
        fw = W - label_w - val_w - right_w - 0.25
        band = 0.0
    else:
        H_forest = n_lines * lh * (1.12 if picks else 1.0)
        fw = W - label_w - val_w
        if picks:
            tile = min(1.6, (W - 2 * 0.35 - 3 * img_gap) / 6)
            band = 0.1 + lab_h + tile
        else:
            tile = band = 0.0
        H = top_pad + H_forest + bot_pad + band
    fig = plt.figure(figsize=(W, H))
    fig.set_layout_engine("none")
    f_bot = band + bot_pad                                       # the forest's floor, inches
    ax = fig.add_axes([label_w / W, f_bot / H, fw / W, H_forest / H])

    # each group on its own scale, zero at the same place in every group
    # each group scaled by its largest distance from zero, so a loss reads as far left as a win reads right
    lo_frac = 1.06
    y = n_lines - 0.5
    for g, head in GROUPS:
        got = [m for m in ms if m in v[g]]
        if not got:
            continue
        M = max(max(abs(v[g][m]["hi"]), abs(v[g][m]["lo"])) for m in got) or 1.0
        fig.text(0.02 / W, (f_bot + y / n_lines * H_forest) / H, head, ha="left", va="center",
                 fontsize=fs, color=st.INK)
        y -= 1
        for m in got:
            r = v[g][m]
            u = lambda x: x / M  # noqa: E731
            ax.plot([u(r["lo"]), u(r["hi"])], [y, y], color=P.COLOUR["v_write"], lw=2.4,
                    solid_capstyle="butt", zorder=3)
            ax.plot([u(r["mean"])], [y], marker="o", ms=7.5, color=P.COLOUR["v_write"], mec="white",
                    mew=0.9, ls="none", zorder=4)
            yf = (f_bot + y / n_lines * H_forest) / H
            fig.text(0.02 / W, yf, "   " + _rowlabel(v, g, m), ha="left", va="center", fontsize=fs,
                     color=st.INK)
            fig.text((label_w + fw + 0.06) / W, yf, _valtext(r), ha="left", va="center", fontsize=fs,
                     color=st.INK)
            y -= 1
    ax.set_ylim(0, n_lines)
    ax.set_xlim(-lo_frac, 1.04)
    ax.axvline(0, color=st.INK, lw=1.1, zorder=2)
    ax.set_yticks([])
    ax.set_xticks([0])
    ax.set_xticklabels(["0"], fontsize=fs)
    ax.tick_params(axis="x", length=3, pad=2)
    ax.grid(False)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(st.AXIS)
    # the directions sit on the line of the zero tick, clear of it either side
    for txt, dx, ha in (("V-Write better", 9, "left"), ("worse", -9, "right")):
        ax.annotate(txt, xy=(0, 0), xycoords=("data", "axes fraction"), xytext=(dx, -5),
                    textcoords="offset points", ha=ha, va="top", fontsize=fs, color=st.INK)
    # faint separators between the groups
    y = n_lines
    for g, _ in GROUPS:
        got = [m for m in ms if m in v[g]]
        if not got:
            continue
        if y < n_lines:
            ax.axhline(y, color=st.GRID, lw=0.9, zorder=0)
        y -= 1 + len(got)

    if picks:
        meths = (("vwrite_mean", P.COLOUR["v_write"]), ("attr_patch", P.COLOUR["attr_patch"]))
        pair_w = 2 * tile + img_gap
        for j, p in enumerate(picks):
            if side:                 # a column at the right, one pair per row
                x0 = W - right_w
                top = H - top_pad - j * (lab_h + tile + pair_gap)
            else:                    # a row under the forest, centred
                gap = 0.35
                x0 = (W - 3 * pair_w - 2 * gap) / 2 + j * (pair_w + gap)
                top = band - 0.1
            fig.text((x0 + pair_w / 2) / W, (top - lab_h / 2) / H, _obj(p), ha="center", va="center",
                     fontsize=fs, color=st.INK)
            for k, (meth, col) in enumerate(meths):
                a = fig.add_axes([(x0 + k * (tile + img_gap)) / W, (top - lab_h - tile) / H,
                                  tile / W, tile / H])
                _tile(a, photos[p["id"]], recs[p["id"]], p, meth, col)
    hs = [Line2D([], [], color=P.COLOUR["v_write"], lw=2.4, marker="o", ms=7, mec="white"),
          Patch(fc=P.COLOUR["v_write"], alpha=0.62, ec="none"),
          Patch(fc=P.COLOUR["attr_patch"], alpha=0.62, ec="none"),
          Line2D([], [], color=st.INK, lw=1.3)]
    ls = ["V-Write minus best baseline, 95% interval", "V-Write top tenth", "attribution patching top tenth",
          "human outline"]
    if not picks:
        hs, ls = hs[:1], ls[:1]
    st.key_band(fig, hs, ls, fontsize=fs, ncol=len(hs), columnspacing=1.2, handlelength=1.5)
    return st.save(fig, P._at(out, "vwrite_gains"), check=True, tight=True)


def _obj(p):
    return f"{p['target']}, {100 * float(p['area']):.0f}%"


def _rowlabel(v, g, m):
    return f"{m}, vs {NAME.get(v[g][m]['base'], v[g][m]['base'])}"


def _valtext(r):
    return f"{_f(r['mean'], '{:+.2f}')} [{_f(r['lo'])}, {_f(r['hi'])}]"


#: ``tools/make_captions.py`` calls this for the caption, so the text is built from what is on disk
vwrite_gains.caption = _import_caption
