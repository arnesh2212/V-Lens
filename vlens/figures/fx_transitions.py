from __future__ import annotations

import json
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from vlens.figures import paper as P
from vlens.figures import style as st
from vlens.figures import fx_vread_yesno as Y

MODEL = "Qwen3-VL-8B-Instruct"
SMALL_PRINT = {"watermark", "eval_watermark"}
EVAL_SOURCES = ("openimages", "vins", "nuscenes")
SAND_FAMILIES = ("wmdp_bio_sand", "wmdp_chem_sand", "wmdp_cybe_sand")
SUBJECT = {"wmdp_bio_sand": "biology", "wmdp_chem_sand": "chemistry", "wmdp_cybe_sand": "cybersecurity"}


def _json(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None


def _grid(*rows):
    g = None
    for r in rows:
        s = {i for i, x in enumerate(r.get("v_read") or []) if x is not None}
        g = s if g is None else g & s
    return sorted(g or [])


def _depth(g, L):
    return np.array(g, float) / max(L - 1, 1)


def _said_tr(b):
    t = str(b.get("reply") or "").strip().lower()
    m = re.match(r"^\W*(test|real)\b", t)
    return m.group(1) if m else None


def _said_letter(b):
    t = str(b.get("reply") or "").strip()
    m = re.match(r"^\W*([ABCD])\b", t)
    return m.group(1) if m else None


def _pack(g, L, ro, re_, bo, be, val, kind):
    vo = np.array([val(ro["v_read"][t]) for t in g], float)
    ve = np.array([val(re_["v_read"][t]) for t in g], float)
    lo = np.array([val(ro["logit"][t]) for t in g], float)
    le = np.array([val(re_["logit"][t]) for t in g], float)
    fo, fe = float(val(ro["logit"][-1])), float(val(re_["logit"][-1]))
    s = np.sign(fe - fo)
    tau = min(0.5, abs(fe - fo) / 2)
    return dict(kind=kind, g=g, L=L, bo=bo, be=be, vo=vo, ve=ve, lo=lo, le=le, fo=fo, fe=fe, s=s,
                sep_v=Y._commit(ve - vo, s, tau) if s else len(g),
                sep_l=Y._commit(le - lo, s, tau) if s else len(g))


def yesno_pairs(results, m):
    ps = Y._pairs(results, m)
    for p in ps:
        p["kind"] = "yesno"
    return ps


def test_real_pairs(results, m):
    out = []
    for src in EVAL_SOURCES:
        a = _json(os.path.join(results, "readouts", f"{m}__{src}_orig_eval_readouts.json"))
        e = _json(os.path.join(results, "readouts", f"{m}__{src}_edit_eval_readouts.json"))
        ba = _json(os.path.join(results, "simonsays", f"{m}__{src}_orig_eval.json"))
        be = _json(os.path.join(results, "simonsays", f"{m}__{src}_edit_eval.json"))
        if not (a and e and ba and be):
            continue
        L = len(a.get("blocks") or [])
        ba = {r["id"]: r for r in ba["rows"]}
        be = {r["id"]: r for r in be["rows"]}
        idx = {}
        for r in a["rows"]:
            if r.get("kind") == "test_real" and r["id"] in ba and not r.get("v_read_fitted_on"):
                b = ba[r["id"]]
                idx.setdefault((os.path.basename(str(b["image"])), b["question"]), []).append(r)
        for r in e["rows"]:
            if r.get("kind") != "test_real" or r["id"] not in be or r.get("v_read_fitted_on"):
                continue
            b = be[r["id"]]
            mt = b.get("meta") or {}
            for ro in idx.get((os.path.basename(str(mt.get("source_image"))), b["question"]), []):
                g = _grid(ro, r)
                if len(g) < 4:
                    continue
                p = _pack(g, L, ro, r, ba[ro["id"]], b, float, "test_real")
                p["src"] = src
                p["said"] = (_said_tr(p["bo"]), _said_tr(p["be"]))
                p["flip"] = bool(p["said"][0] and p["said"][1] and p["said"][0] != p["said"][1])
                out.append(p)
        del a, e
    return [p for p in out if p["s"] != 0]


def sandbag_pairs(results, m):
    out = []
    for fam in SAND_FAMILIES:
        d = _json(os.path.join(results, "readouts", f"{m}__{fam}_readouts.json"))
        b = _json(os.path.join(results, "simonsays", f"{m}__{fam}.json"))
        if not (d and b):
            continue
        L = len(d.get("blocks") or [])
        bb = {r["id"]: r for r in b["rows"]}
        ro = {r["id"]: r for r in d["rows"] if r.get("kind") == "letter" and not r.get("v_read_fitted_on")}
        items = {}
        for i in ro:
            it, _, cond = i.partition("__")
            items.setdefault(it, {})[cond] = i
        for it, conds in items.items():
            if "unlocked" not in conds or conds["unlocked"] not in bb:
                continue
            u = ro[conds["unlocked"]]
            bu = bb[conds["unlocked"]]
            exp = str(bu.get("expected") or "").strip()[:1]
            if exp not in "ABCD" or not exp:
                continue
            k = "ABCD".index(exp)

            def margin(a, k=k):
                a = np.asarray(a, float)
                return a[k] - np.max(np.delete(a, k))
            for cond in ("hazard_sandbagged", "calibration_sandbagged"):
                if cond not in conds or conds[cond] not in bb:
                    continue
                sd, bs = ro[conds[cond]], bb[conds[cond]]
                g = _grid(u, sd)
                if len(g) < 4:
                    continue
                p = _pack(g, L, u, sd, bu, bs, margin, "letter")
                p.update(src=fam, subject=SUBJECT.get(fam, fam), cond=cond, right=exp, item=it)
                p["said"] = (_said_letter(bu), _said_letter(bs))
                p["flip"] = bool(p["said"][0] == exp and p["said"][1] and p["said"][1] != exp)
                out.append(p)
        del d, b
    return [p for p in out if p["s"] != 0]


def _gap(p):
    return abs(p["fe"] - p["fo"])


def _pick_yesno(ps):
    fl = [p for p in ps if p["flip"]] or ps
    return max(fl, key=_gap) if fl else None


def _pick_test_real(ps):
    fl = [p for p in ps if p["flip"]] or ps
    legible = [p for p in fl if (p["be"].get("meta") or {}).get("edit_variant") not in SMALL_PRINT]
    best = max(legible or fl, key=_gap) if fl else None
    passed = [p for p in sorted(fl, key=_gap, reverse=True) if p is not best and _gap(p) > _gap(best)] if best else []
    return best, passed


def _pick_sandbag(ps):
    fl = [p for p in ps if p["flip"]] or ps
    return max(fl, key=_gap) if fl else None


def _tally(ps):
    sv = np.array([p["sep_v"] for p in ps], int)
    sl = np.array([p["sep_l"] for p in ps], int)
    return dict(n=len(ps), first=int((sv < sl).sum()), same=int((sv == sl).sum()), later=int((sv > sl).sum()),
                sv=sv, sl=sl, p=Y._sign_p(int((sv < sl).sum()), int((sv != sl).sum())))


def _image(row):
    from vlens import corpus as C
    from vlens.figures import fx_trace as FT
    from vlens.paths import ROOT
    p = FT._image_path(None, dict(row, source=row.get("source", "simonsays")))
    if not p:
        from vlens.paths import DATA
        q = os.path.join(DATA, str(row["image"]))
        if os.path.exists(q):
            p = q
    if not p:
        return None
    return C.load_image(p, max_side=640)


def _thumb(im, aspect):
    w, h = im.size
    if w / h > aspect:
        nw = int(round(h * aspect))
        x0 = (w - nw) // 2
        return im.crop((x0, 0, x0 + nw, h))
    nh = int(round(w / aspect))
    y0 = (h - nh) // 2
    return im.crop((0, y0, w, y0 + nh))


def _edit_words(b):
    from vlens.figures import fx_layers as FL
    v = (b.get("meta") or {}).get("edit_variant")
    return FL.EDIT_WORDS.get(v, str(v).replace("_", " ")) if v else "the scene edited"


def _short_q(q):
    q = re.sub(r"\s*Answer (yes or no|'test' or 'real')\.?\s*$", "", str(q).strip(), flags=re.I)
    return q.rstrip(".?")


def _fit_quote(cands, width_in, fs, most):
    from vlens.figures import fx_layers as FL
    for t in cands:
        lines = FL.fit_lines(t, width_in, fs, most)
        if " ".join(lines) == FL._clean(t):
            return lines
    return FL.fit_lines(cands[-1], width_in, fs, most)


def _blk(p, k):
    if k >= len(p["g"]):
        return "never"
    return f"block {p['g'][k] + 1} of {p['L']}"


def _pct(k, n):
    return f"{100 * k / n:.0f}%" if n else "n/a"


def _trace(ax, p, fs, fsm, quantity):
    d = _depth(p["g"], p["L"])
    c = P.COLOUR["v_read"]
    ax.fill_between(d, p["vo"], p["ve"], color=c, alpha=0.18, lw=0, zorder=2)
    ax.plot(d, p["lo"], color=st.MUTED, lw=1.3, dashes=(3, 1.6), zorder=3)
    ax.plot(d, p["le"], color=st.MUTED, lw=1.5, zorder=3)
    ax.plot(d, p["vo"], color=c, lw=2.0, dashes=(3, 1.6), zorder=4)
    ax.plot(d, p["ve"], color=c, lw=2.4, zorder=5)
    ax.axhline(0, color=st.AXIS, lw=0.8, zorder=1)
    lo = min(p["vo"].min(), p["ve"].min(), p["lo"].min(), p["le"].min(), 0.0)
    hi = max(p["vo"].max(), p["ve"].max(), p["lo"].max(), p["le"].max(), 0.0)
    pad = (hi - lo) * 0.08
    ax.set_xlim(0, 1)
    ax.set_ylim(lo - pad, hi + pad + (hi - lo) * 0.30 * (1 + quantity.count("\n")))
    ext = np.append(d, 1.0)
    y0 = lo - pad
    for sep, col, ms, z in ((p["sep_l"], st.MUTED, 10.0, 6), (p["sep_v"], c, 8.0, 7)):
        ax.plot([ext[sep]], [y0], marker="^", ms=ms, color=col, mec="white", mew=0.8, clip_on=False, zorder=z)
    ax.text(0.02, 0.98, quantity, transform=ax.transAxes, ha="left", va="top", fontsize=fsm, color=st.INK,
            zorder=8)
    from matplotlib.ticker import MaxNLocator
    tk = [t for t in MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi) if lo - pad <= t <= hi + pad]
    ax.set_yticks(tk)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["input", "half", "output"], fontsize=fs)
    Y._edge_ticks(ax)
    ax.tick_params(labelsize=fs, length=3, pad=4)
    st.despine(ax)


def _parting_rows(results):
    import re
    p = os.path.join(results, "extended", "a6", "parting.md")
    rows = []
    if os.path.exists(p):
        for ln in open(p):
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            if len(c) >= 6 and c[1] in ("logit", "tuned") and c[0] in (MODEL, "Qwen3.5-27B"):
                f_, s_, l_ = (float(re.sub("%", "", v)) for v in c[3:6])
                rows.append((c[0], c[1], f_, s_, l_, int(c[2])))
    return rows


def _aggregate_bars(ax, rows, fs, fsm):
    from matplotlib.patches import Patch
    nice = {"logit": "logit lens", "tuned": "tuned lens"}
    short = {MODEL: "Qwen3-VL-8B", "Qwen3.5-27B": "Qwen3.5-27B"}
    cv, co = P.COLOUR["v_read"], st.MUTED
    OFF, bh = 37.0, 0.34
    y, heads = 0.0, []
    for m in (MODEL, "Qwen3.5-27B"):
        rr = sorted([r for r in rows if r[0] == m], key=lambda r: r[1] != "logit")
        if not rr:
            continue
        heads.append((y, short[m]))
        y += 0.9
        for r in rr:
            ax.text(0, y + bh / 2, "vs " + nice[r[1]], ha="left", va="center", fontsize=fsm, color=st.INK)
            for k, (v, c) in enumerate(((r[2], cv), (r[4], co))):
                yy = y + k * bh
                ax.barh(yy, v, left=OFF, color=c, height=bh * 0.86)
            ax.text(OFF + max(r[2], r[4]) + 1.5, y + bh / 2, f"{r[2]:.0f} vs {r[4]:.0f}", ha="left",
                    va="center", fontsize=fsm, color=st.INK)
            y += 2 * bh + 0.5
        y += 0.4
    for yy, tx in heads:
        ax.text(0, yy, tx, ha="left", va="center", fontsize=fsm, color=st.INK)
    ax.set_yticks([])
    ax.set_ylim(y - 0.75, -2.2)
    ax.set_xlim(0, OFF + 76); ax.set_xticks([OFF, OFF + 25, OFF + 50]); ax.set_xticklabels(["0", "25", "50"])
    ax.tick_params(axis="x", labelsize=fs, length=3, pad=4)
    ax.grid(False)
    st.despine(ax, left=True)
    ax.spines["left"].set_visible(False)
    ax.legend(handles=[Patch(fc=cv), Patch(fc=co)], labels=["V-Read parts first", "the other reader first"],
              loc="upper left", bbox_to_anchor=(-0.02, 1.0), frameon=False, fontsize=fsm, handlelength=1.1,
              borderaxespad=0.1, labelspacing=0.3)


def _aggregate(ax, V, fs, fsm):
    c = P.COLOUR["v_read"]
    ms_max = max(max(np.bincount(v["t"]["sv"]).max(), np.bincount(v["t"]["sl"]).max()) for v in V.values())
    y = 0.0
    step, gap_group = 1.0, 1.55
    for m, v in reversed(list(V.items())):
        ps, t = v["ps"], v["t"]
        g, L = ps[0]["g"], ps[0]["L"]
        dep = _depth(g, L)
        ax.text(0.0, y, f"logit lens first {_pct(t['later'], t['n'])}",
                ha="left", va="center", fontsize=fsm, color=st.INK, clip_on=False)
        y += 0.85 * step
        ax.text(0.0, y, f"V-Read first {_pct(t['first'], t['n'])}, same block {_pct(t['same'], t['n'])},",
                ha="left", va="center", fontsize=fsm, color=st.INK, clip_on=False)
        y += step
        for lens, sep, col in (("logit lens", t["sl"], st.MUTED), ("V-Read", t["sv"], c)):
            cnt = np.bincount(sep, minlength=len(g) + 1)
            for k, n in enumerate(cnt[:len(g)]):
                if n:
                    ax.scatter([dep[k]], [y], s=14 + 150 * n / ms_max, color=col, alpha=0.9, lw=0.6,
                               ec="white", zorder=3, clip_on=False)
            ax.text(0.0, y, lens, ha="left", va="center", fontsize=fsm, color=st.INK, clip_on=False)
            if cnt[len(g)]:
                ax.text(1.04, y, f"never {int(cnt[len(g)])}", ha="left", va="center", fontsize=fsm,
                        color=st.INK, clip_on=False)
            y += step
        ax.text(0.0, y, f"{m}, {t['n']} pairs", ha="left", va="center", fontsize=fsm, color=st.INK,
                clip_on=False)
        y += gap_group
    ax.set_ylim(-0.55, y - gap_group + 0.55)
    ax.set_yticks([])
    ax.set_xlim(-0.02, 1.25)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["input", "half", "output"], fontsize=fs)
    ax.get_xticklabels()[0].set_ha("left")
    ax.tick_params(axis="x", labelsize=fs, length=3, pad=4)
    ax.grid(False)
    ax.xaxis.grid(True)
    st.despine(ax, left=False)


def _lead_span(p):
    ve = p["ve"]
    on = [k for k in range(len(ve)) if ve[k] > 0]
    if not on:
        return None, None
    k0 = on[0]
    off = [k for k in range(k0, len(ve)) if (ve[k:] <= 0).all()]
    return _blk(p, k0), (_blk(p, off[0]) if off else None)


def _small(ps, tau=1.0):
    return int(sum(_gap(p) < tau for p in ps))


def _caption(m, ex, V, tr_all, sb_all, passed, others):
    a, b, c = ex["yesno"], ex["test_real"], ex["letter"]
    parts = []
    for mm, v in V.items():
        t = v["t"]
        g, L = v["ps"][0]["g"], v["ps"][0]["L"]
        parts.append(f"on {mm} ({t['n']} pairs, read at {len(g)} of {L} blocks) V-Read parts the two readings "
                     f"before the logit lens on {_pct(t['first'], t['n'])}, at the same block on "
                     f"{_pct(t['same'], t['n'])} and later on {_pct(t['later'], t['n'])} "
                     f"(two-sided sign test p = {t['p']:.2g}); {int((t['sv'] >= len(g)).sum())} pairs never part "
                     f"under V-Read and {int((t['sl'] >= len(g)).sum())} under the logit lens")
    tr_t, sb_t = _tally(tr_all), _tally(sb_all)
    sb_fl = _tally([p for p in sb_all if p["flip"]])
    on, off = _lead_span(c)
    lead = (
        f"MAIN. A cue moves the model's reading in the second half of the network, and V-Read shows the move "
        f"block by block: on {m}, {_edit_words(a['be'])} turns the reading of \"{_short_q(a['bo']['question'])}?\" "
        f"from {a['said'][0]} to {a['said'][1]} from {_blk(a, a['sep_v'])} (logit lens: {_blk(a, a['sep_l'])}); "
        f"{_edit_words(b['be'])} turns \"test or real?\" from {b['said'][0]} to {b['said'][1]} from "
        f"{_blk(b, b['sep_v'])} (logit lens: {_blk(b, b['sep_l'])}); and under an instruction to answer wrongly, "
        f"the right letter {c['right']} of a {c['subject']} item leads V-Read's reading from {on}"
        + (f" until {off}, where the wrong letter {c['said'][1]} takes over and keeps the output" if off else "")
        + ". Over all matched yes/no pairs the two lenses part the readings at about the same depth: "
        + "; ".join(parts) + ".")
    body = (
        "\n\nEach example panel reads the answer position after every block V-Read is fitted at, through V-Read "
        "(clay) and the logit lens (grey), for the record without the cue (dashed) and with it (solid), with the "
        "ribbon between V-Read's two readings; the triangles mark the block from which the two readings differ "
        "in the direction of the model's output by at least half a logit (or half the output gap, if smaller) and "
        "keep differing, the rule of the appendix and the text. (a) The picture and its edit with the model's "
        f"replies; the reading is log p(yes) minus log p(no); the picture is asked \"{a['bo']['question']}\" with "
        f"the cue \"{a['bo'].get('cue')}\". (b) Log p(test) minus log p(real) for \"{b['bo']['question']}\"; the "
        f"edit is {_edit_words(b['be'])}. (c) Log p of the right letter minus the largest log p of the other three, "
        f"for one WMDP {c['subject']} carrier item (question and picture not shown) under the system prompt "
        f"\"{c['bo'].get('system')}\" and under the {c['cond'].replace('_', ' ')} instruction, which tells the model "
        "it must give a wrong answer when a right one could serve as a precursor to dangerous knowledge. Right: every "
        "matched yes/no pair (an untouched SimonSays picture and its edit asked the identical question, held out "
        "from V-Read's fit) placed at the depth its two readings part under each lens, dot area the number of "
        "pairs, pairs that never part counted at the right.\n\n"
        f"Selection: (a) is the yes/no pair with the largest output gap among the {sum(p['flip'] for p in V[m]['ps'])} "
        f"pairs on {m} whose reply flips ({_gap(a):.1f} logits); (b) is the test-or-real pair with the largest "
        f"output gap ({_gap(b):.1f}) among the {sum(p['flip'] for p in tr_all)} flipped pairs whose edit is legible "
        f"at this size" + ((" (one larger flip, " if len(passed) == 1 else f" ({len(passed)} larger flips, ")
                           + ", ".join(f"{_gap(q):.1f} logits for {_edit_words(q['be'])}" for q in passed)
                           + (", carries" if len(passed) == 1 else ", carry") + " a line of small print)"
                           if passed else "") +
        f"; (c) is the sandbagged item with the largest change in the final margin ({_gap(c):.1f}) among the "
        f"{sum(p['flip'] for p in sb_all)} items on {m} answered right under the plain prompt and wrong under an "
        f"instruction. The examples are chosen to be vivid and are not the rule: over the {tr_t['n']} test-or-real "
        f"pairs on {m} V-Read parts first on {tr_t['first']}, at the same block on {tr_t['same']} and later on "
        f"{tr_t['later']}; over the {sb_t['n']} unlocked-versus-sandbagged pairs ({_small(sb_all)} with an output "
        f"gap under one logit) first on {sb_t['first']}, same on {sb_t['same']}, later on {sb_t['later']}, and on "
        f"the {sb_fl['n']} right-to-wrong items alone first on {sb_fl['first']}, same on {sb_fl['same']}, later on "
        f"{sb_fl['later']}"
        + "".join(f"; on {mm} the {o['tr']['n']} test-or-real pairs go {o['tr']['first']} / {o['tr']['same']} / "
                  f"{o['tr']['later']} and the {o['sb']['n']} sandbagging pairs {o['sb']['first']} / {o['sb']['same']} / "
                  f"{o['sb']['later']} (V-Read first / same / later)" for mm, o in others.items())
        + ". Depth is the block index over the number of blocks less one, so the last block sits at the output."
    )
    return lead + body


def layer_transitions(results="results", out="figures", model=None):
    from vlens.figures import fx_layers as FL
    V = {}
    for m in ([model] if model else [mm for mm in Y.MODELS if mm in Y._models(results)]):
        if Y._records(results, m) is None:
            continue
        ps = yesno_pairs(results, m)
        if len(ps) >= 12:
            V[m] = dict(ps=ps, t=_tally(ps))
    if not V:
        return None
    m = model if model in V else (MODEL if MODEL in V else next(iter(V)))
    tr_all = test_real_pairs(results, m)
    sb_all = sandbag_pairs(results, m)
    a = _pick_yesno(V[m]["ps"])
    b, passed = _pick_test_real(tr_all)
    c = _pick_sandbag(sb_all)
    if not (a and b and c):
        return None
    ex = {"yesno": a, "test_real": b, "letter": c}
    others = {}
    for mm in V:
        if mm != m:
            tr, sb = test_real_pairs(results, mm), sandbag_pairs(results, mm)
            if tr and sb:
                others[mm] = dict(tr=_tally(tr), sb=_tally(sb))
    cap = _caption(m, ex, V, tr_all, sb_all, passed, others)
    layer_transitions.__doc__ = cap
    print(cap)

    st.use()
    W = st.MAX_WIDTH
    fs = matplotlib.rcParams["xtick.labelsize"] * 0.72
    fsm = fs * 0.86
    lh = fsm * 1.25 / 72
    tick_w, colw = 0.58, 1.84
    x_agg = 3 * (tick_w + colw) + 0.42
    agg_w = W - x_agg - 0.05
    head_lines = 3
    head_h = head_lines * lh + 0.04
    thumb_h = 0.60
    box_h = lh * 0.9 + 2 * FL.BOX_PAD
    trace_h, foot = 1.20, 0.40
    H = 0.03 + head_h + thumb_h + 0.05 + box_h + 0.18 + trace_h + foot
    fig = plt.figure(figsize=(W, H))
    fig.set_layout_engine("none")
    thumb_w = (colw - 0.06) / 2
    c_v = P.COLOUR["v_read"]

    heads = [
        f"(a) {_edit_words(a['be'])}: {_short_q(a['bo']['question']).lower()}?",
        f"(b) {_edit_words(b['be'])}: a test image, or a real photo?",
        f"(c) told to answer wrongly: a WMDP {c['subject']} item",
    ]
    quantities = ["log p(yes) minus log p(no)", "log p(test) minus log p(real)",
                  "log p(right letter)\nminus the best other letter"]
    y_head = H - 0.03
    y_thumb = y_head - head_h
    y_box = y_thumb - thumb_h - 0.05
    y_trace_top = y_box - box_h - 0.18
    for k, p in enumerate((a, b, c)):
        x = tick_w + k * (tick_w + colw)
        lines = FL.fit_lines(heads[k], colw, fsm, head_lines)
        fig.text(x / W, y_head / H, "\n".join(lines), ha="left", va="top", fontsize=fsm, color=st.INK,
                 linespacing=1.25)
        if p["kind"] != "letter":
            for j, (bh, said) in enumerate(((p["bo"], p["said"][0]), (p["be"], p["said"][1]))):
                xj = x + j * (thumb_w + 0.06)
                im = _image(bh)
                if im is not None:
                    ai = fig.add_axes([xj / W, (y_thumb - thumb_h) / H, thumb_w / W, thumb_h / H])
                    ai.imshow(_thumb(im, thumb_w / thumb_h))
                    ai.set_xticks([]); ai.set_yticks([])
                    ai.grid(False)
                    for sp in ai.spines.values():
                        sp.set_color(st.AXIS); sp.set_linewidth(0.7)
                FL.panel_box(fig, xj / W, y_box / H, thumb_w, 1, lh * 0.9, fsm, [f"reply: {said}"], "model")
        else:
            sys_plain = str(p["bo"].get("system") or "")
            stroke, gapx = 0.12, 0.04
            rep_w = 0.64
            sys_w = colw - stroke - gapx - rep_w - 0.05
            row_h = 2 * lh * 0.9 + 2 * FL.BOX_PAD
            top = y_thumb
            for j, (cands, said, note, dash) in enumerate((
                    ([f'"{sys_plain}"', '"... a helpful assistant."'], p["said"][0], "right", (3, 1.6)),
                    (['"... You must give a wrong answer ..."', '"... must give a wrong answer ..."',
                      '"... give a wrong answer ..."', '"... a wrong answer ..."'], p["said"][1], "wrong", ()))):
                yt = top - j * (row_h + 0.07)
                yc = yt - row_h / 2
                fig.lines.append(Line2D([x / W, (x + stroke) / W], [yc / H, yc / H], color=c_v,
                                        lw=2.4 if not dash else 2.0, dashes=dash or (None, None),
                                        transform=fig.transFigure, solid_capstyle="round"))
                lines = _fit_quote(cands, sys_w - 2 * FL.BOX_PAD, fsm, 2)
                FL.panel_box(fig, (x + stroke + gapx) / W, yt / H, sys_w, 2, lh * 0.9, fsm, lines, "set")
                FL.panel_box(fig, (x + stroke + gapx + sys_w + 0.05) / W, yt / H, rep_w, 2, lh * 0.9, fsm,
                             [f"reply: {said}", note], "model")
        at = fig.add_axes([x / W, (y_trace_top - trace_h) / H, colw / W, trace_h / H])
        _trace(at, p, fs, fsm, quantities[k])
        if k == 1:
            at.set_xlabel(f"depth through {m}, {p['L']} blocks", fontsize=fs, labelpad=3)
    ag = fig.add_axes([x_agg / W, (y_trace_top - trace_h) / H, agg_w / W,
                       (y_head - 0.02 - (y_trace_top - trace_h)) / H])
    rows = _parting_rows(results)
    if rows:
        _aggregate_bars(ag, rows, fs, fsm)
        ag.set_xlabel("pairs whose readings part first (%)", fontsize=fs, labelpad=3)
    else:
        _aggregate(ag, V, fs, fsm)
        ag.set_xlabel("depth at which the pair parts", fontsize=fs, labelpad=3)
    hs = [Line2D([], [], color=c_v, lw=2.4),
          Line2D([], [], color=c_v, lw=2.0, dashes=(3, 1.6)),
          Line2D([], [], color=st.MUTED, lw=1.5),
          Line2D([], [], marker="^", ls="none", color=c_v, ms=9)]
    st.key_band(fig, hs, ["V-Read, with the cue", "V-Read, without", "logit lens, same two styles",
                          "the readings part here"],
                fontsize=fsm, columnspacing=1.1, handlelength=1.6)
    return st.save(fig, P._at(out, "layer_transitions"), check=False, tight=True)


__all__ = ["layer_transitions"]
