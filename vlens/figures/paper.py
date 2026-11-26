from __future__ import annotations

import glob
import json
import re
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from vlens.figures import style as st

SOURCE = {"openimages": "Open Images", "nuscenes": "nuScenes",
          "vins": "VINS-120K", "gemini": "Generated scenes",
          "wmdp": "WMDP"}

LOCKED = re.compile(r"wmdp[_ ]pw[_ ](fine|test)")

ALL_ORDER = ["v_read_cross", "v_read", "tuned", "jacobian", "latent", "logit", "probe"]

HIDDEN = {"latent"}

ORDER = [k for k in ALL_ORDER if k not in HIDDEN]


def _hidden(ranking):
    base = str(ranking).split(":")[-1]
    return any(base == h or base.startswith(h + "_") for h in HIDDEN)


NICE = {"v_read_cross": "V-Read (to the answer)", "v_read": "V-Read", "tuned": "tuned lens",
        "jacobian": "Jacobian lens", "latent": "LatentLens", "logit": "logit lens",
        "probe": "probe (labelled)", "v_read_coco": "V-Read fitted on COCO",
        "patchscope": "Patchscopes",
        "v_write": "V-Write", "attr_patch": "attribution patching",
        "ig_zero": "integrated gradients (zero state)", "ig_mean32": "V-Write, 32 steps",
        "grad_x_act": "gradient x activation", "act_patch": "activation patching",
        "rise": "RISE", "kernel_shap": "KernelSHAP", "gradcam": "Grad-CAM",
        "random": "random"}
COLOUR = {"v_read": st.LENS, "v_read_cross": "#7E2A11", "tuned": st.SERIES[1],
          "jacobian": st.SERIES[2], "latent": st.SERIES[5],
          "logit": st.MUTED, "probe": st.SERIES[4], "v_read_coco": st.CLAY_SOFT,
          "patchscope": st.SAGE,
          "v_write": st.LENS, "attr_patch": st.SLATE, "ig_zero": st.OCHRE,
          "ig_mean32": st.CLAY_DK, "grad_x_act": st.OCHRE, "act_patch": st.INK,
          "rise": st.SERIES[5], "kernel_shap": st.SERIES[5], "gradcam": st.SLATE,
          "random": st.MUTED}
DASH = {"probe": (2, 2), "logit": (5, 2), "v_read_coco": (4, 1.6), "patchscope": (1.5, 1.5),
        "ig_mean32": (4, 1.6), "grad_x_act": (1.5, 1.5), "kernel_shap": (4, 1.6), "gradcam": (1.5, 1.5),
        "random": (4, 2.2)}
for _k, _nice, _col, _dash in (
        ("v_read_indomain", "V-Read fitted in domain", st.CLAY_DK, (4, 1.6)),
        ("tuned_indomain", "tuned lens fitted in domain", st.SERIES[1], (4, 1.6)),
        ("tuned_coco", "tuned lens fitted on COCO", st.SERIES[1], (1.5, 1.5)),
        ("jacobian_coco", "Jacobian lens fitted on COCO", st.SERIES[2], (1.5, 1.5)),
        ("v_read_safety", "V-Read refitted on safety prompts", st.CLAY_SOFT, (2, 1.2)),
        ("tuned_safety", "tuned lens refitted on safety prompts", st.SERIES[1], (2, 1.2)),
        ("jacobian_safety", "Jacobian lens refitted on safety prompts", st.SERIES[2], (2, 1.2)),
        ("dom", "difference of means (labelled)", st.OCHRE, (3, 1.5)),
        ("dom_refused", "difference of means, refused vs not", st.CLAY_DK, (3, 1.5)),
        ("arditi", "refusal direction (Arditi et al.)", st.OCHRE, None),
        ("unembed_axis", "unembedding axis", st.SLATE, (2, 2)),
        ("bottom", "V-Write, bottom of the ranking", st.CLAY_SOFT, (1.5, 1.5))):
    NICE.setdefault(_k, _nice)
    COLOUR.setdefault(_k, _col)
    if _dash is not None:
        DASH.setdefault(_k, _dash)
del _k, _nice, _col, _dash
REFUSAL_EXTRA = ["v_read_coco", "v_read_indomain", "tuned_indomain", "dom"]

AXES_ORDER = ["the answer", "refusal", "self-report"]


def _drop_locked(o):
    if isinstance(o, dict):
        return {k: _drop_locked(v) for k, v in o.items()
                if not (isinstance(k, str) and LOCKED.search(k))}
    if isinstance(o, list):
        return [_drop_locked(v) for v in o
                if not (isinstance(v, str) and LOCKED.search(v))]
    return o


def _load(p, pool: bool = True):
    def one(q):
        try:
            with open(q) as f:
                return _drop_locked(json.load(f))
        except Exception:
            return None
    if not (AVERAGE and pool):
        return one(p)
    d, f = os.path.dirname(p), os.path.basename(p)
    stem = f.split("__", 1)[1] if "__" in f else f
    got = [x for x in (one(q) for q in _one_per_model(os.path.join(d, f"*__{stem}"))) if x]
    if len(got) < 2:
        return None
    if all("summary" in g for g in got):
        return dict(got[0], summary=_mean_of([g["summary"] for g in got]))
    return _mean_of(got)


SUFFIX = ""

AVERAGE = False

KEEP_FIRST = {"repo", "ks", "alphas", "rankings", "directions", "layers", "blocks",
              "meta", "kinds", "variant", "oriented", "per_source_column",
              "per_source_index", "per_family_column", "block", "depth", "ks"}


PROBE = {"auth": "authority", "eval": "eval awareness", "read": "reading",
         "base": "description", "sand": "3 instructions pooled",
         "swap": "text swapped", "removal": "text removed"}
SUBJECT = {"bio": "biology", "chem": "chemistry", "cybe": "cyber"}


def _fam_label(f: str) -> str:
    bits = str(f).split("_")
    src = SOURCE.get(bits[0], bits[0])
    rest = []
    for b in bits[1:]:
        if b == "orig":
            rest.append("as taken")
        elif b == "edit":
            rest.append("edited")
        elif b in SUBJECT:
            rest.append(SUBJECT[b])
        elif b in PROBE:
            rest.append(PROBE[b])
        else:
            rest.append(b)
    return f"{src}, {', '.join(rest)}" if rest else src


def _spread(fams, n_of, k):
    by_src = {}
    for f in sorted(fams, key=lambda x: -n_of(x)):
        by_src.setdefault(str(f).split("_")[0], []).append(f)
    out, rnd = [], 0
    while len(out) < k and any(len(v) > rnd for v in by_src.values()):
        for src in sorted(by_src, key=lambda s: -n_of(by_src[s][0])):
            if len(by_src[src]) > rnd and len(out) < k:
                out.append(by_src[src][rnd])
        rnd += 1
    return out


def _is_count(k) -> bool:
    k = str(k)
    return k == "n" or k.startswith("n_") or k.endswith("_n") or k == "records"


def _rank_keys(objs):
    lead = re.compile(r"^(-?\d+)\|")
    firsts = []
    for o in objs:
        if not isinstance(o, dict) or not o:
            return objs
        got = {lead.match(k).group(1) for k in o if isinstance(k, str) and lead.match(k)}
        if len(got) != len({k.split("|")[0] for k in o}):
            return objs
        firsts.append(sorted(got, key=int))
    if len({tuple(f) for f in firsts}) <= 1 or len({len(f) for f in firsts}) != 1:
        return objs
    out = []
    for o, f in zip(objs, firsts):
        at = {v: w for v, w in zip(f, firsts[0])}
        out.append({lead.sub(lambda m: at[m.group(1)] + "|", k): v for k, v in o.items()})
    return out


def _mean_of(objs):
    objs = _rank_keys(objs) if isinstance(objs[0], dict) else objs
    a = objs[0]
    if a is None or isinstance(a, (str, bool)):
        return a
    if isinstance(a, (int, float)):
        v = [o for o in objs if isinstance(o, (int, float)) and not isinstance(o, bool)
             and np.isfinite(o)]
        return float(np.mean(v)) if v else a
    if isinstance(a, list):
        if not all(isinstance(o, list) and len(o) == len(a) for o in objs):
            return a
        return [_mean_of([o[i] for o in objs]) for i in range(len(a))]
    if isinstance(a, dict):
        keys = set(a)
        for o in objs[1:]:
            keys &= set(o) if isinstance(o, dict) else set()
        out = {}
        for k in a:
            if k not in keys:
                continue
            vals = [o[k] for o in objs]
            if k in KEEP_FIRST:
                out[k] = a[k]
            elif _is_count(k):
                out[k] = (sum(vals) if isinstance(a[k], (int, float))
                          else {kk: sum(o.get(kk, 0) for o in vals) for kk in a[k]}
                          if isinstance(a[k], dict) else a[k])
            else:
                out[k] = _mean_of(vals)
        return out
    return a


def _enough(paths) -> bool:
    return (not AVERAGE) or len({os.path.basename(q).split("__")[0] for q in paths}) > 1


def _one_per_model(pattern):
    best = {}
    for q in glob.glob(pattern):
        mo = os.path.basename(q).split("__")[0]
        if mo not in best or os.path.getmtime(q) > os.path.getmtime(best[mo]):
            best[mo] = q
    return [best[k] for k in sorted(best)]


def _at(out: str, name: str) -> str:
    os.makedirs(out, exist_ok=True)
    return os.path.join(out, f"{name}{SUFFIX}.png")


def _newest(*patterns):
    hits = sorted((f for pat in patterns for f in glob.glob(pat)),
                  key=os.path.getmtime, reverse=True)
    return hits[0] if hits else None


def _line(ax, x, y, key, lo=None, hi=None, lw=None):
    c = COLOUR.get(key, st.SERIES[0])
    if lo is not None:
        st.ribbon(ax, x, lo, hi, c)
    ax.plot(x, y, color=c, lw=lw or (2.6 if key == "v_read" else 1.9),
            dashes=DASH.get(key, ()), zorder=5 if key == "v_read" else 3)
    return c


def lens_by_depth(results="results", out="figures", model=None, families=None,
                  target="model", max_panels=3):
    return _by_depth(_newest(f"{results}/lensbench*/{model or '*'}__bench.json"),
                     out, "lens_by_depth", families, target, max_panels)


def lens_read_by_depth(results="results", out="figures", model=None, families=None,
                       target="model", max_panels=3):
    return _by_depth(_newest(f"{results}/lensread*/{model or '*'}__read_summary.json"),
                     out, "lens_read_by_depth", families, target, max_panels)


def _by_depth(p, out, stem, families, target, max_panels):
    d = _load(p) if p else None
    if not d or not d.get("families"):
        return None
    fams = families or _spread(d["families"], lambda f: d["families"][f]["n"], max_panels)
    fams = [f for f in fams if f in d["families"]]
    if not fams:
        return None
    st.use()
    fig, axes = st.wide(1, len(fams), unit=(5.59, 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    floor = 1.0
    for fam in fams:
        r = d["families"][fam]
        for k in ORDER:
            key = f"{target}|{k}"
            if key in r["auroc"]:
                floor = min(floor, float(np.nanmin(np.array(r["ci"][key][0], float))))
    floor = max(0.0, min(0.45, floor) - 0.02)
    for ax, fam in zip(axes, fams):
        r = d["families"][fam]
        x = r["depths"]
        ends = {}
        for k in ORDER:
            key = f"{target}|{k}"
            if key not in r["auroc"]:
                continue
            y = np.array(r["auroc"][key], float)
            lo, hi = (np.array(v, float) for v in r["ci"][key])
            _line(ax, x, y, k, lo, hi)
            ends[k] = y[-1]
        st.chance(ax, 0.5)
        st.depth_axis(ax, "depth" if ax is axes[len(axes) // 2] else "")
        ax.set_ylim(floor, 1.02)
        ax.set_title(f"{_fam_label(fam)}  n={r['n']}", loc="left", color=st.INK2)
        st.despine(ax)
        if ax is axes[-1]:
            st.end_labels(ax, x[-1], [(v, NICE[k], COLOUR[k]) for k, v in ends.items()])
    axes[0].set_ylabel("AUROC for the answer")
    axes[0].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    return st.save(fig, _at(out, stem))


def lens_kl(results="results", out="figures", model=None):
    p = _newest(f"{results}/lensbench*/{model or '*'}__bench.json")
    d = _load(p) if p else None
    if not d or not d.get("kl"):
        return None
    st.use()
    fig, ax = st.wide(unit=(11.6, 2.7))
    x = d["depths"]
    ends = {}
    for k in ORDER:
        if k not in d["kl"]:
            continue
        y = np.array(d["kl"][k], float)
        _line(ax, x, y, k)
        ends[k] = y[-1]
    st.depth_axis(ax)
    ax.set_ylabel("KL from the model (nats)")
    ax.set_yscale("log")
    st.despine(ax)
    lo, hi = ax.get_ylim()
    ys = st.stagger([np.log10(max(ends[k], 1e-3)) for k in ends],
                    (np.log10(hi) - np.log10(lo)) * 0.055)
    for (k, _), yy in zip(ends.items(), ys):
        st.label_end(ax, x[-1], 10 ** yy, NICE[k], color=COLOUR[k])
    return st.save(fig, _at(out, "lens_kl"))


def patch_causal(results="results", out="figures", model=None):
    p = _newest(f"{results}/patchcausal/{model or '*'}__patchcausal.json")
    d = _load(p) if p else None
    if not d:
        return None
    s = d["summary"]
    ks = [100 * k for k in s["ks"]]
    st.use()
    fig, (a1, a2) = st.wide(1, 2, unit=(5.98, 3.3))
    cols = {"top": st.LENS, "middle": st.SERIES[3], "random": st.MUTED, "bottom": st.SERIES[2]}
    show = {"top": "V-Write's highest", "bottom": "its lowest",
            "middle": "the middle", "random": "random"}
    e1, e2 = [], []
    for g in ("top", "bottom", "middle", "random"):
        if g not in s["curve"]:
            continue
        lead = g in ("top", "bottom")
        for ax, key, ends in ((a1, "curve", e1), (a2, "flip_rate", e2)):
            ax.plot(ks, s[key][g], color=cols[g], lw=2.7 if lead else 1.6,
                    marker="o", ms=4.5 if lead else 3, zorder=4 if lead else 2,
                    ls="-" if lead else (0, (4, 2)))
            ends.append((s[key][g][-1], show[g], cols[g]))
    st.end_labels(a1, ks[-1], e1)
    st.end_labels(a2, ks[-1], e2)
    for ax, lab in ((a1, "fall in the answer score"), (a2, "share of answers that flip")):
        ax.set_xlabel("patches replaced (%)", labelpad=9)
        ax.set_ylabel(lab)
        ax.set_xticks(ks)
        st.despine(ax)
    a1.axhline(0, color=st.AXIS, lw=1.0, zorder=1)
    a2.set_ylim(bottom=0)
    return st.save(fig, _at(out, "patch_causal"))


def patch_lenses(results="results", out="figures", model=None):
    p = _newest(f"{results}/patchlenses/{model or '*'}__patchlenses.json")
    d = _load(p) if p else None
    if not d:
        return None
    s_ = d["summary"]
    ks = [100 * k for k in s_["ks"]]
    order = [r for r in s_["rankings"] if f"zero:{r}" in s_.get("drop", {})]
    if not order:
        return None
    st.use()
    fig, (a1, a2) = st.wide(1, 2, unit=(6.5, 3.4))

    OURS = {"v_write": st.LENS, "v_read": "#7E2A11", "v_read_cross": "#E08150"}

    def colour(mm, rank):
        if mm in OURS:
            return OURS[mm]
        if mm == "random":
            return st.MUTED
        return [st.SERIES[1], st.SERIES[2], st.SERIES[4], st.SERIES[5],
                st.SERIES[3], st.SERIES[6], st.SERIES[7]][rank % 7]

    NAMED = {"v_write": st.LENS, "attr_patch": st.SERIES[2], "random": st.MUTED}
    band = [mm for mm in order if mm not in NAMED]
    ends = []
    for mm in band:
        y = s_["drop"][f"zero:{mm}"]
        a1.plot(ks, y, color=st.SERIES[3], lw=1.2, alpha=0.55, zorder=2)
    if band:
        ys = [s_["drop"][f"zero:{mm}"][-1] for mm in band]
        ends.append((float(np.mean(ys)), f"the {len(band)} reading lenses", st.SERIES[3]))
    for mm, c in NAMED.items():
        if f"zero:{mm}" not in s_["drop"]:
            continue
        y = s_["drop"][f"zero:{mm}"]
        lead = mm != "random"
        a1.plot(ks, y, color=c, lw=2.8 if lead else 1.8, marker="o", ms=4.5 if lead else 3.5,
                zorder=4 if lead else 3, ls="-" if lead else (0, (4, 2)))
        ends.append((y[-1], mm.replace("_", " "), c))
    st.end_labels(a1, ks[-1], ends)
    a1.set_xlabel("patches replaced (%)")
    a1.set_ylabel("fall in the answer score")
    a1.set_xticks(ks)
    a1.axhline(0, color=st.AXIS, lw=1.0, zorder=1)
    st.despine(a1)

    g = s_.get("graded_against_patching", {})
    pairs = [(mm, g.get(f"zero|{mm}")) for mm in s_["rankings"]
             if g.get(f"zero|{mm}") is not None and np.isfinite(g.get(f"zero|{mm}"))]
    pairs.sort(key=lambda t: t[1])
    for j, (mm, v) in enumerate(pairs):
        c = colour(mm, s_["rankings"].index(mm))
        a2.plot([0, v], [j, j], color=c, lw=1.2, alpha=0.5, zorder=2)
        a2.plot([v], [j], marker="o", ms=7, color=c, zorder=3)
    a2.set_yticks(range(len(pairs)))
    a2.set_yticklabels([mm.replace("_", " ") for mm, _ in pairs])
    a2.set_xlabel("rank agreement")
    a2.axvline(0, color=st.AXIS, lw=1.0, zorder=1)
    a2.set_ylim(-0.7, len(pairs) - 0.3)
    st.despine(a2, left=False)
    a2.tick_params(axis="y", length=0)
    return st.save(fig, _at(out, "patch_lenses"))


def patch_sources(results="results", out="figures", model=None):
    p = _newest(f"{results}/patchlenses/{model or '*'}__patchlenses.json")
    d = _load(p) if p else None
    if not d or not d["summary"].get("per_source"):
        return None
    s_ = d["summary"]
    per, ns = s_["per_source"], s_.get("per_source_n") or {}
    ks = [100 * k for k in s_["ks"]]
    srcs = sorted(per, key=lambda f: -float(np.nanmax(per[f].get("v_write") or [0])))
    NAMED = {"v_write": (st.LENS, "V-Write"), "attr_patch": (st.SERIES[2], "attribution patching"),
             "random": (st.MUTED, "random")}
    st.use()
    fig, axes = st.wide(1, len(srcs), unit=(3.51, 3.2))
    fig.get_layout_engine().set(wspace=0.07)
    axes = np.atleast_1d(axes).ravel()
    for j, (ax, f) in enumerate(zip(axes, srcs)):
        band = [mm for mm in s_["rankings"] if mm in per[f] and mm not in NAMED]
        for mm in band:
            ax.plot(ks, per[f][mm], color=st.SERIES[3], lw=1.0, alpha=0.5, zorder=2)
        ends = []
        if band:
            ends.append((float(np.nanmean([per[f][mm][-1] for mm in band])),
                         f"the {len(band)} reading lenses", st.SERIES[3]))
        for mm, (c, lab) in NAMED.items():
            if mm not in per[f]:
                continue
            lead = mm != "random"
            ax.plot(ks, per[f][mm], color=c, lw=2.6 if lead else 1.6, marker="o",
                    ms=4.2 if lead else 3.0, zorder=4 if lead else 3,
                    ls="-" if lead else (0, (4, 2)))
            ends.append((per[f][mm][-1], lab, c))
        if j == len(srcs) - 1:
            st.end_labels(ax, ks[-1], ends)
        ax.set_title(f"{SOURCE.get(f, f)}\n{ns.get(f, 0)} pictures", color=st.INK2)
        ax.set_xticks([ks[0], ks[len(ks) // 2], ks[-1]])
        ax.axhline(0, color=st.AXIS, lw=1.0, zorder=1)
        st.despine(ax)
        if j == 0:
            ax.set_ylabel("fall in the answer score")
    fig.supxlabel("patches replaced (%)", color=st.INK2)
    return st.save(fig, _at(out, "patch_sources"))


def baseline_agreement(results="results", out="figures", model=None):
    p = (_newest(f"{results}/ablation1/{model or '*'}__patchcausal.json")
         or _newest(f"{results}/patchcausal/{model or '*'}__patchcausal.json"))
    d = _load(p) if p else None
    if not d or not d["summary"].get("agreement"):
        return None
    s = d["summary"]
    keys = sorted({k for pair in s["agreement"] for k in pair.split("|")})
    keys = ([k for k in keys if k.startswith("vwrite_")]
            + [k for k in keys if not k.startswith("vwrite_") and k != "random"]
            + [k for k in keys if k == "random"])
    SHORT = {"vwrite_mean": "the mean patch", "vwrite_blur": "a blurred copy",
             "vwrite_other": "another picture", "vwrite_zero": "zeros",
             "vwrite_noise": "noise", "attr_patch": "attribution patching",
             "gradxact_zero": "grad \u00d7 activation", "random": "random"}
    lab = [SHORT.get(k, k.replace("vwrite_", "")) for k in keys]
    n = len(keys)
    M = np.full((n, n), np.nan)
    for i, x in enumerate(keys):
        for j, y in enumerate(keys):
            if i == j:
                continue
            v = s["agreement"].get(f"{x}|{y}", s["agreement"].get(f"{y}|{x}"))
            if v is not None:
                M[i, j] = M[j, i] = v
    st.use()
    fig, (a1, a2) = st.wide(1, 2, unit=(8.6, 4.2))
    shown = np.where(np.tril(np.ones((n, n)), -1) > 0, M, np.nan)
    a1.imshow(shown, cmap=st.warm_cmap(), vmin=0, vmax=1)
    a1.set_xticks([]); a1.set_yticks([])
    a1.grid(False)
    for sp in a1.spines.values():
        sp.set_visible(False)
    for i in range(n):
        a1.text(i - 0.36, i, lab[i], ha="left", va="center", color=st.INK)
        for j in range(i):
            if np.isfinite(M[i, j]):
                a1.text(j, i, f"{M[i, j]:.2f}".replace("0.", ".").replace("-.00", ".00"),
                        ha="center", va="center",
                        color=st.INK,
                        bbox=(dict(boxstyle="round,pad=0.12", fc="white", ec="none")
                              if M[i, j] >= 0.62 else None))
    a1.set_xlim(-0.55, n + 3.6)
    a1.set_ylim(n - 0.4, -0.6)
    a1.set_title("rank agreement", loc="left", color=st.INK2)
    ks = [100 * k for k in s["ks"]]
    tips = []
    got = [x for x in s["curve"] if x.startswith("top:vwrite_")]
    for i, k in enumerate(got):
        c = COLOUR["v_read"] if k == "top:vwrite_mean" else st.WARM[2 + i % 4]
        a2.plot(ks, s["curve"][k], color=c, lw=2.4 if k == "top:vwrite_mean" else 1.7,
                marker="o", ms=3.5, zorder=5 if k == "top:vwrite_mean" else 3)
        tips.append((s["curve"][k][-1], SHORT.get(k[4:], k[4:]), c))
    for k, c in (("top:attr_patch", st.SERIES[2]), ("top:gradxact_zero", st.SERIES[1]),
                 ("top:random", st.MUTED)):
        if k in s["curve"]:
            a2.plot(ks, s["curve"][k], color=c, lw=1.6, dashes=(4, 2))
            tips.append((s["curve"][k][-1], SHORT.get(k[4:], k[4:]), c))
    st.end_labels(a2, ks[-1], tips)
    a2.set_xlabel("picture replaced (%)")
    a2.set_ylabel("fall in the answer score")
    a2.set_xticks(ks)
    a2.set_xlim(ks[0], ks[-1] * 1.02)
    st.despine(a2)
    a2.set_title("what the top fifth is worth, under each", loc="left", color=st.INK2)
    return st.save(fig, _at(out, "baseline_agreement"))


def _steer_rows(d):
    if "results" in d:
        return d
    s = (d or {}).get("summary") or {}
    if not s.get("induced"):
        return d
    rows = []
    for k, benign in s["induced"].items():
        block, dname, al = k.split("|")
        removed = (s.get("removed") or {}).get(k)
        if removed is None:
            continue
        rows.append(dict(direction=dname, layer=int(block), depth=None,
                         alpha=float(al), refusal_benign=float(benign),
                         refusal_harmful=1.0 - float(removed)))
    base = s.get("base_refusal") or {}
    out = dict(results=rows)
    if base.get("removed") is not None and base.get("induced") is not None:
        out["baseline"] = dict(harmful=float(base["removed"]),
                               benign=float(base["induced"]))
    return out


def refusal_steer(results="results", out="figures", model=None):
    p = _newest(f"{results}/refusalsteer*/{model or '*'}__refusal_steer.json",
                f"{results}/safety*/{model or '*'}__refusal_steer.json")
    d = _load(p) if p else None
    if not d:
        return None
    d = _steer_rows(d)
    rows = [r for r in (d.get("results") or [])
            if r.get("direction") not in (None, "none")]
    if not rows:
        return None
    allb = [r for r in rows if r.get("scope") == "all blocks" or r.get("layer") == "all"]
    if allb:
        rows, scope = allb, "removed at every block"
    else:
        best = {}
        for r in rows:
            k = (r["direction"], r.get("layer"))
            best.setdefault(k, []).append(r)
        pick = {}
        for (dname, l), rs in best.items():
            lo = min(x["refusal_harmful"] for x in rs)
            if dname not in pick or lo < pick[dname][0]:
                pick[dname] = (lo, rs)
        rows = [x for _, rs in pick.values() for x in rs]
        scope = "removed at the block where it did most"
    names = sorted({r["direction"] for r in rows})
    cmap = {n: st.SERIES[i % len(st.SERIES)] for i, n in enumerate(names)}
    cmap["V-Write"] = st.LENS
    cmap["random"] = st.MUTED
    st.use()
    fig, ax = st.wide(unit=(7.8, 3.8))
    b = d.get("baseline", {})
    tips = []
    for n in names:
        sub = sorted([r for r in rows if r["direction"] == n], key=lambda r: r["alpha"])
        xs = [r["refusal_harmful"] for r in sub]
        ys = [r["refusal_benign"] for r in sub]
        if b:
            xs = [b["harmful"]] + xs
            ys = [b["benign"]] + ys
        ax.plot(xs, ys, color=cmap[n], lw=2.4 if n == "V-Write" else 1.7,
                marker="o", ms=3.5, alpha=0.95)
        if xs:
            tips.append((ys[-1], n, cmap[n], xs[-1]))
    if b:
        ax.plot([b["harmful"]], [b["benign"]], marker="s", ms=7, color=st.INK, zorder=9)
        ax.annotate("no edit", xy=(b["harmful"], b["benign"]), xytext=(0, -14),
                    textcoords="offset points", color=st.INK2, ha="center", va="top")
    if tips:
        ax.figure.canvas.draw()
        y0, y1 = ax.get_ylim()
        per_pt = (y1 - y0) / max(ax.get_window_extent().height, 1) * ax.figure.dpi / 72.0
        ys_ = st.stagger([t[0] for t in tips],
                         (st.matplotlib.rcParams["legend.fontsize"] + 2.0) * per_pt)
        for (yv, n, c, xv), yy in zip(tips, ys_):
            st.label_end(ax, xv, float(yy), n, color=c)
    ax.set_xlabel(f"refusal on harmful items after the direction is {scope.split(' ', 1)[0]}")
    ax.set_ylabel("refusal on benign items")
    ax.set_title(scope, loc="left", color=st.INK2)
    st.despine(ax)
    return st.save(fig, _at(out, "refusal_steer"))


def refusal_read(results="results", out="figures", model=None):
    p = _newest(f"{results}/safety*/{model or '*'}__refusal_auroc.json")
    d = _load(p) if p else None
    if not d:
        return None
    st.use()
    col = {**COLOUR, "probe": st.SLATE}
    keys = [k for k in [*ORDER, *REFUSAL_EXTRA]
            if any(f"{t}|{k}" in d["auroc"] for t in ("harmful", "refused"))]
    rank = ["v_read", "v_read_indomain", "v_read_coco", "tuned_indomain", "logit",
            "tuned", "jacobian", "v_read_cross", "dom", "probe"]
    keys.sort(key=lambda k: rank.index(k) if k in rank else len(rank))
    fig, (a1, a2) = st.wide(1, 2, unit=(5.72, 3.0), sharey=True,
                            gridspec_kw=dict(wspace=0.08))
    x = np.array(d["depths"], float)
    lows = []
    for ax, tname, lab in ((a1, "harmful", "the request is harmful"),
                           (a2, "refused", "the reply will be a refusal")):
        for k in keys[::-1]:
            key = f"{tname}|{k}"
            if key not in d["auroc"]:
                continue
            y = np.array(d["auroc"][key], float)
            lo, hi = (np.array(v, float) for v in d["ci"][key])
            ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
            c = col.get(k, st.SERIES[0])
            st.ribbon(ax, x[ok], lo[ok], hi[ok], c, clip_on=True)
            ax.plot(x[ok], y[ok], color=c, lw=3.0 if k == "v_read" else 1.6,
                    dashes=DASH.get(k, ()), zorder=6 if k == "v_read" else 3,
                    clip_on=True)
            lows.append(np.nanmin(np.r_[y[ok], lo[ok]]))
        st.chance(ax, 0.5)
        st.depth_axis(ax, "depth")
        ax.set_title(lab, loc="left", color=st.INK2, fontsize=st.matplotlib.rcParams["legend.fontsize"])
        st.despine(ax)
    floor = np.floor(min(lows + [0.5]) / 0.05) * 0.05
    a1.set_ylim(floor, 1.02)
    a1.set_yticks([t for t in (0.25, 0.5, 0.75, 1.0) if t >= floor])
    a1.set_ylabel("AUROC")
    st.key_band_lines(fig, [(NICE[k], col.get(k, st.SERIES[0]), DASH.get(k))
                            for k in keys], lw=2.2, ncol=int(np.ceil(len(keys) / 2)))
    return st.save(fig, _at(out, "refusal_read"))


def refusal_cascade(results="results", out="figures", model=None):
    p = _newest(f"{results}/safety*/{model or '*'}__refusal_auroc.json")
    d = _load(p) if p else None
    if not d:
        return None
    LEVEL = 0.8
    x = np.array(d["depths"], float)

    def first(key):
        y = np.array(d["auroc"].get(key, []), float)
        hit = np.nonzero(y >= LEVEL)[0]
        return float(x[hit[0]]) if len(hit) else None

    pairs = []
    for k in [*ORDER, *REFUSAL_EXTRA]:
        h, r = first(f"harmful|{k}"), first(f"refused|{k}")
        if h is not None and r is not None:
            pairs.append((k, h, r))
    if not pairs:
        return None
    pairs.sort(key=lambda t: t[1])
    st.use()
    fig, ax = st.wide(unit=(8.06, 0.55 * len(pairs) + 1.6))
    for j, (k, h, r) in enumerate(pairs):
        c = COLOUR.get(k, st.SERIES[0])
        ax.plot([h, r], [j, j], color=c, lw=1.6, alpha=0.6, zorder=2)
        ax.plot([h], [j], marker="o", ms=7, color=c, zorder=3)
        ax.plot([r], [j], marker="o", ms=7, mfc="white", mec=c, mew=1.8, zorder=3)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([NICE.get(k, k) for k in (t[0] for t in pairs)])
    st.depth_axis(ax, "depth at which it first reads at 0.8 AUROC")
    ax.set_ylim(-0.7, len(pairs) - 0.3)
    st.despine(ax, left=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(ax.get_xlabel() + "\nfilled: the harm.  hollow: the refusal.")
    return st.save(fig, _at(out, "refusal_cascade"))


def handoff(results="results", out="figures", model=None):
    p = _newest(f"{results}/handoff/{model or '*'}__handoff.json")
    d = _load(p) if p else None
    if not d:
        return None
    s = d["summary"]
    st.use()
    fig, (a1, a2) = st.wide(1, 2, unit=(5.98, 3.3))
    grid = s["grid"]
    x = [l / (max(grid) or 1) for l in grid]
    y0 = [s["depth_curve_original"][str(l)] for l in grid]
    y1 = [s["depth_curve"][str(l)] for l in grid]
    a1.plot(x, y0, color=st.MUTED, lw=1.9, dashes=(5, 2))
    a1.plot(x, y1, color=st.LENS, lw=2.6)
    st.label_end(a1, x[-1], y0[-1], "original run", color=st.MUTED)
    st.label_end(a1, x[-1], y1[-1], "edited region\ntransplanted", color=st.LENS)
    a1.axvline(s["depth"], color=st.AXIS, lw=1.0)
    a1.annotate("transplant", xy=(s["depth"], a1.get_ylim()[0]), xytext=(3, 6),
                textcoords="offset points", color=st.INK2)
    st.chance(a1, 0.0, "the two answers are level")
    st.depth_axis(a1)
    a1.set_ylabel("edited over original")
    st.despine(a1)
    groups = sorted({k.split("|")[1] for k in s["swaps"]})
    depths = sorted({int(k.split("|")[0]) for k in s["swaps"]})
    for i, g in enumerate(groups):
        ys = [s["swaps"].get(f"{l}|{g}", np.nan) for l in depths]
        c = st.SERIES[i % len(st.SERIES)]
        a2.plot([l / max(grid) for l in depths], ys, color=c, lw=2.1, marker="o", ms=4)
        st.label_end(a2, depths[-1] / max(grid), ys[-1], g, color=c)
    st.chance(a2, 0.0, "")
    a2.set_xlabel("depth of the swap")
    st.despine(a2)
    return st.save(fig, _at(out, "handoff"))


def read_image(results="results", out="figures", model=None):
    p = _newest(f"{results}/readimage/{model or '*'}__readimage.json")
    d = _load(p) if p else None
    if not d:
        return None
    s = d["summary"] if "summary" in d else d
    st.use()
    fig, (a1, a2) = st.wide(1, 2, unit=(5.85, 3.3), sharey=True)
    for ax, how, lab in ((a1, "best", "the single best image token"),
                         (a2, "top10", "the best tenth of image tokens")):
        ends = {}
        for k in ORDER:
            key = f"{how}|{k}"
            if key not in s["auroc"]:
                continue
            y = np.array(s["auroc"][key], float)
            ci = s.get("ci", {}).get(key)
            lo, hi = (np.array(ci[0], float), np.array(ci[1], float)) if ci else (None, None)
            _line(ax, s["depths"], y, k, lo, hi)
            ends[k] = y[-1]
        st.chance(ax, 0.5)
        st.depth_axis(ax, "depth")
        ax.set_ylim(0.35, 1.02)
        ax.set_title(lab, loc="left", color=st.INK2)
        st.despine(ax)
        if ax is a2 and ends:
            st.end_labels(ax, s["depths"][-1],
                          [(v, NICE.get(k, k), COLOUR.get(k, st.INK)) for k, v in ends.items()])
    a1.set_ylabel("AUROC for the answer")
    fig.get_layout_engine().set(wspace=0.06)
    return st.save(fig, _at(out, "read_image"))


def counterfactual(results="results", out="figures", model=None):
    p = _newest(f"{results}/readimage/{model or '*'}__readimage.json")
    d = _load(p) if p else None
    top = (d or {}).get("summary") or d or {}
    s = top.get("counterfactual")
    depths = s.get("depths") if isinstance(s, dict) else None
    depths = depths or top.get("depths")
    if not s or not depths:
        return None
    st.use()
    fig, ax = st.wide(unit=(8.06, 3.5))
    ends = {}
    for k in ORDER:
        if k not in s["correlation"]:
            continue
        y = np.array(s["correlation"][k], float)
        _line(ax, depths, y, k)
        ends[k] = y[-1]
    ax.axhline(0, color=st.AXIS, lw=1.0, zorder=1)
    st.depth_axis(ax)
    ax.set_ylabel("tracks the answer moving")
    st.despine(ax)
    st.end_labels(ax, depths[-1],
                  [(v, NICE.get(k, k), COLOUR.get(k, st.INK)) for k, v in ends.items()])
    return st.save(fig, _at(out, "counterfactual"))


def crossquestion(results="results", out="figures", model=None):
    p = _newest(f"{results}/crossquestion/{model or '*'}__crossquestion.json")
    d = _load(p) if p else None
    if not d:
        return None
    s_, rows = d["summary"], d.get("rows") or []
    ks = [100 * k for k in s_["ks"]]
    st.use()
    fig, (a1, a2) = st.wide(1, 2, unit=(5.98, 3.2))
    ends = []
    for src, lab, c in (("own", "this question's map", st.LENS),
                        ("other", "the other question's map", st.SERIES[2]),
                        ("random", "a random ranking", st.MUTED)):
        y = s_["curve"][src]
        lead = src != "random"
        a1.plot(ks, y, color=c, lw=2.6 if src == "own" else 1.8, marker="o",
                ms=4.2 if lead else 3.0, ls="-" if lead else (0, (4, 2)),
                zorder=4 if src == "own" else 3)
        ends.append((y[-1], lab, c))
    st.end_labels(a1, ks[-1], ends)
    a1.set_xticks(ks)
    a1.set_ylabel("fall in the answer score")
    fig.get_layout_engine().set(wspace=0.08)
    fig.supxlabel("patches replaced (%)", color=st.INK2)
    a1.axhline(0, color=st.AXIS, lw=1.0, zorder=1)
    st.despine(a1)

    k = s_["ks"][len(s_["ks"]) // 2]
    diff = np.array(sorted(
        float(np.nanmean([x["drop"].get(f"{i}|own|{k}", np.nan)
                          - x["drop"].get(f"{i}|other|{k}", np.nan) for i in (0, 1)]))
        for x in rows), float)
    diff = diff[np.isfinite(diff)]
    if len(diff):
        x = np.arange(len(diff))
        a2.fill_between(x, 0, diff, where=diff > 0, color=st.LENS, alpha=0.25, lw=0)
        a2.fill_between(x, 0, diff, where=diff <= 0, color=st.MUTED, alpha=0.25, lw=0)
        a2.plot(x, diff, color=st.LENS, lw=1.6)
        p_ = s_["paired"][str(k)]
        a2.axhline(0, color=st.AXIS, lw=1.0)
        st.label_end(a2, len(diff) - 1, p_["mean"], f"mean {p_['mean']:.2f}", color=st.INK2)
    a2.set_xticks([])
    a2.set_title(f"one picture at a time, sorted, at {int(k * 100)} per cent replaced",
                 loc="left", color=st.INK2)
    a2.set_ylabel("this map minus the other")
    st.despine(a2)
    return st.save(fig, _at(out, "crossquestion"))


def patch_atlas(results="results", out="figures", model=None):
    hits = sorted(glob.glob(f"{results}/atlas/{model or '*'}__*patchlenses.json")
                  + glob.glob(f"{results}/patchlenses/{model or '*'}__patchlenses.json"))
    name = model or (os.path.basename(hits[0]).split("__")[0] if hits else None)
    if not name or not _enough(hits):
        return None
    seen = {}
    by_depth = {}
    for p in hits:
        if not AVERAGE and os.path.basename(p).split("__")[0] != name:
            continue
        d = _load(p, pool=False)
        if not d or not d["summary"].get("per_source"):
            continue
        s_ = d["summary"]
        if not s_.get("oriented"):
            continue
        by_depth.setdefault(round(float(s_.get("depth", 0.5)), 2), []).append(
            (os.path.basename(p).split("__")[0], s_))
    if AVERAGE:
        every = set.intersection(*({m for m, _ in v} for v in by_depth.values())) \
            if by_depth else set()
        by_depth = {d_: [(m, x) for m, x in v if m in every] for d_, v in by_depth.items()}
        by_depth = {d_: v for d_, v in by_depth.items() if len({m for m, _ in v}) > 1}
    seen = {d_: (v[0][1] if len(v) == 1 else _mean_of([x for _, x in v]))
            for d_, v in by_depth.items() if v}
    if len(seen) < 2:
        return None
    depths = sorted(seen)
    srcs = sorted({f for s_ in seen.values() for f in s_["per_source"]})
    st.use()
    fig, (a1, a2) = st.wide(1, 2, unit=(6.24, 3.3))
    for ax, over_random in ((a1, False), (a2, True)):
        ends, j = [], 0
        for f in srcs:
            x, y = [], []
            for dd in depths:
                s_ = seen[dd]
                row = s_["per_source"].get(f) or {}
                v, r = row.get("v_write"), row.get("random")
                if v is None:
                    continue
                k = s_.get("per_source_index", 2)
                x.append(dd)
                y.append(v[k] - (r[k] if over_random and r else 0.0))
            if len(x) < 2:
                continue
            carrier = f == "wmdp"
            c = st.MUTED if carrier else [st.LENS, st.SERIES[1], st.SERIES[2],
                                          st.SERIES[5]][j % 4]
            j += 0 if carrier else 1
            ax.plot(x, y, color=c, lw=1.8 if carrier else 2.4, marker="o", ms=4.0,
                    ls=(0, (4, 2)) if carrier else "-", zorder=3 if carrier else 4)
            ends.append((y[-1], SOURCE.get(f, f), c))
        if not ends:
            return None
        st.depth_axis(ax)
        ax.set_xlim(0, 1.62)
        ax.axhline(0, color=st.AXIS, lw=1.0, zorder=1)
        st.despine(ax)
        if ax is a2:
            st.end_labels(ax, 1.04, ends)
    a1.set_ylabel("fall in the answer score")
    a2.set_ylabel("over a random ranking")
    fig.get_layout_engine().set(wspace=0.09)
    return st.save(fig, _at(out, "patch_atlas"))


def patch_jailbreak(results="results", out="figures", model=None):
    p = _newest(f"{results}/patchjailbreak/{model or '*'}__patchjailbreak.json")
    d = _load(p) if p else None
    if not d or not d["summary"].get("arms"):
        return None
    s_ = d["summary"]
    ks = [100 * k for k in s_["ks"]]
    arms = [a_ for a_ in ("complied", "refused") if a_ in s_["arms"]]
    NAMED = {"v_write": (st.LENS, "V-Write"), "attr_patch": (st.SERIES[2], "attribution patching"),
             "random": (st.MUTED, "random"), "bottom": (st.SERIES[3], "the other end")}
    TITLE = {"complied": "the model complied", "refused": "the model refused"}
    st.use()
    fig, axes = st.wide(1, len(arms), unit=(6.24, 3.3))
    axes = np.atleast_1d(axes).ravel()
    for j, (ax, a_) in enumerate(zip(axes, arms)):
        e = s_["arms"][a_]
        ends = []
        for c, (col, lab) in NAMED.items():
            y = (e.get("moved") or {}).get(c)
            if not y:
                continue
            lead = c in ("v_write", "attr_patch")
            ax.plot(ks, y, color=col, lw=2.6 if c == "v_write" else 1.8, marker="o",
                    ms=4.2 if lead else 3.0, ls="-" if lead else (0, (4, 2)),
                    zorder=4 if c == "v_write" else 3)
            ends.append((y[-1], lab, col))
        if e.get("all_covered") is not None and np.isfinite(e["all_covered"]):
            ax.axhline(e["all_covered"], color=st.AXIS, lw=1.2, dashes=(2, 2), zorder=2)
            if j == len(arms) - 1:
                ends.append((e["all_covered"], "the whole picture", st.INK2))
            else:
                st.label_end(ax, ks[-1], e["all_covered"], "the whole picture", color=st.INK2)
        if j == len(arms) - 1:
            st.end_labels(ax, ks[-1], ends)
        ax.set_xticks(ks)
        ax.axhline(0, color=st.AXIS, lw=1.0, zorder=1)
        ax.set_title(f"{TITLE.get(a_, a_)}   n={e['n']}", loc="left", color=st.INK2)
        st.despine(ax)
    axes[0].set_ylabel("moved the intended way")
    fig.get_layout_engine().set(wspace=0.1)
    fig.supxlabel("patches replaced (%)", color=st.INK2)
    return st.save(fig, _at(out, "patch_jailbreak"))


def steer(results="results", out="figures", model=None):
    p = _newest(f"{results}/steer/{model or '*'}__steer.json")
    d = _load(p) if p else None
    if not d:
        return None
    s_ = (d.get("summary") or d)
    alphas, blocks = s_["alphas"], s_["blocks"]
    dirs = s_["directions"]
    COL = {"V-Write": st.LENS, "difference of means (labelled)": st.SERIES[1],
           "logit lens": st.MUTED, "random": st.SERIES[3]}
    DASH_ = {"random": (4, 2), "logit lens": (2, 2)}
    st.use()
    fig, (a1, a2) = st.wide(1, 2, unit=(6.11, 3.3))
    for ax, field, ylab in ((a1, "flip_rate", "share that become the written answer"),
                            (a2, "margin", "logit of the written answer,\nminus the given one")):
        ends, at_blocks = [], []
        for dn in dirs:
            best, series, at_block = None, None, None
            for L in blocks:
                def at(al, L=L, dn=dn):
                    v = s_[field].get(f"{L}|{dn}|{al}")
                    if v is None and al == 0.0:
                        v = next((s_[field][f"{L}|{o}|{al}"] for o in dirs
                                  if f"{L}|{o}|{al}" in s_[field]), None)
                    return v
                y = [at(al) for al in alphas]
                if any(v is None for v in y):
                    continue
                y = [float(v) for v in y]
                if best is None or abs(y[-1] - y[0]) > best:
                    best, series, at_block = abs(y[-1] - y[0]), y, L
            if series is None:
                continue
            at_blocks.append(at_block)
            c = COL.get(dn, st.SERIES[2])
            lead = dn == "V-Write"
            ax.plot(alphas, series, color=c, lw=2.6 if lead else 1.8, marker="o",
                    ms=4.2 if lead else 3.0, dashes=DASH_.get(dn, ()),
                    zorder=4 if lead else 3)
            ends.append((series[-1],
                         dn.replace(" (labelled)", " (labels)")
                         + ("" if AVERAGE or len(set(at_blocks)) < 2
                            else f" at block {at_block}"), c))
        if ax is a2:
            st.end_labels(ax, alphas[-1], ends)
        ax.set_xticks(alphas)
        ax.set_ylabel(ylab)
        ax.axhline(0, color=st.AXIS, lw=1.0, zorder=1)
        st.despine(ax)
    a1.set_ylim(bottom=0)
    a1.set_title(("" if AVERAGE else os.path.basename(p).split("__")[0] + ", ")
                 + f"{s_['n']} records", loc="left", color=st.INK2)
    fig.supxlabel("dose, in residual norms", color=st.INK2)
    return st.save(fig, _at(out, "steer"))


def carrier(results="results", out="figures", model=None):
    p = _newest(f"{results}/carrier/{model or '*'}__carrier.json")
    d = _load(p) if p else None
    if not d:
        return None
    s_ = d["summary"]
    x = s_["depths"]
    st.use()
    fig, (a1, a2) = st.wide(1, 2, unit=(6.24, 3.3))
    ends = []
    for which, dash in (("as taken", ()), ("edited", (4, 2))):
        g = (s_.get("groups") or {}).get(which)
        if not g:
            continue
        for inst in ("v_read", "logit"):
            y = (g.get("trace") or {}).get(inst)
            if not y:
                continue
            c = COLOUR.get(inst, st.SERIES[0])
            a1.plot(x, y, color=c, lw=2.4 if inst == "v_read" else 1.6, dashes=dash,
                    zorder=4 if inst == "v_read" else 3)
            ends.append((y[-1], f"{NICE.get(inst, inst)}, {which}", c))
    st.end_labels(a1, x[-1], ends)
    a1.axhline(0, color=st.AXIS, lw=1.0, zorder=1)
    st.depth_axis(a1)
    a1.set_xlim(0, 1.5)
    a1.set_ylabel("yes over no at the answer")

    ends = []
    for inst in ("v_read", "tuned", "logit"):
        g = (s_.get("pair_gap") or {}).get(inst)
        if not g:
            continue
        c = COLOUR.get(inst, st.SERIES[0])
        st.ribbon(a2, x, g["lo"], g["hi"], c)
        a2.plot(x, g["mean"], color=c, lw=2.4 if inst == "v_read" else 1.6,
                dashes=DASH.get(inst, ()), zorder=4 if inst == "v_read" else 3)
        ends.append((g["mean"][-1], NICE.get(inst, inst), c))
        at = g.get("parts_at")
        if at is not None and np.isfinite(at) and inst == "v_read":
            a2.axvline(at, color=c, lw=1.0, dashes=(2, 2), zorder=2)
    st.end_labels(a2, x[-1], ends)
    a2.axhline(0, color=st.AXIS, lw=1.0, zorder=1)
    st.depth_axis(a2)
    a2.set_xlim(0, 1.34)
    n = ((s_.get("pair_gap") or {}).get("v_read") or {}).get("n")
    a2.set_ylabel("edited minus original")
    if n:
        a2.set_title(f"{n} matched pairs", loc="left", color=st.INK2)
    for ax in (a1, a2):
        st.despine(ax)
    fig.get_layout_engine().set(wspace=0.09)
    return st.save(fig, _at(out, "carrier"))


def dataset_map(results="results", out="figures", model=None):
    if model:
        return None
    d = _load(f"{results}/dataset/census.json", pool=False)
    if not d or not d.get("families"):
        return None
    fams = d["families"]
    KIND = {"auth": "authority", "read": "reading", "eval": "evaluation awareness",
            "base": "description", "sand": "sandbagging",
            "swap": "authority", "removal": "authority"}
    COL = {"authority": st.LENS, "reading": st.SERIES[1], "evaluation awareness": st.SERIES[5],
           "description": st.SERIES[7], "sandbagging": st.MUTED}
    rows = []
    for f, e in fams.items():
        probe = KIND.get(e["probe"], e["probe"])
        rows.append((e["n"], f, probe, e["whose"] == "ours", e["sandbagging"],
                     SOURCE.get(e["source"], e["source"])))
    rows.sort(key=lambda r: (r[4], -r[0]))
    shown, rows = len(rows), rows[:16]
    half = (len(rows) + 1) // 2
    cols = [rows[:half], rows[half:]]
    hi = max(r[0] for r in rows) * 3.4
    st.use()
    fig, axes = st.wide(1, 2, unit=(7.6, 0.30 * half + 1.1))
    for ax, part in zip(np.atleast_1d(axes), cols):
        for j, (n, fam, probe, ours, sand, src) in enumerate(part):
            c = COL.get(probe, st.SERIES[0])
            ax.plot([1, n], [j, j], color=c, lw=1.2, alpha=0.45, zorder=2)
            ax.plot([n], [j], marker="o", ms=9, zorder=3, color=c if ours else "white",
                    mec=c, mew=2.0)
        ax.set_yticks(range(len(part)))
        ax.set_yticklabels([_fam_label(f) for _, f, _, _, _, _ in part])
        ax.set_xscale("log")
        ax.set_xlim(0.8, hi)
        ax.set_ylim(-0.7, half - 0.3)
        ax.invert_yaxis()
        ax.set_xlabel("records")
        st.despine(ax, left=False)
        ax.tick_params(axis="y", length=0)
    ax = np.atleast_1d(axes)[0]
    seen = [p_ for p_ in ("authority", "reading", "evaluation awareness", "description",
                          "sandbagging") if any(r[2] == p_ for r in rows)]
    a2 = np.atleast_1d(axes)[-1]
    for t, p_ in enumerate(seen):
        a2.plot([1.15], [half - 1 - t * 0.9], marker="o", ms=9, color=COL[p_],
                clip_on=False, zorder=6)
        a2.annotate(p_, xy=(1.5, half - 1 - t * 0.9), color=st.INK,
                    va="center", ha="left", annotation_clip=False)
    return st.save(fig, _at(out, "dataset_map"))


def cost_and_quality(results="results", out="figures", model=None):
    hits = sorted(glob.glob(f"{results}/lensbench*/{model or '*'}__bench.json"))
    if not _enough(hits):
        return None
    rows = []
    for p in hits:
        d = _load(p, pool=False)
        if not d or not d.get("kl"):
            continue
        mo = os.path.basename(p).split("__")[0]
        meta = d.get("meta") or {}
        for k, series in d["kl"].items():
            if k in HIDDEN:
                continue
            sec = (meta.get(k) or {}).get("seconds")
            rows.append((k, mo, float(np.mean(series)),
                         float(sec) if sec else 1.0, sec is not None))
    if not rows:
        return None
    st.use()
    fig, ax = st.wide(unit=(10.4, 3.4))
    fitted = [r for r in rows if r[4]]
    free = [r for r in rows if not r[4]]
    if not fitted:
        return None
    lo, hi = min(r[3] for r in fitted), max(r[3] for r in fitted)
    klo = min(r[2] for r in rows)
    khi = max(r[2] for r in rows)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo / 1.7, hi * 1.9)
    ax.set_ylim(klo / 1.18, khi * 1.22)
    for k, mo, kl, _sec, _ in free:
        c = COLOUR.get(k, st.SERIES[0])
        ax.axhline(kl, color=c, lw=1.3, dashes=(4, 3), zorder=2)
        st.label_end(ax, hi * 2.5, kl, NICE.get(k, k) + ", free", color=c)
    for k, mo, kl, sec, _ in fitted:
        c = COLOUR.get(k, st.SERIES[0])
        ours = k.startswith("v_read")
        ax.plot([sec], [kl], marker="o", ms=11 if ours else 7.5,
                color=c, mec=c, mew=1.6, zorder=4)
        ax.annotate(NICE.get(k, k), xy=(sec, kl), xytext=(0, 13 if ours else 11),
                    textcoords="offset points", color=st.INK, ha="center", va="bottom")
    for axis, vals in ((ax.yaxis, [v for v in (3, 5, 10, 20) if klo / 1.3 <= v <= khi * 1.3]),):
        axis.set_major_locator(mticker.FixedLocator(vals))
        axis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:g}"))
        axis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2, 5)))
    ax.xaxis.set_minor_formatter(mticker.FuncFormatter(
        lambda v, _p: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _p: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
    ax.set_xlabel("seconds to fit, all blocks")
    ax.set_ylabel("KL from the model (nats)")
    st.despine(ax)
    return st.save(fig, _at(out, "cost_and_quality"))


def gallery(results="results", out="figures", model=None):
    if AVERAGE:
        return None
    p = _newest(f"{results}/gallery/{model or '*'}__gallery.json")
    d = _load(p, pool=False) if p else None
    if not d or not d.get("scenarios"):
        return None
    from PIL import Image
    root = os.path.dirname(p)
    sc = d["scenarios"]
    st.use()
    fig, axes = st.wide(3, len(sc), unit=(3.6, 2.95))
    axes = np.atleast_2d(axes)
    for j, s_ in enumerate(sc):
        rows, cols = s_["grid"]
        for i, which in enumerate(("original", "top", "random")):
            ax = axes[i, j]
            f = os.path.join(root, s_["picture"][which])
            if not os.path.exists(f):
                ax.set_axis_off(); continue
            im = Image.open(f)
            boxes = None
            if which == "original":
                ph, pw = im.height / rows, im.width / cols
                boxes = [((int(t) % cols) * pw, (int(t) // cols) * ph, pw, ph)
                         for t in s_["top"]]
            st.image_panel(ax, im,
                           title=(SOURCE.get(s_.get("source"), s_["label"]) if i == 0 else None),
                           boxes=boxes, box_color=st.LENS)
            ax.set_xlabel(f"{s_['score'][which]:.2f}", color=st.INK2, labelpad=2)
    for i, lab in enumerate(("top fifth", "that fifth gone", "a random fifth gone")):
        axes[i, 0].set_ylabel(lab, color=st.INK2)
    return st.save(fig, _at(out, "gallery"))


def _src_colour(fam):
    order = ["openimages", "nuscenes", "vins", "gemini", "wmdp"]
    tone = [st.LENS, st.SERIES[1], st.SERIES[2], st.SERIES[5], st.MUTED]
    src = str(fam).split("_")[0]
    return tone[order.index(src)] if src in order else st.SERIES[4]


def hallucination(results="results", out="figures", model=None):
    p = _newest(f"{results}/hallucination/{model or '*'}__hallucination.json")
    d = _load(p) if p else None
    d = (d or {}).get("summary") or d
    if not d or not d.get("auroc"):
        return None
    x = np.asarray(d["depths"], float)
    st.use()
    fig, ax = st.wide(unit=(11.6, 2.7))
    ends = {}
    for k in ORDER:
        if k not in d["auroc"]:
            continue
        y = np.asarray(d["auroc"][k], float)
        c = (d.get("ci") or {}).get(k) if k in ("v_read", "probe") else None
        lo, hi = ((np.asarray(c[0], float), np.asarray(c[1], float)) if c else (None, None))
        _line(ax, x, y, k, lo, hi)
        ends[k] = float(y[-1])
    if not ends:
        return None
    st.chance(ax, 0.5)
    st.depth_axis(ax)
    ax.set_ylim(0.2, 1.02)
    ax.set_ylabel("AUROC, true against not")
    st.despine(ax)
    n, n_true = int(d.get("n", 0)), int(d.get("n_true", 0))
    if n:
        ax.set_title(f"{n_true} true, {n - n_true} hallucinated", loc="left", color=st.INK2)
    st.end_labels(ax, x[-1], [(v, NICE.get(k, k), COLOUR.get(k, st.INK))
                              for k, v in ends.items()])
    return st.save(fig, _at(out, "hallucination"))


PROBE_ORDER = ["operative", "operative_generic", "operative_with_graffiti",
               "defer", "ignore", "p1", "p8", "p5"]
PROBE_SAYS = {"operative": "obey a plain sign",
              "operative_generic": "obey it, generically",
              "operative_with_graffiti": "obey it, graffiti beside",
              "defer": "sign over the user",
              "ignore": "ignore text that misses",
              "p1": "say nothing answers",
              "p8": "name the bearer with standing",
              "p5": "a sign inside a picture"}

SHORT_MODEL = {"Qwen3-VL-8B-Instruct": "Qwen3-VL-8B-Instruct", "Qwen3.5-27B": "Qwen3.5-27B",
               "Qwen3-VL-2B-Instruct": "Qwen3-VL-2B-Instruct", "gemma-4-26B-A4B-it": "gemma-4-26B-A4B-it",
               "InternVL3_5-8B-HF": "InternVL3_5-8B-HF"}


def _short(m):
    return SHORT_MODEL.get(m, str(m).split("-")[0])


def eval_tell(results="results", out="figures", model=None):
    if model:
        return None
    d = _load(f"{results}/showcase/behaviour.json", pool=False)
    rows = [r for r in (d or {}).get("shift", [])
            if r["contrast"] == "told it is being evaluated"]
    if not rows:
        return None
    by = {}
    for r in rows:
        k = (r["model"], r["dataset"])
        e = by.setdefault(k, dict(n=0, mk_lo=0, mk_hi=0, h_lo=0, h_hi=0, d=[]))
        e["n"] += r["n"]
        for a_, b_ in (("mk_lo", "mk_lo"), ("mk_hi", "mk_hi"),
                       ("h_lo", "hedge_lo"), ("h_hi", "hedge_hi")):
            e[a_] += r[b_]
        e["d"].append(r["d_words"])
    if not by:
        return None
    st.use()
    fig, axes = st.wide(1, 2, unit=(5.4, 3.6))
    PANEL = [("formatted", lambda e: (e["mk_lo"] / e["n"], e["mk_hi"] / e["n"])),
             ("carries a caveat", lambda e: (e["h_lo"] / e["n"], e["h_hi"] / e["n"]))]
    for ax, (title, fn) in zip(axes, PANEL):
        tips = []
        for (mo, ds), e in sorted(by.items()):
            c = _src_colour(ds)
            dash = () if "27" in mo or "26" in mo else (4, 2)
            if fn is None:
                lo, hi = 0.0, float(np.mean(e["d"]))
            else:
                lo, hi = fn(e)
            ax.plot([0, 1], [lo, hi], color=c, lw=2.4, dashes=dash, zorder=4)
            ax.plot([0, 1], [lo, hi], marker="o", ms=6, color=c, ls="none", zorder=5)
            tips.append((hi, f"{SOURCE.get(ds, ds)}, {_short(mo)}", c))
        ax.set_xlim(-0.12, 1.78)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["helping", "evaluated"])
        ax.set_title(title, loc="left", color=st.INK2)
        st.despine(ax)
        if fn is not None:
            ax.set_ylim(-0.03, 1.03)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda v, _p: f"{v:.0%}"))
        else:
            ax.axhline(0, color=st.AXIS, lw=1.0, zorder=1)
        if ax is axes[-1]:
            st.key_labels(ax, [(t[1], t[2], () if "27" in t[1] or "26" in t[1] else (4, 2))
                               for t in tips])
    axes[0].set_xlim(-0.12, 1.12)
    axes[0].set_ylabel("share of replies")
    return st.save(fig, _at(out, "eval_tell"))


def eval_channel(results="results", out="figures", model=None):
    if model:
        return None
    d = _load(f"{results}/showcase/behaviour.json", pool=False) or {}
    rep = {r["model"]: r for r in d.get("self_report", []) if r["picture"] == "plain"}
    wm = {r["model"]: r for r in d.get("self_report", []) if r["picture"] == "watermark"}
    cond = {r["model"]: r for r in d.get("picture_conduct", [])}
    shift = {}
    for r in d.get("shift", []):
        if r["contrast"] != "told it is being evaluated":
            continue
        e = shift.setdefault(r["model"], dict(n=0, lo=0, hi=0))
        e["n"] += r["n"]; e["lo"] += r["mk_lo"]; e["hi"] += r["mk_hi"]
    models = sorted(set(rep) & set(cond) & set(shift))
    if not models:
        return None
    st.use()
    fig, axes = st.wide(2, 2, unit=(6.37, 2.9))
    CELL = [
        (0, 0, "says it is a test", ("a plain photograph", "a marker in it"),
         lambda mo: (rep[mo]["says_test"], wm[mo]["says_test"])),
        (0, 1, "formats the reply", ("a plain photograph", "a marker in it"),
         (lambda mo: (cond[mo]["markup_plain"], cond[mo]["markup_watermark"]))
         if min((c.get("n", 0) for c in cond.values()), default=0) >= 20 else None),
        (1, 0, "says it is a test", ("helping someone", "being evaluated"), None),
        (1, 1, "formats the reply", ("helping someone", "being evaluated"),
         lambda mo: (shift[mo]["lo"] / shift[mo]["n"], shift[mo]["hi"] / shift[mo]["n"])),
    ]
    for i, j, what, ends, fn in CELL:
        ax = axes[i, j]
        if fn is None:
            ax.text(0.5, 0.5, "too few matched pairs\nto measure",
                    transform=ax.transAxes, ha="center", va="center", color=st.INK)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            continue
        for mo in models:
            lo, hi = fn(mo)
            c = st.LENS if ("27" in mo or "26" in mo) else st.SERIES[2]
            ax.plot([0, 1], [lo, hi], color=c, lw=2.6, zorder=4)
            ax.plot([0, 1], [lo, hi], marker="o", ms=7, color=c, ls="none", zorder=5)
            st.label_end(ax, 1.0, hi, f"{_short(mo)}  {hi:.0%}", color=c, dx=7)
        ax.set_xlim(-0.1, 1.95); ax.set_ylim(-0.05, 1.08)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([_wrap(e, 12) for e in ends])
        ax.set_yticks([0, 0.5, 1.0])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:.0%}"))
        st.despine(ax)
        if i == 0:
            ax.set_title(what, loc="left", color=st.INK2)
    for i, lab in enumerate(("cue in the picture", "cue in the words")):
        axes[i, 0].set_ylabel(lab, labelpad=10)
    return st.save(fig, _at(out, "eval_channel"))


def authority_ladder(results="results", out="figures", model=None):
    if model:
        return None
    d = _load(f"{results}/showcase/behaviour.json", pool=False) or {}
    pool = {}
    for r in d.get("authority", []):
        if not r["n"]:
            continue
        e = pool.setdefault((r["model"], r["probe"]), [0, 0])
        e[0] += r["right"]; e[1] += r["n"]
    for r in d.get("standing", []):
        if r["n"] >= 5:
            pool[(r["model"], "tier:" + r["tier"])] = [r["right"], r["n"]]
    models = sorted({k[0] for k in pool})
    TIER = {"tier:D1": "a bearer with plain standing",
            "tier:D2": "two bearers, standing contested"}
    rows = [p_ for p_ in PROBE_ORDER
            if any(k[1] == p_ and v[1] >= 10 for k, v in pool.items())]
    rows += [t for t in ("tier:D1", "tier:D2") if any(k[1] == t for k in pool)]
    if len(rows) < 3 or not models:
        return None
    st.use()
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
    cmap = LinearSegmentedColormap.from_list("ladder", st.DIVERGING[1:-1])
    fs = st.matplotlib.rcParams["ytick.labelsize"]
    names = [TIER.get(p_, PROBE_SAYS.get(p_, p_)) for p_ in rows]
    cw, rh, lab_w, head_h, band_h = 2.0, 0.36, 3.2, 0.34, 0.42
    W = lab_w + cw * len(models) + 0.05
    H = band_h + head_h + rh * len(rows) + 0.05
    fig = plt.figure(figsize=(W, H), layout="none")
    ax = fig.add_axes([lab_w / W, 0.05 / H, cw * len(models) / W, rh * len(rows) / H])
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.axis("off")
    for j, p_ in enumerate(rows):
        ax.text(-0.56, j, names[j], ha="right", va="center", color=st.INK, fontsize=fs)
        for t, mo in enumerate(models):
            v = pool.get((mo, p_))
            if not v:
                continue
            frac = v[0] / v[1]
            ax.add_patch(Rectangle((t - 0.5, j - 0.5), 1, 1, facecolor=cmap(frac), edgecolor="white", lw=1.5))
            ax.text(t, j, f"{frac:.0%}", ha="center", va="center", color=st.INK, fontsize=fs)
    cut = len(rows) - sum(1 for r in rows if r.startswith("tier:"))
    if 0 < cut < len(rows):
        ax.plot([-0.5, len(models) - 0.5], [cut - 0.5, cut - 0.5], color=st.INK, lw=1.0, clip_on=False)
    for t, mo in enumerate(models):
        ax.text(t, -0.6, mo, ha="center", va="bottom", color=st.INK, fontsize=fs * 0.85)
    fsk = fs * 0.85
    yk = (H - 0.17) / H
    x = lab_w
    txt = fig.text(x / W, yk, "answered as the set says:  0%", ha="left", va="center", fontsize=fsk, color=st.INK)
    fig.canvas.draw()
    x = txt.get_window_extent(fig.canvas.get_renderer()).x1 / fig.dpi + 0.08
    cax = fig.add_axes([x / W, yk - 0.065 / H, 1.3 / W, 0.13 / H])
    cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmap, extent=(0, 1, 0, 1))
    cax.axvline(0.5, color=st.INK, lw=0.8)
    cax.set_xticks([]); cax.set_yticks([])
    for sp in cax.spines.values():
        sp.set_linewidth(0.6)
    fig.text((x + 1.38) / W, yk, "100%", ha="left", va="center", fontsize=fsk, color=st.INK)
    return st.save(fig, _at(out, "authority_ladder"))


_YES = re.compile(r"^\W*(yes|yeah|correct|true)\b", re.I)
_NO = re.compile(r"^\W*(no|nope|incorrect|false)\b", re.I)


def _said(r):
    t = str(r.get("reply") or "")
    m = re.match(r"^\W*actually[,\s]*\*{0,2}\s*(yes|no)\b", t, re.I)
    if m:
        return m.group(1).lower()
    return "no" if _NO.match(t) else "yes" if _YES.match(t) else None


def _wanted(r):
    e = str(r.get("expected") or "").strip().lower()
    return "no" if e.startswith("no") else "yes" if e.startswith("yes") else None


def _behaviour_rows(results, model=None, where=None):
    out = []
    for p in sorted(glob.glob(f"{results}/simonsays/{model or '*'}__*.json")):
        if p.endswith("_states.npz"):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        mo = os.path.basename(p).split("__")[0]
        for r in d.get("rows", []):
            if where and not where(r):
                continue
            out.append(dict(r, _model=mo))
    return out


def _wrap(t, n):
    words, lines, cur = str(t).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


HARD_PROBES = ("p8", "p4ref", "p3", "p6", "p7", "p5")


def hard_questions(results="results", out="figures", model=None, root=None, n=4,
                   probes=HARD_PROBES):
    from vlens import corpus as C
    from vlens.data import simonsays as SS
    rows = _behaviour_rows(results, model,
                           lambda r: str(r.get("subcategory")) in probes)
    best = {}
    for r in rows:
        k = r["subcategory"]
        img = os.path.join(root or SS.ROOT, r["image"])
        if not os.path.exists(img):
            continue
        q = str(r.get("question") or "")
        if len(q) < 30:
            continue
        if k not in best or len(q) > len(best[k][0].get("question") or ""):
            best[k] = (r, img)
    pick = [best[k] for k in probes if k in best][:n]
    if len(pick) < 2:
        return None
    st.use()
    k = len(pick)
    w = min(st.MAX_WIDTH, 2.75 * k)
    fig = plt.figure(figsize=(w, w / k * 1.72))
    fig.set_layout_engine("none")
    fs = matplotlib.rcParams["xtick.labelsize"] * 0.95
    pad, top = 0.012, 0.985
    col = (1.0 - pad * (k + 1)) / k
    img_h = 0.585
    for i, (r, img) in enumerate(pick):
        x = pad + i * (col + pad)
        ax = fig.add_axes([x, top - img_h, col, img_h])
        st.image_panel(ax, C.load_image(img, square=448))
        chars = max(int((col * w) / (fs * 0.5 / 72.0)) - 1, 12)
        fig.text(x, top - img_h - 0.045, _wrap(str(r.get("question") or ""), chars),
                 ha="left", va="top", color=st.INK, fontsize=fs, linespacing=1.45)
    return st.save(fig, _at(out, "hard_questions"), check=False)


def read_surface(results="results", out="figures", model=None, key="v_read"):
    p = _newest(f"{results}/lensbench*/{model or '*'}__bench.json")
    d = _load(p) if p else None
    if not d or not d.get("families"):
        return None
    got = {}
    for f, r in d["families"].items():
        if f"model|{key}" not in r["auroc"]:
            continue
        got.setdefault(str(f).split("_")[0], []).append(
            (np.asarray(r["depths"], float), np.asarray(r["auroc"][f"model|{key}"], float),
             int(r["n"])))
    order = [s_ for s_ in ("wmdp", "gemini", "vins", "nuscenes", "openimages") if s_ in got]
    if len(order) < 3:
        return None
    x = got[order[0]][0][0]
    Z = np.stack([np.average(np.stack([r[1] for r in got[s_]]), 0,
                             weights=np.array([r[2] for r in got[s_]], float))
                  for s_ in order])
    from matplotlib.collections import PolyCollection
    st.use()
    fig = plt.figure(figsize=(15.0, 5.4))
    fig.set_layout_engine("none")
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    a2 = fig.add_subplot(1, 2, 2)
    ax.set_box_aspect((2.3, 1.15, 0.72), zoom=1.36)
    verts, faces = [], []
    for i, s_ in enumerate(order):
        c = _src_colour(s_)
        verts.append([(x[0], 0.5)] + list(zip(x, Z[i])) + [(x[-1], 0.5)])
        faces.append(c)
        ax.plot(x, [i] * len(x), Z[i], color=c, lw=3.0, zorder=10 + i)
    ax.add_collection3d(PolyCollection(verts, facecolors=faces, alpha=0.24,
                                       edgecolors="none"), zs=range(len(order)), zdir="y")
    xx, yy = np.meshgrid([0, 1], [-0.45, len(order) - 0.55])
    ax.plot_surface(xx, yy, np.full_like(xx, 0.5, dtype=float),
                    color=st.SUBINK, alpha=0.16, shade=False, zorder=0)
    ax.set_xlim(0, 1); ax.set_ylim(-0.45, len(order) - 0.55); ax.set_zlim(0, 1.04)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["input", "half", "output"])
    ax.set_yticks(range(len(order))); ax.set_yticklabels([""] * len(order))
    ax.set_zticks([]); ax.set_zticklabels([])
    ax.text(0.0, len(order) - 0.5, 0.5, "chance ", color=st.INK,
            ha="right", va="center")
    ax.view_init(elev=18, azim=-62)
    ax.set_facecolor(st.SURFACE)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(st.SURFACE)
        pane.pane.set_edgecolor(st.GRID)
        pane.line.set_color(st.AXIS)
        pane.set_tick_params(colors=st.INK, pad=-1)
    ax.zaxis.set_tick_params(pad=4)
    ax.grid(color=st.GRID)
    ax.set_position([-0.04, -0.02, 0.56, 1.04])

    im = a2.pcolormesh(np.r_[x, x[-1] + (x[-1] - x[-2])] - (x[1] - x[0]) / 2,
                       np.arange(len(order) + 1) - 0.5, Z,
                       cmap=st.warm_cmap(), vmin=0.35, vmax=1.0, shading="flat")
    a2.set_yticks(range(len(order)))
    a2.set_yticklabels([SOURCE.get(s_, s_) for s_ in order])
    a2.set_xticks([x[0], 0.5, x[-1]]); a2.set_xticklabels(["input", "half", "output"])
    a2.set_xlim(x[0] - (x[1] - x[0]) / 2, x[-1] + (x[1] - x[0]) / 2)
    a2.set_xlabel("depth through the network")
    a2.grid(False)
    for sp in a2.spines.values():
        sp.set_visible(False)
    a2.tick_params(length=0)
    for i in range(len(order)):
        hit = [j for j in range(len(x) - 1) if Z[i, j] >= 0.5 and Z[i, j + 1] >= 0.5]
        if hit:
            a2.plot([x[hit[0]]], [i], marker="|", ms=22, mew=3.0, color=st.INK, zorder=5)
    cax = fig.add_axes([0.945, 0.20, 0.014, 0.66])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("AUROC for the answer\nthe model gives")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=st.INK)
    cb.set_ticks([0.4, 0.6, 0.8, 1.0])
    a2.set_position([0.585, 0.20, 0.325, 0.72])
    return st.save(fig, _at(out, "read_surface"), check=False, tight=True)


def read_inversion(results="results", out="figures", model=None, key="v_read"):
    p = _newest(f"{results}/lensbench*/{model or '*'}__bench.json")
    d = _load(p) if p else None
    if not d or not d.get("families"):
        return None
    fams = [f for f in sorted(d["families"]) if f"model|{key}" in d["families"][f]["auroc"]]
    if len(fams) < 3:
        return None
    st.use()
    fig, (a1, a2) = st.wide(1, 2, unit=(6.5, 3.4), sharey=True)
    for ax, signed in ((a1, True), (a2, False)):
        stack = []
        for f in fams:
            r = d["families"][f]
            x = np.asarray(r["depths"], float)
            y = np.asarray(r["auroc"][f"model|{key}"], float)
            if not signed:
                y = 0.5 + np.abs(y - 0.5)
            ax.plot(x, y, color=_src_colour(f), lw=1.3, alpha=0.55, zorder=3)
            stack.append(y)
        ax.plot(x, np.nanmean(stack, 0), color=st.INK, lw=2.6, zorder=5)
        if not signed:
            ses = []
            for f in fams:
                c = (d["families"][f].get("ci") or {}).get(f"model|{key}")
                if c:
                    ses.append(np.abs(np.asarray(c[1], float)
                                      - np.asarray(c[0], float)) / (2 * 1.96))
            if ses:
                null = 0.5 + np.nanmean(ses, 0) * np.sqrt(2 / np.pi)
                ax.plot(x, null, color=st.MUTED, lw=1.3, dashes=(3, 3), zorder=4)
                j = int(0.62 * (len(x) - 1))
                ax.annotate("what folding alone would give", xy=(x[j], null[j]),
                            xytext=(0, -6), textcoords="offset points",
                            color=st.INK, ha="center", va="top")
        st.chance(ax, 0.5)
        st.depth_axis(ax, "depth")
        ax.set_ylim(0, 1.03)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        st.despine(ax)
    a1.set_ylabel(f"AUROC for the answer")
    a1.set_title("as the lens signs it", loc="left", color=st.INK2)
    a2.set_title("with the sign ignored", loc="left", color=st.INK2)
    st.end_labels(a2, x[-1], [(np.nanmean(stack, 0)[-1], "every family, mean", st.INK)]
                  + [(d["families"][f]["auroc"][f"model|{key}"][-1],
                      SOURCE.get(str(f).split("_")[0], f), _src_colour(f))
                     for f in {str(x_).split("_")[0]: x_ for x_ in fams}.values()])
    return st.save(fig, _at(out, "read_inversion"))


def direction_geometry(results="results", out="figures", model=None):
    p = _newest(f"{results}/directions/{model or '*'}__directions.json")
    d = _load(p) if p else None
    d = (d or {}).get("summary") or d
    if not d or not d.get("cosine"):
        return None
    names = list(d["directions"])
    axes_ = [a for a in AXES_ORDER if any(a in v for v in d["moved"].values())] or AXES_ORDER
    n = len(names)
    C = np.array([[float(d["cosine"].get(f"{x}|{y}", 0.0)) for y in names] for x in names])
    G = (C + C.T) / 2
    J = np.eye(n) - np.ones((n, n)) / n
    w, V = np.linalg.eigh(J @ G @ J)
    xy = V[:, ::-1][:, :2] * np.sqrt(np.clip(w[::-1][:2], 0, None))
    st.use()
    fig, (a1, a2) = st.wide(1, 2, unit=(7.02, 4.0))
    mid = xy.mean(0)
    span = float(np.hypot(*(xy.max(0) - xy.min(0)))) or 1.0
    crowd = [sum(1 for j in range(i) if np.hypot(*(xy[i] - xy[j])) < 0.08 * span)
             for i in range(n)]
    for i, nm in enumerate(names):
        ours = nm.startswith("V-Write")
        c = (st.LENS if ours else st.MUTED if nm == "random"
             else st.SERIES[1] if nm.startswith("difference") else st.SERIES[2])
        a1.plot([xy[i, 0]], [xy[i, 1]], marker="o", ms=13 if ours else 8,
                color=c, zorder=4)
        v = xy[i] - mid
        nrm = float(np.hypot(*v)) or 1.0
        dx, dy = 19 * v[0] / nrm, 19 * v[1] / nrm
        dy += crowd[i] * (-26 if dy < 0 else 26)
        a1.annotate(nm.replace("V-Write, ", ""), xy=xy[i], xytext=(dx, dy),
                    textcoords="offset points", color=st.INK,
                    ha="left" if dx > 1 else "right" if dx < -1 else "center",
                    va="bottom" if dy > 1 else "top" if dy < -1 else "center")
    a1.set_xticks([]); a1.set_yticks([])
    for sp in a1.spines.values():
        sp.set_visible(False)
    a1.set_xlabel("angle between them, best plane")
    a1.margins(0.46)

    al = max(d["alphas"])
    M = np.array([[float((d["moved"].get(f"{nm}|{al}") or {}).get(ax, np.nan))
                   for ax in axes_] for nm in names])
    lim = np.nanmax(np.abs(M)) or 1.0
    from matplotlib.patches import Rectangle
    a2.imshow(M, cmap=st.warm_cmap(diverging=True), vmin=-lim, vmax=lim, aspect="auto")
    a2.grid(False)
    a2.set_xticks(range(len(axes_))); a2.set_xticklabels(axes_)
    a2.set_yticks(range(n)); a2.set_yticklabels(names)
    for i in range(n):
        for j in range(len(axes_)):
            if np.isfinite(M[i, j]):
                own = names[i].startswith("V-Write") and names[i].endswith(axes_[j])
                a2.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", color=st.INK)
                if own:
                    a2.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           ec=st.LENS, lw=2.0, zorder=5))
    a2.set_xlabel(f"what moves, at a dose of {al:g}")
    for sp in a2.spines.values():
        sp.set_visible(False)
    a2.tick_params(length=0)
    return st.save(fig, _at(out, "direction_geometry"))


REWRITTEN = ('fx_attrplane', 'fx_bench', 'fx_beyond', 'fx_channel', 'fx_coco', 'fx_commit', 'fx_cuegeom', 'fx_depth', 'fx_detect', 'fx_dirspace', 'fx_evalaware', 'fx_geom', 'fx_halluc', 'fx_layers', 'fx_mapgallery', 'fx_packaging', 'fx_patch', 'fx_patchwords', 'fx_pwlock_story', 'fx_retention', 'fx_standing', 'fx_trace', 'fx_transitions', 'fx_vread_yesno', 'fx_vwrite_gains')


def _resolve(name):
    import importlib
    for mod in reversed(REWRITTEN):
        try:
            m = importlib.import_module(f"vlens.figures.{mod}")
        except Exception:
            continue
        fn = getattr(m, name, None)
        if callable(fn):
            return fn
    return globals().get(name)


def extra_figures():
    import importlib
    out = {}
    for mod in REWRITTEN:
        try:
            m = importlib.import_module(f"vlens.figures.{mod}")
        except Exception:
            continue
        for nm in getattr(m, "__all__", None) or dir(m):
            fn = getattr(m, nm, None)
            if (not nm.startswith("_") and callable(fn) and nm != "all_figures"
                    and getattr(fn, "__module__", "") == f"vlens.figures.{mod}"
                    and nm not in globals()
                    and (fn.__doc__ or "").lstrip().startswith(("MAIN", "APPENDIX"))):
                out[nm] = fn
    return list(out.items())


DROPPED = frozenset({"hard_questions", "packaged_as_a_test", "eval_tell", "eval_channel",
                     "dose_to_flip", "read_surface", "read_inversion", "sandbag_depth",
                     "sandbag_accuracy", "hallucination", "behaviour_directions",
                     "lens_scoreboard", "lens_canyon", "lens_by_depth", "answer_plane",
                     "lens_kl",
                     "carrier",
                     "refusal_read", "refusal_cascade",
                     "vwrite_baselines", "patch_lenses", "patch_sources", "attribution_plane",
                     "signed_handle",
                     "direction_geometry", "refusal_steer",
                     "map_methods", "map_questions", "map_swaps",
                     "layer_story", "scene_store",
                     "edit_series_more", "standing_swap",
                     "eval_depth", "sandbag_channels",
                     "patch_jailbreak", "jailbreak_coverage_fan", "handoff",
                     "read_image", "counterfactual", "transport_geometry"})


def all_figures(results="results", out="figures", model=None, average=False):
    global SUFFIX, AVERAGE
    AVERAGE = bool(average) and not model
    SUFFIX = f"__{model}" if model else ("__averaged" if AVERAGE else "")
    st.STAMP = model or ""
    made = {}
    for fn in (lens_by_depth, lens_read_by_depth, lens_kl, patch_lenses, patch_sources, patch_causal,
               baseline_agreement, patch_atlas, gallery, crossquestion,
               patch_jailbreak, refusal_cascade, steer, carrier, cost_and_quality,
               refusal_steer, refusal_read, handoff, read_image, counterfactual,
               read_surface, read_inversion, direction_geometry, hallucination,
               eval_tell, eval_channel, authority_ladder, hard_questions):
        name = fn.__name__
        if name in DROPPED:
            continue
        use = _resolve(name) or fn
        try:
            p = use(results=results, out=out, model=model)
        except Exception as e:
            p = f"failed: {type(e).__name__}: {e}"
        made[name] = p
    for name, fn in extra_figures():
        if name in DROPPED:
            continue
        try:
            made[name] = fn(results=results, out=out, model=model)
        except Exception as e:
            made[name] = f"failed: {type(e).__name__}: {e}"
    SUFFIX, AVERAGE, st.STAMP = "", False, ""
    return made
