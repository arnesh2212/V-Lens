"""Assemble every headline number, with clustered inference, into one report."""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, "/workspace/vlm/code")
import numpy as np
import pandas as pd

from jvlm import stats as st

RES = sys.argv[1] if len(sys.argv) > 1 else "/workspace/vlm/results"
TAG = sys.argv[2] if len(sys.argv) > 2 else "qwen2b"
OUT = f"{RES}/REPORT"
R: dict = {}
lines: list[str] = []


def sec(title: str):
    lines.append(f"\n## {title}\n")


def say(s: str = ""):
    lines.append(s)


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:.1f}"


# --------------------------------------------------------------- blindsight
p = f"{RES}/blindsight/{TAG}_raw.parquet"
if os.path.exists(p):
    df = pd.read_parquet(p)
    R["blindsight"] = {"n_items": int(df.i.nunique())}
    sec(f"Erasure ({df.i.nunique()} items)")
    rows = []
    for cond in df.cond.unique():
        d = df[df.cond == cond]
        for ch, col in (("word", "word_correct"), ("letter", "letter_correct")):
            m_, lo, hi = st.cluster_bootstrap_ci(d, col)
            rows.append(dict(cond=cond, channel=ch, mean=m_, lo=lo, hi=hi,
                             n=len(d)))
    acc = pd.DataFrame(rows)
    acc.to_csv(f"{RES}/blindsight/{TAG}_accuracy_ci.csv", index=False)
    piv = acc.pivot(index="cond", columns="channel", values="mean")
    say("| condition | says the word | presses the key |")
    say("|---|---|---|")
    order = ["clean", "wrong_image", "abl_all", "abl_e90", "abl_e80", "abl_e70",
             "abl_e60", "abl_hi", "abl_mid", "abl_lo", "abl_prompt", "abl_img",
             "abl_rand", "abl_other", "abl_unemb"]
    for c in [x for x in order if x in piv.index]:
        r = acc[acc.cond == c]
        w = r[r.channel == "word"].iloc[0]
        l = r[r.channel == "letter"].iloc[0]
        say(f"| {c} | {pct(w['mean'])} [{pct(w.lo)}, {pct(w.hi)}] | "
            f"{pct(l['mean'])} [{pct(l.lo)}, {pct(l.hi)}] |")

    for cond in ("abl_all", "abl_e80", "abl_unemb", "abl_rand"):
        if cond not in set(df.cond):
            continue
        inter = st.interaction(df, "clean", cond, "word_margin", "letter_margin")
        mw = st.mcnemar(df, "clean", cond, "word_correct")
        ml = st.mcnemar(df, "clean", cond, "letter_correct")
        R.setdefault("interaction", {})[cond] = dict(
            interaction=inter, mcnemar_word=mw, mcnemar_letter=ml)
        say(f"\n`{cond}` channel x intervention interaction: "
            f"dd = {inter.get('dd', float('nan')):.2f} "
            f"[{inter.get('ci_lo', float('nan')):.2f}, "
            f"{inter.get('ci_hi', float('nan')):.2f}], "
            f"Wilcoxon p = {inter.get('p', float('nan')):.2g}; "
            f"McNemar word p = {mw['p']:.2g} "
            f"(flips {mw['n01']}/{mw['n01']+mw['n10']}), "
            f"letter p = {ml['p']:.2g}.")
    if "energy" in df.columns:
        e = df.groupby("cond").energy.mean()
        R["energy"] = {k: float(v) for k, v in e.items()}
        say("\nRelative residual energy removed, ||dh||/||h||: " +
            ", ".join(f"{k} {v:.3f}" for k, v in e.items() if v > 0))

# ------------------------------------------------------------------- routes
p = f"{RES}/routes/{TAG}_raw.parquet"
if os.path.exists(p):
    df = pd.read_parquet(p)
    sec(f"Report routes ({df.i.nunique()} items)")
    cols = [c for c in df.columns if c.endswith("_correct")]
    g = df.groupby("arm")[cols].mean()
    R["routes"] = g.round(4).to_dict()
    say("| erased component | " + " | ".join(c.replace("_correct", "")
                                             for c in cols) + " |")
    say("|" + "---|" * (len(cols) + 1))
    for arm in ["clean", "full", "par", "perp", "unemb"]:
        if arm in g.index:
            say(f"| {arm} | " + " | ".join(pct(g.loc[arm, c]) for c in cols) + " |")
    gp = f"{RES}/routes/{TAG}_geometry.parquet"
    if os.path.exists(gp):
        geo = pd.read_parquet(gp)
        band = geo.assign(b=(geo.depth * 4).astype(int) / 4).groupby("b")[
            ["cos_V_WU", "frac_par"]].mean()
        R["geometry"] = band.round(4).to_dict()
        say("\nAlignment of the lens direction with the raw unembedding row, "
            "by depth band:")
        say("| depth | cos(v, W_U) | fraction of v in span(W_U) |")
        say("|---|---|---|")
        for b, r in band.iterrows():
            say(f"| {b:.2f} | {r.cos_V_WU:.3f} | {r.frac_par:.3f} |")

# ------------------------------------------------------------ source / reality
p = f"{RES}/source/{TAG}_stats.json"
if os.path.exists(p):
    s_ = json.load(open(p))
    R["source"] = s_
    sec(f"Saying what it saw (n = {s_['n']})")
    say("| question | correct |")
    say("|---|---|")
    for k, lab in (("presence_correct", "is each object visible?"),
                   ("source_correct", "which one can you see?"),
                   ("told_correct", "which one were you only told about?"),
                   ("consistent", "are those two answers different objects?")):
        if k in s_:
            v = s_[k]
            say(f"| {lab} | {pct(v['mean'])} [{pct(v['lo'])}, {pct(v['hi'])}] |")
    if "presence_vs_source_p" in s_:
        say(f"\nPresence against source, Wilcoxon p = "
            f"{s_['presence_vs_source_p']:.2g}.")

p = f"{RES}/reality/{TAG}_stats.json"
if os.path.exists(p):
    s_ = json.load(open(p))
    R["reality"] = s_
    sec("Reality monitoring")
    say(f"- the model answers the reality check correctly "
        f"{pct(s_.get('reality_check'))}% of the time; the plain presence "
        f"question, {pct(s_.get('plain_presence'))}%")
    if "pooled_auc" in s_:
        say("- probe AUC for whether the asserted object is really present: " +
            ", ".join(f"{k} {v:.3f}" for k, v in s_["pooled_auc"].items()))

# ----------------------------------------------------------------- language
p = f"{RES}/language/{TAG}_summary.csv"
if os.path.exists(p):
    d = pd.read_csv(p, index_col=0)
    R["language"] = d.round(4).to_dict()
    sec("How many output buffers")
    cols = [c for c in d.columns if c.endswith("_correct")]
    say("| erased | " + " | ".join(c.replace("_correct", "") for c in cols) + " |")
    say("|" + "---|" * (len(cols) + 1))
    for arm, r in d.iterrows():
        say(f"| {arm} | " + " | ".join(pct(r[c]) for c in cols) + " |")

# ------------------------------------------------------------- depth crossover
p = f"{RES}/depthcross/{TAG}_stats.json"
if os.path.exists(p):
    s_ = json.load(open(p))
    R["depthcross"] = s_
    sec("Where the Jacobian earns its keep")
    say(f"With no intervention the model says the right word "
        f"{pct(s_['clean_word'])}% of the time.")
    say("")
    say("| band of layers | cos(v, W_U) | erasing the lens direction | erasing the row |")
    say("|---|---|---|---|")
    for b, v in s_["bands"].items():
        say(f"| {b} | {v['cos']:.2f} | {pct(v['lens_word'])} | "
            f"{pct(v['unembed_word'])} |")

# -------------------------------------------------------- energy-matched control
p = f"{RES}/energy/{TAG}_summary.csv"
if os.path.exists(p):
    d = pd.read_csv(p, index_col=0)
    R["energy_control"] = d.round(4).to_dict()
    sec("A random control matched in energy, not only in rank")
    say("| ablation | residual energy removed | says the word | presses the key |")
    say("|---|---|---|---|")
    for arm, r in d.iterrows():
        say(f"| {arm} | {r['energy']:.3f} | {pct(r['word_correct'])} | "
            f"{pct(r['letter_correct'])} |")

# ---------------------------------------------------------------- families
p = f"{RES}/families/{TAG}_summary.csv"
if os.path.exists(p):
    d = pd.read_csv(p)
    R["families"] = d.round(4).to_dict("records")
    sec("Concept families beyond nouns")
    say("| family | says the word, clean | erased | presses the key, clean | erased |")
    say("|---|---|---|---|---|")
    for fam in d.family.unique():
        c = d[(d.family == fam) & (d.arm == "clean")].iloc[0]
        e = d[(d.family == fam) & (d.arm == "erase")].iloc[0]
        say(f"| {fam} | {pct(c.word_correct)} | {pct(e.word_correct)} | "
            f"{pct(c.letter_correct)} | {pct(e.letter_correct)} |")

# ------------------------------------------------------------------- patch
p = f"{RES}/patch/{TAG}_stats.json"
if os.path.exists(p):
    s_ = json.load(open(p))
    R["patch"] = s_
    sec("Injecting the two halves of a perceptual direction")
    say(f"Verbalizable fraction of what a photograph writes: "
        f"{pct(s_.get('V_mean'))}% against a random-direction null of "
        f"{pct(s_.get('V_null'))}% (channel row space truncated to rank "
        f"{s_.get('rank')}).")
    for arm in ("verbal", "use", "random"):
        if arm in s_:
            v = s_[arm]
            say(f"- injecting the {arm} copy: word-minus-letter interaction "
                f"dd = {v.get('dd', float('nan')):.2f}, p = "
                f"{v.get('p', float('nan')):.2g}")
    sp = f"{RES}/patch/{TAG}_summary.csv"
    if os.path.exists(sp):
        d = pd.read_csv(sp)
        say("")
        say("| injected | strength | rank of the absent object when naming | "
            "forced choice picks it (%) |")
        say("|---|---|---|---|")
        for _, r in d.iterrows():
            say(f"| {r['arm']} | {r['alpha']:g} | {r['say_rank']:.0f} | "
                f"{pct(r['letter_picks'])} |")

# --------------------------------------------------------------- provenance
p = f"{RES}/provenance/{TAG}_probes.parquet"
if os.path.exists(p):
    df = pd.read_parquet(p)
    sec("Provenance")
    g = df.groupby(["probe", "view"]).auc.agg(["mean", "sem", "count"])
    R["provenance"] = g["mean"].round(4).to_dict()
    say("| probe | whole residual | outside workspace | random subspace | workspace |")
    say("|---|---|---|---|---|")
    for probe in ("content", "provenance"):
        cells = []
        for v in ("full", "orthogonal", "random", "jspace"):
            try:
                cells.append(pct(g.loc[(probe, v), "mean"]))
            except KeyError:
                cells.append("n/a")
        say(f"| {probe} | " + " | ".join(cells) + " |")
    bp = f"{RES}/provenance/{TAG}_behaviour.parquet"
    if os.path.exists(bp):
        b = pd.read_parquet(bp)
        R["provenance_behaviour"] = dict(
            source_correct=float(b.source_correct.mean()),
            content_correct=float(b.content_correct.mean()), n=len(b))
        say(f"\nAsked directly, the model identifies which channel a concept "
            f"came from {pct(b.source_correct.mean())}% of the time, while "
            f"answering a content question of identical format "
            f"{pct(b.content_correct.mean())}% of the time (n={len(b)}).")

# ------------------------------------------------------------------ forgery
for f in sorted(glob.glob(f"{RES}/forgery/*_summary.json")):
    s = json.load(open(f))
    name = os.path.basename(f).replace("_summary.json", "")
    R.setdefault("forgery", {})[name] = s
    sec(f"Forgery: {name}")
    say(f"Median readout rank of the target on a clean image: "
        f"{s['lens_rank_clean']:.0f}.")
    say("")
    say("| output-preservation weight | readout rank | in top 10 | output identical | both |")
    say("|---|---|---|---|---|")
    for r in s.get("frontier", []):
        say(f"| {r['beta']:g} | {r['lens_rank_adv']:.0f} | {pct(r['hit_top10'])} | "
            f"{pct(r['output_unchanged'])} | {pct(r['forgery_rate'])} |")
    b = s.get("best", {})
    if b:
        say(f"\nBest operating point: {pct(b['hit_top10'])}% of images put the "
            f"planted word in the readout's top 10 while "
            f"{pct(b['output_unchanged'])}% of generations are byte-identical "
            f"({pct(b['forgery_rate'])}% achieve both).")

# -------------------------------------------------------------- competition
p = f"{RES}/competition/{TAG}_stats.json"
if os.path.exists(p):
    s = json.load(open(p))
    R["competition"] = s
    sec("Cross-modal competition")
    say(f"Effective number of object words the workspace holds at once "
        f"(participation ratio): {s.get('participation_mean', float('nan')):.2f}.")
    say("")
    say("| relationship | Spearman rho | p |")
    say("|---|---|---|")
    for k in sorted(k for k in s if k.startswith("rho_")):
        say(f"| {k[4:].replace('_', ' vs ', 1).replace('_',' ')} | "
            f"{s[k]:+.3f} | {s['p' + k[3:]]:.2g} |")

# ------------------------------------------------------------------- gate 0
p = f"{RES}/gate0/{TAG}_summary.csv"
if os.path.exists(p):
    s = pd.read_csv(p)
    sec("Readout: Jacobian lens against logit lens")
    best = s.loc[s.groupby("jac").auc.idxmax()]
    R["gate0"] = best.round(4).to_dict("records")
    say("| lens | best depth | present-vs-absent AUC | median rank of the target |")
    say("|---|---|---|---|")
    for _, r in best.iterrows():
        say(f"| {'Jacobian' if r.jac else 'logit'} | {r.depth:.2f} | "
            f"{r.auc:.3f} | {r.median_rank_target:.0f} |")
    # earliest depth at which each lens puts the target inside the top 10
    for jac in (1, 0):
        d = s[(s.jac == jac) & (s.top10_target > 0.25)]
        if len(d):
            R.setdefault("surfacing_depth", {})["jacobian" if jac else "logit"] = \
                float(d.depth.min())
    if "surfacing_depth" in R:
        say("\nEarliest depth at which the object's name reaches the readout's "
            "top 10 on a quarter of images: " +
            ", ".join(f"{k} {v:.2f}" for k, v in R["surfacing_depth"].items()))

# ----------------------------------------------------------------- lens qual
p = f"{RES}/lensqual/{TAG}_quality.csv"
if os.path.exists(p):
    q = pd.read_csv(p)
    R["lens_quality"] = q.round(4).to_dict("list")
    sec("Lens diagnostics")
    if "cross_lens_cos_v" in q:
        say(f"- concept directions from a text-fitted lens agree with the "
            f"multimodal one at mean cosine {q.cross_lens_cos_v.mean():.3f}")
    say(f"- top-1 agreement with the logit lens ranges "
        f"{q.top1_agreement.min():.3f} to {q.top1_agreement.max():.3f} across depth")
    say(f"- mean cos(v_c, W_U[c]) = {q.cos_v_unembed.mean():.3f}")

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
json.dump(R, open(f"{OUT}.json", "w"), indent=2, default=float)
open(f"{OUT}.md", "w").write("# Results\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nwrote {OUT}.json and {OUT}.md")
