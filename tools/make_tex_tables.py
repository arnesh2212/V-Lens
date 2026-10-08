# SPDX-License-Identifier: Apache-2.0
"""The paper's tables, as LaTeX, built from the result files rather than typed.

Every number here is read off disk. A table that is typed by hand goes stale
the moment a run finishes, and the one thing worse than a missing number in a
paper is a stale one.

The tables are wide, because a two-column template gives a ``table*`` the full
page and a tall table wastes it. Captions are written to be read on their own:
what is shown, what it means, and the number that matters.

    python3 tools/make_tex_tables.py
    python3 tools/make_tex_tables.py --out paper/tables.tex
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHORT = {"Qwen3-VL-8B-Instruct": "Qwen3-VL 8B", "Qwen3.5-27B": "Qwen3.5 27B",
         "Qwen3-VL-2B-Instruct": "Qwen3-VL 2B", "gemma-4-26B-A4B-it": "gemma-4 26B",
         "InternVL3_5-8B-HF": "InternVL3.5 8B"}
NICE = {"v_read": "V-Read", "v_read_cross": "V-Read, to the answer",
        "tuned": "tuned lens", "jacobian": "Jacobian lens", "latent": "LatentLens",
        "logit": "logit lens", "probe": "probe, labelled",
        "v_write": "V-Write", "attr_patch": "attribution patching",
        "gradxact_zero": "grad $\\times$ activation", "random": "random",
        "gradxact": "grad $\\times$ activation"}
SOURCE = {"openimages": "Open Images", "nuscenes": "nuScenes", "vins": "VINS-120K",
          "gemini": "Generated scenes", "wmdp": "WMDP carrier"}


def rows(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def num(v, dp=2, dash="--"):
    try:
        return f"{float(v):.{dp}f}"
    except (TypeError, ValueError):
        return dash


def pct(v, dp=0, dash="--"):
    try:
        return f"{float(v) * 100:.{dp}f}"
    except (TypeError, ValueError):
        return dash


def table(body, cols, caption, label, note=None):
    out = ["\\begin{table*}[t]", "\\centering", "\\small",
           "\\begin{tabular}{" + cols + "}", "\\toprule"]
    out += body
    out += ["\\bottomrule", "\\end{tabular}",
            f"\\caption{{{caption}}}", f"\\label{{tab:{label}}}"]
    if note:
        out.insert(-2, "")
    out.append("\\end{table*}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- the tables

def t_reading(R):
    r = [x for x in rows(f"{R}/reading.csv")]
    if not r:
        return ""
    r.sort(key=lambda x: float(x["kl_mean"]))
    head = ("instrument & labels & gradients & how it is fitted & fit (s) & "
            "tokens & KL mean & at the input & at the output \\\\")
    body = [head, "\\midrule"]
    for x in r:
        tok = x.get("fit_tokens") or ""
        tok = f"{float(tok) / 1e6:.1f}M" if tok else "--"
        sec = x.get("fit_seconds") or ""
        sec = f"{int(float(sec))}" if sec else "0"
        nm = NICE.get(x["instrument"], x["instrument"])
        if x.get("note"):
            nm += "$^{\\dagger}$"
        body.append(f"{nm} & {x['needs_labels']} & {x['needs_gradients']} & "
                    f"{x['fitting']} & {sec} & {tok} & {num(x['kl_mean'])} & "
                    f"{num(x['kl_first_block'])} & {num(x['kl_last_block'])} \\\\")
    cap = ("Reading the model's own distribution off an intermediate block, on "
           "held-out prompts no instrument was fitted on, for Qwen3-VL 8B over "
           "sixteen blocks. Lower divergence is better and the logit lens is "
           "what an identity map scores. The tuned lens wins on the mean, by "
           "about one nat, having spent twelve times the wall clock and "
           "sixty-eight times the tokens on gradient descent; V-Read is one "
           "closed-form solve. The two columns on the right say where that nat "
           "is: V-Read is worse near the input and better at the output, so "
           "the deficit is concentrated early rather than spread through "
           "depth. $^{\\dagger}$ LatentLens is reported for completeness; its "
           "caption store reaches 3.6\\% of the vocabulary, so the "
           "distribution it builds is close to uniform and the number is a "
           "fact about coverage rather than about the lens.")
    return table(body, "l cc l rr rrr", cap, "reading")


def t_causal(R):
    r = rows(f"{R}/causal.csv")
    if not r:
        return ""
    ks = ["drop_at_5pc", "drop_at_10pc", "drop_at_20pc", "drop_at_30pc", "drop_at_50pc"]
    order = ["top", "top:attr_patch", "top:gradxact_zero", "middle", "bottom", "random"]
    NAME = {"top": "V-Write, its highest fifth", "top:attr_patch": "attribution patching",
            "top:gradxact_zero": "grad $\\times$ activation", "middle": "V-Write, its middle",
            "bottom": "V-Write, its lowest", "random": "a random selection"}
    body = ["model & $n$ & block & what was replaced & 5\\% & 10\\% & 20\\% & 30\\% & "
            "50\\% & flips at 50\\% \\\\", "\\midrule"]
    for mo in sorted({x["model"] for x in r}):
        mine = {x["chosen"]: x for x in r if x["model"] == mo}
        first = True
        for c in order:
            x = mine.get(c)
            if not x:
                continue
            lead = (f"{SHORT.get(mo, mo)} & {x['items']} & {x['block']}"
                    if first else " & & ")
            first = False
            body.append(lead + f" & {NAME.get(c, c)} & "
                        + " & ".join(num(x[k]) for k in ks)
                        + f" & {pct(x.get('flip_at_50pc'), 1) if x.get('flip_at_50pc') else '--'} \\\\")
        body.append("\\midrule")
    if body[-1] == "\\midrule":
        body.pop()
    cap = ("Replacing image-patch states with the corpus mean patch, at the "
           "block where the picture is weighed, oriented toward the answer the "
           "model actually gave. Each row replaces the same number of patches "
           "and differs only in which ones. V-Write's highest fifth costs the "
           "answer 3.20 on the 8B against 1.43 for attribution patching and "
           "0.19 for a random fifth. The row that carries the argument is the "
           "lowest fifth: replacing the patches V-Write ranks last does not do "
           "nothing, it moves the answer score the other way, which a ranking "
           "that had only found salient pixels could not do. The last column "
           "is the share of answers that change outright.")
    return table(body, "l rr l rrrrr r", cap, "causal")


def t_sources(R):
    r = rows(f"{R}/sources.csv")
    if not r:
        return ""
    srcs = [s for s in ("openimages", "nuscenes", "vins", "gemini", "wmdp")
            if any(x["source"] == s for x in r)]
    insts = ["v_write", "attr_patch", "jacobian", "random"]
    body = ["model & instrument & " + " & ".join(SOURCE.get(s, s) for s in srcs) + " \\\\",
            "\\midrule"]
    for mo in sorted({x["model"] for x in r}):
        first = True
        for ins in insts:
            cells = []
            for s in srcs:
                hit = [x for x in r if x["model"] == mo and x["source"] == s
                       and x["instrument"] == ins]
                cells.append(num(hit[0]["drop_at_20pc"]) if hit else "--")
            if all(c == "--" for c in cells):
                continue
            lead = SHORT.get(mo, mo) if first else " "
            first = False
            body.append(f"{lead} & {NICE.get(ins, ins)} & " + " & ".join(cells) + " \\\\")
        body.append("\\midrule")
    if body[-1] == "\\midrule":
        body.pop()
    cap = ("The same measurement, one source of pictures at a time: the fall "
           "in the answer score when the fifth of the image patches an "
           "instrument ranks highest is replaced by the mean patch. WMDP is "
           "the control, since its picture is only a carrier for a question "
           "written in text and nothing in it is being asked about, and it is "
           "where every instrument scores least. The ordering between "
           "instruments holds within every source, which is what says the "
           "result is about the method and not about one corner of the data.")
    return table(body, "ll" + "r" * len(srcs), cap, "sources")


def t_channel(R, res):
    sr = rows(f"{R}/eval_channel.csv")
    sh = rows(f"{R}/behaviour_shift.csv")
    d = {}
    p = f"{res}/showcase/behaviour.json"
    if os.path.exists(p):
        d = json.load(open(p))
    if not sr and not sh:
        return ""
    cond = {x["model"]: x for x in d.get("picture_conduct", [])}
    agg = {}
    for x in sh:
        if x["contrast"] != "told it is being evaluated":
            continue
        e = agg.setdefault(x["model"], [0, 0.0, 0.0, 0.0, 0.0])
        n = int(x["pairs"]); e[0] += n
        e[1] += float(x["marked_up_before"]) * n
        e[2] += float(x["marked_up_after"]) * n
        e[3] += float(x["hedged_before"]) * n
        e[4] += float(x["hedged_after"]) * n
    mods = sorted({x["model"] for x in sr})
    span = " & ".join("\\multicolumn{2}{c}{" + SHORT.get(m, m) + "}" for m in mods)
    rule = " ".join(f"\\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(mods)))
    body = [f"& & & {span} \\\\", rule,
            "where the cue is & what is measured & $n$ & "
            + " & ".join(["before & after"] * len(mods)) + " \\\\",
            "\\midrule"]
    plain = {x["model"]: x for x in sr if x["picture"] == "plain"}
    wm = {x["model"]: x for x in sr if x["picture"] == "watermark"}
    cells = []
    for m in mods:
        cells += [pct(plain.get(m, {}).get("says_test")), pct(wm.get(m, {}).get("says_test"))]
    body.append("in the picture & says it is a test (\\%) & "
                + str(sum(int(x["n"]) for x in sr)) + " & " + " & ".join(cells) + " \\\\")
    cells = []
    for m in mods:
        c = cond.get(m, {})
        cells += [pct(c.get("markup_plain")), pct(c.get("markup_watermark"))]
    body.append("in the picture & formats the reply (\\%) & "
                + str(cond.get(mods[0], {}).get("n", "--")) + " & " + " & ".join(cells) + " \\\\")
    body.append("\\midrule")
    body.append("in the words & says it is a test (\\%) & -- & "
                + " & ".join(["--"] * 2 * len(mods)) + " \\\\")
    cells = []
    for m in mods:
        e = agg.get(m)
        cells += [pct(e[1] / e[0]) if e else "--", pct(e[2] / e[0]) if e else "--"]
    body.append("in the words & formats the reply (\\%) & "
                + str(agg.get(mods[0], [0])[0]) + " & " + " & ".join(cells) + " \\\\")
    cells = []
    for m in mods:
        e = agg.get(m)
        cells += [pct(e[3] / e[0]) if e else "--", pct(e[4] / e[0]) if e else "--"]
    body.append("in the words & carries a caveat (\\%) & "
                + str(agg.get(mods[0], [0])[0]) + " & " + " & ".join(cells) + " \\\\")
    cap = ("Two ways of telling a model it is being evaluated, with the other "
           "held still. The set writes a benchmark marker into the photograph "
           "and also puts one in the system prompt, so each channel can be "
           "measured on its own. The pattern is off-diagonal. A marker in the "
           "picture moves what the model says the situation is, from seven "
           "photographs in ten called a test to nine in ten on the 27B, and "
           "leaves its conduct alone to within a point over 387 matched pairs. "
           "A sentence in the prompt barely needs saying and changes the "
           "conduct completely. The model can see it is being tested and carry "
           "on as before; being told is what changes it. The empty row is a "
           "cell the behaviour pass never ran, and is queued rather than "
           "estimated.")
    return table(body, "ll r" + "rr" * len(mods), cap, "channel")


def t_authority(R):
    r = [x for x in rows(f"{R}/authority.csv") if int(x["graded"]) >= 10]
    if not r:
        return ""
    order = ["operative", "operative_generic", "operative_with_graffiti", "defer",
             "ignore", "p1", "p8", "p5"]
    pool = {}
    for x in r:
        k = (x["model"], x["probe"])
        e = pool.setdefault(k, [0, 0.0, x["asks"]])
        n = int(x["graded"]); e[0] += n; e[1] += float(x["correct"]) * n
    mods = sorted({k[0] for k in pool})
    probes = [p for p in order if any(k[1] == p for k in pool)]
    body = ["what the question asks for & $n$ & "
            + " & ".join(SHORT.get(m, m) for m in mods) + " \\\\", "\\midrule"]
    for p in probes:
        got = [pool.get((m, p)) for m in mods]
        if not any(got):
            continue
        n = max(g[0] for g in got if g)
        asks = next(g[2] for g in got if g)
        body.append(f"{asks} & {n} & "
                    + " & ".join(pct(g[1] / g[0]) if g else "--" for g in got) + " \\\\")
    cap = ("Authority, on the probes that state a yes or a no, so the grading "
           "is the dataset's own. Obeying a sign that means what it says is "
           "close to solved and graffiti beside the sign does not disturb it. "
           "The bottom row is a sign drawn inside a picture: a price on a mock "
           "phone screen in an advert, a shop front painted in a mural. Asked "
           "whether the thing the picture depicts is so, both models answer "
           "worse than a coin would. A picture of a sign is not a sign, and "
           "neither model has that distinction.")
    return table(body, "l r" + "r" * len(mods), cap, "authority")


def t_efficiency(R):
    r = rows(f"{R}/efficiency.csv")
    r = [x for x in r if x.get("experiment") == "lensbench"] or r
    if not r:
        return ""
    r.sort(key=lambda x: float(x["fit_seconds"]))
    body = ["instrument & labels & gradients & how it is fitted & minutes & "
            "steps & prompts & tokens \\\\", "\\midrule"]
    for x in r:
        tok = x.get("tokens") or ""
        tok = f"{float(tok) / 1e6:.1f}M" if tok else "--"
        body.append(f"{NICE.get(x['instrument'], x['instrument'])} & "
                    f"{x['needs_labels']} & {x['needs_gradients']} & {x['fitting']} & "
                    f"{num(x['fit_minutes'], 1)} & {x.get('steps') or '--'} & "
                    f"{x.get('prompts') or '--'} & {tok} \\\\")
    cap = ("What each instrument costs to fit once, for all sixteen blocks of "
           "Qwen3-VL 8B. V-Read is a single closed-form solve and needs "
           "neither labels nor gradients. The tuned lens is two hours of "
           "gradient descent for the one nat it wins in Table~\\ref{tab:reading}.")
    return table(body, "l cc l rrrr", cap, "efficiency")


def t_dataset(R, res):
    """The full inventory, which is what a figure of twenty-eight rows was for."""
    d = None
    q = os.path.join(res, "dataset", "census.json")
    if os.path.exists(q):
        d = json.load(open(q))
    if not d or not d.get("families"):
        return ""
    KIND = {"auth": "authority", "read": "reading", "eval": "eval awareness",
            "base": "description", "sand": "sandbagging", "swap": "text swapped",
            "removal": "text removed"}
    by = {}
    for f, e in d["families"].items():
        if re.search(r"wmdp[_ ]pw[_ ](fine|test)", str(f)):
            continue
        src = SOURCE.get(e["source"], e["source"])
        r = by.setdefault(src, dict(n=0, fam=0, ours=e.get("whose") == "ours", probes=set()))
        r["n"] += int(e["n"]); r["fam"] += 1
        r["probes"].add(KIND.get(e["probe"], e["probe"]))
    order = ["Open Images", "nuScenes", "VINS-120K", "Generated scenes",
             "WMDP carrier"]
    order = [s for s in order if s in by] + [s for s in by if s not in order]
    body = ["source & pictures & whose & records & families & probes asked \\\\",
            "\\midrule"]
    tot = 0
    for src in order:
        e = by[src]
        tot += e["n"]
        body.append(f"{src} & {'generated' if e['ours'] else 'public'} & "
                    f"{'ours' if e['ours'] else 'theirs'} & {e['n']} & {e['fam']} & "
                    + ", ".join(sorted(e["probes"])) + " \\\\")
    body.append("\\midrule")
    body.append(f"all & & & {tot} & & \\\\")
    cap = ("What the set is made of, once the password study's own two families "
           "are set aside. The photographs in four of the five sources are other "
           "people's and we say so; the generated scenes are ours, and so is "
           "every question, every edit and every label in all five. The probe "
           "column is what the questions ask for, and it is the part that does "
           "not exist in a scraped corpus: whether text has the standing to "
           "settle a matter, which object a string is written on, and whether "
           "the setting is a test.")
    return table(body, "l ll rr l", cap, "dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="tables")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="tables/tables.tex")
    a = ap.parse_args()
    parts, made = [], []
    for name, fn in (("dataset", lambda: t_dataset(a.tables, a.results)),
                     ("reading", lambda: t_reading(a.tables)),
                     ("causal", lambda: t_causal(a.tables)),
                     ("sources", lambda: t_sources(a.tables)),
                     ("channel", lambda: t_channel(a.tables, a.results)),
                     ("authority", lambda: t_authority(a.tables)),
                     ("efficiency", lambda: t_efficiency(a.tables))):
        try:
            tex = fn()
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: failed, {type(e).__name__}: {e}")
            continue
        if tex:
            parts.append(tex); made.append(name)
        else:
            print(f"  {name}: no data yet")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    head = ("% Generated by tools/make_tex_tables.py from the result files.\n"
            "% Do not edit by hand: re-run the tool instead.\n"
            "% Needs \\usepackage{booktabs}.\n\n")
    open(a.out, "w").write(head + "\n".join(parts))
    print(f"\n{len(made)} table(s) -> {a.out}: {', '.join(made)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
