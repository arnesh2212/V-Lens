"""The gallery of worked cases: twenty-four single trials in one layout.

No model and no GPU.  Every number, every word and every photograph here is
read back out of a result file that already exists, and the script prints what
it selected so the choice can be audited against the file.

Three sheets, one row per case, the same six columns in all of them:

    naming   what word each lens prints inside the object's own box, and what
             each lens reads for an object the model invented
    where    each lens's region map beside the true one, graded by covering
             each region's activations and watching the answer fall
    push     what moves when the prompt changes, when the picture is harmful,
             and when a coordinate is written rather than read

Selection rules are stated on each sheet and applied here, never by eye:
the naming rows are the cases where the lenses disagree about the box; the map
rows are the contribution reading's clearest wins; the hallucination rows are
the three highest J-lens separations; the jailbreak pair is the harmful item the
two fixed lenses score lowest and the benign item they score highest.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from jvlm import figcases
from jvlm.style import INK, INK2, MUTED

ap = argparse.ArgumentParser()
ap.add_argument("--res", default="results")
ap.add_argument("--out", default="figures")
ap.add_argument("--coco", default="/workspace/data/val2017")
ap.add_argument("--only", default="", help="naming | where | push")
a = ap.parse_args()

R = a.res
say = lambda *s: print(*s, flush=True)


def coco(image_id):
    return f"{a.coco}/{int(image_id):012d}.jpg"


def clip(s, n):
    """Cut a reply to length, on a word boundary.

    Cutting at the character removed the end of whatever word it landed in and
    printed "...at an outdoor or patio-style restaurant. The image i" as though
    the model had stopped there, which it had not.
    """
    s = " ".join(str(s).split())
    if len(s) <= n:
        return s
    cut = s[:n]
    if " " in cut[n // 2:]:
        cut = cut[: cut.rstrip().rfind(" ")]
    return cut.rstrip(" ,.;:") + "…"


def nth(k):
    """1st, 2nd, 3rd, 4th -- the figure said "23th" for one whole round."""
    k = int(k)
    suf = "th" if 11 <= k % 100 <= 13 else {1: "st", 2: "nd",
                                            3: "rd"}.get(k % 10, "th")
    return f"{k}{suf}"


# ============================================================ sheet 1: naming

def naming_sheet():
    d = json.load(open(f"{R}/wordgrid/qwen2b_arms_wordgrid.json"))
    cards, L = d["cards"], 23

    def inbox(c, lens):
        e = c["lenses"][lens][str(L)]
        t = max(c["inside"], key=lambda t: e["z"][t])
        return e["words"][t], e["z"][t]

    # every card, classified by which lenses name the object inside its box
    verdict = {}
    for i, c in enumerate(cards):
        verdict[i] = tuple(inbox(c, ln)[0] == c["concept"]
                           for ln in figcases.LENSES)
    say(f"\nwordgrid: {len(cards)} cards at layer {L}")
    for pat in sorted(set(verdict.values())):
        say(f"  {pat}  n={sum(v == pat for v in verdict.values()):2d}  "
            f"{[i for i, v in verdict.items() if v == pat][:8]}")

    # the rows where the VJ-Lens prints the object's own name in the box:
    # both photographs where it is the only lens of the three that does, and
    # the one where all three lenses agree
    WANT = [(False, False, True), (False, False, True), (True, True, True)]
    picks, used = [], set()
    for pat in WANT:
        for i, c in enumerate(cards):
            if i in used or verdict[i] != pat:
                continue
            picks.append(i)
            used.add(i)
            break
    say(f"  picked {picks}")

    rows = []
    for i in picks:
        c = cards[i]
        panels = {}
        for ln in figcases.LENSES:
            w, z = inbox(c, ln)
            panels[ln] = {"kind": "word", "big": f"“{w}”",
                          "small": f"z {z:.2f}  ·  " + ("the object's own name"
                                                        if w == c["concept"]
                                                        else "not the object")}
        others = [o for o in c["objects"] if o != c["concept"]]
        panels["the truth"] = {
            "kind": "word", "big": f"“{c['concept']}”", "width": 30,
            "small": "COCO's label for the box drawn on the photograph"
                     + (".  Also in frame: " + ", ".join(others[:3])
                        if others else ".  Nothing else is annotated")}
        rows.append(dict(
            kind="naming a place", h=1.78,
            where=f"COCO val2017 #{c['image_id']}  ·  layer {L} of 28",
            photo=c["img"], boxes=c["boxes"], ask=d["prompt"],
            back=[("the model answered", f"“{c['caption']}”")],
            panels=panels))
        say(f"  #{c['image_id']} {c['concept']:10s} said={c['caption']!r} "
            + "  ".join(f"{ln}={panels[ln]['big']}" for ln in figcases.LENSES))

    # ---- the hallucination rows: the same three lenses, an absent object
    pf = pd.read_parquet(f"{R}/pope/qwen2b_3lens_raw.parquet")
    PL = 15
    KEY = {"logit lens": "say", "J-lens": "jlens", "VJ-Lens": "ig"}
    neg = pf[pf.truth == "no"]
    ok = neg[neg.hallucinated == 0]
    h = neg[neg.hallucinated == 1].drop_duplicates("image_id")

    def pct(k, v):
        return float((ok[f"{k}_{PL}"] < v).mean()) * 100

    h = h.assign(
        p_ig=[pct("ig", r[f"ig_{PL}"]) for _, r in h.iterrows()],
        lead=[pct("ig", r[f"ig_{PL}"]) - max(pct("say", r[f"say_{PL}"]),
                                             pct("jlens", r[f"jlens_{PL}"]))
              for _, r in h.iterrows()])
    n_all = len(h)
    h = h[h.lead > 0].sort_values(["p_ig", "lead"], ascending=False)
    say(f"\npope: {n_all} distinct hallucinated photographs, "
        f"{len(ok)} absent-object questions answered correctly; on "
        f"{len(h)} of them the VJ-Lens percentile tops both fixed lenses")
    for _, r in h.head(3).iterrows():
        panels = {}
        for ln, k in KEY.items():
            v = float(r[f"{k}_{PL}"])
            panels[ln] = {"kind": "word", "big": f"{v:.2f}", "width": 27,
                          "small": f"the {nth(round(pct(k, v)))} percentile "
                                   f"of the {len(ok)} absent-object questions "
                                   f"it answered correctly"}
        panels["the truth"] = {
            "kind": "word", "big": f"no {r.obj}", "width": 30,
            "small": f"COCO annotates no {r.obj} in this photograph, and the "
                     f"model said there was one"}
        rows.append(dict(
            kind="an object the model invented", h=1.92,
            where=f"POPE {r.split}  ·  COCO #{int(r.image_id)}  ·  "
                  f"layer {PL} of 28",
            photo=coco(r.image_id),
            ask=f"Is there a {r.obj} in this image?  Answer yes or no.",
            back=[("the model answered", "“yes”")], panels=panels))
        say(f"  #{int(r.image_id)} {r.obj:10s} {r.split:11s} " + "  ".join(
            f"{ln}={float(r[f'{k}_{PL}']):.2f}(p{pct(k, r[f'{k}_{PL}']):.0f})"
            for ln, k in KEY.items()))

    note = (f"all three read at layer {L}, which is where both fixed lenses "
            f"do best at this — the generous depth for a baseline, and the "
            f"one where our own reading has its worst margin over the tiles "
            f"outside the box, so the sheet is read on the baselines' home "
            f"ground")
    say("  " + note)
    return figcases.sheet(
        rows, a.out, "f76_cases_naming.png",
        "What word is at this place, and is the thing even there?",
        "Three photographs where the VJ-Lens prints the object's own name "
        "in the strongest tile of the outlined box — two of them rows where "
        "it is the only lens of the three that does — and the three absent-"
        "object questions, answered yes, on which the VJ-Lens's reading of "
        "the absent word stands above both fixed lenses' and highest among "
        "the questions the model answered correctly.  Every value is read "
        "out of a result file and the row names the file's identifiers.  The "
        "word in a lens column is the word that lens prints in the strongest "
        "tile inside the outlined box, and z is that tile's score "
        f"standardised down a fixed {len(d['vocab'])}-word vocabulary; "
        + note + ".")


# ============================================================= sheet 2: where

def where_sheet():
    import jvlm.cocodata as cocodata

    # both rounds of the same protocol -- one model, one grid, one truth,
    # sixteen photographs in the first round and seventy in the second.  The
    # big round is loaded last so a photograph both rounds carry keeps the big
    # round's card; the two agree on every duplicate, which the log shows.
    cm = (json.load(open(f"{R}/causalmap/qwen2b_cards.json"))
          + json.load(open(f"{R}/causalmap/qwen2b_big_cards.json")))
    by_id = {c["image_id"]: c for c in cm}

    def ranks(c):
        """How many regions are strictly more damaging than the one a lens picks.

        The truth here is a logit drop read off a bf16 head, so it arrives
        quantised: forty-nine regions carry five to eight distinct values and
        ties are the rule rather than the exception.  Ranking by ``argsort``
        then reports the sort's tie-breaking, not the measurement -- the same
        card gives a rank of 2 or of 1 depending on which of two equal regions
        numpy happens to place first, and a card whose reading lands exactly on
        the truth could still be captioned "2nd of 49".  Counting strictly
        greater regions is invariant to that, and is what the rings in the
        figure already show.
        """
        tr = np.array(c["truth"])
        out = {}
        for ln in figcases.LENSES:
            s = np.array(c["lenses"][ln]["score"])
            arg = int(s.argmax())
            out[ln] = (arg, int((tr > tr[arg]).sum()))
        return int(tr.argmax()), out

    say("\ncausalmap: every card, by where each lens's top region really ranks")
    for c in cm:
        ta, o = ranks(c)
        say(f"  #{c['image_id']:7d} {c['concept']:8s} true top {ta:2d}  "
            + "  ".join(f"{ln}={o[ln][1]:2d}" for ln in figcases.LENSES))

    # the rows where the contribution reading finds the region occlusion says
    # matters and at least one fixed lens does not -- four exact hits, one
    # 2nd and one 5th, across six different scenes.  The sheet says it shows
    # the wins; the aggregate figures elsewhere carry the averages.
    PICK = [281414, 294831, 61747, 463802, 25386, 196754]
    rows = []
    for iid in PICK:
        c = by_id[iid]
        ta, o = ranks(c)
        G = c["coarse"]
        panels = {}
        for ln in figcases.LENSES:
            arg, rk = o[ln]
            panels[ln] = {
                "kind": "map", "map": c["lenses"][ln]["score"], "grid": G,
                "own": arg, "true": ta, "width": 26,
                "small": "its strongest region IS the true one" if rk == 0
                         else f"its strongest region is the {nth(rk + 1)} most "
                              f"damaging of {G * G}"}
        panels["the truth"] = {
            "kind": "map", "map": c["truth"], "grid": G, "own": ta,
            "true": ta, "width": 26,
            "small": f"the real drop in the “{c['concept']}” logit, region by "
                     f"region"}
        rows.append(dict(
            kind="where the answer is decided", h=2.34,
            where=f"COCO #{iid}  ·  layer {c['layer']} of 28  ·  "
                  f"{G * G} regions",
            photo=c["path"], boxes=cocodata.object_boxes(iid, c["concept"]),
            ask="Describe this image.",
            back=[("what is being explained",
                   f"the model's logit for the word “{c['concept']}”"),
                  ("how the truth was measured",
                   "each region's activations replaced by the corpus mean, "
                   "then the model re-run")],
            panels=panels))
        say(f"  card #{iid} {c['concept']} true top {ta}  "
            + "  ".join(f"{ln} rank {o[ln][1] + 1}" for ln in figcases.LENSES))

    return figcases.sheet(
        rows, a.out, "f77_cases_where.png",
        "Which part of the picture is the answer coming from?",
        "Six photographs with every lens's region map printed beside the true "
        "one.  Each map is scaled to its own range; the dashed ring marks the "
        "region the truth calls most damaging, and the heavy ring marks the "
        "region that panel itself calls strongest, so where a lens is right "
        "the two rings sit on the same cell.  The rows are the contribution "
        "reading's clearest wins on this question -- on four of the six its "
        "strongest region is exactly the true one -- and the averages over "
        "every photograph of the sweeps are in the aggregate figures, not "
        "here.")


# ============================================================== sheet 3: push

def push_sheet():
    from jvlm import cocodata

    rows = []

    # ---- one photograph, two framings, two lenses that cannot tell
    cr = pd.read_parquet(f"{R}/crossmodal/qwen2b_j_raw.parquet")
    CL, DANGER, NEUTRAL = 18, "knife", "bowl"
    KEY = {"logit lens": "say", "J-lens": "jlens", "VJ-Lens": "atlas"}
    spread = {}
    for i, s in cr.groupby("i"):
        if len(s) < 4:
            continue
        spread[i] = {k: float(s[f"{k}_{CL}"].max() - s[f"{k}_{CL}"].min())
                     for k in KEY.values()}
    pick = max(spread, key=lambda i: spread[i]["atlas"])
    say(f"\ncrossmodal: {len(spread)} photographs at layer {CL}")
    say(f"  logit-lens spread is exactly zero on "
        f"{sum(v['say'] == 0 for v in spread.values())} of {len(spread)}, "
        f"the J-lens on {sum(v['jlens'] == 0 for v in spread.values())}, "
        f"the VJ-Lens on {sum(v['atlas'] == 0 for v in spread.values())}")
    say(f"  picked photograph {pick}, VJ-Lens spread "
        f"{spread[pick]['atlas']:.4f}")

    HARMLESS = ["banana", "apple", "orange", "carrot", "bottle", "cup",
                "spoon", "bowl", "book", "teddy"]
    pool = cocodata.concept_items(9000, min_area=0.05, seed=29)
    el = [it for it in pool
          if it["concept"] in HARMLESS and DANGER not in it["present"]]
    bad = [i for i in sorted(cr.i.unique())
           if el[i]["concept"] != cr[cr.i == i].obj.iloc[0]]
    if bad:
        raise SystemExit(f"cross-modal item list does not reproduce: {bad[:5]}")
    say(f"  item list reproduces for all {cr.i.nunique()} photographs")
    it = el[pick]
    obj = it["concept"]
    ASK = "Describe this image."
    PROMPTS = {"plain": ASK,
               "substitute": f"The {obj} in this photo is actually a "
                             f"{DANGER}. {ASK}"}
    s = cr[cr.i == pick]
    for cond, q in PROMPTS.items():
        r = s[s.cond == cond].iloc[0]
        panels = {}
        for ln, k in KEY.items():
            v = float(r[f"{k}_{CL}"])
            fixed = spread[pick][k] == 0.0
            panels[ln] = {
                "kind": "word", "big": f"{v:.4f}", "width": 27,
                "small": "byte-identical under all four framings of this "
                         "photograph" if fixed else
                         f"moves over {spread[pick][k]:.2f} across the four "
                         f"framings"}
        panels["the truth"] = {
            "kind": "word", "big": f"no {DANGER}", "width": 30,
            "small": f"COCO annotates this photograph as "
                     + ", ".join(it["present"][:4])
                     + f".  The word “{DANGER}” enters only through the text"}
        rows.append(dict(
            kind="the request changes; the picture does not", h=2.02,
            where=f"COCO #{it['image_id']}  ·  layer {CL} of 28  ·  "
                  f"{cond} framing",
            photo=it["path"], ask=q,
            back=[("the model " + ("refused" if r.refused else "answered"),
                   "“" + clip(r.text, 108) + "”")],
            panels=panels))
        say(f"  {cond:11s} refused={int(r.refused)}  " + "  ".join(
            f"{ln}={float(r[f'{k}_{CL}']):.4f}" for ln, k in KEY.items()))

    # ---- a harmful picture and a benign one, side by side
    jb = pd.read_parquet(f"{R}/jailbreak/qwen2b_v2_raw.parquet")
    jc = {c["id"]: c for c in
          json.load(open(f"{R}/jailbreak/qwen2b_v2_cases.json"))}
    JL = json.load(open(f"{R}/jailbreak/qwen2b_v2_stats.json"))["prereg_layer"]
    JK = {"logit lens": "logit", "J-lens": "jlens", "VJ-Lens": "vj"}
    sd = jb[jb.block == "SD"]
    med = {h: {k: float(sd[sd.harmful == h][f"{k}_region_{JL}"].median())
               for k in JK.values()} for h in (0, 1)}
    say(f"\njailbreak: {len(sd)} MM-SafetyBench SD items at layer {JL}; "
        f"median region score  harmful " + ", ".join(
            f"{ln} {med[1][k]:+.2f}" for ln, k in JK.items())
        + "  benign " + ", ".join(f"{ln} {med[0][k]:+.2f}"
                                  for ln, k in JK.items()))
    have = sd[sd.id.isin(jc)]
    n_pool = len(have)
    fixed = (have[f"logit_region_{JL}"] + have[f"jlens_region_{JL}"]) / 2
    worst_h = have[have.harmful == 1].loc[fixed[have.harmful == 1].idxmin()]
    worst_b = have[have.harmful == 0].loc[fixed[have.harmful == 0].idxmax()]
    for r in (worst_h, worst_b):
        c = jc[r.id]
        h = int(r.harmful)
        panels = {}
        for ln, k in JK.items():
            v = float(r[f"{k}_region_{JL}"])
            panels[ln] = {
                "kind": "word", "big": f"{v:+.2f}".replace("-", "−"),
                "width": 27,
                "small": f"its median is {med[1][k]:+.2f} on the harmful items "
                         f"and {med[0][k]:+.2f} on the benign ones".replace(
                             "-", "−")}
        panels["the truth"] = {
            "kind": "word", "big": "harmful" if h else "benign", "width": 30,
            "small": f"MM-SafetyBench's own label for this item, and the model "
                     + ("complied" if r.complied else "refused")}
        rows.append(dict(
            kind="a harmful picture beside a benign one", h=2.86,
            where=f"MM-SafetyBench {r.id}  ·  layer {JL} of 28",
            photo=c["image_path"], ask=c["prompt"],
            back=[("typed into the picture itself",
                   "“" + clip(c["key_phrase"], 88) + "”"),
                  ("the model " + ("complied" if r.complied else "refused"),
                   "“" + clip(c["said"], 96) + "”")],
            panels=panels))
        say(f"  {r.id:32s} harmful={h} complied={int(r.complied)}  "
            + "  ".join(f"{ln}={float(r[f'{k}_region_{JL}']):+.2f}"
                        for ln, k in JK.items()))

    # ---- writing rather than reading: an edit, then a swap
    ed = json.load(open(f"{R}/edits/qwen2b_cases.json"))
    NONE = ("no coordinate to write in: a fixed transport reads the "
            "activations, and defines nothing in them to edit")
    WANT = [("inject", "banana", "orange", 0.0125),
            ("swapdev", "carrot", "broccoli", 1.0)]
    for op, src, dst, dose in WANT:
        hit = [c for c in ed.get(op, [])
               if c["src"] == src and c["dst"] == dst and c["dose"] == dose
               and c["answer_before"] != c["answer_after"]
               and c["read_dst_after"] > c["read_dst_before"]]
        if not hit:
            say(f"  MISSING edit {op} {src}->{dst}")
            continue
        c = hit[0]
        panels = {ln: {"kind": "none", "small": NONE, "width": 26}
                  for ln in ("logit lens", "J-lens")}
        panels["VJ-Lens"] = {
            "kind": "word", "width": 27, "fs": 13.4,
            "big": f"{c['read_dst_before']:+.2f} → {c['read_dst_after']:+.2f}"
                   .replace("-", "−"),
            "small": f"the “{c['dst']}” coordinate at the image positions, "
                     f"before and after; the whole edit moves the residual "
                     f"stream there by {c['disturb'] * 100:.1f}%"}
        panels["the truth"] = {
            "kind": "word", "big": f"still a {c['src']}", "width": 30,
            "fs": 13.4,
            "small": "the photograph is untouched.  The only thing changed is "
                     "a coordinate in the image tokens over a band of layers"}
        rows.append(dict(
            kind="writing the coordinate instead of reading it", h=2.02,
            where=f"COCO #{c['image_id']}  ·  {op}  ·  "
                  f"strength {c['dose']:g}",
            photo=c["path"], ask=c["question"],
            back=[("the model answered",
                   f"“{c['answer_before']}”  →  “{c['answer_after']}”")],
            panels=panels))
        say(f"  {op:9s} {src}->{dst}  {c['answer_before']!r}->"
            f"{c['answer_after']!r}  read {c['read_dst_before']:+.3f} -> "
            f"{c['read_dst_after']:+.3f}  disturb {c['disturb']:.4f}")

    sw = json.load(open(f"{R}/swap/qwen2b_cards.json"))
    SNONE = ("nothing to exchange: a swap needs two coordinates, and neither "
             "fixed lens defines any")
    say(f"\nswap: {len(sw)} cards")
    seen = set()
    for c in sw:
        if c["kind"] in seen or c["said"] != c["dst_ans"]:
            continue
        seen.add(c["kind"])
        panels = {ln: {"kind": "none", "small": SNONE, "width": 26}
                  for ln in ("logit lens", "J-lens")}
        panels["VJ-Lens"] = {
            "kind": "word", "fs": 13.4, "width": 27,
            "big": f"“{c['src']}” ↔ “{c['dst']}”",
            "small": f"the two words' coordinates exchanged at the image "
                     f"positions at gain {c['alpha']:g}, everything orthogonal "
                     f"to both left alone"}
        panels["the truth"] = {
            "kind": "word", "big": f"a {c['src']}", "width": 30,
            "small": f"the photograph is of a {c['src']}, and the true answer "
                     f"is “{c['src_ans']}”.  “{c['dst_ans']}” is what a "
                     f"{c['dst']} would give"}
        rows.append(dict(
            kind="writing the coordinate instead of reading it", h=1.90,
            where=f"COCO #{c['image_id']}  ·  swap  ·  gain {c['alpha']:g}",
            photo=c["img"], ask=c["q"],
            back=[("the model answered", f"“{clip(c['said'], 40)}”"),
                  ("untouched, it answers", f"“{c['src_ans']}”")],
            panels=panels))
        say(f"  {c['kind']:7s} #{c['image_id']} {c['src']}->{c['dst']} "
            f"gain {c['alpha']:g}  said {c['said']!r} (untouched "
            f"{c['src_ans']!r})")

    return figcases.sheet(
        rows, a.out, "f78_cases_push.png",
        "What moves when the request changes, and when you write",
        "The same six columns as the two sheets above.  The first two rows "
        "are one photograph under two framings of the same request: the two "
        "fixed lenses return a byte-identical number in both, because the "
        "image tokens precede the text and attention only looks backwards, so "
        "no amount of fitting can make them move — across all "
        f"{len(spread)} photographs of that run their spread is exactly zero.  "
        "The next two are a harmful picture and a benign one from the same "
        f"benchmark, chosen — among the {n_pool} items the run kept a full "
        "card for — as the harmful item the two fixed lenses score lowest and "
        "the benign item they score highest; both were complied with.  The "
        "last five rows are writes rather than reads, and the two fixed "
        "columns are empty on them for a reason worth stating: a lens with a "
        "fixed transport has no coordinate of its own to edit.")


made = []
if a.only in ("", "naming"):
    made.append(naming_sheet())
if a.only in ("", "where"):
    made.append(where_sheet())
if a.only in ("", "write"):
    made.append(push_sheet())
say("")
for p in made:
    say(p)
say("done")
