"""Pull real model outputs out of the result files, as an HTML fragment.

Margins between logits are the right thing to measure and the wrong thing to
read.  This assembles the same claims in the model's own words, taking the first
qualifying items rather than choosing among them.
"""
from __future__ import annotations

import glob
import html
import json
import os
import sys

import pandas as pd

RES = sys.argv[1] if len(sys.argv) > 1 else "/workspace/vlm/results"
TAG = sys.argv[2] if len(sys.argv) > 2 else "qwen2b"
OUT = sys.argv[3] if len(sys.argv) > 3 else f"{RES}/examples.html"
N = 4
parts: list[str] = []


def esc(x) -> str:
    return html.escape(str(x))


def card(title: str, sub: str, rows: list[tuple[str, str, str]]) -> str:
    """rows: (label, text, css class for the label)"""
    body = "".join(
        f'<div class="exrow"><div class="exlab {cls}">{esc(lab)}</div>'
        f'<div class="extext">{esc(txt)}</div></div>'
        for lab, txt, cls in rows)
    return (f'<div class="excard"><div class="exhead">{esc(title)}</div>'
            f'<div class="exsub">{esc(sub)}</div>{body}</div>')


# ---- erasure: the free-form description, before and after
p = f"{RES}/blindsight/{TAG}_raw.parquet"
if os.path.exists(p):
    df = pd.read_parquet(p)
    if "text" in df.columns:
        clean = df[(df.cond == "clean") & df.text.notna()].set_index("i")
        abl = df[(df.cond == "abl_all") & df.text.notna()].set_index("i")
        shared = [i for i in clean.index if i in abl.index]
        shown = 0
        for i in shared:
            c, a = clean.loc[i], abl.loc[i]
            if float(c.report_said) != 1.0 or float(a.report_said) != 0.0:
                continue
            parts.append(card(
                f"{c.concept}", "asked to list the objects in the photograph",
                [("before", str(c.text).strip(), "w"),
                 ("after erasing the concept", str(a.text).strip(), "k")]))
            shown += 1
            if shown >= N:
                break

# ---- forgery: the description is unchanged, the readout is not
for f in sorted(glob.glob(f"{RES}/forgery/*_plant_raw.parquet")):
    d = pd.read_parquet(f)
    hit = d[(d.text_same == 1) & (d.lens_rank_adv < 10)]
    for _, r in hit.head(N).iterrows():
        parts.append(card(
            f"planted: {r.target}",
            f"L-infinity budget {r.eps:g}/255; the readout moves from rank "
            f"{int(r.lens_rank_clean)} to rank {int(r.lens_rank_adv)}",
            [("clean image", str(r.clean_text).strip(), "w"),
             ("perturbed image", str(r.adv_text).strip(), "k")]))
    break

# ---- reality monitoring: what it says when the text asserts something false
p = f"{RES}/reality/{TAG}_behaviour.parquet"
if os.path.exists(p):
    d = pd.read_parquet(p)
    wrong = d[(d.really == 0) & (d.check_correct == 0)]
    if len(wrong):
        parts.append(card(
            "asserted but absent",
            f"the text claims the scene contains an object that is not in it; the "
            f"model confirms it is really visible on "
            f"{100 * (1 - d[d.really == 0].check_correct.mean()):.0f}% of such "
            f"photographs",
            [(f"{r.concept}", f"image {int(r.image_id)}: says it is really there",
              "w") for _, r in wrong.head(N).iterrows()]))

frag = "\n".join(parts) if parts else "<p>No qualitative examples available yet.</p>"
open(OUT, "w").write(frag)
print(f"wrote {OUT}  ({len(parts)} cards)")
json.dump({"n_cards": len(parts)}, open(OUT + ".json", "w"))
