"""Induced anomia: it can pick the object out of two and cannot name it.

Anomic aphasia is the clinical picture where a patient recognises an object,
can say what it is for, can pick it from a line-up -- and cannot retrieve its
name, so they talk around it.  The production-channel thesis predicts exactly
that picture from a rank-k projection: erase a concept's directions and naming
should fail while recognition survives.

The forced-choice experiments already show the numbers.  This shows the words.
Four prompts on every photograph, under three interventions, with the model's
own text kept verbatim:

    free      "Describe this image in one sentence."
    name      "What is the main object in this photograph?  One word."
    meta      "Name the main object.  If you cannot name it, say why."
    choice    the two-alternative letter question, as the control that the
              model still knows perfectly well what it is looking at

and the readouts are: did it say the word, did it produce a circumlocution --
a superordinate or a near neighbour instead of the name -- and did it report
difficulty.  The random arm removes the same number of directions somewhere
else, so "it got damaged" is a hypothesis with a control attached.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time

sys.path.insert(0, "code")
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch

import jvlm
from jvlm import cocodata, corpus, lexicon, readout, stats as st, tasks

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--lens", required=True)
ap.add_argument("--n", type=int, default=140)
ap.add_argument("--max-new", type=int, default=40)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--out", default="results/anomia")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lens = (jvlm.lens_from_checkpoint(a.lens) if a.lens.endswith(".ckpt")
        else __import__("jlens").lens.JacobianLens.load(a.lens))
L = [l for l in lens.source_layers]
sc = tasks.Scorer(m)
print(m, f"\nerasing over layers {L[0]}..{L[-1]}", flush=True)

FREE = "Describe this image in one sentence."
NAME = "What is the main object in this photograph? Answer with one word."
META = ("Name the main object in this photograph. If you cannot name it, say "
        "why not.")
SUPER = {  # what a talker-around reaches for when the name will not come
    "cow": ["animal", "cattle", "bovine", "livestock", "farm"],
    "dog": ["animal", "pet", "canine", "puppy"],
    "cat": ["animal", "pet", "feline", "kitten"],
    "horse": ["animal", "equine", "pony", "mare", "stallion"],
    "sheep": ["animal", "lamb", "livestock", "flock"],
    "bird": ["animal", "creature", "wings", "feather"],
    "elephant": ["animal", "mammal", "trunk", "tusk"],
    "bear": ["animal", "mammal", "creature"],
    "zebra": ["animal", "stripes", "striped"],
    "giraffe": ["animal", "mammal", "tall", "neck"],
    "car": ["vehicle", "automobile", "auto"],
    "bus": ["vehicle", "transit", "coach"],
    "truck": ["vehicle", "lorry"],
    "train": ["vehicle", "rail", "railway", "locomotive"],
    "boat": ["vessel", "craft", "watercraft", "ship"],
    "bicycle": ["vehicle", "bike", "cycle", "two-wheeler"],
    "motorcycle": ["vehicle", "bike", "motorbike"],
    "airplane": ["aircraft", "plane", "jet", "vehicle"],
    "pizza": ["food", "dish", "meal", "slice"],
    "sandwich": ["food", "dish", "meal", "snack"],
    "banana": ["fruit", "food", "produce"],
    "apple": ["fruit", "food", "produce"],
    "orange": ["fruit", "food", "citrus", "produce"],
    "chair": ["furniture", "seat", "seating"],
    "couch": ["furniture", "seat", "sofa", "seating"],
    "bed": ["furniture", "mattress", "bedding"],
    "laptop": ["computer", "device", "machine", "screen"],
    "clock": ["timepiece", "device", "time"],
    "umbrella": ["object", "canopy", "parasol", "shade"],
    "kite": ["object", "toy", "flying"],
    "book": ["object", "publication", "reading"],
    "toilet": ["fixture", "bathroom", "porcelain"],
    "vase": ["container", "vessel", "pottery", "ornament"],
    "bottle": ["container", "vessel"],
    "cup": ["container", "vessel", "mug", "drink"],
    "bowl": ["container", "vessel", "dish"],
    "spoon": ["utensil", "cutlery"],
    "fork": ["utensil", "cutlery"],
    "knife": ["utensil", "cutlery", "blade"],
}
TROUBLE = re.compile(
    r"\b(cannot|can't|cant|unable|not able|difficult|hard to|struggl|"
    r"not sure|unsure|don't know|do not know|unclear|can not)\b", re.I)


def circumlocution(text: str, concept: str) -> float:
    """A superordinate or near neighbour, with the name itself absent."""
    if sc.said(text, concept):
        return 0.0
    pool = list(SUPER.get(concept, ["object", "thing", "item"]))
    pool += [w for w in (lexicon.SYN.get(concept) or []) if w != concept]
    pat = r"\b(" + "|".join(re.escape(w) for w in pool) + r")"
    return float(bool(re.search(pat, text, flags=re.I)))


items = cocodata.concept_items(a.n, min_area=0.12, seed=77, hard=True,
                               tokenizer=m.tokenizer)
rows, cards = [], []
t0 = time.time()
for k, it in enumerate(items):
    c, d = it["concept"], it["distractor"]
    ids = cocodata.concept_token_ids(m.tokenizer, c)
    im = corpus.load_image(it["path"])
    spec = {l: readout.lens_vectors(m, lens, ids, l) for l in L}
    g = torch.Generator().manual_seed(500 + k)
    rspec = {}
    for l in L:
        q = torch.linalg.qr(torch.randn(m.d_model, len(ids), generator=g))[0]
        rspec[l] = q.T.to(m.device).to(lens.jacobians[l].dtype)

    texts = {}
    for arm, sp in (("clean", None), ("erased", spec), ("random", rspec)):
        iv = (readout.NoIntervention() if sp is None else
              readout.Intervention(m, sp, channel="all", mode="ablate",
                                   alpha=1.0))
        with iv:
            out = {}
            for tag_, q in (("free", FREE), ("name", NAME), ("meta", META)):
                out[tag_] = sc.generate(iv.set_batch(m.chat(q, [im])),
                                        max_new_tokens=a.max_new).strip()

            def multi(prompts, _iv=iv):
                return sc.multi_last_logits(
                    _iv.set_batch(m.chat_batch(prompts, [im])))

            ch = tasks.run_two_channels(sc, c, d, multi)
        texts[arm] = out
        rec = dict(i=k, image_id=it["image_id"], concept=c, distractor=d,
                   arm=arm, cond=arm, rank=len(ids),
                   letter_correct=ch["letter_correct"],
                   word_correct=ch["word_correct"],
                   energy=float(np.mean(iv.energy))
                   if getattr(iv, "energy", None) else 0.0)
        for tag_, txt in out.items():
            rec[f"{tag_}_said"] = float(sc.said(txt, c))
            rec[f"{tag_}_circum"] = circumlocution(txt, c)
            rec[f"{tag_}_trouble"] = float(bool(TROUBLE.search(txt)))
            rec[f"{tag_}_text"] = txt[:180]
        rows.append(rec)

    if (texts["clean"]["name"] and sc.said(texts["clean"]["name"], c)
            and not sc.said(texts["erased"]["name"], c)):
        cards.append({"concept": c, "image_id": int(it["image_id"]),
                      "path": it["path"], **{f"{a_}_{t_}": texts[a_][t_]
                                             for a_ in texts for t_ in
                                             texts[a_]}})
    if k % 15 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m  "
              f"{len(cards)} anomic cases", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")
json.dump(cards, open(f"{a.out}/{tag}_cases.json", "w"), indent=1)

cols = [c for c in df.columns if c.endswith(("_said", "_circum", "_trouble"))]
agg = df.groupby("arm")[cols + ["letter_correct", "word_correct", "energy"]] \
        .mean().reindex(["clean", "erased", "random"])
agg["n"] = df.groupby("arm").size().reindex(["clean", "erased", "random"])
agg.to_csv(f"{a.out}/{tag}_summary.csv")
print("\n", agg.round(3).T.to_string())

out = {"repo": a.repo, "n": int(df.i.nunique()),
       "anomic_cases": len(cards), "summary": st.jsonable(agg.round(4).to_dict()),
       "minutes": round((time.time() - t0) / 60, 1)}
for v in ("name_said", "free_said", "letter_correct", "name_circum",
          "meta_trouble"):
    out[f"erased_vs_clean_{v}"] = st.mcnemar(df, "clean", "erased", v)
    out[f"random_vs_clean_{v}"] = st.mcnemar(df, "clean", "random", v)
    out[f"erased_vs_random_{v}"] = st.mcnemar(df, "random", "erased", v)
print("\nnaming, erased vs clean:", out["erased_vs_clean_name_said"])
print("choice, erased vs clean:", out["erased_vs_clean_letter_correct"])
print("circumlocution, erased vs random:", out["erased_vs_random_name_circum"])
json.dump(out, open(f"{a.out}/{tag}_stats.json", "w"), indent=2, default=float)
