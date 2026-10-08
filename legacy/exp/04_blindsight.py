"""Induced blindsight: is workspace membership necessary for report and
unnecessary for discrimination?

Each COCO item gives both halves of the dissociation:

  necessity   -- the concept IS in the image; ablate its J-lens direction at the
                 image-token positions and ask whether the model can still name
                 it (report) and still pick it out of two options (2AFC).
  sufficiency -- the distractor is NOT in the image; add its J-lens direction and
                 ask whether the model now names it, and whether 2AFC follows.

The forced-choice answer is a letter, never the concept word, so an intervention
on the concept direction cannot trivially destroy both measures at once.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time

sys.path.insert(0, "code")
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch

import jvlm
from jvlm import cocodata, corpus, readout, tasks

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--lens", required=True)
ap.add_argument("--n", type=int, default=120)
ap.add_argument("--max-pixels", type=int, default=200704)
ap.add_argument("--lo", type=float, default=0.10, help="layer band start, as depth")
ap.add_argument("--hi", type=float, default=1.0, help="layer band end, as depth")
ap.add_argument("--out", default="results/blindsight")
ap.add_argument("--min-area", type=float, default=0.08)
ap.add_argument("--tag", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.repo.split("/")[-1]

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
lens = (
    jvlm.lens_from_checkpoint(a.lens)
    if a.lens.endswith(".ckpt")
    else __import__("jlens").lens.JacobianLens.load(a.lens)
)
sc = tasks.Scorer(m)
L = [l for l in lens.source_layers if a.lo <= l / (m.n_layers - 1) <= a.hi]


def band(lo: float, hi: float) -> list[int]:
    return [l for l in L if lo <= l / (m.n_layers - 1) <= hi]


# Layer bands.  "e80"/"e90" stop short of the output: at the last layers J_l is
# close to the identity, so erasing the lens direction there is barely
# distinguishable from subtracting the unembedding row itself.  If report still
# collapses when the final layers are spared, the effect is not output
# suppression.
BANDS = {"all": L, "lo": band(0.10, 0.50), "mid": band(0.25, 0.60),
         "hi": band(0.50, 0.95), "e60": band(0.10, 0.60),
         "e70": band(0.10, 0.70), "e80": band(0.10, 0.80),
         "e90": band(0.10, 0.90)}
empty = [k for k, v in BANDS.items() if not v]
if empty:
    raise ValueError(f"empty layer bands {empty}: an intervention over no layers "
                     f"is a silent no-op recorded as an ablation")
print(m, lens, "\nbands: " + "  ".join(
    f"{k}={v[0]}-{v[-1]}({len(v)})" for k, v in BANDS.items()), flush=True)

items = cocodata.concept_items(a.n, min_area=a.min_area, seed=11, hard=True,
                               tokenizer=m.tokenizer)
rng = random.Random(3)
_cache: dict[tuple[str, int], torch.Tensor] = {}


def spec(concept: str) -> dict[int, torch.Tensor]:
    """J-lens directions for every surface form of ``concept``.

    A rank-1 ablation of just ``" cow"`` leaves ``" cows"``/``"Cow"`` intact and
    barely dents the model; erasing the span of all surface forms is what
    actually removes the concept from the workspace.
    """
    out = {}
    for l in L:
        key = (concept, l)
        if key not in _cache:
            ids = cocodata.concept_token_ids(m.tokenizer, concept)
            _cache[key] = readout.lens_vectors(m, lens, ids, l)
        out[l] = _cache[key]
    return out


def unemb_spec(concept: str) -> dict[int, torch.Tensor]:
    """The logit-lens control: the raw unembedding rows, no Jacobian transport.

    Ablating these is what you would do without a lens at all.  The comparison
    says whether the averaged Jacobian is doing real work or whether the
    dissociation would fall out of the unembedding geometry alone.
    """
    W = m._lm_head.weight[cocodata.concept_token_ids(m.tokenizer, concept)]
    W = W.detach().float()
    W = W / W.norm(dim=-1, keepdim=True)
    return {l: W for l in L}


def rand_spec(seed: int, k: int) -> dict[int, torch.Tensor]:
    """Matched-rank random subspace: the norm/dimension control for ablation."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for l in L:
        V = torch.randn(k, m.d_model, generator=g)
        out[l] = (V / V.norm(dim=-1, keepdim=True)).to(m.device)
    return out


# name, mode, direction, alpha, channel, exclude_last, band, generate?
#   band "all" = every fitted layer, "mid" = 0.25-0.60 depth (well upstream of
#   the output, so a collapse there cannot be final-position suppression).
#   exclude_last leaves the response cue alone for the same reason.
CONDITIONS = [
    ("clean",         None,     None,   0.0, "all", 0,  "all", True),
    ("wrong_image",   None,     None,   0.0, "all", 0,  "all", True),
    # necessity, full protocol (Arditi: every layer, every position)
    ("abl_all",       "ablate", "c",    1.0, "all", 0,  "all", True),
    # how deep does the erasure have to reach?
    ("abl_e90",       "ablate", "c",    1.0, "all", 0,  "e90", False),
    ("abl_e80",       "ablate", "c",    1.0, "all", 0,  "e80", True),
    ("abl_e60",       "ablate", "c",    1.0, "all", 0,  "e60", False),
    ("abl_hi",        "ablate", "c",    1.0, "all", 0,  "hi",  False),
    ("abl_mid",       "ablate", "c",    1.0, "all", 0,  "mid", False),
    # which channel carries it?
    ("abl_img",       "ablate", "c",    1.0, "img", 0,  "all", False),
    # dose-response
    ("abl_all_a0.25", "ablate", "c",   0.25, "all", 0,  "all", False),
    ("abl_all_a0.5",  "ablate", "c",    0.5, "all", 0,  "all", False),
    ("abl_all_a0.75", "ablate", "c",   0.75, "all", 0,  "all", False),
    ("abl_all_a1.5",  "ablate", "c",    1.5, "all", 0,  "all", False),
    # controls
    ("abl_rand",      "ablate", "rand", 1.0, "all", 0,  "all", True),
    ("abl_other",     "ablate", "o",    1.0, "all", 0,  "all", True),
    ("abl_unemb",     "ablate", "u",    1.0, "all", 0,  "all", True),
]

#: Conditions that also get the open-vocabulary cloze, the presence questions,
#: and a forced choice whose options exclude the concept being written.
#: Injecting v_d makes d the wrong answer in a c-versus-d choice, so that
#: comparison cannot separate "hallucinates in open report" from "changes its
#: decision"; a c-versus-other choice can.  The depth and dose sweeps skip these
#: -- they only need the two channels, and the extras dominate the runtime.
EXTENDED = {"clean", "wrong_image", "abl_all", "abl_rand", "abl_unemb"}

rows = []
t0 = time.time()
for k, it in enumerate(items):
    c, d = it["concept"], it["distractor"]
    other = rng.choice([x for x in cocodata.SINGLE_WORD
                        if x not in it["present"] and x not in (c, d)])
    im = corpus.load_image(it["path"])
    # A different photo, same prompt: if the letter channel is genuinely reading
    # the picture, its accuracy must fall apart here.  Without this control a
    # preserved forced choice could just be a prior over the two option words --
    # so the substitute photograph must contain neither option, or the control
    # is answering the question by accident.
    foil = next((x for j in range(1, len(items))
                 for x in [items[(k + j) % len(items)]]
                 if c not in x["present"] and d not in x["present"]), items[k - 1])
    foil_im = corpus.load_image(foil["path"])

    for name, mode, which, alpha, channel, excl, bandname, do_gen in CONDITIONS:
        shown = foil_im if name == "wrong_image" else im
        if mode is None:
            iv = readout.NoIntervention()
        else:
            nvar = len(cocodata.concept_token_ids(m.tokenizer, c))
            dirs = {"c": spec(c), "d": spec(d), "o": spec(other),
                    "u": unemb_spec(c), "rand": rand_spec(k, nvar)}[which]
            iv = readout.Intervention(m, dirs, channel=channel, mode=mode,
                                      alpha=alpha, exclude_last=excl,
                                      layers=BANDS[bandname])

        def multi(prompts, image=shown, _iv=iv):
            return sc.multi_last_logits(_iv.set_batch(m.chat_batch(prompts, [image])))

        def make(prompt, image=shown, _iv=iv):
            return _iv.set_batch(m.chat(prompt, [image]))

        with iv:
            r = tasks.run_two_channels(sc, c, d, multi)
            if name in EXTENDED:
                r.update(tasks.run_cloze_presence(sc, c, d, multi))
                ctl = tasks.run_two_channels(sc, c, other, multi)
                r["ctl_word_correct"] = ctl["word_correct"]
                r["ctl_letter_correct"] = ctl["letter_correct"]
                r["ctl_letter_margin"] = ctl["letter_margin"]
            if do_gen:
                r.update(tasks.run_free_report(sc, c, d, make))
        rows.append(dict(i=k, image_id=it["image_id"], concept=c, distractor=d,
                         other=other, area=it["area"], cond=name,
                         mode=mode or "none", alpha=alpha, channel=channel,
                         band=bandname, exclude_last=excl,
                         energy=float(np.mean(iv.energy)) if getattr(iv, "energy", None)
                         else 0.0, **r))

    if k % 10 == 0:
        el = time.time() - t0
        print(f"  {k}/{len(items)}  {el:.0f}s  "
              f"eta {el/(k+1)*(len(items)-k-1)/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(f"{a.out}/{tag}_raw.parquet")

df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/{tag}_raw.parquet")

agg = df.groupby("cond").agg(
    n=("i", "count"),
    word_margin=("word_margin", "mean"),
    word_correct=("word_correct", "mean"),
    letter_margin=("letter_margin", "mean"),
    letter_correct=("letter_correct", "mean"),
    presence=("presence_margin", "mean"),
    report_said=("report_said", "mean"),
    inject_said=("inject_said", "mean"),
).reindex([x[0] for x in CONDITIONS])
agg.to_csv(f"{a.out}/{tag}_summary.csv")
print("\n", agg.to_string())
json.dump({"repo": a.repo, "lens": a.lens, "n": len(items),
           "layers": [L[0], L[-1]], "minutes": round((time.time() - t0) / 60, 1)},
          open(f"{a.out}/{tag}_meta.json", "w"), indent=2)
