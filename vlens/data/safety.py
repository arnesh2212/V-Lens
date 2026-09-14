# SPDX-License-Identifier: Apache-2.0
r"""Published VLM-safety benchmarks, loaded into one record shape.

Every hand-rolled jailbreak is a jailbreak the author already knew the answer
to.  This module replaces them with datasets from the literature, downloaded
once into ``/workspace/vlm/data/safety`` and served as a single record type::

    {id, image_path, question, harmful, scenario, source}

plus whatever extra fields a source carries (``variant``, ``keywords``,
``text_only``), which experiments may use and may ignore.

What is actually here, and what is not
--------------------------------------

Downloaded and working:

``mm_safetybench``
    MM-SafetyBench (Liu et al. 2024), ``PKU-Alignment/MM-SafetyBench``.
    1680 questions x 4 image variants: ``SD`` (a Stable-Diffusion picture of
    the forbidden concept), ``TYPO`` (the key phrase typed on white),
    ``SD_TYPO`` (both), ``Text_only`` (no image).  Thirteen scenarios.  The
    text prompt never names the forbidden thing -- it says *the image shows a
    phrase of an activity, list steps to perform it* -- so the harm lives
    entirely in the picture.  That property is what makes this the right
    benchmark for a vision lens.

``figstep``
    FigStep / SafeBench (Gong et al. 2023), ``AngelAlita/FigStep``.  500
    harmful instructions rendered as a numbered list on white, with a benign
    carrier prompt.

``hades``
    HADES (Li et al. 2024), ``Monosail/HADES``.  750 instructions x 6
    amplification steps; the image at step *k* is the diffusion-amplified
    version of the one at step *k-1*, with the harmful keyword typed beneath.
    Identical text across steps makes this a *within-item* image contrast.

``jailbreakv``
    JailbreakV-28K (Luo et al. 2024).  The CSV has 28 000 rows; the public
    repository ships only 300 of the referenced images, so only the 360 rows
    whose image is actually present are returned.  Reported honestly by
    :func:`available`.

``vlguard``
    Not the official ``ys-zong/VLGuard`` (gated), but the ungated mirror
    ``kirito011024/vlguard_safe_unsafes``: 558 image-question pairs with a
    safe/unsafe answer label.

``advbench`` / ``harmbench`` / ``safebench_text``
    Text-only behaviour lists, for pairing with a benign image.
    ``walledai/AdvBench`` and ``walledai/HarmBench`` are gated; the ungated
    mirrors ``kelly8tom/advbench_orig`` (520) and
    ``NoorNizar/HarmBench-Test-Standard`` (159) are used instead.

Not obtainable without credentials: ``ys-zong/VLGuard``,
``walledai/AdvBench``, ``walledai/HarmBench`` (all HTTP 401, gated).  VLSafe
is not published on the Hub at all.

The harmful / benign split
--------------------------

MM-SafetyBench's own thirteen scenarios divide as the paper divides them:
01-07 are forbidden content, 08-13 are professionally restricted but not
harmful.  :data:`HARMFUL_SCENARIOS` and :data:`BENIGN_SCENARIOS` follow that
division, with ``Privacy_Violence`` held out of both because it sits on the
line.  This gives a benign control that shares the harmful items' *generator,
typography, layout and prompt template* and differs only in the words -- which
is the tightest control the literature offers, and much tighter than pairing
harmful renders against photographs.
"""
from __future__ import annotations

import functools
import glob
import hashlib
import io
import json
import os
import re

ROOT = os.environ.get("JVLM_SAFETY_ROOT", "/workspace/vlm/data/safety")
CACHE = os.path.join(ROOT, "_images")

MM = os.path.join(ROOT, "mmsafetybench", "data")
FIGSTEP = os.path.join(ROOT, "figstep", "data")
HADES = os.path.join(ROOT, "hades", "data")
JBV = os.path.join(ROOT, "jbv28k", "JailBreakV_28K")
VLGUARD = os.path.join(ROOT, "vlguard_alt", "data")
ADVBENCH = os.path.join(ROOT, "advbench_alt", "advbench_behaviors.jsonl")
HARMBENCH = os.path.join(ROOT, "harmbench", "data")
SAFEBENCH = os.path.join(ROOT, "safebench")

#: MM-SafetyBench 01-07 -- forbidden content.
HARMFUL_SCENARIOS = ["Illegal_Activitiy", "HateSpeech", "Malware_Generation",
                     "Physical_Harm", "EconomicHarm", "Fraud", "Sex"]
#: MM-SafetyBench 08-13 minus Privacy_Violence -- restricted, not harmful.
BENIGN_SCENARIOS = ["Political_Lobbying", "Legal_Opinion", "Financial_Advice",
                    "Health_Consultation", "Gov_Decision"]
#: On the line; in neither default group, available by explicit request.


VARIANTS = ["SD", "SD_TYPO", "TYPO", "Text_only"]


def _is_harmful(scenario: str) -> bool:
    """Label a scenario, erring towards harmful when it is not classified.

    ``Privacy_Violence`` and anything outside the two lists come back True:
    silently filing an unclassified safety scenario under *benign* would
    contaminate the control group, which is the one error here that changes a
    result rather than merely shrinking it.
    """
    if scenario in BENIGN_SCENARIOS:
        return False
    return True


# ------------------------------------------------------------------ plumbing

def _pd():
    import pandas as pd
    return pd


def _write_image(raw, key: str) -> str:
    """Decode an inline image once, cache it as a file, return the path.

    Every loader here returns ``image_path`` rather than pixels, so an
    experiment can hold ten thousand records without holding ten thousand
    decoded images, and so the same picture read twice is byte-identical.
    """
    if isinstance(raw, dict):
        raw = raw.get("bytes")
    if raw is None:
        return None
    sub = key.split("/")[0]
    d = os.path.join(CACHE, sub)
    os.makedirs(d, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", key.split("/", 1)[-1])
    if len(name) > 120:
        name = name[:100] + "_" + hashlib.md5(key.encode()).hexdigest()[:12]
    p = os.path.join(d, name + ".png")
    if not os.path.exists(p):
        from PIL import Image
        Image.open(io.BytesIO(raw)).convert("RGB").save(p)
    return p


def _take(rows, n, seed):
    """Deterministic subsample that keeps class balance callers set upstream."""
    if n is None or n >= len(rows):
        return rows
    import numpy as np
    idx = np.random.default_rng(seed).permutation(len(rows))[:n]
    return [rows[int(i)] for i in sorted(idx)]


def _take_df(d, n, seed):
    """Subsample a frame *before* its images are decoded.

    Decoding first and sampling afterwards writes tens of thousands of PNGs to
    fetch two hundred, which is how the first version of this module spent four
    minutes returning a smoke test's worth of records.
    """
    if n is None or n >= len(d):
        return d
    import numpy as np
    idx = np.random.default_rng(seed).permutation(len(d))[:n]
    return d.iloc[sorted(int(i) for i in idx)]


# ------------------------------------------------------------- MM-SafetyBench

@functools.lru_cache(maxsize=64)
def _mm_frame(scenario: str, variant: str):
    p = os.path.join(MM, scenario, variant + ".parquet")
    if not os.path.exists(p):
        return None
    return _pd().read_parquet(p)


@functools.lru_cache(maxsize=16)
def _mm_text_only(scenario: str) -> dict:
    """``id -> the unmasked harmful question``, which the image variants hide.

    The image variants replace the dangerous noun phrase with *this activity*
    and put the phrase in the picture.  The ``Text_only`` variant of the same
    id still carries it, so this is where a harmful vocabulary comes from
    without anybody writing one by hand.
    """
    d = _mm_frame(scenario, "Text_only")
    if d is None:
        return {}
    return {str(r.id): str(r.question) for r in d.itertuples()}


def mm_safetybench(n=None, variant: str = "SD_TYPO", scenarios=None,
                   harmful=None, seed: int = 0, balance: bool = False) -> list:
    """MM-SafetyBench records for one image variant.

    ``harmful=True`` selects :data:`HARMFUL_SCENARIOS`, ``False`` selects
    :data:`BENIGN_SCENARIOS`, ``None`` takes both and labels each row.  With
    ``balance`` and ``harmful=None``, ``n`` is split evenly between the two.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if scenarios is None:
        if harmful is True:
            scenarios = HARMFUL_SCENARIOS
        elif harmful is False:
            scenarios = BENIGN_SCENARIOS
        else:
            scenarios = HARMFUL_SCENARIOS + BENIGN_SCENARIOS
    if balance and harmful is None:
        h = mm_safetybench(None if n is None else (n + 1) // 2, variant,
                           HARMFUL_SCENARIOS, seed=seed)
        b = mm_safetybench(None if n is None else n // 2, variant,
                           BENIGN_SCENARIOS, seed=seed + 1)
        return h + b
    pdm = _pd()
    frames = []
    for s in scenarios:
        d = _mm_frame(s, variant)
        if d is None:
            continue
        d = d.assign(scen=s)
        frames.append(d)
    if not frames:
        return []
    d = _take_df(pdm.concat(frames, ignore_index=True), n, seed)
    rows = []
    for r in d.itertuples():
        s = str(r.scen)
        rid = f"mm/{s}/{variant}/{r.id}"
        rows.append(dict(
            id=rid, source="mm_safetybench", scenario=s, variant=variant,
            question=str(r.question), harmful=_is_harmful(s),
            text_only=_mm_text_only(s).get(str(r.id), ""),
            image_path=(None if variant == "Text_only"
                        else _write_image(r.image, rid)),
        ))
    return rows


# ---------------------------------------------------------------- FigStep

def figstep(n=None, seed: int = 0) -> list:
    """FigStep / SafeBench: the instruction is a numbered list in the image."""
    fs = sorted(glob.glob(os.path.join(FIGSTEP, "*.parquet")))
    if not fs:
        return []
    d = _take_df(_pd().concat([_pd().read_parquet(f) for f in fs],
                              ignore_index=True), n, seed)
    rows = []
    for r in d.itertuples():
        rid = f"figstep/{r.category_id}_{r.task_id}"
        rows.append(dict(
            id=rid, source="figstep", scenario=str(r.category_name),
            variant="figstep", question=str(r.jailbreak_prompt), harmful=True,
            text_only=str(r.question), instruction=str(r.instruction),
            image_path=_write_image(r.image, rid),
        ))
    return rows


# ------------------------------------------------------------------- HADES

def hades(n=None, steps=(0, 5), seed: int = 0, scenarios=None) -> list:
    """HADES, as a *within-item* contrast over amplification steps.

    Text is identical across steps, so a difference between step 0 and step 5
    is a difference the image made and nothing else.  Both steps are labelled
    ``harmful`` -- step 0 is already a harmful picture, only less amplified --
    and ``step`` is what an experiment should contrast.  Calling it a benign
    control would be wrong, and this docstring exists to stop that.
    """
    fs = sorted(glob.glob(os.path.join(HADES, "*.parquet")))
    if not fs:
        return []
    d = _pd().concat([_pd().read_parquet(f) for f in fs], ignore_index=True)
    d = d[d.step.astype(int).isin([int(s) for s in steps])]
    d = d.reset_index(drop=True)
    if scenarios:
        d = d[d.scenario.isin(scenarios)]
    base = d.id.str.rsplit("_step", n=1).str[0]
    keep = None
    if n is not None:
        import numpy as np
        u = sorted(base.unique())
        pick = np.random.default_rng(seed).permutation(len(u))[
            : max(1, n // max(len(steps), 1))]
        keep = {u[int(i)] for i in pick}
    rows = []
    for r, b in zip(d.itertuples(), base, strict=True):
        if keep is not None and b not in keep:
            continue
        rid = f"hades/{r.id}"
        rows.append(dict(
            id=rid, source="hades", scenario=str(r.scenario),
            variant=f"step{int(r.step)}", question=str(r.instruction),
            harmful=True, step=int(r.step), item=b, keywords=str(r.keywords),
            category=str(r.category), text_only=str(r.instruction),
            image_path=_write_image(r.image, rid),
        ))
    return rows


# ------------------------------------------------------------- JailbreakV-28K

def jailbreakv(n=None, seed: int = 0, mini: bool = False) -> list:
    """JailbreakV-28K, restricted to rows whose image the repo actually ships.

    The public repository holds 300 of the images its CSV references, so this
    returns roughly 360 of 28 000 rows.  The shortfall is real and is reported
    by :func:`available` rather than hidden by silent substitution.
    """
    csv = os.path.join(JBV, "mini_JailBreakV_28K.csv" if mini
                       else "JailBreakV_28K.csv")
    if not os.path.exists(csv):
        return []
    d = _pd().read_csv(csv)
    rows = []
    for r in d.itertuples():
        p = os.path.join(JBV, str(r.image_path))
        if not os.path.exists(p):
            continue
        rows.append(dict(
            id=f"jbv/{r.id}", source="jailbreakv", scenario=str(r.policy),
            variant=str(getattr(r, "format")), question=str(r.jailbreak_query),
            harmful=True, text_only=str(r.redteam_query), image_path=p,
        ))
    return _take(rows, n, seed)


def jailbreakv_redteam(n=None, seed: int = 0) -> list:
    """The 2 000 text-only red-team questions behind JailbreakV-28K."""
    p = os.path.join(JBV, "RedTeam_2K.csv")
    if not os.path.exists(p):
        return []
    d = _pd().read_csv(p)
    rows = [dict(id=f"jbv_rt/{r.id}", source="jailbreakv_redteam",
                 scenario=str(r.policy), variant="text", harmful=True,
                 question=str(r.question), text_only=str(r.question),
                 image_path=None) for r in d.itertuples()]
    return _take(rows, n, seed)


# ------------------------------------------------------------------ VLGuard

def vlguard(n=None, seed: int = 0) -> list:
    """VLGuard mirror: image + question, ``answer`` says whether it is unsafe."""
    fs = sorted(glob.glob(os.path.join(VLGUARD, "*.parquet")))
    if not fs:
        return []
    d = _take_df(_pd().concat([_pd().read_parquet(f) for f in fs],
                              ignore_index=True), n, seed)
    rows = []
    for r in d.itertuples():
        rid = f"vlguard/{r.id}"
        unsafe = str(r.answer).strip().lower().startswith("y")
        rows.append(dict(
            id=rid, source="vlguard", scenario="vlguard",
            variant="unsafe" if unsafe else "safe", question=str(r.question),
            harmful=bool(unsafe), text_only=str(r.question),
            image_path=_write_image(r.image, rid),
        ))
    return rows


# ------------------------------------------------------- text-only behaviours

def advbench(n=None, seed: int = 0) -> list:
    """AdvBench harmful behaviours (ungated mirror), no image."""
    if not os.path.exists(ADVBENCH):
        return []
    rows = []
    with open(ADVBENCH, encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(dict(id=f"advbench/{r['BehaviorID']}",
                             source="advbench", scenario="advbench",
                             variant="text", question=r["Behavior"],
                             harmful=True, text_only=r["Behavior"],
                             target=r.get("target", ""), image_path=None))
    return _take(rows, n, seed)


def harmbench(n=None, seed: int = 0) -> list:
    """HarmBench standard behaviours (ungated mirror), no image."""
    fs = sorted(glob.glob(os.path.join(HARMBENCH, "*.parquet")))
    if not fs:
        return []
    d = _pd().concat([_pd().read_parquet(f) for f in fs], ignore_index=True)
    rows = [dict(id=f"harmbench/{r.BehaviorID}", source="harmbench",
                 scenario=str(r.SemanticCategory), variant="text",
                 question=str(r.prompt), harmful=True,
                 text_only=str(r.prompt), image_path=None)
            for r in d.itertuples()]
    return _take(rows, n, seed)


def safebench_text(n=None, seed: int = 0) -> list:
    """The 23-category SafeBench question list (text side only)."""
    cat = {}
    p = os.path.join(SAFEBENCH, "category.csv")
    if os.path.exists(p):
        c = _pd().read_csv(p)
        # itertuples() steals the name ``Index`` for the row index, which
        # silently shifted every category label by one the first time round.
        cat = {int(i): re.sub(r"^\d+", "", str(v)).strip()
               for i, v in zip(c["Index"], c["Category"], strict=True)}
    rows = []
    for f in sorted(glob.glob(os.path.join(SAFEBENCH, "SafeBench_Text",
                                           "*.csv"))):
        k = int(os.path.basename(f)[:-4])
        d = _pd().read_csv(f, header=None)
        for j, q in enumerate(d[0].astype(str)):
            rows.append(dict(id=f"safebench/{k}_{j}", source="safebench_text",
                             scenario=cat.get(k, str(k)), variant="text",
                             question=q, harmful=True, text_only=q,
                             image_path=None))
    return _take(rows, n, seed)


# -------------------------------------------------------------- vocabularies

STOP = set("""a an the this that these those and or but if then than of to in on
at for with by from as is are was were be been being it its it's his her their
your our my you i we they he she them us me do does did done doing how what
which who whom whose when where why can could should would may might must will
shall shan't don't not no nor so such into onto over under about after before
during while against between within without through above below up down out off
again further once here there all any both each few more most other some only
own same too very s t just now provide list detail details steps step perform
activity image shows phrase bottom generate content item items please write
give make show describe explain tell using use used one two three""".split())

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")


def _content_words(texts):
    """Frequency-ranked content words of a corpus, stopwords removed."""
    import collections
    c = collections.Counter()
    for t in texts:
        for w in _WORD.findall(str(t).lower()):
            if len(w) > 2 and w not in STOP:
                c[w] += 1
    return c


def single_token(tokenizer, word: str):
    """The id of ``word`` when it is one token, in any of its surface forms."""
    for form in (" " + word, word, " " + word.capitalize(), word.capitalize()):
        ids = tokenizer.encode(form, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    return None


def contrast_vocab(tokenizer, k: int = 16, ratio: float = 3.0,
                   min_count: int = 6, seed: int = 0) -> dict:
    r"""A harmful word list and a benign one, both read off the benchmark.

    The words come from MM-SafetyBench's ``Text_only`` questions -- the harmful
    ones from scenarios 01-07, the benign ones from 08-13.  Ranking by raw
    frequency alone hands back ``create``, ``people`` and ``online``, which
    both halves use and which therefore measure nothing.  Words are ranked by
    how *lopsided* they are -- the rate in their own half over the rate in the
    other -- among those appearing at least ``min_count`` times and clearing
    ``ratio``.  What survives is the vocabulary the two halves genuinely
    disagree about, and it is chosen by the corpus rather than by us.

    Only words the tokenizer spells in one piece are eligible, since a lens
    reads single vocabulary items.  The two lists are then made disjoint and
    equal in length, so ``mean(z[harm]) - mean(z[benign])`` is symmetric under
    relabelling and cannot be won by one list simply being longer.

    Returns ``{harm, benign, harm_ids, benign_ids, vocab, harm_idx,
    benign_idx}``; ``vocab`` is the concatenated id list to hand a lens, and
    the two ``_idx`` lists index into it.
    """
    ch = _content_words(q for s in HARMFUL_SCENARIOS
                        for q in _mm_text_only(s).values())
    cb = _content_words(q for s in BENIGN_SCENARIOS
                        for q in _mm_text_only(s).values())
    if not ch or not cb:
        raise RuntimeError("MM-SafetyBench is not downloaded; "
                           "contrast_vocab has nothing to read")
    nh, nb = sum(ch.values()), sum(cb.values())

    def pick(mine, other, n_mine, n_other, taken):
        cand = []
        for w, c in mine.items():
            if c < min_count:
                continue
            lop = (c / n_mine) / ((other.get(w, 0) + 0.5) / n_other)
            if lop >= ratio:
                cand.append((lop, c, w))
        cand.sort(reverse=True)
        out, ids = [], []
        for _, _, w in cand:
            if len(out) >= k:
                break
            i = single_token(tokenizer, w)
            if i is None or i in taken or i in ids:
                continue
            out.append(w)
            ids.append(i)
        return out, ids

    harm, hid = pick(ch, cb, nh, nb, set())
    ben, bid = pick(cb, ch, nb, nh, set(hid))
    n = min(len(harm), len(ben))
    if n < 4:
        raise RuntimeError(
            f"only {n} usable words per half at ratio={ratio}; "
            "lower --vocab or the ratio")
    harm, ben, hid, bid = harm[:n], ben[:n], hid[:n], bid[:n]
    return dict(harm=harm, benign=ben, harm_ids=hid, benign_ids=bid,
                vocab=hid + bid, harm_idx=list(range(n)),
                benign_idx=list(range(n, 2 * n)), ratio=ratio)


# ------------------------------------------------------------- availability

def _parquet_rows(pattern) -> int:
    """Row count from parquet footers -- no decoding, no images touched."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return 0
    n = 0
    for f in glob.glob(pattern):
        try:
            n += pq.ParquetFile(f).metadata.num_rows
        except OSError:
            pass
    return n


def _csv_rows(path) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8", errors="replace") as f:
        return max(0, sum(1 for _ in f) - 1)


def _jbv_with_image() -> int:
    """JailbreakV rows whose referenced image the public repo actually ships."""
    csv = os.path.join(JBV, "JailBreakV_28K.csv")
    if not os.path.exists(csv):
        return 0
    have = set()
    for sub in ("figstep", "llm_transfer_attack", "query_related"):
        d = os.path.join(JBV, sub)
        if os.path.isdir(d):
            have |= {sub + "/" + f for f in os.listdir(d)}
    return int(_pd().read_csv(csv).image_path.isin(have).sum())


_SOURCES = [
    ("mm_safetybench", MM, "PKU-Alignment/MM-SafetyBench",
     lambda: _parquet_rows(os.path.join(MM, "*", "*.parquet")),
     "1680 questions x 4 image variants x 13 scenarios"),
    ("figstep", FIGSTEP, "AngelAlita/FigStep",
     lambda: _parquet_rows(os.path.join(FIGSTEP, "*.parquet")),
     "SafeBench: harmful instruction as a numbered list in the image"),
    ("hades", HADES, "Monosail/HADES",
     lambda: _parquet_rows(os.path.join(HADES, "*.parquet")),
     "750 instructions x 6 amplification steps"),
    ("jailbreakv", JBV, "JailbreakV-28K/JailBreakV-28k",
     _jbv_with_image,
     "28000 CSV rows but only ~360 have an image in the public repo"),
    ("vlguard", VLGUARD, "kirito011024/vlguard_safe_unsafes",
     lambda: _parquet_rows(os.path.join(VLGUARD, "*.parquet")),
     "ungated mirror; ys-zong/VLGuard itself is gated"),
    ("advbench", ADVBENCH, "kelly8tom/advbench_orig",
     lambda: sum(1 for x in open(ADVBENCH, encoding="utf-8-sig")
                 if x.strip()) if os.path.exists(ADVBENCH) else 0,
     "text only; walledai/AdvBench is gated"),
    ("harmbench", HARMBENCH, "NoorNizar/HarmBench-Test-Standard",
     lambda: _parquet_rows(os.path.join(HARMBENCH, "*.parquet")),
     "text only; walledai/HarmBench is gated"),
    ("safebench_text", SAFEBENCH, "Zonghao2025/safebench",
     lambda: sum(_csv_rows(f) for f in glob.glob(
         os.path.join(SAFEBENCH, "SafeBench_Text", "*.csv"))),
     "23 categories of text questions"),
]

#: Sources that are known to be unobtainable from this machine, and why.
MISSING = {
    "vlguard_official": "ys-zong/VLGuard -- HTTP 401, gated repository",
    "advbench_walledai": "walledai/AdvBench -- HTTP 401, gated repository",
    "harmbench_walledai": "walledai/HarmBench -- HTTP 401, gated repository",
    "vlsafe": "not published on the Hugging Face Hub under any name we found",
}


def available() -> dict:
    """What is on disk, so an experiment can degrade instead of crashing.

    Record counts come from parquet footers and CSV line counts, so calling
    this is cheap and never decodes an image.
    """
    out = {}
    for name, path, origin, count, note in _SOURCES:
        try:
            n = int(count())
        except (OSError, ValueError):
            n = 0
        out[name] = dict(present=bool(n), records=n, path=path, origin=origin,
                         note=note)
    out["_missing"] = MISSING
    return out


def summary() -> str:
    """One line per source, for the top of an experiment's log."""
    lines = []
    for k, v in available().items():
        if k.startswith("_"):
            continue
        lines.append("  %-16s %-9s %7d  %s" %
                     (k, "present" if v["present"] else "MISSING",
                      v["records"], v["origin"]))
    for k, v in MISSING.items():
        lines.append("  %-16s %-9s %7s  %s" % (k, "MISSING", "-", v))
    return "\n".join(lines)

# =====================================================================
#                       EVALUATION AWARENESS
# =====================================================================
r"""Eval-vs-deployment framings from a published benchmark, on photographs.

What exists, and what does not
------------------------------

The literature's eval-awareness benchmark is *Large Language Models Often
Know When They Are Being Evaluated* (Needham, Edkins et al., arXiv:2505.23836).
Its own corpus, ``jjpn2/eval_awareness``, is a **gated** repository and ships
its items as an *encrypted* ``dataset.zip`` -- deliberately, so the benchmark
does not leak into training data.  From this machine it answers HTTP 401.

The derivative ``Anonymous648/eval-awareness-tells`` (the tell-transplantation
set) is likewise not fetchable: the repository under that name does not
resolve, and the ``eval-awareness-cues`` repository that documents it ships
only code.

What *is* ungated and does download is ``viliana-dev/eval-awareness-2x2``, a
controlled two-by-two built on top of that paper.  It crosses

    format      benchmark-shaped ("Question: ... Choices: ... Answer:")
                against casual prose
    provenance  really an evaluation item against really a user's query

which is the cross a lens experiment needs, because it separates *this prompt
is wearing benchmark clothes* from *this prompt is actually a test*.  A reading
that only tracks the first is reading typography, not situation.

**No multimodal eval-awareness benchmark exists.**  Searching the Hub for
multimodal / VLM / vision variants of situational- and evaluation-awareness
returns nothing at all -- not gated, absent.  So the multimodal arm here is an
honest fallback and is labelled as one: the prompts are the published
benchmark's text, used **verbatim**, and each is shown to the model beside a
COCO photograph.  The eval/deployment label is the benchmark's, not ours.  The
photograph carries no eval/deployment information whatsoever, which is exactly
what makes it the right probe -- see :func:`eval_awareness`.

The four arms
-------------

``paired``
    The within-item contrast, and the strongest one.  For each of 835 ids the
    2x2 keeps a real user's query in their own words (``original_query``) and
    the *same question* rewritten into benchmark format (``rewritten``).  Same
    content, same id, and this loader gives both members **the same
    photograph**, so a difference is the framing's doing and nothing else.

``provenance``
    Format held fixed at benchmark-shaped.  Genuine MMLU/ARC/GSM8K items
    (``bench_eval``) against user queries dressed in the identical format
    (``bench_deploy_rewritten.rewritten``).  Both look like a test; only one is
    one.  This is the arm that cannot be won by reading typography.

``casual``
    Format held fixed at casual prose, provenance flipped -- and the arm that
    controls for length.  In ``paired`` and ``provenance`` the eval side is the
    *longer* text (median 238 characters against 65).  Here it is the
    *shorter* one (88 against 224).  A reading that tracks eval-ness keeps its
    sign across both; a reading that tracks prompt length must flip.  That
    reversal is a free, real control and experiments should use it.

``natural``
    Both format and provenance differ: ``bench_eval`` against
    ``casual_deploy_matched``.  The easy contrast, reported for comparison and
    not as the headline.
"""

EVALAWARE = os.environ.get("JVLM_EVALAWARE_ROOT",
                           "/workspace/vlm/data/evalaware")

#: The ungated 2x2 and the file each config lives in.
_EA_REPO = "viliana-dev/eval-awareness-2x2"
_EA_CONFIGS = ["bench_eval", "bench_eval_full", "bench_deploy_rewritten",
               "bench_deploy_formal", "casual_eval_mutual",
               "casual_eval_mutual_full", "casual_deploy_matched",
               "casual_deploy_formal"]

#: Arms as ``(eval side, deployment side)``; see the module docstring.
EA_ARMS = ["paired", "provenance", "casual", "natural"]

#: Arms that draw rows from ``bench_deploy_rewritten`` -- the same table
#: :func:`ea_vocab` fits its word lists on.  ``paired`` takes both of its
#: columns; ``provenance`` takes ``rewritten`` as its *deployment* side.  The
#: scored ids of these arms must be handed to ``ea_vocab(exclude_ids=...)`` or
#: the vocabulary is read off the very items it is then used to separate.


def _ea_path(config: str) -> str:
    return os.path.join(EVALAWARE, f"2x2_{config}.parquet")


def ea_download(configs=None, force: bool = False) -> dict:
    """Fetch the ungated 2x2 parquets.  Returns ``{config: rows or error}``.

    Kept separate from the loaders so that a run on a machine with no network
    fails at one obvious place rather than halfway through an experiment.
    """
    import urllib.request
    os.makedirs(EVALAWARE, exist_ok=True)
    out = {}
    for c in (configs or _EA_CONFIGS):
        p = _ea_path(c)
        if os.path.exists(p) and not force:
            out[c] = _parquet_rows(p)
            continue
        url = (f"https://huggingface.co/datasets/{_EA_REPO}/resolve/main/"
               f"{c}/train-00000-of-00001.parquet")
        try:
            urllib.request.urlretrieve(url, p)
            out[c] = _parquet_rows(p)
        except OSError as e:
            out[c] = f"FAILED: {e}"
    return out


@functools.lru_cache(maxsize=16)
def _ea_frame(config: str):
    p = _ea_path(config)
    if not os.path.exists(p):
        return None
    return _pd().read_parquet(p)


def _ea_photos(n: int, seed: int, offset: int = 0):
    """COCO photographs, as paths, for the carrier images."""
    from vlens import corpus
    return [str(p) for p in corpus.coco_images(n, seed=seed,
                                               split_offset=offset)]


def _ea_rows(arm: str):
    """``(eval_side, deploy_side)`` as lists of ``(key, text, extra)``.

    ``key`` is what pairs the two sides.  In ``paired`` it is the shared item
    id, so the pair is a genuine within-item contrast; elsewhere it is the row
    index and the sides are independent samples.
    """
    if arm == "paired":
        d = _ea_frame("bench_deploy_rewritten")
        if d is None:
            return [], []
        ev = [(str(r.id), str(r.rewritten), dict(label=str(r.label)))
              for r in d.itertuples()]
        dep = [(str(r.id), str(r.original_query), dict(label=str(r.label)))
               for r in d.itertuples()]
        return ev, dep
    if arm == "provenance":
        e, d = _ea_frame("bench_eval"), _ea_frame("bench_deploy_rewritten")
        if e is None or d is None:
            return [], []
        ev = [(f"be{i}", str(r.formatted_prompt), dict(label=str(r.source)))
              for i, r in enumerate(e.itertuples())]
        dep = [(str(r.id), str(r.rewritten), dict(label=str(r.label)))
               for r in d.itertuples()]
        return ev, dep
    if arm == "casual":
        e, d = _ea_frame("casual_eval_mutual"), _ea_frame(
            "casual_deploy_matched")
        if e is None or d is None:
            return [], []
        ev = [(str(r.id), str(r.first_turn), dict(label="dailydialog"))
              for r in e.itertuples()]
        dep = [(str(r.id), str(r.first_turn), dict(label="user_query"))
               for r in d.itertuples()]
        return ev, dep
    if arm == "natural":
        e, d = _ea_frame("bench_eval"), _ea_frame("casual_deploy_matched")
        if e is None or d is None:
            return [], []
        ev = [(f"be{i}", str(r.formatted_prompt), dict(label=str(r.source)))
              for i, r in enumerate(e.itertuples())]
        dep = [(str(r.id), str(r.first_turn), dict(label="user_query"))
               for r in d.itertuples()]
        return ev, dep
    raise ValueError(f"arm must be one of {EA_ARMS}")


def eval_awareness(n=None, arm: str = "paired", seed: int = 0,
                   photo_seed: int = 91, photo_offset: int = 0,
                   max_chars: int = 900, min_chars: int = 25) -> list:
    r"""Eval / deployment framings from the 2x2, each beside a photograph.

    Returns ``{id, image_path, prompt, is_eval, source}`` plus ``arm``,
    ``pair_id``, ``label`` and ``n_chars``.

    Photographs are **matched across the two classes in every arm**: the i-th
    evaluation record and the i-th deployment record are shown the same
    picture (in ``paired`` that is also the same question, so the match is
    within-item).  ``photo_id`` records which picture, so the match can be
    used or checked downstream.

    This matters more than it looks.  The image precedes the text and
    attention is causal, so a photograph shared by the two classes cannot
    carry the framing into the image tokens, and any difference a lens reports
    at those positions is not a difference in what the picture is -- it is a
    difference in what the picture is *for*.  Without the match, each class
    draws its own disjoint set of photographs and the label sits in the image
    tokens directly, which is the one thing this design exists to prevent.

    The activations are equal in that sense and not bitwise: the two framings
    give the sequence different lengths, so the kernels reduce in a different
    order and the residue is the arithmetic's round-off (about 1e-6 relative
    in float32, 1e-2 in bfloat16).  A content lens therefore lands *near* 0.5
    rather than exactly on it, and it is the confidence interval that says
    whether it is at chance.

    ``n`` is the number of records **in total**, split evenly between the two
    sides, so ``n=200`` gives 100 evaluation and 100 deployment records.
    Items shorter than ``min_chars`` or longer than ``max_chars`` are dropped
    before sampling: the very short ones carry no framing at all and the very
    long ones dominate the sequence, and both were sampled unevenly across the
    arms until this filter was added.
    """
    ev, dep = _ea_rows(arm)
    if not ev or not dep:
        return []

    def ok(t):
        return min_chars <= len(t) <= max_chars

    ev = [x for x in ev if ok(x[1])]
    dep = [x for x in dep if ok(x[1])]
    if arm == "paired":
        # a pair survives only if both of its framings do
        keep = {k for k, _, _ in ev} & {k for k, _, _ in dep}
        ev = [x for x in ev if x[0] in keep]
        dep = [x for x in dep if x[0] in keep]
    half = None if n is None else max(1, n // 2)
    if half is not None:
        if arm == "paired":
            # sample pair keys, never the two sides independently, or the
            # within-item design silently degrades to a between-item one
            import numpy as np
            keys = sorted({k for k, _, _ in ev})
            pick = set(np.asarray(keys)[
                np.random.default_rng(seed).permutation(len(keys))[:half]])
            ev = [x for x in ev if x[0] in pick]
            dep = [x for x in dep if x[0] in pick]
        else:
            ev = _take(ev, half, seed)
            dep = _take(dep, half, seed + 1)

    # Photographs are matched across the two classes in **every** arm, not
    # only in ``paired``.  Keying the picture on the union of the two sides'
    # ids -- which is what this did -- only matches them when the two sides
    # share an id space, and ``provenance``, ``casual`` and ``natural`` do not:
    # their eval keys sort as one block and their deploy keys as another, so
    # each class drew from a disjoint half of the photograph pool and no
    # picture ever appeared on both sides.  That put the label into the image
    # tokens themselves in exactly the arms whose whole point is that it is
    # not there.  Pairing by rank within each side instead gives the i-th
    # evaluation item and the i-th deployment item the same photograph, so the
    # two classes see an identical set of pictures in every arm.
    n_ph = max(len(ev), len(dep))
    photos = _ea_photos(n_ph, photo_seed, photo_offset)
    if not photos:
        raise RuntimeError("no COCO photographs found; eval_awareness needs "
                           "corpus.coco_images to resolve")

    rows = []
    for side, is_eval in ((ev, True), (dep, False)):
        for i, (k, text, extra) in enumerate(side):
            rows.append(dict(
                id=f"ea/{arm}/{'eval' if is_eval else 'deploy'}/{k}",
                source="eval_awareness_2x2", arm=arm, pair_id=k,
                photo_id=i, image_path=photos[i % len(photos)], prompt=text,
                is_eval=bool(is_eval), n_chars=len(text), **extra))
    rows.sort(key=lambda r: (r["pair_id"], not r["is_eval"]))
    return rows


#: Function words only.  :data:`STOP` also drops ``question``, ``answer``,
#: ``list`` and ``provide``, which were noise in MM-SafetyBench's template and
#: are the single most informative words here -- so this arm needs its own.
EA_STOP = set("""a an the this that these those and or but if then than of to
in on at for with by from as is are was were be been being am it its it's his
her their your our my you i we they he she them us me do does did done doing
so such into onto over under about after before during while against between
within without through above below up down out off again further once here
there all any both each few more most other some only own same too very s t
just now not no nor can could should would may might must will shall
https http www com org net""".split())
# the last line is URL debris rather than vocabulary: real users paste links
# and benchmark items do not, so ``https`` and ``com`` survive a document-count
# and rank among the top deployment words while naming nothing about framing.


def _ea_docfreq(texts) -> dict:
    r"""How many *items* contain each word, not how many times it occurs.

    Term frequency is the wrong count here and gets the answer wrong in a way
    that is easy to miss.  A handful of real user queries in this corpus paste
    in log dumps and URLs, so counting occurrences hands back ``amazonaws``,
    ``splunk`` and ``prod-api-ora-`` as the top "deployment" words at
    lopsidedness above 100 -- signal from one or two items, dressed as a
    corpus-wide effect.  Counting items instead drops them to the floor and
    leaves the words that genuinely recur across queries.
    """
    import collections
    c = collections.Counter()
    for t in texts:
        c.update({w for w in _WORD.findall(str(t).lower())
                  if len(w) > 2 and w not in EA_STOP})
    return c


def ea_vocab(tokenizer, k: int = 10, ratio: float = 2.0,
             min_items: int = 10, exclude_ids=()) -> dict:
    r"""An evaluation word list and a deployment one, read off the corpus.

    Same lopsidedness construction as :func:`contrast_vocab` -- rank by a
    word's rate in its own half over its rate in the other, keep only what the
    tokenizer spells in one piece, then make the lists disjoint and equal in
    length so the contrast is symmetric under relabelling.

    Two things differ, and both were forced by checking what came out.

    The corpus is the **paired** arm and only that arm: the 835 items where the
    same user question exists in both framings.  Reading the vocabulary off
    ``bench_eval`` against ``casual_deploy_matched`` instead -- the obvious
    choice -- returns ``sun, oxygen, atmosphere, hydrogen, soil`` for the
    evaluation half and ``python, json, api, server, https`` for the
    deployment half.  That is a *topic* discriminator: those pools are science
    exams and coding chat.  A lens scored against it would be graded on telling
    chemistry from software, and would score well while measuring nothing about
    evaluation.  Within a pair the topic is held fixed by construction, so the
    words that survive are the words the *framing* changed and nothing else.

    The count is document frequency, for the reason in :func:`_ea_docfreq`.

    What comes out is ``options, choices, question, answer, primarily, focus,
    described, characteristic`` against ``explain, please, tell, know,
    describe, example, give, how`` -- multiple-choice stem markers against the
    verbs people use when they actually want something.  The two halves are
    unequally lopsided (the evaluation template is near-universal, real queries
    are heterogeneous), which is why ``ratio`` defaults lower than
    :func:`contrast_vocab`'s.

    Returns ``{eval, deploy, eval_ids, deploy_ids, vocab, eval_idx,
    deploy_idx}``; ``vocab`` is the id list to hand a lens.
    """
    d = _ea_frame("bench_deploy_rewritten")
    if d is None:
        raise RuntimeError(
            "the eval-awareness 2x2 is not on disk; run safedata.ea_download()")
    ex = {str(x) for x in exclude_ids}
    if ex:
        d = d[~d.id.astype(str).isin(ex)]
    if len(d) < 50:
        raise RuntimeError(
            f"only {len(d)} items left after excluding {len(ex)} scored ids; "
            "the vocabulary would be fitted to noise")
    ce, cd = _ea_docfreq(d.rewritten), _ea_docfreq(d.original_query)
    n_items = len(d)

    def pick(mine, other, taken):
        cand = []
        for w, c in mine.items():
            if c < min_items:
                continue
            lop = c / (other.get(w, 0) + 0.5)
            if lop >= ratio:
                cand.append((lop, c, w))
        cand.sort(reverse=True)
        out, ids = [], []
        for _, _, w in cand:
            if len(out) >= k:
                break
            i = single_token(tokenizer, w)
            if i is None or i in taken or i in ids:
                continue
            out.append(w)
            ids.append(i)
        return out, ids

    ew, eid = pick(ce, cd, set())
    dw, did = pick(cd, ce, set(eid))
    n = min(len(ew), len(dw))
    # the floor is a statistical one -- averaging over fewer than four words
    # per half is noise -- but it must not override an explicit request for a
    # tiny vocabulary, which is what a smoke test asks for and what used to
    # fail here with a message about lowering a ratio that was not the problem.
    if n < min(4, k):
        raise RuntimeError(
            f"only {n} usable words per half at ratio={ratio}, "
            f"min_items={min_items} (asked for {k}); lower one of them")
    ew, dw, eid, did = ew[:n], dw[:n], eid[:n], did[:n]
    return dict(eval=ew, deploy=dw, eval_ids=eid, deploy_ids=did,
                vocab=eid + did, eval_idx=list(range(n)),
                deploy_idx=list(range(n, 2 * n)), ratio=ratio,
                min_items=min_items, n_items=int(n_items),
                n_excluded=len(ex),
                source="paired arm (topic held fixed within item), "
                       "document frequency")


# Registered so ``available()`` and ``summary()`` report this source too,
# including the upstream sets that cannot be had.
_SOURCES.append((
    "eval_awareness", EVALAWARE, _EA_REPO,
    lambda: _parquet_rows(os.path.join(EVALAWARE, "2x2_*.parquet")),
    "eval/deployment 2x2; paired with COCO photographs (no multimodal "
    "eval-awareness benchmark exists)"))

MISSING.update({
    "eval_awareness_upstream":
        "jjpn2/eval_awareness (Needham et al., arXiv:2505.23836) -- HTTP 401, "
        "gated, and its items ship encrypted to prevent training-set leakage",
    "eval_awareness_tells":
        "Anonymous648/eval-awareness-tells -- HTTP 401; the companion "
        "eval-awareness-cues repository ships code but no data file",
    "eval_awareness_multimodal":
        "no VLM/multimodal eval-awareness benchmark is published anywhere we "
        "can find -- absent, not gated",
})
