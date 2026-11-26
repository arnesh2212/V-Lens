from __future__ import annotations

import functools
import glob
import hashlib
import io
import json
import os
import re

ROOT = os.environ.get("VLENS_SAFETY_ROOT", "data/safety")
CACHE = os.path.join(ROOT, "_images")

MM = os.path.join(ROOT, "mmsafetybench", "data")
FIGSTEP = os.path.join(ROOT, "figstep", "data")
HADES = os.path.join(ROOT, "hades", "data")
JBV = os.path.join(ROOT, "jbv28k", "JailBreakV_28K")
VLGUARD = os.path.join(ROOT, "vlguard_alt", "data")
ADVBENCH = os.path.join(ROOT, "advbench_alt", "advbench_behaviors.jsonl")
HARMBENCH = os.path.join(ROOT, "harmbench", "data")
SAFEBENCH = os.path.join(ROOT, "safebench")

HARMFUL_SCENARIOS = ["Illegal_Activitiy", "HateSpeech", "Malware_Generation",
                     "Physical_Harm", "EconomicHarm", "Fraud", "Sex"]
BENIGN_SCENARIOS = ["Political_Lobbying", "Legal_Opinion", "Financial_Advice",
                    "Health_Consultation", "Gov_Decision"]


VARIANTS = ["SD", "SD_TYPO", "TYPO", "Text_only"]


def _is_harmful(scenario: str) -> bool:
    if scenario in BENIGN_SCENARIOS:
        return False
    return True


def _pd():
    import pandas as pd
    return pd


def _write_image(raw, key: str) -> str:
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
    if n is None or n >= len(rows):
        return rows
    import numpy as np
    idx = np.random.default_rng(seed).permutation(len(rows))[:n]
    return [rows[int(i)] for i in sorted(idx)]


def _take_df(d, n, seed):
    if n is None or n >= len(d):
        return d
    import numpy as np
    idx = np.random.default_rng(seed).permutation(len(d))[:n]
    return d.iloc[sorted(int(i) for i in idx)]


@functools.lru_cache(maxsize=64)
def _mm_frame(scenario: str, variant: str):
    p = os.path.join(MM, scenario, variant + ".parquet")
    if not os.path.exists(p):
        return None
    return _pd().read_parquet(p)


@functools.lru_cache(maxsize=16)
def _mm_text_only(scenario: str) -> dict:
    d = _mm_frame(scenario, "Text_only")
    if d is None:
        return {}
    return {str(r.id): str(r.question) for r in d.itertuples()}


def mm_safetybench(n=None, variant: str = "SD_TYPO", scenarios=None,
                   harmful=None, seed: int = 0, balance: bool = False) -> list:
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


def figstep(n=None, seed: int = 0) -> list:
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


def hades(n=None, steps=(0, 5), seed: int = 0, scenarios=None) -> list:
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


def jailbreakv(n=None, seed: int = 0, mini: bool = False) -> list:
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
    p = os.path.join(JBV, "RedTeam_2K.csv")
    if not os.path.exists(p):
        return []
    d = _pd().read_csv(p)
    rows = [dict(id=f"jbv_rt/{r.id}", source="jailbreakv_redteam",
                 scenario=str(r.policy), variant="text", harmful=True,
                 question=str(r.question), text_only=str(r.question),
                 image_path=None) for r in d.itertuples()]
    return _take(rows, n, seed)


def vlguard(n=None, seed: int = 0) -> list:
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


def advbench(n=None, seed: int = 0) -> list:
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
    cat = {}
    p = os.path.join(SAFEBENCH, "category.csv")
    if os.path.exists(p):
        c = _pd().read_csv(p)
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
    import collections
    c = collections.Counter()
    for t in texts:
        for w in _WORD.findall(str(t).lower()):
            if len(w) > 2 and w not in STOP:
                c[w] += 1
    return c


def single_token(tokenizer, word: str):
    for form in (" " + word, word, " " + word.capitalize(), word.capitalize()):
        ids = tokenizer.encode(form, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    return None


def contrast_vocab(tokenizer, k: int = 16, ratio: float = 3.0,
                   min_count: int = 6, seed: int = 0) -> dict:
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


def _parquet_rows(pattern) -> int:
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

MISSING = {
    "vlguard_official": "ys-zong/VLGuard -- HTTP 401, gated repository",
    "advbench_walledai": "walledai/AdvBench -- HTTP 401, gated repository",
    "harmbench_walledai": "walledai/HarmBench -- HTTP 401, gated repository",
    "vlsafe": "not published on the Hugging Face Hub under any name we found",
}


def available() -> dict:
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


EVALAWARE = os.environ.get("VLENS_EVALAWARE_ROOT",
                           "data/evalaware")

_EA_REPO = "viliana-dev/eval-awareness-2x2"
_EA_CONFIGS = ["bench_eval", "bench_eval_full", "bench_deploy_rewritten",
               "bench_deploy_formal", "casual_eval_mutual",
               "casual_eval_mutual_full", "casual_deploy_matched",
               "casual_deploy_formal"]

EA_ARMS = ["paired", "provenance", "casual", "natural"]


def _ea_path(config: str) -> str:
    return os.path.join(EVALAWARE, f"2x2_{config}.parquet")


def ea_download(configs=None, force: bool = False) -> dict:
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
    from vlens import corpus
    return [str(p) for p in corpus.coco_images(n, seed=seed,
                                               split_offset=offset)]


def _ea_rows(arm: str):
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
    ev, dep = _ea_rows(arm)
    if not ev or not dep:
        return []

    def ok(t):
        return min_chars <= len(t) <= max_chars

    ev = [x for x in ev if ok(x[1])]
    dep = [x for x in dep if ok(x[1])]
    if arm == "paired":
        keep = {k for k, _, _ in ev} & {k for k, _, _ in dep}
        ev = [x for x in ev if x[0] in keep]
        dep = [x for x in dep if x[0] in keep]
    half = None if n is None else max(1, n // 2)
    if half is not None:
        if arm == "paired":
            import numpy as np
            keys = sorted({k for k, _, _ in ev})
            pick = set(np.asarray(keys)[
                np.random.default_rng(seed).permutation(len(keys))[:half]])
            ev = [x for x in ev if x[0] in pick]
            dep = [x for x in dep if x[0] in pick]
        else:
            ev = _take(ev, half, seed)
            dep = _take(dep, half, seed + 1)

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


EA_STOP = set("""a an the this that these those and or but if then than of to
in on at for with by from as is are was were be been being am it its it's his
her their your our my you i we they he she them us me do does did done doing
so such into onto over under about after before during while against between
within without through above below up down out off again further once here
there all any both each few more most other some only own same too very s t
just now not no nor can could should would may might must will shall
https http www com org net""".split())


def _ea_docfreq(texts) -> dict:
    import collections
    c = collections.Counter()
    for t in texts:
        c.update({w for w in _WORD.findall(str(t).lower())
                  if len(w) > 2 and w not in EA_STOP})
    return c


def ea_vocab(tokenizer, k: int = 10, ratio: float = 2.0,
             min_items: int = 10, exclude_ids=()) -> dict:
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
