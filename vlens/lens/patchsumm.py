# SPDX-License-Identifier: Apache-2.0
"""Turn per-record patch-ranking rows into the table the paper reports.

Kept apart from the experiment so a finished run can be summarised again from
its saved rows without touching the card, and so the tool that does that and
the experiment itself cannot drift.

Every ranking is measured against the random ranking under the *same*
replacement: a drop measured after replacing patches with zero means nothing
beside one measured after replacing them with the mean.

The password study's own two families never reach the table. Its fine-tuning
half is what the lock is trained on, so a general result computed over it is
computed partly over training data, and its test half is that study's
counterpart to it. The guard lives here rather than only in the samplers so
that a run made before the split, summarised again from its rows, comes out
on the same footing as one made after.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def key_for(tag: str, mm: str) -> str:
    return mm if tag == "mean" else f"zero:{mm}"


def summarise(rows: Sequence[dict], ks: Sequence[float], rankings: Sequence[str],
              **head) -> dict:
    """``{drop, flip, ratio, over_random, agreement, per_family, ...}`` from rows."""
    from vlens.data import simonsays as SS
    rows = [x for x in rows if not SS.is_password_locking(x)]
    mean = lambda f: float(np.nanmean([f(x) for x in rows]))
    summ = dict(**head, n=len(rows), ks=list(ks), rankings=list(rankings),
                drop={}, ratio={}, flip={}, agreement={})
    # how many records each ranking was measured on. A run resumed with more
    # rankings than it started with measures the new ones only on the records
    # still to come, and a comparison between two rankings is a comparison
    # only where these two numbers are the same
    summ["n_ranking"] = {mm: sum(1 for x in rows if f"zero|{mm}|{ks[0]}" in x["after"]
                                 or f"mean|{mm}|{ks[0]}" in x["after"])
                         for mm in rankings}
    for mm in rankings:
        if any(mm in x.get("agree", {}) for x in rows):     # V-Write is the reference
            summ["agreement"][mm] = mean(lambda x, mm=mm: x.get("agree", {}).get(mm, np.nan))
    for tag in ("mean", "zero"):
        for mm in rankings:
            if not any(f"{tag}|{mm}|{ks[0]}" in x["after"] for x in rows):
                continue
            key = key_for(tag, mm)
            summ["drop"][key] = [mean(lambda x, k=k, mm=mm, t=tag:
                                      x["s0"] - x["after"].get(f"{t}|{mm}|{k}", np.nan)) for k in ks]
            summ["flip"][key] = [mean(lambda x, k=k, mm=mm, t=tag:
                                      float(x["after"].get(f"{t}|{mm}|{k}", np.nan) <= 0)) for k in ks]
    summ["graded_against_patching"] = {
        f"{tag}|{mm}": mean(lambda x, k=f"{tag}|{mm}": x.get("truth_agreement", {}).get(k, np.nan))
        for tag in ("mean", "zero") for mm in rankings
        if any(f"{tag}|{mm}" in x.get("truth_agreement", {}) for x in rows)}
    for tag in ("mean", "zero"):
        base = summ["drop"].get(key_for(tag, "random"))
        if not base:
            continue
        for mm in rankings:
            key = key_for(tag, mm)
            if key not in summ["drop"]:
                continue
            # the ratio is reported only where the random baseline is a real
            # positive fall, since a multiple of a negative number says
            # nothing; the difference is reported always
            summ["ratio"][key] = [(summ["drop"][key][j] / base[j]) if base[j] > 0.05
                                  else float("nan") for j in range(len(ks))]
            summ.setdefault("over_random", {})[key] = [summ["drop"][key][j] - base[j]
                                                       for j in range(len(ks))]
    # the paired table across families, on the twenty per cent column, under
    # the zero replacement, which V-Write was not built around
    j20 = int(np.argmin(np.abs(np.array(ks) - 0.2)))
    fam = {}
    for x in rows:
        fam.setdefault(x["family"], []).append(x)
    drop_of = lambda xs, mm: float(np.nanmean(
        [y["s0"] - y["after"].get(f"zero|{mm}|{ks[j20]}", np.nan) for y in xs]))
    summ["per_family"] = {f: {mm: drop_of(xs, mm) for mm in rankings}
                          for f, xs in fam.items() if len(xs) >= 12}
    summ["per_family_column"] = float(ks[j20])
    summ["per_source_column"] = float(ks[j20])
    summ["per_source_index"] = j20
    # the same table by source dataset. A family can be too thin to average
    # on its own while its dataset is not, and five went into this benchmark,
    # so the paper reports whether the result holds on each of the five.
    src = {}
    for x in rows:
        src.setdefault(str(x["family"]).split("_")[0], []).append(x)
    curve_of = lambda xs, mm: [float(np.nanmean(
        [y["s0"] - y["after"].get(f"zero|{mm}|{k}", np.nan) for y in xs])) for k in ks]
    summ["per_source"] = {f: {mm: curve_of(xs, mm) for mm in rankings}
                          for f, xs in src.items() if len(xs) >= 12}
    summ["per_source_n"] = {f: len(xs) for f, xs in src.items()}
    return summ


def report(summ: dict, ks: Sequence[float], rankings: Sequence[str], say: Callable = print) -> None:
    """Print the drop tables, both replacements, then the zero-replaced margins."""
    counts = {mm: c for mm, c in (summ.get("n_ranking") or {}).items() if c}
    if len(set(counts.values())) > 1:
        say("\nrankings measured on different numbers of records, so compare them with care:")
        say("  " + ", ".join(f"{mm} {c}" for mm, c in counts.items()))
    for tag, lab in (("zero", "replaced by zero, which V-Write does not integrate from"),
                     ("mean", "replaced by the mean, which V-Write integrates from")):
        if not any(key_for(tag, mm) in summ["drop"] for mm in rankings):
            continue
        say(f"\nfall in the answer score, patches {lab} ({summ['n']} records)")
        say("  ranking        " + "".join(f"{int(k * 100):>8}%" for k in ks))
        for mm in rankings:
            key = key_for(tag, mm)
            if key in summ["drop"]:
                say(f"  {mm:<14}" + "".join(f"{v:>9.2f}" for v in summ["drop"][key]))
    for field, lab, fmt in (("over_random", "with the random ranking subtracted", "{:>9.2f}"),
                            ("ratio", "as a multiple of random, where it is big enough to divide by",
                             "{:>9.1f}")):
        if not summ.get(field):
            continue
        say(f"\n  {lab}, patches replaced by zero")
        for mm in rankings:
            key = key_for("zero", mm)
            if key in summ[field] and mm != "random":
                say(f"  {mm:<14}" + "".join(("      nan" if not np.isfinite(v) else fmt.format(v))
                                            for v in summ[field][key]))


def report_sources(summ: dict, rankings: Sequence[str], say: Callable = print) -> None:
    """The twenty-per-cent column split by the dataset each record came from."""
    per = summ.get("per_source") or {}
    if not per:
        return
    ns = summ.get("per_source_n") or {}
    srcs = sorted(per)
    say(f"\nfall in the answer score at {int(summ.get('per_source_column', 0.2) * 100)}%, "
        "by the dataset the picture came from")
    say("  ranking        " + "".join(f"{s[:9]:>11}" for s in srcs))
    say("  records        " + "".join(f"{ns.get(s, 0):>11}" for s in srcs))
    j = summ.get("per_source_index", 0)
    at = lambda s_, mm: (per[s_].get(mm) or [float("nan")])[j]
    for mm in rankings:
        if any(np.isfinite(at(s, mm)) for s in srcs):
            say(f"  {mm:<14}" + "".join(f"{at(s, mm):>11.2f}" for s in srcs))
