# SPDX-License-Identifier: Apache-2.0
"""Everything that has finished, in one page of numbers.

Reads whatever result files exist and prints a digest: which models have which
experiments, and the headline number from each. Safe to run at any time; it
reports what is missing rather than failing.

    python3 tools/digest.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ORDER = ["v_read", "tuned", "jacobian", "latent", "logit", "probe"]


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def at(depths, vals, x):
    if not depths:
        return float("nan")
    return vals[int(np.argmin(np.abs(np.array(depths, float) - x)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--into", default=None,
                    help="a markdown file whose Status section this replaces")
    a = ap.parse_args()
    R = a.results
    lines: list[str] = []
    say = lines.append if a.into else print

    say("=" * 78)
    say("BEHAVIOUR")
    for pat, lab in ((f"{R}/simonsays/*__*.json", "capped"),
                     (f"{R}/simonsays_full/*__*.json", "full")):
        tot = {}
        for p in glob.glob(pat):
            d = load(p)
            # a behaviour file has rows; a lens file has AUROC curves.  Filtering
            # by name would drop the reading-binding families, whose names end in
            # "_read" exactly like the lens output does.
            if d and "n" in d and "rows" in d and "auroc" not in d:
                tot.setdefault(d.get("repo", "?").split("/")[-1], [0, 0])
                tot[d["repo"].split("/")[-1]][0] += d["n"]
                tot[d["repo"].split("/")[-1]][1] += 1
        for k, (n, f) in sorted(tot.items()):
            say(f"  {lab:<7} {k:<28} {n:>6} rows over {f} families")

    say("=" * 78)
    say("LENS BENCHMARK  (KL to the model, lower is better; mean over fitted blocks)")
    for p in sorted(glob.glob(f"{R}/lensbench*/*__bench.json")):
        d = load(p)
        if not d:
            continue
        tag = os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p).split("__")[0]
        say(f"  {tag}")
        kl = d.get("kl", {})
        if kl:
            say("    " + "  ".join(f"{k} {np.mean(v):.2f}" for k, v in sorted(kl.items())))
        mt = d.get("meta", {})
        tu = mt.get("tuned") or {}
        say(f"    budget actually spent: tuned {tu.get('steps')} steps x "
            f"{tu.get('tokens_per_step')} tokens, lr {tu.get('lr_chosen')}, "
            f"weight decay {tu.get('wd_chosen')}")
        say(f"    chosen: lambda {(mt.get('v_read') or {}).get('lam')}, "
            f"latent temp {(mt.get('latent') or {}).get('temp')}, "
            f"caption store {(mt.get('latent') or {}).get('n_captions')}")
        fams = d.get("families", {})
        if fams:
            say("    AUROC for the model's own answer, at a quarter depth:")
            agg = {}
            for fam, r in fams.items():
                for k in ORDER:
                    key = f"model|{k}"
                    if key in r["auroc"]:
                        agg.setdefault(k, []).append(at(r["depths"], r["auroc"][key], 0.25))
            say("      " + "  ".join(f"{k} {np.nanmean(v):.3f}" for k, v in agg.items()))
            say(f"      over {len(fams)} families, "
                f"{sum(r['n'] for r in fams.values())} scored items")

    say("=" * 78)
    say("LENS, CROSS-FITTED WITHIN FAMILY")
    for p in sorted(glob.glob(f"{R}/lensread*/*__read_summary.json")):
        d = load(p)
        if not d:
            continue
        agg = {}
        for fam, r in (d.get("families") or {}).items():
            for k in ORDER:
                key = f"model|{k}"
                if key in r.get("auroc", {}):
                    agg.setdefault(k, []).append(at(r["depths"], r["auroc"][key], 0.25))
        if agg:
            say(f"  {os.path.basename(p).split('__')[0]:<28} " +
                "  ".join(f"{k} {np.nanmean(v):.3f}" for k, v in agg.items()))

    say("=" * 78)
    say("EVERY INSTRUMENT RANKING IMAGE PATCHES  (patches replaced by zero)")
    for p in sorted(glob.glob(f"{R}/patchlenses/*__patchlenses.json")):
        d = load(p)
        if not d:
            continue
        s_ = d["summary"]
        ks = s_["ks"]
        j = int(np.argmin(np.abs(np.array(ks) - 0.2)))
        rank = [r for r in s_["rankings"] if f"zero:{r}" in s_.get("drop", {})]
        say(f"  {os.path.basename(p).split('__')[0]:<28} n={s_['n']:<5} block {s_.get('block')}")
        say("    " + "".join(f"{int(k * 100):>8}%" for k in ks) + "   fall in the answer score")
        for mm in rank:
            say(f"    {mm:<15}" + "".join(f"{v:>9.2f}" for v in s_["drop"][f"zero:{mm}"]))
        g = s_.get("graded_against_patching") or {}
        if g:
            say("    rank agreement with what covering each part of the picture does")
            for mm in rank:
                v = g.get(f"zero|{mm}")
                if v is not None and np.isfinite(v):
                    say(f"      {mm:<15}{v:>7.3f}")
        fam = s_.get("per_family") or {}
        if fam and rank:
            base = "v_write" if "v_write" in rank else rank[0]
            rivals = [r for r in rank if r not in (base, "random")]
            say(f"    families where {base} leads, at {int(100 * ks[j])} per cent covered "
                f"({len(fam)} families)")
            for r in rivals:
                won = sum(1 for f in fam if np.isfinite(fam[f].get(base, np.nan))
                          and np.isfinite(fam[f].get(r, np.nan))
                          and fam[f][base] > fam[f][r])
                ok = sum(1 for f in fam if np.isfinite(fam[f].get(base, np.nan))
                         and np.isfinite(fam[f].get(r, np.nan)))
                say(f"      against {r:<15} {won}/{ok}")

    say("=" * 78)
    say("CAUSAL PATCH VALIDATION  (fall in the answer score at 20 per cent replaced)")
    for p in sorted(glob.glob(f"{R}/patchcausal/*__patchcausal.json")):
        d = load(p)
        if not d:
            continue
        s = d["summary"]
        j = int(np.argmin(np.abs(np.array(s["ks"]) - 0.2)))
        say(f"  {os.path.basename(p).split('__')[0]:<28} n={s['n_items']:<4} " +
            "  ".join(f"{g} {s['curve'][g][j]:.2f}" for g in ("top", "middle", "random", "bottom")
                      if g in s["curve"]))
        ag = s.get("agreement") or {}
        # baselines that keep the state near the data (the image's own mean, a
        # blurred image, another image) are comparable to each other; zero and
        # gaussian noise put it somewhere the model never goes, so their
        # rankings are reported apart rather than averaged in
        ON = ("vwrite_mean", "vwrite_blur", "vwrite_other")
        OFF = ("vwrite_zero", "vwrite_noise")
        near = [v for k, v in ag.items()
                if all(p_ in ON for p_ in k.split("|"))]
        far = [v for k, v in ag.items()
               if any(p_ in OFF for p_ in k.split("|")) and "random" not in k]
        rnd = [v for k, v in ag.items() if "random" in k]
        if near:
            say(f"    rank agreement, baselines that stay on the manifold: {np.nanmean(near):.2f}")
        if far:
            say(f"    rank agreement, zero and noise against the rest:     {np.nanmean(far):.2f}")
        if rnd:
            say(f"    rank agreement with a random ranking:                {np.nanmean(rnd):.2f}")
        for k in ("top:vwrite_mean", "top:vwrite_blur", "top:vwrite_other",
                  "top:vwrite_noise", "top:vwrite_zero", "top:gradxact", "top:random"):
            if k in s.get("curve", {}):
                say(f"      {k.replace('top:', ''):<14} " +
                    " ".join(f"{v:6.2f}" for v in s["curve"][k]))

    say("=" * 78)
    say("OCCLUSION IN THE PICTURE, AND THE MINIMAL REGION")
    for p in sorted(glob.glob(f"{R}/occlusion/*__occlusion.json")):
        d = load(p)
        if not d:
            continue
        s = d["summary"]
        j = int(np.argmin(np.abs(np.array(s["ks"]) - 0.2)))
        say(f"  {os.path.basename(p).split('__')[0]:<28} n={s['n']}")
        for how in s["methods"]:
            say(f"    {how:<12} top {s['a3'][f'{how}|top'][j]:>6.2f}   "
                f"random {s['a3'][f'{how}|random'][j]:>6.2f}")
        say(f"    minimal region {s['minimal_fraction'] * 100:.1f}% of the picture; "
            f"precision {s['precision']:.2f} recall {s['recall']:.2f} F1 {s['f1']:.2f}")

    say("=" * 78)
    say("SAFETY")
    for p in sorted(glob.glob(f"{R}/safety*/*__refusal_auroc.json")):
        d = load(p)
        if not d:
            continue
        say(f"  {os.path.basename(p).split('__')[0]:<28} n={d['n']}")
        for t in ("harmful", "refused"):
            row = {k: at(d["depths"], d["auroc"][f"{t}|{k}"], 0.25)
                   for k in ORDER if f"{t}|{k}" in d["auroc"]}
            if row:
                say(f"    {t:<9} at a quarter depth: " +
                    "  ".join(f"{k} {v:.2f}" for k, v in row.items()))
    for p in sorted(glob.glob(f"{R}/safety*/*__refusal_steer.json")):
        d = load(p)
        if not d:
            continue
        b = d.get("baseline", {})
        say(f"  steering, baseline refusal harmful {b.get('harmful', float('nan')):.2f} "
            f"benign {b.get('benign', float('nan')):.2f}")
        best = {}
        for r in d["results"]:
            if r.get("direction") in (None, "none"):
                continue
            k = r["direction"]
            cur = best.get(k)
            if cur is None or r["refusal_harmful"] < cur["refusal_harmful"]:
                best[k] = r
        for k, r in sorted(best.items()):
            say(f"    {k:<22} best ablation {r['refusal_harmful']:.2f} at alpha {r['alpha']}, "
                f"over-refusal added {r['refusal_benign']:.2f}, coherent {r['coherent_h']:.2f}")
    for p in sorted(glob.glob(f"{R}/safety*/*__refusal_patch.json")):
        d = load(p)
        if not d or not d.get("drop"):
            continue
        j = int(np.argmin(np.abs(np.array(d["ks"]) - 0.2)))
        say("    patches on the jailbreak image, 20 per cent replaced: " +
            "  ".join(f"{g} {d['drop'][g][j]:.2f}" for g in d["drop"]))

    say("=" * 78)
    say("HANDOFF OF A VISUAL FACT")
    for p in sorted(glob.glob(f"{R}/handoff/*__handoff.json")):
        d = load(p)
        if not d:
            continue
        s = d["summary"]
        say(f"  {os.path.basename(p).split('__')[0]:<28} n={s['n']}")
        say(f"    edited answer recovered {s['recovered']:.2f} of the way; "
            f"a control region of the same size {s['recovered_control']:.2f}")
        say(f"    answer flipped: edit {s['flipped']:.2f}, control {s['flipped_control']:.2f}, "
            f"noise {s['flipped_noise']:.2f}")

    say("=" * 78)
    say("PASSWORD LOCKING")
    for p in sorted(glob.glob(f"{R}/pwlock/*__pwlock.json")):
        d = load(p)
        if not d:
            continue
        st = d.get("stages", {})
        say(f"  {os.path.basename(p).split('__pwlock')[0]}")
        if "E1" in st:
            e = st["E1"]
            say(f"    clean {e['clean']['accuracy']:.3f} [{e['clean']['lo']:.3f},{e['clean']['hi']:.3f}]  "
                f"marked {e['marked']['accuracy']:.3f} [{e['marked']['lo']:.3f},{e['marked']['hi']:.3f}]  "
                f"designated wrong letter: marked {e['sandbag_rate_marked']:.3f} clean {e['sandbag_rate_clean']:.3f}")
        if "E2" in st:
            e = st["E2"]
            say(f"    designs seen in training {e['seen']['accuracy']:.3f}, never seen {e['unseen']['accuracy']:.3f}")
        if "E7" in st:
            e = st["E7"]
            for t in ("marked", "sandbagged"):
                row = {k: at(e["depths"], e["auroc"][f"{t}|{k}"], 0.25)
                       for k in ORDER if f"{t}|{k}" in e["auroc"]}
                if row:
                    say(f"    {t:<11} at a quarter depth: " +
                        "  ".join(f"{k} {v:.2f}" for k, v in row.items()))
        if "E8" in st and st["E8"].get("drop"):
            e = st["E8"]
            j = int(np.argmin(np.abs(np.array(e["ks"]) - 0.2)))
            say("    patches, 20 per cent replaced: " +
                "  ".join(f"{g} {e['drop'][g][j]:.2f}" for g in e["drop"]) +
                (f";  top tenth landing on the mark {e['overlap_with_mark']:.2f}"
                 if e.get("overlap_with_mark") is not None else ""))
    say("=" * 78)
    if a.into:
        body = "\n".join(lines)
        print(body)
        if os.path.exists(a.into):
            txt = open(a.into).read()
            head = "## Status\n"
            if head in txt:
                keep = txt[: txt.index(head) + len(head)]
                note = ("\nRegenerated by `tools/digest.py --into` from the files in "
                        "`results/`. Nothing here is typed by hand; an empty section is "
                        "an experiment that has not landed yet, and `tools/progress.py` "
                        "says where it is in the queue.\n\n")
                open(a.into, "w").write(keep + note + body + "\n")
                print(f"\nwrote the Status section of {a.into}")


if __name__ == "__main__":
    sys.exit(main())
