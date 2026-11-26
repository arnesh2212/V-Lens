from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
import numpy as np
import torch


def source_path(r):
    img = r["image"]
    src = (r.get("meta") or {}).get("source_image")
    if src:
        base = os.path.dirname(os.path.dirname(os.path.dirname(img)))
        for cand in (os.path.join(base, src), os.path.join(base, "Original", "Images", os.path.basename(src))):
            if os.path.exists(cand):
                return cand
        return None
    if "Gemini Dataset" in img:
        stem = os.path.basename(img).split("_")[0]
        root = img.split("Gemini Dataset")[0] + "Gemini Dataset"
        for c in glob.glob(os.path.join(root, "Original*", f"{stem}.*")) + glob.glob(os.path.join(root, "*", "*", f"{stem}.*")):
            if "Edited" not in c:
                return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="results/extended/g_edits")
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--diff-thr", type=int, default=28)
    ap.add_argument("--min-frac", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import vlens
    from vlens import corpus
    from vlens.data import simonsays as S
    from vlens.lens import attribution, depth, scores

    recs = [r for r in S.load() if r["variant"] == "edited" and r["kind"] in ("yesno", "test_real")]
    cases = []
    for r in recs:
        sp = source_path(r)
        if sp and os.path.exists(r["image"]):
            cases.append((r, sp))
    if a.limit:
        cases = cases[: a.limit]
    say(f"{len(cases)} edited short-answer records with a source frame (of {len(recs)})")
    name = a.repo.split("/")[-1]
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    allL = list(range(max(m.n_deepstack, 1), m.n_layers - 1))
    l = min(allL, key=lambda x: abs(depth.to_frac(m, x) - a.depth))
    say(f"{name}: V-Write at block {l} ({depth.to_frac(m, l):.2f} of the depth)")
    os.makedirs(a.out, exist_ok=True)
    outp = os.path.join(a.out, f"{name}__g_edits.json")
    out, t0 = [], time.time()
    for ci, (r, sp) in enumerate(cases):
        try:
            fn = scores.score_fn(TS, r["kind"], r.get("expected"))
            img_e = corpus.load_image(r["image"], square=a.square)
            img_s = corpus.load_image(sp, square=a.square)
            b_e = m.chat(r["question"], [img_e], system=r.get("system"), cue=r.get("cue"))
            b_s = m.chat(r["question"], [img_s], system=r.get("system"), cue=r.get("cue"))
            pos, He, hLe, lge = attribution.states_at(m, b_e, [l])
            pos_s, Hs, hLs, lgs = attribution.states_at(m, b_s, [l])
            if len(pos) != len(pos_s):
                say(f"  skip {r['id']}: token counts differ"); continue
            P = len(pos); G = int(round(math.sqrt(P)))
            s_e = float(fn(hLe[None], lge).float().cpu()[0]); s_s = float(fn(hLs[None], lgs).float().cpu()[0])
            M = attribution.pool_mask(attribution.edit_mask(sp, r["image"], thr=a.diff_thr), G)
            inm = (M >= a.min_frac).reshape(-1)[:P] if M.size >= P else np.zeros(P, bool)
            meta = r.get("meta") or {}
            rec = dict(id=r["id"], family=r["family"], edit_family=meta.get("edit_family"),
                       edit_variant=meta.get("edit_variant") or meta.get("visual_condition"),
                       subcategory=r["subcategory"], condition=r["condition"], kind=r["kind"],
                       expected=r.get("expected"), score_src=s_s, score_edit=s_e,
                       mask_area=float(inm.mean()), n_mask=int(inm.sum()))
            if inm.sum() > 0:
                h = He[l]; hs = Hs[l]
                mu = h.mean(0, keepdim=True).expand_as(h).clone()
                p_mu = attribution.path_reading(m, b_e, fn, l, pos, h, mu, a.steps)
                p_src = attribution.path_reading(m, b_e, fn, l, pos, h, hs, a.steps)
                with torch.no_grad():
                    X1 = h.clone(); X1[torch.as_tensor(inm)] = hs[torch.as_tensor(inm)]
                    X2 = h.clone(); X2[torch.as_tensor(~inm)] = hs[torch.as_tensor(~inm)]
                    v1 = float(attribution.hooked_axis(m, b_e, fn, l, pos, X1[None]).float().cpu()[0])
                    v2 = float(attribution.hooked_axis(m, b_e, fn, l, pos, X2[None]).float().cpu()[0])

                def share(v):
                    av = np.abs(np.asarray(v, float)); return float(av[inm].sum() / max(av.sum(), 1e-9))
                rec.update(share_mu=share(p_mu), share_src=share(p_src),
                           top1_mu=bool(inm[int(np.argmax(np.abs(p_mu)))]),
                           top1_src=bool(inm[int(np.argmax(np.abs(p_src)))]),
                           unedit_mask=v1 - s_e, unedit_rest=v2 - s_e,
                           pred_unedit_mask=float(-np.asarray(p_src)[inm].sum()),
                           pred_unedit_rest=float(-np.asarray(p_src)[~inm].sum()))
            out.append(rec)
            if ci % 10 == 0:
                say(f"  {ci + 1}/{len(cases)} {r['id']:<26} {str(rec['edit_family']):<18} src {s_s:+.2f} edit {s_e:+.2f} "
                    f"area {rec['mask_area']:.2f} share {rec.get('share_mu', float('nan')):.2f}/{rec.get('share_src', float('nan')):.2f} "
                    f"({(time.time() - t0) / 60:.1f} min)")
        except Exception as e:
            say(f"  fail {r['id']}: {type(e).__name__}: {str(e)[:120]}")
        if ci % 50 == 0:
            json.dump(dict(repo=a.repo, block=l, cases=out), open(outp, "w"), default=float)
    json.dump(dict(repo=a.repo, block=l, cases=out), open(outp, "w"), default=float)
    say(f"wrote {outp} ({len(out)} records)")
    say("DONE g_edits")


if __name__ == "__main__":
    main()
