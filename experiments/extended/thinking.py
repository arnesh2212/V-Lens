from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import numpy as np

LETTERS = "ABCD"
FAMS = ["openimages_orig_auth", "openimages_edit_auth", "vins_orig_auth", "nuscenes_edit_auth"]
SAND = ["wmdp_bio_sand", "wmdp_chem_sand", "wmdp_cybe_sand"]


def load_vread(model_name):
    from vlens.lens.fitall import _load
    cache = os.environ.get("VLENS_MAP_CACHE", "")
    best = None
    for s in glob.glob(os.path.join(cache, "*.pt.spec")):
        try:
            spec = json.loads(open(s).read())
        except Exception:
            continue
        if str(spec[0]).split("/")[-1] == model_name and len(spec[2]) >= 2 and os.path.exists(s[:-5]):
            maps, _lat, _ids, _meta = _load(s[:-5])
            if "v_read" in maps:
                best = {int(l): v for l, v in maps["v_read"].items() if str(l).lstrip("-").isdigit()}
                break
    return best


def run(a):
    import torch
    from scipy.stats import spearmanr
    import vlens
    from vlens import corpus as C
    from vlens.data import simonsays as SS
    from vlens.lens import attribution as AT, readout, scores
    from vlens.lens.hooks import ActivationRecorder
    from g_edits import source_path

    name = a.repo.split("/")[-1]
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    part = os.path.join(ROOT, a.out, f"{name}__j3.jsonl")
    done = {json.loads(l)["id"] for l in open(part)} if os.path.exists(part) else set()
    rng = np.random.default_rng(0)
    recs = []
    for f in FAMS:
        R = [r for r in SS.load(families=[f]) if r["kind"] == "yesno"]
        recs += [R[i] for i in rng.permutation(len(R))[: a.per_family]]
    for f in SAND:
        recs += [r for r in SS.load(families=[f]) if r["kind"] == "letter"]
    if a.limit:
        recs = recs[: a.limit]
    m = vlens.load_vlm(a.repo)
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    vr = load_vread(name)
    print(f"{len(recs)} records; V-Read {'found at ' + str(sorted(vr)) if vr else 'NOT found (logit lens only)'}", flush=True)
    LAST = m.n_layers - 1
    blocks = sorted(vr) if vr else list(range(0, LAST, max(1, LAST // 16)))
    Lw = max(int(round(0.5 * LAST)), 1)
    close_id = tok.convert_tokens_to_ids("</think>")
    letter_ids = torch.tensor([TS.letters[c][0] for c in LETTERS])

    def score_vec(lg, kind, final):
        if kind == "yesno":
            return scores.score_fn(TS, "yesno", None)(None, lg).float()
        lp = torch.log_softmax(lg.float(), -1)[..., letter_ids.to(lg.device)]
        k = LETTERS.index(final) if final in LETTERS else 0
        oth = torch.cat([lp[..., :k], lp[..., k + 1:]], -1).max(-1).values
        return lp[..., k] - oth

    def reads(H, kind, final):
        out = {"logit": [], "v_read": []}
        for l in blocks:
            h = H[l].float()
            with torch.no_grad():
                out["logit"].append(float(score_vec(m.unembed(h[None].to(torch.bfloat16)), kind, final)[0]))
                if vr and l in vr:
                    A, b = vr[l]
                    t = h @ A.float().T.to(h.device) + b.float().to(h.device)
                    out["v_read"].append(float(score_vec(m.unembed(t[None].to(torch.bfloat16)), kind, final)[0]))
        return out

    fh = open(part, "a")
    for i, r in enumerate(recs):
        if r["id"] in done:
            continue
        img = C.load_image(r["image"], square=a.square)
        os.environ["VLENS_THINKING"] = "1"
        b_on = m.chat(r["question"], [img], system=r["system"], cue=r["cue"])
        P = b_on["input_ids"].shape[1]
        with torch.no_grad():
            gen = m.hf.generate(**{k: v for k, v in b_on.items() if not k.startswith("_")},
                                max_new_tokens=a.max_think + 24, do_sample=False)[0, P:].tolist()
        truncated = close_id not in gen[: a.max_think + 1]
        if truncated:
            thought = gen[: a.max_think] + [close_id] + tok.encode("\n\n", add_special_tokens=False)
        else:
            j = gen.index(close_id)
            after = gen[j + 1:]
            k = 0
            while k < len(after) and not tok.decode([after[k]]).strip():
                k += 1
            thought = gen[: j + 1] + after[:k]
        ids = torch.tensor([b_on["input_ids"][0].tolist() + thought], device=m.device)
        b_full = {}
        for kk, vv in b_on.items():
            if (torch.is_tensor(vv) and vv.dim() >= 2 and vv.shape[0] == 1 and vv.shape[1] == P
                    and kk not in ("input_ids", "attention_mask")):
                vv = torch.cat([vv, torch.zeros((1, len(thought)) + tuple(vv.shape[2:]), dtype=vv.dtype, device=vv.device)], 1)
            b_full[kk] = vv
        b_full["input_ids"] = ids
        b_full["attention_mask"] = torch.ones_like(ids)
        S = ids.shape[1]
        with torch.no_grad(), ActivationRecorder(m.layers, at=sorted(set(blocks) | {LAST})) as rec:
            m.forward(b_full)
            Hpre = {l: rec.activations[l][0, P - 1] for l in blocks}
            Hpost = {l: rec.activations[l][0, S - 1] for l in blocks}
            lg_post = m.unembed(rec.activations[LAST][0, S - 1][None])
        if r["kind"] == "yesno":
            final = "yes" if float(scores.score_fn(TS, "yesno", None)(None, lg_post)[0]) > 0 else "no"
        else:
            final = LETTERS[int(torch.log_softmax(lg_post.float(), -1)[0, letter_ids.to(lg_post.device)].argmax())]
        row = dict(id=r["id"], family=r["family"], kind=r["kind"], subcategory=r["subcategory"], final=final,
                   correct=r.get("expected"), think_tokens=len(thought), truncated=truncated,
                   thought=tok.decode(thought)[:400], pre=reads(Hpre, r["kind"], final), post=reads(Hpost, r["kind"], final),
                   blocks=blocks, n_layers=m.n_layers)
        os.environ["VLENS_THINKING"] = "0"
        b_off = m.chat(r["question"], [img], system=r["system"], cue=r["cue"])
        with torch.no_grad(), ActivationRecorder(m.layers, at=[LAST]) as rec:
            m.forward(b_off)
            lg_off = m.unembed(rec.activations[LAST][0, b_off["input_ids"].shape[1] - 1][None])
        if r["kind"] == "yesno":
            row["final_off"] = "yes" if float(scores.score_fn(TS, "yesno", None)(None, lg_off)[0]) > 0 else "no"
        else:
            row["final_off"] = LETTERS[int(torch.log_softmax(lg_off.float(), -1)[0, letter_ids.to(lg_off.device)].argmax())]
        if a.vwrite:
            maps = {}
            for tag, bb in (("think", b_full), ("off", b_off)):
                pos = readout.image_positions(m, bb)
                fn = lambda hL, lg, kk=r["kind"], ff=final: score_vec(lg, kk, ff)
                with torch.no_grad():
                    _, H, _, _ = AT.states_at(m, bb, [Lw])
                h = H[Lw]; mu = h.mean(0, keepdim=True).expand_as(h)
                try:
                    with AT.SkipTo(m, bb, Lw, fn=fn, pos=pos, h=h, base=mu) as bh:
                        maps[tag] = np.asarray(AT.path_reading(m, bh, fn, Lw, pos, h, mu, steps=8), np.float32)
                except Exception as e:
                    row["vwrite_error"] = f"{type(e).__name__}: {e}"[:200]
            if len(maps) == 2 and maps["think"].shape == maps["off"].shape:
                row["vwrite_spearman"] = float(spearmanr(maps["think"], maps["off"]).correlation)
                sp = source_path(r) if r["variant"] == "edited" else None
                if sp:
                    G = int(round(len(maps["off"]) ** 0.5))
                    cov = AT.pool_mask(AT.edit_mask(sp, r["image"], side=a.square), G); on = cov > 0.1
                    if on.any() and not on.all():
                        for tag, w in maps.items():
                            pw = np.clip(w, 0, None)
                            row[f"edit_share_{tag}"] = float(pw[on].sum() / max(pw.sum(), 1e-12))
                        row["edit_area"] = float(on.mean())
        fh.write(json.dumps(row) + "\n"); fh.flush()
        if i % 20 == 0:
            print(f"  {i}/{len(recs)} {r['id']} final {final} off {row['final_off']} think {len(thought)} tok"
                  f"{' (truncated)' if truncated else ''}", flush=True)
    fh.close()
    print("DONE j3 run", flush=True)


def summarise(a):
    from vlens.lens.metrics import auroc
    name = a.model
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.out, f"{name}__j3.jsonl"))]
    if a.completed_only:
        rows = [r for r in rows if not r["truncated"]]
    L = [f"# J3 reasoning mode, {name}" + (" (completed thoughts only)" if a.completed_only else ""), "",
         f"{len(rows)} records; thought capped at the run's --max-think tokens; "
         f"{np.mean([r['truncated'] for r in rows]):.2f} of thoughts did not close and were closed by force.", ""]
    for kind, R in (("yesno", [r for r in rows if r["kind"] == "yesno"]), ("letter", [r for r in rows if r["kind"] == "letter"])):
        if not R:
            continue
        agree = np.mean([r["final"] == r["final_off"] for r in R])
        acc_on = np.mean([str(r["final"]).lower() == str(r["correct"]).lower() for r in R if r.get("correct")]) if any(r.get("correct") for r in R) else np.nan
        acc_off = np.mean([str(r["final_off"]).lower() == str(r["correct"]).lower() for r in R if r.get("correct")]) if any(r.get("correct") for r in R) else np.nan
        L += [f"## {kind} (n = {len(R)})", "", f"answer with thinking = answer without: {agree:.2f}; accuracy with / without thinking: "
              f"{acc_on:.2f} / {acc_off:.2f}; median thought {np.median([r['think_tokens'] for r in R]):.0f} tokens", ""]
        blocks = R[0]["blocks"]; nl = R[0]["n_layers"]
        L += ["| depth | " + " | ".join(f"{b / max(nl - 1, 1):.2f}" for b in blocks) + " |", "|---|" + "---|" * len(blocks)]
        for where in ("pre", "post"):
            for lens in ("v_read", "logit"):
                vals = []
                for bi in range(len(blocks)):
                    xs = [r[where][lens][bi] for r in R if len(r[where][lens]) > bi]
                    if kind == "yesno":
                        y = [r["final"] == "yes" for r in R if len(r[where][lens]) > bi]
                        vals.append(auroc(np.asarray(y, int), np.asarray(xs)) if 0 < sum(y) < len(y) else np.nan)
                    else:
                        vals.append(float(np.mean(np.asarray(xs) > 0)))
                L.append(f"| {where} {lens} | " + " | ".join("-" if not np.isfinite(v) else f"{v:.2f}" for v in vals) + " |")
        L.append("")
        L.append("(yes/no: AUROC of the reading for the final answer; letters: share of records where the final letter "
                 "already leads, chance about 0.25)")
        sp = [r["vwrite_spearman"] for r in R if "vwrite_spearman" in r]
        es = [(r["edit_share_think"], r["edit_share_off"], r["edit_area"]) for r in R if "edit_share_think" in r]
        if sp:
            L.append(f"V-Write map with vs without the thought: Spearman median {np.median(sp):.2f} (n = {len(sp)})")
        if es:
            e = np.asarray(es)
            L.append(f"share on the edit mask: with thought {e[:, 0].mean():.2f}, without {e[:, 1].mean():.2f}, area {e[:, 2].mean():.2f} (n = {len(e)})")
        L.append("")
    open(os.path.join(ROOT, a.out, f"{name}__j3_summary{'_completed' if a.completed_only else ''}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--repo", default="Qwen/Qwen3.5-9B"); r.add_argument("--out", default="results/extended/j3")
    r.add_argument("--per-family", type=int, default=60); r.add_argument("--max-think", type=int, default=256)
    r.add_argument("--square", type=int, default=448); r.add_argument("--limit", type=int, default=0)
    r.add_argument("--no-vwrite", dest="vwrite", action="store_false", default=True)
    s = sub.add_parser("summarise"); s.add_argument("--model", default="Qwen3.5-9B"); s.add_argument("--out", default="results/extended/j3")
    s.add_argument("--completed-only", action="store_true")
    a = ap.parse_args()
    {"run": run, "summarise": summarise}[a.cmd](a)


if __name__ == "__main__":
    main()
