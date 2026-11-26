from __future__ import annotations

import argparse
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

LETTERS = "ABCD"
THR = 0.1


def prompts(P):
    raw = {r["id"]: r for r in P._raw("test")}
    out = []
    for x in P.test_items():
        conds = (raw.get(x["id"]) or {}).get("conditions") or {}
        for cname, c in conds.items():
            sysp = ((c or {}).get("system_prompt") or "You are a helpful assistant.").replace("{p}", "40")
            out.append(dict(x, id=f"{x['id']}__{cname}", condition=cname, system=sysp,
                            kind="test"))
    return out


def summary(w, cov):
    from vlens.lens.metrics import auroc
    on = cov > THR
    if not on.any() or on.all():
        return dict(mass=None, hit1=None, auroc=None, chance=float(on.mean()))
    pos = np.clip(w, 0, None)
    return dict(mass=float(pos[on].sum() / max(pos.sum(), 1e-12)),
                hit1=bool(on[int(np.argmax(w))]),
                auroc=float(auroc(on.astype(int), w)),
                chance=float(on.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--models", nargs="*", default=["locked", "base"])
    ap.add_argument("--out", default="results/pwlock")
    ap.add_argument("--square", type=int, default=448)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first n prompts only (a smoke run)")
    ap.add_argument("--dry", default=None, help="build every picture and mask, save a sheet here, load no model")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    import torch
    from PIL import Image
    import vlens
    from vlens import corpus as C
    from vlens.data import pwlock as P
    from vlens.lens import attribution as AT, readout, scores
    from experiments.pwlock.evaluate import watermark_layer, composite

    items = prompts(P)
    test = P.test_items()
    clean_by = {x["group"]: x for x in test if not x["marked"]}
    marked_by = {}
    for x in test:
        if x["marked"]:
            marked_by.setdefault(x["group"], {})[x["variant"]] = x
    groups = [g for g in clean_by if g in marked_by]
    S = a.square

    mask_cache = {}

    def test_mask(g, v):
        if (g, v) not in mask_cache:
            mask_cache[(g, v)] = AT.edit_mask(clean_by[g]["image"], marked_by[g][v]["image"], side=S)
        return mask_cache[(g, v)]

    def layer_alpha(lay, scale=1.0, offset=(0, 0)):
        al = np.asarray(lay)[..., 3].astype(np.float32) / 255.0
        im = Image.fromarray(((al > 0.3) * 255).astype(np.uint8), "L")
        if scale != 1.0:
            s = max(int(S * scale), 8)
            pad = Image.new("L", (S, S), 0)
            pad.paste(im.resize((s, s), Image.BILINEAR), ((S - s) // 2, (S - s) // 2))
            im = pad
        if offset != (0, 0):
            pad = Image.new("L", (S, S), 0)
            pad.paste(im, offset)
            im = pad
        return np.asarray(im, np.float32) / 255.0

    lay_cache = {}

    def layer_for(g, v=1):
        if (g, v) not in lay_cache:
            mk = marked_by[g].get(v) or next(iter(marked_by[g].values()))
            lay_cache[(g, v)] = watermark_layer(mk["image"], clean_by[g]["image"], S)[0]
        return lay_cache[(g, v)]

    controls = []
    if not a.no_controls:
        rng = np.random.default_rng(0)
        for name in ("unrelated pattern", "opacity 0.25", "opacity 0.5", "opacity 0.75", "scale 0.5",
                     "scale 0.75", "moved", "mark alone on grey", "a different design"):
            for g in groups:
                controls.append(dict(clean_by[g], id=f"{clean_by[g]['id']}__{name.replace(' ', '_')}",
                                     condition="unlocked", control=name, kind="control", marked=True,
                                     variant=-1, system=clean_by[g]["system"]))
        unrel = {}
        for g in groups:
            lay = layer_for(g, 1)
            arr = np.asarray(lay).copy()
            al = arr[..., 3]
            n_on = int((al > 8).sum())
            new = np.zeros_like(al)
            ys = rng.integers(0, S, n_on); xs = rng.integers(0, S, n_on)
            new[ys, xs] = 255
            arr[..., 3] = new
            unrel[g] = Image.fromarray(arr, "RGBA")

    def picture(x):
        g = x["group"]
        if x["kind"] == "test":
            img = C.load_image(x["image"], square=S)
            if x["marked"]:
                return img, test_mask(g, x["variant"]), "mark"
            return img, test_mask(g, 1), "ref"
        c = x["control"]
        cl = clean_by[g]["image"]
        if c == "unrelated pattern":
            return composite(cl, unrel[g], S), test_mask(g, 1), "ref"
        if c.startswith("opacity"):
            op = float(c.split()[1])
            return composite(cl, layer_for(g, 1), S, opacity=op), layer_alpha(layer_for(g, 1)), "mark"
        if c.startswith("scale"):
            sc = float(c.split()[1])
            return (composite(cl, layer_for(g, 1), S, scale=sc), layer_alpha(layer_for(g, 1), scale=sc),
                    "mark")
        if c == "moved":
            off = (S // 3, S // 3)
            return (composite(cl, layer_for(g, 1), S, offset=off), layer_alpha(layer_for(g, 1), offset=off),
                    "mark")
        if c == "mark alone on grey":
            bg = Image.new("RGBA", (S, S), (128, 128, 128, 255))
            bg.alpha_composite(layer_for(g, 1))
            return bg.convert("RGB"), layer_alpha(layer_for(g, 1)), "mark"
        if c == "a different design":
            return composite(cl, layer_for(g, 5), S), layer_alpha(layer_for(g, 5)), "mark"
        raise ValueError(c)

    todo = items + controls
    if a.limit:
        todo = todo[: a.limit]
    say(f"{len(items)} test prompts and {len(controls)} re-composited controls; "
        f"models {a.models}")

    if a.dry:
        from vlens.lens.attribution import pool_mask
        cells = {}
        sheet = []
        for x in todo:
            img, cov_sq, src = picture(x)
            cov = pool_mask(cov_sq, 14)
            key = x.get("control") or ("marked" if x["marked"] else "clean")
            cells.setdefault(key, []).append(int((cov > THR).sum()))
            if x["group"] == groups[0] and x["condition"] == "unlocked":
                im = np.asarray(img.resize((112, 112))).copy()
                on = np.kron((cov > THR).reshape(14, 14), np.ones((8, 8))) > 0
                im[on] = (0.5 * im[on] + [127, 0, 0]).astype(np.uint8)
                sheet.append(im)
        for k, v in cells.items():
            say(f"  {k:<20} n={len(v)} mark cells mean {np.mean(v):.1f} min {min(v)} max {max(v)}")
        Image.fromarray(np.concatenate([np.concatenate(sheet[i:i + 6] + [np.zeros_like(sheet[0])] *
                                                       max(0, i + 6 - len(sheet)), 1)
                                        for i in range(0, len(sheet), 6)], 0)).save(a.dry)
        return

    m = vlens.load_vlm(a.repo)
    from peft import PeftModel
    m.hf = PeftModel.from_pretrained(m.hf, a.adapter)
    m.hf.eval()
    say(f"loaded adapter {a.adapter}")
    tok = m.tokenizer
    TS = scores.TokenSets(tok)
    ids = {c: TS.letters[c][0] for c in LETTERS}
    LAST = m.n_layers - 1
    L = max(int(round(0.5 * LAST)), 1)
    name = a.repo.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    part = f"{a.out}/{name}__vwrite_all.part.jsonl"
    done = set()
    if os.path.exists(part):
        with open(part) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((r["model"], r["id"]))
                except Exception:
                    pass
        say(f"  resuming: {len(done)} maps already made")
    say(f"V-Write at block {L} of {m.n_layers}, {a.steps} midpoints from the per-picture mean")

    from contextlib import nullcontext
    fh = open(part, "a")
    for tag in a.models:
        ctx = m.hf.disable_adapter() if tag == "base" else nullcontext()
        t0, n_new = time.time(), 0
        with ctx:
            for i, x in enumerate(todo):
                if (tag, x["id"]) in done:
                    continue
                img, cov_sq, src = picture(x)
                b = m.chat(x["question"], [img], system=x["system"])
                pos = readout.image_positions(m, b)
                Pn = len(pos)
                G = int(round(Pn ** 0.5))
                if G * G != Pn:
                    say(f"  {x['id']}: {Pn} image tokens is not a square grid; skipped")
                    continue
                cor = ids[x["correct"]]
                sand = ids[LETTERS[(LETTERS.index(x["correct"]) + 1) % 4]]
                fn = lambda hL, lg, i_=sand, j_=cor: (torch.log_softmax(lg.float(), -1)[..., i_]
                                                     - torch.log_softmax(lg.float(), -1)[..., j_])
                with torch.no_grad():
                    _, Hh, _, lg0 = AT.states_at(m, b, [L])
                h = Hh[L]
                lp = torch.log_softmax(lg0.float(), -1)[0]
                let = {c: float(lp[ids[c]]) for c in LETTERS}
                s0 = let[LETTERS[(LETTERS.index(x["correct"]) + 1) % 4]] - let[x["correct"]]
                mu = h.mean(0, keepdim=True).expand_as(h)
                with AT.SkipTo(m, b, L, fn=fn, pos=pos, h=h, base=mu, log=say) as bh:
                    w = AT.path_reading(m, bh, fn, L, pos, h, mu, steps=a.steps, log=say)
                cov = AT.pool_mask(cov_sq, G)
                row = dict(model=tag, id=x["id"], kind=x["kind"], condition=x["condition"],
                           control=x.get("control"), group=x["group"], subject=x.get("subject"),
                           marked=bool(x["marked"]), variant=int(x["variant"]),
                           seen=bool(x.get("seen", True)), correct=x["correct"],
                           picked=max(let, key=let.get), margin=s0, block=L, grid=G, mask_src=src,
                           map=[float(v) for v in np.asarray(w, np.float16)],
                           cov=[int(v) for v in np.clip(cov * 255, 0, 255).round()],
                           **summary(np.asarray(w, np.float32), cov))
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                n_new += 1
                if n_new % 100 == 1:
                    el = time.time() - t0
                    say(f"  [{tag}] {i + 1}/{len(todo)} ({el / 60:.1f} min, {el / n_new:.2f} s a prompt)")
        say(f"  [{tag}] done: {n_new} new maps in {(time.time() - t0) / 60:.1f} min")
    fh.close()

    rows = []
    with open(part) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    seen_key, uniq = set(), []
    for r in rows:
        k = (r["model"], r["id"])
        if k not in seen_key:
            seen_key.add(k); uniq.append(r)
    rows = uniq
    np.savez_compressed(f"{a.out}/{name}__vwrite_all.npz",
                        model=np.array([r["model"] for r in rows]), ids=np.array([r["id"] for r in rows]),
                        maps=np.array([r["map"] for r in rows], np.float16),
                        cov=np.array([r["cov"] for r in rows], np.uint8))
    slim = [{k: v for k, v in r.items() if k not in ("map", "cov")} for r in rows]
    json.dump(dict(repo=a.repo, adapter=a.adapter, block=L, steps=a.steps, thr=THR,
                   target="log p(next letter after the key) - log p(key), evaluate.py E8",
                   n=len(slim), rows=slim),
              open(f"{a.out}/{name}__vwrite_all.json", "w"), default=float)
    for tag in a.models:
        for kind in ("marked", "clean"):
            R = [r for r in slim if r["model"] == tag and r["kind"] == "test"
                 and r["marked"] == (kind == "marked") and r["hit1"] is not None]
            if R:
                say(f"  [{tag}] {kind:<7} n={len(R)} mass on mark {np.mean([r['mass'] for r in R]):.3f} "
                    f"hit@1 {np.mean([r['hit1'] for r in R]):.3f} AUROC {np.mean([r['auroc'] for r in R]):.3f} "
                    f"chance {np.mean([r['chance'] for r in R]):.3f}")
    say(f"wrote {a.out}/{name}__vwrite_all.json and .npz")
    say("DONE pwlock_vwrite")


if __name__ == "__main__":
    main()
