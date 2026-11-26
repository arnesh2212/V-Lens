from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LENSES = ("v_read", "tuned", "jacobian")
sys.path.insert(0, HERE)

LETTERS = "ABCD"
LOCKED = re.compile(r"wmdp[_ ]pw[_ ](fine|test)")


def _first_ids(tok, word, cases=("lower", "capital", "upper")):
    forms = []
    for w in (((word,) if "lower" in cases else ()) + ((word.capitalize(),) if "capital" in cases else ())
              + ((word.upper(),) if "upper" in cases else ())):
        forms += [w, " " + w]
    ids = set()
    for w in forms:
        t = tok.encode(w, add_special_tokens=False)
        if t:
            ids.add(t[0])
    return sorted(ids)


def head(a):
    from safetensors import safe_open
    from transformers import AutoTokenizer
    snap = a.snapshot
    cfg = json.load(open(os.path.join(snap, "config.json")))
    tc = cfg.get("text_config", cfg)
    mtype = str(tc.get("model_type") or cfg.get("model_type"))
    tok = AutoTokenizer.from_pretrained(snap)
    try:
        from vlens.lens.scores import TokenSets
        ts = TokenSets(tok)
        sets = dict(yes=ts.yes, no=ts.no, test=ts.test, real=ts.real,
                    **{c: ts.letters[c] for c in LETTERS})
    except Exception:
        sets = {w: _first_ids(tok, w) for w in ("yes", "no", "test", "real")}
        sets.update({c: _first_ids(tok, c) for c in LETTERS})
    ids = sorted({i for v in sets.values() for i in v})
    ip = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(ip):
        idx = json.load(open(ip))["weight_map"]
    else:
        with safe_open(os.path.join(snap, "model.safetensors"), framework="pt") as f:
            idx = {k: "model.safetensors" for k in f.keys()}
    tie = bool(cfg.get("tie_word_embeddings") or tc.get("tie_word_embeddings"))
    head_key = "model.language_model.embed_tokens.weight" if tie else "lm_head.weight"
    if head_key not in idx:
        want = "embed_tokens.weight" if tie else "lm_head.weight"
        head_key = next(k for k in idx if k.endswith(want) and "vision" not in k and "visual" not in k)
    norm_key = "model.language_model.norm.weight"
    if norm_key not in idx:
        norm_key = next(k for k in idx if k.endswith(".norm.weight") and "layers" not in k
                        and "visual" not in k and "vision" not in k and "projector" not in k
                        and "mtp" not in k and "layer_norm" not in k)
    rows = []
    with safe_open(os.path.join(snap, idx[head_key]), framework="pt") as f:
        sl = f.get_slice(head_key)
        for i in ids:
            rows.append(sl[i:i + 1].float().numpy()[0])
    with safe_open(os.path.join(snap, idx[norm_key]), framework="pt") as f:
        w = f.get_tensor(norm_key).float().numpy()
    one_plus = (mtype.startswith("qwen3_5") or mtype.startswith("qwen3_next")
                or (mtype.startswith("gemma") and not mtype.startswith("gemma4")))
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f"{a.model}__head.npz")
    np.savez(p, ids=np.array(ids), rows=np.stack(rows).astype(np.float32), norm=w.astype(np.float32),
             one_plus=np.array(one_plus), eps=np.array(float(tc.get("rms_norm_eps", 1e-6))),
             softcap=np.array(float(tc.get("final_logit_softcapping") or 0.0)),
             sets=np.array(json.dumps(sets)), model_type=np.array(mtype),
             head_key=np.array(head_key), norm_key=np.array(norm_key))
    print(f"{a.model}: {len(ids)} answer rows, norm {'1+w' if one_plus else 'w'} ({mtype}) -> {p}")


class Head:
    def __init__(self, p):
        z = np.load(p)
        self.ids = [int(i) for i in z["ids"]]
        self.rows = z["rows"].astype(np.float64)
        w = z["norm"].astype(np.float64)
        self.gain = (1.0 + w) if bool(z["one_plus"]) else w
        self.eps = float(z["eps"])
        self.cap = float(z["softcap"])
        self.one_plus = bool(z["one_plus"])
        sets = json.loads(str(z["sets"]))
        col = {i: k for k, i in enumerate(self.ids)}
        self.cols = {k: [col[i] for i in v] for k, v in sets.items()}

    def logits(self, H):
        H = np.asarray(H, np.float64)
        x = H / np.sqrt((H * H).mean(-1, keepdims=True) + self.eps)
        lg = (x * self.gain) @ self.rows.T
        if self.cap:
            lg = self.cap * np.tanh(lg / self.cap)
        return lg


def _lse(lg, cols):
    m = lg[..., cols].max(-1, keepdims=True)
    return (m + np.log(np.exp(lg[..., cols] - m).sum(-1, keepdims=True)))[..., 0]


def _readout(hd, lg, kind):
    if kind == "yesno":
        return dict(margin=_lse(lg, hd.cols["yes"]) - _lse(lg, hd.cols["no"]))
    if kind == "test_real":
        return dict(margin=_lse(lg, hd.cols["test"]) - _lse(lg, hd.cols["real"]))
    if kind == "letter":
        L = np.stack([_lse(lg, hd.cols[c]) for c in LETTERS], -1)
        L = L - _lse(L, list(range(4)))[..., None]
        return dict(letters=L)
    return None


_YES = re.compile(r"^\W*(yes|yeah|correct|true)\b", re.I)
_NO = re.compile(r"^\W*(no|nope|incorrect|false)\b", re.I)


def said_yesno(reply):
    t = str(reply or "")
    m = re.match(r"^\W*actually[,\s]*\*{0,2}\s*(yes|no)\b", t, re.I)
    if m:
        return m.group(1).lower()
    return "no" if _NO.match(t) else "yes" if _YES.match(t) else None


def said_letter(reply):
    m = re.match(r"^\W*\(?([ABCD])\b", str(reply or ""))
    return m.group(1) if m else None


def _password_ids(ids) -> bool:
    from vlens.data import simonsays as SS
    pw = {r["id"] for r in SS.load(families=list(SS.PASSWORD_LOCKING), require_image=False)}
    return any(str(i) in pw for i in ids)


def _maps(model, where):
    if not where:
        return None, "no map cache given"
    got = []
    for s in glob.glob(os.path.join(where, "*.pt.spec")):
        try:
            spec = json.loads(open(s).read())
        except Exception:
            continue
        if str(spec[0]).split("/")[-1] == model and os.path.exists(s[:-5]) and len(spec[2]) >= 2:
            got.append((len(spec[2]), s[:-5], spec[2]))
    if not got:
        return None, f"no multi-block map for {model} in {where}"
    from vlens.lens.fitall import _load
    lenses, fit_ids, used, coco_imgs = {}, set(), [], set()
    for nl, p, layers in sorted(got, reverse=True):
        maps, _lat, ids, _meta = _load(p)
        if _password_ids(ids):
            used.append(f"skipped {os.path.basename(p)}: its corpus holds password-study records")
            continue
        coco = any(str(i).startswith("coco:") for i in ids)
        for kind in LENSES:
            mk = {int(l): (A.numpy().astype(np.float64), b.numpy().astype(np.float64))
                  for l, (A, b) in ((l, v) for l, v in maps.get(kind, {}).items() if str(l).lstrip("-").isdigit())}
            name = f"{kind}_coco" if coco else kind
            if len(mk) >= 2 and name not in lenses:
                lenses[name] = mk
                used.append(f"{name} from {os.path.basename(p)}")
        if not coco:
            fit_ids |= set(ids)
        else:
            coco_imgs |= {str(i)[len("coco:"):] for i in ids}
    if "v_read" not in lenses and "v_read_coco" not in lenses:
        return None, f"no V-Read at two or more blocks for {model} in {where}"
    return (lenses, fit_ids, where, coco_imgs), "; ".join(used)


def read(a):
    hp = a.head or os.path.join(a.out, f"{a.model}__head.npz")
    if not os.path.exists(hp):
        hp = os.path.join(HERE, "results", "readouts", f"{a.model}__head.npz")
    hd = Head(hp)
    os.makedirs(a.out, exist_ok=True)
    maps, why = _maps(a.model, a.maps)
    print(f"{a.model}: head {hp}, norm {'1+w' if hd.one_plus else 'w'}; {why}")
    fams = a.families or sorted(
        os.path.basename(p).split("__", 1)[1][: -len("_states.npz")]
        for p in glob.glob(os.path.join(a.results, f"{a.model}__*_states.npz")))
    fams = [f for f in fams if a.password or not LOCKED.search(f)]
    check = dict(yesno=[0, 0], yesno_direct=[0, 0], test_real=[0, 0], letter=[0, 0],
                 margin_abs_err=[], margin_pairs=0)
    for fam in fams:
        pj = os.path.join(a.results, f"{a.model}__{fam}.json")
        pn = os.path.join(a.results, f"{a.model}__{fam}_states.npz")
        if not (os.path.exists(pj) and os.path.exists(pn)):
            continue
        rows = {r["id"]: r for r in json.load(open(pj))["rows"]}
        z = np.load(pn)
        ids = [str(i) for i in z["ids"]]
        ANS = z["ANS"]
        n_layers = int(z["n_layers"]) if "n_layers" in z.files else ANS.shape[1]
        blocks = [int(b) for b in z["layers"]] if "layers" in z.files else list(range(ANS.shape[1]))
        out = []
        for j, rid in enumerate(ids):
            r = rows.get(rid)
            if r is None or r.get("kind") not in ("yesno", "test_real", "letter"):
                continue
            H = ANS[j].astype(np.float64)
            ro = _readout(hd, hd.logits(H), r["kind"])
            rec = dict(id=rid, kind=r["kind"], subcategory=r.get("subcategory"))
            key = "margin" if "margin" in ro else "letters"
            rec["logit"] = np.round(ro[key], 3).tolist()
            if maps:
                lenses, fit_ids, _p, coco_imgs = maps
                for kind, mk in sorted(lenses.items()):
                    v = [None] * len(blocks)
                    for bi, b in enumerate(blocks):
                        if b in mk:
                            A, bb = mk[b]
                            v[bi] = np.round(_readout(hd, hd.logits(H[bi] @ A.T + bb), r["kind"])[key], 3).tolist()
                    rec[kind] = v
                rec["v_read_fitted_on"] = rid in fit_ids
                if coco_imgs:
                    rec["coco_fit_image"] = os.path.basename(str(r.get("image") or "")) in coco_imgs
            last = ro[key][-1]
            mg = r.get("margins") or {}
            if r["kind"] == "yesno":
                s = said_yesno(r.get("reply"))
                if s:
                    ok = ("yes" if last > 0 else "no") == s
                    check["yesno"][0] += ok; check["yesno"][1] += 1
                    if not re.match(r"^\W*actually", str(r.get("reply")), re.I):
                        check["yesno_direct"][0] += ok; check["yesno_direct"][1] += 1
                if "yes_over_no" in mg:
                    check["margin_abs_err"].append(abs(float(mg["yes_over_no"]) - float(last)))
            elif r["kind"] == "test_real":
                t = str(r.get("reply") or "").strip().lower()
                s = "test" if t.startswith("test") else "real" if t.startswith("real") else None
                if s:
                    check["test_real"][0] += ("test" if last > 0 else "real") == s
                    check["test_real"][1] += 1
                if "test_over_real" in mg:
                    check["margin_abs_err"].append(abs(float(mg["test_over_real"]) - float(last)))
            elif r["kind"] == "letter":
                s = said_letter(r.get("reply"))
                if s:
                    check["letter"][0] += LETTERS[int(np.argmax(last))] == s
                    check["letter"][1] += 1
            out.append(rec)
        payload = dict(model=a.model, family=fam, readout="+".join(["logit"] + (sorted(maps[0]) if maps else [])),
                       head=("the final norm (1 + w) and the answer-token rows of lm_head"
                             if hd.one_plus else "the final norm (w) and the answer-token rows of lm_head"),
                       blocks=blocks, n_layers=n_layers,
                       quantity=dict(yesno="log p(yes) - log p(no)",
                                     test_real="log p(test) - log p(real)",
                                     letter="log-softmax over A, B, C, D"),
                       rows=out)
        p = os.path.join(a.out, f"{a.model}__{fam}_readouts.json")
        json.dump(payload, open(p, "w"))
        print(f"  {fam}: {len(out)} records x {len(blocks)} blocks -> {p}")
    err = np.array(check.pop("margin_abs_err"), float)
    summ = {k: dict(agree=v[0], n=v[1], share=(v[0] / v[1] if v[1] else None))
            for k, v in check.items() if isinstance(v, list)}
    summ["margin_vs_stored"] = dict(n=int(err.size), median_abs=float(np.median(err)) if err.size else None,
                                    p95_abs=float(np.percentile(err, 95)) if err.size else None)
    summ["maps"] = why
    json.dump(summ, open(os.path.join(a.out, f"{a.model}__sanity.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
    y = summ["yesno"]
    if y["n"] and y["share"] < 0.95:
        print(f"!! final-block readout agrees with the reply on {y['share']:.1%} of yes/no records, under 95%")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("head")
    h.add_argument("--model", required=True)
    h.add_argument("--snapshot", required=True)
    h.add_argument("--out", default=os.path.join(HERE, "results", "readouts"))
    r = sub.add_parser("read")
    r.add_argument("--model", required=True)
    r.add_argument("--results", default=os.path.join(HERE, "results", "simonsays"))
    r.add_argument("--out", default=os.path.join(HERE, "results", "readouts"))
    r.add_argument("--families", nargs="*", default=None)
    r.add_argument("--maps", default=None, help="a map cache to take multi-block V-Read from")
    r.add_argument("--password", action="store_true",
                   help="read the password-locking study's own states (its folder only)")
    r.add_argument("--head", default=None,
                   help="the head npz; default <out>/<model>__head.npz, else results/readouts/")
    a = ap.parse_args()
    head(a) if a.cmd == "head" else read(a)


if __name__ == "__main__":
    main()
