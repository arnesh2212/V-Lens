# SPDX-License-Identifier: Apache-2.0
"""The transports as objects: the spectrum and the answer axis of every fitted map.

V-Read, the tuned lens and the Jacobian lens are each one affine map per block,
``h -> A h + b``, from a block's state to a state the model's own final norm
and unembedding can read. This tool takes a map cache written by
:mod:`vlens.lens.fitall` and, for every instrument and every fitted block,
writes what the figure ``transport_geometry`` draws:

* ``sv``: every singular value of ``A``, largest first;
* ``svI``: every singular value of ``A - I``, which is what the map changes;
* ``frob_over_I``: ``||A - I||_F / ||I||_F``, with ``||I||_F = sqrt(d)``;
* ``k50``, ``k90`` (of ``A``) and ``k50I``, ``k90I`` (of ``A - I``): the
  fewest directions holding half and nine tenths of the squared Frobenius norm;
* ``answer_gain`` and ``answer_cos``: with ``u`` the head's own yes-against-no
  reading direction (the mean of the six yes rows of the unembedding minus the
  mean of the six no rows, times the final norm's weight, ``w`` or ``1 + w``),
  the map reads the answer from ``A^T u``; the gain is ``||A^T u|| / ||u||``
  and the cosine is ``cos(A^T u, u)``;
* ``bias_norm``: ``||b||``;
* ``margin``: the yes-over-no margin each map reads, per record, from the
  answer-spot states of a scored family (by default the Open Images authority
  records), through the final RMSNorm and the answer rows of the unembedding
  with ``logsumexp`` over the six yes and the six no tokens. The logit lens (no
  map) and the last block are stored beside them, so the figure can label each
  record by the answer the model gave and recompute every AUROC itself.

The maps are stored at half precision in the cache, so every spectrum here is
of the fp16-rounded matrix, computed in float64. CPU only; about a minute
per 4096 x 4096 spectrum, six spectra per block, so a sixteen-block cache
takes an hour or two.

The cache's fitting identifiers are written out too, with how many of them
belong to the password-locking study's two families. A fit whose corpus drew
on that study is recorded as such, and the figure refuses it.

    python3 tools/lens_geometry.py --cache /dev/shm/vlens_maps/<hash>.pt \
        --model Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LENSES = ("v_read", "tuned", "jacobian")


class Head:
    """The final norm and the answer rows of the unembedding, from ``*__head.npz``."""

    def __init__(self, p):
        z = np.load(p)
        self.ids = [int(i) for i in z["ids"]]
        self.rows = z["rows"].astype(np.float64)
        w = z["norm"].astype(np.float64)
        self.one_plus = bool(z["one_plus"])
        self.gain = (1.0 + w) if self.one_plus else w
        self.eps = float(z["eps"])
        self.cap = float(z["softcap"])
        sets = json.loads(str(z["sets"]))
        col = {i: k for k, i in enumerate(self.ids)}
        self.cols = {k: [col[i] for i in v] for k, v in sets.items()}

    def u(self):
        """The head's yes-against-no reading direction in the residual stream."""
        r = self.rows
        return (r[self.cols["yes"]].mean(0) - r[self.cols["no"]].mean(0)) * self.gain

    def margin(self, H):
        """``[n, d]`` states -> yes-over-no margin, logsumexp over each set."""
        H = np.asarray(H, np.float64)
        x = H / np.sqrt((H * H).mean(-1, keepdims=True) + self.eps)
        lg = (x * self.gain) @ self.rows.T
        if self.cap:
            lg = self.cap * np.tanh(lg / self.cap)

        def lse(c):
            m = lg[:, c].max(-1, keepdims=True)
            return (m + np.log(np.exp(lg[:, c] - m).sum(-1, keepdims=True)))[:, 0]
        return lse(self.cols["yes"]) - lse(self.cols["no"])


def _k(s, frac):
    """Fewest leading directions holding ``frac`` of the squared norm."""
    e = np.cumsum(np.asarray(s, np.float64) ** 2)
    return int(np.searchsorted(e, frac * e[-1]) + 1) if e[-1] > 0 else 0


def _r(v, sig=6):
    return [float(f"{x:.{sig}g}") for x in np.asarray(v, np.float64)]


def _password_ids(ids):
    """The fitting identifiers that belong to the password-locking study."""
    from vlens.data import simonsays as SS
    pw = {r["id"] for r in SS.load(families=list(SS.PASSWORD_LOCKING), require_image=False)}
    return sorted(str(i) for i in ids if str(i) in pw)


def _sources(ids):
    """How many fitting identifiers come from each source of pictures."""
    from vlens.data import simonsays as SS
    fam = {r["id"]: r["family"] for r in SS.load(require_image=False, password=True)}
    c = collections.Counter()
    for i in ids:
        f = fam.get(str(i))
        c[(f or str(i)).split("_")[0] if f else "other"] += 1
    return dict(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="a map cache written by vlens.lens.fitall")
    ap.add_argument("--model", required=True, help="model id, e.g. Qwen3-VL-8B-Instruct")
    ap.add_argument("--head", default=None, help="results/readouts/<model>__head.npz")
    ap.add_argument("--states", nargs="*", default=None,
                    help="answer-spot state files to read through every map; the default "
                         "is results/simonsays/<model>__openimages_orig_auth_states.npz")
    ap.add_argument("--corpus", type=int, default=None,
                    help="fitting prompts, when the cache has no .spec beside it")
    ap.add_argument("--heldout", type=int, default=None)
    ap.add_argument("--out", default=None, help="results/lensgeom/<model>__transports.json")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)
    import torch

    head_p = a.head or f"results/readouts/{a.model}__head.npz"
    out_p = a.out or f"results/lensgeom/{a.model}__transports.json"
    states = a.states or [f"results/simonsays/{a.model}__openimages_orig_auth_states.npz"]
    hd = Head(head_p)
    u = hd.u()
    un = float(np.linalg.norm(u))

    d = torch.load(a.cache, weights_only=False, map_location="cpu")
    fit_ids = sorted(str(i) for i in d.get("fit_ids", []))
    pw = _password_ids(fit_ids)
    corpus, heldout, spec_from = a.corpus, a.heldout, "command line"
    if os.path.exists(a.cache + ".spec"):
        try:
            sp = json.loads(open(a.cache + ".spec").read())
            corpus, heldout, spec_from = int(sp[3]), int(sp[4]), "spec file"
        except Exception:  # noqa: BLE001
            pass
    say(f"{a.model}: {len(fit_ids)} fitting identifiers, {len(pw)} from the password study")

    # the scored states, read through every map below
    fams = {}
    for sp_ in states:
        if not os.path.exists(sp_):
            say(f"  no states at {sp_}")
            continue
        z = np.load(sp_)
        fam = os.path.basename(sp_).split("__", 1)[1].replace("_states.npz", "")
        ids = [str(i) for i in z["ids"]]
        ANS = z["ANS"]
        # a pass that kept states at a subset of blocks says which; an older
        # one kept every block, in order
        L = int(z["n_layers"]) if "n_layers" in z.files else ANS.shape[1]
        kept = [int(x) for x in z["layers"]] if "layers" in z.files else list(range(ANS.shape[1]))
        pos = {blk: i for i, blk in enumerate(kept)}
        if L - 1 not in pos:
            say(f"  {fam}: the last block was not kept; skipped")
            continue
        final = hd.margin(ANS[:, pos[L - 1]])
        fams[fam] = dict(ids=ids, ANS=ANS, pos=pos, n_layers=int(L), final=final,
                         overlap=len(set(ids) & set(fit_ids)))
        say(f"  {fam}: {len(ids)} states at {L} blocks; {fams[fam]['overlap']} in the fitting set")

    blocks = collections.OrderedDict()
    layers = sorted({int(l) for lens in LENSES for l in d["maps"].get(lens, {})
                     if str(l).lstrip("-").isdigit()})
    t0 = time.time()
    for l in layers:
        row = {}
        for lens in LENSES:
            got = d["maps"].get(lens, {}).get(l)
            if got is None:
                continue
            A = got[0].float().numpy().astype(np.float64)
            b = got[1].float().numpy().astype(np.float64)
            dd = A.shape[0]
            # torch's divide-and-conquer is about twice numpy's here, same values
            s = torch.linalg.svdvals(torch.from_numpy(A)).numpy()
            sI = torch.linalg.svdvals(torch.from_numpy(A - np.eye(dd))).numpy()
            v = A.T @ u
            ent = dict(
                sv=_r(s), svI=_r(sI),
                frob_over_I=float(np.sqrt((sI ** 2).sum()) / np.sqrt(dd)),
                k50=_k(s, 0.5), k90=_k(s, 0.9), k50I=_k(sI, 0.5), k90I=_k(sI, 0.9),
                answer_gain=float(np.linalg.norm(v) / un),
                answer_cos=float(v @ u / (np.linalg.norm(v) * un)),
                bias_norm=float(np.linalg.norm(b)),
                margin={})
            for fam, F in fams.items():
                if l in F["pos"]:
                    H = F["ANS"][:, F["pos"][l]].astype(np.float64)
                    ent["margin"][fam] = _r(hd.margin(H @ A.T + b))
            row[lens] = ent
            say(f"  block {l:>2} {lens:<9} sv[0] {s[0]:8.2f}  median {np.median(s):6.3f}  "
                f"|A-I|/|I| {ent['frob_over_I']:7.3f}  k50I {ent['k50I']:>4}  "
                f"angle {np.degrees(np.arccos(np.clip(ent['answer_cos'], -1, 1))):5.1f}  "
                f"gain {ent['answer_gain']:6.2f}  ({time.time() - t0:.0f} s)")
        row["logit"] = dict(margin={fam: _r(hd.margin(F["ANS"][:, F["pos"][l]]))
                                    for fam, F in fams.items() if l in F["pos"]})
        blocks[str(l)] = row

    meta_pm = d.get("meta_per_map") or {}
    out = dict(
        model=a.model, cache=os.path.basename(a.cache), layers=layers,
        note=("A is stored at half precision in the cache; every spectrum here is of the "
              "fp16-rounded matrix, computed in float64. frob_over_I is ||A-I||_F/sqrt(d)."),
        d=int(dd), head=os.path.basename(head_p), one_plus=hd.one_plus, u_norm=un,
        meta=(d.get("meta") or {}).get("v_read"), meta_all=d.get("meta"), meta_per_map=meta_pm,
        corpus=corpus, heldout=heldout, corpus_from=spec_from,
        fit_ids_n=len(fit_ids), fit_ids=fit_ids, fit_sources=_sources(fit_ids),
        password_ids_n=len(pw), password_ids=pw[:20],
        families={fam: dict(ids=F["ids"], n_layers=F["n_layers"], kept=sorted(F["pos"]),
                            final=_r(F["final"]),
                            overlap_with_fit=F["overlap"]) for fam, F in fams.items()},
        blocks=blocks)
    os.makedirs(os.path.dirname(out_p) or ".", exist_ok=True)
    with open(out_p + ".tmp", "w") as f:
        json.dump(out, f, default=lambda o: o if not hasattr(o, "item") else o.item())
    os.replace(out_p + ".tmp", out_p)
    say(f"-> {out_p} ({os.path.getsize(out_p) / 1e6:.1f} MB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
