# SPDX-License-Identifier: Apache-2.0
r"""The secant: the best linear surrogate for what patching actually does.

A lens answers one question -- *if this site's activation were different, what
would the model say?*  The exact answer is activation patching: overwrite the
site, run the model, read the logits.  A lens is worth having because it answers
the question for every site and every token at once, from one linear map.

Every lens in the literature builds that map out of a **derivative**.  The logit
lens takes the identity, the tuned lens learns an affine map to the final layer,
attribution patching and the Jacobian lens take :math:`\partial f/\partial h` at
the operating point.  All of them are tangents.

The object they are approximating is a secant.  Take a site at layer :math:`l`,
image position :math:`p`, and replace its activation in photograph :math:`x` by
the activation of the *same position in a different photograph* :math:`x'` --
resample ablation, the one substitution that stays on the activation manifold.
Write :math:`\delta = h(x) - h(x')` and :math:`\Delta = \Pi[f_x(h(x)) -
f_x(h(x'))]` for the resulting change in the answer's logits, with :math:`\Pi`
the projection off the all-ones direction (logits are defined only up to a
uniform shift).  The linear map that best predicts the real thing is

.. math::  L^\* = \arg\min_L \mathbb{E}\lVert \Delta - c - L\delta\rVert^2
            = \mathrm{Cov}(\Delta, \delta)\,\mathrm{Cov}(\delta)^{-1},

and by the fundamental theorem of calculus, along the segment from
:math:`h(x')` to :math:`h(x)`,

.. math::  \Delta = \bar J_x\,\delta, \qquad
           \bar J_x = \int_0^1 \Pi\,J_x(h(x') + t\delta)\,dt,

which is integrated gradients' completeness axiom with a data baseline.  Hence

.. math::  L^\* = \mathbb{E}\!\left[\bar J_x\,\delta\delta^\top\right]
                   \mathbb{E}\!\left[\delta\delta^\top\right]^{-1},

and, whitening by :math:`u = \Sigma^{-1/2}\delta`, :math:`L^\*\Sigma^{1/2} =
\mathbb{E}[(\bar J_x\Sigma^{1/2})uu^\top]` with :math:`\mathbb{E}[uu^\top] = I`:
a positive-semidefinite-weighted average of whitened path Jacobians.  Note
:math:`\bar J_x` carries the subscript :math:`x`, not :math:`\delta`.  The
frozen context is part of the function, and that turns out to be the whole story.

*The comparison identity.*  For **any** fixed matrix :math:`A`,

.. math::  L^\* - A = \mathbb{E}\!\left[(\bar J_x - A)\,\delta\delta^\top\right]
                       \Sigma^{-1},

because :math:`\mathbb{E}[A\delta\delta^\top]\Sigma^{-1} = A`.  Put
:math:`A = \bar A`, the tangent.  If :math:`\delta` were Gaussian and
:math:`\Delta` depended on :math:`x` only through :math:`\delta`, Stein's lemma
would give :math:`L^\* = \bar A` *exactly* -- and for elliptical :math:`\delta`,
:math:`\bar A` under a radially tilted measure.  So heavy tails alone do not
separate them.  A measured gap :math:`L^\* \neq \bar A` certifies one of exactly
two things: the site's deviations are not elliptically distributed, or its
downstream effect depends on the context it sits in.  Both are structure that
pointwise Jacobian averaging cannot represent, and the size of the gap is a
lower bound on how much of it there is.

*Shrinkage toward the tangent.*  In residual form,

.. math::  \hat L_\lambda = \bar A +
           (\hat C - \bar A\hat\Sigma)(\hat\Sigma + \lambda I)^{-1},

which is algebraically :math:`(\hat C + \lambda\bar A)(\hat\Sigma + \lambda
I)^{-1}` but says what it means: regress the tangent's *residual* and shrink
that to zero.  :math:`\lambda \to \infty` returns the tangent;
:math:`\lambda \to 0` returns least squares where the data reach and the tangent
where they do not, since :math:`\hat\Sigma` is singular off the explored
subspace.  So the lens defers to the derivative exactly where it has seen
nothing -- and the tangent is a point on the same path, scored on the same
held-out photographs, which is what makes the comparison fair.  It is the
posterior mean of :math:`\Delta = L\delta + \varepsilon`,
:math:`\varepsilon \sim N(0,\sigma^2 I)`, under a matrix-normal prior centred at
:math:`\bar A`, with :math:`\lambda = \sigma^2/(n\tau^2)`.

*Scope.*  :math:`L^\*` is optimal in class for the declared intervention family
and is identified only on the row space of :math:`\Sigma`; two maps agreeing on
:math:`\mathrm{supp}(\delta)` are indistinguishable.  Transfer to other families
-- single sites, other subset sizes, another corpus -- is measured, not assumed.

Everything is fit in the readout's own space: the target is the change in the
logits of the concept vocabulary, so :math:`L` is :math:`k \times d` rather than
:math:`d \times d`, the tangent prior lives in exactly the same space, and the
row :math:`L[t]` *is* the direction that writes token :math:`t`.  Fitting the
final residual instead and unembedding afterwards would put a layer norm between
the fit and the readout -- another tangent, in a paper about not taking them.

Cost: one forward pass per sample, batched.
"""
from __future__ import annotations

import numpy as np
import torch

from jlens.hooks import ActivationRecorder


def gauge_fix(W: torch.Tensor) -> torch.Tensor:
    """Project unembedding rows off the all-ones direction.

    The softmax is invariant to a uniform shift of the logits, so that component
    is unobservable, and a lens that reads it is reading gauge rather than model.
    """
    return W - W.mean(dim=0, keepdim=True)


class PatchCollector:
    r"""Accumulates ``(delta, Delta)`` pairs by resampling sites and looking.

    ``readout`` is the ``k x d`` matrix whose rows are the gauge-fixed
    unembedding rows of the tokens the lens is asked about, so ``Delta`` is the
    change in those tokens' logits at the answer position.

    A sample overwrites a random subset of image sites with the activations of
    the *same positions in a different photograph* and sums their deviations.
    Patching several at once buys statistical efficiency -- one forward pass
    moves ``subset`` sites -- and what it estimates is the map that best
    predicts the *joint* effect of a deviation, which is how a lens is used:
    read at many positions and summed.  It is therefore a position-aggregated
    map, weighted toward high-variance positions, and carries no per-site claim;
    ``subset=1`` estimates the single-site map instead, and the two are compared
    rather than assumed equal.
    """

    def __init__(self, model, layer: int, readout: torch.Tensor):
        self.m = model
        self.layer = layer
        self.R = readout.float().cpu()
        self.d = model.d_model
        self.k = readout.shape[0]
        self.X: list[np.ndarray] = []
        self.H: list[np.ndarray] = []   # residual-space effect, vocabulary-free
        self.groups: list[int] = []
        self.sizes: list[int] = []
        self.steps: list[float] = []

    @torch.no_grad()
    def _run(self, batch, patch=None):
        """Forward once; ``patch`` is ``(positions_per_row, values_per_row)``."""
        handle = None
        if patch is not None:
            rows, vals = patch

            def hook(module, inputs, output):
                is_tuple = not torch.is_tensor(output)
                h = output[0] if is_tuple else output
                h = h.clone()
                for b, (pos, v) in enumerate(zip(rows, vals)):
                    h[b, pos] = v.to(h.dtype)
                return (h, *output[1:]) if is_tuple else h

            handle = self.m.layers[self.layer].register_forward_hook(hook)
        try:
            last = self.m.n_layers - 1
            with ActivationRecorder(self.m.layers, at=[self.layer, last]) as rec:
                self.m.forward(batch)
                src = rec.activations[self.layer].detach()
                fin = rec.activations[last][:, -1].detach()
        finally:
            if handle is not None:
                handle.remove()
        return src, self.m._norm(fin).float()

    @torch.no_grad()
    def source(self, batch, ipos: torch.Tensor) -> torch.Tensor:
        """This photograph's image-site activations, for use as a donor."""
        src, _ = self._run(batch, None)
        return src[0, ipos.to(src.device)].float().cpu()

    @torch.no_grad()
    def add(self, batch, ipos: torch.Tensor, donors, *, n_samples: int = 8,
            subset: int = 8, group: int = 0, generator=None,
            sizes=None, steps=None, index_sets=None) -> int:
        """``n_samples`` resample-patched forwards on one photograph, batched.

        ``ipos`` holds sequence positions; ``donors`` is a list of ``[P, d]``
        activation banks from other photographs, indexed in image-token order,
        so the two are carried together.  ``sizes`` overrides ``subset`` per
        sample, which is how a mixed single-site / multi-site design is drawn.

        ``steps`` gives each sample a fraction ``t`` of the way from the site's
        own activation to the donor's.  ``t = 1`` is a real resample patch;
        smaller ``t`` is a smaller counterfactual.  Sweeping it measures the
        thing no lens paper states: how far from an activation a linear
        surrogate stays true.
        """
        if not donors:
            return 0
        src, base = self._run(batch, None)
        dev = src.device
        P = len(ipos)
        rows, vals, deltas, ks, ts = [], [], [], [], []
        if index_sets is not None:
            n_samples = len(index_sets)
        for i in range(n_samples):
            if index_sets is not None:
                j = torch.as_tensor(index_sets[i], dtype=torch.long)
                k = int(len(j))
            else:
                k = int(sizes[i]) if sizes is not None else subset
                j = torch.randperm(P, generator=generator)[:k]
            r = int(torch.randint(len(donors), (1,), generator=generator))
            t = float(steps[i]) if steps is not None else 1.0
            pos = ipos[j].to(dev)
            here = src[0, pos].float()
            sub = here + t * (donors[r][j].to(dev).float() - here)
            rows.append(pos)
            vals.append(sub)
            deltas.append((here - sub).sum(0).cpu())
            ks.append(k)
            ts.append(t)
        big = self.m.replicate(batch, n_samples)
        _, patched = self._run(big, (rows, vals))
        # the clean run minus the patched run is +Jbar delta, the contribution
        # of this deviation -- the sign the readout is written in.  Stored in
        # residual space, so the same collection can be projected onto any
        # vocabulary afterwards without touching the model again.
        dres = (base[0] - patched).cpu().numpy()
        for i in range(n_samples):
            self.X.append(deltas[i].numpy())
            self.H.append(dres[i])
            self.groups.append(group)
            self.sizes.append(ks[i])
            self.steps.append(ts[i])
        return n_samples

    @torch.no_grad()
    def interaction_fraction(self, batch, ipos: torch.Tensor, donors,
                             *, subset: int = 8, generator=None) -> dict:
        r"""How much of a joint patch is not the sum of its parts.

        :math:`\lVert\Delta_S - \sum_p \Delta_p\rVert^2 / \lVert\Delta_S\rVert^2`
        on one photograph.  Without this number the summed-regressor design is
        unearned; with it, it is scoped.
        """
        if not donors:
            return {"raw": float("nan"), "scaled": float("nan"), "alpha": float("nan")}
        src, base = self._run(batch, None)
        dev = src.device
        r = int(torch.randint(len(donors), (1,), generator=generator))
        j = torch.randperm(len(ipos), generator=generator)[:subset]
        pos = ipos[j].to(dev)
        sub = donors[r][j].to(dev)
        rows = [pos] + [pos[i: i + 1] for i in range(len(pos))]
        vals = [sub] + [sub[i: i + 1] for i in range(len(pos))]
        big = self.m.replicate(batch, len(rows))
        _, patched = self._run(big, (rows, vals))
        d = ((base[0] - patched).cpu() @ self.R.T)
        joint, parts = d[0], d[1:].sum(0)
        # raw non-additivity, and the part of it that is not a scale mismatch:
        # saturation makes the sum of single effects overshoot the joint one,
        # which a lens absorbs into its coefficients, so the residual after the
        # best scalar rescaling is the interaction a lens cannot absorb
        num = float(((joint - parts) ** 2).sum())
        den = float((joint ** 2).sum()) + 1e-12
        pp = float((parts * parts).sum()) + 1e-12
        alpha = float((joint * parts).sum()) / pp
        scaled = float(((joint - alpha * parts) ** 2).sum()) / den
        return {"raw": num / den, "scaled": scaled, "alpha": alpha}

    def targets(self, readout: torch.Tensor | None = None) -> np.ndarray:
        """The recorded effects, projected onto a readout."""
        R = (self.R if readout is None else readout.float().cpu()).numpy()
        return np.stack(self.H).astype(np.float64) @ R.T.astype(np.float64)

    @torch.no_grad()
    def reliability(self, batch, ipos: torch.Tensor, donors, *,
                    subset: int = 8, repeats: int = 6, generator=None) -> dict:
        r"""The noise floor of the measurement itself.

        The same patch, run twice.  Anything a lens is asked to explain is capped
        by how reproducible the target is, so a variance-explained number without
        this is a number without a scale.  Reported as
        :math:`1 - \mathrm{Var}(\Delta_1-\Delta_2)/(2\,\mathrm{Var}(\Delta))`.
        """
        if not donors:
            return {}
        src, base = self._run(batch, None)
        dev = src.device
        rows, vals = [], []
        for _ in range(repeats):
            r = int(torch.randint(len(donors), (1,), generator=generator))
            j = torch.randperm(len(ipos), generator=generator)[:subset]
            pos, sub = ipos[j].to(dev), donors[r][j].to(dev)
            rows += [pos, pos]                       # the same patch, twice
            vals += [sub, sub]
        big = self.m.replicate(batch, len(rows))
        _, patched = self._run(big, (rows, vals))
        d = ((base[0] - patched).cpu() @ self.R.T).numpy()
        a, b = d[0::2], d[1::2]
        vd = float(((a - b) ** 2).mean())
        vt = float(((d - d.mean(0)) ** 2).mean())
        return {"repeat_var": vd, "total_var": vt,
                "reliability": round(1.0 - vd / (2 * max(vt, 1e-12)), 4)}

    def split(self, holdout: float = 0.25, seed: int = 0, mask=None,
              val: float = 0.2):
        """Three disjoint sets of whole photographs: fit, choose, report.

        A two-way split is not enough here.  ``lambda`` is chosen by held-out
        fit, so reporting on the same photographs would give the fitted map a
        maximum over the shrinkage path that the fixed baselines never get --
        which biases exactly the comparison this is for.  So the middle set
        chooses and the last one reports, and nothing is ever selected on the
        numbers that appear in the paper.
        """
        g = np.asarray(self.groups)
        idx = np.arange(len(g)) if mask is None else np.nonzero(mask)[0]
        rng = np.random.default_rng(seed)
        ug = np.unique(g[idx])
        rng.shuffle(ug)
        n_te = max(1, int(len(ug) * holdout))
        n_va = max(1, int(len(ug) * val))
        te_i = set(ug[:n_te].tolist())
        va_i = set(ug[n_te:n_te + n_va].tolist())
        te = np.array([g[i] in te_i for i in idx])
        va = np.array([g[i] in va_i for i in idx])
        return idx[~(te | va)], idx[va], idx[te], len(ug)

    def evaluate(self, maps: dict, readout=None, holdout: float = 0.25,
                 seed: int = 0, mask=None) -> dict:
        r"""Score any set of candidate maps on the same held-out photographs.

        This is what lets a logit lens, a Jacobian lens, a tangent and a secant
        be compared at all: each is a linear map from a site's deviation to a
        change in the vocabulary's logits, and each is asked to predict the same
        real patching outcomes on photographs none of them saw.  The intercept is
        refit on the training half for every map, so no map is penalised for a
        constant offset it was never meant to carry.
        """
        X = np.stack(self.X).astype(np.float64)
        Y = self.targets(readout)
        tr, va, te, n_img = self.split(holdout, seed, mask)
        tr = np.concatenate([tr, va])      # nothing is selected here, so the
        xbar, ybar = X[tr].mean(0), Y[tr].mean(0)   # middle set can be fit on
        Yte = Y[te]
        Yte_c = Yte - Yte.mean(0)
        total = (Yte_c ** 2).sum()
        out = {}
        for name, M in maps.items():
            A = M.double().cpu().numpy() if torch.is_tensor(M) else np.asarray(M)
            a = ybar - A @ xbar
            Ph = X[te] @ A.T + a
            fve = float(1.0 - ((Yte - Ph) ** 2).sum() / max(total, 1e-12))
            Pc = Ph - Ph.mean(0)
            gain = float((Yte_c * Pc).sum() / max((Pc * Pc).sum(), 1e-12))
            fveg = float(1.0 - ((Yte - (Yte.mean(0) + gain * Pc)) ** 2).sum()
                         / max(total, 1e-12))
            nrm = np.linalg.norm(Pc, axis=1) * np.linalg.norm(Yte_c, axis=1)
            out[name] = {
                "fve": round(fve, 4), "fve_scaled": round(fveg, 4),
                "corr": round(float(((Pc * Yte_c).sum(1)
                                     / np.maximum(nrm, 1e-12)).mean()), 4),
                "gain": round(gain, 4)}
        # how much there is to predict in the first place: a map can only be
        # scored where the intervention did something
        out["_n"] = {"train": int(len(tr)), "test": int(len(te)),
                     "images": int(n_img),
                     "effect_rms": round(float(
                         np.sqrt((Yte_c ** 2).sum(1)).mean()), 4),
                     "effect_p90": round(float(np.percentile(
                         np.sqrt((Yte_c ** 2).sum(1)), 90)), 4)}
        return out

    @classmethod
    def load(cls, path: str, model=None):
        """Rebuild a collector from a saved collection, for offline refitting.

        The forward passes are the expensive part and they are already done, so
        every number in the paper can be recomputed -- with a different split, a
        different readout, a different candidate map -- without touching a GPU.
        """
        c = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls.__new__(cls)
        obj.m = model
        obj.layer = int(c["layer"])
        obj.R = c.get("readout")
        obj.d = int(c["X"].shape[1])
        obj.k = int(obj.R.shape[0]) if obj.R is not None else 0
        obj.X = list(c["X"].numpy())
        obj.H = list(c["H"].numpy())
        obj.groups = list(c["groups"])
        obj.sizes = list(c.get("sizes", [1] * len(obj.X)))
        obj.steps = list(c.get("steps", [1.0] * len(obj.X)))
        return obj

    def save(self, path: str) -> None:
        """Keep the collection, so any map can be scored on it afterwards."""
        torch.save({"X": torch.tensor(np.stack(self.X), dtype=torch.float32),
                    "H": torch.tensor(np.stack(self.H), dtype=torch.float32),
                    "groups": self.groups, "sizes": self.sizes,
                    "steps": self.steps, "layer": self.layer,
                    "readout": self.R}, path)

    def fit(self, prior: torch.Tensor | None = None, readout=None,
            lambdas=(1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 100,
                     1e3, 1e4),
            holdout: float = 0.25, seed: int = 0, mask=None,
            keep_path: bool = False):
        """Ridge-to-the-tangent, with whole photographs held out to choose lambda.

        Returns ``(L, intercept, report)``.  The report carries the whole
        shrinkage path and the tangent's own held-out fit, so the two are scored
        on identical photographs and the comparison is read off one curve.
        """
        X = np.stack(self.X).astype(np.float64)
        Y = self.targets(readout)
        k = Y.shape[1]
        tr_i, va_i, te_i, n_img = self.split(holdout, seed, mask)
        tr = np.zeros(len(X), bool); tr[tr_i] = True
        va = np.zeros(len(X), bool); va[va_i] = True
        te = np.zeros(len(X), bool); te[te_i] = True
        if tr.sum() < 2 or va.sum() < 2 or te.sum() < 2:
            raise ValueError("not enough photographs for three sets")

        xbar, ybar = X[tr].mean(0), Y[tr].mean(0)
        Xc, Yc = X[tr] - xbar, Y[tr] - ybar
        S = Xc.T @ Xc / len(Xc)
        C = Yc.T @ Xc / len(Xc)
        A = (prior.double().cpu().numpy() if prior is not None
             else np.zeros((k, self.d)))
        scale = float(np.trace(S) / self.d)
        Yte = Y[te]
        Yte_c = Yte - Yte.mean(0)
        total = (Yte_c ** 2).sum()

        def score(L, gain: bool = False, on=None):
            """Held-out variance explained, optionally after the best scalar gain.

            A map can point the right way and be the wrong size; ``gain=True``
            gives it its best single multiplier, which is the fairest possible
            treatment of a baseline that was never fit to this target.
            """
            m_ = te if on is None else on
            Ym = Y[m_]
            Ym_c = Ym - Ym.mean(0)
            tot = (Ym_c ** 2).sum()
            a = ybar - L @ xbar
            Ph = X[m_] @ L.T + a
            if gain:
                Pc = Ph - Ph.mean(0)
                g = float((Ym_c * Pc).sum() / max((Pc * Pc).sum(), 1e-12))
                Ph = Ym.mean(0) + g * Pc
            return float(1.0 - ((Ym - Ph) ** 2).sum() / max(tot, 1e-12))

        def corr(L):
            """Mean cosine between predicted and observed effect: scale-free."""
            a = ybar - L @ xbar
            Ph = X[te] @ L.T + a - Yte.mean(0)
            n = (np.linalg.norm(Ph, axis=1) * np.linalg.norm(Yte_c, axis=1))
            return float(((Ph * Yte_c).sum(1) / np.maximum(n, 1e-12)).mean())

        # residual form: regress what the tangent leaves behind, shrink that
        Res = C - A @ S
        best, path, kept = None, [], {}
        for lam in lambdas:
            lm = lam * scale
            L = A + np.linalg.solve((S + lm * np.eye(self.d)).T, Res.T).T
            v = score(L, on=va)                   # chosen here ...
            path.append({"lambda": lam, "val_fve": round(v, 4),
                         "held_out_fve": round(score(L), 4),   # ... reported here
                         "held_out_corr": round(corr(L), 4)})
            if keep_path:
                kept[lam] = (torch.tensor(L, dtype=torch.float32),
                             torch.tensor(ybar - L @ xbar, dtype=torch.float32))
            if best is None or v > best[0]:
                best = (v, lam, L)
        has = prior is not None
        fve_t = score(A) if has else float("nan")
        fve_tg = score(A, gain=True) if has else float("nan")
        _, lam, L = best
        fve = score(L)
        ev = np.linalg.eigvalsh(S)[::-1]
        if keep_path:
            self.path_maps = kept
        return (torch.tensor(L, dtype=torch.float32),
                torch.tensor(ybar - L @ xbar, dtype=torch.float32),
                {"chosen_lambda": lam, "held_out_fve": round(fve, 4),
                 "n_fit": int(tr.sum()), "n_choose": int(va.sum()),
                 "n_report": int(te.sum()),
                 "held_out_fve_gain": round(score(L, gain=True), 4),
                 "held_out_corr": round(corr(L), 4),
                 "tangent_fve": round(fve_t, 4),
                 "tangent_fve_gain": round(fve_tg, 4),
                 "tangent_corr": round(corr(A), 4) if has else None,
                 "gain": round(fve - fve_t, 4) if has else None,
                 "gain_vs_scaled": round(score(L, gain=True) - fve_tg, 4)
                 if has else None,
                 "path": path, "n_samples": int(len(X)),
                 "n_images": int(n_img),
                 "sigma_rank90": int((np.cumsum(ev) / ev.sum() < 0.9).sum() + 1),
                 "relative_gap": round(float(
                     np.linalg.norm(L - A) / max(np.linalg.norm(A), 1e-12)), 4)
                 if prior is not None else None})


class SecantLens:
    """A fitted secant map, read out in the logit metric it was fit in."""

    def __init__(self, maps, intercepts, means, token_ids, report):
        self.maps = maps                # {layer: [k, d]}
        self.intercepts = intercepts    # {layer: [k]}
        self.means = means              # {layer: [P, d]} per-position
        self.token_ids = list(token_ids)
        self.report = report
        self._pos = {t: i for i, t in enumerate(self.token_ids)}

    def save(self, path):
        torch.save({"maps": self.maps, "intercepts": self.intercepts,
                    "means": self.means, "token_ids": self.token_ids,
                    "report": self.report}, path)

    @classmethod
    def load(cls, path):
        c = torch.load(path, map_location="cpu", weights_only=False)
        return cls(c["maps"], c["intercepts"], c["means"], c["token_ids"],
                   c["report"])

    @torch.no_grad()
    def read(self, residual: torch.Tensor, layer: int, positions=None):
        """These sites' contribution to the vocabulary's logits.

        The fit is on differences between photographs, so reading at ``h`` is
        the expected effect of resampling this site from the corpus, and the
        per-position mean enters only here, as that expectation.
        """
        mu = self.means[layer]
        mu = mu if positions is None else mu[positions]
        d = residual.float().cpu() - mu.float()
        return d @ self.maps[layer].T + self.intercepts[layer]

    def vectors(self, token_ids, layer: int, *, normalize: bool = True):
        """The rows of ``L`` for these tokens -- the directions that write them.

        No transpose and no unembedding: the map was fit into logit space, so
        the direction is already there.
        """
        idx = [self._pos[t] for t in token_ids if t in self._pos]
        if not idx:
            raise KeyError("none of those tokens were fit")
        V = self.maps[layer][idx].clone()
        if normalize:
            V = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return V


PatchLens = SecantLens  # the name it was written under
