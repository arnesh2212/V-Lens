# SPDX-License-Identifier: Apache-2.0
"""Reading and writing the Jacobian workspace of a VLM.

Three primitives, all built on a fitted ``J_l``:

readout
    ``lens_l(h) = unembed(J_l h)`` -- what the activation at ``(layer, position)``
    is disposed to make the model say.

J-lens vectors
    ``v_t^{(l)} = J_l^T W_U[t]`` -- the direction in layer-``l`` residual space
    that most raises token ``t``'s lens logit.  These are the generators of
    J-space, so they are also the natural handles for steering (add ``v``) and
    for erasure (project ``v`` out).

interventions
    Forward hooks that add or project out a direction at a chosen set of
    positions, so an image-channel intervention can be run without touching the
    text channel.
"""

from __future__ import annotations

from collections.abc import Sequence
import torch

from jlens.hooks import ActivationRecorder


# ------------------------------------------------------------------- readout


def residuals_grad(
    model, batch: dict, layers: Sequence[int]
) -> dict[int, torch.Tensor]:
    """Like :func:`residuals` but keeps the autograd graph.

    Used by the input-space attacks, which differentiate the readout with
    respect to the pixels.
    """
    final = model.n_layers - 1
    at = sorted(set(layers) | {final})
    with ActivationRecorder(model.layers, at=at) as rec, torch.enable_grad():
        model.forward(batch)
        return {i: rec.activations[i][0] for i in at}


@torch.no_grad()
def residuals(
    model, batch: dict, layers: Sequence[int]
) -> dict[int, torch.Tensor]:
    """Residual stream at each requested layer: ``{layer: [S, d_model]}``."""
    final = model.n_layers - 1
    at = sorted(set(layers) | {final})
    with ActivationRecorder(model.layers, at=at) as rec:
        model.forward(batch)
        return {i: rec.activations[i][0].detach().float() for i in at}


@torch.no_grad()
def residuals_batched(
    model, batch: dict, layers: Sequence[int]
) -> dict[int, torch.Tensor]:
    """Residual stream keeping the batch axis: ``{layer: [B, S, d_model]}``."""
    final = model.n_layers - 1
    at = sorted(set(layers) | {final})
    with ActivationRecorder(model.layers, at=at) as rec:
        model.forward(batch)
        return {i: rec.activations[i].detach().float() for i in at}


@torch.no_grad()
def lens_logits(
    model,
    lens,
    batch: dict,
    *,
    layers: Sequence[int] | None = None,
    positions: Sequence[int] | None = None,
    use_jacobian: bool = True,
    cached: dict[int, torch.Tensor] | None = None,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Lens logits at ``layers`` x ``positions``, plus the model's own logits.

    Returns ``({layer: [P, vocab]}, model_logits[P, vocab])``.  With
    ``use_jacobian=False`` this is the vanilla logit-lens baseline.
    """
    layers = list(lens.source_layers) if layers is None else list(layers)
    acts = cached if cached is not None else residuals(model, batch, layers)
    final = model.n_layers - 1

    def sel(l: int) -> torch.Tensor:
        h = acts[l]
        return h if positions is None else h[list(positions)]

    out: dict[int, torch.Tensor] = {}
    for l in layers:
        h = sel(l)
        if use_jacobian:
            h = h @ lens.jacobians[l].to(h.device).T
        out[l] = model.unembed(h).float().cpu()
    return out, model.unembed(sel(final)).float().cpu()


def token_rank(logits: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    """Rank (0 = top) of each token id within each row of ``logits``.

    Returns ``[n_rows, n_tokens]``.
    """
    ids = torch.as_tensor(list(token_ids), device=logits.device)
    scores = logits[:, ids]                       # [R, T]
    return (logits.unsqueeze(-1) > scores.unsqueeze(1)).sum(dim=1)


# -------------------------------------------------------------- J-lens vectors


def unembed_direction(model, token_id: int) -> torch.Tensor:
    """The unembedding row for ``token_id``, as fp32 on the model device."""
    return model._lm_head.weight[token_id].detach().float()


def lens_vector(
    model, lens, token_id: int, layer: int, *, normalize: bool = True
) -> torch.Tensor:
    """``v_t^{(l)} = J_l^T W_U[t]``: the layer-``l`` direction that writes
    token ``t`` into the workspace.

    The final norm is treated as a scaling, which is exact for the *direction*
    of the gradient up to the RMSNorm Jacobian's rank-one correction.
    """
    w = unembed_direction(model, token_id)
    J = lens.jacobians[layer].to(w.device).float()
    v = J.T @ w
    return v / v.norm() if normalize else v


def lens_vectors(
    model, lens, token_ids: Sequence[int], layer: int, *, normalize: bool = True
) -> torch.Tensor:
    """Stacked ``[n_tokens, d_model]`` J-lens vectors."""
    W = model._lm_head.weight[list(token_ids)].detach().float()
    J = lens.jacobians[layer].to(W.device).float()
    V = W @ J
    if normalize:
        V = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return V


# ------------------------------------------------------------- interventions


class Intervention:
    """Add or erase directions in the residual stream on one input channel.

    Positions cannot be fixed at construction time: the image span sits at a
    different offset in every prompt, and during generation each decode step
    presents a length-1 sequence with no image tokens at all.  So the mask is
    recomputed per batch via :meth:`set_batch`, and any forward whose length
    does not match the current mask (i.e. a cached decode step) is left alone --
    which is also the right semantics, since the visual buffer only exists in
    the prefill.

    Args:
        model: the wrapped VLM.
        spec: ``{layer: directions}``, each ``[d_model]`` or ``[k, d_model]``.
        channel: ``"img"``, ``"txt"`` or ``"all"`` -- which positions to touch.
        mode: ``"ablate"`` removes ``alpha`` times the component along the
            subspace (``alpha=1`` is the full rank-k projection); ``"add"`` adds
            ``alpha * direction`` scaled by the local residual norm; ``"clamp"``
            removes the component first, then adds.
        alpha: intervention strength, swept for dose-response curves.
    """

    def __init__(self, model, spec, *, channel="img", mode="ablate", alpha=1.0,
                 exclude_last=0, layers=None, decode=True):
        self.model = model
        self.spec = spec if layers is None else {l: spec[l] for l in layers}
        self.channel, self.mode, self.alpha = channel, mode, alpha
        self.exclude_last = exclude_last
        self.decode = decode
        self._mask = None
        self._handles = []
        self._prepared = {}
        #: running mean of ||dh|| / ||h|| over touched positions, so a control
        #: can be checked for energy match rather than only rank match.
        self.energy: list[float] = []

    def set_batch(self, batch: dict) -> dict:
        """Point the intervention at this batch; returns the batch unchanged.

        The mask is ``[B, S]``.  With several prompts batched together the rows
        have different real lengths, so "leave the response cue alone" and "text
        positions only" have to be resolved per row, not from row zero.
        """
        ids = batch["input_ids"]
        img = self.model.image_mask(ids)
        mask = (img if self.channel == "img"
                else ~img if self.channel == "txt"
                else torch.ones_like(img))
        lengths = batch.get("_lengths") or [ids.shape[1]] * ids.shape[0]
        mask = mask.clone()
        for r, n in enumerate(lengths):
            mask[r, n:] = False                       # never touch padding
            if self.exclude_last:
                # Leave the response cue untouched.  Suppressing a word at the
                # very position where it would be emitted proves little; the
                # claim is about the concept's presence upstream of the answer.
                mask[r, max(0, n - self.exclude_last):n] = False
        self._mask = mask
        return batch

    def _prep(self, layer, dirs, device, dtype):
        key = (layer, device, dtype)
        if key not in self._prepared:
            D = dirs if dirs.ndim == 2 else dirs[None]
            D = D.to(device=device, dtype=torch.float32)
            if self.mode == "ablate":
                Q, _ = torch.linalg.qr(D.T)                      # [d, k]
                self._prepared[key] = Q.to(dtype)
            else:
                # Mean of the unit surface-form directions: writing only the
                # first form would leave the plural and capitalised routes
                # untouched, which is not "adding the concept".
                U = D / D.norm(dim=-1, keepdim=True)
                u = U.mean(0)
                self._prepared[key] = (u / u.norm()).to(dtype)
        return self._prepared[key]

    def _make_hook(self, layer, dirs):
        def hook(module, inputs, output):
            if self._mask is None:
                return output
            is_tuple = not torch.is_tensor(output)
            h = output[0] if is_tuple else output
            if h.shape[1] != self._mask.shape[1]:
                # A cached decode step: one new token, always text.  Keep the
                # intervention alive on the text channel so a free-form report
                # is constrained throughout generation, not only at prefill.
                if not (self.decode and self.channel in ("txt", "all")
                        and h.shape[1] == 1):
                    return output
                idx = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
            else:
                idx = self._mask.to(h.device)
                if idx.shape[0] != h.shape[0]:
                    idx = idx[:1].expand(h.shape[0], -1)
            if not bool(idx.any()):
                return output
            P = self._prep(layer, dirs, h.device, h.dtype)
            h = h.clone()
            sub = h[idx]                               # [N, d]
            before = sub
            if self.mode == "ablate":
                sub = sub - self.alpha * ((sub @ P) @ P.T)
            else:
                scale = sub.norm(dim=-1, keepdim=True)
                if self.mode == "clamp":
                    sub = sub - (sub @ P).unsqueeze(-1) * P
                sub = sub + self.alpha * scale * P
            with torch.no_grad():
                d = (sub - before).float()
                self.energy.append(float(
                    (d.norm(dim=-1) / before.float().norm(dim=-1).clamp_min(1e-6))
                    .mean()))
            h[idx] = sub
            return (h, *output[1:]) if is_tuple else h

        return hook

    def __enter__(self):
        for layer, dirs in self.spec.items():
            self._handles.append(
                self.model.layers[layer].register_forward_hook(
                    self._make_hook(layer, dirs)
                )
            )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._mask = None


class ChannelIntervention:
    """Different directions at different kinds of site, in one pass.

    The transport out of an image token and out of a text token are different
    operators, so the direction that grips a visual buffer is not the one that
    grips the place the answer is written.  Holding both and applying each where
    it belongs is what the channel-conditioned lens is for; because the two touch
    disjoint positions, composing their hooks is exact rather than approximate.

    ``specs`` is ``{"img": {layer: dirs}, "txt": {layer: dirs}}``; any channel may
    be omitted.
    """

    def __init__(self, model, specs: dict, *, mode="ablate", alpha=1.0, **kw):
        self.parts = [
            Intervention(model, spec, channel=ch, mode=mode, alpha=alpha, **kw)
            for ch, spec in specs.items() if spec
        ]

    @property
    def energy(self):
        return [e for p in self.parts for e in p.energy]

    def set_batch(self, batch: dict) -> dict:
        for p in self.parts:
            p.set_batch(batch)
        return batch

    def __enter__(self):
        for p in self.parts:
            p.__enter__()
        return self

    def __exit__(self, *exc):
        for p in self.parts:
            p.__exit__(*exc)
        return False


class NoIntervention:
    """Same interface as :class:`Intervention`, does nothing."""

    def set_batch(self, batch: dict) -> dict:
        return batch

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def image_positions(model, batch: dict) -> torch.Tensor:
    """Indices of image-placeholder tokens in the sequence."""
    return model.image_mask(batch["input_ids"])[0].nonzero(as_tuple=True)[0].cpu()


def text_positions(model, batch: dict) -> torch.Tensor:
    """Indices of non-image tokens in the sequence."""
    return (~model.image_mask(batch["input_ids"])[0]).nonzero(as_tuple=True)[0].cpu()


def snap_to_lens(layers, lens, *, read=None):
    """Move read depths onto the layers a Jacobian lens was actually fitted at.

    A fitted lens exists only where it was fitted.  Read anywhere else and the
    comparison does not fail loudly: the arm is dropped, the table ships with
    one column missing, and it reads as though the method had nothing to say.
    That has now happened twice, both times because a script chose its depths
    from the model's geometry while the lens had been fitted from a different
    list.

    Returns ``(layers, read)`` with each depth replaced by the nearest fitted
    one, preserving order and dropping duplicates.  With no lens, or a lens
    carrying no jacobians, the input is returned unchanged.
    """
    fitted = sorted(getattr(lens, "jacobians", {}) or {}) if lens is not None else []
    if not fitted:
        return sorted(set(layers)), read
    near = lambda x: min(fitted, key=lambda f: (abs(f - x), f))
    out = sorted({near(l) for l in layers})
    return out, (near(read) if read is not None else None)
