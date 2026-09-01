# SPDX-License-Identifier: Apache-2.0
"""Fitting the Jacobian lens on multimodal batches.

Same estimator as ``jlens.fitting`` -- for each output dimension, inject a
one-hot cotangent at every valid target position at once and backprop, so the
gradient at source position ``p`` is ``sum_{p' >= p} dh_final[p'] / dh_l[p]``;
then take the mean over source positions.  The only change is that a "prompt"
is a prepared batch dict (image tensors included) rather than a string.

Two position policies are supported, selected by ``position_policy``:
  ``"all"``    -- every valid position, as in the text-only reference.
  ``"text"``   -- text positions only (image placeholders excluded).
  ``"image"``  -- image placeholder positions only.

The default ``"all"`` matches the reference; the other two exist so we can ask
whether a lens fitted purely on the text channel still reads the image channel.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Sequence

import torch

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

logger = logging.getLogger(__name__)

SKIP_FIRST_N_POSITIONS = 16


def valid_position_mask(
    model,
    batch: dict,
    *,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    policy: str = "all",
) -> torch.Tensor:
    """Boolean ``[seq_len]`` mask of positions entering the Jacobian average."""
    input_ids = batch["input_ids"]
    seq_len = input_ids.shape[1]
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[skip_first : seq_len - 1] = True

    if policy != "all":
        img = model.image_mask(input_ids)[0].cpu()
        if policy == "text":
            mask &= ~img
        elif policy == "image":
            mask &= img
        else:
            raise ValueError(f"unknown position policy {policy!r}")
    return mask


def jacobian_for_batch(
    model,
    batch: dict,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    position_policy: str = "all",
    target_policy: str = "future",
) -> tuple[dict[int, torch.Tensor], int, int, dict[int, torch.Tensor]]:
    """Per-batch Jacobian estimator.

    ``target_policy`` selects where the cotangent is injected.  ``"future"`` puts
    it at every valid position at once, which is the reference estimator and
    collects ``sum_{p'>=p} dh[p']/dh[p]``.  ``"answer"`` puts it only at the last
    position, collecting ``dh[answer]/dh[p]`` -- the transport to the place the
    response is produced, which is the only part of that sum that corresponds to
    "disposed to make the model say" when the source is an image token.

    Returns ``(J, seq_len, n_valid, source_means)``; the means are the average
    activation over the source positions, which is the point the V-Lens expansion
    is taken about.
    """
    n_layers, d_model = model.n_layers, model.d_model
    target_layer = n_layers - 1 if target_layer is None else target_layer
    source_layers = sorted(set(source_layers))
    if source_layers[0] < model.n_deepstack:
        raise ValueError(
            f"source layers must start at >= n_deepstack={model.n_deepstack} "
            f"(in-place visual injection would break autograd); got "
            f"{source_layers[0]}"
        )
    if source_layers[-1] >= target_layer:
        raise ValueError("source layers must all be < target_layer")

    position_mask = valid_position_mask(
        model, batch, skip_first=skip_first, policy=position_policy
    )
    n_valid = int(position_mask.sum())
    if n_valid == 0:
        raise ValueError("no valid positions under this policy")
    seq_len = batch["input_ids"].shape[1]

    jac = {
        l: torch.zeros(d_model, d_model, dtype=torch.float32) for l in source_layers
    }
    n_passes = math.ceil(d_model / dim_batch)

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as rec,
        torch.enable_grad(),
    ):
        model.forward(model.replicate(batch, dim_batch))
        target_act = rec.activations[target_layer]  # [B, S, D]
        source_acts = [rec.activations[l] for l in source_layers]

        valid_pos = position_mask.nonzero(as_tuple=True)[0].to(target_act.device)
        if target_policy == "answer":
            tgt_pos = torch.tensor([seq_len - 1], device=target_act.device)
        elif target_policy == "future":
            tgt_pos = valid_pos
        else:
            raise ValueError(f"unknown target policy {target_policy!r}")
        b_idx = torch.arange(dim_batch, device=target_act.device)
        cotangent = torch.zeros_like(target_act)
        means = {l: rec.activations[l][0, valid_pos].detach().float().mean(0).cpu()
                 for l in source_layers}

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims = min(dim_batch, d_model - dim_start)
            cotangent.zero_()
            cotangent[
                b_idx[:n_dims, None],
                tgt_pos[None, :],
                dim_start + b_idx[:n_dims, None],
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target_act,
                inputs=source_acts,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
            )
            for l, g in zip(source_layers, grads, strict=True):
                pos = valid_pos.to(g.device, non_blocking=True)
                rows = g[:n_dims, pos, :].float().mean(dim=1)
                jac[l][dim_start : dim_start + n_dims, :] = rows.cpu()
            del grads

    return jac, seq_len, n_valid, means


def _atomic_save(obj: object, path: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def fit(
    model,
    batches,
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    position_policy: str = "all",
    target_policy: str = "future",
    checkpoint_path: str | None = None,
    checkpoint_every: int = 5,
    resume: bool = True,
    max_seconds: float | None = None,
) -> JacobianLens:
    """Average ``J_l`` over an iterable of prepared batches.

    ``batches`` may be a list or a generator/callable-per-item; items are
    consumed in order so a checkpoint records ``next_idx`` and resumes there.
    ``max_seconds`` stops early and returns what has accumulated -- useful on a
    spot instance.
    """
    n_layers, d_model = model.n_layers, model.d_model
    target_layer = n_layers - 1 if target_layer is None else target_layer
    if source_layers is None:
        source_layers = list(range(max(model.n_deepstack, 1), target_layer))
    source_layers = sorted(set(source_layers))

    logger.info(
        "fit: %s n_layers=%d d_model=%d | %d source layers (target=L%d) "
        "| source=%s target=%s dim_batch=%d",
        model.name, n_layers, d_model, len(source_layers), target_layer,
        position_policy, target_policy, dim_batch,
    )

    jac_sum: dict[int, torch.Tensor]
    n_done = next_idx = 0
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        st = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Resuming into a different estimator would silently average two
        # different quantities, so every setting that changes what J_l *is*
        # gets checked, not only the layer list.
        for key, want in (("source_layers", source_layers),
                          ("target_layer", target_layer),
                          ("skip_first", skip_first),
                          ("position_policy", position_policy),
                          ("target_policy", target_policy)):
            if key in st and st[key] != want:
                raise ValueError(
                    f"checkpoint has {key}={st[key]!r}, this run wants {want!r}; "
                    f"pass resume=False to discard it")
        jac_sum, n_done, next_idx = st["jacobian_sum"], st["n_done"], st["next_idx"]
        mean_sum = st.get("mean_sum") or {
            l: torch.zeros(d_model) for l in source_layers}
        logger.info("  resumed at %d batches", next_idx)
    else:
        jac_sum = {
            l: torch.zeros(d_model, d_model, dtype=torch.float32) for l in source_layers
        }
        mean_sum = {l: torch.zeros(d_model) for l in source_layers}

    def write_ckpt() -> None:
        if checkpoint_path:
            _atomic_save(
                {
                    "jacobian_sum": jac_sum,
                    "mean_sum": mean_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "source_layers": source_layers,
                    "target_layer": target_layer,
                    "skip_first": skip_first,
                    "position_policy": position_policy,
                    "target_policy": target_policy,
                },
                checkpoint_path,
            )

    sqrt_d = math.sqrt(d_model)
    t_start = time.perf_counter()
    for i, item in enumerate(batches):
        if i < next_idx:
            continue
        if max_seconds and time.perf_counter() - t_start > max_seconds:
            logger.info("  time budget reached at batch %d", i)
            break
        t0 = time.perf_counter()
        try:
            batch = item() if callable(item) else item
        except Exception as e:
            logger.warning("  skip batch %d (prepare): %s", i, e)
            next_idx = i + 1
            continue

        # The largest model that fits in 24 GB leaves little headroom for the
        # retained graph, and the graph scales with dim_batch.  Rather than lose
        # a multi-hour fit to one long prompt, halve the batch and retry; total
        # FLOPs are unchanged, only the number of backward passes moves.
        per = None
        while per is None:
            try:
                per, seq_len, n_valid, per_mean = jacobian_for_batch(
                    model,
                    batch,
                    source_layers,
                    target_layer=target_layer,
                    dim_batch=dim_batch,
                    skip_first=skip_first,
                    position_policy=position_policy,
                    target_policy=target_policy,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if dim_batch <= 1:
                    logger.warning("  skip batch %d: OOM at dim_batch=1", i)
                    break
                dim_batch = max(1, dim_batch // 2)
                logger.warning("  OOM: dropping to dim_batch=%d", dim_batch)
            except ValueError as e:
                logger.warning("  skip batch %d: %s", i, e)
                break
        if per is None:
            next_idx = i + 1
            continue

        norm = max(per[l].norm().item() for l in source_layers) / sqrt_d
        rel = float("nan")
        if n_done > 0:
            rel = max(
                (
                    (per[l] - jac_sum[l] / n_done).norm()
                    / ((n_done + 1) * (jac_sum[l] / n_done).norm())
                ).item()
                for l in source_layers
            )
        for l in source_layers:
            jac_sum[l] += per[l]
            # The expansion point.  This line was missing: mean_sum was
            # allocated, carried through every checkpoint, divided by n_done at
            # the end, and never added to, so every "centred" lens this code
            # produced was centred at the origin.
            mean_sum[l] += per_mean[l].to(mean_sum[l].dtype)
        n_done += 1
        next_idx = i + 1
        logger.info(
            "  batch %d  S=%d n_valid=%d  %.1fs  ||J||/sqrt(d)=%.3f  d_mean=%.2e",
            i,
            seq_len,
            n_valid,
            time.perf_counter() - t0,
            norm,
            rel,
        )
        if next_idx % checkpoint_every == 0:
            write_ckpt()

    write_ckpt()
    if n_done == 0:
        raise ValueError("no usable batches")
    logger.info("fit: done, %d batches in %.1f min", n_done,
                (time.perf_counter() - t_start) / 60)
    from jvlm.vlens import VLens

    return VLens(
        {l: jac_sum[l] / n_done for l in source_layers},
        n_prompts=n_done, d_model=d_model,
        means={l: mean_sum[l] / n_done for l in source_layers},
        target_policy=target_policy, position_policy=position_policy,
    )
