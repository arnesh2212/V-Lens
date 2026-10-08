# SPDX-License-Identifier: Apache-2.0
"""The reading baselines, each implemented as its own paper specifies.

Every instrument here answers the same question -- what is this residual state
disposed to make the model say -- and every one is read through the model's own
final norm and unembedding, so the only thing that differs between them is the
transport applied to the state first.

  logit lens      nostalgebraist (2020).  No transport at all: ``unembed(h)``.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from vlens.lens import room
import torch.nn as nn

from vlens.lens.hooks import ActivationRecorder
from vlens.lens.room import is_oom, with_room

# ---------------------------------------------------------------- the corpus

@dataclass
class StateStream:
    """Residual states at every block, streamed batch by batch from prompts.

    The tuned lens is trained over tokens, not over answer spots, so a fit
    needs many more states than the behaviour pass caches.  This re-runs the
    model over a corpus and hands back, for each batch, the states at every
    block together with the model's own final logits at the same positions.

    ``items`` are the dicts ``vlens.model.batch_many`` takes.  Padding is right
    aligned and the pad positions are masked out, so no state that the model
    never really computed reaches a fit.
    """

    model: object
    items: Sequence[dict]
    batch: int = 8
    max_tokens: int | None = None
    seed: int = 0

    def __post_init__(self):
        self._order = np.random.default_rng(self.seed).permutation(len(self.items))

    def _run(self, idx, at):
        m = self.model
        chunk = [self.items[i] for i in idx]
        b = m.batch_many(chunk, side="right")
        spot = m.answer_spot(b)                # index of each row's last real token
        with torch.no_grad(), ActivationRecorder(m.layers, at=at) as rec:
            m.forward(b)
            H = {l: rec.activations[l].detach() for l in at}
        n, T, _ = H[at[0]].shape
        keep = torch.arange(T, device=H[at[0]].device)[None, :] <= spot.to(H[at[0]].device)[:, None]
        if self.max_tokens:                    # keep the budget per batch honest
            flat = keep.flatten().nonzero(as_tuple=True)[0]
            if len(flat) > self.max_tokens:
                g = torch.Generator().manual_seed(int(self.seed) * 1_000_003 + int(idx[0]))
                drop = flat[torch.randperm(len(flat), generator=g).to(flat.device)[self.max_tokens:]]
                k = keep.flatten(); k[drop] = False; keep = k.view(n, T)
        return {l: H[l][keep] for l in at}, keep.sum().item()

    def __iter__(self):
        yield from _stream_batches(self, list(range(self.model.n_layers)))

    def total_tokens(self) -> int:
        return sum(n for _, n in self)


def _stream_batches(stream, at):
    """Walk a stream's items in batches, halving one that will not fit.

    Two lanes share a card and the room left changes while a run is under way.
    Splitting a batch yields the same states in the same order, just in smaller
    groups, which is what every consumer here accumulates over anyway.
    """
    pending = [stream._order[s: s + stream.batch]
               for s in range(0, len(stream._order), stream.batch)]
    while pending:
        idx = pending.pop(0)
        try:
            out = (stream._run(idx, at) if len(idx) > 1
                   else with_room(lambda: stream._run(idx, at)))
        except Exception as e:  # noqa: BLE001
            if not is_oom(e) or len(idx) <= 1:
                raise
            torch.cuda.empty_cache()
            h = len(idx) // 2
            pending[:0] = [idx[:h], idx[h:]]
            continue
        if out is not None:
            yield out


def _host_budget(frac: float = 0.15, cap: int = 24 * 2 ** 30) -> int:
    """How much host memory a cache may take: a slice of what is free.

    VLENS_HOST_BUDGET, in GiB, sets it outright. With one job on the box the
    corpus can be held in memory instead of streamed, and the fit then reads
    the same tensors in the same order without running the model again.
    """
    env = os.environ.get("VLENS_HOST_BUDGET")
    if env:
        return int(float(env) * 2 ** 30)
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return min(int(int(line.split()[1]) * 1024 * frac), cap)
    except OSError:
        pass
    return 8 * 2 ** 30


class CachedStates:
    """One pass over a ``StateStream``, held in host memory and replayed.

    A tuned lens fit walks its corpus hundreds of times over, and the states do
    not change between walks, so re-running the model for each one costs far
    more than the fit itself.  The first pass is kept and every later pass
    reads it back.  Only the blocks a fit actually reads are stored; if they
    would not fit in ``budget`` the stream is passed through untouched.
    """

    def __init__(self, stream: StateStream, layers: Sequence[int], *, device=None,
                 budget: int | None = None, log: Callable = print):
        m = stream.model
        self.stream = stream
        self.device = device or getattr(m, "device", "cpu")
        self.budget = _host_budget() if budget is None else budget
        want = sorted({*[int(l) for l in layers], m.n_layers - 1})
        self._batches: list = []
        size = 0
        for H, n in stream:
            size += sum(H[l].numel() * H[l].element_size() for l in want)
            if size > self.budget:
                self._batches, self.cached = [], False
                log(f"  corpus states would pass {self.budget / 2 ** 30:.0f} GiB, streaming instead")
                return
            self._batches.append(({l: H[l].to("cpu", copy=True) for l in want}, n))
        self.cached = True
        self.bytes = size
        log(f"  corpus states kept in memory: {size / 2 ** 30:.2f} GiB, "
            f"{len(self._batches)} batches, blocks {want[0]}..{want[-1]}")

    def __iter__(self):
        if not self.cached:
            yield from self.stream
            return
        for H, n in self._batches:
            yield {l: v.to(self.device, non_blocking=True) for l, v in H.items()}, n

    def total_tokens(self) -> int:
        return sum(n for _, n in self._batches) if self.cached else self.stream.total_tokens()


def corpus_items(records: Sequence[dict], loader: Callable, *, n: int, square: int = 448,
                 seed: int = 0) -> list[dict]:
    """Take ``n`` records and turn them into model inputs, images loaded.

    The record id travels with the item under ``_id`` so a caller can subtract
    the fitting corpus from whatever it scores.
    """
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(records))[:n]
    out = []
    for i in pick:
        r = records[int(i)]
        out.append(dict(question=r["question"], image=loader(r["image"], square=square),
                        system=r.get("system"), cue=r.get("cue"), _id=r.get("id"),
                        _group=(r.get("meta") or {}).get("image_number")))
    return out


def fit_ids(*item_lists) -> set:
    """Every record id that went into a fit, for excluding from a scored set."""
    out = set()
    for lst in item_lists:
        for it in lst:
            if it.get("_id"):
                out.add(str(it["_id"]))
    return out


# ---------------------------------------------------------------- tuned lens

class Translator(nn.Module):
    """One tuned-lens translator: ``h -> h + W h + b``, ``W`` and ``b`` zeroed.

    Zeroing the update makes the map the identity at initialisation, which is
    what the paper specifies and what the reference implementation does.  A
    rank ``r`` factorises ``W = U V^T`` and gives the anchor lens.
    """

    def __init__(self, d: int, rank: int | None = None):
        super().__init__()
        self.rank = rank
        if rank:
            self.U = nn.Parameter(torch.zeros(d, rank))
            self.V = nn.Parameter(torch.zeros(rank, d))
            nn.init.normal_(self.V, std=d ** -0.5)   # V random, U zero: the product is still zero
        else:
            self.W = nn.Parameter(torch.zeros(d, d))
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        upd = (h @ self.U) @ self.V if self.rank else h @ self.W.T
        return h + upd + self.b

    def affine(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The equivalent ``(A, b)`` with ``A = I + W``, for saving and reuse."""
        d = self.b.shape[0]
        W = (self.U @ self.V).T if self.rank else self.W
        return (torch.eye(d, device=W.device, dtype=W.dtype) + W).detach(), self.b.detach()


#: the published schedule, kept in one place so a deviation has to be explicit.
#: ``lr`` is the reference implementation's ``lr_scale``: its SGD step is
#: ``lr_scale * (1 - momentum)``, so the paper's "learning rate 1" is a step of
#: 0.1 (tuned_lens/scripts/train_loop.py), and :func:`sgd_step` does the same.
PAPER = dict(steps=250, tokens_per_step=2 ** 18, lr=1.0, momentum=0.9, nesterov=True,
             weight_decay=1e-3, clip=1.0, schedule="linear-decay")


def sgd_step(lr_scale: float) -> float:
    """The SGD step the reference tuned lens takes for a given ``lr_scale``."""
    return float(lr_scale) * (1.0 - PAPER["momentum"])


def fit_tuned_lens_cross(m, stream: "ImageToAnswerStream", *, layers: Sequence[int],
                         steps: int = 250, tokens_per_step: int = 2 ** 16,
                         lr: float = PAPER["lr"], weight_decay: float = PAPER["weight_decay"],
                         clip: float = PAPER["clip"], chunk: int = 1024,
                         log: Callable = print) -> dict:
    """The tuned lens's own objective, taken at the answer spot instead of locally.

    Same translator, optimiser and schedule as the local tuned lens, with the
    target distribution read at the answer spot instead of at the patch.
    """
    dev = m.device
    tr = {l: Translator(m.d_model, None).to(dev).float() for l in layers}
    params = [p for t in tr.values() for p in t.parameters()]
    opt = torch.optim.SGD(params, lr=sgd_step(lr), momentum=PAPER["momentum"],
                          nesterov=PAPER["nesterov"], weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1.0 - s / steps))
    it, spent, hist = iter(stream), 0, []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        got, tot = 0, 0.0
        while got < tokens_per_step:
            try:
                X, Y, n = next(it)
            except StopIteration:
                it = iter(stream); X, Y, n = next(it)
            for s0 in range(0, n, chunk):
                sl = slice(s0, min(s0 + chunk, n))
                with torch.no_grad():
                    ref = torch.log_softmax(m.unembed(Y[sl]).float(), -1)
                    p_ = ref.exp()
                w = (sl.stop - sl.start) / tokens_per_step
                for l in layers:
                    lg = torch.log_softmax(
                        m.unembed(tr[l](X[l][sl].float()).to(X[l].dtype)).float(), -1)
                    loss = (p_ * (ref - lg)).sum(-1).mean() * w
                    loss.backward()
                    tot += float(loss.detach())
                del ref, p_
            got += n; spent += n
            del X, Y
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step(); sched.step()
        hist.append(tot)
        if step % 25 == 0 or step == steps - 1:
            log(f"    cross tuned lens step {step + 1}/{steps}  KL {tot:.4f}  "
                f"tokens {spent / 1e6:.2f}M")
    out = {}
    for l, t in tr.items():
        A, b = t.affine()
        out[l] = (A.float().cpu(), b.float().cpu())
    out["meta"] = dict(kind="image token -> answer-spot distribution, tuned lens objective",
                       steps=steps, tokens_per_step=tokens_per_step, lr=lr,
                       tokens_spent=int(spent), kl_last=hist[-1] if hist else None)
    return out


def fit_tuned_lens(m, stream: StateStream, *, layers: Sequence[int] | None = None,
                   steps: int = PAPER["steps"], tokens_per_step: int = PAPER["tokens_per_step"],
                   lr: float = PAPER["lr"], weight_decay: float = PAPER["weight_decay"],
                   clip: float = PAPER["clip"], rank: int | None = None, chunk: int = 1024, log_every: int = 25,
                   init: dict | None = None, log: Callable = print) -> dict:
    """Train one translator per block with the published objective and optimiser.

    All translators are trained at once against the same forward passes, which
    is what makes the token budget affordable: one pass over the corpus feeds
    every layer's loss.  ``rank`` gives the low-rank anchor variant; ``init``
    supplies starting ``(A, b)`` per layer, used only for the diagnostic run
    that starts the baseline from our own closed-form solution.

    Returns ``{layer: (A, b)}`` on the CPU plus a ``meta`` entry recording the
    budget actually spent, so a table can state it.
    """
    dev = m.device
    LAST = m.n_layers - 1
    layers = sorted(layers if layers is not None else range(LAST))
    tr = {l: Translator(m.d_model, rank).to(dev).float() for l in layers}
    if init:
        for l, (A, b) in init.items():
            if l not in tr:
                continue
            with torch.no_grad():
                W = (A.float() - torch.eye(m.d_model)).to(dev)
                if rank:
                    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
                    r = min(rank, S.shape[0])
                    tr[l].U.copy_((U[:, :r] * S[:r]).float()); tr[l].V.copy_(Vh[:r].float())
                else:
                    tr[l].W.copy_(W)
                tr[l].b.copy_(b.float().to(dev))
    params = [p for t in tr.values() for p in t.parameters()]
    opt = torch.optim.SGD(params, lr=sgd_step(lr), momentum=PAPER["momentum"],
                          nesterov=PAPER["nesterov"], weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1.0 - s / steps))

    chunk = int(os.environ.get("VLENS_TL_CHUNK", chunk))   # small GPUs need smaller slices
    it = iter(stream)
    spent, hist = 0, []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        got, tot = 0, 0.0
        while got < tokens_per_step:
            try:
                H, n = next(it)
            except StopIteration:                      # another epoch over the corpus
                it = iter(stream); H, n = next(it)
            # the vocabulary is wide enough that a whole batch of log-softmaxes
            # will not fit, so the tokens are walked in chunks; the gradient is
            # the same, only the peak memory differs.  A block whose backward
            # has already landed is remembered, so waiting for the other lane
            # and trying again resumes at the block that did not fit rather
            # than counting the earlier ones twice.
            for s0 in range(0, n, chunk):
                sl = slice(s0, min(s0 + chunk, n))
                done_l: set = set()

                def chunk_step(sl=sl, done_l=done_l):
                    with torch.no_grad():
                        ref = torch.log_softmax(m.unembed(H[LAST][sl]).float(), -1)
                        p = ref.exp()
                    w = (sl.stop - sl.start) / tokens_per_step
                    out = 0.0
                    for l in layers:
                        if l in done_l:
                            continue
                        lg = torch.log_softmax(
                            m.unembed(tr[l](H[l][sl].float()).to(H[l].dtype)).float(), -1)
                        loss = (p * (ref - lg)).sum(-1).mean() * w
                        loss.backward()
                        out += float(loss.detach())
                        done_l.add(l)
                    del ref, p
                    return out

                tot += with_room(chunk_step, log=log)
            got += n; spent += n
            del H
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, clip))
        opt.step(); sched.step()
        hist.append(tot)
        if step % max(log_every, 1) == 0 or step == steps - 1:
            log(f"    tuned lens step {step + 1}/{steps}  KL {tot / len(layers):.4f} per block  "
                f"lr {sched.get_last_lr()[0]:.4f}  grad norm {gnorm:.3f}  tokens {spent / 1e6:.2f}M")
    maps = {}
    for l, t in tr.items():
        A, b = t.affine()
        maps[l] = (A.float().cpu(), b.float().cpu())
    maps["meta"] = dict(PAPER, steps=steps, tokens_per_step=tokens_per_step, lr=lr,
                        weight_decay=weight_decay, rank=rank, tokens_spent=int(spent),
                        kl_first=hist[0] if hist else None, kl_last=hist[-1] if hist else None,
                        paper="Belrose et al. 2023, arXiv:2303.08112")
    return maps


# ---------------------------------------------------------------- jacobian lens

SKIP_FIRST_N_POSITIONS = 16


def valid_position_mask(seq_len: int, spot: int, *, skip_first: int = SKIP_FIRST_N_POSITIONS):
    """Positions to average the Jacobian over: past the sinks, before the last.

    The reference implementation drops the first 16 positions, which behave as
    attention sinks, and the final position, which has no next-token target.
    Here the "final position" is the row's answer spot, because everything to
    its right is padding.
    """
    mask = torch.zeros(seq_len, dtype=torch.bool)
    hi = max(skip_first, int(spot))
    mask[skip_first:hi] = True
    if mask.sum() == 0:                       # a short prompt: keep what there is
        mask[: max(int(spot), 1)] = True
    return mask


def jacobian_for_item(m, item: dict, source_layers: Sequence[int], *, dim_batch: int = 16,
                      skip_first: int = SKIP_FIRST_N_POSITIONS) -> tuple[dict, int]:
    """``J_l`` for one multimodal prompt, by the reference estimator.

    One forward on the prompt replicated ``dim_batch`` times, graph retained,
    then ``ceil(d_model / dim_batch)`` backward passes.  Batch element ``b``
    carries a one-hot cotangent at output dimension ``dim_start + b``, set at
    every valid target position; the gradient at each source position is
    therefore summed over current and future target positions, and we average
    over source positions.  That is the reduction the paper uses.
    """
    d = m.d_model
    target = m.n_layers - 1
    src = sorted(source_layers)
    J = {l: torch.zeros(d, d, dtype=torch.float32) for l in src}
    b = m.batch_many([item] * dim_batch, side="right")
    spot = int(m.answer_spot(b)[0])
    with ActivationRecorder(m.layers, at=[*src, target], start_graph_at=min(src)) as rec, torch.enable_grad():
        m.forward(b)
        tgt = rec.activations[target]
        srcs = [rec.activations[l] for l in src]
        T = tgt.shape[1]
        pos = valid_position_mask(T, spot, skip_first=skip_first).nonzero(as_tuple=True)[0].to(tgt.device)
        rows = torch.arange(dim_batch, device=tgt.device)
        cot = torch.zeros_like(tgt)
        n_passes = math.ceil(d / dim_batch)
        for k, d0 in enumerate(range(0, d, dim_batch)):
            nd = min(dim_batch, d - d0)
            cot.zero_()
            cot[rows[:nd, None], pos[None, :], d0 + rows[:nd, None]] = 1.0
            grads = torch.autograd.grad(outputs=tgt, inputs=srcs, grad_outputs=cot,
                                        retain_graph=(k < n_passes - 1))
            for l, g in zip(src, grads):
                J[l][d0: d0 + nd, :] = g[:nd, pos.to(g.device), :].float().mean(1).cpu()
            del grads
    return J, int(pos.numel())


def fit_jacobian_lens(m, items: Sequence[dict], *, source_layers: Sequence[int] | None = None,
                      dim_batch: int = 16, snapshots: Sequence[int] | None = None,
                      log: Callable = print) -> dict:
    """Average ``J_l`` over prompts, as a running mean.

    The paper fits on 1000 sequences of 128 tokens of generic text.  A
    multimodal prompt here is several hundred positions, so far fewer prompts
    reach the same number of source positions; the count actually used is
    recorded in ``meta`` and stated wherever the number appears.
    """
    src = sorted(source_layers if source_layers is not None else range(m.n_layers - 1))
    acc = {l: torch.zeros(m.d_model, m.d_model, dtype=torch.float32) for l in src}
    n_pos = 0
    #: the running mean at a few prompt counts, for showing the average has
    #: settled long before the budget runs out
    taken: dict[int, dict] = {}
    want = sorted(set(snapshots or ()))
    for i, it in enumerate(items):
        # a smaller group of cotangents is the same Jacobian, just slower
        def one(it=it):
            db = dim_batch
            while True:
                try:
                    return jacobian_for_item(m, it, src, dim_batch=db)
                except Exception as e:  # noqa: BLE001
                    if not is_oom(e) or db <= 1:
                        raise
                    db = max(db // 2, 1)
                    torch.cuda.empty_cache()
                    log(f"    not enough room; {db} directions at a time")
        J, npos = with_room(one, log=log)
        for l in src:
            acc[l] += J[l]
        n_pos += npos
        if i % 4 == 0 or i == len(items) - 1:
            log(f"    jacobian lens {i + 1}/{len(items)} prompts, {n_pos} source positions")
        if (i + 1) in want:
            taken[i + 1] = {l: (acc[l].clone() / (i + 1), torch.zeros(m.d_model)) for l in src}
            taken[i + 1]["meta"] = dict(n_prompts=i + 1, n_source_positions=n_pos)
        torch.cuda.empty_cache()
    out = {l: (acc[l] / len(items), torch.zeros(m.d_model)) for l in src}
    out["meta"] = dict(n_prompts=len(items), n_source_positions=n_pos, dim_batch=dim_batch,
                       skip_first=SKIP_FIRST_N_POSITIONS,
                       paper="Verbalizable Representations Form a Global Workspace, Anthropic 2026")
    if want:
        out["snapshots"] = taken
    return out


# ---------------------------------------------------------------- latent lens

class LatentLens:
    """A datastore of contextual text representations, queried by cosine similarity.

    Built as the paper builds it: a caption corpus is run through the same
    model and the contextual representation at every position of every caption
    is kept, capped at ``per_token`` representations per vocabulary token per
    layer.  A query state is then matched against the store by cosine
    similarity and described by its ``k`` nearest entries.

    ``distribution`` is our addition: the retrieved tokens, weighted by a
    softmax over their similarities, give a distribution over the vocabulary so
    that the same score functions used for every other instrument apply.  The
    published method stops at the retrieved words.
    """

    def __init__(self, keys: dict[int, torch.Tensor], toks: dict[int, torch.Tensor],
                 vocab: int, meta: dict | None = None):
        self.keys = keys        # {layer: [n, d] float16, L2 normalised}
        self.toks = toks        # {layer: [n] int32}
        self.vocab = vocab
        self.meta = meta or {}

    @property
    def layers(self):
        return sorted(self.keys)

    def nearest(self, h: torch.Tensor, layer: int, k: int = 5):
        """Top-``k`` entries for each row of ``h`` ``[n, d]``: (similarity, token)."""
        K = self.keys[layer].to(h.device)
        q = torch.nn.functional.normalize(h.float(), dim=-1).to(K.dtype)
        sim = q @ K.T
        s, i = sim.topk(min(k, K.shape[0]), dim=-1)
        return s.float(), self.toks[layer].to(h.device)[i]

    def similarity_logits(self, h: torch.Tensor, layer: int,
                          floor: float | None = None) -> torch.Tensor:
        """A vocabulary-wide score from the store: per token, its best match to ``h``.

        The published method returns the ``k`` nearest entries as words.  To put
        it on the same footing as the other instruments it has to produce a
        number for whichever tokens a score asks about, and those tokens are
        rarely among the five nearest.  So every entry votes for its own token
        and a token's score is the best cosine similarity any of its entries
        achieves; tokens the store never saw get ``floor``.  This uses the same
        reference space and the same similarity the paper does, and it is the
        most generous reading of it that a scored comparison allows.  It is our
        adaptation, and is labelled as such wherever the number appears.
        """
        # the store reaches a few per cent of the vocabulary, so the floor is
        # what almost every token gets; it is chosen on held-out prompts
        floor = self.meta.get("floor", -1.0) if floor is None else floor
        K = self.keys[layer].to(h.device)
        q = torch.nn.functional.normalize(h.float(), dim=-1).to(K.dtype)
        sim = (q @ K.T).float()                        # [n, entries]
        t = self.toks[layer].to(h.device).long()
        out = torch.full((h.shape[0], self.vocab), float(floor), device=h.device)
        out.scatter_reduce_(1, t[None, :].expand(h.shape[0], -1), sim, reduce="amax",
                            include_self=True)
        return out

    def distribution(self, h: torch.Tensor, layer: int, k: int = 5,
                     temp: float = 0.05, floor: float | None = None):
        """The similarity scores as a distribution, for the divergence measure.

        ``temp`` and ``floor`` are both chosen on held-out prompts rather than
        fixed, so the baseline is not penalised for a scale we happened to pick.
        """
        return torch.softmax(self.similarity_logits(h, layer, floor) / temp, dim=-1)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(dict(keys={l: v.cpu() for l, v in self.keys.items()},
                        toks={l: v.cpu() for l, v in self.toks.items()},
                        vocab=self.vocab, meta=self.meta), path)
        return path

    @classmethod
    def load(cls, path: str) -> "LatentLens":
        d = torch.load(path, map_location="cpu")
        return cls(d["keys"], d["toks"], d["vocab"], d.get("meta"))


def caption_corpus(m, items, *, prompt: str = "Describe this image in one sentence.",
                   gen: int = 40, batch: int = 8, log=print) -> list[str]:
    """The model's own one-sentence descriptions of the given pictures."""
    import torch
    tok = m.tokenizer
    out = []
    for s in range(0, len(items), batch):
        chunk = list(items[s: s + batch])
        pack = [dict(question=prompt, image=it["image"], system=None, cue=None) for it in chunk]
        b = m.batch_many(pack, side="left")
        with torch.no_grad():
            g = m.hf.generate(**{k: v for k, v in b.items() if not k.startswith("_")},
                              max_new_tokens=gen, do_sample=False,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
        n_in = b["input_ids"].shape[1]
        for j in range(g.shape[0]):
            t = tok.decode(g[j, n_in:], skip_special_tokens=True).strip()
            if len(t.split()) >= 4:
                out.append(t)
        if (s // batch) % 20 == 0:
            log(f"    captions {min(s + batch, len(items))}/{len(items)}")
    return out


def latent_corpus(m, spare: Sequence[dict], loader: Callable, *, results: str | None = None,
                  model_name: str = "", n: int = 6000, square: int = 448,
                  log: Callable = print) -> list[str]:
    """The strings the LatentLens datastore is built from.

    The paper builds its store from Visual Genome captions -- descriptive
    sentences about images.  Those are not on this machine, so the closest
    thing available is used: this model's own one-sentence descriptions of the
    held-back pictures, its open-ended replies from the behaviour pass, and the
    dataset's own prompt text.  Only replies to records that are themselves
    outside every scored set are taken, so the store never holds a scored
    item's own words; that costs the baseline some corpus and is the price of
    the comparison being clean.
    """
    import glob as _glob
    import json as _json
    cap_items = corpus_items(spare, loader, n=min(n, 400), square=square, seed=11)
    caps, seen = list(caption_corpus(m, cap_items, log=log)), set()
    spare_ids = {str(r.get("id")) for r in spare}

    def add(t):
        t = (t or "").strip()
        if len(t.split()) >= 4 and t not in seen:
            seen.add(t); caps.append(t)

    n_reply = 0
    if results and model_name:
        for p_ in sorted(_glob.glob(f"{results}/{model_name}__*.json")):
            try:
                d_ = _json.load(open(p_))
            except Exception:  # noqa: BLE001
                continue
            for r in d_.get("rows", []):
                if str(r.get("id")) in spare_ids:
                    add(r.get("reply")); n_reply += 1
    for r in spare:
        add(r.get("cue")); add(r.get("question"))
    rng = np.random.default_rng(0)
    caps = [caps[i] for i in rng.permutation(len(caps))]
    log(f"  latent lens corpus: {len(caps)} distinct strings "
        f"({n_reply} of them replies to held-out records), using {min(len(caps), n)}")
    return caps[:n], cap_items


def build_latent_lens(m, captions: Sequence[str], *, layers: Sequence[int],
                      per_token: int = 20, batch: int = 16, log: Callable = print) -> LatentLens:
    """Run the caption corpus through the model and keep the contextual states.

    ``per_token`` caps how many representations any one vocabulary token
    contributes at a layer, which is the cap the paper uses and what keeps the
    store from being dominated by function words.
    """
    tok = m.tokenizer
    keys = {l: [] for l in layers}
    toks = {l: [] for l in layers}
    count = {l: {} for l in layers}
    for s in range(0, len(captions), batch):
        chunk = list(captions[s: s + batch])
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=64)
        enc = {k: v.to(m.device) for k, v in enc.items()}
        with torch.no_grad(), ActivationRecorder(m.layers, at=list(layers)) as rec:
            m.forward(enc)          # text only: the trunk takes ids and a mask
            for l in layers:
                H = rec.activations[l]
                mask = enc["attention_mask"].bool()
                ids = enc["input_ids"][mask]
                V = torch.nn.functional.normalize(H[mask].float(), dim=-1).half().cpu()
                c = count[l]
                sel = []
                for j, t in enumerate(ids.tolist()):
                    if c.get(t, 0) < per_token:
                        c[t] = c.get(t, 0) + 1; sel.append(j)
                if sel:
                    sel = torch.tensor(sel)
                    keys[l].append(V[sel]); toks[l].append(ids.cpu()[sel].int())
        if (s // batch) % 20 == 0:
            log(f"    latent lens {s + len(chunk)}/{len(captions)} captions, "
                f"{sum(len(x) for x in keys[layers[0]])} entries at layer {layers[0]}")
    K = {l: torch.cat(keys[l]) for l in layers if keys[l]}
    T = {l: torch.cat(toks[l]) for l in layers if toks[l]}
    vocab = int(m._lm_head.weight.shape[0])   # padded, so wider than len(tok)
    return LatentLens(K, T, vocab, dict(n_captions=len(captions), per_token=per_token,
                                           entries={l: int(len(v)) for l, v in K.items()},
                                           paper="LatentLens, arXiv:2602.00462",
                                           note="scored readout is our adaptation"))


# ---------------------------------------------------------------- saving a fit

def save_maps(maps: dict, path: str) -> str:
    """Save ``{layer: (A, b)}`` plus ``meta``, halved on disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta = maps.get("meta", {})
    torch.save({"maps": {int(k): (maps[k][0], maps[k][1]) for k in maps if k != "meta"},
                "meta": meta}, path)
    with open(path.replace(".pt", ".json"), "w") as f:
        json.dump(meta, f, indent=1, default=float)
    return path


def load_maps(path: str) -> dict:
    d = torch.load(path, map_location="cpu")
    out = {int(k): (A.float(), b.float()) for k, (A, b) in d["maps"].items()}
    out["meta"] = d.get("meta", {})
    return out


# ---------------------------------------------------------------- streaming ridge

class RidgeAccumulator:
    """Sufficient statistics for the V-Read map, accumulated over a stream.

    A closed-form fit ordinarily needs every state in memory at once, which a
    corpus of tens of millions of tokens does not allow.  Ridge only ever uses
    ``X^T X`` and ``X^T Y``, both ``d x d``, so they are accumulated batch by
    batch and the solve happens once at the end.  The result is the exact fit
    on the whole corpus, which means V-Read is fitted on precisely the tokens
    the tuned lens is trained on and neither instrument sees more data than the
    other.
    """

    def __init__(self, layers: Sequence[int], d: int, device="cpu"):
        self.layers = list(layers)
        self.d = d
        self.G = {l: torch.zeros(d, d, dtype=torch.float64, device=device) for l in self.layers}
        self.C = {l: torch.zeros(d, d, dtype=torch.float64, device=device) for l in self.layers}
        self.sx = {l: torch.zeros(d, dtype=torch.float64, device=device) for l in self.layers}
        self.sy = torch.zeros(d, dtype=torch.float64, device=device)
        self.n = 0

    def add_xy(self, X: dict, Y: torch.Tensor) -> None:
        """Accumulate against an explicit target, for a cross-position fit."""
        Y = Y.double()
        self.sy += Y.sum(0)
        for l in self.layers:
            Xl = X[l].double()
            self.G[l] += Xl.T @ Xl
            self.C[l] += Xl.T @ Y
            self.sx[l] += Xl.sum(0)
        self.n += Y.shape[0]

    def add(self, H: dict, final_layer: int) -> None:
        Y = H[final_layer].double()
        self.sy += Y.sum(0)
        for l in self.layers:
            X = H[l].double()
            self.G[l] += X.T @ X
            self.C[l] += X.T @ Y
            self.sx[l] += X.sum(0)
        self.n += Y.shape[0]

    def solve_many(self, lams) -> dict:
        """One solution per penalty, so the penalty can be chosen on held-out data."""
        return {float(l): self.solve(lam=float(l)) for l in lams}

    @property
    def mu_y(self) -> torch.Tensor:
        """The mean target state: what a ridge solution is pulled towards."""
        return (self.sy / float(self.n)).float().cpu()

    def solve(self, lam: float = 0.1, toward_identity: bool = False) -> dict:
        """``{layer: (A, b)}``, the centred ridge solution on everything seen.

        ``toward_identity`` penalises ``||A - I||`` instead of ``||A||``, which
        is the tuned lens's own parameterisation -- it learns an update to the
        state rather than a map from scratch, so its penalty pulls towards
        leaving the state alone.  Regressing ``Y - X`` on ``X`` and adding the
        identity back gives the same thing in closed form.  At ``lam = 0`` the
        two are the same solution.
        """
        out = {}
        n = float(self.n)
        my = (self.sy / n)

        def one(l, dev):
            # a dense solve at the model's width wants several hundred
            # megabytes of workspace, and the card is usually shared. The
            # accumulator itself is the expensive thing to build; where the
            # solve happens is not, so a full card falls back to the host
            # rather than losing the pass that filled it.
            I = torch.eye(self.d, dtype=torch.float64, device=dev)
            mx = self.sx[l].to(dev) / n
            G = self.G[l].to(dev) - n * torch.outer(mx, mx)
            C = self.C[l].to(dev) - n * torch.outer(mx, my.to(dev))
            lam_eff = lam * G.diagonal().mean()
            if toward_identity:
                W = torch.linalg.solve(G + lam_eff * I, C - G).T
                return I + W, my.to(dev) - mx - W @ mx
            A = torch.linalg.solve(G + lam_eff * I, C).T
            return A, my.to(dev) - A @ mx

        dev = self.G[self.layers[0]].device
        for l in self.layers:
            try:
                A, b = one(l, dev)
            except Exception as e:  # noqa: BLE001
                if not room.is_oom(e):
                    raise
                torch.cuda.empty_cache()
                A, b = one(l, "cpu")
            out[l] = (A.float().cpu(), b.float().cpu())
        out["meta"] = dict(kind="v_read/ridge", lam=lam, tokens=int(self.n),
                           toward_identity=bool(toward_identity))
        return out


# ---------------------------------------------------------------- held-out KL

@torch.no_grad()
def heldout_kl(m, stream: StateStream, layers: Sequence[int], maps: dict,
               latent: "LatentLens | None" = None, chunk: int = 1024) -> tuple[dict, int]:
    """Mean ``D_KL(p_model || p_lens)`` per layer for every instrument at once.

    This is the tuned lens paper's own headline number and needs no labels, so
    it is the cleanest way to rank instruments.  ``maps`` is
    ``{name: {layer: (A, b)}}``; the logit lens is added for free as the
    identity and never has to be passed in.
    """
    LAST = m.n_layers - 1
    chunk = int(os.environ.get("VLENS_KL_CHUNK", chunk))   # small GPUs need smaller slices
    names = [*maps, "logit", *(["latent"] if latent is not None else [])]
    acc = {k: {l: 0.0 for l in layers} for k in names}
    ntok = 0
    for H, n in stream:
        for s0 in range(0, n, chunk):
            sl = slice(s0, min(s0 + chunk, n))
            ref = torch.log_softmax(m.unembed(H[LAST][sl]).float(), -1)
            p = ref.exp()
            for l in layers:
                h = H[l][sl].float()
                cand = {"logit": h}
                for k, M in maps.items():
                    A, b = M[l]
                    cand[k] = h @ A.T.to(h.device, h.dtype) + b.to(h.device, h.dtype)
                for k, v in cand.items():
                    lg = torch.log_softmax(m.unembed(v.to(H[l].dtype)).float(), -1)
                    acc[k][l] += float((p * (ref - lg)).sum(-1).sum())
                if latent is not None:
                    q = latent.distribution(h, l, temp=latent.meta.get("temp", 0.05)).clamp_min(1e-9)
                    lg = (q / q.sum(-1, keepdim=True)).log()
                    acc["latent"][l] += float((p * (ref - lg)).sum(-1).sum())
            del ref, p
        ntok += n
        del H
    for k in acc:
        for l in acc[k]:
            acc[k][l] /= max(ntok, 1)
    return acc, ntok


def pick_lr(m, stream: StateStream, layers: Sequence[int], held: StateStream, *,
            candidates=(1.0, 0.1, 0.01), decays=(1e-4, 1e-3, 1e-1),
            probe_steps: int = 40, tokens_per_step: int = 2 ** 16,
            probe_layers: int | None = 4,
            rank: int | None = None, log: Callable = print):
    """Choose the tuned lens learning rate by held-out KL.

    Rather than report a baseline we have handicapped, a short probe is run at
    each candidate and the one with the lowest held-out divergence is used for
    the real fit.  The published value stays first in the list, so if it wins
    it is what gets used.

    The published protocol uses one learning rate for every block, so the probe
    runs on an evenly spaced few of them rather than all; the fit that follows
    uses every block.
    """
    probe = list(layers)
    if probe_layers and len(probe) > probe_layers:
        probe = [probe[int(round(i))] for i in
                 np.linspace(0, len(probe) - 1, probe_layers)]
        probe = sorted(set(probe))
        log(f"    choosing the learning rate on blocks {probe} of {len(layers)}")
    best, best_kl = None, float("inf")
    for lr in candidates:
        for wd in decays:
            M = fit_tuned_lens(m, stream, layers=probe, steps=probe_steps,
                               tokens_per_step=min(tokens_per_step, 2 ** 16), lr=lr,
                               rank=rank, weight_decay=wd, log=lambda *x: None)
            kl, _ = heldout_kl(m, held, probe, {"t": {l: M[l] for l in probe}})
            v = float(np.mean([kl["t"][l] for l in probe]))
            log(f"    lr {lr:<6} weight decay {wd:<6} -> held-out KL {v:.4f}")
            if np.isfinite(v) and v < best_kl:
                best, best_kl = (lr, wd), v
    if best is None:
        best = (candidates[-1], decays[0])
        log(f"    every candidate diverged; using lr {best[0]}, weight decay {best[1]}")
    else:
        log(f"    using lr {best[0]}, weight decay {best[1]} (held-out KL {best_kl:.4f})")
    return best


def apply_gain(M: dict, layers: Sequence[int], mu_y: torch.Tensor, gain) -> dict:
    """Push a ridge map away from the mean target state by ``gain``.

    ``h -> mu + g (A h + b - mu)`` is affine again, so the result is still one
    ``(A, b)`` per layer and nothing downstream has to know.
    """
    g = (lambda l: float(gain[l])) if isinstance(gain, dict) else (lambda l: float(gain))
    mu = mu_y.float()
    return {l: (g(l) * M[l][0], g(l) * M[l][1] + (1.0 - g(l)) * mu) for l in layers}


def mean_last_state(m, held: StateStream) -> torch.Tensor:
    """The mean final-block state over the held-out split, the point a gain pivots on."""
    tot, n = None, 0
    for H, k in held:
        x = H[m.n_layers - 1].float().sum(0)
        tot = x if tot is None else tot + x
        n += k
    return (tot / max(n, 1)).to(m.device)


def pick_gain(m, M: dict, held: StateStream, layers: Sequence[int], mu_y: torch.Tensor,
              candidates=(1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0),
              log: Callable = print) -> tuple[dict, dict]:
    """Undo the shrinkage a ridge fit builds in, chosen on held-out KL.

    Least squares pulls its solution towards the mean target state, and the
    pull that is best for squared error is not the one that is best for the
    distribution the state decodes to.  The tuned lens never pays this because
    it optimises the divergence directly; one scalar per block, chosen on the
    same held-out prompts, puts the two on the same footing.
    """
    best = {int(l): (1.0, float("inf")) for l in layers}
    for g in candidates:
        cand = apply_gain(M, layers, mu_y, g)
        kl, _ = heldout_kl(m, held, layers, {"v": cand})
        for l in layers:
            v = float(kl["v"][l])
            if np.isfinite(v) and v < best[int(l)][1]:
                best[int(l)] = (g, v)
        log(f"    gain {g:<5} -> held-out KL {np.mean([kl['v'][l] for l in layers]):.4f}")
    chosen = {int(l): best[int(l)][0] for l in layers}
    kl_at = float(np.mean([best[int(l)][1] for l in layers]))
    log(f"    using gain {sorted(set(chosen.values()))} (held-out KL {kl_at:.4f})")
    return apply_gain(M, layers, mu_y, chosen), dict(gain=chosen, gain_grid=list(candidates),
                                                     heldout_kl=kl_at)


def pick_lam(m, acc: "RidgeAccumulator", held: StateStream, layers: Sequence[int],
             candidates=(1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
             gain_grid=(1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0),
             log: Callable = print) -> tuple[float, dict]:
    """Choose V-Read's penalty and gain by held-out KL, the way the tuned lens
    learning rate is chosen, so both instruments get the same treatment."""
    # the tuned lens fits an independent translator per block, so the penalty
    # is chosen per block too rather than shared across them
    per = {int(l): (None, float("inf"), False) for l in layers}
    for toward in (False, True):
        for lam in candidates:
            M = acc.solve(lam=lam, toward_identity=toward)
            kl, _ = heldout_kl(m, held, layers, {"v": {l: M[l] for l in layers}})
            v = float(np.mean([kl["v"][l] for l in layers]))
            log(f"    lambda {lam:<7} {'towards the identity' if toward else 'towards zero':<20}"
                f" -> held-out KL {v:.4f}")
            for l in layers:
                x = float(kl["v"][l])
                if np.isfinite(x) and x < per[int(l)][1]:
                    per[int(l)] = (lam, x, toward)
    combos = sorted({(per[int(l)][0], per[int(l)][2]) for l in layers},
                    key=lambda c: (c[1], c[0] if c[0] is not None else 0.0))
    solved = {c: acc.solve(lam=c[0], toward_identity=c[1]) for c in combos if c[0] is not None}
    best_maps = {l: solved[(per[int(l)][0], per[int(l)][2])][l] for l in layers}
    best_kl = float(np.mean([per[int(l)][1] for l in layers]))
    lam_per = {int(l): per[int(l)][0] for l in layers}
    toward_per = {int(l): bool(per[int(l)][2]) for l in layers}
    best = max(set(lam_per.values()), key=list(lam_per.values()).count)
    log(f"    using lambda {sorted(set(lam_per.values()))}, "
        f"{'towards the identity' if any(toward_per.values()) else 'towards zero'}"
        f"{' on some blocks' if len(set(toward_per.values())) > 1 else ''} "
        f"(held-out KL {best_kl:.4f})")
    meta = dict(kind="v_read/ridge", tokens=int(acc.n), lam=best, lam_grid=list(candidates),
                lam_per_layer=lam_per, toward_identity=toward_per,
                heldout_kl_before_gain=best_kl)
    if gain_grid:
        tuned_maps, gmeta = pick_gain(m, best_maps, held, layers, acc.mu_y,
                                      candidates=tuple(gain_grid), log=log)
        best_maps = dict(tuned_maps)
        meta = dict(meta, **gmeta)
    best_maps["meta"] = meta
    return best, best_maps


def pick_temp(m, latent: "LatentLens", held: StateStream, layers: Sequence[int],
              candidates=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
              floors=(-1.0, -0.5, 0.0, 0.25, 0.5, 0.75),
              log: Callable = print) -> tuple[float, float]:
    """The temperature and floor that turn LatentLens similarities into a
    distribution, chosen on held-out prompts so its divergence is measured at
    its own best setting rather than one we happened to pick.

    The floor is the score every token the store never saw receives, and the
    store covers a few per cent of the vocabulary, so it is the larger of the
    two knobs: leave it fixed and the measurement is mostly a measurement of
    how much of the vocabulary the corpus happened to reach.
    """
    LAST = m.n_layers - 1
    best, best_kl = (candidates[0], floors[0]), float("inf")
    for f in floors:
        for t in candidates:
            tot, ntok = 0.0, 0
            with torch.no_grad():
                for H, n in held:
                    for s0 in range(0, n, 512):
                        sl = slice(s0, min(s0 + 512, n))
                        ref = torch.log_softmax(m.unembed(H[LAST][sl]).float(), -1)
                        p = ref.exp()
                        for l in layers:
                            q = latent.distribution(H[l][sl].float(), l, temp=t,
                                                    floor=f).clamp_min(1e-9)
                            tot += float((p * (ref - q.log())).sum(-1).sum())
                        ntok += (sl.stop - sl.start) * len(layers)
                    del H
            v = tot / max(ntok, 1)
            log(f"    latent temp {t:<6} floor {f:<6} -> held-out KL {v:.4f}")
            if np.isfinite(v) and v < best_kl:
                best, best_kl = (t, f), v
    log(f"    using latent temperature {best[0]} and floor {best[1]} "
        f"(held-out KL {best_kl:.4f})")
    return best


# ---------------------------------------------------------- cross-position fits

class ImageToAnswerStream:
    """Image-token states at each block, paired with the answer spot's final state.

    Streams the pairs a cross-position map is fitted on. The target is the
    model's own final state at its answer spot, so no labels are used.
    """

    def __init__(self, model, items: Sequence[dict], batch: int = 4, seed: int = 0,
                 max_patches: int | None = 256):
        self.model, self.items, self.batch = model, items, batch
        self.max_patches = max_patches
        self._order = np.random.default_rng(seed).permutation(len(items))

    def __iter__(self):
        yield from _stream_batches(self, None)

    def _run(self, idx, _at=None):
        m = self.model
        LAST = m.n_layers - 1
        if True:
            chunk = [self.items[i] for i in idx]
            b = m.batch_many(chunk, side="right")
            spot = m.answer_spot(b)
            mask = m.image_mask(b["input_ids"])
            at = None
            with torch.no_grad(), ActivationRecorder(m.layers, at=list(range(m.n_layers))) as rec:
                m.forward(b)
                H = {l: rec.activations[l].detach() for l in range(m.n_layers)}
            Xs, Ys, n = {}, [], 0
            for r in range(len(chunk)):
                pos = mask[r].nonzero(as_tuple=True)[0]
                if pos.numel() == 0:
                    continue
                if self.max_patches and pos.numel() > self.max_patches:
                    # keyed to the record, not to the batch it landed in, so the
                    # same patches are kept however the batches are grouped
                    g = torch.Generator().manual_seed(int(self.seed) * 1_000_003 + int(idx[r]))
                    keep = torch.randperm(pos.numel(), generator=g)[: self.max_patches]
                    pos = pos[keep.sort().values.to(pos.device)]
                y = H[LAST][r, spot[r].to(H[LAST].device)]
                Ys.append(y[None].expand(pos.numel(), -1))
                for l in range(m.n_layers):
                    Xs.setdefault(l, []).append(H[l][r, pos])
                n += int(pos.numel())
            if not Ys:
                return None
            out = ({l: torch.cat(v) for l, v in Xs.items()}, torch.cat(Ys), n)
            del H
            return out


class CachedPairs:
    """One pass over an ``ImageToAnswerStream``, held in host memory and replayed.

    The cross-position fit walks its corpus as many times over as the local one
    does, and the pairs do not change between walks. Only the blocks a fit
    reads are stored; past the budget the stream is passed through untouched.
    """

    def __init__(self, stream: "ImageToAnswerStream", layers: Sequence[int], *, device=None,
                 budget: int | None = None, log: Callable = print):
        self.stream = stream
        self.device = device or getattr(stream.model, "device", "cpu")
        self.budget = _host_budget() if budget is None else budget
        want = [int(l) for l in layers]
        self._batches: list = []
        size = 0
        for X, Y, n in stream:
            size += sum(X[l].numel() * X[l].element_size() for l in want)
            size += Y.numel() * Y.element_size()
            if size > self.budget:
                self._batches, self.cached = [], False
                log(f"  cross pairs would pass {self.budget / 2 ** 30:.0f} GiB, streaming instead")
                return
            self._batches.append(({l: X[l].to("cpu", copy=True) for l in want},
                                  Y.to("cpu", copy=True), n))
        self.cached = True
        self.bytes = size
        log(f"  cross pairs kept in memory: {size / 2 ** 30:.2f} GiB, {len(self._batches)} batches")

    def __iter__(self):
        if not self.cached:
            yield from self.stream
            return
        for X, Y, n in self._batches:
            yield ({l: v.to(self.device, non_blocking=True) for l, v in X.items()},
                   Y.to(self.device, non_blocking=True), n)


def pick_cross(m, fit_items, held_items, layers, *, fit_batch: int = 8,
               lams=(1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0), lrs=(1.0, 0.1, 0.01),
               steps: int = 60, tokens_per_step: int = 2 ** 16, log: Callable = print):
    """Choose the ridge penalty and the learning rate for the cross-position pair.

    Both are selected by squared error against the answer-spot state on pairs
    neither was fitted on, so the two sides are searched the same way.
    """
    HX, HY = [], []
    for X, Y, n in ImageToAnswerStream(m, held_items, batch=fit_batch, seed=5):
        HX.append({l: X[l].float().cpu() for l in layers}); HY.append(Y.float().cpu())

    def err(maps):
        tot = 0.0
        for Xc, Yc in zip(HX, HY):
            for l in layers:
                A, b = maps[l]
                tot += float(((Xc[l] @ A.T + b) - Yc).pow(2).mean())
        return tot / max(len(HX) * len(layers), 1)

    fit_stream = CachedPairs(ImageToAnswerStream(m, fit_items, batch=fit_batch, seed=1),
                             layers, log=log)
    acc = RidgeAccumulator(layers, m.d_model, device=m.device)
    for X, Y, n in fit_stream:
        acc.add_xy({l: X[l].float() for l in layers}, Y.float())
    best_lam, best_e, best_maps = None, None, None
    for lam in lams:
        M = acc.solve(lam=lam)
        e = err(M)
        log(f"    cross ridge lambda {lam:<7} -> held-out error {e:.4f}")
        if best_e is None or e < best_e:
            best_lam, best_e, best_maps = lam, e, M
    best_lr, best_le, best_t = None, None, None
    for lr in lrs:
        T = fit_tuned_lens_cross(m, fit_stream, layers=layers, steps=steps,
                                 tokens_per_step=tokens_per_step, lr=lr, log=lambda *x: None)
        e = err({l: T[l] for l in layers})
        log(f"    cross tuned lr {lr:<7} -> held-out error {e:.4f}")
        if best_le is None or e < best_le:
            best_lr, best_le, best_t = lr, e, T
    log(f"    cross: lambda {best_lam}, tuned lr {best_lr}")
    return best_maps, best_lr
