# SPDX-License-Identifier: Apache-2.0
"""Choose batch sizes from the card you actually have, not the one we had.

Every experiment here carries hand-tuned ``--chunk``, ``--ig-steps`` and
``--mean-images`` numbers, and they were tuned against a 24 GB card holding a
2B model.  On a 96 GB card they leave most of the memory idle; on a 32B model
they OOM.  Both failures cost a run, and the second cost several.

Two things are offered.  :func:`plan` estimates what will fit from the model's
own dimensions, which is enough to size a queue before anything is launched.
:func:`fit_chunk` measures instead of estimating -- it runs the real call with a
chunk size, halves it on an out-of-memory error, and returns what worked.  The
estimate is for planning; the measurement is for running, because an estimate
that is wrong by 20% still loses the job.

The estimate is deliberately crude.  A path chunk holds one copy of the graph
from the read layer to the end of the stack, so its cost is roughly

    chunk x seq x d_model x (n_layers - read_layer) x bytes x overhead

with an overhead factor covering attention workspace, the recorder's own copies
and allocator fragmentation.  Fragmentation is the reason for the margin rather
than a tighter formula: the allocator will refuse a request long before the
card is full, and ``expandable_segments`` only softens that.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def free_bytes(device: int = 0) -> int:
    """Free VRAM, from the driver rather than from torch's cache."""
    if not torch.cuda.is_available():
        return 0
    free, _total = torch.cuda.mem_get_info(device)
    return int(free)


def total_bytes(device: int = 0) -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.mem_get_info(device)[1])


@dataclass
class Plan:
    """What the card can hold, for one experiment shape."""

    free_gb: float
    weights_gb: float
    per_row_gb: float
    chunk: int
    steps: int
    mean_images: int
    region_batch: int

    def flags(self) -> str:
        """The command-line flags this plan implies."""
        return (f"--ig-steps {self.steps} --chunk {self.region_batch} "
                f"--mean-images {self.mean_images}")

    def __str__(self) -> str:
        return (f"{self.free_gb:.0f} GB free, weights {self.weights_gb:.0f} GB, "
                f"{self.per_row_gb:.2f} GB per path row -> chunk {self.chunk}, "
                f"steps {self.steps}, regions/batch {self.region_batch}")


def plan(model, *, seq_len: int, read_layer: int, steps: int = 8,
         margin: float = 0.25, device: int = 0) -> Plan:
    """Size a run from the model's dimensions and the card's free memory.

    ``margin`` is held back from the free figure.  A quarter sounds generous
    and is not: the allocator fragments, the vision tower's workspace is not in
    this arithmetic, and a run that dies at 90% through has cost more than the
    memory it saved.
    """
    d = int(getattr(model, "d_model", 0) or 0)
    n_layers = int(getattr(model, "n_layers", 0) or 0)
    if not d or not n_layers:
        raise ValueError("model must expose d_model and n_layers")

    weights = sum(p.numel() * p.element_size()
                  for p in getattr(model, "hf", model).parameters())
    depth_kept = max(n_layers - read_layer, 1)

    # one path row: the graph from the read layer down, in bf16, with room for
    # attention workspace and the recorder's own copies
    OVERHEAD = 6.0
    per_row = seq_len * d * depth_kept * 2 * OVERHEAD

    free = free_bytes(device)
    usable = max(free * (1.0 - margin), 0.0)
    chunk = max(int(usable // max(per_row, 1)), 1)
    chunk = min(chunk, steps)

    # covering regions is a plain forward, so it is far cheaper per row
    region_row = seq_len * d * n_layers * 2 * 1.5
    region_batch = max(min(int(usable // max(region_row, 1)), 64), 1)

    # the corpus mean is one forward per photograph and accumulates on the CPU,
    # so the card does not bound it; more photographs is simply a better mean
    mean_images = 120 if free > 60e9 else (90 if free > 40e9 else 60)

    return Plan(free_gb=free / 1e9, weights_gb=weights / 1e9,
                per_row_gb=per_row / 1e9, chunk=chunk, steps=steps,
                mean_images=mean_images, region_batch=region_batch)


def fit_chunk(call, chunk: int, *, floor: int = 1, label: str = "") -> tuple:
    """Run ``call(chunk)``, halving on OOM until it fits or the floor is hit.

    Returns ``(result, chunk_used)``.  Measuring beats estimating: the estimate
    above is good enough to plan a queue and not good enough to bet a run on,
    because the allocator's behaviour under fragmentation is not a function of
    the dimensions.
    """
    while True:
        try:
            return call(chunk), chunk
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk <= floor:
                raise
            chunk = max(chunk // 2, floor)
            print(f"    {label or 'call'}: out of memory, retrying at "
                  f"chunk {chunk}", flush=True)


def describe(model, *, seq_len: int, read_layer: int) -> str:
    """A one-line summary for a log header."""
    p = plan(model, seq_len=seq_len, read_layer=read_layer)
    d = int(getattr(model, "d_model", 0))
    n = int(getattr(model, "n_layers", 0))
    return (f"{n} layers, d_model {d}, {seq_len} tokens, read at {read_layer}"
            f"  |  {p}")
