# SPDX-License-Identifier: Apache-2.0
"""Making room on a shared card.

Two lanes run on one GPU and their heavy phases do not line up, so a moment
when nothing fits is usually a moment rather than a wall. Everything here
follows the same policy: make the piece of work smaller if it can be made
smaller, and otherwise wait for the other lane and try again. A job that has
already spent an hour is worth more than the minutes spent waiting, and none
of it changes the arithmetic -- a smaller batch is the same states, a smaller
group of cotangents is the same Jacobian.
"""
from __future__ import annotations

import os
import time
from typing import Callable

import torch


#: A full card does not announce itself the same way twice. Torch raises
#: ``OutOfMemoryError``, Triton raises a plain ``RuntimeError`` whose message
#: says so, and the allocator's own mapping failure is a third. Catching only
#: the first lets the other two end a job that had hours in it.
#: The card being full reaches us under several names. cuBLAS and cuDNN
#: report a failed workspace allocation as a handle-creation failure, and
#: Triton reports it as a plain runtime error; none of them say "out of
#: memory" in the words torch does.
_OOM_TEXT = ("out of memory", "outofmemory", "memory mapping failed",
             "cuda error: out of memory", "cublas_status_alloc_failed",
             "cudnn_status_alloc_failed", "cublas_status_not_initialized",
             "cudaerrormemoryallocation")


def is_oom(exc: BaseException) -> bool:
    """Whether an exception is the card being full, however it was raised.

    A caller should know that recovering in-process is only sometimes
    possible: a failed allocation torch raised itself leaves the context
    usable, while one that surfaced from cuBLAS during a backward pass often
    does not, and the honest recovery there is a new process.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, (RuntimeError, MemoryError, getattr(torch, "AcceleratorError", ()))):
        return any(t in str(exc).lower() for t in _OOM_TEXT)
    return False


def with_room(fn: Callable, *, tries: int = 5, wait: float = 45.0,
              log: Callable | None = None):
    """Run ``fn``; if the card is full, wait for the other lane and try again."""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if not is_oom(e) or k == tries - 1:
                raise
            torch.cuda.empty_cache()
            if log:
                log(f"    the card is full; waiting {wait:.0f}s for room ({k + 1}/{tries - 1})")
            time.sleep(wait)


def free_mib() -> int:
    """Free memory on the card, in MiB, or -1 where there is no card."""
    if not torch.cuda.is_available():
        return -1
    free, _total = torch.cuda.mem_get_info()
    return int(free // (1024 * 1024))


def wait_for_free(mib: int, *, minutes: float = 90.0, every: float = 60.0,
                  log: Callable | None = None) -> bool:
    """Wait until ``mib`` is free, or give up and let the caller try anyway.

    A failed allocation inside a backward pass can leave the CUDA context
    unusable, so retrying in the same process does not always work and a job
    that starts into a full card can lose an hour of accumulation before it
    finds out. Waiting for room before the model is even loaded costs nothing
    and is the only recovery that always works.
    """
    if mib <= 0:
        return True
    end = time.time() + minutes * 60.0
    said = False
    while True:
        got = free_mib()
        if got < 0 or got >= mib:
            if said and log:
                log(f"  {got} MiB free now; starting")
            return True
        if time.time() >= end:
            if log:
                log(f"  still only {got} MiB free after {minutes:.0f} min; starting anyway")
            return False
        if not said and log:
            log(f"  waiting for {mib} MiB on the card; {got} MiB free")
            said = True
        time.sleep(every)


def weights_mib(repo: str, hf_home: str | None = None) -> int:
    """How much the weights of ``repo`` take on disk, in MiB, or 0 if unknown.

    A fit needs the model resident plus room to work, and the model's size is
    the one part of that which can be known before anything is loaded.
    """
    import glob as _glob
    home = hf_home or os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    d = os.path.join(home, "hub", "models--" + repo.replace("/", "--"))
    tot = 0
    for f in _glob.glob(os.path.join(d, "**", "*.safetensors"), recursive=True):
        try:
            tot += os.path.getsize(os.path.realpath(f))
        except OSError:
            pass
    return int(tot // (1024 * 1024))
