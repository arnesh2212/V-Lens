# SPDX-License-Identifier: Apache-2.0
"""Write a fit to the cache and read it back, in thirty seconds and no card.

A sixteen-block fit is eight hours, and the only sign that it failed to cache
is one line in a log that the next experiment has already scrolled past. This
exercises the same save and load on tensors small enough to run anywhere, so
the round trip is checked before a run relies on it rather than after.

    python3 tools/check_fit_cache.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from vlens.lens import fitall as FA  # noqa: E402


def main():
    d = tempfile.mkdtemp(prefix="vlens_cache_check_")
    try:
        blocks = [0, 3, 7]
        maps = {"v_read": {l: (torch.randn(64, 64), torch.randn(64)) for l in blocks},
                "tuned": {l: (torch.randn(64, 64), torch.randn(64)) for l in blocks}}
        # the note each fit carries about itself, which is what a save has to
        # step around rather than try to read as a transport
        maps["v_read"]["meta"] = dict(lam=1e-3, gain=1.3, tokens=243667, shrink="identity")
        maps["tuned"]["meta"] = dict(steps=250, lr=0.1, wd=1e-3)
        path = os.path.join(d, "check.pt")
        FA._save(path, maps, None, {"id-1", "id-2"}, {"v_read": dict(seconds=1)},
                 spec='["a model", "a digest"]')
        got = FA._load(path)
        assert got is not None, "nothing came back"
        gm, _, ids, _ = got
        assert sorted(gm) == ["tuned", "v_read"], sorted(gm)
        for k in ("v_read", "tuned"):
            assert sorted(l for l in gm[k] if isinstance(l, int)) == blocks
            assert gm[k]["meta"] == maps[k]["meta"], k
            for l in blocks:
                a0, b0 = maps[k][l]
                a1, b1 = gm[k][l]
                assert a1.shape == a0.shape and b1.shape == b0.shape
                err = (a1 - a0).abs().max().item() / max(a0.abs().max().item(), 1e-9)
                assert err < 2e-3, f"{k} block {l} came back {err:.2e} out"
        assert ids == {"id-1", "id-2"}
        assert open(path + ".spec").read() == '["a model", "a digest"]'
        # and an entry under the wrong name is taken over rather than refitted
        moved = os.path.join(d, "older.pt")
        os.rename(path, moved)
        os.rename(path + ".spec", moved + ".spec")
        assert FA._adopt('["a model", "a digest"]', path, log=lambda *_: None)
        assert os.path.exists(path) and os.path.exists(path + ".spec")
        print("fit cache round trip: saved, loaded, adopted under a new name")
        return 0
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
