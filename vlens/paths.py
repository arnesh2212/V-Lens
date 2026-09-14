# SPDX-License-Identifier: Apache-2.0
"""Where things live.  One place, so no script computes its own ROOT.

The repository root is two levels above this file.  Results and figures sit
under it on both machines; the datasets sit under ``VLENS_DATA`` on the box
and next to the repository here.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
PAPER = os.path.join(ROOT, "paper")
ASSETS = os.path.join(ROOT, "assets")
DATA = os.environ.get("VLENS_DATA") or (
    "/workspace/vlm/data" if os.path.isdir("/workspace/vlm/data")
    else os.path.join(ROOT, "VLM V-Lens Final Dataset", "VLM V-Lens Final Dataset"))
