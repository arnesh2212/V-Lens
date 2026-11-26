import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlens.figures import style
from vlens.figures import fx_bench
from vlens.figures import fx_beyond
from vlens.figures import fx_channel
from vlens.figures import fx_coco
from vlens.figures import fx_commit
from vlens.figures import fx_cuegeom
from vlens.figures import fx_dirspace
from vlens.figures import fx_evalaware
from vlens.figures import fx_geom
from vlens.figures import fx_halluc
from vlens.figures import fx_mapgallery
from vlens.figures import fx_patch
from vlens.figures import fx_patchwords
from vlens.figures import fx_pwlock_story
from vlens.figures import fx_retention
from vlens.figures import fx_standing
from vlens.figures import fx_transitions
from vlens.figures import paper

FIGURES = [
    (paper.authority_ladder, "averaged"),
    (fx_commit.commitment_lattice, "averaged"),
    (fx_bench.cost_and_quality, "averaged"),
    (fx_bench.lens_scoreboard_readouts, "averaged"),
    (fx_mapgallery.map_edits, "averaged"),
    (fx_patch.patch_causal, "averaged"),
    (fx_standing.standing_plane, "averaged"),
    (fx_beyond.vwrite_beyond, "averaged"),
    (fx_coco.coco_pointing, "averaged"),
    (fx_channel.channel_plane, "Qwen3-VL-8B-Instruct"),
    (fx_geom.handoff, "Qwen3-VL-8B-Instruct"),
    (fx_geom.handoff, "Qwen3.5-27B"),
    (fx_retention.retention, "Qwen3-VL-8B-Instruct"),
    (fx_cuegeom.cue_geometry, "single"),
    (fx_dirspace.direction_space, "single"),
    (fx_evalaware.eval_awareness, "single"),
    (fx_halluc.hallucination_depth, "single"),
    (fx_transitions.layer_transitions, "single"),
    (fx_patchwords.patch_words, "single"),
    (fx_pwlock_story.pwlock_story, "single"),
]


def main():
    ap = argparse.ArgumentParser(description="Draw every figure of the paper from the result files.")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for fn, how in FIGURES:
        model = None if how in ("averaged", "single") else how
        paper.AVERAGE = how == "averaged"
        paper.SUFFIX = "__averaged" if how == "averaged" else (f"__{model}" if model else "")
        style.STAMP = model or ""
        kw = {"model": model} if model else {}
        try:
            print(fn.__name__, how, fn(results=a.results, out=a.out, **kw), flush=True)
        except (OSError, KeyError, ValueError) as e:
            print(fn.__name__, how, "skipped:", type(e).__name__, str(e)[:160], flush=True)


if __name__ == "__main__":
    main()
