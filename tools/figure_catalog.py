# SPDX-License-Identifier: Apache-2.0
"""A catalogue of every figure, sorted so the paper's set can be chosen from it.

Each figure carries three tags: its kind (qualitative, quantitative, or dataset x
method), its breadth (broad: across the whole set or several tasks; niche: one
probe, one scene or one study), and the pillar it serves (method against the
published baselines, the method on general data, the dataset, the two together).
Beside the tags stand its placement from the caption, the claim it opens with,
the models it is drawn for, and whether it has passed the visual check.

    python3 tools/figure_catalog.py            # writes FIGURES.md
"""
from __future__ import annotations

import glob
import importlib
import inspect
import json
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: name: (kind, breadth, pillar)
TAGS = {
    # the lens against published lenses, across the set
    "lens_scoreboard": ("quantitative", "broad", "method vs baselines"),
    "lens_scoreboard_readouts": ("quantitative", "broad", "method vs baselines"),
    "lens_canyon": ("quantitative", "broad", "method vs baselines"),
    "lens_by_depth": ("quantitative", "broad", "method vs baselines"),
    "lens_kl": ("quantitative", "broad", "method vs baselines"),
    "cost_and_quality": ("quantitative", "broad", "method vs baselines"),
    "early_detection": ("quantitative", "broad", "method on general data"),
    "transport_geometry": ("quantitative", "niche", "method vs baselines"),
    # attribution against published attribution methods
    "vwrite_baselines": ("quantitative", "broad", "method vs baselines"),
    "attribution_plane": ("quantitative", "broad", "method vs baselines"),
    "patch_causal": ("quantitative", "broad", "method vs baselines"),
    "patch_lenses": ("quantitative", "broad", "method vs baselines"),
    "baseline_agreement": ("quantitative", "niche", "method vs baselines"),
    "patch_atlas": ("quantitative", "broad", "method vs baselines"),
    "crossquestion": ("quantitative", "broad", "method vs baselines"),
    "steer": ("quantitative", "broad", "method vs baselines"),
    "signed_handle": ("qualitative", "niche", "method vs baselines"),
    "gallery": ("qualitative", "broad", "method vs baselines"),
    "map_methods": ("qualitative", "broad", "method vs baselines"),
    "map_questions": ("qualitative", "niche", "method vs baselines"),
    # general data
    "coco_general": ("quantitative", "broad", "method on general data"),
    "coco_pointing": ("qualitative", "broad", "method on general data"),
    "refusal_read": ("quantitative", "broad", "method on general data"),
    "refusal_steer": ("quantitative", "broad", "method on general data"),
    "refusal_cascade": ("quantitative", "niche", "method on general data"),
    "patch_jailbreak": ("quantitative", "niche", "method on general data"),
    "jailbreak_coverage_fan": ("quantitative", "niche", "method on general data"),
    # the method read across the set's sources and probes
    "patch_sources": ("dataset x method", "broad", "dataset x method"),
    "answer_plane": ("dataset x method", "niche", "dataset x method"),
    "commitment_lattice": ("dataset x method", "broad", "dataset x method"),
    "packaging_lattice": ("dataset x method", "niche", "dataset x method"),
    "channel_plane": ("dataset x method", "niche", "dataset x method"),
    "standing_plane": ("dataset x method", "niche", "dataset x method"),
    "handoff": ("dataset x method", "niche", "dataset x method"),
    "direction_geometry": ("dataset x method", "niche", "dataset x method"),
    "eval_depth": ("dataset x method", "niche", "dataset x method"),
    "sandbag_channels": ("dataset x method", "niche", "dataset x method"),
    "map_edits": ("dataset x method", "niche", "dataset x method"),
    "map_swaps": ("dataset x method", "niche", "dataset x method"),
    "method_source_matrix": ("dataset x method", "broad", "dataset x method"),
    # the dataset, with the model's answers through depth
    "layer_story": ("qualitative", "broad", "dataset"),
    "scene_battery": ("qualitative", "niche", "dataset"),
    "scene_store": ("qualitative", "niche", "dataset"),
    "authority_cases": ("qualitative", "broad", "dataset"),
    "edit_series": ("qualitative", "niche", "dataset"),
    "edit_series_more": ("qualitative", "niche", "dataset"),
    "eval_series": ("qualitative", "niche", "dataset"),
    "standing_swap": ("qualitative", "niche", "dataset"),
    "authority_ladder": ("quantitative", "broad", "dataset"),
    "dataset_map": ("quantitative", "broad", "dataset"),
    # the password-locking study
    "pwlock_transfer": ("quantitative", "niche", "dataset x method"),
    "pwlock_latent": ("dataset x method", "niche", "dataset x method"),
    "pwlock_conditions": ("quantitative", "niche", "dataset x method"),
}
CHECKED = "figures/.checked.json"   # names that passed the visual check, with the date


def figures():
    from vlens.figures import paper as P
    out = {}
    mods = [P] + [importlib.import_module(f"vlens.figures.{m}") for m in P.REWRITTEN]
    for mod in mods:
        for n in dir(mod):
            f = getattr(mod, n)
            if (callable(f) and getattr(f, "__module__", "") == mod.__name__
                    and (f.__doc__ or "").lstrip().startswith(("MAIN", "APPENDIX"))):
                out[n] = f        # a later module wins, as the build does
    return {k: v for k, v in out.items() if k not in P.DROPPED}


def main():
    checked = json.load(open(CHECKED)) if os.path.exists(CHECKED) else {}
    rows = []
    for n, f in sorted(figures().items()):
        doc = inspect.cleandoc(f.__doc__ or "")
        where = "main" if doc.startswith("MAIN") else "appendix"
        claim = re.sub(r"^(MAIN|APPENDIX)[.:]?\s*", "", doc.split("\n\n")[0]).replace("\n", " ")
        files = sorted(os.path.basename(p) for p in glob.glob(f"figures/{n}__*.png"))
        models = ", ".join(sorted({p.split("__")[1][:-4] for p in files})) or "not drawn yet"
        kind, breadth, pillar = TAGS.get(n, ("untagged", "untagged", "untagged"))
        rows.append((pillar, kind, breadth, n, where, models, n in checked, claim))
    order = ["method vs baselines", "method on general data", "dataset x method", "dataset", "untagged"]
    rows.sort(key=lambda r: (order.index(r[0]) if r[0] in order else 9, r[1], r[2], r[3]))
    md = ["# Figure catalogue", "",
          "Generated by `tools/figure_catalog.py`. Every figure the build draws, tagged so the paper's",
          "set can be chosen: kind, breadth (broad = across the set or several tasks, niche = one probe,",
          "scene or study), and pillar. Captions are in CAPTIONS.md.", ""]
    cur = None
    for pillar, kind, breadth, n, where, models, ok, claim in rows:
        if pillar != cur:
            md += ["", f"## {pillar}", "",
                   "| figure | kind | breadth | caption says | drawn for | checked | claim |",
                   "|---|---|---|---|---|---|---|"]
            cur = pillar
        md.append(f"| `{n}` | {kind} | {breadth} | {where} | {models} | {'yes' if ok else 'no'} | "
                  f"{claim[:220]}{'...' if len(claim) > 220 else ''} |")
    open("FIGURES.md", "w").write("\n".join(md) + "\n")
    n_drawn = sum(1 for r in rows if r[5] != "not drawn yet")
    print(f"{len(rows)} figures, {n_drawn} drawn, {sum(1 for r in rows if r[6])} checked -> FIGURES.md")


if __name__ == "__main__":
    main()
