"""Print ``--layers a,b,c,d`` for a model, chosen from its own depth.

Hard-coded layer indices are the quiet way a queue breaks when a new
architecture arrives: 8,15,23,30 is sensible on a 36-block Qwen and out of
range on a 28-block one.  This reads the config only (no weights) and places
the reads at the same fractions of depth the experiments use.
"""
import sys

from transformers import AutoConfig

repo = sys.argv[1]
try:
    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True)
    n = None
    for probe in (cfg, getattr(cfg, "text_config", None),
                  getattr(cfg, "llm_config", None)):
        if probe is not None and getattr(probe, "num_hidden_layers", None):
            n = int(probe.num_hidden_layers)
    if not n:
        raise ValueError("no num_hidden_layers")
    at = sorted({max(1, min(n - 2, round(f * n)))
                 for f in (0.25, 0.45, 0.65, 0.85)})
    print("--layers " + ",".join(str(x) for x in at))
except Exception as e:                      # no flag beats a wrong flag
    print("", end="")
    print(f"pick_layers: {type(e).__name__}: {e}", file=sys.stderr)
