"""Draw the prompt-invariance figure.  Pure plotting -- no model, no GPU."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from jvlm import figcausal

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--jraw", default=None)
ap.add_argument("--out", default="/workspace/vlm/figures")
a = ap.parse_args()
df = pd.read_parquet(a.raw)
dj = pd.read_parquet(a.jraw) if a.jraw and os.path.exists(a.jraw) else None
print(figcausal.promptblind(df, dj, a.out))
