# SPDX-License-Identifier: Apache-2.0
"""Render named figures into a folder, per model suffix: python3 tools/render_figs.py OUT name1,name2 [averaged,8B,27B]"""
import sys
import warnings
warnings.simplefilter("ignore")
sys.path.insert(0, ".")
from vlens.figures import paper as P, style as st  # noqa: E402

out, names = sys.argv[1], sys.argv[2].split(",")
which = (sys.argv[3] if len(sys.argv) > 3 else "averaged,8B,27B").split(",")
runs = {"averaged": (None, True), "8B": ("Qwen3-VL-8B-Instruct", False), "27B": ("Qwen3.5-27B", False)}
for w in which:
    model, avg = runs[w]
    P.AVERAGE = avg; P.SUFFIX = f"__{model}" if model else "__averaged"; st.STAMP = model or ""
    for n in names:
        f = P._resolve(n)
        try:
            print(n, w, f(results="results", out=out, model=model), flush=True)
        except Exception as e:  # noqa: BLE001
            print(n, w, "ERR", type(e).__name__, str(e)[:160], flush=True)
