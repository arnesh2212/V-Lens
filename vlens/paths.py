import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
PAPER = os.path.join(ROOT, "paper")
ASSETS = os.path.join(ROOT, "assets")
DATA = os.environ.get("VLENS_DATA") or os.path.join(ROOT, "data", "simonsays")
