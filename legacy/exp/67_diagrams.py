"""Render the two method schematics.  No model, no GPU."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jvlm import figdiagram

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="figures")
a = ap.parse_args()
print(figdiagram.forward_pass(a.out))
print(figdiagram.edit_space(a.out))
