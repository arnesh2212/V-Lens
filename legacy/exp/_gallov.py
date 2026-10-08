"""Shared helpers for the worked-example galleries: heatmap-over-image overlay."""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys, os
sys.path.insert(0, "code")
from jvlm import style

def load_rgb(path, square=None):
    im = Image.open(path).convert("RGB")
    if square:
        im = im.resize((square, square), Image.LANCZOS)
    return np.asarray(im)

def ramp(hue):
    # transparent -> hue, for overlay
    from matplotlib.colors import to_rgb
    r,g,b = to_rgb(hue)
    return LinearSegmentedColormap.from_list("r", [(r,g,b,0.0),(r,g,b,0.92)])

def overlay(ax, img, m, grid, hue, title, q=0.55):
    """img: HxWx3 array. m: length grid*grid list. Draws image + heatmap overlay."""
    ax.imshow(img, extent=[0,1,0,1], origin="upper", aspect="auto", interpolation="lanczos")
    a = np.array(m, float).reshape(grid, grid)
    a = a - a.min()
    if a.max() > 0: a = a / a.max()
    # soft threshold so only the strong cells paint
    a = np.clip((a - q) / (1 - q), 0, 1) if q < 1 else a
    ax.imshow(a, extent=[0,1,0,1], origin="upper", aspect="auto",
              cmap=ramp(hue), vmin=0, vmax=1, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title, loc="left", color=style.INK, fontsize=13, pad=4)
