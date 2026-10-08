"""Regenerate every paper figure from whatever results exist."""
import sys, os, glob, json
sys.path.insert(0, "code")
import pandas as pd
from jvlm import style, figures, vjfigs
style.use()

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
TAG = sys.argv[3] if len(sys.argv) > 3 else "qwen2b"
os.makedirs(OUT, exist_ok=True)
made = []

def try_(fn, *args, **kw):
    try:
        made.append(fn(*args, **kw))
    except Exception as e:
        print(f"  skip {fn.__name__}: {type(e).__name__}: {e}")

p = f"{RES}/blindsight/{TAG}_raw.parquet"
if os.path.exists(p):
    df = pd.read_parquet(p)
    print(f"blindsight: {df.i.nunique()} items")
    try_(figures.f1_dissociation, df, OUT)
    try_(figures.f1b_localisation, df, OUT)
    try_(figures.f10_depth, df, OUT)
    try_(figures.f2_dose, df, OUT)
    try_(figures.f3_injection, df, OUT)

p = f"{RES}/routes/{TAG}_raw.parquet"
if os.path.exists(p):
    try_(figures.f7_routes, pd.read_parquet(p), OUT)
p = f"{RES}/routes/{TAG}_geometry.parquet"
if os.path.exists(p):
    try_(figures.f8_geometry, pd.read_parquet(p), OUT)
p = f"{RES}/competition/{TAG}_raw.parquet"
if os.path.exists(p):
    try_(figures.f9_competition, pd.read_parquet(p), OUT)

sp, sj = f"{RES}/secant/{TAG}_summary.csv", f"{RES}/secant/{TAG}_stats.json"
if os.path.exists(sp) and os.path.exists(sj):
    rf = f"{RES}/refit/{TAG}_stats.json"
    try_(figures.f21_secant, json.load(open(sj)), pd.read_csv(sp), OUT,
         refit=json.load(open(rf)) if os.path.exists(rf) else None)
    try_(figures.f22_certificate, json.load(open(sj)), OUT)
    try_(figures.f23_tradeoff, json.load(open(sj)), pd.read_csv(sp), OUT)
    if os.path.exists(rf):
        try_(figures.f25_crossing, json.load(open(rf)), pd.read_csv(sp), OUT)
        try_(figures.f26_radius, json.load(open(rf)), OUT)

# the clean-vocabulary re-run supersedes the first pass, whose concept set
# admitted the first fragment of two multi-token words in their place -- 29%
# of its trials swapped toward a fragment rather than toward a word.  The
# figure has to be drawn from the same trials the table reports, or the
# caption and the plot disagree about the headline.
sw = f"{RES}/swap/{TAG}_clean_raw.parquet"
if not os.path.exists(sw):
    sw = f"{RES}/swap/{TAG}_raw.parquet"
if os.path.exists(sw):
    print(f"  f40_swap from {os.path.basename(sw)}")
    try_(figures.f40_swap, pd.read_parquet(sw), OUT)
aj = f"{RES}/attacks/{TAG}_stats.json"
if os.path.exists(aj):
    try_(figures.f42_attacks, json.load(open(aj)), OUT)
pv = f"{RES}/posvar/{TAG}_stats.json"
if os.path.exists(pv):
    try_(figures.f43_posvar, json.load(open(pv)), OUT)

cj = f"{RES}/compare/{TAG}_stats.json"
if os.path.exists(cj):
    try_(figures.f41_lenses, json.load(open(cj)), OUT)

dc, dj = f"{RES}/deletion/{TAG}_curve.csv", f"{RES}/deletion/{TAG}_stats.json"
if os.path.exists(dc) and os.path.exists(dj):
    try_(figures.f32_deletion, pd.read_csv(dc), json.load(open(dj)), OUT)

vp, vj = f"{RES}/vmap/{TAG}_summary.csv", f"{RES}/vmap/{TAG}_stats.json"
if os.path.exists(vp) and os.path.exists(vj):
    try_(figures.f31_validate, pd.read_csv(vp), json.load(open(vj)), OUT)

p = f"{RES}/traject/{TAG}_traject.json"
if os.path.exists(p):
    try_(figures.f27_traject, json.load(open(p)), OUT)

p = f"{RES}/evalgate/{TAG}_raw.parquet"
if os.path.exists(p):
    try_(figures.f24_evalgate, pd.read_parquet(p), OUT)

p = f"{RES}/families/{TAG}_raw.parquet"
if os.path.exists(p):
    try_(figures.f11_families, pd.read_parquet(p), OUT)
gp, pp = f"{RES}/perceptual/{TAG}_geometry.parquet", f"{RES}/perceptual/{TAG}_prediction.csv"
if os.path.exists(gp):
    try_(figures.f12_verbalizable, pd.read_parquet(gp),
         pd.read_csv(pp) if os.path.exists(pp) else None, OUT)
p = f"{RES}/perceptual/{TAG}_raw.parquet"
if os.path.exists(p):
    try_(figures.f13_perceptual, pd.read_parquet(p), OUT)

p = f"{RES}/collateral/{TAG}_raw.parquet"
if os.path.exists(p):
    try_(figures.f14_collateral, pd.read_parquet(p), OUT)
p = f"{RES}/patch/{TAG}_raw.parquet"
if os.path.exists(p):
    try_(figures.f15_patch, pd.read_parquet(p), OUT)
bp, pp2 = f"{RES}/reality/{TAG}_behaviour.parquet", f"{RES}/reality/{TAG}_probes.parquet"
if os.path.exists(bp):
    try_(figures.f16_reality, pd.read_parquet(bp),
         pd.read_parquet(pp2) if os.path.exists(pp2) else None, OUT)
frames = {}
for nm, tg in (("Qwen3-VL 2B", "qwen2b"), ("Qwen3-VL 4B", "qwen4b"),
               ("Qwen3-VL 8B", "qwen8b"), ("InternVL 3.5 8B", "internvl8b")):
    f = f"{RES}/routes/{tg}_raw.parquet"
    if os.path.exists(f):
        frames[nm] = pd.read_parquet(f)
if len(frames) >= 2:
    try_(figures.f17_scaling, frames, OUT)

import glob as _g
pairs = sorted(_g.glob(f"{RES}/forgery/img_plant_*_clean.png"))
if pairs:
    stem = pairs[0].replace("_clean.png", "")
    fr = f"{RES}/forgery/{TAG}_plant_raw.parquet"
    bits = ["the model describes it", "the same description, word for word",
            "the monitor now reads the planted word"]
    try_(figures.forgery_pair, f"{stem}_clean.png", f"{stem}_adv.png",
         f"{stem}_delta.png", OUT, bits)

p = f"{RES}/depthcross/{TAG}_raw.parquet"
if os.path.exists(p):
    try_(figures.f18_depthcross, pd.read_parquet(p), OUT)
p = f"{RES}/language/{TAG}_raw.parquet"
if os.path.exists(p):
    try_(figures.f19_language, pd.read_parquet(p), OUT)
p = f"{RES}/source/{TAG}_raw.parquet"
if os.path.exists(p):
    try_(figures.f20_source, pd.read_parquet(p), OUT)

p = f"{RES}/gate0/{TAG}_summary.csv"
if os.path.exists(p):
    try_(figures.f4_readout, pd.read_csv(p), OUT)

p = f"{RES}/provenance/{TAG}_probes.parquet"
if os.path.exists(p):
    try_(figures.f5_provenance, pd.read_parquet(p), OUT)

for f in sorted(glob.glob(f"{RES}/forgery/*_raw.parquet")):
    tag = os.path.basename(f).replace("_raw.parquet", "")
    try_(figures.f6_forgery, pd.read_parquet(f), OUT, name=f"f6_{tag}.png")

# ------------------------------------------------- VJ-Lens layerwise figures
cp = f"{RES}/compare/{TAG}_point.parquet"
cd_ = f"{RES}/compare/{TAG}_del.parquet"
rb = f"{RES}/ribbon/{TAG}_ribbon.parquet"
rj = f"{RES}/ribbon/{TAG}_stats.json"

if os.path.exists(cp):
    try_(vjfigs.f51_strip, pd.read_parquet(cp), OUT,
         dele=pd.read_parquet(cd_) if os.path.exists(cd_) else None)

if os.path.exists(rb):
    stt = json.load(open(rj)) if os.path.exists(rj) else {}
    stt["_dir"] = f"{RES}/ribbon"
    try_(vjfigs.f52_ribbon, pd.read_parquet(rb), OUT, stats=stt)
elif os.path.exists(cp):
    try_(vjfigs.f52_ribbon, pd.read_parquet(cp), OUT)

_del = cd_ if os.path.exists(cd_) else f"{RES}/deletion/{TAG}_raw.parquet"
if os.path.exists(_del):
    try_(vjfigs.f53_deletion3, pd.read_parquet(_del), OUT)

for _tg in (f"{TAG}_full", TAG):
    _rp, _rj = f"{RES}/region/{_tg}_raw.parquet", f"{RES}/region/{_tg}_stats.json"
    if os.path.exists(_rp):
        try_(vjfigs.f54_regionsweep, pd.read_parquet(_rp), OUT,
             stats=json.load(open(_rj)) if os.path.exists(_rj) else None)
        break

print("\n".join(str(x) for x in made))
