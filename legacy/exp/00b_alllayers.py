"""Does fitting ALL layers cost more than fitting 3? (backward dominates)"""
import sys, time, logging, torch, os
sys.path.insert(0, "code")
os.environ.setdefault("HF_HUB_OFFLINE","1")
logging.getLogger("httpx").setLevel(logging.WARNING)
from PIL import Image, ImageDraw
import jvlm
logging.basicConfig(level=logging.INFO, format="%(message)s")
REPO = sys.argv[1]
m = jvlm.load_vlm(REPO, max_pixels=200704)
im = Image.new("RGB",(448,448),(230,230,235)); d=ImageDraw.Draw(im)
d.ellipse([60,60,200,200],fill=(200,40,40)); d.rectangle([250,250,390,390],fill=(40,90,200))
batch = m.chat("Describe this image.", [im])
print(m, "S=", batch["input_ids"].shape[1])
for name, layers in [("3 layers", [3, m.n_layers//2, m.n_layers-4]),
                     ("all layers", list(range(m.n_deepstack, m.n_layers-1)))]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0=time.time(); J,S,nv = jvlm.jacobian_for_batch(m, batch, layers, dim_batch=8); dt=time.time()-t0
    print(f"{name:12s} n={len(layers):3d}  {dt:6.1f}s  peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB")
