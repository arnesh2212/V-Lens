"""Smoke test: adapter correctness + Jacobian fit cost on this GPU."""
import sys, time, logging, torch
sys.path.insert(0, "/workspace/vlm/code")
from PIL import Image, ImageDraw
import jvlm
from jvlm import readout

logging.basicConfig(level=logging.INFO, format="%(message)s")
REPO = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-VL-2B-Instruct"
MAXPIX = int(sys.argv[2]) if len(sys.argv) > 2 else 200704  # 256 visual tokens

def demo_image():
    im = Image.new("RGB", (448, 448), (230, 230, 235))
    d = ImageDraw.Draw(im)
    d.ellipse([60, 60, 200, 200], fill=(200, 40, 40))          # red ball
    d.rectangle([250, 250, 390, 390], fill=(40, 90, 200))      # blue square
    return im

print(f"=== loading {REPO}")
t0 = time.time()
m = jvlm.load_vlm(REPO, max_pixels=MAXPIX)
print(m, f"({time.time()-t0:.0f}s)  vocab={m.vocab_size} img_tok={m.image_token_id}")
print("mem", torch.cuda.memory_allocated()/2**30, "GiB")

# ---- 1. forward works, image span is found
batch = m.chat("Describe this image.", [demo_image()])
print("batch keys:", {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in batch.items()})
ipos = readout.image_positions(m, batch)
print(f"seq_len={batch['input_ids'].shape[1]}  n_image_tokens={len(ipos)}  span={ipos.min().item()}..{ipos.max().item()}")

# ---- 2. replication is numerically consistent (batch elem 0 == batch of 1)
with torch.no_grad():
    h1 = m.forward(batch).last_hidden_state[0]
    hB = m.forward(m.replicate(batch, 4)).last_hidden_state
err = (hB - h1[None]).abs().max().item()
rel = err / h1.abs().max().item()
print(f"replication max abs err={err:.4g} rel={rel:.4g}  ({'OK' if rel < 5e-2 else 'FAIL'})")

# ---- 3. model actually answers
with torch.no_grad():
    out = m.hf.generate(**batch, max_new_tokens=30, do_sample=False)
print("gen:", m.tokenizer.decode(out[0, batch["input_ids"].shape[1]:], skip_special_tokens=True))

# ---- 4. cost of one Jacobian batch
mid = m.n_layers // 2
layers = sorted({max(m.n_deepstack, 1), mid, m.n_layers - 4})
for db in (8, 16, 32):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.time()
        J, S, nv = jvlm.jacobian_for_batch(m, batch, layers, dim_batch=db)
        dt = time.time() - t0
        print(f"dim_batch={db:3d}  {dt:6.1f}s  S={S} valid={nv}  "
              f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB  "
              f"||J||/sqrt(d)={max(J[l].norm().item() for l in layers)/m.d_model**0.5:.3f}")
    except torch.cuda.OutOfMemoryError:
        print(f"dim_batch={db:3d}  OOM"); torch.cuda.empty_cache()
print(f"\n=> all-layer fit would be ~{len(range(m.n_deepstack, m.n_layers-1))} layers")
