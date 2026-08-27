"""Fit a Jacobian lens on a VLM.  Checkpointed and resumable (spot-safe)."""
import argparse, logging, os, sys, time, json
sys.path.insert(0, "/workspace/vlm/code")
for n in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)
import torch
import jvlm
from jvlm import corpus

p = argparse.ArgumentParser()
p.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
p.add_argument("--kind", default="mm", choices=["mm", "text", "mix"])
p.add_argument("--n", type=int, default=100)
p.add_argument("--dim-batch", type=int, default=8)
p.add_argument("--max-pixels", type=int, default=200704)
p.add_argument("--policy", default="all", choices=["all", "text", "image"],
               help="which positions the average runs over (V-Lens: image)")
p.add_argument("--target", default="future", choices=["future", "answer"],
               help="where the cotangent goes (V-Lens: answer)")
p.add_argument("--out", default="/workspace/vlm/results/lenses")
p.add_argument("--max-seconds", type=float, default=None)
a = p.parse_args()

os.makedirs(a.out, exist_ok=True)
tag = (f"{a.repo.split('/')[-1]}__{a.kind}__{a.policy}__{a.target}__n{a.n}"
       if a.target != "future" else
       f"{a.repo.split('/')[-1]}__{a.kind}__{a.policy}__n{a.n}")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(f"/workspace/vlm/logs/fit_{tag}.log")],
)
log = logging.getLogger("fit")
log.info("=== %s", tag)

m = jvlm.load_vlm(a.repo, max_pixels=a.max_pixels)
log.info("%s", m)
batches = corpus.build_batches(m, a.kind, a.n, seed=0)
log.info("corpus: %d items", len(batches))

t0 = time.time()
lens = jvlm.fit(
    m, batches,
    dim_batch=a.dim_batch,
    position_policy=a.policy,
    target_policy=a.target,
    checkpoint_path=f"{a.out}/{tag}.ckpt",
    checkpoint_every=5,
    max_seconds=a.max_seconds,
)
lens.save(f"{a.out}/{tag}.pt")
meta = dict(repo=a.repo, kind=a.kind, policy=a.policy, target=a.target,
            n_requested=a.n,
            n_fitted=lens.n_prompts, d_model=m.d_model, n_layers=m.n_layers,
            source_layers=lens.source_layers, max_pixels=a.max_pixels,
            minutes=round((time.time()-t0)/60, 1))
json.dump(meta, open(f"{a.out}/{tag}.json", "w"), indent=2)
log.info("saved %s/%s.pt  %s", a.out, tag, meta)
