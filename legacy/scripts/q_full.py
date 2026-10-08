"""Write the run queue for the 40 GB card.

Ordering principle: safety leads, because that is where this reading claims
something no fixed-transport lens can claim at all, and a safety result that
holds on one model is an anecdote.  So the pairing experiment and the matched
causal intervention run on every family we fit, not just the flagship.

Two rules the earlier boxes taught us, both encoded here rather than remembered:

  * every script that can show a Jacobian-lens arm is given one, by
    interpolating scripts/find_lens.py at start time.  Without the flag the arm
    is dropped in silence and the table ships two-armed as if the method had
    failed.
  * layers are never hard-coded.  ``--n-layers`` lets each script place its own
    reads between the deepstack and the last block, which is the only form that
    survives a change of architecture.
"""
import os
import sys

Q = "queue.txt"
ALLOC = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
FIND = "$(python3 scripts/find_lens.py {repo})"

Q8 = "Qwen/Qwen3-VL-8B-Instruct"
Q4 = "Qwen/Qwen3-VL-4B-Instruct"
Q2 = "Qwen/Qwen3-VL-2B-Instruct"
I8 = "OpenGVLab/InternVL3-8B-hf"
L7 = "llava-hf/llava-1.5-7b-hf"
G12 = "google/gemma-3-12b-it"

rows = []


def add(name, cmd, need=20000, tries=2):
    rows.append([name, str(tries), str(need), "1", ALLOC + cmd])


def lens(repo):
    return FIND.format(repo=repo)


def fit(name, repo, n=64, need=30000):
    at = "$(python3 scripts/pick_layers.py %s)" % repo
    add(name, f"python code/exp/01_fit_lens.py --repo {repo} --kind mm "
              f"--n {n} --policy all {at} --dim-batch 8", need=need, tries=3)


# ---- the safety pair, which is the point of the paper -----------------------
def pair(name, repo, tag, per=110):
    add(name, f"python code/exp/95_pairing.py --repo {repo} {lens(repo)} "
              f"--per-source {per} --mean-images 60 --ig-steps 4 --boot 2000 "
              f"--tag {tag}", need=24000)


def matched(name, repo, tag, n=150):
    add(name, f"python code/exp/90_matched.py --repo {repo} {lens(repo)} "
              f"--n {n} --mean-images 60 --coco-n 80 --tag {tag}", need=24000)


def base(name, repo, tag, n=120):
    add(name, f"python code/exp/96_baselines.py --repo {repo} {lens(repo)} "
              f"--n {n} --mean-images 80 --n-layers 4 --ig-steps 8 --tag {tag}",
        tries=3)


def diss(name, repo, tag, n=80):
    add(name, f"python code/exp/92_dissoc.py --repo {repo} {lens(repo)} "
              f"--n {n} --scan 4000 --mean-images 90 --n-layers 4 --gen-n 24 "
              f"--boot 2000 --null-perm 200 --chunk-margin 4 "
              f"--out results/dissoc --tag {tag}")


def nat(name, repo, tag, n=80):
    add(name, f"python code/exp/93_natural.py --repo {repo} {lens(repo)} "
              f"--n {n} --scan 900 --mean-images 80 --n-layers 3 --tag {tag}")


def inter(name, repo, tag, n=64):
    add(name, f"python code/exp/91_interaction.py --repo {repo} {lens(repo)} "
              f"--n {n} --mean-images 80 --n-layers 3 --tag {tag}")


# =================== 1. the flagship: safety first, then the novel claims ====
diss("diss_8b", Q8, "qwen8b")          # already running; kept so it can retry
pair("pair_8b", Q8, "qwen8b")
matched("matched_8b", Q8, "qwen8b")
inter("inter_8b", Q8, "qwen8b")
add("fair_8b", f"python code/exp/89_fair.py --repo {Q8} {lens(Q8)} --n 90 "
               f"--mean-images 90 --n-layers 3 --ig-steps 4 --tag qwen8b")

# =================== 2. the other transports have to exist before their arms =
fit("fit_intern8", I8)
fit("fit_llava7", L7, need=28000)

# =================== 3. the rest of the flagship evidence ====================
nat("nat_8b", Q8, "qwen8b")
add("steer_8b", f"python code/exp/99_steer.py --repo {Q8} --n 48 "
                f"--mean-images 60 --ig-steps 8 --tag qwen8b")
add("inpaint_8b", f"python code/exp/94_inpaint.py --repo {Q8} {lens(Q8)} "
                  f"--n 64 --pool-n 400 --mean-images 60 --n-layers 3 "
                  f"--tag qwen8b")
add("wino_8b", f"python code/exp/79_winoground.py --repo {Q8} {lens(Q8)} "
               f"--n 400 --mean-images 90 --n-layers 4 --ig-steps 4 "
               f"--tag qwen8b")
add("sugar_8b", f"python code/exp/82_sugarcrepe.py --repo {Q8} {lens(Q8)} "
                f"--n 400 --mean-images 60 --n-layers 6 --ig-steps 4 "
                f"--tag qwen8b")
add("census_8b", f"python code/exp/88_census.py --repo {Q8} --images 500 "
                 f"--mean-images 120 --tag qwen8b")

# =================== 4. every family, safety first ==========================
for nm, repo, tag in (("i8", I8, "intern8"), ("4b", Q4, "qwen4b"),
                      ("l7", L7, "llava7"), ("2b", Q2, "qwen2b")):
    # The replications answer "does it hold on another architecture", which a
    # smaller sample settles.  At the flagship's sample each of these costs
    # three hours and the four of them would outlast every other job in the
    # queue put together.
    pair(f"pair_{nm}", repo, tag, per=60)
    matched(f"matched_{nm}", repo, tag, n=60)
    base(f"base_{nm}", repo, tag)
    diss(f"diss_{nm}", repo, tag)
    inter(f"inter_{nm}", repo, tag)
    nat(f"nat_{nm}", repo, tag)

# =================== 5. the big one, last, where an OOM costs nothing =======
fit("fit_g12", G12, n=48, need=34000)
pair("pair_g12", G12, "gemma12b", per=40)
base("base_g12", G12, "gemma12b", n=80)
diss("diss_g12", G12, "gemma12b", n=48)

keep = set(sys.argv[1:])
tmp = Q + ".tmp"
with open(tmp, "w") as f:
    f.write("\n".join("|".join(r) for r in rows) + "\n")
os.replace(tmp, Q)
done = sum(os.path.exists(f"logs/{r[0]}.done") for r in rows)
print(f"{len(rows)} jobs queued, {done} already done")
for i, r in enumerate(rows, 1):
    m = " [done]" if os.path.exists(f"logs/{r[0]}.done") else (
        " [running]" if r[0] in keep else "")
    print(f"  {i:2d}. {r[0]:<14}{m}")
