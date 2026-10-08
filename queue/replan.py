"""Rebalance the two lanes: InternVL moves to lane8 after the 8B work; consistency fixes."""
import shutil

Q = "/workspace/temp/vlm/queue/"
for f in ("main_one.q", "lane8.q"):
    shutil.copy(Q + f, Q + f + ".bak_replan")
mo = open(Q + "main_one.q").read().split("\n")
l8 = open(Q + "lane8.q").read().split("\n")
mo = [l for l in mo if l.strip()]
l8 = [l for l in l8 if l.strip()]

SAFE = "JVLM_SAFETY_ROOT=/dev/shm/vlens_safety JVLM_HARD_ROOT=/dev/shm/vlens_safety "
BUD = "VLENS_HOST_BUDGET=48 VLENS_MAP_BUDGET=20 "


def coco_baselines(repo, short, hf):
    return [f"HF_HOME={hf} python3 tools/patchscope_readouts.py --repo {repo} --results results/coco --out results/coco_readouts",
            f"python3 tools/probe_readouts.py --model {short} --results results/coco --out results/coco_readouts"]


# ---- main_one: 27B, then gemma
i_int = next(k for k, l in enumerate(mo) if "rm -rf /dev/shm/gemma_hf" in l)
internvl = mo[i_int + 1:]          # rotate to InternVL ... to the end
mo = mo[:i_int + 1]
# 27B: refusal re-read with all three lenses once they exist, COCO baselines
i = next(k for k, l in enumerate(mo) if "warm_fit.py --repo Qwen/Qwen3.5-27B --want v_read tuned jacobian" in l)
mo[i + 1:i + 1] = [
    f"REDO=1 VLENS_NO_FIT=1 {BUD}{SAFE}HF_HOME=/dev/shm/qwen_hf python3 experiments/safety/refusal.py --repo Qwen/Qwen3.5-27B --n 300 --variant TYPO --batch 8 --stages read --lens-sources shared coco indomain --want v_read tuned jacobian --out results/safety",
    *coco_baselines("Qwen/Qwen3.5-27B", "Qwen3.5-27B", "/dev/shm/qwen_hf")]
# the 27B COCO lens bench only restates the COCO readouts; the KL/cost bench stays
mo = [l for l in mo if not ("lens_bench.py --repo Qwen/Qwen3.5-27B --results results/coco" in l)]
# gemma: no smoke tests (the layout is covered by the behaviour pass), patching at n 600,
# refusal reading with every lens, COCO baselines
mo = [l for l in mo if not (("smoke_layout.py" in l or "verify_batching.py" in l) and "gemma" in l)]
out = []
for l in mo:
    if "patch_causal.py --repo google/gemma-4-26B-A4B-it --n 1000" in l:
        l = l.replace("--n 1000", "--n 600")
    if "refusal.py --repo google/gemma-4-26B-A4B-it" in l:
        l = (f"VLENS_NO_FIT=1 {BUD}{SAFE}HF_HOME=/dev/shm/gemma_hf python3 experiments/safety/refusal.py "
             "--repo google/gemma-4-26B-A4B-it --n 300 --variant TYPO --fracs 0.3 0.5 0.7 --batch 8 "
             "--stages read steer --lens-sources shared coco indomain --want v_read tuned jacobian --out results/safety")
    out.append(l)
    if "layer_readouts.py read --model gemma-4-26B-A4B-it --results results/coco" in l:
        out += coco_baselines("google/gemma-4-26B-A4B-it", "gemma-4-26B-A4B-it", "/dev/shm/gemma_hf")
mo = out

# ---- InternVL on lane8, weights on the workspace disk in place of the 8B weights,
# once every 8B job is through (the 8B weights can be fetched again)
HFI = "/workspace/temp/vlm/hf_internvl"
iv = []
for l in internvl:
    if "rotate_shm.sh" in l:
        l = ("bash -c \"rm -rf /workspace/temp/vlm/hf/hub/models--Qwen--Qwen3-VL-8B-Instruct; df -h / | tail -1; "
             f"mkdir -p {HFI}; HF_HOME={HFI} HF_HUB_OFFLINE=0 python3 -c 'from huggingface_hub import snapshot_download as s; "
             "print(s(\\\"OpenGVLab/InternVL3_5-8B-HF\\\", allow_patterns=[\\\"*.json\\\",\\\"*.safetensors\\\",\\\"*.txt\\\",\\\"*.model\\\",\\\"*.py\\\",\\\"*.jinja\\\"]))'; df -h / | tail -1\"")
    if "rm -rf /dev/shm/internvl_hf" in l or "read_image.py" in l:
        continue
    if ("smoke_layout.py" in l) or ("verify_batching.py" in l):
        pass   # a new family's layout and batching are checked once, they are quick
    l = l.replace("/dev/shm/internvl_hf", HFI)
    iv.append(l)
    if "layer_readouts.py read --model InternVL3_5-8B-HF --results results/coco" in l:
        iv += coco_baselines("OpenGVLab/InternVL3_5-8B-HF", "InternVL3_5-8B-HF", HFI)
# the 8B COCO baselines, before the 8B weights go
j = next(k for k, l in enumerate(l8) if "layer_readouts.py read --model Qwen3-VL-8B-Instruct --results results/coco" in l)
l8[j + 1:j + 1] = coco_baselines("Qwen/Qwen3-VL-8B-Instruct", "Qwen3-VL-8B-Instruct", "/workspace/temp/vlm/hf")
l8 += iv
open(Q + "main_one.q", "w").write("\n".join(mo) + "\n")
open(Q + "lane8.q", "w").write("\n".join(l8) + "\n")
print("main_one", len(mo), "lines; lane8", len(l8), "lines")
