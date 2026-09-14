# V-Lens on the custom datasets

Two contributions, evaluated together: the V-Lens (V-Read, an unlabelled per-layer
map from the answer-spot state to the final state; V-Write, a path-integrated
gradient attribution and steering direction) and five custom datasets that ask a
vision-language model whether text in a picture has the standing to settle a
question, whether it can tell a benchmark item from a real photo, and whether it
hides what it knows. Results page: the "Reading the sign" artifact, built from
`paper/datasets.html`.

## Layout

```
code/jvlm/          library (model adapter, loaders, reader, readout, plotting)
  model.py          VLM adapter: chat prompts with cue / system / history, batching, unembed
  vdata.py          loaders for the five datasets; VLENS_DATA points at the data root
  reader.py         V-Read: cross-fitted affine maps per layer
  verdict.py, axes.py   scores: yes over no, test over real, letters, pairs
  plot.py           Palatino, no bold, validated palette
code/exp/d*.py      one script per experiment (below); older 00_..34_ scripts are the
                    earlier Jacobian-lens study and are not needed for this page
scripts/            box queue scripts (*.sh), report and page builders (*.py)
assets/             the TeX Gyre Pagella fonts embedded into the built page
```

Not versioned (see `.gitignore`): the datasets, `results/`, `figures/`, `paper/`,
`logs/`, the vendored `ref_jlens_repo/`, and every note except this file.

## Setup

One 24 GB card runs one 8B model at a time. On the box:

```bash
mkdir -p /workspace/vlm && cd /workspace/vlm
git clone <this repo> .
git clone https://github.com/anthropics/jacobian-lens ref_jlens_repo/jacobian-lens   # reference lens code
ln -sfn /workspace/vlm/ref_jlens_repo/jacobian-lens/jlens code/jlens
pip install torch transformers accelerate scikit-learn pillow matplotlib numpy
# data: copy "VLM V-Lens Final Dataset/VLM V-Lens Final Dataset/*" to /workspace/vlm/data/
#       (the five dataset folders; NoUse/ is not needed except NoUse/subset_50.json for WMDP)
export VLENS_DATA=/workspace/vlm/data
```

Models are pulled from the Hub on first use (`Qwen/Qwen3-VL-8B-Instruct`,
`OpenGVLab/InternVL3-8B-hf`, and `Qwen/Qwen3-8B` as the text judge). No token is
needed. From a laptop, push code with:

```bash
BOX=root@<box> bash scripts/push_code.sh     # rsyncs code/ and scripts/, never the jlens symlink
```

## Running the experiments

Run in this order per model; every script writes JSON under `results/vlens/`
and prints a summary. `R` is a repo id, for example `Qwen/Qwen3-VL-8B-Instruct`.

```bash
cd /workspace/vlm; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 code/exp/d01_behaviour.py --repo $R                 # replies, first-token margins, answer-spot states
python3 code/exp/d02_judge.py results/vlens/<model>__*.json # parse yes/no, letters, test/real; Qwen3-8B judge for open answers
python3 code/exp/d03_vlens.py --repo $R --parts A B         # V-Read curves per family; V-Write attribution on the edits
python3 code/exp/d05_wmdp.py --repo $R                      # WMDP sandbagging: 100 items, five prompts, image variants, read and write
python3 code/exp/d06_more.py --repo $R --parts S W T X      # steering, awareness steering, word strips, map transfer
python3 code/exp/d08_directions.py --repo $R                # awareness / sandbag / authority / knowledge directions and cross-effects
python3 code/exp/d09_authority.py --repo $R                 # looks-generated check, answer arrival, bearer removal
python3 code/exp/d11_depicted.py --repo $R                  # depicted-sign attribution with mean-replacement truth
python3 code/exp/d12_bearer.py --repo $R                    # which bearer class each Authority reply followed (text judge)
```

The Authority open prompts should be run with room to answer: repeat d01 for the
four `gemini_*` families with `--gen-open 160 --no-states --out results/vlens_long`
and judge that folder; the report prefers those replies when they are judged.

`scripts/*.sh` are the queues that were actually used (`chain_beh.sh`,
`night.sh`, `more.sh`, `directions.sh`, `long.sh`, `depicted.sh`, ...). Each
waits for the previous log's `DONE` marker, so they can all be started at once.
`scripts/boxmon.sh` writes a status line every two minutes to `logs/status.txt`
and an `ALERT` line on a traceback or a dead queue. Use `--limit 2` on any script
for a smoke test.

Two traps: `pkill -f <pattern>` matches the ssh shell that runs it, so kill with
`pkill -x -f "bash scripts/name.sh"` in a separate call; and a device-side CUDA
assert poisons the process, so a crashed part must be re-run in a new process.

## Building the page

On the laptop, after pulling `results/vlens/*.json` and `results/vlens_long/*.json`:

```bash
python3 scripts/d04_report.py           # results/vlens/report.json: every table's numbers
python3 scripts/d07_figs.py             # figures/vlens/*.png: reading, attribution, steering, sandbagging, 3-D surfaces
python3 scripts/behaviour_figs.py       # prompt-kind bars, decoy replies, nuScenes obedience
python3 scripts/authority_showcase.py   # four scenes with verbatim replies
python3 scripts/dataset_panels.py       # example panels of each dataset
python3 scripts/build_datasets_page.py  # fills the <!--TABLE:name--> blocks in paper/datasets.html
ALLOW_MISSING=1 python3 scripts/build_artifact.py datasets   # paper/datasets_artifact.html, images inlined as WebP
```

Prose lives in `paper/datasets.html` between the table markers and is never
touched by the builders. The page is published as an Artifact from the built
file.

## Conventions

Figures: Palatino (TeX Gyre Pagella), no bold, no em dashes, one scale per axis.
WMDP question bodies are never printed on the page: letters, accuracies and the
model's own self-reports only. Datasets that were specified but not generated
(Open Images and VINS edits) contribute nothing and the page says so.

## Credit

The reference lens implementation vendored under `ref_jlens_repo/` is
Gurnee, Sofroniew et al., *Verbalizable Representations Form a Global Workspace
in Language Models*, Transformer Circuits, July 2026 (Apache-2.0); `jvlm` reuses
its `ActivationRecorder` directly.
