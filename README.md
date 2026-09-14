# V-Lens

Reading and writing a vision-language model's decision at the answer spot, and
five custom datasets that ask whether text in a picture has the standing to
settle a question, whether the model can tell a benchmark item from a real
photo, and whether it hides what it knows.

Two instruments, both fitted without labels. V-Read: one affine map per layer
from the answer-spot state at that layer to the final state, fitted on a few
hundred unlabelled states of the same prompt family, read through the model's
own unembedding. V-Write: the gradient of a score with respect to every earlier
state, path-integrated from a baseline, summed over groups of positions, and
used as a steering direction.

Two pages come out of this repository: the V-Lens page (`paper/tool.html`,
the method on safety and grounding sets) and the datasets page
(`paper/datasets.html`, the five custom sets with and without the lens).

## Layout

```
vlens/                 the library
  model.py             VLM adapter: chat prompts with cue / system / history, batching, unembedding
  corpus.py            image loading and the COCO fitting corpus
  tasks.py             the Scorer: first-token log-probabilities, greedy replies
  results.py           reading behaviour rows and cached states
  paths.py             where results, figures, paper and data live
  data/                datasets: the five custom sets, safety sets, the hard presence set, CLEVR, COCO
  lens/                readout hooks, caches, depth, axes and scores, V-Read (reader), V-Write (verdict),
                       attribution, steering, the probe / tuned-lens / GLIMPSE baselines
  judge/               the regex refusal judge, the guard models, the rubric judge for open replies
  figures/             plot (the house style), datasets (every figure of the datasets page),
                       vlens_page (every figure of the V-Lens page)
experiments/
  datasets/            one script per experiment on the custom datasets (below)
  vlens/               one script per experiment of the V-Lens page (gate, channels, ... , tasks)
build/                 report -> figures -> page -> artifact, and the V-Lens page's tables
queue/                 the box queue for the datasets pipeline
tools/                 push / pull / monitor, and two maintenance scripts
legacy/                the superseded VJ-Lens study, not maintained (see legacy/README.md)
jlens -> ref_jlens_repo/jacobian-lens/jlens   the vendored reference lens code (ActivationRecorder)
```

Not versioned (see `.gitignore`): the datasets, `results/`, `figures/`, `paper/`,
`logs/`, `ref_jlens_repo/`, and every note except this file.

## Setup

One 24 GB card runs one 8B model at a time.

```bash
git clone <this repo> /workspace/vlm && cd /workspace/vlm
git clone https://github.com/anthropics/jacobian-lens ref_jlens_repo/jacobian-lens   # the jlens symlink points into it
pip install -e .                                   # or just run from the repository root: every script adds it to sys.path
# data: copy the five dataset folders under "VLM V-Lens Final Dataset/VLM V-Lens Final Dataset/" to /workspace/vlm/data/
export VLENS_DATA=/workspace/vlm/data              # defaults to that path on the box, and to the dataset folder here
```

Models are pulled from the Hub on first use (`Qwen/Qwen3-VL-8B-Instruct`,
`OpenGVLab/InternVL3-8B-hf`; `Qwen/Qwen3-8B` as the text judge). No token is
needed. From a laptop:

```bash
BOX=root@<host> bash tools/push_code.sh            # code only; results come back with tools/pull.sh
```

## The datasets pipeline

Per model, in this order. Every script writes JSON under `results/vlens/` and
prints a summary; `--limit 2` on any of them is a smoke test.

```bash
cd /workspace/vlm; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; R=Qwen/Qwen3-VL-8B-Instruct
python3 experiments/datasets/behaviour.py --repo $R                # replies, first-token margins, answer-spot states
python3 experiments/datasets/judge.py results/vlens/<model>__*.json  # parse yes/no, letters, test/real; rubric judge for open replies
python3 experiments/datasets/lens.py --repo $R --parts A B         # V-Read curves per family; V-Write attribution on the edits
python3 experiments/datasets/wmdp.py --repo $R                     # WMDP sandbagging: 100 items, five prompts, image variants, read and write
python3 experiments/datasets/steering.py --repo $R --parts S W T X # steering, awareness steering, word strips, map transfer
python3 experiments/datasets/directions.py --repo $R               # awareness / sandbag / authority / knowledge directions, cross-effects
python3 experiments/datasets/authority.py --repo $R                # looks-generated check, answer arrival, bearer removal
python3 experiments/datasets/depicted.py --repo $R                 # depicted-sign attribution with the mean-replacement truth
python3 experiments/datasets/behaviour.py --repo $R --families gemini_auth gemini_swap gemini_removal gemini_eval \
        --gen-open 160 --no-states --out results/vlens_long        # the Authority open prompts with room to answer
python3 experiments/datasets/judge.py results/vlens_long/<model>__*.json
python3 experiments/datasets/bearer.py --repo $R                   # which bearer class each Authority reply followed
```

`bash queue/datasets.sh $R` runs all of it (`SMOKE=1` for the two-item
version), and `tools/boxmon.sh logs/datasets.log "DONE datasets"` watches the
log and writes an `ALERT` line on a traceback or a dead queue.

Two traps: `pkill -f <pattern>` matches the ssh shell that runs it, so kill
with `pkill -x -f "bash queue/datasets.sh"` in a separate call; and a
device-side CUDA assert poisons the process, so a crashed part is re-run in a
new process.

## The V-Lens page pipeline

The scripts under `experiments/vlens/` are the experiments of the V-Lens page
in the order they were built: `gate` (does the lens earn the rest), `channels`
and `calibrate_channels` (where to read), `gallery` and `slice` (per-patch
maps), `axes_sweep`, `reader` and `reader_eval` (V-Read against the probe and
the raw reading), `cache_sets` and `judge_replies` (the audited safety sets and
their guard-model labels), `reader_loo` (leave-one-benchmark-out), `steer` and
`refusal_steer`, `regions` (grounding), `atlas` (attribution against
patching), `hookcheck`, `prefill`, `cascade`, `twostep`, `tunedcheck`,
`scenarios`, `tasks`, plus `probe_sweep` and `fit_tuned` for the baselines.
Each takes `--repo` and writes to its own folder under `results/`.

## Building the pages

On the laptop, after `tools/pull.sh`:

```bash
python3 build/report.py                 # results/vlens/report.json: every table's numbers of the datasets page
python3 build/figures.py datasets       # figures/vlens/*.png, from vlens/figures/datasets.py
python3 build/page.py                   # fills the <!--TABLE:name--> blocks in paper/datasets.html
python3 build/artifact.py datasets      # paper/datasets_artifact.html, images inlined as WebP (ALLOW_MISSING=1 to build with gaps)

python3 build/figures.py vlens          # figures/*.png of the V-Lens page, from vlens/figures/vlens_page.py
python3 build/tables.py                 # fills the tables of paper/tool.html from results/
python3 build/artifact.py tool          # paper/tool_artifact.html
```

Prose lives in the HTML between the table markers and is never touched by the
builders. The built file is published as an Artifact.

## Conventions

Figures: TeX Gyre Pagella (Palatino), no bold, no em dashes, one scale per
axis; `vlens.figures.plot.save` logs any overlapping text. WMDP question bodies
are never printed on a page: letters, accuracies and the model's own
self-reports only. Datasets that were specified but not generated (the Open
Images and VINS edits) contribute nothing and the page says so.

## Credit

The reference lens implementation vendored under `ref_jlens_repo/` is
Gurnee, Sofroniew et al., *Verbalizable Representations Form a Global Workspace
in Language Models*, Transformer Circuits, July 2026 (Apache-2.0); `vlens`
reuses its `ActivationRecorder` directly.
