# The production channel of a vision–language model

Porting Anthropic's Jacobian lens (July 2026) to vision–language models, and
finding that the subspace it identifies carries **verbal report** rather than
perception or decision.

> Erase a concept from it and the model's English naming falls from 99.6% to
> 2.1% while it still presses the right key 96.6% of the time and still names the
> object in Chinese 76.1% of the time. Saying is not choosing.

## Layout

```
code/jvlm/        the library
  model.py        VLM adapter: batching, prompt replication, image reconstruction
  fitting.py      the averaged-Jacobian estimator, generalised to image batches
  readout.py      lens readout, lens vectors, the intervention hook
  tasks.py        the two response channels (a word, or an arbitrary key)
  percept.py      perceptual directions and their split into verbal / use copies
  lexicon.py      the same concept through four output vocabularies
  concepts.py     concept families beyond nouns: colour, count, spatial relation
  cocodata.py     items, hard distractors, tokenisation
  stats.py        concept-clustered bootstrap, signed-rank, McNemar
  figures.py      every paper figure
  style.py        Palatino, no bold, validated categorical palette
code/exp/         one script per experiment, numbered in the order they were built
scripts/          sync, GPU lanes, watchdog, analysis
paper/            the artifact source, its build script, and the results ledger
results/          everything the experiments wrote
figures/          every figure, plus WebP copies for the artifact
```

## Running it

Fitting a lens is the only expensive step. The cost is `d_model` backward passes
per prompt and is *independent of how many source layers are requested*, so
always fit every layer:

```bash
python code/exp/01_fit_lens.py --repo Qwen/Qwen3-VL-2B-Instruct --kind mm --n 120
```

Then any experiment against that lens, for example the headline one:

```bash
python code/exp/10_routes.py --lens results/lenses/<...>.pt --n 110 --tag qwen2b
```

`scripts/lane_{a,b,c,d}.sh` are the queues that were actually used on a single
24 GB card; every step is skipped when its output already exists, so a
preemption costs at most the step in flight. `scripts/analyse.sh` turns results
into the report, the figures and the qualitative examples;
`scripts/refresh.sh` pulls them down and rebuilds the artifact.

## Two implementation notes that cost time

Qwen3-VL injects DeepStack visual features **in place** into the first three
language-model layers. The Jacobian fit marks its graph root with
`requires_grad_(True)`, and an in-place write to that tensor is a hard autograd
error, so source layers must start at or after `n_deepstack`.

Prompts about the same photograph are batched with **right** padding and read at
each row's own final position. A causal mask never looks rightward, so the real
tokens are unaffected; left padding would have needed the model's position and
vision bookkeeping to cooperate, and right padding does not.

## Credit

The lens, the workspace, and the text-only evidence this extends are
Gurnee, Sofroniew et al., *Verbalizable Representations Form a Global Workspace
in Language Models*, Transformer Circuits, 6 July 2026. `ref_jlens_repo/`
vendors their reference implementation (Apache-2.0); `jvlm.fitting` is their
estimator generalised to multimodal batches, and `jvlm` reuses their
`ActivationRecorder` and `JacobianLens` directly so a lens fitted here loads in
their code and vice versa.
