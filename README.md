# V-Lens

Code for **"Says Who? We Read and We Write the Decisions of Vision-Language Models"**.

V-Lens looks at a vision-language model's answer at the *answer spot*, the last prompt position before the reply.

- **V-Read** reads the answer from any middle layer, using a linear map fitted in closed form.
- **V-Write** shows which image patches pushed the answer, and by how much.

`vlens/` is the library. `experiments/` and `tools/` reproduce the paper.

## Install

```bash
git clone git@github.com:arnesh2212/V-Lens.git && cd V-Lens
pip install -e .                  # the library
pip install -r requirements.txt   # extra packages for the experiments
```

## Data

The SimonSays benchmark is on the Hugging Face Hub: [`Ironman2212/SimonSays`](https://huggingface.co/datasets/Ironman2212/SimonSays).

```python
from datasets import load_dataset
ds = load_dataset("Ironman2212/SimonSays", "authority", split="test")
```

To run the experiments, download it into `data/simonsays/`:

```bash
python3 tools/fetch_simonsays.py
```

## Quick example

```python
import torch, vlens
from vlens import corpus as C
from vlens.lens import attribution as AT, readout, scores

m = vlens.load_vlm("Qwen/Qwen3-VL-8B-Instruct")
fn = scores.score_fn(scores.TokenSets(m.tokenizer), "yesno")   # log-odds of "yes" over "no"
l = m.n_layers // 2

# V-Write: one value per image token, for the answer score at layer l
b = m.chat("Is there a dog in the image? Answer yes or no.", [C.load_image("photo.jpg", square=448)])
pos = readout.image_positions(m, b)
with torch.no_grad():
    _, Hs, _, lg0 = AT.states_at(m, b, [l])
fn, _ = scores.orient(fn, "yesno", float(fn(None, lg0).float().cpu()[0]))
h = Hs[l]
patch_map = AT.path_reading(m, b, fn, l, pos, h, h.mean(0, keepdim=True).expand_as(h), steps=8)
```

V-Read is fitted with `vlens.lens.tuned.fit_ridge_cv`. See `tools/layer_readouts.py` for V-Read and `tools/vwrite_maps.py` for V-Write in full.

## Reproducing the paper

All commands are in [REPRODUCE.md](REPRODUCE.md), grouped by the paper's tables and figures. Every script takes `--help`.

Models: `Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3.5-27B` and `google/gemma-4-31B-it`, with more in the appendix. All run in bfloat16 on one GPU; the 27B and 31B models need 80 GB or more.

## License

The code is Apache-2.0. The dataset has its own licences; see the dataset card.
