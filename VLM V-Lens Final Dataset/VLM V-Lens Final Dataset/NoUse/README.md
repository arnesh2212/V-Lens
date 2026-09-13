# Handoff — prompts and their images

```
images/         202 nuScenes frames, 1600x900, no boxes drawn on them
prompts.csv     one row per sample: the image, and the prompts for it
prompts.jsonl   the same, machine-readable
```

1163 samples on 202 images. Several samples share an image — each one targets a
different string in that frame — so join on `id`, not on the image filename.

## Columns

| column | meaning |
|---|---|
| `id` | sample id |
| `image` | path relative to this folder |
| `target_string` | the string in the image this sample is about |
| `decoy_string` | a string **not** present in this image |
| `cue_neutral` / `cue_named` / `cue_decoy` | the bare instruction, no chat template |
| `prompt_neutral` / `prompt_named` / `prompt_decoy` | the same cues wrapped in LLaVA-1.5's format |

Use the `prompt_*` columns for LLaVA-1.5. For any other model, take the `cue_*`
strings and wrap them in that model's own chat template.

## The three prompts

Same image, same pixels. Only the instruction changes.

* **neutral** — asks for text without naming any
* **named** — names `target_string`, which is in the image
* **decoy** — names `decoy_string`, which is not

Neutral and named ask whether naming a string changes anything. Decoy is the
control for that comparison: it names a string in the same way, but one that
isn't there, so it separates the effect of having a quoted string in the prompt
from the effect of that string being in the picture.

`decoy_string` is checked against every string in the image, not just the ones
this sample uses, and substring overlaps are excluded too.

## One thing not to change

**The instruction comes before `<image>`, not after.**

These models are causal, so anything at an image position only sees tokens before
it. If the instruction is moved after `<image>` — which is the more common
layout — it can no longer affect the image positions at all, and the three
conditions become identical wherever those positions are read.

Keep the order: instruction, then image, then the question.

## If more is needed

`../experiment/00_data/rows.jsonl` has the same samples plus the pixel boxes for
the text and its object, the per-string boxes, and the image-token index sets
derived from them — for scoring what should be decodable at a given image token.
`../experiment/` has the runner and analysis if they're wanted.
