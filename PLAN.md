# Induced Blindsight — locked research plan

*Revised 2026-08-26 after prior-art recon. Supersedes the framing in `plan.md`.*

## Why the original framing changed

The recon found the "silent visual monologue" readout is **scooped several times over**:

| Prior work | What it already did |
|---|---|
| Neo et al., ICLR 2025 (2410.07149) | logit lens on LLaVA visual tokens; object info spatially localized to its patch |
| Jiang et al., ICLR 2025 (2410.02762) | per-patch logit lens → zero-shot segmentation (mIoU 54.26); **PROJECTAWAY** = directional ablation of an object concept in image tokens |
| Esmaeilkhani & Latecki (2602.01530) | turned per-patch logit-lens maps into a training loss |
| Liu et al., COLM 2025 (2510.04819) | "33.3% of BLINK Art Style: perception present in the LM, not surfaced" |
| Hawrani, LessWrong 10 Jul 2026 + repo `jude-sph/J-lens-Vision` | ported J-lens to LLaVA-1.5-7B image tokens; per-patch transport; probe-vs-lens; a *natural* report/discrimination dissociation |

Hawrani also published the damaging control: at the LM layers, **plain logit lens gives essentially the same answer as J-lens**. So J-lens cannot be sold as a better microscope.

**Consequence.** J-lens is not our microscope. It is our *operational definition of "can say."* That is the only thing that licenses the workspace framing, and it is what makes the causal claim below meaningful.

## The claim

> Membership in the verbalizable workspace is **causally necessary for verbal report** and
> **causally unnecessary for forced-choice discrimination** — and because the workspace does not
> record where its contents came from, an image can write into it.

Three sections, each novel, each positive whichever way the number falls.

### S1 — Induced blindsight (the core causal result)

Blindsight is *defined* by chance-level report with above-chance forced choice. We induce exactly
that pattern on demand.

Design is a full 2x2, not a single dissociation:

|                      | free report of concept c | 2AFC discrimination of c |
|----------------------|--------------------------|--------------------------|
| ablate J-lens dir    | collapses                | preserved                |
| ablate probe dir ⟂ J | preserved                | collapses                |

Plus: **dose-response** in ablation strength α (the two thresholds should separate), and
**reversal** by re-injecting the direction (Arditi's activation-addition converse).

Controls: matched-norm random direction; orthogonal-concept direction; ablation at text
positions instead of image positions; LEACE-style check that the concept is genuinely
linearly undetectable after erasure.
Named baseline: **PROJECTAWAY** — it ablates and measures only the destroyed half.

### S2 — The workspace is modality-blind (why S3 is possible)

Provenance (did concept c enter through the image or through the text?) is linearly decodable
from the residual stream at high AUC but at **chance from its J-space component**.
Content without source ⇒ the model cannot introspect on whether it saw or was told.
This is the mechanistic counterpart of the instruction/data inseparability results.

### S3 — Workspace injection (the attack)

L∞-bounded image perturbation optimised against the *Jacobian readout* so the lens reports a
chosen word at image positions **while the generated text is unchanged**.
Positioning: the **dual of Obfuscated Activations** (2412.09565) — that work *hides* a concept
from a latent monitor; we *plant* one. Distinguished from Image Hijacks (2309.00236) by the
output-unchanged rate, which we must report alongside the lens-hit rate.

## Non-negotiable reporting

- Always run **logit lens side by side** with J-lens. If they agree, say so.
- Report the **output-unchanged rate** for every attack.
- Report the **second diagonal** of the 2x2, or it is a single dissociation and reviewers will say so.
- Cite Hawrani's post *and* repo generously; cite PROJECTAWAY as the baseline ablation.

## Compute

Single A30 24GB. Full 24-layer lens on Qwen3-VL-2B = 26 s/prompt ⇒ 120 prompts ≈ 52 min.
Cost is set by `d_model` backward passes per prompt and is independent of the number of
source layers, so always fit every layer.

Models: Qwen3-VL-2B/4B/8B-Instruct, InternVL3.5-2B/4B (different encoder + connector),
Qwen3-VL-4B-Thinking. Data: COCO val2017 + panoptic + captions (local).
