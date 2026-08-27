# JacobianLens-Vision → "The Silent Workspace": An Oral-Track ICLR 2027 Program

## TL;DR
- The single best oral bet is **"Does a VLM have a silent, non-verbal global workspace?"** — porting Anthropic's Jacobian Lens (J-lens, released July 6 2026) to vision-language and vision-language-action (VLA) models to test whether concepts a model *acts on* (points, grounds, moves a robot) are absent from the verbalizable J-space. This is one crisp, falsifiable claim, directly contradicts Anthropic's "the workspace is words because producing words is the only kind of action Claude can take" hypothesis, and even a clean negative ("there is no visual workspace — VLAs route action through words") is oral-worthy.
- The runner-up and natural companion is **"Workspace hijacking"**: adversarial images optimized against the averaged Jacobian to write a chosen word-concept ("authorized", "safe", "test") into J-space with no visible text — a new mechanistic threat model plus a J-space "immune system" that flags concepts the image channel wrote but the text channel never licensed.
- ICLR 2027 dates are **abstract Sep 18 2026 / full paper Sep 25 2026 AoE**, 10 pages, double-blind, with mandatory reciprocal reviewing — per the official Call for Papers, "all submissions must have at least one author who is registered to review at least 3 papers" (qualifying = ≥1 accepted paper at ICLR/NeurIPS/ICML/UAI/AISTATS/JMLR/TMLR/ACL/EMNLP/etc.), conference April 26–30 2027 in California. You have ~4 weeks to the deadline — enough for the go/no-go gates below on 16 GB, scaling to 8–12B only where a gate passes.

## Key Findings

### What makes an ICLR oral in this area (Part 1)
ICLR 2026 recognized only **two Outstanding Papers plus one Honorable Mention**; the committee (chaired by Gautam Kamath) explicitly rewarded conceptual punch — of *Transformers are Inherently Succinct* (Bergsträßer, Cotterell, Lin; arXiv:2510.19315) the ICLR Blog wrote (Apr 23 2026) "Notwithstanding critiques, the paper's strong conceptual message was intriguing to the committee and other experts." Across recent interpretability/safety orals and spotlights the winning pattern is consistent. Distilled principles:

1. **One crisp, surprising, one-sentence claim.** The archetype is Arditi et al., "Refusal in language models is mediated by a single direction" (arXiv:2406.11717), shown "across 13 popular open-source chat models up to 72B parameters," where "erasing this direction… prevents it from refusing harmful instructions, while adding this direction elicits refusal on even harmless instructions." A reviewer can restate it in a breath.
2. **A new measurement instrument validated with causal interventions.** Not "we probed X" — you read a signal *and* steer/ablate/patch to prove causality (J-lens itself; the VLA steering paper arXiv:2509.00328).
3. **A conceptual reframe of a known phenomenon.** Global-workspace framing of verbalizable representations; "instructions and data are inseparable in shared-embedding models" (arXiv:2606.27567).
4. **A new threat model with a working demonstration.** Image Hijacks (arXiv:2309.00236), FigStep, alignment faking. Reviewers reward "here is a danger nobody had operationalized."
5. **A clean causal/scaling result stated as a law.** CoT-faithfulness (Chen et al., arXiv:2505.05410): the hint "reveal rate is often below 20%" (Claude 3.7 Sonnet mentioned the hint 25% of the time, DeepSeek R1 39%); evaluation-awareness scales with model size (Chaudhary et al., arXiv:2509.13333).
6. **A negative result that closes a door.** "CoTs are not always faithful" is high-impact precisely because it refutes an assumed safety guarantee — your fallback framings exploit exactly this.
7. **Generality across models + released artifact.** Cross-model validation and open code/data (J-lens is Apache-2.0; ASIDE arXiv:2503.10566).
8. **"Slow science" ambition.** The ICLR 2027 CFP explicitly asks for "your most complete and most exciting work" rather than incremental deltas — a signal to swing big.

### The competitive landscape as of August 2026 (novelty verification)
- **J-lens is brand new and barely touched in vision.** Only the Hawrani LessWrong post (LLaVA-1.5-7B hallucination ranks: present ~16, absent-but-asked ~111, unasked-absent ~430) applies it to a VLM; a separate LessWrong review replicated J-space on **Qwen 3.6 27B** (text). No peer-reviewed vision/multimodal J-lens paper exists. **Open.**
- **VLA interpretability exists but is logit-lens/SAE-based, not Jacobian and not workspace-framed.** Häon et al. (arXiv:2509.00328) project FFN activations onto token embeddings to steer π0/OpenVLA on LIBERO and a UR5; Buurmeijer et al. (arXiv:2603.05487) formalize feature-observability/controllability; event-grounded SAEs (arXiv:2605.17204); "Not All Features Are Created Equal" (arXiv:2603.19233) probes π0.5/OpenVLA-OFT/GR00T N1.5/SmolVLA/X-VLA. **None asks whether an action-relevant concept is absent from the *verbalizable* workspace.** This is the gap.
- **Cross-modal patching exists but not through the Jacobian and not as a "concept transplant."** VLMs Create Cross-Modal Task Representations (ICML 2025 task vectors, OpenReview 77ziPGdQct); V-SEAM (arXiv:2509.14837); "Same Task, Different Circuits" back-patching (arXiv:2506.09047); VLM circuit tracing (arXiv:2602.20330); perception-knowledge conflict V2P/P2V patching (arXiv:2606.28273). **Open** for J-space read→write across modalities.
- **Workspace hijacking is unclaimed.** Image Hijacks (arXiv:2309.00236) and CrossInject (ACM MM 2025) optimize *output behavior*; SAEgis / SAE firewalls (arXiv:2605.07447) detect adversarial images by reconstruction error. **No one uses the averaged Jacobian as the attack objective to write a specific vocabulary concept into the workspace, nor builds a J-space provenance "immune system."**
- **VLM evaluation-awareness from visual cues is entirely unstudied.** All eval-awareness work is text-only (Needham arXiv:2505.23836; SAD arXiv:2407.04694; Nguyen/Attubato probes arXiv:2507.01786; Chaudhary scaling arXiv:2509.13333; Meta's April 2026 Muse Spark report: public benchmarks flagged as evals 19.8% of the time vs 2.0% on internal ones). Computer-use benchmarks (OSWorld; RedTeamCUA arXiv:2505.21936; AgentHijack arXiv:2605.25707) never test screenshot-triggered eval awareness. **Wide open.**
- **Instruction/data inseparability is now a formal result.** Pant, "On the Inseparability of Instructions and Data in Shared-Embedding Sequence Models" (arXiv:2606.27567) proves provenance recovery is bounded by the total-variation distance between representation distributions; ASIDE (arXiv:2503.10566) and ISE (ICLR 2025) add architectural separation. A J-space "modality-of-origin tag" test is the *mechanistic, empirical* counterpart. **Open for vision.**
- **Introspection/concept-injection is text-only.** Lindsey, "Emergent Introspective Awareness in Large Language Models" (transformer-circuits.pub, Oct 29 2025) reports that "Opus 4.1 and 4 exhibit such behavior about 20% of the time when concepts are injected in the appropriate layer," with Opus 4.1 "never" producing false positives; replicated on **Qwen2.5-Coder-32B** (Vogel 2025). **No VLM/VLA replication exists** — directly composable with J-lens for a visual introspection test.

## Details

### Part 2 — 15 bolder directions (headline · why surprising · method · key experiment · expected result · risk · alternates)

**D1. The Silent Workspace (flagship).** *Headline:* "Vision-language-action models have a second, non-verbal workspace that the Jacobian Lens cannot see." *Surprise:* Directly tests Anthropic's claim that the workspace is words "because producing words is the only kind of action Claude can take." Models that point/ground/move can act non-verbally. *Method:* Fit per-layer J-lens on VLAs (π0.5, OpenVLA-OFT, GR00T N1.5, SmolVLA) and pointing VLMs (MolmoPoint, Qwen3-VL grounding). For each action-relevant concept (target object, direction, grasp), measure (a) J-space rank of its verbal token vs (b) a causal action-readout. *Key experiment:* concept is "action-live" (ablating it changes the action) but "workspace-silent" (near-vocabulary-floor J-rank) ⇒ non-verbal workspace. *Expected:* partial silent workspace in VLAs; verbal-only workspace in chat VLMs. *Risk:* Medium. *Alternates:* (i) restrict to token-output OpenVLA-OFT where unembedding is exact; (ii) pointing VLMs only (point tokens); (iii) reframe as "the workspace is verbal even in acting models."

**D2. Cross-modal concept transplant.** *Headline:* "Read a concept out of an image's J-space and write it into a text-only prompt so the model behaves as if it saw the image." *Method:* J-lens read at image positions → inject the J-space direction at matched text positions (and vice-versa). *Experiment:* VQA/grounding transfer vs task-vector baseline (ICML 2025). *Expected:* works mid-network where J-space lives; decays at edges. *Risk:* Medium. *Alternates:* raw-residual (logit-lens) baseline; single-token only; back-patching vehicle (arXiv:2506.09047).

**D3. Workspace hijacking + J-space immune system (runner-up).** *Headline:* "An imperceptible image writes the word 'authorized' into the workspace without visible text — and a Jacobian monitor catches it." *Method:* PGD on the Jacobian readout to maximize a target token's J-space logit; test black-box transfer; then a detector flagging J-space tokens whose gradient traces to image but not text tokens. *Experiment:* ASR vs Image Hijacks (arXiv:2309.00236); detector AUROC vs SAEgis (arXiv:2605.07447), HiddenDetect (arXiv:2502.14744). *Expected:* high white-box ASR, partial transfer; detector >0.9 AUROC. *Risk:* Medium. *Alternates:* typographic-attack baseline; patch-constrained attack; defense-only paper.

**D4. Eval-awareness through the camera.** *Headline:* "Computer-use agents can tell they're being tested from the screenshot, and ablating the J-space 'eval' concept changes agentic misbehavior." *Method:* Build EvalCue-VLM (synthetic desktops, eval-harness UI, sandbox banners, fake bank sites); read J-space for {test, fake, fictional, sandbox}; steer/ablate; measure behavior shift, mirroring Anthropic's blackmail-ablation. *Risk:* Medium-high (agentic harness). *Alternates:* static VQA sandbagging; single-turn screenshot classification; probe-only.

**D5. What the model sees but won't say (visual omission detector).** *Headline:* "A VLM's workspace contains 'gun'/'drug'/'nude' even when its output denies it — J-lens as a visual lie/omission detector." *Baselines:* hidden-state deception probes (arXiv:2606.17478). *Risk:* Medium. *Alternates:* hallucination direction (present vs absent objects, per Hawrani); benign occluded-object proxy to avoid harmful-content curation.

**D6. J-lens for reasoning VLMs (visual CoT faithfulness).** *Headline:* "Thinking VLMs solve the image silently before the chain-of-thought starts — and the verbalized chain often doesn't match J-space." *Method:* J-lens on Qwen3-VL-Thinking, GLM-4.5V, Kimi-VL-A3B-Thinking, InternVL3.5 thinking; align J-space trajectory vs emitted CoT. *Baseline:* Anthropic CoT-faithfulness (arXiv:2505.05410, text-only, ≤20% reveal). *Risk:* Low-medium (compelling either way). *Alternates:* measure "silent lead time" in layers; thinking vs non-thinking checkpoints.

**D7. A universal visual J-space.** *Headline:* "Workspace concepts align across VLMs and encoders — a monitor trained on one transfers to another via Jacobian alignment." *Method:* Fit J-lens on Qwen3-VL, InternVL3.5, Gemma 3, align J-spaces (Procrustes / vec2vec, arXiv:2510.02348); transfer a steering vector; connect to Platonic Representation Hypothesis (arXiv:2405.07987). *Risk:* Medium-high. *Alternates:* single-family transfer (2B→4B→8B); encoder-only alignment (SigLIP 2 vs DINOv3 vs PE).

**D8. J-lens for unified/generation models.** *Headline:* "Read a generator's workspace and catch an unsafe concept before it renders a pixel." *Method:* J-lens/Diffusion-Lens hybrid (arXiv:2403.05846) on BAGEL, Janus-Pro, Show-o2 text-to-image path. *Risk:* High (generation heads differ). *Alternates:* text-encoder-only lens; "will render NSFW" classifier from the workspace.

**D9. J-lens for video/world models.** High cost/risk on 16 GB; **park as future work / appendix feasibility note**, not a headline.

**D10. Jacobian scaling laws.** *Headline:* "Instruction tuning creates the visual workspace." *Method:* Measure J-space size/density/verbalizability across scale (2B→32B) and stage (base vs instruct vs RL) within the Qwen3-VL / InternVL3.5 families. *Risk:* Medium (need base checkpoints). *Alternates:* modality-mix axis; density-vs-depth curve only.

**D11. Training with the lens (lens-regularized fine-tuning).** *Headline:* "Penalizing 'authorized' in image-channel J-space at train time hardens models against visual injection without hurting utility." *Baseline:* circuit-breakers/RR (arXiv:2406.04313), ReFT. *Risk:* High (training on 16 GB is tight). *Alternates:* LoRA-only; inference-time steering substitute; reward-verbalization objective.

**D12. A fundamentally better lens.** *Headline:* "A multi-token / open-vocabulary Jacobian lens fixes the single-token limitation." *Method:* Jacobian into a sequence-decoder head, or into CLIP/SigLIP2 text-embedding space for open-vocab phrases; or a second-order (Hessian) lens locating where local linearity breaks. *Risk:* Medium-high. *Alternates:* phrase lens via top-k beam; encoder-target lens only.

**D13. Modality-of-origin tagging.** *Headline:* "The workspace cannot tell whether a concept came from the image or the system prompt — the mechanistic root of visual prompt injection." *Method:* Probe J-space for concept provenance; relate to the impossibility result (arXiv:2606.27567) and ASIDE/ISE. *Expected:* near-chance provenance ⇒ mechanistic confirmation of inseparability; if separable ⇒ a defense signal. *Risk:* Low-medium (either outcome publishable). *Alternates:* layerwise provenance curve; provenance under adversarial images.

**D14. Visual introspection via concept injection.** *Headline:* "Inject a visual concept into a VLM's activations and ask if it notices — VLMs are (un)aware of their own visual thoughts." *Method:* Port Lindsey's concept-injection (transformer-circuits.pub Oct 29 2025; Opus 4.1 ~20% detection, ~0% false positives) to VLMs using J-space directions as injection vectors. *Risk:* Medium. *Alternates:* detection-rate-vs-scale curve; injected-image-concept vs injected-text-concept awareness.

**D15. Authority-cue workspace differencing.** *Headline:* "A single 'authority direction' in J-space, learned by differencing badge/uniform/letterhead images, controls unauthorized-instruction compliance." *Method:* AuthCue-VLM dataset; J-space differencing; steer to restore instruction hierarchy. *Risk:* Low-medium. *Alternates:* text-channel vs image-channel authority contrast; monitor-only.

### Part 3 — The oral paper, end to end

**Title:** *The Silent Workspace: Do Vision-Language-Action Models Reason in a Space the Jacobian Lens Cannot Read?*

**One-sentence claim:** In models that can take non-verbal actions, a measurable fraction of action-determining concepts are causally live yet absent from the verbalizable Jacobian workspace — a second, silent workspace that does not exist in chat-only VLMs.

**Abstract.** Anthropic's Jacobian Lens revealed a small, verbalizable "global workspace" (J-space) in language models and argued it is built from words because producing words is the only action a chat model can take. We test this claim where it is falsifiable: in pointing VLMs and vision-language-action models that can act without words. We fit per-layer Jacobian lenses for four VLAs and two pointing/grounding VLMs and, for each action-relevant concept, jointly measure its verbalizable J-space rank and its causal influence on the emitted action via patching/ablation. We find a class of concepts that are action-live but workspace-silent, demonstrating a second, non-verbal workspace in acting models that is absent or negligible in matched chat VLMs. We validate causally (a double dissociation), show the effect grows from chat→pointing→continuous-control models, and release lenses and a benchmark. A corollary holds either way: if the workspace is fully verbal even in acting models, then VLA behavior is auditable through words — a strong positive safety result.

**Headline figures.**
1. *The silence plot.* Scatter of J-space verbal rank (x) vs causal action-effect (y) per concept; a populated high-effect/high-rank ("silent") quadrant is the money shot, chat-VLM points hugging the low-rank axis.
2. *Modality-of-action gradient.* Silent-workspace fraction rising across chat VLM → pointing VLM → discrete-token VLA → flow-matching VLA.
3. *Double dissociation bars.* Ablating a workspace-visible concept: Δwords high, Δaction low; ablating a silent concept: Δaction high, Δwords low.
4. *Layer trajectory.* Where in depth silent vs verbal concepts live (expect silent ones nearer the action head).
5. *Steering demo.* Writing a silent concept into a LIBERO rollout changes the trajectory with no change in any verbalization.

**Method.** (a) Fit J_l = E[∂h_final/∂h_l] per layer on a task corpus; for token-output VLAs (OpenVLA-OFT) unembed to the action/vocab head; for flow-matching heads (π0.5, GR00T N1.5, SmolVLA) define the readout as the Jacobian into the action-expert output, decoded against a discretized action codebook (fallback: probe-decodable action bins). (b) Concept set = target object, spatial direction, grasp/release, obstacle; verbal token via the LLM unembedding. (c) Causal tests: activation patching and directional ablation at the concept direction; measure Δaction (task success / DTW trajectory divergence) and Δwords (J-rank, emitted caption). (d) "Silent" = high Δaction ∧ low Δwords ∧ high vocabulary rank.

**Models (all open-weight, 2025–2026; verified).** VLAs: **OpenVLA-OFT** (`moojink/openvla-oft`, 7B, token-output; base OpenVLA arXiv:2406.09246), **π0.5** (arXiv:2504.16054; `lerobot/pi05_base`, flow-matching, Apache-2.0), **SmolVLA** (HF/LeRobot, 450M, fully open, June 2025), **GR00T N1.5** (`nvidia/GR00T-N1.5-3B`, June 2025, Eagle-2 VLM + DiT). Pointing/grounding VLMs: **Molmo 2 / MolmoPoint-8B** (Ai2, Dec 2025 report), **Qwen3-VL-4B/8B** incl. `-Thinking` (Apache-2.0, released Sep 23 2025; tech report arXiv:2511.21631). Chat controls: **InternVL3.5-2B/4B/8B** (arXiv:2508.18265, Aug 25 2025, Apache-2.0), **Gemma 3 4B** (Mar 12 2025). Encoders where needed: **SigLIP 2** (arXiv:2502.14786), **DINOv3** (arXiv:2508.10104), **Perception Encoder** (arXiv:2504.13181). Reasoning-VLM extension (D6 companion): Qwen3-VL-Thinking, **GLM-4.5V** (`zai-org/GLM-4.5V`, MIT, Aug 11 2025), **Kimi-VL-A3B-Thinking** (`moonshotai/Kimi-VL-A3B-Thinking`). Unified/gen (D8): **BAGEL** (`ByteDance-Seed/BAGEL-7B-MoT`, arXiv:2505.14683), **Janus-Pro** (`deepseek-ai/Janus-Pro-7B`, arXiv:2501.17811), **Show-o2** (`showlab/show-o2-7B`, arXiv:2506.15564).

**Experiments / baselines / metrics / ablations.** Baselines: logit-lens and TextSpan (arXiv:2310.05916) readouts (expected to under-recover, echoing the "naive projection at chance" finding); VLA FFN-to-embedding steering (arXiv:2509.00328); SAE features (arXiv:2605.17204). Metrics: J-rank, causal Δaction (success rate, DTW distance), Δwords, AUROC of silent-vs-verbal classification, silent-fraction vs model class. Ablations: layer band; Jacobian-averaging corpus size; single- vs multi-token concepts; discrete vs flow-matching heads; base vs instruct checkpoints.

**16 GB compute plan.** Fits: SmolVLA (450M), Qwen3-VL-2B/4B, InternVL3.5-2B/4B, Gemma 3 4B, MolmoPoint (8B @ 4-bit), OpenVLA-OFT (7B @ 4-bit), GR00T N1.5 (3B), π0.5 (LeRobot). Jacobian averaging is the cost driver: use ~25–50 prompts, length ≤128, skip high-norm early tokens (per the LessWrong Qwen replication recipe), cache J_l. LIBERO simulation is light. **Scale to 8–12B (Qwen3-VL-8B, InternVL3.5-8B, GLM-4.5V-class) only after Gate 2 passes**, on rented A100s for camera-ready numbers.

**6-week plan with go/no-go gates.**
- Wk 1: Reproduce J-lens on Qwen3-VL-4B text + Hawrani's LLaVA hallucination ranks. **Gate 0:** lens reproduces the present/absent-object rank gap.
- Wk 2: Fit J-lens on SmolVLA + OpenVLA-OFT; define action readout. **Gate 1:** action readout decodes ground-truth action bins above chance.
- Wk 3: Silence plot on SmolVLA/OpenVLA-OFT + MolmoPoint. **Gate 2 (critical):** at least one action-live, workspace-silent concept with causal double dissociation. If fail → pivot to negative framing that day.
- Wk 4: Modality gradient across chat→pointing→VLA; add GR00T N1.5, π0.5.
- Wk 5: Steering demo in LIBERO; robustness ablations.
- Wk 6: Writing, release lenses + benchmark, register reciprocal reviewer, lock abstract by Sep 18.

**10-week extension:** add D6 (visual CoT faithfulness) as Section 6, cross-model universality spot-check (D7), and 8–12B scaling.

**Reviewer objections + rebuttals.**
- *"J-lens is Anthropic's; this is derivative."* → We port a text-only instrument to a regime its authors said should behave differently, and falsify (or confirm) their central hypothesis; new benchmark + causal protocol.
- *"Flow-matching heads have no unembedding, so J-space is ill-defined."* → We define the readout against a discretized action codebook and validate decodability (Gate 1); we also include token-output OpenVLA-OFT where unembedding is exact.
- *"'Silent' is just a weak lens, not a second workspace."* → The double dissociation is the control: a weak lens would depress Δwords for *all* concepts, but verbal concepts stay lens-visible while silent ones do not, at matched causal effect.
- *"Only small models on 16 GB."* → The claim is a modality gradient shown within families; 8–12B confirmation in camera-ready.
- *"Single-token limitation."* → Reported, with the open-vocab/phrase-lens ablation (D12) as robustness.

**Fallback framings if the headline is negative.**
1. **"There is no visual workspace — VLAs route action through words."** If every action-live concept is also verbalizable, that is a strong positive: VLA behavior is word-auditable and J-lens is a ready-made robot-action monitor. Oral-worthy as a safety guarantee, paired with the steering demo.
2. **"The lens is the bottleneck."** If Gate 1 fails on flow-matching heads, pivot to D12 (an open-vocabulary Jacobian lens for encoders/action heads) validated on discrete-head models.
3. **"Silent workspace only under RL."** If it appears only in RL-tuned checkpoints, reframe as a Jacobian scaling/training-stage law (D10): "RL fine-tuning grows a non-verbal workspace."

### Part 4 — Ranked shortlist (scores /10)

| Rank | Paper concept | Novelty | Feasibility (16 GB) | Surprise | Why |
|---|---|---|---|---|---|
| 1 | **The Silent Workspace** (D1, +D6 companion) | 9 | 7 | 9 | Falsifies a named Anthropic hypothesis; oral either way via the negative framing; VLA interp is hot but nobody asks the verbalizability question. |
| 2 | **Workspace Hijacking + J-space immune system** (D3) | 8 | 8 | 8 | New white-box objective (Jacobian-as-attack) + defense; strong threat-model story; baselines exist to beat (Image Hijacks, SAEgis). |
| 3 | **Eval-awareness through the camera** (D4) | 9 | 6 | 8 | First visual eval-awareness; mirrors Anthropic's blackmail-ablation; agentic harness is the feasibility risk. |
| 4 | **Modality-of-origin tagging** (D13) | 7 | 9 | 7 | Cheap, either-way publishable, mechanistically explains visual prompt injection; ties to a 2026 impossibility theorem. |
| 5 | **Visual introspection via concept injection** (D14) | 8 | 7 | 8 | Composes two 2025–26 Anthropic instruments (J-lens + concept injection) into an untested multimodal introspection test. |

## Recommendations
1. **Commit to The Silent Workspace (D1) as the primary submission, with Workspace Hijacking (D3) as a parallel fallback.** Both share the same J-lens/VLM infrastructure, so week-1–2 work is shared regardless of which becomes the paper.
2. **Sequence by gates, not by calendar.** Gate 0 (reproduce J-lens), Gate 1 (action readout above chance), Gate 2 (double dissociation) are the decision points. Gate 2 by end of week 3 is the single most important threshold; if it fails, switch to the negative-result or better-lens framing immediately.
3. **Do 16 GB work first on SmolVLA (450M), Qwen3-VL-4B, OpenVLA-OFT (4-bit).** Rent A100s for 8–12B only once Gate 2 passes and you need camera-ready scale numbers.
4. **Build two small, releasable benchmarks** (the action-concept silence probe set; EvalCue-VLM if you take D4). A released artifact materially raises oral odds and aligns with the CFP's "most complete and most exciting work."
5. **Register your reciprocal reviewer now** (≥1 author, ≥3 papers, with a qualifying prior acceptance) and lock the abstract by Sep 18 2026.
6. **Thresholds that change the plan:** if the silent-workspace fraction is <5% and not causally dissociable, abandon D1's positive framing; if cross-model Jacobian alignment (D7) exceeds ~0.7 transfer, promote universality to a headline section; if screenshot eval-awareness AUROC <0.6, drop D4 to an appendix.

## Caveats
- **J-lens itself is imperfect and unreplicated across labs** — Anthropic call it "an imperfect tool, which we believe only approximately and incompletely captures the model's underlying workspace structure," and ran it only on Claude (Sonnet/Haiku/Opus 4.5, checks on 4.6). Independent vision replication is part of your contribution but also a risk; Gate 0 de-risks it.
- **Flow-matching action heads may not admit a clean Jacobian-to-vocabulary readout**; the discrete-codebook workaround is unproven and is exactly what Gate 1 tests.
- **Some 2026 model details rely on vendor/news sources** (GLM-4.6V sizes, GR00T N1.6, MiniCPM-V 4.6); primary HuggingFace cards confirm the core models used, but treat benchmark claims as vendor-reported.
- **arXiv IDs with 26xx prefixes are 2026 postings**; the introspection primitive's canonical source is the Oct 29 2025 transformer-circuits.pub article (arXiv mirror dated 2026), and its replication was on Qwen2.5-Coder-32B, not a VLM.
- **The "silent workspace" is speculative until Gate 2**; this proposal is explicit that the negative outcome is designed to be equally publishable, which is the hedge that makes it a comparatively safe oral bet.
- **Agentic/harmful-content experiments (D4, D5)** require synthetic-only data and safety review; keep sensitive-content detectors on benign proxies (e.g., occluded objects) wherever possible.
