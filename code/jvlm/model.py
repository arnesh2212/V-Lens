# SPDX-License-Identifier: Apache-2.0
"""Multimodal LensModel adapter.

The reference ``jlens`` implementation is typed against a text-only decoder:
``encode(str) -> input_ids`` then ``forward(input_ids)``.  A VLM needs a whole
batch dict (pixel values, grids, mRoPE position ids), so we keep the same
*shape* of interface but pass prepared batches around instead of strings.

Everything the Jacobian estimator needs is unchanged:
  * ``layers``   -- the LM residual blocks, hook boundaries
  * ``forward``  -- builds an autograd graph through those blocks
  * ``unembed``  -- final norm + LM head
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def _resolve(obj: Any, path: str) -> Any:
    return functools.reduce(getattr, path.split("."), obj)


@dataclass(frozen=True)
class VLMLayout:
    """Where the pieces live inside a HF ``*ForConditionalGeneration``.

    Attributes:
        mm: dotted path to the multimodal trunk -- the module that accepts
            ``pixel_values`` and returns LM hidden states.
        text: dotted path to the bare text decoder (owns ``layers``/``norm``).
        layers/norm/embed: attribute names on the text decoder.
        lm_head: attribute name on the top-level model.
        n_deepstack: number of leading LM layers that receive an in-place
            additive visual injection (Qwen3-VL DeepStack).  Source layers must
            start at or after this index: the injection writes in place, and an
            in-place write to a tensor we marked ``requires_grad_(True)`` is a
            hard autograd error.
        image_replicate: batch dict keys repeated along dim 0 when the prompt is
            replicated for the Jacobian estimator.
    """

    mm: str
    text: str
    layers: str = "layers"
    norm: str = "norm"
    embed: str = "embed_tokens"
    lm_head: str = "lm_head"
    n_deepstack: int = 0
    image_replicate: tuple[str, ...] = (
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "image_sizes",
    )


_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("model", "model.language_model"),
    ("model", "model.text_model"),
    ("model", "model"),
    ("language_model", "language_model"),
)


def _find_layout(hf: nn.Module) -> VLMLayout:
    for mm_path, text_path in _CANDIDATES:
        try:
            text = _resolve(hf, text_path)
            _resolve(hf, mm_path)
        except AttributeError:
            continue
        if all(hasattr(text, a) for a in ("layers", "norm", "embed_tokens")) and hasattr(
            hf, "lm_head"
        ):
            n_ds = 0
            vcfg = getattr(hf.config, "vision_config", None)
            idx = getattr(vcfg, "deepstack_visual_indexes", None)
            if idx is not None:
                n_ds = len(idx)
            return VLMLayout(mm=mm_path, text=text_path, n_deepstack=n_ds)
    raise ValueError(
        f"could not locate the text decoder inside {type(hf).__name__}; "
        "pass layout= explicitly"
    )


class VLMLensModel:
    """A :mod:`jlens`-compatible wrapper around a HuggingFace VLM.

    Mutates the wrapped model in place: all parameters get
    ``requires_grad_(False)``.  The Jacobian fit differentiates with respect to
    *activations* only, which also keeps the retained graph small -- nothing
    upstream of the graph root builds a graph at all, so the vision tower costs
    memory only for its own forward.
    """

    def __init__(
        self,
        hf_model: nn.Module,
        processor: Any,
        *,
        layout: VLMLayout | None = None,
        name: str = "",
    ) -> None:
        self.hf = hf_model
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.name = name or type(hf_model).__name__

        hf_model.eval()
        for p in hf_model.parameters():
            p.requires_grad_(False)

        self.layout = layout or _find_layout(hf_model)
        self._mm = _resolve(hf_model, self.layout.mm)
        self._text = _resolve(hf_model, self.layout.text)
        self.layers: nn.ModuleList = getattr(self._text, self.layout.layers)
        self._norm: nn.Module = getattr(self._text, self.layout.norm)
        self._embed: nn.Module = getattr(self._text, self.layout.embed)
        self._lm_head: nn.Module = getattr(hf_model, self.layout.lm_head)

        tcfg = hf_model.config.get_text_config()
        self.n_layers = int(tcfg.num_hidden_layers)
        self.d_model = int(tcfg.hidden_size)
        self.vocab_size = int(tcfg.vocab_size)
        self._softcap = getattr(tcfg, "final_logit_softcapping", None)
        self.n_deepstack = self.layout.n_deepstack
        if len(self.layers) != self.n_layers:
            raise ValueError(
                f"config says {self.n_layers} layers, found {len(self.layers)}"
            )

        self.image_token_id = self._find_image_token_id()

    # ---------------------------------------------------------------- helpers

    def _find_image_token_id(self) -> int | None:
        cfg = self.hf.config
        for attr in ("image_token_id", "image_token_index"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int):
                return v
        tok = self.tokenizer
        for s in ("<|image_pad|>", "<image>", "<IMG_CONTEXT>", "<start_of_image>"):
            try:
                i = tok.convert_tokens_to_ids(s)
            except Exception:
                continue
            if i is not None and i >= 0 and i != getattr(tok, "unk_token_id", -1):
                return int(i)
        return None

    @property
    def device(self) -> torch.device:
        return self._embed.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self._embed.weight.dtype

    def image_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Boolean ``[B, S]`` mask marking image-placeholder positions."""
        if self.image_token_id is None:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        return input_ids == self.image_token_id

    # ------------------------------------------------------------- input prep

    def build(self, text: str, images: list | None = None, **kw) -> dict:
        """Processor call -> batch dict on the model device."""
        kwargs = dict(text=[text], return_tensors="pt")
        if images:
            kwargs["images"] = images
        kwargs.update(kw)
        batch = self.processor(**kwargs)
        return {
            k: (v.to(self.device) if torch.is_tensor(v) else v)
            for k, v in dict(batch).items()
        }

    def chat_prompt(self, text: str, images: list | None = None, *, system: str | None = None,
                    cue: str | None = None, history: list | None = None) -> str:
        """The chat-templated prompt string.

        ``cue`` is text placed *before* the image inside the user turn, so that
        it can influence what is written into the image positions; ``text``
        goes after.  ``system`` fills the system slot when given.  ``history``
        is a list of prior (user_text, assistant_text) pairs for a follow-up turn.
        """
        def user_turn(t, with_image):
            c: list[dict] = []
            if with_image and cue:
                c.append({"type": "text", "text": cue})
            if with_image:
                for _ in images or []:
                    c.append({"type": "image"})
            c.append({"type": "text", "text": t})
            return {"role": "user", "content": c}

        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": [{"type": "text", "text": system}]})
        # the image and the cue belong to the first user turn; a follow-up turn is text only
        for j, (u, a_) in enumerate(history or []):
            msgs.append(user_turn(u, with_image=(j == 0)))
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": a_}]})
        msgs.append(user_turn(text, with_image=not history))
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def chat(self, text: str, images: list | None = None, *, system: str | None = None,
             cue: str | None = None, history: list | None = None, **kw) -> dict:
        """Same as :meth:`build` but wrapped in the model's chat template."""
        prompt = self.chat_prompt(text, images, system=system, cue=cue, history=history)
        return self.build(prompt, images, **kw)

    def chat_batch(self, prompts: list[str], images: list | None = None) -> dict:
        """One batch holding several prompts about the same image.

        Every experiment here asks four or more questions about one photograph,
        and running them as separate forwards spends most of the time re-encoding
        the same picture.  Prompts are right-padded: a causal mask never looks
        rightward, so tokens before the padding are unaffected, and the logits we
        want sit at each row's own final real position (returned as ``lengths``).
        """
        batches = [self.chat(p, images) for p in prompts]
        lengths = [b["input_ids"].shape[1] for b in batches]
        smax = max(lengths)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        out: dict = {}
        for k in batches[0]:
            v0 = batches[0][k]
            if not torch.is_tensor(v0):
                out[k] = v0
            elif k in self.layout.image_replicate:
                out[k] = torch.cat([b[k] for b in batches], dim=0)
            elif v0.ndim == 2 and v0.shape[0] == 1:
                fill = {"input_ids": pad_id, "attention_mask": 0}.get(k, 0)
                rows = [
                    torch.nn.functional.pad(b[k], (0, smax - b[k].shape[1]), value=fill)
                    for b in batches
                ]
                out[k] = torch.cat(rows, dim=0)
            elif v0.ndim >= 2 and v0.shape[0] == 1:
                raise ValueError(
                    f"chat_batch does not know how to stack {k!r} "
                    f"(shape {tuple(v0.shape)}); keeping row 0 would silently "
                    f"apply one prompt's value to every row")
            else:
                out[k] = v0
        out["_lengths"] = lengths
        return out

    def replicate(self, batch: dict, n: int) -> dict:
        """Repeat a batch-of-1 ``n`` times along the batch axis.

        Sequence tensors are ``expand``ed (no copy).  Vision tensors are packed
        variable-length along dim 0 for the Qwen/InternVL families, so those are
        ``repeat``ed instead -- ``n`` copies of the patches feed ``n`` copies of
        the image-token span.
        """
        if n == 1:
            return batch
        out = {}
        for k, v in batch.items():
            if not torch.is_tensor(v):
                out[k] = v
            elif k in self.layout.image_replicate:
                out[k] = v.repeat(n, *([1] * (v.ndim - 1)))
            elif v.ndim >= 2 and v.shape[0] == 1:
                out[k] = v.expand(n, *v.shape[1:])
            else:
                out[k] = v
        return out

    # --------------------------------------------------------------- run/read

    def forward(self, batch: dict) -> Any:
        """Run the multimodal trunk (vision tower + LM blocks), no LM head."""
        kw = {k: v for k, v in batch.items()
              if k != "labels" and not k.startswith("_")}
        kw["use_cache"] = False
        return self._mm(**kw)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        w = self._lm_head.weight
        logits = self._lm_head(self._norm(residual.to(w.dtype).to(w.device)))
        if self._softcap is not None:
            logits = self._softcap * torch.tanh(logits / self._softcap)
        return logits

    def __repr__(self) -> str:
        return (
            f"VLMLensModel({self.name}, n_layers={self.n_layers}, "
            f"d_model={self.d_model}, deepstack={self.n_deepstack})"
        )


# --------------------------------------------------------------------- loading

_MODEL_REGISTRY = {
    "qwen3vl": "Qwen3VLForConditionalGeneration",
    "internvl": "InternVLForConditionalGeneration",
    "gemma3": "Gemma3ForConditionalGeneration",
    "llava": "LlavaForConditionalGeneration",
}


def load_vlm(
    repo: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    attn: str = "sdpa",
    max_pixels: int | None = None,
    device_map: str | dict | None = None,
) -> VLMLensModel:
    """Load a VLM by HF repo id and wrap it.

    ``max_pixels`` caps the visual token budget for Qwen-family processors,
    which is the main lever on sequence length (and therefore fit cost).
    """
    import transformers

    cfg = transformers.AutoConfig.from_pretrained(repo, trust_remote_code=True)
    arch = (cfg.architectures or ["AutoModelForImageTextToText"])[0]
    cls = getattr(transformers, arch, None)
    if cls is None:
        cls = transformers.AutoModelForImageTextToText

    proc_kw: dict = {"trust_remote_code": True}
    if max_pixels is not None:
        proc_kw["max_pixels"] = max_pixels
    try:
        processor = transformers.AutoProcessor.from_pretrained(repo, **proc_kw)
    except (TypeError, ValueError):
        # only the Qwen-family processors take a token budget; the rest resize
        # to a fixed grid and reject the argument
        proc_kw.pop("max_pixels", None)
        processor = transformers.AutoProcessor.from_pretrained(repo, **proc_kw)

    load_kw = dict(dtype=dtype, attn_implementation=attn,
                   trust_remote_code=True)
    if device_map is not None:
        # a model too large for one card: accelerate places the layers and the
        # hooks still fire, but activations then live on several devices, so
        # anything that indexes them must move its own tensors rather than
        # assume a single device
        load_kw["device_map"] = device_map
        model = cls.from_pretrained(repo, **load_kw)
    else:
        model = cls.from_pretrained(repo, **load_kw).to(device)
    return VLMLensModel(model, processor, name=repo)


def unpatchify_qwen(pixel_values, grid_thw, processor) -> "np.ndarray":
    """Invert the Qwen image processor's patch packing, back to an RGB array.

    The processor flattens an image into patches with a ten-way ``view`` and a
    ``permute``; recovering the picture means undoing exactly that, which is the
    only way to *show* an adversarial perturbation rather than merely quote its
    L-infinity norm.  Returns ``uint8`` ``[H, W, 3]``.
    """
    import numpy as np

    ip = processor.image_processor
    ps = int(getattr(ip, "patch_size", 16))
    ms = int(getattr(ip, "merge_size", 2))
    tp = int(getattr(ip, "temporal_patch_size", 2))
    gt, gh, gw = [int(x) for x in grid_thw[0].tolist()]
    c = 3

    x = pixel_values.detach().float().cpu()
    x = x.reshape(gt, gh // ms, gw // ms, ms, ms, c, tp, ps, ps)
    # forward permute was (0,1,4,7,5,8,3,2,6,9) on the 10-dim view; with the
    # batch axis dropped the inverse is this ordering
    x = x.permute(0, 5, 6, 1, 3, 7, 2, 4, 8)      # gt, c, tp, gh/ms, ms, ps, gw/ms, ms, ps
    x = x.reshape(gt, c, tp, gh * ps, gw * ps)[0, :, 0]     # c, H, W

    mean = np.array(getattr(ip, "image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
    std = np.array(getattr(ip, "image_std", [0.5, 0.5, 0.5]), dtype=np.float32)
    img = x.numpy().transpose(1, 2, 0) * std + mean
    return np.clip(img * 255.0, 0, 255).astype("uint8")
