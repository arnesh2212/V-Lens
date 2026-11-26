from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def _resolve(obj: Any, path: str) -> Any:
    return functools.reduce(getattr, path.split("."), obj)


@dataclass(frozen=True)
class VLMLayout:

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
        self._thinking_flag = None
        if len(self.layers) != self.n_layers:
            raise ValueError(
                f"config says {self.n_layers} layers, found {len(self.layers)}"
            )

        self.image_token_id = self._find_image_token_id()


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
        if self.image_token_id is None:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        return input_ids == self.image_token_id


    def build(self, text: str, images: list | None = None, **kw) -> dict:
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
        for j, (u, a_) in enumerate(history or []):
            msgs.append(user_turn(u, with_image=(j == 0)))
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": a_}]})
        msgs.append(user_turn(text, with_image=not history))
        return self.apply_template(msgs)

    def apply_template(self, msgs: list[dict]) -> str:
        kw = dict(tokenize=False, add_generation_prompt=True)
        think = os.environ.get("VLENS_THINKING") == "1"
        if self._thinking_flag is not False:
            try:
                out = self.processor.apply_chat_template(msgs, enable_thinking=think, **kw)
                self._thinking_flag = True
                return out
            except (TypeError, ValueError, KeyError):
                self._thinking_flag = False
        return self.processor.apply_chat_template(msgs, **kw)

    def chat(self, text: str, images: list | None = None, *, system: str | None = None,
             cue: str | None = None, history: list | None = None, **kw) -> dict:
        prompt = self.chat_prompt(text, images, system=system, cue=cue, history=history)
        return self.build(prompt, images, **kw)

    def chat_batch(self, prompts: list[str], images: list | None = None) -> dict:
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

    def batch_many(self, items: list[dict], *, side: str = "right") -> dict:
        texts = [self.chat_prompt(it["question"], [it["image"]] if it.get("image") is not None else None,
                                  system=it.get("system"), cue=it.get("cue")) for it in items]
        imgs = [it["image"] for it in items if it.get("image") is not None]
        old = getattr(self.tokenizer, "padding_side", None)
        try:
            if old is not None:
                self.tokenizer.padding_side = side
            kw = dict(text=texts, return_tensors="pt", padding=True)
            if imgs:
                if "gemma" in type(self.processor).__name__.lower():
                    kw["images"] = [[it["image"]] if it.get("image") is not None else [] for it in items]
                else:
                    kw["images"] = imgs
            batch = self.processor(**kw)
        finally:
            if old is not None:
                self.tokenizer.padding_side = old
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        am = batch.get("attention_mask")
        batch["_lengths"] = (am.sum(-1).tolist() if am is not None
                             else [batch["input_ids"].shape[1]] * len(items))
        batch["_side"] = side
        return batch

    def answer_spot(self, batch: dict) -> torch.Tensor:
        S = batch["input_ids"].shape[1]
        if batch.get("_side") == "left":
            return torch.full((batch["input_ids"].shape[0],), S - 1, dtype=torch.long)
        return torch.tensor([n - 1 for n in batch["_lengths"]], dtype=torch.long)

    def replicate(self, batch: dict, n: int) -> dict:
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


    def forward(self, batch: dict) -> Any:
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


def load_vlm(
    repo: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    attn: str = "sdpa",
    max_pixels: int | None = None,
    device_map: str | dict | None = None,
) -> VLMLensModel:
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
        proc_kw.pop("max_pixels", None)
        processor = transformers.AutoProcessor.from_pretrained(repo, **proc_kw)
    _gst = os.environ.get("VLENS_GEMMA_SOFT_TOKENS")
    if _gst and "gemma" in type(processor).__name__.lower():
        ip = getattr(processor, "image_processor", None)
        for attr in ("max_soft_tokens", "image_seq_length"):
            if ip is not None and hasattr(ip, attr):
                setattr(ip, attr, int(_gst))
        if hasattr(processor, "image_seq_length"):
            processor.image_seq_length = int(_gst)

    load_kw = dict(dtype=dtype, attn_implementation=attn,
                   trust_remote_code=True)
    if device_map is not None:
        load_kw["device_map"] = device_map
        model = cls.from_pretrained(repo, **load_kw)
    else:
        model = cls.from_pretrained(repo, **load_kw).to(device)
    return VLMLensModel(model, processor, name=repo)
