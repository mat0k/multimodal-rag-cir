"""
BGE-VL-MLLM-S1 retriever wrapper (LLaVA-NeXT-Mistral-7B based, MegaPairs-only).

Used as the CLEAN teacher for retriever -> retriever distillation: S1 is trained
*exclusively* on MegaPairs (synthetic triplets from open-domain images) and has
NOT seen FashionIQ/CIRR training data — unlike LamRA-Ret (M-BEIR) or
BGE-VL-MLLM-S2 (fine-tuned on MMEB). Never swap this for S2.

Exposes the same interface as LamRARetRetriever so the existing eval and
precompute paths work unchanged:
    encode_images(list[PIL])            -> [B, 4096] L2-normalized
    encode_composed(list[PIL], list[str]) -> [B, 4096] L2-normalized

Inference follows the official model card:
    inputs = model.data_process(images=..., text=..., q_or_c="q"|"c",
                                task_instruction=<CIR instruction>)
    emb    = model(**inputs, output_hidden_states=True)[:, -1, :]   # last token
    emb    = F.normalize(emb, dim=-1)
Images are resized to 512x512 (the resolution the model was trained at).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel

# The model's own default CIR instruction (from modeling_llavanext_for_embedding.py)
DEFAULT_CIR_INSTRUCTION = (
    "Retrieve the target image that best meets the combined criteria by using "
    "both the provided image and the image retrieval instructions: "
)
IMAGE_SIZE = 512


class BGEVLRetriever:
    """BGE-VL-MLLM-S1 embedding retriever (clean teacher candidate)."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        cir_instruction: str = DEFAULT_CIR_INSTRUCTION,
        image_size: int = IMAGE_SIZE,
    ):
        self.device = device
        self.cir_instruction = cir_instruction
        self.image_size = image_size

        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype
        )
        self.model.set_processor(model_path)
        self._patch_transformers_compat()
        self.model.eval()
        self.model.to(device)

    def _patch_transformers_compat(self) -> None:
        """Shim for newer transformers (>=4.47) vs the model's shipped custom code.

        The custom forward calls `self.image_newline`, which lived on
        LlavaNextForConditionalGeneration in older transformers. After the
        refactor it is an nn.Parameter on the INNER model (`self.model`), so the
        top-level lookup fails. We alias it back (via object.__setattr__ so we
        don't double-register the parameter). vision_tower / multi_modal_projector
        / pack_image_features are still reachable and need no patching.
        """
        # (1) image_newline moved to the inner model.
        if not hasattr(self.model, "image_newline"):
            inner = getattr(self.model, "model", None)
            newline = getattr(inner, "image_newline", None)
            if newline is None:
                raise AttributeError(
                    "Could not locate `image_newline` on the BGE-VL model or its inner model; "
                    "transformers API may have changed again."
                )
            object.__setattr__(self.model, "image_newline", newline)

        # (2) pack_image_features now returns a LIST of per-image tensors; the
        #     custom code expects the older concatenated TENSOR. Restore that.
        _orig_pack = self.model.pack_image_features

        def _compat_pack(*args, **kwargs):
            feats, lens = _orig_pack(*args, **kwargs)
            if isinstance(feats, (list, tuple)):
                feats = torch.cat(list(feats), dim=0)
            return feats, lens

        object.__setattr__(self.model, "pack_image_features", _compat_pack)

    # ------------------------------------------------------------------
    def _prep(self, images: list[Image.Image]) -> list[Image.Image]:
        # Match the model's own preprocessing order: resize -> convert("RGB").
        return [im.resize((self.image_size, self.image_size)).convert("RGB") for im in images]

    def _build_inputs(self, images, texts, q_or_c: str):
        """Replicates model.data_process() but takes PIL images instead of paths.

        The shipped data_process() hardcodes Image.open(path), so it cannot accept
        in-memory PIL images (our datasets yield PIL). We reuse the model's own
        prepare_text_input() so the prompt construction stays byte-identical —
        it only checks `image is not None`, never the image content.
        """
        instruction = self.cir_instruction if "q" in q_or_c else None
        text_input = [
            self.model.prepare_text_input(
                image=im, text=tx, q_or_c=q_or_c, task_instruction=instruction
            )
            for im, tx in zip(images, texts)
        ]
        inputs = self.model.processor(
            images=images, text=text_input, return_tensors="pt", padding=True
        )
        return inputs.to(self.device)

    @torch.no_grad()
    def _embed(self, inputs) -> torch.Tensor:
        """Last-layer hidden state at the last token, L2-normalized.

        The model card does `model(...)[:, -1, :]`, which assumed the older API
        returned a bare tensor. Newer transformers returns a ModelOutput, so we
        pull the last-layer hidden states explicitly. NOTE: we must use hidden
        states (4096-d), never `logits` (vocab-sized).
        """
        out = self.model(**inputs, output_hidden_states=True)
        if isinstance(out, torch.Tensor):
            hidden = out
        elif getattr(out, "hidden_states", None) is not None:
            hidden = out.hidden_states[-1]
        elif getattr(out, "last_hidden_state", None) is not None:
            hidden = out.last_hidden_state
        else:
            raise RuntimeError(f"Cannot locate hidden states on model output: {type(out)}")
        emb = hidden[:, -1, :]                   # last-token embedding
        return F.normalize(emb.float(), dim=-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Candidate / gallery side: one embedding per image. -> [B, 4096]"""
        imgs = self._prep(images)
        return self._embed(self._build_inputs(imgs, [None] * len(imgs), "c"))

    @torch.no_grad()
    def encode_composed(
        self, images: list[Image.Image], captions: list[str]
    ) -> torch.Tensor:
        """Composed query side: (reference image + modification text). -> [B, 4096]"""
        imgs = self._prep(images)
        return self._embed(self._build_inputs(imgs, list(captions), "q"))
