"""
LamRA-Ret retriever wrapper — Qwen2.5-VL-7B based embedding retriever.

Standalone INFERENCE wrapper used to evaluate LamRA-Ret as a candidate TEACHER
for retriever -> retriever distillation. It is intentionally separate from the
VISTA student and from the LamRA *reranker* wrapper (src/rerankers/...); nothing
here touches existing retriever/reranker code.

Embedding mechanism — faithful to the official LamRA `demo.py` (qwen2.5vl branch):
  * '<emb>' is added as a NEW token at load time. It is only a POSITION SENTINEL:
    the embedding is the last-layer hidden state at the token immediately BEFORE
    '<emb>'. Because attention is causal and '<emb>' sits at the very end, the
    extracted token never attends to '<emb>', so the freshly-resized '<emb>' row
    is irrelevant to the feature. The lm_head is likewise unused (we read
    hidden_states[-1], not logits).
  * The feature is L2-normalized.

Prompt templates — verbatim from LamRA `dataset/datasets_circo.py` (composed CIR):
  composed query  : [image] + "{instruction} {caption}"
                    + "\nSummarize above image and sentence in one word: "
  candidate image : [image] + "\nSummarize above image in one word: "
  with assistant turn content = "<emb>."
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen2VLImageProcessor,
    Qwen2VLVideoProcessor,
)

# LamRA's CIRCO (composed image retrieval) eval instruction.
DEFAULT_CIR_INSTRUCTION = "I'm looking for a similar everyday image with the described changes."
EMB_TOKEN = "<emb>"


class LamRARetRetriever:
    """LamRA-Ret embedding retriever (teacher candidate)."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        cir_instruction: str = DEFAULT_CIR_INSTRUCTION,
    ):
        self.device = device
        self.cir_instruction = cir_instruction

        # --- Processor: built explicitly. AutoProcessor cannot resolve this
        #     checkpoint (its preprocessor_config is not auto-discovered in
        #     transformers 4.57), so we assemble image + video + tokenizer +
        #     chat_template by hand.
        image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        try:
            video_processor = Qwen2VLVideoProcessor.from_pretrained(model_path)
        except Exception:
            video_processor = Qwen2VLVideoProcessor()
        chat_template = json.loads(
            (Path(model_path) / "chat_template.json").read_text()
        )["chat_template"]
        self.processor = Qwen2_5_VLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
        )
        self.tokenizer = tokenizer

        # --- Model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
        )

        # --- '<emb>' sentinel (added AFTER weights are loaded).
        n_added = tokenizer.add_tokens([EMB_TOKEN])
        if n_added > 0:
            self.model.resize_token_embeddings(len(tokenizer))
        self.emb_token_id = tokenizer.convert_tokens_to_ids(EMB_TOKEN)

        self.model.to(device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Message builders (verbatim LamRA composed-CIR templates)
    # ------------------------------------------------------------------
    def _composed_message(self, image: Image.Image, caption: str) -> list[dict]:
        text = f"{self.cir_instruction} {caption}"
        return [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
                {"type": "text", "text": "\nSummarize above image and sentence in one word: "},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
        ]

    def _image_message(self, image: Image.Image) -> list[dict]:
        return [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "\nSummarize above image in one word: "},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
        ]

    # ------------------------------------------------------------------
    # Core encoding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode(self, messages_list: list[list[dict]]) -> torch.Tensor:
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]
        image_inputs, video_inputs = process_vision_info(messages_list)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        hidden = self.model(
            **inputs, output_hidden_states=True, return_dict=True
        ).hidden_states[-1]

        ids = inputs["input_ids"]
        emb_pos = torch.argmax((ids == self.emb_token_id).int(), dim=1)  # first '<emb>'
        feats = hidden[torch.arange(hidden.size(0), device=hidden.device), emb_pos - 1]
        return F.normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Candidate / gallery side: one embedding per image. -> [B, D]"""
        return self._encode([self._image_message(im) for im in images])

    @torch.no_grad()
    def encode_composed(
        self, images: list[Image.Image], captions: list[str]
    ) -> torch.Tensor:
        """Composed query side: (reference image + modification text). -> [B, D]"""
        return self._encode(
            [self._composed_message(im, c) for im, c in zip(images, captions)]
        )
