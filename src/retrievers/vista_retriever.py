from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.retrievers.backbones.vista.modeling import Visualized_BGE
from src.retrievers.base import TwoEncoderVLM


@dataclass
class VisionEncoderOutput:
    image_embeds: torch.Tensor


@dataclass
class TextEncoderOutput:
    text_embeds: torch.Tensor


class VistaImageProcessor:
    """Adapter that exposes HF-like processor output expected by dataset loaders."""

    def __init__(self, preprocess_fn: Any):
        self.preprocess_fn = preprocess_fn

    def __call__(self, image: Any, return_tensors: str = "pt", **_: Any) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("VistaImageProcessor currently supports return_tensors='pt' only.")

        tensor = self.preprocess_fn(image)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return {"pixel_values": tensor}


class VistaVisionEncoder(nn.Module):
    def __init__(self, backbone: Visualized_BGE):
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values: torch.Tensor) -> VisionEncoderOutput:
        return VisionEncoderOutput(image_embeds=self.backbone.encode_image(pixel_values))


class VistaTextEncoder(nn.Module):
    def __init__(self, backbone: Visualized_BGE):
        super().__init__()
        self.backbone = backbone

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> TextEncoderOutput:
        tokenized = {"input_ids": input_ids, "attention_mask": attention_mask}
        return TextEncoderOutput(text_embeds=self.backbone.encode_text(tokenized))


class VistaBGERetriever(TwoEncoderVLM):
    def __init__(self, backbone: Visualized_BGE):
        super().__init__()
        self.backbone = backbone
        self.vision = VistaVisionEncoder(backbone)
        self.text = VistaTextEncoder(backbone)
        self.image_processor = VistaImageProcessor(backbone.preprocess_val)
        self.tokenizer = backbone.tokenizer

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> "VistaBGERetriever":
        # model_name_or_path is used as BGE model id or local config path for the text branch.
        model_weight = kwargs.pop("checkpoint_path", kwargs.pop("model_weight", None))
        if not model_weight:
            raise ValueError(
                "A checkpoint path is required. Provide --checkpoint_path or pass model_weight in --retriever_init_kwargs."
            )

        backbone = Visualized_BGE(
            model_name_bge=model_name_or_path,
            model_weight=model_weight,
            **kwargs,
        )
        return cls(backbone)
