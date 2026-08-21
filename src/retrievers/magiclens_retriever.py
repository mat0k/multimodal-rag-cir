"""MagicLens retriever wrapper exposing the `TwoEncoderVLM` contract.

Mirrors `VistaBGERetriever` / `VistaMMRetriever` so MagicLens runs through the
existing `evaluate_fashioniq` / `evaluate_cirr` pipelines and the importlib
`load_retriever` plugin path with no changes to either.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.retrievers.base import TwoEncoderVLM
from src.retrievers.backbones.magiclens.modeling import MagicLens
from src.retrievers.vista_retriever import VistaImageProcessor
from src.utils.tensor import make_normalized


@dataclass
class VisionEncoderOutput:
    image_embeds: torch.Tensor


class MagicLensVisionEncoder(nn.Module):
    def __init__(self, backbone: MagicLens):
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values: torch.Tensor) -> VisionEncoderOutput:
        return VisionEncoderOutput(image_embeds=self.backbone.encode_image(pixel_values))


class MagicLensTextEncoder(nn.Module):
    """Placeholder for the `legacy_fusion` path, which MagicLens cannot support.

    MagicLens has no text-only tower: its pooler consumes a fused
    image+text sequence, so a text embedding alone is not defined. Only
    `query_embedding_mode='vista_mm'` (native multimodal encoding) is available.
    """

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        raise NotImplementedError(
            "MagicLens has no text-only encoder; use query_embedding_mode='vista_mm' "
            "(native multimodal query encoding) rather than 'legacy_fusion'."
        )


class MagicLensRetriever(TwoEncoderVLM):
    """Wraps a `MagicLens` backbone for the evaluation pipelines."""

    def __init__(self, backbone: MagicLens):
        super().__init__()
        self.backbone = backbone
        self.vision = MagicLensVisionEncoder(backbone)
        self.text = MagicLensTextEncoder()
        # VistaImageProcessor is a generic adapter over any callable transform
        # (PIL -> tensor); it carries no VISTA-specific behaviour.
        self.image_processor = VistaImageProcessor(backbone.preprocess_val)
        self.tokenizer = backbone.tokenizer

    def encode_query_mm(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokenized = {"input_ids": input_ids, "attention_mask": attention_mask}
        return make_normalized(self.backbone.encode_mm(pixel_values, tokenized))

    @classmethod
    def from_pretrained(cls, model_name_or_path: str = "base", **kwargs: Any) -> "MagicLensRetriever":
        """Build from converted weights.

        `model_name_or_path` is the MagicLens size ('base' or 'large'); the
        converted `.pt` state_dict is passed as `checkpoint_path`.
        """
        checkpoint_path = kwargs.pop("checkpoint_path", None) or kwargs.pop("model_weight", None)
        if not checkpoint_path:
            raise ValueError(
                "A checkpoint path is required. Pass --checkpoint_path (or model_weight in "
                "--retriever_init_kwargs) pointing at the converted MagicLens .pt state_dict, "
                "produced by scripts/convert_magiclens_weights.py."
            )
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"MagicLens checkpoint not found: {checkpoint_path}")

        backbone = MagicLens(model_size=model_name_or_path or "base", **kwargs)
        backbone.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        backbone.eval()
        return cls(backbone)


__all__ = ["MagicLensRetriever"]
