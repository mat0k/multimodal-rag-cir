from typing import Any

import torch

from src.retrievers.vista_retriever import VistaBGERetriever
from src.utils.tensor import make_normalized


class VistaMMRetriever(VistaBGERetriever):
    """Vista retriever that exposes native multimodal query encoding."""

    def encode_query_mm(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokenized = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return make_normalized(self.backbone.encode_mm(pixel_values, tokenized))


__all__ = ["VistaMMRetriever"]