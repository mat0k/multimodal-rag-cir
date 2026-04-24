from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TrainingBatch:
    """Collated batch ready to pass into VISTA backbone.forward()."""
    ref_images: torch.Tensor       # [B, C, H, W]
    input_ids: torch.Tensor        # [B, T]
    attention_mask: torch.Tensor   # [B, T]
    target_images: torch.Tensor    # [B, C, H, W]


class LaSCoCollator:
    """Collates LaSCo triplets into VISTA-compatible training batches."""

    def __call__(self, samples: list[dict[str, Any]]) -> TrainingBatch:
        return TrainingBatch(
            ref_images=torch.stack([s["ref_image"] for s in samples]),
            input_ids=torch.stack([s["input_ids"] for s in samples]),
            attention_mask=torch.stack([s["attention_mask"] for s in samples]),
            target_images=torch.stack([s["target_image"] for s in samples]),
        )
