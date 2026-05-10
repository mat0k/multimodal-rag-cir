"""
Collator for LaSCo distillation batches.

Each sample contains:
  - VISTA query inputs (ref image + tokenized text)
  - 1 positive image
  - K negative images
  - Teacher scores (pos_score, neg_scores[K])
"""
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class DistillBatch:
    ref_images: Tensor          # [B, C, H, W]
    input_ids: Tensor           # [B, T]
    attention_mask: Tensor      # [B, T]
    pos_images: Tensor          # [B, C, H, W]
    neg_images: Tensor          # [B, K, C, H, W]
    teacher_pos_scores: Tensor  # [B]
    teacher_neg_scores: Tensor  # [B, K]


class DistillCollator:
    def __call__(self, samples: list[dict]) -> DistillBatch:
        return DistillBatch(
            ref_images=torch.stack([s["ref_image"] for s in samples]),
            input_ids=torch.stack([s["input_ids"] for s in samples]),
            attention_mask=torch.stack([s["attention_mask"] for s in samples]),
            pos_images=torch.stack([s["pos_image"] for s in samples]),
            neg_images=torch.stack([s["neg_images"] for s in samples]),
            teacher_pos_scores=torch.stack([s["teacher_pos_score"] for s in samples]),
            teacher_neg_scores=torch.stack([s["teacher_neg_scores"] for s in samples]),
        )
