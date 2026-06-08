"""
Collator for LaSCo relation-based (group-matrix) distillation batches.

The batch unit is a GROUP. A batch of B groups, each of size G, is stacked so
the leading dimension is B and the second is G:

  ref_images     : [B, G, C, H, W]
  cand_images    : [B, G, C, H, W]
  input_ids      : [B, G, T]
  attention_mask : [B, G, T]
  teacher_matrix : [B, G, G]
  qids           : [B, G]

NOTE: `batch_size` in the training config therefore means "groups per step";
the effective number of encoded items per step is B * G references plus
B * G candidates.
"""
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RelationBatch:
    ref_images: Tensor       # [B, G, C, H, W]
    cand_images: Tensor      # [B, G, C, H, W]
    input_ids: Tensor        # [B, G, T]
    attention_mask: Tensor   # [B, G, T]
    teacher_matrix: Tensor   # [B, G, G]
    qids: Tensor             # [B, G]


class RelationCollator:
    def __call__(self, samples: list[dict]) -> RelationBatch:
        return RelationBatch(
            ref_images=torch.stack([s["ref_images"] for s in samples]),
            cand_images=torch.stack([s["cand_images"] for s in samples]),
            input_ids=torch.stack([s["input_ids"] for s in samples]),
            attention_mask=torch.stack([s["attention_mask"] for s in samples]),
            teacher_matrix=torch.stack([s["teacher_matrix"] for s in samples]),
            qids=torch.stack([s["qids"] for s in samples]),
        )
