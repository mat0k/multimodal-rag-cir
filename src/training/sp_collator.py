"""
Collator for SP (candidate-candidate) relation distillation batches.

A batch is N (query, target) pairs from one cluster:
  ref_images         : [N, C, H, W]
  target_images      : [N, C, H, W]
  input_ids          : [N, T]
  attention_mask     : [N, T]
  teacher_target_emb : [N, d_teacher]
  target_ids         : list[str]   (for CE duplicate-positive masking)
  qids               : [N]
"""
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SPBatch:
    ref_images: Tensor
    target_images: Tensor
    input_ids: Tensor
    attention_mask: Tensor
    teacher_target_emb: Tensor
    target_ids: list
    qids: Tensor
    # Feature-based KD only (query-side embedding matching). None for SP/RKD/CRD.
    teacher_query_emb: Tensor | None = None


class SPCollator:
    def __call__(self, samples: list[dict]) -> SPBatch:
        has_q = "teacher_query_emb" in samples[0]
        return SPBatch(
            ref_images=torch.stack([s["ref_image"] for s in samples]),
            target_images=torch.stack([s["target_image"] for s in samples]),
            input_ids=torch.stack([s["input_ids"] for s in samples]),
            attention_mask=torch.stack([s["attention_mask"] for s in samples]),
            teacher_target_emb=torch.stack([s["teacher_target_emb"] for s in samples]),
            target_ids=[s["target_id"] for s in samples],
            qids=torch.tensor([s["qid"] for s in samples], dtype=torch.long),
            teacher_query_emb=(
                torch.stack([s["teacher_query_emb"] for s in samples]) if has_q else None
            ),
        )
