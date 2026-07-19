"""
Projection heads for CRD (Contrastive Representation Distillation, Tian et al.
ICLR'20). Student and teacher reps live in different dims (student 768, teacher
3584), so each is projected into a shared L2-normalized space before the
contrastive loss.

Head capacity is configurable so we can test the "projector absorbs the signal"
hypothesis:
  - mlp      : 2-layer MLP (expressive; default — can absorb the alignment)
  - linear   : single Linear (original CRD; limited absorption)
  - identity : no projection (only valid if input dim == shared dim). Forces the
               NATIVE embedding to carry the alignment (used on the student side).

Training-only: discarded at inference (retrieval uses the raw student embedding;
the projector only carries gradients back into it).
"""
import torch
import torch.nn.functional as F
from torch import nn


def _build_head(in_dim: int, out_dim: int, kind: str) -> nn.Module:
    if kind == "identity":
        if in_dim != out_dim:
            raise ValueError(f"identity head needs in_dim==out_dim ({in_dim} vs {out_dim})")
        return nn.Identity()
    if kind == "linear":
        return nn.Linear(in_dim, out_dim)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(in_dim, out_dim * 2), nn.ReLU(), nn.Linear(out_dim * 2, out_dim))
    raise ValueError(f"unknown head kind: {kind!r}")


class CRDProjector(nn.Module):
    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        proj_dim: int = 128,
        student_head: str = "mlp",
        teacher_head: str = "mlp",
    ):
        super().__init__()
        # If the student head is identity, the shared space is the student dim.
        shared = student_dim if student_head == "identity" else proj_dim
        self.shared_dim = shared
        self.g_s = _build_head(student_dim, shared, student_head)
        self.g_t = _build_head(teacher_dim, shared, teacher_head)

    def project_student(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.g_s(x), dim=-1)

    def project_teacher(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.g_t(x), dim=-1)
