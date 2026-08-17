"""
Projection heads for FEATURE-based distillation (FitNets-style hint regression,
EmbedDistill-style embedding matching).

Two modes, deliberately opposed so the pair diagnoses projector ABSORPTION:

  mode="up"    Trainable Linear(student_dim -> teacher_dim), separate heads for
               the query and candidate encoders (they are different encoders:
               encode_mm vs encode_image). Alignment happens in the TEACHER's
               space. This is the EmbedDistill-preferred direction, but the head
               (~3.1M params/side) can absorb the alignment while the student's
               native embedding barely moves.

  mode="none"  NO trainable parameters at all. Used when the teacher has been
               pre-projected offline to student_dim (see
               scripts/precompute_teacher_pca768.py). Alignment then happens
               directly on the student's NATIVE retrieval embedding, so
               absorption is structurally impossible.

Training-only in both modes: the heads are discarded at inference — retrieval
always uses the raw student embedding, so the zero-inference-cost result holds.
"""
from __future__ import annotations

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
        return nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
    raise ValueError(f"unknown head kind: {kind!r}")


class FeatureProjector(nn.Module):
    """Maps student embeddings into the space the feature loss is computed in."""

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        mode: str = "up",
        head: str = "linear",
    ):
        super().__init__()
        self.mode = mode
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim

        if mode == "none":
            if student_dim != teacher_dim:
                raise ValueError(
                    f"mode='none' requires the teacher to be pre-projected to the student dim "
                    f"({student_dim}), got teacher_dim={teacher_dim}. Run "
                    f"scripts/precompute_teacher_pca768.py and point the config at the "
                    f"*_pca{student_dim}.pt caches."
                )
            self.q_head = nn.Identity()
            self.c_head = nn.Identity()
        elif mode == "up":
            self.q_head = _build_head(student_dim, teacher_dim, head)
            self.c_head = _build_head(student_dim, teacher_dim, head)
        else:
            raise ValueError(f"unknown mode: {mode!r} (expected 'up' or 'none')")

    @property
    def has_params(self) -> bool:
        return any(p.requires_grad for p in self.parameters())

    def project_query(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.q_head(x), dim=-1)

    def project_candidate(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.c_head(x), dim=-1)


def squared_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean over batch of the SUMMED-over-dims squared L2 distance.

    Both inputs are L2-normalized, so this equals 2 - 2*cos and lives in [0, 4]
    regardless of dimensionality. Deliberately NOT nn.MSELoss, whose default
    reduction divides by the feature dim (4096) and would yield ~1e-4 values,
    forcing an absurd loss weight — the same score-scale trap that invalidated
    the earlier LaSCo distillation runs.
    """
    return (a - b).pow(2).sum(-1).mean()
