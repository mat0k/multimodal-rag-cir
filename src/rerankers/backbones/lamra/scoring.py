from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def extract_emb_token_feature(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    emb_token_id: int,
) -> torch.Tensor:
    """Extract feature vectors from the token right before `<emb>`.

    This follows LamRA demo behavior and returns one feature vector per sequence.
    """
    if hidden_states.ndim != 3:
        raise ValueError(f"Expected hidden_states shape [B, T, D], got {tuple(hidden_states.shape)}")

    emb_token_mask = input_ids == emb_token_id
    if not torch.all(emb_token_mask.any(dim=1)):
        raise ValueError("Embedding token not found in one or more sequences.")

    emb_positions = torch.argmax(emb_token_mask.int(), dim=1)
    feature_positions = torch.clamp(emb_positions - 1, min=0)
    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_indices, feature_positions]


def l2_normalize(embeddings: torch.Tensor) -> torch.Tensor:
    """L2-normalize embeddings along the last dimension."""
    return F.normalize(embeddings, dim=-1)


def pairwise_similarity(
    left_embeddings: torch.Tensor,
    right_embeddings: torch.Tensor,
    score_type: Literal["dot", "cosine"] = "cosine",
) -> torch.Tensor:
    """Compute pairwise similarity matrix where higher means better match."""
    if score_type not in {"dot", "cosine"}:
        raise ValueError(f"Unsupported score_type '{score_type}'. Use 'dot' or 'cosine'.")

    left = left_embeddings
    right = right_embeddings
    if score_type == "cosine":
        left = l2_normalize(left)
        right = l2_normalize(right)

    return left @ right.T


def pointwise_relevance_score(
    query_embedding: torch.Tensor,
    candidate_embedding: torch.Tensor,
    score_type: Literal["dot", "cosine"] = "cosine",
) -> torch.Tensor:
    """Return a continuous scalar relevance score for one query-candidate pair."""
    if query_embedding.ndim != 2 or candidate_embedding.ndim != 2:
        raise ValueError(
            "query_embedding and candidate_embedding must both be 2D tensors with shape [B, D]."
        )
    if query_embedding.shape[0] != 1 or candidate_embedding.shape[0] != 1:
        raise ValueError(
            "pointwise_relevance_score expects a single query and a single candidate (batch size 1)."
        )

    score_matrix = pairwise_similarity(
        left_embeddings=query_embedding,
        right_embeddings=candidate_embedding,
        score_type=score_type,
    )
    return score_matrix[0, 0]


__all__ = [
    "extract_emb_token_feature",
    "l2_normalize",
    "pairwise_similarity",
    "pointwise_relevance_score",
]
