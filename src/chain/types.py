from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


DatasetName = Literal["cirr", "fashioniq"]
StageName = Literal["retriever_candidates", "chain_rerank"]


@dataclass(frozen=True)
class QueryRecord:
	"""Dataset-agnostic query payload used across chain stages."""

	query_id: str
	dataset: DatasetName
	split: str
	reference_name: str
	text_edit: str
	target_name: str | None = None
	query_class: str | None = None
	pair_id: int | str | None = None
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateScore:
	"""Single candidate entry with a continuous relevance score."""

	candidate_id: str
	score: float
	rank: int | None = None
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateListRecord:
	"""Stage-A retriever output for one query."""

	query: QueryRecord
	candidates: list[CandidateScore]
	retriever_top_m: int
	target_in_top_m: bool | None = None
	target_rank_in_top_m: int | None = None
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankedRecord:
	"""Stage-B reranker output for one query."""

	query: QueryRecord
	candidates: list[CandidateScore]
	rerank_top_n: int
	target_in_rerank_top_n: bool | None = None
	target_rank_in_rerank: int | None = None
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChainRunMetadata:
	"""Top-level metadata for either Stage-A or Stage-B runs."""

	stage: StageName
	dataset: DatasetName
	split: str
	created_at_utc: str
	seed: int | None = None
	candidate_depth_m: int | None = None
	rerank_depth_n: int | None = None
	retriever: dict[str, Any] = field(default_factory=dict)
	reranker: dict[str, Any] = field(default_factory=dict)
	runtime: dict[str, Any] = field(default_factory=dict)
	metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
	"CandidateListRecord",
	"CandidateScore",
	"ChainRunMetadata",
	"DatasetName",
	"QueryRecord",
	"RerankedRecord",
	"StageName",
]
