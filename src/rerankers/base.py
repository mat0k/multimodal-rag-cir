from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class RerankQuery:
	"""Generic query container for reranking.

	CIR-compatible required fields:
	- reference_image: the reference image item/tensor/path for the query
	- text_edit: the relative text modification for the query

	metadata can carry dataset-specific fields (e.g., pair_id, class label).
	"""

	reference_image: Any
	text_edit: str
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankCandidate:
	"""Single candidate item to be scored against a query."""

	image: Any
	candidate_id: str | None = None
	metadata: dict[str, Any] = field(default_factory=dict)


class BaseReranker(ABC):
	"""Common interface for rerankers used in standalone and two-stage pipelines.

	Implementations must return continuous scores where higher means more relevant.
	"""

	@abstractmethod
	def score(self, query: RerankQuery, candidate: RerankCandidate) -> float:
		"""Score a single candidate for a given query."""
		raise NotImplementedError

	@abstractmethod
	def score_batch(
		self,
		query: RerankQuery,
		candidates: Sequence[RerankCandidate],
	) -> list[float]:
		"""Score multiple candidates for a given query.

		The returned list must align with the input candidate order.
		"""
		raise NotImplementedError

