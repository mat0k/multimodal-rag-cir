from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from tqdm.auto import tqdm

from src.chain.types import CandidateListRecord, CandidateScore, RerankedRecord
from src.datasets.cirr import CIRR, build_cirr_dataset
from src.datasets.fashioniq import FashionIQ, build_fashioniq_dataset
from src.rerankers.base import BaseReranker, RerankCandidate, RerankQuery


class _ChainImageResolver:
	"""Resolve image paths for references and candidates from query metadata."""

	def __init__(self) -> None:
		self._cirr_image_datasets: dict[str, CIRR] = {}
		self._fashioniq_image_datasets: dict[str, FashionIQ] = {}

	def _get_cirr_images(self, split: str) -> CIRR:
		if split not in self._cirr_image_datasets:
			self._cirr_image_datasets[split] = build_cirr_dataset(
				split=split,
				mode="images",
				image_transform=None,
				caption_transform=None,
			)
		return self._cirr_image_datasets[split]

	def _get_fashioniq_images(self, split: str) -> FashionIQ:
		if split not in self._fashioniq_image_datasets:
			self._fashioniq_image_datasets[split] = build_fashioniq_dataset(
				split=split,
				mode="images",
				image_transform=None,
				caption_transform=None,
			)
		return self._fashioniq_image_datasets[split]

	def resolve_reference_image(self, record: CandidateListRecord) -> str:
		query = record.query
		if query.dataset == "cirr":
			image_dataset = self._get_cirr_images(query.split)
			relpath = image_dataset.name_to_relpath[query.reference_name]
			return str(Path(image_dataset.images_dirpath) / relpath)

		if query.dataset == "fashioniq":
			if not query.query_class:
				raise ValueError(f"FashionIQ query_class missing for query_id={query.query_id}.")
			image_dataset = self._get_fashioniq_images(query.split)
			return str(image_dataset.get_image_path(query.query_class, query.reference_name))

		raise ValueError(f"Unsupported dataset: {query.dataset}")

	def resolve_candidate_image(self, record: CandidateListRecord, candidate: CandidateScore) -> str:
		query = record.query
		if query.dataset == "cirr":
			image_dataset = self._get_cirr_images(query.split)
			relpath = image_dataset.name_to_relpath[candidate.candidate_id]
			return str(Path(image_dataset.images_dirpath) / relpath)

		if query.dataset == "fashioniq":
			query_class = query.query_class
			candidate_class = candidate.metadata.get("class") if candidate.metadata else None
			if candidate_class is not None:
				query_class = str(candidate_class)
			if not query_class:
				raise ValueError(f"FashionIQ class missing for query_id={query.query_id}.")
			image_dataset = self._get_fashioniq_images(query.split)
			return str(image_dataset.get_image_path(query_class, candidate.candidate_id))

		raise ValueError(f"Unsupported dataset: {query.dataset}")


def rerank_candidate_records(
	reranker: BaseReranker,
	records: Sequence[CandidateListRecord],
	rerank_top_n: int,
	use_tqdm: bool = False,
	fuse_with_retrieval_scores: bool = False,
	retrieval_score_weight: float = 0.0,
	reranker_score_weight: float = 1.0,
) -> tuple[list[RerankedRecord], dict[str, float]]:
	"""Rerank Stage-A candidates and return Stage-B records plus latency stats."""
	if rerank_top_n <= 0:
		raise ValueError("rerank_top_n must be > 0.")

	resolver = _ChainImageResolver()
	ranked_records: list[RerankedRecord] = []

	total_scored_pairs = 0
	total_source_candidates = 0
	query_latencies: list[float] = []
	start_time = time.perf_counter()

	iterable = records
	if use_tqdm:
		iterable = tqdm(records, desc="Stage-B reranking")

	for record in iterable:
		query_start = time.perf_counter()
		query = record.query
		source_candidates = list(record.candidates[:rerank_top_n])
		total_source_candidates += len(record.candidates)

		target_in_source_top_n: bool | None = None
		target_rank_in_source_top_n: int | None = None
		if query.target_name is not None:
			source_ids = [item.candidate_id for item in source_candidates]
			target_in_source_top_n = query.target_name in source_ids
			if target_in_source_top_n:
				target_rank_in_source_top_n = source_ids.index(query.target_name) + 1

		reference_image = resolver.resolve_reference_image(record)
		rerank_query = RerankQuery(
			reference_image=reference_image,
			text_edit=query.text_edit,
			metadata={
				"query_id": query.query_id,
				"dataset": query.dataset,
				"split": query.split,
				"reference_name": query.reference_name,
				"target_name": query.target_name,
				"query_class": query.query_class,
				"pair_id": query.pair_id,
			},
		)

		candidate_items: list[RerankCandidate] = []
		retriever_scores: list[float] = []
		for candidate in source_candidates:
			candidate_path = resolver.resolve_candidate_image(record, candidate)
			candidate_items.append(
				RerankCandidate(
					image=candidate_path,
					candidate_id=candidate.candidate_id,
					metadata={
						"image_path": candidate_path,
						**dict(candidate.metadata),
					},
				)
			)
			retriever_scores.append(float(candidate.score))

		reranker_scores: list[float] = []
		if candidate_items:
			reranker_scores = reranker.score_batch(query=rerank_query, candidates=candidate_items)
			if len(reranker_scores) != len(candidate_items):
				raise ValueError(
					f"score_batch returned {len(reranker_scores)} scores for {len(candidate_items)} candidates."
				)

		combined_scores: list[float] = []
		for idx, reranker_score in enumerate(reranker_scores):
			if fuse_with_retrieval_scores:
				combined = retrieval_score_weight * retriever_scores[idx] + reranker_score_weight * reranker_score
			else:
				combined = reranker_score
			combined_scores.append(float(combined))

		sorted_positions = sorted(
			range(len(candidate_items)),
			key=lambda pos: combined_scores[pos],
			reverse=True,
		)

		reranked_candidates: list[CandidateScore] = []
		for new_rank, source_idx in enumerate(sorted_positions, start=1):
			source_candidate = source_candidates[source_idx]
			reranked_candidates.append(
				CandidateScore(
					candidate_id=source_candidate.candidate_id,
					score=combined_scores[source_idx],
					rank=new_rank,
					metadata={
						**dict(source_candidate.metadata),
						"retriever_score": float(retriever_scores[source_idx]),
						"reranker_score": float(reranker_scores[source_idx]),
						"source_rank": source_candidate.rank,
					},
				)
			)

		target_in_rerank: bool | None = None
		target_rank_in_rerank: int | None = None
		if query.target_name is not None:
			target_ids = [item.candidate_id for item in reranked_candidates]
			target_in_rerank = query.target_name in target_ids
			if target_in_rerank:
				target_rank_in_rerank = target_ids.index(query.target_name) + 1

		query_elapsed = time.perf_counter() - query_start
		query_latencies.append(float(query_elapsed))
		total_scored_pairs += len(candidate_items)

		ranked_records.append(
			RerankedRecord(
				query=query,
				candidates=reranked_candidates,
				rerank_top_n=rerank_top_n,
				target_in_rerank_top_n=target_in_rerank,
				target_rank_in_rerank=target_rank_in_rerank,
				metadata={
					"query_latency_seconds": float(query_elapsed),
					"scored_pairs": float(len(candidate_items)),
					"source_candidate_count": float(len(record.candidates)),
					"source_rerank_pool_size": float(len(source_candidates)),
					"target_in_source_top_m": record.target_in_top_m,
					"target_rank_in_source_top_m": record.target_rank_in_top_m,
					"target_in_source_top_n": target_in_source_top_n,
					"target_rank_in_source_top_n": target_rank_in_source_top_n,
					"retriever_top_m": float(record.retriever_top_m),
					"fuse_with_retrieval_scores": bool(fuse_with_retrieval_scores),
					"retrieval_score_weight": float(retrieval_score_weight),
					"reranker_score_weight": float(reranker_score_weight),
				},
			)
		)

	elapsed_seconds = time.perf_counter() - start_time
	num_queries = len(ranked_records)

	latency_stats = {
		"num_queries": float(num_queries),
		"scored_pairs": float(total_scored_pairs),
		"latency_seconds": float(elapsed_seconds),
		"latency_seconds_per_query": float(elapsed_seconds / max(1, num_queries)),
		"latency_seconds_per_scored_pair": float(elapsed_seconds / max(1, total_scored_pairs)),
		"avg_query_latency_seconds": float(sum(query_latencies) / max(1, len(query_latencies))),
		"avg_source_candidate_pool_size": float(total_source_candidates / max(1, num_queries)),
		"avg_rerank_candidate_pool_size": float(total_scored_pairs / max(1, num_queries)),
		"rerank_top_n": float(rerank_top_n),
		"fuse_with_retrieval_scores": float(1.0 if fuse_with_retrieval_scores else 0.0),
		"retrieval_score_weight": float(retrieval_score_weight),
		"reranker_score_weight": float(reranker_score_weight),
	}

	return ranked_records, latency_stats


__all__ = ["rerank_candidate_records"]
