from __future__ import annotations

from typing import Mapping, Sequence

from src.chain.types import RerankedRecord


def evaluate_cirr_chain_records(
	records: Sequence[RerankedRecord],
	k_values: Sequence[int] = (1, 5, 10, 50),
	latency_stats: Mapping[str, float] | None = None,
) -> dict[str, float]:
	"""Evaluate CIRR chain reranking records on recall and coverage metrics."""
	hits = {int(k): 0 for k in k_values}
	eligible = {int(k): 0 for k in k_values}

	num_queries = len(records)
	num_queries_with_target = 0
	coverage_hits = 0

	total_source_pool = 0.0
	total_rerank_pool = 0.0

	for record in records:
		query = record.query
		target_name = query.target_name

		total_source_pool += float(record.metadata.get("source_candidate_count", len(record.candidates)))
		total_rerank_pool += float(len(record.candidates))

		if target_name is None:
			continue
		num_queries_with_target += 1

		if bool(record.metadata.get("target_in_source_top_m", False)):
			coverage_hits += 1

		ranked_ids = [item.candidate_id for item in record.candidates]
		for k in k_values:
			k_int = int(k)
			if len(ranked_ids) < k_int:
				continue
			eligible[k_int] += 1
			if target_name in ranked_ids[:k_int]:
				hits[k_int] += 1

	metrics: dict[str, float] = {
		"num_queries": float(num_queries),
		"num_queries_with_target": float(num_queries_with_target),
		"avg_source_candidate_pool_size": float(total_source_pool / max(1, num_queries)),
		"avg_rerank_candidate_pool_size": float(total_rerank_pool / max(1, num_queries)),
		"target_coverage_at_m": float((coverage_hits / max(1, num_queries_with_target)) * 100.0),
	}

	for k in k_values:
		k_int = int(k)
		name = f"chain_recall_at{k_int}"
		if eligible[k_int] == 0:
			metrics[name] = float("nan")
		else:
			metrics[name] = float((hits[k_int] / eligible[k_int]) * 100.0)

	if latency_stats is not None:
		for key in (
			"scored_pairs",
			"latency_seconds",
			"latency_seconds_per_query",
			"latency_seconds_per_scored_pair",
		):
			if key in latency_stats:
				metrics[key] = float(latency_stats[key])

	return metrics


__all__ = ["evaluate_cirr_chain_records"]
