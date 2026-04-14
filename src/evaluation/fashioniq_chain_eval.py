from __future__ import annotations

from typing import Mapping, Sequence

from src.chain.types import RerankedRecord


def evaluate_fashioniq_chain_records(
	records: Sequence[RerankedRecord],
	metric_prefix: str = "val",
	k_values: Sequence[int] = (1, 5, 10, 15),
	latency_stats: Mapping[str, float] | None = None,
) -> dict[str, float]:
	"""Evaluate FashionIQ chain reranking records with class-level metrics."""
	classes = sorted({str(record.query.query_class) for record in records if record.query.query_class})
	if not classes:
		classes = ["dress", "shirt", "toptee"]

	hits = {cls: {int(k): 0 for k in k_values} for cls in classes}
	retrieval_hits = {cls: {int(k): 0 for k in k_values} for cls in classes}
	eligible = {cls: {int(k): 0 for k in k_values} for cls in classes}
	class_queries = {cls: 0 for cls in classes}
	class_coverage_hits = {cls: 0 for cls in classes}
	class_latency_seconds = {cls: 0.0 for cls in classes}
	class_scored_pairs = {cls: 0.0 for cls in classes}

	num_queries = len(records)
	num_queries_with_target = 0

	total_source_pool = 0.0
	total_rerank_pool = 0.0

	for record in records:
		query = record.query
		query_class = str(query.query_class) if query.query_class else "unknown"
		if query_class not in class_queries:
			class_queries[query_class] = 0
			class_coverage_hits[query_class] = 0
			class_latency_seconds[query_class] = 0.0
			class_scored_pairs[query_class] = 0.0
			hits[query_class] = {int(k): 0 for k in k_values}
			retrieval_hits[query_class] = {int(k): 0 for k in k_values}
			eligible[query_class] = {int(k): 0 for k in k_values}

		total_source_pool += float(record.metadata.get("source_candidate_count", len(record.candidates)))
		total_rerank_pool += float(len(record.candidates))

		class_latency_seconds[query_class] += float(record.metadata.get("query_latency_seconds", 0.0))
		class_scored_pairs[query_class] += float(record.metadata.get("scored_pairs", len(record.candidates)))

		target_name = query.target_name
		if target_name is None:
			continue

		num_queries_with_target += 1
		class_queries[query_class] += 1
		if bool(record.metadata.get("target_in_source_top_m", False)):
			class_coverage_hits[query_class] += 1

		ranked_ids = [item.candidate_id for item in record.candidates]
		source_rank = record.metadata.get("target_rank_in_source_top_n")
		source_rank_int = int(source_rank) if source_rank is not None else None
		for k in k_values:
			k_int = int(k)
			if len(ranked_ids) < k_int:
				continue
			eligible[query_class][k_int] += 1
			if source_rank_int is not None and source_rank_int <= k_int:
				retrieval_hits[query_class][k_int] += 1
			if target_name in ranked_ids[:k_int]:
				hits[query_class][k_int] += 1

	metrics: dict[str, float] = {
		"num_queries": float(num_queries),
		"num_queries_with_target": float(num_queries_with_target),
		"avg_source_candidate_pool_size": float(total_source_pool / max(1, num_queries)),
		"avg_rerank_candidate_pool_size": float(total_rerank_pool / max(1, num_queries)),
	}

	coverage_values: list[float] = []
	for cls in sorted(class_queries.keys()):
		class_count = class_queries[cls]
		coverage = float((class_coverage_hits[cls] / max(1, class_count)) * 100.0)
		metrics[f"{metric_prefix}_{cls}_target_coverage_at_m"] = coverage
		metrics[f"{metric_prefix}_{cls}_latency_seconds"] = float(class_latency_seconds[cls])
		metrics[f"{metric_prefix}_{cls}_latency_seconds_per_query"] = float(
			class_latency_seconds[cls] / max(1, class_count)
		)
		metrics[f"{metric_prefix}_{cls}_scored_pairs"] = float(class_scored_pairs[cls])
		coverage_values.append(coverage)

		for k in k_values:
			k_int = int(k)
			retrieval_key = f"{metric_prefix}_{cls}_retrieval_recall_at{k_int}"
			key = f"{metric_prefix}_{cls}_chain_recall_at{k_int}"
			if eligible[cls][k_int] == 0:
				metrics[retrieval_key] = float("nan")
				metrics[key] = float("nan")
			else:
				metrics[retrieval_key] = float((retrieval_hits[cls][k_int] / eligible[cls][k_int]) * 100.0)
				metrics[key] = float((hits[cls][k_int] / eligible[cls][k_int]) * 100.0)

	metrics[f"{metric_prefix}_avg_target_coverage_at_m"] = float(sum(coverage_values) / max(1, len(coverage_values)))

	for k in k_values:
		k_int = int(k)
		class_retrieval_values = [
			metrics[f"{metric_prefix}_{cls}_retrieval_recall_at{k_int}"]
			for cls in sorted(class_queries.keys())
			if f"{metric_prefix}_{cls}_retrieval_recall_at{k_int}" in metrics
		]
		class_metric_values = [
			metrics[f"{metric_prefix}_{cls}_chain_recall_at{k_int}"]
			for cls in sorted(class_queries.keys())
			if f"{metric_prefix}_{cls}_chain_recall_at{k_int}" in metrics
		]
		valid_retrieval = [value for value in class_retrieval_values if value == value]
		valid = [value for value in class_metric_values if value == value]
		metrics[f"{metric_prefix}_avg_retrieval_recall_at{k_int}"] = float(
			sum(valid_retrieval) / max(1, len(valid_retrieval))
		)
		metrics[f"{metric_prefix}_avg_chain_recall_at{k_int}"] = float(sum(valid) / max(1, len(valid)))

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


__all__ = ["evaluate_fashioniq_chain_records"]
