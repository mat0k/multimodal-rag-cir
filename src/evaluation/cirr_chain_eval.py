from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from src.chain.types import RerankedRecord


def _load_cirr_subset_map(annotations_path: str | Path) -> dict[int, dict]:
	"""Build pairid → {members, reference, target} from CIRR annotation file."""
	with open(annotations_path) as f:
		entries = json.load(f)
	subset_map: dict[int, dict] = {}
	for e in entries:
		subset_map[int(e["pairid"])] = {
			"members": set(e["img_set"]["members"]),
			"reference": e["reference"],
			"target": e["target_hard"],
		}
	return subset_map


def _compute_subset_recall(
	record: RerankedRecord,
	subset_map: dict[int, dict],
	subset_k_values: Sequence[int],
) -> dict[str, int] | None:
	"""Return {k: hit} for subset recall, or None if pair_id missing from map."""
	pair_id = record.query.pair_id
	if pair_id is None:
		return None
	info = subset_map.get(int(pair_id))
	if info is None:
		return None

	reference = info["reference"]
	target = info["target"]
	members = info["members"] - {reference}   # exclude reference from subset

	# Map candidate_id → reranked position (1-indexed); unseen members get inf rank
	ranked_ids = [item.candidate_id for item in record.candidates]
	rank_lookup = {cid: i + 1 for i, cid in enumerate(ranked_ids)}

	subset_ranked = sorted(members, key=lambda m: rank_lookup.get(m, float("inf")))
	return {k: int(target in subset_ranked[:k]) for k in subset_k_values}


def evaluate_cirr_chain_records(
	records: Sequence[RerankedRecord],
	k_values: Sequence[int] = (1, 5, 10, 15),
	latency_stats: Mapping[str, float] | None = None,
	cirr_annotations_path: str | Path | None = None,
	subset_k_values: Sequence[int] = (1, 2, 3),
) -> dict[str, float]:
	"""Evaluate CIRR chain reranking records on recall and coverage metrics."""
	hits = {int(k): 0 for k in k_values}
	retrieval_hits = {int(k): 0 for k in k_values}
	eligible = {int(k): 0 for k in k_values}

	num_queries = len(records)
	num_queries_with_target = 0
	coverage_hits = 0

	total_source_pool = 0.0
	total_rerank_pool = 0.0

	subset_map = _load_cirr_subset_map(cirr_annotations_path) if cirr_annotations_path is not None else None
	subset_hits = {int(k): 0 for k in subset_k_values}
	subset_eligible = 0

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
		source_rank = record.metadata.get("target_rank_in_source_top_n")
		source_rank_int = int(source_rank) if source_rank is not None else None
		for k in k_values:
			k_int = int(k)
			if len(ranked_ids) < k_int:
				continue
			eligible[k_int] += 1
			if source_rank_int is not None and source_rank_int <= k_int:
				retrieval_hits[k_int] += 1
			if target_name in ranked_ids[:k_int]:
				hits[k_int] += 1

		if subset_map is not None:
			sub = _compute_subset_recall(record, subset_map, subset_k_values)
			if sub is not None:
				subset_eligible += 1
				for k in subset_k_values:
					subset_hits[int(k)] += sub[k]

	metrics: dict[str, float] = {
		"num_queries": float(num_queries),
		"num_queries_with_target": float(num_queries_with_target),
		"avg_source_candidate_pool_size": float(total_source_pool / max(1, num_queries)),
		"avg_rerank_candidate_pool_size": float(total_rerank_pool / max(1, num_queries)),
		"target_coverage_at_m": float((coverage_hits / max(1, num_queries_with_target)) * 100.0),
	}

	for k in k_values:
		k_int = int(k)
		retrieval_name = f"retrieval_recall_at{k_int}"
		name = f"chain_recall_at{k_int}"
		if eligible[k_int] == 0:
			metrics[retrieval_name] = float("nan")
			metrics[name] = float("nan")
		else:
			metrics[retrieval_name] = float((retrieval_hits[k_int] / eligible[k_int]) * 100.0)
			metrics[name] = float((hits[k_int] / eligible[k_int]) * 100.0)

	if subset_map is not None:
		for k in subset_k_values:
			k_int = int(k)
			if subset_eligible == 0:
				metrics[f"subset_recall_at{k_int}"] = float("nan")
			else:
				metrics[f"subset_recall_at{k_int}"] = float((subset_hits[k_int] / subset_eligible) * 100.0)

		# Summary average: mean of global R@1,5,10 + subset R@1,2,3 (6 metrics)
		summary_keys = [f"chain_recall_at{k}" for k in (1, 5, 10)] + \
		               [f"subset_recall_at{k}" for k in (1, 2, 3)]
		summary_vals = [metrics[key] for key in summary_keys if key in metrics and not (metrics[key] != metrics[key])]
		if summary_vals:
			metrics["summary_average"] = float(sum(summary_vals) / len(summary_vals))

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
