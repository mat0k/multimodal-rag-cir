from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from tqdm.auto import tqdm

from src.datasets.fashioniq import FashionIQ, build_fashioniq_dataset
from src.rerankers.base import BaseReranker, RerankCandidate, RerankQuery


@dataclass(frozen=True)
class FashionIQStandaloneQuery:
	query_class: str
	reference_name: str
	target_name: str
	text_edit: str
	reference_image: object


CandidatePoolBuilder = Callable[[FashionIQStandaloneQuery, dict[str, list[str]], random.Random], list[str]]


def _resolve_k_values(k_values: Sequence[int]) -> tuple[int, ...]:
	# Step-5 requirement: always report R@1 and R@5, and include R@10 when pool size allows.
	resolved = set(int(k) for k in k_values)
	resolved.update({1, 5, 10})
	return tuple(sorted(resolved))


def build_sampled_candidate_pool(
	query: FashionIQStandaloneQuery,
	class_to_candidate_ids: dict[str, list[str]],
	rng: random.Random,
	num_random_distractors: int = 31,
	include_target: bool = True,
	remove_reference: bool = True,
) -> list[str]:
	"""Create a temporary class-specific pool with target + sampled distractors."""
	class_candidates = list(class_to_candidate_ids.get(query.query_class, []))
	if not class_candidates:
		raise ValueError(f"No candidate gallery found for class '{query.query_class}'.")

	excluded = set()
	if remove_reference:
		excluded.add(query.reference_name)

	available = [candidate_id for candidate_id in class_candidates if candidate_id not in excluded]
	if include_target and query.target_name not in available:
		raise ValueError(
			f"Target '{query.target_name}' is unavailable for class '{query.query_class}' after filtering."
		)

	distractor_pool = [candidate_id for candidate_id in available if candidate_id != query.target_name]
	num_to_sample = min(num_random_distractors, len(distractor_pool))
	sampled = rng.sample(distractor_pool, k=num_to_sample) if num_to_sample > 0 else []

	if include_target:
		sampled.append(query.target_name)

	rng.shuffle(sampled)
	return sampled


def rerank_query_candidates(
	reranker: BaseReranker,
	query: FashionIQStandaloneQuery,
	candidate_ids: Sequence[str],
	image_dataset: FashionIQ,
) -> list[str]:
	"""Score candidate pool and return candidate ids sorted by descending score."""
	if not candidate_ids:
		return []

	rerank_query = RerankQuery(
		reference_image=query.reference_image,
		text_edit=query.text_edit,
		metadata={
			"class": query.query_class,
			"reference_name": query.reference_name,
			"target_name": query.target_name,
		},
	)

	candidate_items: list[RerankCandidate] = []
	for candidate_id in candidate_ids:
		image_path = str(Path(image_dataset.get_image_path(query.query_class, candidate_id)))
		candidate_items.append(
			RerankCandidate(
				image=image_path,
				candidate_id=candidate_id,
				metadata={"class": query.query_class, "image_path": image_path},
			)
		)

	scores = reranker.score_batch(query=rerank_query, candidates=candidate_items)
	ranked_pairs = sorted(zip(candidate_ids, scores), key=lambda item: item[1], reverse=True)
	return [candidate_id for candidate_id, _ in ranked_pairs]


def evaluate_fashioniq_standalone_rerank(
	reranker: BaseReranker,
	split: str = "val",
	caption_joiner: str = " ",
	metric_prefix: str = "val",
	num_random_distractors: int = 31,
	max_queries: int | None = None,
	seed: int = 42,
	k_values: Sequence[int] = (1, 5, 10),
	use_tqdm: bool = False,
	triplet_dataset: FashionIQ | None = None,
	image_dataset: FashionIQ | None = None,
	candidate_pool_builder: CandidatePoolBuilder | None = None,
) -> dict[str, float]:
	"""Evaluate standalone LamRA reranking on FashionIQ candidate pools.

	Candidate source is modular and can later be replaced by retriever top-k without
	changing the scoring/ranking path.
	"""
	if split == "test":
		raise ValueError("Standalone rerank evaluation requires a split with targets. Use split='val' or 'train'.")

	eval_start_time = time.perf_counter()

	if image_dataset is None:
		image_dataset = build_fashioniq_dataset(
			split=split,
			mode="images",
			image_transform=None,
			caption_transform=None,
			caption_joiner=caption_joiner,
		)
	if triplet_dataset is None:
		triplet_dataset = build_fashioniq_dataset(
			split=split,
			mode="triplets",
			image_transform=None,
			caption_transform=None,
			caption_joiner=caption_joiner,
		)

	k_values = _resolve_k_values(k_values)

	class_to_candidate_ids = {cls: list(image_dataset.images[cls]) for cls in image_dataset.classes}
	rng = random.Random(seed)

	pool_builder = candidate_pool_builder or (
		lambda query, ids_by_class, random_state: build_sampled_candidate_pool(
			query=query,
			class_to_candidate_ids=ids_by_class,
			rng=random_state,
			num_random_distractors=num_random_distractors,
			include_target=True,
			remove_reference=True,
		)
	)

	query_count = len(triplet_dataset)
	if max_queries is not None:
		query_count = min(query_count, max_queries)

	hits: dict[str, dict[int, int]] = {cls: {k: 0 for k in k_values} for cls in image_dataset.classes}
	eligible: dict[str, dict[int, int]] = {cls: {k: 0 for k in k_values} for cls in image_dataset.classes}
	pool_sizes: list[int] = []
	class_query_counts: dict[str, int] = {cls: 0 for cls in image_dataset.classes}
	class_scored_pairs: dict[str, int] = {cls: 0 for cls in image_dataset.classes}
	class_latency_seconds: dict[str, float] = {cls: 0.0 for cls in image_dataset.classes}
	total_scored_pairs = 0

	indices = range(query_count)
	if use_tqdm:
		indices = tqdm(indices, desc="Evaluating FashionIQ standalone reranker")

	for idx in indices:
		query_start_time = time.perf_counter()
		sample = triplet_dataset[idx]
		target_name = sample.get("target_name")
		if not target_name:
			raise ValueError(
				"FashionIQ standalone rerank evaluation requires non-empty target_name for every query."
			)

		query = FashionIQStandaloneQuery(
			query_class=sample["class"],
			reference_name=sample["candidate_name"],
			target_name=target_name,
			text_edit=sample["transformed_caption"] if isinstance(sample["transformed_caption"], str) else "",
			reference_image=sample["candidate"],
		)

		if not query.text_edit:
			# image_transform/caption_transform are intentionally None in standalone mode,
			# so transformed_caption is expected to be raw caption string.
			raw_triplet = triplet_dataset.annotations[query.query_class][triplet_dataset.get_class_index(idx)[1]]
			query = FashionIQStandaloneQuery(
				query_class=query.query_class,
				reference_name=query.reference_name,
				target_name=query.target_name,
				text_edit=caption_joiner.join(raw_triplet["captions"]),
				reference_image=query.reference_image,
			)

		candidate_ids = pool_builder(query, class_to_candidate_ids, rng)
		if query.target_name not in candidate_ids:
			raise ValueError(
				f"Target '{query.target_name}' missing from candidate pool for class '{query.query_class}'."
			)

		ranked_ids = rerank_query_candidates(
			reranker=reranker,
			query=query,
			candidate_ids=candidate_ids,
			image_dataset=image_dataset,
		)
		query_elapsed = time.perf_counter() - query_start_time

		pool_sizes.append(len(ranked_ids))
		total_scored_pairs += len(ranked_ids)
		class_query_counts[query.query_class] += 1
		class_scored_pairs[query.query_class] += len(ranked_ids)
		class_latency_seconds[query.query_class] += query_elapsed
		for k in k_values:
			if len(ranked_ids) < k:
				continue
			eligible[query.query_class][k] += 1
			if query.target_name in ranked_ids[:k]:
				hits[query.query_class][k] += 1

	elapsed_seconds = time.perf_counter() - eval_start_time
	scored_queries = len(pool_sizes)

	metrics: dict[str, float] = {
		"num_queries": float(query_count),
		"avg_candidate_pool_size": float(sum(pool_sizes) / max(1, len(pool_sizes))),
		"num_random_distractors": float(num_random_distractors),
		"scored_pairs": float(total_scored_pairs),
		"latency_seconds": float(elapsed_seconds),
		"latency_seconds_per_query": float(elapsed_seconds / max(1, scored_queries)),
		"latency_seconds_per_scored_pair": float(elapsed_seconds / max(1, total_scored_pairs)),
	}

	for cls in image_dataset.classes:
		metrics[f"{metric_prefix}_{cls}_latency_seconds"] = float(class_latency_seconds[cls])
		metrics[f"{metric_prefix}_{cls}_latency_seconds_per_query"] = float(
			class_latency_seconds[cls] / max(1, class_query_counts[cls])
		)
		metrics[f"{metric_prefix}_{cls}_scored_pairs"] = float(class_scored_pairs[cls])
		for k in k_values:
			metric_name = f"{metric_prefix}_{cls}_recall_at{k}"
			if eligible[cls][k] == 0:
				metrics[metric_name] = float("nan")
			else:
				metrics[metric_name] = float((hits[cls][k] / eligible[cls][k]) * 100.0)

	for k in k_values:
		class_metric_names = [f"{metric_prefix}_{cls}_recall_at{k}" for cls in image_dataset.classes]
		valid_values = [metrics[name] for name in class_metric_names if metrics[name] == metrics[name]]
		metrics[f"{metric_prefix}_avg_recall_at{k}"] = float(sum(valid_values) / max(1, len(valid_values)))

	return metrics


__all__ = [
	"CandidatePoolBuilder",
	"FashionIQStandaloneQuery",
	"build_sampled_candidate_pool",
	"evaluate_fashioniq_standalone_rerank",
	"rerank_query_candidates",
]

