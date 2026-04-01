from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from tqdm.auto import tqdm

from src.datasets.cirr import CIRR, build_cirr_dataset
from src.rerankers.base import BaseReranker, RerankCandidate, RerankQuery


@dataclass(frozen=True)
class CIRRStandaloneQuery:
	pair_id: int | str
	reference_name: str
	target_name: str
	text_edit: str
	reference_image: object
	group_members: list[str]


CandidatePoolBuilder = Callable[[CIRRStandaloneQuery, list[str], random.Random], list[str]]


def _resolve_k_values(k_values: Sequence[int]) -> tuple[int, ...]:
	# Step-5 requirement: always report R@1 and R@5, and include R@10 when pool size allows.
	resolved = set(int(k) for k in k_values)
	resolved.update({1, 5, 10})
	return tuple(sorted(resolved))


def build_sampled_candidate_pool(
	query: CIRRStandaloneQuery,
	all_candidate_ids: list[str],
	rng: random.Random,
	num_random_distractors: int = 31,
	include_target: bool = True,
	remove_reference: bool = True,
) -> list[str]:
	"""Create a temporary pool with target + sampled distractors.

	This mirrors the intended Phase-2 standalone setup and is intentionally
	independent from any retriever top-k source.
	"""
	excluded = set()
	if remove_reference:
		excluded.add(query.reference_name)

	available = [candidate_id for candidate_id in all_candidate_ids if candidate_id not in excluded]
	if include_target and query.target_name not in available:
		raise ValueError(f"Target '{query.target_name}' is unavailable after filtering.")

	distractor_pool = [candidate_id for candidate_id in available if candidate_id != query.target_name]
	num_to_sample = min(num_random_distractors, len(distractor_pool))
	sampled = rng.sample(distractor_pool, k=num_to_sample) if num_to_sample > 0 else []

	if include_target:
		sampled.append(query.target_name)

	# Keep deterministic ordering policy under fixed seed while avoiding bias from always-last target.
	rng.shuffle(sampled)
	return sampled


def rerank_query_candidates(
	reranker: BaseReranker,
	query: CIRRStandaloneQuery,
	candidate_ids: Sequence[str],
	image_dataset: CIRR,
) -> list[str]:
	"""Score candidate pool and return candidate ids sorted by descending score."""
	if not candidate_ids:
		return []

	rerank_query = RerankQuery(
		reference_image=query.reference_image,
		text_edit=query.text_edit,
		metadata={
			"pair_id": query.pair_id,
			"reference_name": query.reference_name,
			"target_name": query.target_name,
		},
	)

	candidate_items: list[RerankCandidate] = []
	for candidate_id in candidate_ids:
		relpath = image_dataset.name_to_relpath[candidate_id]
		image_path = str(Path(image_dataset.images_dirpath) / relpath)
		candidate_items.append(
			RerankCandidate(
				image=image_path,
				candidate_id=candidate_id,
				metadata={"image_path": image_path},
			)
		)

	scores: list[float] = []
	for idx, candidate in enumerate(candidate_items):
		if idx == 0:
			print("DEBUG_CIRR_EVAL: before first single score call", flush=True)
		if idx == 1:
			print("DEBUG_CIRR_EVAL: before second single score call", flush=True)
		score = reranker.score(query=rerank_query, candidate=candidate)
		scores.append(score)
		if idx == 0:
			print(f"DEBUG_CIRR_EVAL: after first single score returns score={score}", flush=True)
		if idx == 1:
			print(f"DEBUG_CIRR_EVAL: after second single score returns score={score}", flush=True)
	ranked_pairs = sorted(zip(candidate_ids, scores), key=lambda item: item[1], reverse=True)
	return [candidate_id for candidate_id, _ in ranked_pairs]


def evaluate_cirr_standalone_rerank(
	reranker: BaseReranker,
	split: str = "val",
	num_random_distractors: int = 31,
	max_queries: int | None = None,
	seed: int = 42,
	k_values: Sequence[int] = (1, 5, 10),
	use_tqdm: bool = False,
	triplet_dataset: CIRR | None = None,
	image_dataset: CIRR | None = None,
	candidate_pool_builder: CandidatePoolBuilder | None = None,
) -> dict[str, float]:
	"""Evaluate standalone LamRA reranking on CIRR candidate pools.

	Candidate source is modular and can later be replaced by retriever top-k without
	changing the scoring/ranking path.
	"""
	if split == "test1":
		raise ValueError("Standalone rerank evaluation requires a split with targets. Use split='val' or 'train'.")

	print("DEBUG_CIRR_EVAL: entering evaluate_cirr_standalone_rerank", flush=True)

	if image_dataset is None:
		print("DEBUG_CIRR_EVAL: creating/loading image dataset", flush=True)
		image_dataset = build_cirr_dataset(split=split, mode="images", image_transform=None, caption_transform=None)
	if triplet_dataset is None:
		print("DEBUG_CIRR_EVAL: creating/loading triplet dataset", flush=True)
		triplet_dataset = build_cirr_dataset(split=split, mode="triplets", image_transform=None, caption_transform=None)

	print(f"DEBUG_CIRR_EVAL: triplet dataset length = {len(triplet_dataset)}", flush=True)
	print(f"DEBUG_CIRR_EVAL: image dataset length = {len(image_dataset)}", flush=True)

	k_values = _resolve_k_values(k_values)

	all_candidate_ids = list(image_dataset.name_to_relpath.keys())
	rng = random.Random(seed)
	pool_builder = candidate_pool_builder or (
		lambda query, ids, random_state: build_sampled_candidate_pool(
			query=query,
			all_candidate_ids=ids,
			rng=random_state,
			num_random_distractors=num_random_distractors,
			include_target=True,
			remove_reference=True,
		)
	)

	query_count = len(triplet_dataset)
	if max_queries is not None:
		query_count = min(query_count, max_queries)

	hits = {k: 0 for k in k_values}
	eligible = {k: 0 for k in k_values}
	pool_sizes: list[int] = []

	indices = range(query_count)
	if use_tqdm:
		indices = tqdm(indices, desc="Evaluating CIRR standalone reranker")

	print("DEBUG_CIRR_EVAL: starting query loop", flush=True)

	for idx in indices:
		if idx == 0:
			print("DEBUG_CIRR_EVAL: fetching first query", flush=True)
		sample = triplet_dataset[idx]
		query = CIRRStandaloneQuery(
			pair_id=sample["pair_id"],
			reference_name=sample["reference_name"],
			target_name=sample["target_name"],
			text_edit=sample["caption"],
			reference_image=sample["reference"],
			group_members=sample["group_members"],
		)

		if idx == 0:
			print(
				f"DEBUG_CIRR_EVAL: first query id / target id = {query.pair_id} / {query.target_name}",
				flush=True,
			)

		if idx == 0:
			print("DEBUG_CIRR_EVAL: building candidate pool", flush=True)
		candidate_ids = pool_builder(query, all_candidate_ids, rng)
		if idx == 0:
			print(f"DEBUG_CIRR_EVAL: candidate pool size = {len(candidate_ids)}", flush=True)
		if query.target_name not in candidate_ids:
			raise ValueError(f"Target '{query.target_name}' missing from candidate pool for pair_id={query.pair_id}.")

		if idx == 0:
			print("DEBUG_CIRR_EVAL: entering single-item scoring path", flush=True)
		ranked_ids = rerank_query_candidates(
			reranker=reranker,
			query=query,
			candidate_ids=candidate_ids,
			image_dataset=image_dataset,
		)
		if idx == 0:
			print("DEBUG_CIRR_EVAL: after first scoring call returns", flush=True)

		pool_sizes.append(len(ranked_ids))
		for k in k_values:
			if len(ranked_ids) < k:
				continue
			eligible[k] += 1
			if query.target_name in ranked_ids[:k]:
				hits[k] += 1

	metrics: dict[str, float] = {
		"num_queries": float(query_count),
		"avg_candidate_pool_size": float(sum(pool_sizes) / max(1, len(pool_sizes))),
		"num_random_distractors": float(num_random_distractors),
	}
	print("DEBUG_CIRR_EVAL: before metrics aggregation", flush=True)

	for k in k_values:
		metric_name = f"recall_at{k}"
		if eligible[k] == 0:
			metrics[metric_name] = float("nan")
		else:
			metrics[metric_name] = float((hits[k] / eligible[k]) * 100.0)

	print("DEBUG_CIRR_EVAL: before function return", flush=True)

	return metrics


__all__ = [
	"CIRRStandaloneQuery",
	"CandidatePoolBuilder",
	"build_sampled_candidate_pool",
	"evaluate_cirr_standalone_rerank",
	"rerank_query_candidates",
]

