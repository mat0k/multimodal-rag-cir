from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.chain.types import CandidateListRecord, CandidateScore, QueryRecord
from src.datasets.fashioniq import build_fashioniq_dataset
from src.evaluation.fashioniq_eval import (
	generate_fashioniq_index_features,
	generate_fashioniq_predicted_features,
	resolve_caption_joiner,
	resolve_caption_order_mode,
	resolve_fashioniq_eval_protocol,
)
from src.retrievers.base import TwoEncoderVLM
from src.utils.tensor import make_normalized


def _join_captions(captions: list[str], joiner: str, reverse_order: bool = False) -> str:
	ordered = list(reversed(captions)) if reverse_order else list(captions)
	return joiner.join(ordered)


def build_fashioniq_candidate_records(
	model: TwoEncoderVLM,
	top_m: int = 100,
	query_embedding_mode: str = "vista_mm",
	fusion_type: str = "sum",
	batch_size: int = 64,
	num_workers: int = 4,
	use_tqdm: bool = False,
	max_queries: int | None = None,
	eval_protocol: str = "val_split",
	caption_joiner_override: Optional[str] = None,
	caption_order_mode: str = "original",
) -> tuple[list[CandidateListRecord], dict[str, float]]:
	"""Build class-specific top-M retriever candidate lists for FashionIQ."""
	if top_m <= 0:
		raise ValueError("top_m must be > 0.")

	protocol_cfg = resolve_fashioniq_eval_protocol(eval_protocol)
	split = str(protocol_cfg["split"])
	caption_joiner = resolve_caption_joiner(
		protocol_caption_joiner=str(protocol_cfg["caption_joiner"]),
		caption_joiner_override=caption_joiner_override,
	)
	resolved_caption_order_mode = resolve_caption_order_mode(caption_order_mode)

	index_dataset = build_fashioniq_dataset(
		split=split,
		mode="images",
		image_transform=model.image_processor,
		caption_transform=model.tokenizer,
		max_length_tokenizer=77,
		caption_joiner=caption_joiner,
	)
	triplet_dataset = build_fashioniq_dataset(
		split=split,
		mode="triplets",
		image_transform=model.image_processor,
		caption_transform=model.tokenizer,
		max_length_tokenizer=77,
		caption_joiner=caption_joiner,
		reverse_caption_order=False,
	)

	index_features, index_names, index_classes = generate_fashioniq_index_features(
		clip_model=model,
		index_dataset=index_dataset,
		batch_size=batch_size,
		num_workers=num_workers,
		use_tqdm=use_tqdm,
		accelerator=None,
	)

	predicted_features, reference_names, target_names, triplet_classes = generate_fashioniq_predicted_features(
		clip_model=model,
		triplet_dataset=triplet_dataset,
		query_embedding_mode=query_embedding_mode,
		fusion_type=fusion_type,
		batch_size=batch_size,
		num_workers=num_workers,
		use_tqdm=use_tqdm,
		accelerator=None,
		skip_targets=False,
	)

	if resolved_caption_order_mode == "both":
		triplet_dataset_reversed = build_fashioniq_dataset(
			split=split,
			mode="triplets",
			image_transform=model.image_processor,
			caption_transform=model.tokenizer,
			max_length_tokenizer=77,
			caption_joiner=caption_joiner,
			reverse_caption_order=True,
		)
		reversed_features, rev_reference_names, rev_target_names, rev_triplet_classes = (
			generate_fashioniq_predicted_features(
				clip_model=model,
				triplet_dataset=triplet_dataset_reversed,
				query_embedding_mode=query_embedding_mode,
				fusion_type=fusion_type,
				batch_size=batch_size,
				num_workers=num_workers,
				use_tqdm=use_tqdm,
				accelerator=None,
				skip_targets=False,
			)
		)

		if (
			rev_reference_names != reference_names
			or rev_target_names != target_names
			or rev_triplet_classes != triplet_classes
		):
			raise ValueError("FashionIQ caption-order averaging failed due to query order mismatch.")

		predicted_features = make_normalized((predicted_features + reversed_features) / 2.0)

	total_queries = len(reference_names)
	if max_queries is not None:
		total_queries = min(total_queries, max_queries)

	predicted_features = predicted_features[:total_queries]
	reference_names = reference_names[:total_queries]
	target_names = target_names[:total_queries]
	triplet_classes = triplet_classes[:total_queries]

	class_to_index_positions: dict[str, list[int]] = {cls: [] for cls in index_dataset.classes}
	for pos, cls in enumerate(index_classes):
		class_to_index_positions[str(cls)].append(pos)

	records: list[CandidateListRecord] = []
	covered = 0
	queries_with_target = 0
	candidate_counts: list[int] = []

	for idx in range(total_queries):
		query_class = str(triplet_classes[idx])
		positions = class_to_index_positions.get(query_class, [])
		if not positions:
			raise ValueError(f"No index candidates found for class '{query_class}'.")

		cls_index_features = index_features[positions]
		cls_index_names = [str(index_names[pos]) for pos in positions]

		scores = torch.matmul(predicted_features[idx], cls_index_features.T).detach().cpu()
		sorted_idx = torch.argsort(scores, descending=True)

		ranked_names = np.array(cls_index_names, dtype=object)[sorted_idx.numpy()]
		ranked_scores = scores[sorted_idx].numpy()

		reference_name = str(reference_names[idx])
		target_raw = target_names[idx]
		target_name = str(target_raw) if target_raw is not None else None

		keep_mask = np.array([name != reference_name for name in ranked_names], dtype=bool)
		filtered_names = ranked_names[keep_mask]
		filtered_scores = ranked_scores[keep_mask]

		final_names = filtered_names[:top_m]
		final_scores = filtered_scores[:top_m]

		target_in_top_m: bool | None = None
		target_rank_in_top_m: int | None = None
		if target_name is not None:
			queries_with_target += 1
			target_positions = np.where(final_names == target_name)[0]
			target_in_top_m = len(target_positions) > 0
			target_rank_in_top_m = int(target_positions[0] + 1) if target_in_top_m else None
			if target_in_top_m:
				covered += 1

		cls_from_index, local_idx = triplet_dataset.get_class_index(idx)
		if str(cls_from_index) != query_class:
			raise ValueError(f"FashionIQ query alignment mismatch at idx={idx}.")
		annotation = triplet_dataset.annotations[query_class][local_idx]
		text_edit = _join_captions(
			captions=list(annotation["captions"]),
			joiner=caption_joiner,
			reverse_order=False,
		)

		candidates = [
			CandidateScore(
				candidate_id=str(candidate_id),
				score=float(score),
				rank=rank,
				metadata={"class": query_class},
			)
			for rank, (candidate_id, score) in enumerate(zip(final_names, final_scores), start=1)
		]

		records.append(
			CandidateListRecord(
				query=QueryRecord(
					query_id=f"fashioniq:{split}:{query_class}:{idx}",
					dataset="fashioniq",
					split=split,
					reference_name=reference_name,
					text_edit=text_edit,
					target_name=target_name,
					query_class=query_class,
					metadata={
						"eval_protocol": eval_protocol,
						"caption_order_mode": resolved_caption_order_mode,
					},
				),
				candidates=candidates,
				retriever_top_m=top_m,
				target_in_top_m=target_in_top_m,
				target_rank_in_top_m=target_rank_in_top_m,
				metadata={
					"candidate_count_after_reference_removal": int(len(filtered_names)),
					"class": query_class,
				},
			)
		)
		candidate_counts.append(len(candidates))

	stats = {
		"num_queries": float(len(records)),
		"num_queries_with_target": float(queries_with_target),
		"top_m": float(top_m),
		"avg_candidates_per_query": float(sum(candidate_counts) / max(1, len(candidate_counts))),
		"target_coverage_at_m": float((covered / max(1, queries_with_target)) * 100.0),
	}
	return records, stats


__all__ = ["build_fashioniq_candidate_records"]
