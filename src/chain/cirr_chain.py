from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.chain.types import CandidateListRecord, CandidateScore, QueryRecord
from src.datasets.cirr import build_cirr_dataset
from src.evaluation.cirr_eval import generate_cirr_index_features, generate_cirr_predicted_features
from src.retrievers.base import TwoEncoderVLM


def build_cirr_candidate_records(
	model: TwoEncoderVLM,
	split: str = "val",
	top_m: int = 100,
	query_embedding_mode: str = "vista_mm",
	fusion_type: str = "sum",
	batch_size: int = 64,
	num_workers: int = 4,
	use_tqdm: bool = False,
	max_queries: int | None = None,
) -> tuple[list[CandidateListRecord], dict[str, float]]:
	"""Build top-M retriever candidate lists for CIRR queries."""
	if top_m <= 0:
		raise ValueError("top_m must be > 0.")

	index_dataset = build_cirr_dataset(
		split=split,
		mode="images",
		image_transform=model.image_processor,
		caption_transform=model.tokenizer,
		max_length_tokenizer=77,
	)
	triplet_dataset = build_cirr_dataset(
		split=split,
		mode="triplets",
		image_transform=model.image_processor,
		caption_transform=model.tokenizer,
		max_length_tokenizer=77,
	)

	index_features, index_names = generate_cirr_index_features(
		clip_model=model,
		index_dataset=index_dataset,
		batch_size=batch_size,
		num_workers=num_workers,
		use_tqdm=use_tqdm,
		accelerator=None,
	)

	predicted_features, reference_names, target_names, _, pair_ids = generate_cirr_predicted_features(
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

	total_queries = len(reference_names)
	if max_queries is not None:
		total_queries = min(total_queries, max_queries)

	predicted_features = predicted_features[:total_queries]
	reference_names = reference_names[:total_queries]
	target_names = target_names[:total_queries]
	pair_ids = pair_ids[:total_queries]

	similarities = predicted_features @ index_features.T
	similarities_cpu = similarities.detach().cpu()
	sorted_indices = torch.argsort(similarities_cpu, dim=1, descending=True)
	sorted_scores = torch.gather(similarities_cpu, 1, sorted_indices).numpy()
	sorted_names = np.array(index_names, dtype=object)[sorted_indices.numpy()]

	records: list[CandidateListRecord] = []
	covered = 0
	candidate_counts: list[int] = []

	for idx in range(total_queries):
		pair_id_raw = pair_ids[idx]
		pair_id = pair_id_raw.item() if hasattr(pair_id_raw, "item") else pair_id_raw
		reference_name = str(reference_names[idx])
		target_name = str(target_names[idx])

		triplet = triplet_dataset.triplets[idx]
		text_edit = str(triplet["caption"])

		if str(triplet["reference"]) != reference_name:
			raise ValueError(f"CIRR query alignment mismatch at idx={idx}.")

		row_names = sorted_names[idx]
		row_scores = sorted_scores[idx]
		keep_mask = np.array([name != reference_name for name in row_names], dtype=bool)

		filtered_names = row_names[keep_mask]
		filtered_scores = row_scores[keep_mask]

		final_names = filtered_names[:top_m]
		final_scores = filtered_scores[:top_m]

		target_positions = np.where(final_names == target_name)[0]
		target_in_top_m = len(target_positions) > 0
		target_rank_in_top_m = int(target_positions[0] + 1) if target_in_top_m else None
		if target_in_top_m:
			covered += 1

		candidates = [
			CandidateScore(
				candidate_id=str(candidate_id),
				score=float(score),
				rank=rank,
				metadata={},
			)
			for rank, (candidate_id, score) in enumerate(zip(final_names, final_scores), start=1)
		]

		records.append(
			CandidateListRecord(
				query=QueryRecord(
					query_id=f"cirr:{split}:{pair_id}",
					dataset="cirr",
					split=split,
					reference_name=reference_name,
					text_edit=text_edit,
					target_name=target_name,
					pair_id=pair_id,
					metadata={"group_size": len(triplet.get("img_set", {}).get("members", []))},
				),
				candidates=candidates,
				retriever_top_m=top_m,
				target_in_top_m=target_in_top_m,
				target_rank_in_top_m=target_rank_in_top_m,
				metadata={
					"candidate_count_after_reference_removal": int(len(filtered_names)),
				},
			)
		)
		candidate_counts.append(len(candidates))

	stats = {
		"num_queries": float(len(records)),
		"top_m": float(top_m),
		"avg_candidates_per_query": float(sum(candidate_counts) / max(1, len(candidate_counts))),
		"target_coverage_at_m": float((covered / max(1, len(records))) * 100.0),
	}
	return records, stats


__all__ = ["build_cirr_candidate_records"]
