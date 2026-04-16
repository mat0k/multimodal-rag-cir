import argparse
import random
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.rerankers.base import RerankCandidate, RerankQuery
from src.rerankers.qwen3vl_rank import Qwen3VLReranker


def build_cirr_case(
	idx: int,
	triplet_dataset: Any,
	image_dataset: Any,
	rng: random.Random,
	num_distractors: int,
) -> tuple[RerankQuery, list[RerankCandidate], str, dict[str, Any]]:
	sample = triplet_dataset[idx]
	target_name = sample["target_name"]
	reference_name = sample["reference_name"]
	text_edit = sample["caption"]

	all_candidate_ids = [
		name
		for name in image_dataset.name_to_relpath.keys()
		if name != reference_name and name != target_name
	]
	sampled = rng.sample(all_candidate_ids, k=min(num_distractors, len(all_candidate_ids)))
	sampled.append(target_name)
	rng.shuffle(sampled)

	query = RerankQuery(
		reference_image=sample["reference"],
		text_edit=text_edit,
		metadata={
			"dataset": "cirr",
			"pair_id": sample["pair_id"],
			"reference_name": reference_name,
			"target_name": target_name,
		},
	)

	candidates: list[RerankCandidate] = []
	for candidate_id in sampled:
		relpath = image_dataset.name_to_relpath[candidate_id]
		image_path = str(Path(image_dataset.images_dirpath) / relpath)
		candidates.append(
			RerankCandidate(
				image=image_path,
				candidate_id=candidate_id,
				metadata={"image_path": image_path},
			)
		)

	case_meta = {
		"id": int(sample["pair_id"]),
		"reference_name": reference_name,
		"target_name": target_name,
	}
	return query, candidates, target_name, case_meta


def build_fashioniq_case(
	idx: int,
	triplet_dataset: Any,
	image_dataset: Any,
	rng: random.Random,
	num_distractors: int,
	caption_joiner: str,
) -> tuple[RerankQuery, list[RerankCandidate], str, dict[str, Any]]:
	sample = triplet_dataset[idx]
	query_class = sample["class"]
	reference_name = sample["candidate_name"]
	target_name = sample["target_name"]

	text_edit = sample["transformed_caption"] if isinstance(sample["transformed_caption"], str) else ""
	if not text_edit:
		_, local_idx = triplet_dataset.get_class_index(idx)
		text_edit = caption_joiner.join(triplet_dataset.annotations[query_class][local_idx]["captions"])

	all_candidate_ids = [
		name
		for name in image_dataset.images[query_class]
		if name != reference_name and name != target_name
	]
	sampled = rng.sample(all_candidate_ids, k=min(num_distractors, len(all_candidate_ids)))
	sampled.append(target_name)
	rng.shuffle(sampled)

	query = RerankQuery(
		reference_image=sample["candidate"],
		text_edit=text_edit,
		metadata={
			"dataset": "fashioniq",
			"class": query_class,
			"reference_name": reference_name,
			"target_name": target_name,
		},
	)

	candidates: list[RerankCandidate] = []
	for candidate_id in sampled:
		image_path = image_dataset.get_image_path(query_class, candidate_id)
		candidates.append(
			RerankCandidate(
				image=image_path,
				candidate_id=candidate_id,
				metadata={"class": query_class, "image_path": image_path},
			)
		)

	case_meta = {
		"id": idx,
		"class": query_class,
		"reference_name": reference_name,
		"target_name": target_name,
	}
	return query, candidates, target_name, case_meta


def main() -> None:
	parser = argparse.ArgumentParser(description="Qwen3-VL tiny reranker sanity check.")
	parser.add_argument(
		"--config",
		type=str,
		default="configs/reranker/qwen3vl_reranker_2b.yaml",
		help="Path to reranker config YAML.",
	)
	parser.add_argument(
		"--dataset",
		type=str,
		default="fashioniq",
		choices=["cirr", "fashioniq"],
		help="Dataset for sanity check.",
	)
	parser.add_argument(
		"--single_score_check",
		action="store_true",
		help="Run one score(query, candidate) check and exit.",
	)
	parser.add_argument("--idx", type=int, default=0, help="Query index to evaluate.")
	parser.add_argument("--num_distractors", type=int, default=2, help="Distractors per query.")
	parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
	args = parser.parse_args()

	print("Loading Qwen3-VL reranker from config...")
	reranker = Qwen3VLReranker.from_config(args.config)
	print("Processor/model initialization: OK")
	print(f"Resolved checkpoint source: {reranker.model_name_or_path}")
	processor_class = type(reranker.processor).__name__ if reranker.processor is not None else "None"
	model_class = type(reranker.model).__name__ if reranker.model is not None else "None"
	print(f"Processor class: {processor_class}")
	print(f"Model class: {model_class}")

	rng = random.Random(args.seed)

	if args.dataset == "cirr":
		from src.datasets.cirr import build_cirr_dataset

		image_dataset = build_cirr_dataset(split="val", mode="images", image_transform=None, caption_transform=None)
		triplet_dataset = build_cirr_dataset(split="val", mode="triplets", image_transform=None, caption_transform=None)
		query, candidates, target_name, case_meta = build_cirr_case(
			idx=args.idx,
			triplet_dataset=triplet_dataset,
			image_dataset=image_dataset,
			rng=rng,
			num_distractors=args.num_distractors,
		)
	else:
		from src.datasets.fashioniq import build_fashioniq_dataset

		image_dataset = build_fashioniq_dataset(
			split="val",
			mode="images",
			image_transform=None,
			caption_transform=None,
			caption_joiner=" ",
		)
		triplet_dataset = build_fashioniq_dataset(
			split="val",
			mode="triplets",
			image_transform=None,
			caption_transform=None,
			caption_joiner=" ",
		)
		query, candidates, target_name, case_meta = build_fashioniq_case(
			idx=args.idx,
			triplet_dataset=triplet_dataset,
			image_dataset=image_dataset,
			rng=rng,
			num_distractors=args.num_distractors,
			caption_joiner=" ",
		)

	pool_size = len(candidates)
	print(f"Dataset: {args.dataset}")
	print(f"Case: {case_meta}")
	print(f"Pool size: {pool_size} (target + {max(0, pool_size - 1)} distractors)")

	if args.single_score_check:
		candidate = candidates[0]
		score_value = reranker.score(query=query, candidate=candidate)
		if not isinstance(score_value, float):
			raise TypeError("score(query, candidate) did not return float.")
		print("Single score check: OK")
		print(f"Candidate id: {candidate.candidate_id}")
		print(f"Target id: {target_name}")
		print(f"Score value: {score_value:.6f}")
		return

	scores = reranker.score_batch(query=query, candidates=candidates)
	if not all(isinstance(score, float) for score in scores):
		raise TypeError("Non-float score produced by reranker.")

	ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
	ranked_ids = [item[0].candidate_id for item in ranked]
	target_rank = ranked_ids.index(target_name) + 1

	print(f"Target rank: {target_rank}/{len(ranked)}")
	print("Top candidates:")
	for rank_idx, (candidate, score) in enumerate(ranked[: min(10, len(ranked))], start=1):
		marker = "<TARGET>" if candidate.candidate_id == target_name else ""
		print(f"  {rank_idx:02d}. {candidate.candidate_id} | score={score:.6f} {marker}")


if __name__ == "__main__":
	main()
