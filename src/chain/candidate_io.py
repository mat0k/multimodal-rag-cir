from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from src.chain.types import CandidateListRecord, CandidateScore, QueryRecord, RerankedRecord


def _ensure_parent_dir(output_path: str) -> None:
	parent = os.path.dirname(output_path)
	if parent:
		os.makedirs(parent, exist_ok=True)


def _is_jsonl(path: str) -> bool:
	return path.lower().endswith(".jsonl")


def _is_json(path: str) -> bool:
	return path.lower().endswith(".json")


def _save_raw_records(records: Sequence[dict[str, Any]], output_path: str) -> None:
	_ensure_parent_dir(output_path)

	if _is_jsonl(output_path):
		with open(output_path, "w", encoding="utf-8") as file_obj:
			for record in records:
				file_obj.write(json.dumps(record, ensure_ascii=True) + "\n")
		return

	if _is_json(output_path):
		with open(output_path, "w", encoding="utf-8") as file_obj:
			json.dump(list(records), file_obj, indent=2)
		return

	raise ValueError("Unsupported extension. Use .jsonl or .json")


def _load_raw_records(input_path: str) -> list[dict[str, Any]]:
	if not Path(input_path).exists():
		raise FileNotFoundError(f"File not found: {input_path}")

	if _is_jsonl(input_path):
		records: list[dict[str, Any]] = []
		with open(input_path, "r", encoding="utf-8") as file_obj:
			for line_idx, line in enumerate(file_obj, start=1):
				payload = line.strip()
				if not payload:
					continue
				try:
					parsed = json.loads(payload)
				except json.JSONDecodeError as exc:
					raise ValueError(f"Invalid JSONL at line {line_idx} in {input_path}") from exc
				if not isinstance(parsed, dict):
					raise ValueError(f"JSONL line {line_idx} must be an object.")
				records.append(parsed)
		return records

	if _is_json(input_path):
		with open(input_path, "r", encoding="utf-8") as file_obj:
			parsed = json.load(file_obj)
		if not isinstance(parsed, list):
			raise ValueError(f"JSON file {input_path} must contain a top-level list.")
		records = [item for item in parsed if isinstance(item, dict)]
		if len(records) != len(parsed):
			raise ValueError("All items in top-level list must be objects.")
		return records

	raise ValueError("Unsupported extension. Use .jsonl or .json")


def _candidate_score_from_dict(payload: dict[str, Any]) -> CandidateScore:
	return CandidateScore(
		candidate_id=str(payload["candidate_id"]),
		score=float(payload["score"]),
		rank=int(payload["rank"]) if payload.get("rank") is not None else None,
		metadata=dict(payload.get("metadata", {})),
	)


def _query_record_from_dict(payload: dict[str, Any]) -> QueryRecord:
	dataset = str(payload["dataset"])
	if dataset not in {"cirr", "fashioniq"}:
		raise ValueError(f"Unsupported dataset '{dataset}'.")

	return QueryRecord(
		query_id=str(payload["query_id"]),
		dataset=dataset,
		split=str(payload["split"]),
		reference_name=str(payload["reference_name"]),
		text_edit=str(payload.get("text_edit", "")),
		target_name=str(payload["target_name"]) if payload.get("target_name") is not None else None,
		query_class=str(payload["query_class"]) if payload.get("query_class") is not None else None,
		pair_id=payload.get("pair_id"),
		metadata=dict(payload.get("metadata", {})),
	)


def candidate_list_record_from_dict(payload: dict[str, Any]) -> CandidateListRecord:
	candidates = [_candidate_score_from_dict(item) for item in payload.get("candidates", [])]

	return CandidateListRecord(
		query=_query_record_from_dict(dict(payload["query"])),
		candidates=candidates,
		retriever_top_m=int(payload.get("retriever_top_m", len(candidates))),
		target_in_top_m=(
			bool(payload["target_in_top_m"])
			if payload.get("target_in_top_m") is not None
			else None
		),
		target_rank_in_top_m=(
			int(payload["target_rank_in_top_m"])
			if payload.get("target_rank_in_top_m") is not None
			else None
		),
		metadata=dict(payload.get("metadata", {})),
	)


def reranked_record_from_dict(payload: dict[str, Any]) -> RerankedRecord:
	candidates = [_candidate_score_from_dict(item) for item in payload.get("candidates", [])]

	return RerankedRecord(
		query=_query_record_from_dict(dict(payload["query"])),
		candidates=candidates,
		rerank_top_n=int(payload.get("rerank_top_n", len(candidates))),
		target_in_rerank_top_n=(
			bool(payload["target_in_rerank_top_n"])
			if payload.get("target_in_rerank_top_n") is not None
			else None
		),
		target_rank_in_rerank=(
			int(payload["target_rank_in_rerank"])
			if payload.get("target_rank_in_rerank") is not None
			else None
		),
		metadata=dict(payload.get("metadata", {})),
	)


def _validate_common_query_record(query: QueryRecord) -> None:
	if not query.query_id:
		raise ValueError("query_id must be non-empty.")
	if not query.reference_name:
		raise ValueError(f"reference_name must be non-empty for query_id={query.query_id}.")
	if query.dataset == "fashioniq" and not query.query_class:
		raise ValueError(f"fashioniq query must define query_class for query_id={query.query_id}.")


def _validate_candidate_ids(
	query_id: str,
	reference_name: str,
	candidates: list[CandidateScore],
) -> None:
	candidate_ids = [item.candidate_id for item in candidates]

	if len(candidate_ids) != len(set(candidate_ids)):
		raise ValueError(f"Duplicate candidate ids detected for query_id={query_id}.")
	if reference_name in candidate_ids:
		raise ValueError(f"Reference image leaked into candidate list for query_id={query_id}.")


def _validate_target_fields(
	query_id: str,
	target_name: str | None,
	ranked_ids: list[str],
	is_present_flag: bool | None,
	rank_field: int | None,
	context_name: str,
) -> None:
	if target_name is None:
		return

	detected_rank = None
	if target_name in ranked_ids:
		detected_rank = ranked_ids.index(target_name) + 1

	if is_present_flag is not None:
		detected_present = detected_rank is not None
		if detected_present != is_present_flag:
			raise ValueError(
				f"Inconsistent {context_name} presence flag for query_id={query_id}. "
				f"Flag={is_present_flag}, detected={detected_present}."
			)

	if rank_field is not None and detected_rank != rank_field:
		raise ValueError(
			f"Inconsistent {context_name} rank for query_id={query_id}. "
			f"Field={rank_field}, detected={detected_rank}."
		)


def validate_candidate_records(records: Sequence[CandidateListRecord]) -> None:
	seen_query_ids: set[str] = set()

	for record in records:
		query = record.query
		_validate_common_query_record(query)

		if query.query_id in seen_query_ids:
			raise ValueError(f"Duplicate query_id detected: {query.query_id}")
		seen_query_ids.add(query.query_id)

		if record.retriever_top_m <= 0:
			raise ValueError(f"retriever_top_m must be > 0 for query_id={query.query_id}.")

		_validate_candidate_ids(
			query_id=query.query_id,
			reference_name=query.reference_name,
			candidates=record.candidates,
		)

		if query.dataset == "fashioniq":
			for item in record.candidates:
				candidate_class = item.metadata.get("class")
				if candidate_class is not None and str(candidate_class) != query.query_class:
					raise ValueError(
						"FashionIQ class mismatch for query_id="
						f"{query.query_id}: query_class={query.query_class}, candidate_class={candidate_class}"
					)

		ranked_ids = [item.candidate_id for item in record.candidates]
		_validate_target_fields(
			query_id=query.query_id,
			target_name=query.target_name,
			ranked_ids=ranked_ids,
			is_present_flag=record.target_in_top_m,
			rank_field=record.target_rank_in_top_m,
			context_name="target_in_top_m",
		)


def validate_reranked_records(records: Sequence[RerankedRecord]) -> None:
	seen_query_ids: set[str] = set()

	for record in records:
		query = record.query
		_validate_common_query_record(query)

		if query.query_id in seen_query_ids:
			raise ValueError(f"Duplicate query_id detected: {query.query_id}")
		seen_query_ids.add(query.query_id)

		if record.rerank_top_n <= 0:
			raise ValueError(f"rerank_top_n must be > 0 for query_id={query.query_id}.")

		_validate_candidate_ids(
			query_id=query.query_id,
			reference_name=query.reference_name,
			candidates=record.candidates,
		)

		if query.dataset == "fashioniq":
			for item in record.candidates:
				candidate_class = item.metadata.get("class")
				if candidate_class is not None and str(candidate_class) != query.query_class:
					raise ValueError(
						"FashionIQ class mismatch for query_id="
						f"{query.query_id}: query_class={query.query_class}, candidate_class={candidate_class}"
					)

		ranked_ids = [item.candidate_id for item in record.candidates]
		_validate_target_fields(
			query_id=query.query_id,
			target_name=query.target_name,
			ranked_ids=ranked_ids,
			is_present_flag=record.target_in_rerank_top_n,
			rank_field=record.target_rank_in_rerank,
			context_name="target_in_rerank_top_n",
		)


def save_candidate_records(records: Sequence[CandidateListRecord], output_path: str) -> None:
	validate_candidate_records(records)
	payload = [asdict(record) for record in records]
	_save_raw_records(payload, output_path)


def load_candidate_records(input_path: str) -> list[CandidateListRecord]:
	payload = _load_raw_records(input_path)
	records = [candidate_list_record_from_dict(item) for item in payload]
	validate_candidate_records(records)
	return records


def save_reranked_records(records: Sequence[RerankedRecord], output_path: str) -> None:
	validate_reranked_records(records)
	payload = [asdict(record) for record in records]
	_save_raw_records(payload, output_path)


def load_reranked_records(input_path: str) -> list[RerankedRecord]:
	payload = _load_raw_records(input_path)
	records = [reranked_record_from_dict(item) for item in payload]
	validate_reranked_records(records)
	return records


__all__ = [
	"candidate_list_record_from_dict",
	"load_candidate_records",
	"load_reranked_records",
	"reranked_record_from_dict",
	"save_candidate_records",
	"save_reranked_records",
	"validate_candidate_records",
	"validate_reranked_records",
]
