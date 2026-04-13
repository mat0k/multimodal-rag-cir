from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.chain.candidate_io import (
	save_candidate_records,
	save_reranked_records,
	validate_candidate_records,
	validate_reranked_records,
)
from src.chain.cirr_chain import build_cirr_candidate_records
from src.chain.fashioniq_chain import build_fashioniq_candidate_records
from src.chain.pipeline import rerank_candidate_records
from src.evaluation.cirr_chain_eval import evaluate_cirr_chain_records
from src.evaluation.fashioniq_chain_eval import evaluate_fashioniq_chain_records
from src.rerankers.lamra_rank import LamRARanker
from src.retrievers.base import TwoEncoderVLM
from src.utils.io import save_to_json


def load_yaml_config(config_path: str) -> dict[str, Any]:
	try:
		import yaml
	except ModuleNotFoundError as exc:
		raise ModuleNotFoundError(
			"PyYAML is required to load config files. Install with `pip install pyyaml`."
		) from exc

	with open(config_path, "r", encoding="utf-8") as file_obj:
		payload = yaml.safe_load(file_obj) or {}

	if not isinstance(payload, dict):
		raise ValueError(f"Config at '{config_path}' must parse to a mapping.")
	return payload


def resolve_fashioniq_study_setting(
	study_setting: str,
	caption_joiner_mode: str,
	caption_order_mode: str,
) -> tuple[str, str]:
	if study_setting == "caption_ablation_v1":
		return "and", "both"
	return caption_joiner_mode, caption_order_mode


def resolve_caption_joiner_override(mode: str) -> str | None:
	if mode == "protocol_default":
		return None
	if mode == "space":
		return " "
	if mode == "and":
		return " and "
	raise ValueError(f"Unsupported fashioniq_caption_joiner_mode '{mode}'.")


def resolve_device(device_arg: str) -> torch.device:
	if device_arg == "auto":
		if torch.cuda.is_available():
			print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
			return torch.device("cuda")
		print("No GPU available, using CPU.")
		return torch.device("cpu")
	return torch.device(device_arg)


def load_retriever(
	retriever_module: str,
	retriever_class: str,
	model_name_or_path: str,
	device: torch.device,
	init_kwargs: dict[str, Any],
) -> TwoEncoderVLM:
	try:
		module = importlib.import_module(retriever_module)
	except ModuleNotFoundError as exc:
		raise ModuleNotFoundError(
			f"Could not import retriever module '{retriever_module}'."
		) from exc

	if not hasattr(module, retriever_class):
		raise AttributeError(
			f"Retriever class '{retriever_class}' not found in module '{retriever_module}'."
		)

	retriever_cls = getattr(module, retriever_class)
	if hasattr(retriever_cls, "from_pretrained"):
		model = retriever_cls.from_pretrained(model_name_or_path, **init_kwargs)
	else:
		model = retriever_cls(model_name_or_path=model_name_or_path, **init_kwargs)

	if hasattr(model, "to"):
		model = model.to(device)
	return model


def _assert_no_reference_leakage(label: str, records: list[Any]) -> None:
	for record in records:
		reference_name = record.query.reference_name
		candidate_ids = [item.candidate_id for item in record.candidates]
		if reference_name in candidate_ids:
			raise AssertionError(
				f"Reference leakage detected in {label} for query_id={record.query.query_id}."
			)


def _assert_coverage_present(metrics: dict[str, float], dataset: str) -> None:
	coverage_keys = [key for key in metrics.keys() if "coverage_at_m" in key]
	if not coverage_keys:
		raise AssertionError(f"No coverage metric found for dataset={dataset}.")

	for key in coverage_keys:
		value = metrics[key]
		if value != value:
			raise AssertionError(f"Coverage metric is NaN: {key}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Tiny end-to-end chain sanity check (Stage A + Stage B).")
	parser.add_argument("--config", type=str, default="configs/chain/vista_lamra_chain.yaml", help="Chain config path.")
	parser.add_argument("--datasets", nargs="+", default=["cirr", "fashioniq"], choices=["cirr", "fashioniq"], help="Datasets to sanity-check.")
	parser.add_argument("--top_m", type=int, default=20, help="Retriever top-M candidates for sanity run.")
	parser.add_argument("--rerank_top_n", type=int, default=20, help="Rerank top-N candidates for sanity run.")
	parser.add_argument("--max_queries", type=int, default=2, help="Small query cap per dataset.")
	parser.add_argument("--batch_size", type=int, default=8, help="Retriever batch size.")
	parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers for sanity check.")
	parser.add_argument("--device", type=str, default="", help="Retriever device override.")
	parser.add_argument("--query_embedding_mode", type=str, default="", choices=["", "vista_mm", "legacy_fusion"], help="Retriever query embedding mode override.")
	parser.add_argument("--fusion_type", type=str, default="", help="Retriever fusion type override.")
	parser.add_argument("--output_dir", type=str, default="results/chain_sanity_small", help="Output directory for sanity artifacts.")
	parser.add_argument("--tqdm", action="store_true", help="Enable progress bars.")
	args = parser.parse_args()

	if args.rerank_top_n > args.top_m:
		raise ValueError("rerank_top_n must be <= top_m for sanity check.")

	cfg = load_yaml_config(args.config)
	retriever_cfg = dict(cfg.get("retriever", {}))
	runtime_cfg = dict(cfg.get("runtime", {}))
	datasets_cfg = dict(cfg.get("datasets", {}))
	stage_b_cfg = dict(cfg.get("stage_b", {}))

	model_name_or_path = str(retriever_cfg.get("model_name_or_path", ""))
	retriever_module = str(retriever_cfg.get("module", "src.retrievers.vista_retriever"))
	retriever_class = str(retriever_cfg.get("class", "VistaBGERetriever"))
	checkpoint_path = str(retriever_cfg.get("checkpoint_path", ""))
	init_kwargs = dict(retriever_cfg.get("init_kwargs", {}))
	if checkpoint_path:
		init_kwargs["checkpoint_path"] = checkpoint_path

	if not model_name_or_path:
		raise ValueError("retriever.model_name_or_path is required in chain config.")

	query_embedding_mode = args.query_embedding_mode or str(runtime_cfg.get("query_embedding_mode", "vista_mm"))
	fusion_type = args.fusion_type or str(runtime_cfg.get("fusion_type", "sum"))
	device = resolve_device(args.device or str(runtime_cfg.get("device", "auto")))

	retriever = load_retriever(
		retriever_module=retriever_module,
		retriever_class=retriever_class,
		model_name_or_path=model_name_or_path,
		device=device,
		init_kwargs=init_kwargs,
	)
	reranker_cfg = str(stage_b_cfg.get("reranker_config", "configs/reranker/lamra_rank.yaml"))
	reranker = LamRARanker.from_config(reranker_cfg)

	os.makedirs(args.output_dir, exist_ok=True)

	report: dict[str, Any] = {
		"generated_at_utc": datetime.now(timezone.utc).isoformat(),
		"config": {
			"chain_config": args.config,
			"reranker_config": reranker_cfg,
			"datasets": list(args.datasets),
			"top_m": int(args.top_m),
			"rerank_top_n": int(args.rerank_top_n),
			"max_queries": int(args.max_queries),
			"batch_size": int(args.batch_size),
			"num_workers": int(args.num_workers),
			"query_embedding_mode": query_embedding_mode,
			"fusion_type": fusion_type,
		},
		"datasets": {},
	}

	for dataset in args.datasets:
		if dataset == "cirr":
			cirr_cfg = dict(datasets_cfg.get("cirr", {}))
			split = str(cirr_cfg.get("split", "val"))
			candidate_records, stage_a_stats = build_cirr_candidate_records(
				model=retriever,
				split=split,
				top_m=args.top_m,
				query_embedding_mode=query_embedding_mode,
				fusion_type=fusion_type,
				batch_size=args.batch_size,
				num_workers=args.num_workers,
				use_tqdm=bool(args.tqdm),
				max_queries=args.max_queries,
			)
		elif dataset == "fashioniq":
			fashioniq_cfg = dict(datasets_cfg.get("fashioniq", {}))
			eval_protocol = str(fashioniq_cfg.get("eval_protocol", "val_split"))
			study_setting = str(fashioniq_cfg.get("study_setting", "default"))
			caption_joiner_mode = str(fashioniq_cfg.get("caption_joiner_mode", "protocol_default"))
			caption_order_mode = str(fashioniq_cfg.get("caption_order_mode", "original"))
			effective_joiner_mode, effective_order_mode = resolve_fashioniq_study_setting(
				study_setting,
				caption_joiner_mode,
				caption_order_mode,
			)
			candidate_records, stage_a_stats = build_fashioniq_candidate_records(
				model=retriever,
				top_m=args.top_m,
				query_embedding_mode=query_embedding_mode,
				fusion_type=fusion_type,
				batch_size=args.batch_size,
				num_workers=args.num_workers,
				use_tqdm=bool(args.tqdm),
				max_queries=args.max_queries,
				eval_protocol=eval_protocol,
				caption_joiner_override=resolve_caption_joiner_override(effective_joiner_mode),
				caption_order_mode=effective_order_mode,
			)
		else:
			raise ValueError(f"Unsupported dataset: {dataset}")

		validate_candidate_records(candidate_records)
		_assert_no_reference_leakage("stage_a", candidate_records)

		candidates_path = os.path.join(args.output_dir, f"{dataset}_candidates_sanity.jsonl")
		save_candidate_records(candidate_records, candidates_path)

		reranked_records, latency_stats = rerank_candidate_records(
			reranker=reranker,
			records=candidate_records,
			rerank_top_n=args.rerank_top_n,
			use_tqdm=bool(args.tqdm),
			fuse_with_retrieval_scores=False,
			retrieval_score_weight=0.0,
			reranker_score_weight=1.0,
		)

		validate_reranked_records(reranked_records)
		_assert_no_reference_leakage("stage_b", reranked_records)

		reranked_path = os.path.join(args.output_dir, f"{dataset}_reranked_sanity.jsonl")
		save_reranked_records(reranked_records, reranked_path)

		if dataset == "cirr":
			metrics = evaluate_cirr_chain_records(reranked_records, latency_stats=latency_stats)
		else:
			split = reranked_records[0].query.split if reranked_records else "val"
			metric_prefix = "original" if split == "test" else "val"
			metrics = evaluate_fashioniq_chain_records(
				reranked_records,
				metric_prefix=metric_prefix,
				latency_stats=latency_stats,
			)

		_assert_coverage_present(metrics, dataset=dataset)

		report["datasets"][dataset] = {
			"stage_a": stage_a_stats,
			"stage_b": {
				"num_reranked_queries": float(len(reranked_records)),
				"latency_seconds": float(latency_stats.get("latency_seconds", 0.0)),
				"latency_seconds_per_query": float(latency_stats.get("latency_seconds_per_query", 0.0)),
				"latency_seconds_per_scored_pair": float(latency_stats.get("latency_seconds_per_scored_pair", 0.0)),
			},
			"metrics": metrics,
			"artifacts": {
				"candidates": os.path.basename(candidates_path),
				"reranked": os.path.basename(reranked_path),
			},
		}

	report_path = os.path.join(args.output_dir, "chain_sanity_report.json")
	save_to_json(report, report_path)

	print("==== Chain Sanity Summary ====")
	print(f"output_dir: {args.output_dir}")
	print(f"datasets: {args.datasets}")
	print(f"top_m: {args.top_m} | rerank_top_n: {args.rerank_top_n} | max_queries: {args.max_queries}")
	print(f"report: {report_path}")
	print("CHAIN_SANITY_SUMMARY_JSON=")
	print(json.dumps(report, indent=2))


if __name__ == "__main__":
	main()
