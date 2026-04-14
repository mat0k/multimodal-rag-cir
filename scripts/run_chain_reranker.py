from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import platform
from pathlib import Path
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.chain.candidate_io import load_candidate_records, save_reranked_records
from src.chain.pipeline import rerank_candidate_records
from src.evaluation.cirr_chain_eval import evaluate_cirr_chain_records
from src.evaluation.fashioniq_chain_eval import evaluate_fashioniq_chain_records
from src.rerankers.lamra_rank import LamRARanker
from src.utils.io import prepend_key_to_dict, save_records_to_csv, save_to_csv, save_to_json


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


def _resolve_enabled_datasets(config: dict[str, Any], datasets_override: list[str] | None) -> list[str]:
	if datasets_override:
		return list(datasets_override)

	datasets_cfg = config.get("datasets", {})
	resolved = [
		name
		for name in ("cirr", "fashioniq")
		if bool(datasets_cfg.get(name, {}).get("enabled", False))
	]

	if not resolved:
		raise ValueError("No enabled dataset in config. Enable one dataset or pass --datasets.")
	return resolved


def _as_bool_or_default(value: str, default: bool) -> bool:
	if not value:
		return default
	if value.lower() == "true":
		return True
	if value.lower() == "false":
		return False
	raise ValueError(f"Invalid boolean string '{value}'. Use true/false.")


def _merge_args_with_config(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
	chain_cfg = dict(config.get("chain", {}))
	stage_b_cfg = dict(config.get("stage_b", {}))
	outputs_cfg = dict(config.get("outputs", {}))

	if not args.datasets:
		args.datasets = _resolve_enabled_datasets(config=config, datasets_override=None)

	if args.rerank_top_n <= 0:
		default_top_n = stage_b_cfg.get("rerank_top_n", chain_cfg.get("top_m", 100))
		args.rerank_top_n = int(default_top_n)
	if not args.k_values:
		args.k_values = [int(k) for k in stage_b_cfg.get("k_values", chain_cfg.get("k_values", [1, 5, 10, 15]))]

	fuse_default = bool(stage_b_cfg.get("fuse_with_retrieval_scores", False))
	args.fuse_with_retrieval_scores = _as_bool_or_default(args.fuse_with_retrieval_scores, fuse_default)

	if args.retrieval_score_weight is None:
		args.retrieval_score_weight = float(stage_b_cfg.get("retrieval_score_weight", 0.0))
	if args.reranker_score_weight is None:
		args.reranker_score_weight = float(stage_b_cfg.get("reranker_score_weight", 1.0))

	if not args.reranker_config:
		args.reranker_config = str(stage_b_cfg.get("reranker_config", "configs/reranker/lamra_rank.yaml"))

	if not args.candidate_run_dir:
		args.candidate_run_dir = str(stage_b_cfg.get("candidate_run_dir", ""))
	if not args.cirr_candidates_file:
		args.cirr_candidates_file = str(stage_b_cfg.get("cirr_candidates_file", ""))
	if not args.fashioniq_candidates_file:
		args.fashioniq_candidates_file = str(stage_b_cfg.get("fashioniq_candidates_file", ""))

	if not args.output_path:
		args.output_path = str(outputs_cfg.get("root_dir", "results"))
	if not args.run_name:
		args.run_name = str(outputs_cfg.get("stage_b_run_name", f"chain_rerank_top{args.rerank_top_n}"))

	if not args.k_values:
		raise ValueError("k_values must contain at least one integer cutoff.")
	if any(int(k) <= 0 for k in args.k_values):
		raise ValueError("k_values must be positive integers.")

	return args


def _resolve_candidate_file(dataset: str, args: argparse.Namespace) -> str:
	if dataset == "cirr":
		candidate_file = args.cirr_candidates_file
		default_name = "cirr_candidates.jsonl"
	elif dataset == "fashioniq":
		candidate_file = args.fashioniq_candidates_file
		default_name = "fashioniq_candidates.jsonl"
	else:
		raise ValueError(f"Unsupported dataset '{dataset}'.")

	if not candidate_file and args.candidate_run_dir:
		candidate_file = os.path.join(args.candidate_run_dir, default_name)

	if not candidate_file:
		raise ValueError(
			f"No candidate file found for dataset '{dataset}'. "
			"Pass --candidate_run_dir or explicit --*_candidates_file."
		)

	if not os.path.isabs(candidate_file) and args.candidate_run_dir and not os.path.exists(candidate_file):
		candidate_file = os.path.join(args.candidate_run_dir, candidate_file)

	if not os.path.exists(candidate_file):
		raise FileNotFoundError(f"Candidate file not found for dataset '{dataset}': {candidate_file}")

	return candidate_file


def _build_runtime_context(
	args: argparse.Namespace,
	reranker: LamRARanker,
	run_output_dir: str,
	candidate_file_map: dict[str, str],
) -> dict[str, Any]:
	requested_device = str(load_yaml_config(args.reranker_config).get("runtime", {}).get("device", "auto"))
	resolved_device = str(reranker.device)

	cuda_available = bool(torch.cuda.is_available())
	cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
	cuda_device_index: int | None = None
	cuda_device_name: str | None = None

	if resolved_device.startswith("cuda") and cuda_available:
		if ":" in resolved_device:
			cuda_device_index = int(resolved_device.split(":", maxsplit=1)[1])
		else:
			cuda_device_index = int(torch.cuda.current_device())
		cuda_device_name = str(torch.cuda.get_device_name(cuda_device_index))

	return {
		"generated_at_utc": datetime.now(timezone.utc).isoformat(),
		"runtime": {
			"python_version": sys.version.split()[0],
			"platform": platform.platform(),
			"torch_version": torch.__version__,
			"cuda_version": torch.version.cuda,
			"cudnn_version": torch.backends.cudnn.version(),
		},
		"device": {
			"requested": requested_device,
			"resolved": resolved_device,
			"cuda_available": cuda_available,
			"cuda_device_count": cuda_device_count,
			"cuda_device_index": cuda_device_index,
			"cuda_device_name": cuda_device_name,
		},
		"stage": {
			"name": "chain_rerank",
			"datasets": list(args.datasets),
			"rerank_top_n": int(args.rerank_top_n),
			"k_values": [int(k) for k in args.k_values],
			"max_queries": args.max_queries,
			"fuse_with_retrieval_scores": bool(args.fuse_with_retrieval_scores),
			"retrieval_score_weight": float(args.retrieval_score_weight),
			"reranker_score_weight": float(args.reranker_score_weight),
			"tqdm": bool(args.tqdm),
			"candidate_files": dict(candidate_file_map),
		},
		"reranker": {
			"config": args.reranker_config,
			"model_name_or_path": reranker.model_name_or_path,
			"dtype": str(reranker.dtype),
		},
		"run": {
			"output_dir": run_output_dir,
		},
	}


def _build_prefixed_metrics(
	dataset: str,
	record_split: str,
	metrics_raw: dict[str, float],
) -> dict[str, float]:
	if dataset == "cirr":
		return prepend_key_to_dict(f"cirr_{record_split}_", metrics_raw)

	if dataset == "fashioniq":
		return prepend_key_to_dict("fashioniq_", metrics_raw)

	raise ValueError(f"Unsupported dataset '{dataset}'.")


def _build_structured_metrics(
	prefixed_metrics: dict[str, float],
	runtime_context: dict[str, Any],
) -> list[dict[str, Any]]:
	records: list[dict[str, Any]] = []
	runtime_summary = {
		"device": runtime_context["device"]["resolved"],
		"gpu_name": runtime_context["device"]["cuda_device_name"],
		"rerank_top_n": runtime_context["stage"]["rerank_top_n"],
		"fuse_with_retrieval_scores": runtime_context["stage"]["fuse_with_retrieval_scores"],
		"retrieval_score_weight": runtime_context["stage"]["retrieval_score_weight"],
		"reranker_score_weight": runtime_context["stage"]["reranker_score_weight"],
	}

	for full_metric_name, value in prefixed_metrics.items():
		dataset = "unknown"
		split = "unknown"
		metric = full_metric_name
		protocol = {
			"name": "two_stage_chain",
			"split": split,
			"candidate_pool": "retriever top-m candidates reranked by LamRA",
			"reference_removed": True,
			"runtime": runtime_summary,
			"score_type": "dataset_level",
		}

		if full_metric_name.startswith("cirr_"):
			dataset = "cirr"
			remainder = full_metric_name.removeprefix("cirr_")
			split, _, metric = remainder.partition("_")
			protocol["split"] = split
			if metric.startswith("chain_recall_at") or metric.startswith("retrieval_recall_at"):
				protocol["score_type"] = "dataset_level"
			elif metric == "target_coverage_at_m":
				protocol["score_type"] = "upper_bound"
			elif metric.startswith("latency_seconds_per"):
				protocol["score_type"] = "efficiency"
			elif metric == "latency_seconds":
				protocol["score_type"] = "system_latency"
			else:
				protocol["score_type"] = "run_stat"

		elif full_metric_name.startswith("fashioniq_"):
			dataset = "fashioniq"
			metric = full_metric_name.removeprefix("fashioniq_")
			if metric.startswith("val_"):
				split = "val"
				metric = metric.removeprefix("val_")
			elif metric.startswith("original_"):
				split = "test"
				metric = metric.removeprefix("original_")
			protocol["split"] = split

			if metric.startswith("avg_chain_recall_at") or metric.startswith("avg_retrieval_recall_at"):
				protocol["score_type"] = "macro_average"
			elif "_chain_recall_at" in metric or "_retrieval_recall_at" in metric:
				protocol["score_type"] = "dataset_level"
			elif "coverage" in metric:
				protocol["score_type"] = "upper_bound"
			elif metric.endswith("latency_seconds_per_query") or metric.endswith("latency_seconds_per_scored_pair"):
				protocol["score_type"] = "efficiency"
			elif metric.endswith("latency_seconds"):
				protocol["score_type"] = "system_latency"
			else:
				protocol["score_type"] = "run_stat"

		records.append(
			{
				"dataset": dataset,
				"split": split,
				"metric": metric,
				"value": value,
				"protocol": protocol,
			}
		)

	return records


def _build_protocol_summary(args: argparse.Namespace, runtime_context: dict[str, Any]) -> dict[str, Any]:
	return {
		"validated": True,
		"mode": "two_stage_chain",
		"datasets": list(args.datasets),
		"candidate_files": runtime_context["stage"]["candidate_files"],
		"rerank_top_n": int(args.rerank_top_n),
		"k_values": [int(k) for k in args.k_values],
		"fuse_with_retrieval_scores": bool(args.fuse_with_retrieval_scores),
		"retrieval_score_weight": float(args.retrieval_score_weight),
		"reranker_score_weight": float(args.reranker_score_weight),
		"policy": {
			"reference_removed": True,
			"target_injection": False,
			"candidate_source": "stage_a_retriever_top_m",
		},
		"runtime": runtime_context,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Run Stage-B chain reranking over Stage-A candidate files.")
	parser.add_argument("--config", type=str, default="configs/chain/vista_lamra_chain.yaml", help="Path to chain YAML config.")
	parser.add_argument("--datasets", nargs="+", default=[], choices=["cirr", "fashioniq"], help="Datasets to process.")

	parser.add_argument("--candidate_run_dir", type=str, default="", help="Directory containing cirr_candidates.jsonl/fashioniq_candidates.jsonl")
	parser.add_argument("--cirr_candidates_file", type=str, default="", help="Explicit CIRR candidates file path.")
	parser.add_argument("--fashioniq_candidates_file", type=str, default="", help="Explicit FashionIQ candidates file path.")

	parser.add_argument("--reranker_config", type=str, default="", help="LamRA reranker config file path.")
	parser.add_argument("--rerank_top_n", type=int, default=-1, help="Top-N candidates to rerank per query.")
	parser.add_argument("--k_values", nargs="+", type=int, default=[], help="Recall@K cutoffs to report before/after rerank.")
	parser.add_argument("--max_queries", type=int, default=None, help="Optional cap per dataset for debug/smoke runs.")

	parser.add_argument("--fuse_with_retrieval_scores", type=str, default="", choices=["", "true", "false"], help="Fuse retrieval and reranker scores.")
	parser.add_argument("--retrieval_score_weight", type=float, default=None, help="Weight for retriever score during fusion.")
	parser.add_argument("--reranker_score_weight", type=float, default=None, help="Weight for reranker score during fusion.")
	parser.add_argument("--tqdm", action="store_true", help="Enable progress bars.")

	parser.add_argument("--output_path", type=str, default="", help="Root output directory.")
	parser.add_argument("--run_name", type=str, default="", help="Output run folder name.")
	args = parser.parse_args()

	config = load_yaml_config(args.config)
	args = _merge_args_with_config(args=args, config=config)

	reranker = LamRARanker.from_config(args.reranker_config)

	run_output_dir = os.path.join(args.output_path, args.run_name)
	os.makedirs(run_output_dir, exist_ok=True)

	prefixed_metrics: dict[str, float] = {}
	candidate_file_map: dict[str, str] = {}
	reranked_file_map: dict[str, str] = {}

	for dataset in args.datasets:
		candidate_file = _resolve_candidate_file(dataset=dataset, args=args)
		candidate_records = load_candidate_records(candidate_file)
		if args.max_queries is not None:
			candidate_records = candidate_records[: args.max_queries]

		short_records = [
			record.query.query_id
			for record in candidate_records
			if len(record.candidates) < int(args.rerank_top_n)
		]
		if short_records:
			preview = ", ".join(short_records[:3])
			raise ValueError(
				"rerank_top_n exceeds available Stage-A candidates for one or more queries. "
				f"rerank_top_n={args.rerank_top_n}, first query_ids: {preview}"
			)

		reranked_records, latency_stats = rerank_candidate_records(
			reranker=reranker,
			records=candidate_records,
			rerank_top_n=int(args.rerank_top_n),
			use_tqdm=bool(args.tqdm),
			fuse_with_retrieval_scores=bool(args.fuse_with_retrieval_scores),
			retrieval_score_weight=float(args.retrieval_score_weight),
			reranker_score_weight=float(args.reranker_score_weight),
		)

		if dataset == "cirr":
			raw_metrics = evaluate_cirr_chain_records(
				reranked_records,
				k_values=tuple(int(k) for k in args.k_values),
				latency_stats=latency_stats,
			)
			record_split = reranked_records[0].query.split if reranked_records else "val"
		elif dataset == "fashioniq":
			record_split = reranked_records[0].query.split if reranked_records else "val"
			metric_prefix = "original" if record_split == "test" else "val"
			raw_metrics = evaluate_fashioniq_chain_records(
				reranked_records,
				metric_prefix=metric_prefix,
				k_values=tuple(int(k) for k in args.k_values),
				latency_stats=latency_stats,
			)
		else:
			raise ValueError(f"Unsupported dataset '{dataset}'.")

		prefixed_metrics.update(_build_prefixed_metrics(dataset=dataset, record_split=record_split, metrics_raw=raw_metrics))

		reranked_file_name = f"{dataset}_reranked.jsonl"
		save_reranked_records(reranked_records, os.path.join(run_output_dir, reranked_file_name))

		candidate_file_map[dataset] = candidate_file
		reranked_file_map[dataset] = reranked_file_name

	save_to_csv(prefixed_metrics, os.path.join(run_output_dir, "metrics.csv"))
	save_to_json(prefixed_metrics, os.path.join(run_output_dir, "metrics.json"))
	save_to_json(prefixed_metrics, os.path.join(run_output_dir, "metrics_raw.json"))

	runtime_context = _build_runtime_context(
		args=args,
		reranker=reranker,
		run_output_dir=run_output_dir,
		candidate_file_map=candidate_file_map,
	)
	runtime_context["stage"]["reranked_files"] = reranked_file_map
	save_to_json(runtime_context, os.path.join(run_output_dir, "evaluation_runtime.json"))

	structured_metrics = _build_structured_metrics(prefixed_metrics=prefixed_metrics, runtime_context=runtime_context)
	save_records_to_csv(
		structured_metrics,
		os.path.join(run_output_dir, "metrics_structured.csv"),
		fieldnames=["dataset", "split", "metric", "value", "protocol"],
	)
	save_to_json(structured_metrics, os.path.join(run_output_dir, "metrics_structured.json"))

	protocol_summary = _build_protocol_summary(args=args, runtime_context=runtime_context)
	save_to_json(protocol_summary, os.path.join(run_output_dir, "evaluation_protocol.json"))

	run_config = vars(args).copy()
	run_config["resolved_device"] = str(reranker.device)
	run_config["candidate_files"] = candidate_file_map
	run_config["reranked_files"] = reranked_file_map
	save_to_json(run_config, os.path.join(run_output_dir, "run_config.json"))

	print("==== Chain Reranker Summary ====")
	print(f"run_output_dir: {run_output_dir}")
	print(f"datasets: {args.datasets}")
	print(f"rerank_top_n: {args.rerank_top_n}")
	for key, value in prefixed_metrics.items():
		print(f"{key}: {value:.4f}")


if __name__ == "__main__":
	main()
