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

from src.evaluation.cirr_rerank_eval import evaluate_cirr_standalone_rerank
from src.evaluation.fashioniq_rerank_eval import evaluate_fashioniq_standalone_rerank
from src.rerankers.base import BaseReranker
from src.rerankers.factory import build_reranker_from_config
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


def resolve_dataset_name(config: dict[str, Any], dataset_override: str | None) -> str:
	if dataset_override:
		return dataset_override

	datasets_cfg = config.get("datasets", {})
	enabled = [name for name in ("cirr", "fashioniq") if datasets_cfg.get(name, {}).get("enabled", False)]

	if len(enabled) == 1:
		return enabled[0]
	if len(enabled) == 0:
		raise ValueError("No enabled dataset in config. Enable one dataset or pass --dataset.")
	raise ValueError("Multiple datasets are enabled in config. Pass --dataset to select one.")


def run_standalone_evaluation(
	dataset: str,
	reranker: BaseReranker,
	config: dict[str, Any],
) -> dict[str, float]:
	pipeline_cfg = config.get("pipeline", {})
	if pipeline_cfg.get("mode", "standalone") != "standalone":
		raise ValueError(
			"This runner currently supports standalone mode only. "
			"Set pipeline.mode=standalone for Phase 2 evaluation."
		)

	candidate_cfg = config.get("standalone_candidate_pool", {})
	num_random_distractors = int(candidate_cfg.get("num_random_negatives", 31))
	seed = int(config.get("seed", 42))
	use_tqdm = bool(config.get("runtime", {}).get("tqdm", False))

	if dataset == "cirr":
		cirr_cfg = config.get("datasets", {}).get("cirr", {})
		split = str(cirr_cfg.get("split", "val"))
		k_values = tuple(int(k) for k in cirr_cfg.get("global_k_values", [1, 5, 10]))
		return evaluate_cirr_standalone_rerank(
			reranker=reranker,
			split=split,
			num_random_distractors=num_random_distractors,
			seed=seed,
			k_values=k_values,
			use_tqdm=use_tqdm,
		)

	if dataset == "fashioniq":
		fashioniq_cfg = config.get("datasets", {}).get("fashioniq", {})
		split = str(fashioniq_cfg.get("split", "val"))
		eval_protocol = str(fashioniq_cfg.get("eval_protocol", "val_split"))
		k_values = tuple(int(k) for k in fashioniq_cfg.get("recall_k_values", [5, 10, 50]))

		caption_joiner_val = str(fashioniq_cfg.get("caption_joiner_val_split", " "))
		caption_joiner_original = str(fashioniq_cfg.get("caption_joiner_original_split", " and "))
		caption_joiner = caption_joiner_original if eval_protocol == "original_split" else caption_joiner_val
		metric_prefix = "original" if eval_protocol == "original_split" else "val"

		return evaluate_fashioniq_standalone_rerank(
			reranker=reranker,
			split=split,
			caption_joiner=caption_joiner,
			metric_prefix=metric_prefix,
			num_random_distractors=num_random_distractors,
			seed=seed,
			k_values=k_values,
			use_tqdm=use_tqdm,
		)

	raise ValueError(f"Unsupported dataset '{dataset}'. Supported: cirr, fashioniq.")


def build_runtime_context(
	dataset: str,
	config: dict[str, Any],
	args: argparse.Namespace,
	run_output_dir: str,
	resolved_device_override: str | None = None,
) -> dict[str, Any]:
	model_cfg = config.get("model", {})
	runtime_cfg = config.get("runtime", {})
	candidate_cfg = config.get("standalone_candidate_pool", {})

	requested_device = str(runtime_cfg.get("device", "auto"))
	cuda_available = bool(torch.cuda.is_available())
	cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
	cuda_device_index: int | None = None
	cuda_device_name: str | None = None

	if resolved_device_override:
		resolved_device = resolved_device_override
	elif requested_device == "auto":
		resolved_device = "cuda" if cuda_available else "cpu"
	else:
		resolved_device = requested_device

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
		"run": {
			"dataset": dataset,
			"config_path": args.config,
			"output_dir": run_output_dir,
		},
		"model": {
			"name": model_cfg.get("name"),
			"model_name_or_path": model_cfg.get("model_name_or_path"),
			"revision": model_cfg.get("revision"),
			"trust_remote_code": model_cfg.get("trust_remote_code"),
			"local_files_only": model_cfg.get("local_files_only"),
			"dtype": runtime_cfg.get("dtype"),
			"requested_device": requested_device,
			"resolved_device": resolved_device,
			"checkpoint_path": model_cfg.get("checkpoint_path"),
		},
		"candidate_pool": {
			"mode": config.get("pipeline", {}).get("mode", "standalone"),
			"include_true_target": candidate_cfg.get("include_true_target", True),
			"num_random_negatives": int(candidate_cfg.get("num_random_negatives", 31)),
			"configured_pool_size": int(candidate_cfg.get("num_random_negatives", 31))
			+ (1 if bool(candidate_cfg.get("include_true_target", True)) else 0),
		},
	}


def build_prefixed_metrics(dataset: str, metrics: dict[str, float]) -> dict[str, float]:
	if dataset == "cirr":
		return prepend_key_to_dict("cirr_val_", metrics)
	if dataset == "fashioniq":
		protocol_prefix = ""
		if any(key.startswith("val_") for key in metrics):
			protocol_prefix = "val"
		elif any(key.startswith("original_") for key in metrics):
			protocol_prefix = "original"

		latency_keys = {
			"latency_seconds",
			"latency_seconds_per_query",
			"latency_seconds_per_scored_pair",
		}
		normalized: dict[str, float] = {}
		for key, value in metrics.items():
			if key in latency_keys and protocol_prefix:
				normalized[f"{protocol_prefix}_{key}"] = value
			else:
				normalized[key] = value

		return prepend_key_to_dict("fashioniq_", normalized)
	raise ValueError(f"Unsupported dataset '{dataset}'. Supported: cirr, fashioniq.")


def build_structured_metrics(
	dataset: str,
	prefixed_metrics: dict[str, float],
	runtime_context: dict[str, Any],
	config: dict[str, Any],
) -> list[dict[str, Any]]:
	records: list[dict[str, Any]] = []
	runtime_summary = {
		"device": runtime_context["device"]["resolved"],
		"gpu_name": runtime_context["device"]["cuda_device_name"],
		"dtype": runtime_context["model"].get("dtype"),
	}

	pool_cfg = runtime_context.get("candidate_pool", {})

	for full_metric_name, value in prefixed_metrics.items():
		split = "unknown"
		metric = full_metric_name
		protocol = {
			"name": "standalone_rerank",
			"split": split,
			"candidate_pool": "sampled target+random negatives",
			"reference_removed": True,
			"query_embedding_mode": "n/a_reranker",
			"runtime": runtime_summary,
			"score_type": "dataset_level",
			"candidate_pool_config": {
				"num_random_negatives": pool_cfg.get("num_random_negatives"),
				"configured_pool_size": pool_cfg.get("configured_pool_size"),
				"include_true_target": pool_cfg.get("include_true_target"),
			},
		}

		if dataset == "cirr" and full_metric_name.startswith("cirr_val_"):
			split = str(config.get("datasets", {}).get("cirr", {}).get("split", "val"))
			metric = full_metric_name.removeprefix("cirr_val_")
			protocol["split"] = split
			protocol["candidate_pool"] = "sampled pool from split gallery, reference removed"

			if metric.startswith("latency_seconds_per"):
				protocol["score_type"] = "efficiency"
			elif metric == "latency_seconds":
				protocol["score_type"] = "system_latency"
			elif metric in {"num_queries", "avg_candidate_pool_size", "num_random_distractors", "scored_pairs"}:
				protocol["score_type"] = "run_stat"

		elif dataset == "fashioniq" and full_metric_name.startswith("fashioniq_"):
			split = str(config.get("datasets", {}).get("fashioniq", {}).get("split", "val"))
			metric = full_metric_name.removeprefix("fashioniq_")
			protocol["split"] = split
			protocol["candidate_pool"] = "class-specific sampled pool from split gallery, reference removed"

			if "avg_recall_at" in metric:
				protocol["score_type"] = "macro_average"
			elif metric.endswith("latency_seconds_per_query") or metric.endswith("latency_seconds_per_scored_pair"):
				protocol["score_type"] = "efficiency"
			elif metric.endswith("latency_seconds"):
				protocol["score_type"] = "system_latency"
			elif metric.endswith("scored_pairs") or metric in {
				"num_queries",
				"avg_candidate_pool_size",
				"num_random_distractors",
				"scored_pairs",
			}:
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


def build_protocol_validation_summary(
	dataset: str,
	config: dict[str, Any],
	runtime_context: dict[str, Any],
) -> dict[str, Any]:
	summary: dict[str, Any] = {
		"validated": True,
		"mode": "standalone_rerank",
		"runtime": runtime_context,
		"dataset": dataset,
		"candidate_pool": {
			"type": "target_plus_random_negatives",
			"source": "split_gallery",
			"reference_removed": True,
			"num_random_negatives": runtime_context["candidate_pool"].get("num_random_negatives"),
		},
	}

	if dataset == "cirr":
		summary["evaluation"] = {
			"split": config.get("datasets", {}).get("cirr", {}).get("split", "val"),
			"metrics": ["R@1", "R@5", "R@10"],
		}
	elif dataset == "fashioniq":
		fashioniq_cfg = config.get("datasets", {}).get("fashioniq", {})
		summary["evaluation"] = {
			"split": fashioniq_cfg.get("split", "val"),
			"protocol": fashioniq_cfg.get("eval_protocol", "val_split"),
			"classes": fashioniq_cfg.get("classes", ["dress", "shirt", "toptee"]),
			"metrics": ["R@1", "R@5", "R@10", "Avg R@1/5/10"],
		}

	return summary


def select_main_metrics(dataset: str, metrics: dict[str, float]) -> dict[str, float]:
	ordered_keys: list[str]
	if dataset == "cirr":
		ordered_keys = [
			"recall_at1",
			"recall_at5",
			"recall_at10",
			"latency_seconds",
			"latency_seconds_per_query",
			"latency_seconds_per_scored_pair",
		]
	else:
		ordered_keys = sorted(
			[key for key in metrics.keys() if key.endswith("avg_recall_at5") or key.endswith("avg_recall_at10") or key.endswith("avg_recall_at50")]
		)
		ordered_keys.extend([
			"latency_seconds",
			"latency_seconds_per_query",
			"latency_seconds_per_scored_pair",
		])

	summary: dict[str, float] = {}
	for key in ordered_keys:
		if key in metrics:
			summary[key] = float(metrics[key])
	return summary


def main() -> None:
	parser = argparse.ArgumentParser(description="Run standalone reranker evaluation.")
	parser.add_argument("--config", type=str, default="configs/reranker/qwen3vl_reranker_2b.yaml", help="Path to reranker YAML config.")
	parser.add_argument("--dataset", type=str, default="", choices=["cirr", "fashioniq"], help="Dataset override. If omitted, inferred from config enabled datasets.")
	parser.add_argument("--output_dir", type=str, default="", help="Optional output directory override.")
	args = parser.parse_args()

	config = load_yaml_config(args.config)
	dataset = resolve_dataset_name(config=config, dataset_override=args.dataset or None)
	outputs_cfg = config.get("outputs", {})
	root_output_dir = args.output_dir or str(outputs_cfg.get("root_dir", "results"))
	run_name = f"{config.get('experiment_name', 'qwen3vl-reranker')}-{dataset}-standalone"
	run_output_dir = os.path.join(root_output_dir, run_name)
	os.makedirs(run_output_dir, exist_ok=True)

	reranker = build_reranker_from_config(args.config)
	metrics_raw = run_standalone_evaluation(dataset=dataset, reranker=reranker, config=config)
	metrics = build_prefixed_metrics(dataset=dataset, metrics=metrics_raw)

	save_to_csv(metrics, os.path.join(run_output_dir, "metrics.csv"))
	save_to_json(metrics, os.path.join(run_output_dir, "metrics.json"))
	save_to_json(metrics_raw, os.path.join(run_output_dir, "metrics_raw.json"))

	runtime_context = build_runtime_context(
		dataset=dataset,
		config=config,
		args=args,
		run_output_dir=run_output_dir,
		resolved_device_override=str(reranker.device),
	)
	save_to_json(runtime_context, os.path.join(run_output_dir, "evaluation_runtime.json"))

	structured_metrics = build_structured_metrics(
		dataset=dataset,
		prefixed_metrics=metrics,
		runtime_context=runtime_context,
		config=config,
	)
	save_records_to_csv(
		structured_metrics,
		os.path.join(run_output_dir, "metrics_structured.csv"),
		fieldnames=["dataset", "split", "metric", "value", "protocol"],
	)
	save_to_json(structured_metrics, os.path.join(run_output_dir, "metrics_structured.json"))

	protocol_validation = build_protocol_validation_summary(
		dataset=dataset,
		config=config,
		runtime_context=runtime_context,
	)
	save_to_json(protocol_validation, os.path.join(run_output_dir, "evaluation_protocol.json"))

	save_to_json(config, os.path.join(run_output_dir, "run_config.json"))

	model_path = config.get("model", {}).get("model_name_or_path", "")
	pool_cfg = config.get("standalone_candidate_pool", {})
	configured_pool_size = int(pool_cfg.get("num_random_negatives", 31)) + (
		1 if bool(pool_cfg.get("include_true_target", True)) else 0
	)
	main_metrics = select_main_metrics(dataset=dataset, metrics=metrics_raw)

	print("==== Standalone Reranker Summary ====")
	print(f"dataset: {dataset}")
	print(f"checkpoint_path: {model_path}")
	print(f"candidate_pool_size_configured: {configured_pool_size}")
	print("main_metrics:")
	if not main_metrics:
		print("  (no main metrics found)")
	else:
		for metric_name, metric_value in main_metrics.items():
			print(f"  {metric_name}: {metric_value:.4f}")
	print(f"output_dir: {run_output_dir}")


if __name__ == "__main__":
	main()

