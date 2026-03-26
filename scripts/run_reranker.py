import argparse
from datetime import datetime, timezone
import os
import platform
import sys
from typing import Any

import torch

from src.evaluation.cirr_rerank_eval import evaluate_cirr_standalone_rerank
from src.evaluation.fashioniq_rerank_eval import evaluate_fashioniq_standalone_rerank
from src.rerankers.lamra_rank import LamRARanker
from src.utils.io import save_to_csv, save_to_json


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
	reranker: LamRARanker,
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
			use_tqdm=True,
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
			use_tqdm=True,
		)

	raise ValueError(f"Unsupported dataset '{dataset}'. Supported: cirr, fashioniq.")


def build_runtime_context(
	dataset: str,
	config: dict[str, Any],
	args: argparse.Namespace,
) -> dict[str, Any]:
	model_cfg = config.get("model", {})
	runtime_cfg = config.get("runtime", {})
	candidate_cfg = config.get("standalone_candidate_pool", {})

	requested_device = str(runtime_cfg.get("device", "auto"))
	if requested_device == "auto":
		resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
	else:
		resolved_device = requested_device

	return {
		"generated_at_utc": datetime.now(timezone.utc).isoformat(),
		"runtime": {
			"python_version": sys.version.split()[0],
			"platform": platform.platform(),
			"torch_version": torch.__version__,
			"cuda_version": torch.version.cuda,
			"cudnn_version": torch.backends.cudnn.version(),
		},
		"run": {
			"dataset": dataset,
			"config_path": args.config,
			"output_dir": args.output_dir,
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
		},
		"candidate_pool": {
			"mode": config.get("pipeline", {}).get("mode", "standalone"),
			"include_true_target": candidate_cfg.get("include_true_target", True),
			"num_random_negatives": int(candidate_cfg.get("num_random_negatives", 31)),
			"configured_pool_size": int(candidate_cfg.get("num_random_negatives", 31))
			+ (1 if bool(candidate_cfg.get("include_true_target", True)) else 0),
		},
	}


def select_main_metrics(dataset: str, metrics: dict[str, float]) -> dict[str, float]:
	ordered_keys: list[str]
	if dataset == "cirr":
		ordered_keys = ["recall_at1", "recall_at5", "recall_at10"]
	else:
		ordered_keys = sorted(
			[key for key in metrics.keys() if key.endswith("avg_recall_at5") or key.endswith("avg_recall_at10") or key.endswith("avg_recall_at50")]
		)

	summary: dict[str, float] = {}
	for key in ordered_keys:
		if key in metrics:
			summary[key] = float(metrics[key])
	return summary


def main() -> None:
	parser = argparse.ArgumentParser(description="Run standalone LamRA reranker evaluation.")
	parser.add_argument("--config", type=str, default="configs/reranker/lamra_rank.yaml", help="Path to reranker YAML config.")
	parser.add_argument("--dataset", type=str, default="", choices=["cirr", "fashioniq"], help="Dataset override. If omitted, inferred from config enabled datasets.")
	parser.add_argument("--output_dir", type=str, default="", help="Optional output directory override.")
	args = parser.parse_args()

	config = load_yaml_config(args.config)
	dataset = resolve_dataset_name(config=config, dataset_override=args.dataset or None)

	reranker = LamRARanker.from_config(args.config)
	metrics = run_standalone_evaluation(dataset=dataset, reranker=reranker, config=config)

	outputs_cfg = config.get("outputs", {})
	root_output_dir = args.output_dir or str(outputs_cfg.get("root_dir", "results"))
	run_name = f"{config.get('experiment_name', 'lamra-reranker')}-{dataset}-standalone"
	run_output_dir = os.path.join(root_output_dir, run_name)
	os.makedirs(run_output_dir, exist_ok=True)

	save_to_csv(metrics, os.path.join(run_output_dir, "metrics.csv"))
	save_to_json(metrics, os.path.join(run_output_dir, "metrics.json"))

	runtime_context = build_runtime_context(dataset=dataset, config=config, args=args)
	save_to_json(runtime_context, os.path.join(run_output_dir, "evaluation_runtime.json"))
	save_to_json(config, os.path.join(run_output_dir, "run_config.json"))

	model_path = config.get("model", {}).get("model_name_or_path", "")
	pool_cfg = config.get("standalone_candidate_pool", {})
	configured_pool_size = int(pool_cfg.get("num_random_negatives", 31)) + (
		1 if bool(pool_cfg.get("include_true_target", True)) else 0
	)
	main_metrics = select_main_metrics(dataset=dataset, metrics=metrics)

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

