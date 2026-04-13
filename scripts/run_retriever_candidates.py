from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
import platform
from pathlib import Path
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.chain.candidate_io import load_candidate_records, save_candidate_records
from src.chain.cirr_chain import build_cirr_candidate_records
from src.chain.fashioniq_chain import build_fashioniq_candidate_records
from src.retrievers.base import TwoEncoderVLM
from src.utils.io import save_to_csv, save_to_json


FASHIONIQ_STUDY_SETTINGS = {"default", "caption_ablation_v1"}


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
	if study_setting not in FASHIONIQ_STUDY_SETTINGS:
		supported = ", ".join(sorted(FASHIONIQ_STUDY_SETTINGS))
		raise ValueError(f"Unsupported fashioniq_study_setting '{study_setting}'. Supported: {supported}")

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
			f"Could not import retriever module '{retriever_module}'. "
			"Create this module under src/retrievers before running candidate generation."
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


def _resolve_runtime_context(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
	cuda_available = bool(torch.cuda.is_available())
	cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
	cuda_device_index: int | None = None
	cuda_device_name: str | None = None

	if device.type == "cuda" and cuda_available:
		cuda_device_index = device.index if device.index is not None else int(torch.cuda.current_device())
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
			"requested": args.device,
			"resolved": str(device),
			"type": device.type,
			"cuda_available": cuda_available,
			"cuda_device_count": cuda_device_count,
			"cuda_device_index": cuda_device_index,
			"cuda_device_name": cuda_device_name,
		},
		"stage": {
			"name": "retriever_candidates",
			"datasets": list(args.datasets),
			"top_m": int(args.top_m),
			"max_queries": args.max_queries,
			"batch_size": int(args.batch_size),
			"num_workers": int(args.num_workers),
			"query_embedding_mode": args.query_embedding_mode,
			"fusion_type": args.fusion_type,
			"fashioniq_eval_protocol": args.fashioniq_eval_protocol,
			"fashioniq_study_setting": args.fashioniq_study_setting,
			"fashioniq_caption_joiner_mode": args.fashioniq_caption_joiner_mode,
			"fashioniq_caption_order_mode": args.fashioniq_caption_order_mode,
			"tqdm": bool(args.tqdm),
		},
		"retriever": {
			"module": args.retriever_module,
			"class": args.retriever_class,
			"model_name_or_path": args.model_name_or_path,
			"checkpoint_path": args.checkpoint_path,
			"init_kwargs": json.loads(args.retriever_init_kwargs),
		},
	}


def _build_protocol_summary(args: argparse.Namespace) -> dict[str, Any]:
	summary: dict[str, Any] = {
		"validated": True,
		"mode": "stage_a_retriever_candidates",
		"candidate_depth": int(args.top_m),
		"reference_removed": True,
		"datasets": {},
	}

	if "cirr" in args.datasets:
		summary["datasets"]["cirr"] = {
			"split": args.cirr_split,
			"candidate_pool": "retriever top-m from full validation gallery, reference removed",
			"query_id": "cirr:{split}:{pair_id}",
			"metrics_ready_for_chain": ["coverage@M", "rerank_recall_at1", "rerank_recall_at5", "rerank_recall_at10", "rerank_recall_at50"],
		}

	if "fashioniq" in args.datasets:
		split = "val" if args.fashioniq_eval_protocol == "val_split" else "test"
		summary["datasets"]["fashioniq"] = {
			"split": split,
			"eval_protocol": args.fashioniq_eval_protocol,
			"candidate_pool": "retriever top-m from class-specific gallery, reference removed",
			"query_id": "fashioniq:{split}:{class}:{index}",
			"metrics_ready_for_chain": ["coverage@M", "rerank_recall_at5", "rerank_recall_at10", "rerank_recall_at50"],
		}

	return summary


def _merge_args_with_config(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
	retriever_cfg = config.get("retriever", {})
	runtime_cfg = config.get("runtime", {})
	chain_cfg = config.get("chain", {})
	datasets_cfg = config.get("datasets", {})
	outputs_cfg = config.get("outputs", {})

	if not args.model_name_or_path:
		args.model_name_or_path = str(retriever_cfg.get("model_name_or_path", ""))
	if not args.retriever_module:
		args.retriever_module = str(retriever_cfg.get("module", "src.retrievers.vista_retriever"))
	if not args.retriever_class:
		args.retriever_class = str(retriever_cfg.get("class", "VistaBGERetriever"))
	if not args.checkpoint_path:
		args.checkpoint_path = str(retriever_cfg.get("checkpoint_path", ""))
	if args.retriever_init_kwargs == "":
		args.retriever_init_kwargs = json.dumps(retriever_cfg.get("init_kwargs", {}))

	if not args.datasets:
		args.datasets = _resolve_enabled_datasets(config=config, datasets_override=None)

	if args.top_m <= 0:
		args.top_m = int(chain_cfg.get("top_m", 100))
	if args.max_queries is None:
		args.max_queries = chain_cfg.get("max_queries")

	if args.batch_size <= 0:
		args.batch_size = int(runtime_cfg.get("batch_size", 64))
	if args.num_workers < 0:
		args.num_workers = int(runtime_cfg.get("num_workers", 4))
	if not args.query_embedding_mode:
		args.query_embedding_mode = str(runtime_cfg.get("query_embedding_mode", "vista_mm"))
	if not args.fusion_type:
		args.fusion_type = str(runtime_cfg.get("fusion_type", "sum"))
	if not args.device:
		args.device = str(runtime_cfg.get("device", "auto"))
	if not args.tqdm:
		args.tqdm = bool(runtime_cfg.get("tqdm", False))

	cirr_cfg = datasets_cfg.get("cirr", {})
	fashioniq_cfg = datasets_cfg.get("fashioniq", {})
	if not args.cirr_split:
		args.cirr_split = str(cirr_cfg.get("split", "val"))
	if not args.fashioniq_eval_protocol:
		args.fashioniq_eval_protocol = str(fashioniq_cfg.get("eval_protocol", "val_split"))
	if not args.fashioniq_study_setting:
		args.fashioniq_study_setting = str(fashioniq_cfg.get("study_setting", "default"))
	if not args.fashioniq_caption_joiner_mode:
		args.fashioniq_caption_joiner_mode = str(fashioniq_cfg.get("caption_joiner_mode", "protocol_default"))
	if not args.fashioniq_caption_order_mode:
		args.fashioniq_caption_order_mode = str(fashioniq_cfg.get("caption_order_mode", "original"))

	if not args.output_path:
		args.output_path = str(outputs_cfg.get("root_dir", "results"))
	if not args.run_name:
		args.run_name = str(outputs_cfg.get("run_name", f"{args.retriever_class}-candidates-top{args.top_m}"))

	if not args.model_name_or_path:
		raise ValueError("model_name_or_path must be provided either via CLI or config.retriever.model_name_or_path")

	return args


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate Stage-A retriever top-M candidate lists for chain reranking.")
	parser.add_argument("--config", type=str, default="configs/chain/vista_lamra_chain.yaml", help="Path to chain YAML config.")

	parser.add_argument("--model_name_or_path", type=str, default="", help="Base model name or local path.")
	parser.add_argument("--retriever_module", type=str, default="", help="Python module containing retriever class.")
	parser.add_argument("--retriever_class", type=str, default="", help="Retriever class name inside retriever module.")
	parser.add_argument("--retriever_init_kwargs", type=str, default="", help="Extra retriever kwargs in JSON format.")
	parser.add_argument("--checkpoint_path", type=str, default="", help="Retriever checkpoint path.")

	parser.add_argument("--datasets", nargs="+", default=[], choices=["cirr", "fashioniq"], help="Datasets to generate candidates for.")
	parser.add_argument("--cirr_split", type=str, default="", help="CIRR split override (default from config).")
	parser.add_argument("--fashioniq_eval_protocol", type=str, default="", choices=["val_split", "original_split"], help="FashionIQ eval protocol override.")
	parser.add_argument("--fashioniq_study_setting", type=str, default="", choices=["default", "caption_ablation_v1"], help="FashionIQ study preset.")
	parser.add_argument("--fashioniq_caption_joiner_mode", type=str, default="", choices=["protocol_default", "space", "and"], help="FashionIQ caption joiner mode.")
	parser.add_argument("--fashioniq_caption_order_mode", type=str, default="", choices=["original", "both"], help="FashionIQ caption order mode.")

	parser.add_argument("--top_m", type=int, default=-1, help="Retriever top-M depth to save for each query.")
	parser.add_argument("--max_queries", type=int, default=None, help="Optional cap on number of queries per dataset.")
	parser.add_argument("--batch_size", type=int, default=-1, help="Evaluation batch size.")
	parser.add_argument("--num_workers", type=int, default=-1, help="DataLoader workers.")
	parser.add_argument("--query_embedding_mode", type=str, default="", choices=["vista_mm", "legacy_fusion"], help="Retriever query embedding path.")
	parser.add_argument("--fusion_type", type=str, default="", help="Fusion type for legacy_fusion mode.")
	parser.add_argument("--device", type=str, default="", help="Device override: auto, cpu, cuda, cuda:0, etc.")
	parser.add_argument("--tqdm", action="store_true", help="Enable tqdm progress bars.")

	parser.add_argument("--output_path", type=str, default="", help="Directory where outputs are saved.")
	parser.add_argument("--run_name", type=str, default="", help="Optional run folder name override.")
	args = parser.parse_args()

	config = load_yaml_config(args.config)
	args = _merge_args_with_config(args=args, config=config)

	effective_joiner_mode, effective_order_mode = resolve_fashioniq_study_setting(
		args.fashioniq_study_setting,
		args.fashioniq_caption_joiner_mode,
		args.fashioniq_caption_order_mode,
	)
	caption_joiner_override = resolve_caption_joiner_override(effective_joiner_mode)

	device = resolve_device(args.device)
	init_kwargs: dict[str, Any] = json.loads(args.retriever_init_kwargs or "{}")
	if args.checkpoint_path:
		init_kwargs["checkpoint_path"] = args.checkpoint_path

	model = load_retriever(
		retriever_module=args.retriever_module,
		retriever_class=args.retriever_class,
		model_name_or_path=args.model_name_or_path,
		device=device,
		init_kwargs=init_kwargs,
	)

	run_output_dir = os.path.join(args.output_path, args.run_name)
	os.makedirs(run_output_dir, exist_ok=True)

	summary_metrics: dict[str, float] = {}
	candidate_file_map: dict[str, str] = {}

	if "cirr" in args.datasets:
		cirr_records, cirr_stats = build_cirr_candidate_records(
			model=model,
			split=args.cirr_split,
			top_m=args.top_m,
			query_embedding_mode=args.query_embedding_mode,
			fusion_type=args.fusion_type,
			batch_size=args.batch_size,
			num_workers=args.num_workers,
			use_tqdm=args.tqdm,
			max_queries=args.max_queries,
		)
		cirr_path = os.path.join(run_output_dir, "cirr_candidates.jsonl")
		save_candidate_records(cirr_records, cirr_path)
		reloaded = load_candidate_records(cirr_path)
		if len(reloaded) != len(cirr_records):
			raise ValueError("CIRR candidate file roundtrip validation failed.")

		candidate_file_map["cirr"] = "cirr_candidates.jsonl"
		summary_metrics.update({f"cirr_{k}": float(v) for k, v in cirr_stats.items()})

	if "fashioniq" in args.datasets:
		fashioniq_records, fashioniq_stats = build_fashioniq_candidate_records(
			model=model,
			top_m=args.top_m,
			query_embedding_mode=args.query_embedding_mode,
			fusion_type=args.fusion_type,
			batch_size=args.batch_size,
			num_workers=args.num_workers,
			use_tqdm=args.tqdm,
			max_queries=args.max_queries,
			eval_protocol=args.fashioniq_eval_protocol,
			caption_joiner_override=caption_joiner_override,
			caption_order_mode=effective_order_mode,
		)
		fashioniq_path = os.path.join(run_output_dir, "fashioniq_candidates.jsonl")
		save_candidate_records(fashioniq_records, fashioniq_path)
		reloaded = load_candidate_records(fashioniq_path)
		if len(reloaded) != len(fashioniq_records):
			raise ValueError("FashionIQ candidate file roundtrip validation failed.")

		candidate_file_map["fashioniq"] = "fashioniq_candidates.jsonl"
		summary_metrics.update({f"fashioniq_{k}": float(v) for k, v in fashioniq_stats.items()})

	runtime_context = _resolve_runtime_context(args=args, device=device)
	runtime_context["stage"]["fashioniq_caption_joiner_mode_effective"] = effective_joiner_mode
	runtime_context["stage"]["fashioniq_caption_order_mode_effective"] = effective_order_mode
	runtime_context["stage"]["candidate_files"] = candidate_file_map

	save_to_json(runtime_context, os.path.join(run_output_dir, "retrieval_runtime.json"))
	save_to_json(_build_protocol_summary(args=args), os.path.join(run_output_dir, "retrieval_protocol.json"))
	save_to_json(summary_metrics, os.path.join(run_output_dir, "candidate_summary.json"))
	save_to_csv(summary_metrics, os.path.join(run_output_dir, "candidate_summary.csv"))

	run_config = vars(args).copy()
	run_config["resolved_device"] = str(device)
	run_config["fashioniq_caption_joiner_mode_effective"] = effective_joiner_mode
	run_config["fashioniq_caption_order_mode_effective"] = effective_order_mode
	run_config["candidate_files"] = candidate_file_map
	save_to_json(run_config, os.path.join(run_output_dir, "run_config.json"))

	print("==== Retriever Candidate Generation Summary ====")
	print(f"run_output_dir: {run_output_dir}")
	print(f"datasets: {args.datasets}")
	print(f"top_m: {args.top_m}")
	for key, value in summary_metrics.items():
		print(f"{key}: {value:.4f}")


if __name__ == "__main__":
	main()
