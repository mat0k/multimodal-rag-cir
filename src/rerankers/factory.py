from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.rerankers.base import BaseReranker
from src.rerankers.lamra_rank import LamRARanker
from src.rerankers.qwen3vl_rank import Qwen3VLRanker


def _load_yaml_config(config_path: str) -> dict[str, Any]:
	path = Path(config_path)
	if not path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")

	try:
		import yaml
	except ModuleNotFoundError as exc:
		raise ModuleNotFoundError(
			"PyYAML is required to load YAML configs. Install with `pip install pyyaml`."
		) from exc

	with path.open("r", encoding="utf-8") as file_obj:
		loaded = yaml.safe_load(file_obj) or {}

	if not isinstance(loaded, dict):
		raise ValueError(f"Config must parse to a mapping, got {type(loaded).__name__}.")
	return loaded


def resolve_reranker_type(config: Mapping[str, Any] | str) -> str:
	config_data = _load_yaml_config(config) if isinstance(config, str) else dict(config)

	reranker_cfg = dict(config_data.get("reranker", {}))
	model_cfg = dict(config_data.get("model", {}))

	raw_type = (
		reranker_cfg.get("type")
		or model_cfg.get("type")
		or config_data.get("type")
		or "lamra"
	)

	normalized = str(raw_type).strip().lower().replace("-", "_")

	lamra_aliases = {"lamra", "lamra_rank", "lamrarank"}
	qwen_aliases = {"qwen3vl", "qwen3_vl", "qwen3vl_2b", "qwen2b", "qwen_2b"}

	if normalized in lamra_aliases:
		return "lamra"
	if normalized in qwen_aliases:
		return "qwen3vl"

	raise ValueError(
		f"Unsupported reranker type '{raw_type}'. Supported values include: lamra, qwen3vl."
	)


def build_reranker_from_config(config: Mapping[str, Any] | str) -> BaseReranker:
	config_data = _load_yaml_config(config) if isinstance(config, str) else dict(config)
	reranker_type = resolve_reranker_type(config_data)

	if reranker_type == "qwen3vl":
		return Qwen3VLRanker.from_config(config_data)

	return LamRARanker.from_config(config_data)
