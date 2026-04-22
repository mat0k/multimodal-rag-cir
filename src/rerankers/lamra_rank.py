from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import (
	AutoTokenizer,
	Qwen2VLImageProcessor,
	Qwen2VLVideoProcessor,
	Qwen2_5_VLForConditionalGeneration,
	Qwen2_5_VLProcessor,
)

from src.rerankers.backbones.lamra.preprocess import (
	load_image_item,
	process_messages_to_inputs,
)
from src.rerankers.backbones.lamra.prompts import build_pointwise_relevance_message
from src.rerankers.base import BaseReranker, RerankCandidate, RerankQuery

try:
	from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - depends on local env setup
	process_vision_info = None


class LamRARanker(BaseReranker):
	"""Generic LamRA reranker wrapper for pointwise multimodal relevance scoring.

	This class is intentionally dataset-agnostic and can be used for:
	- standalone candidate-pool testing
	- two-stage reranking on retriever top-k
	- teacher scoring in distillation pipelines
	"""
	reranker_type = "lamra"

	def __init__(
		self,
		model_name_or_path: str,
		device: str = "auto",
		dtype: str = "bfloat16",
		trust_remote_code: bool = True,
		local_files_only: bool = True,
		revision: str = "main",
		low_cpu_mem_usage: bool = True,
		emb_token: str = "<emb>",
		image_root: str | None = None,
		rgb_only: bool = True,
		resize_short_edge: int | None = None,
		center_crop: int | None = None,
		interpolation: str = "bicubic",
	) -> None:
		self.model_name_or_path = model_name_or_path
		self.device = self._resolve_device(device)
		self.dtype = self._resolve_dtype(dtype)
		self.trust_remote_code = trust_remote_code
		self.local_files_only = local_files_only
		self.revision = revision
		self.low_cpu_mem_usage = low_cpu_mem_usage
		self.emb_token = emb_token
		self.image_root = image_root
		self.rgb_only = rgb_only
		self.resize_short_edge = resize_short_edge
		self.center_crop = center_crop
		self.interpolation = interpolation

		self.model: Qwen2_5_VLForConditionalGeneration | None = None
		self.processor: Any = None
		self.tokenizer: Any = None
		self.emb_token_id: int | None = None

		self.load_model()

	@classmethod
	def from_config(cls, config: Mapping[str, Any] | str) -> "LamRARanker":
		"""Build a ranker from a dict-like config or YAML file path."""
		if isinstance(config, str):
			config_data = cls._load_config_file(config)
		else:
			config_data = dict(config)

		model_cfg = dict(config_data.get("model", {}))
		runtime_cfg = dict(config_data.get("runtime", {}))
		image_cfg = dict(config_data.get("image_io", {}))

		model_name_or_path = (
			model_cfg.get("checkpoint_path")
			or model_cfg.get("model_name_or_path")
			or model_cfg.get("name")
		)
		if not model_name_or_path:
			raise ValueError("Config must define model.checkpoint_path (or model.model_name_or_path/model.name).")

		return cls(
			model_name_or_path=str(model_name_or_path),
			device=str(runtime_cfg.get("device", "auto")),
			dtype=str(runtime_cfg.get("dtype", "bfloat16")),
			trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
			local_files_only=bool(model_cfg.get("local_files_only", True)),
			revision=str(model_cfg.get("revision", "main")),
			image_root=str(image_cfg.get("image_root")) if image_cfg.get("image_root") else None,
			rgb_only=bool(image_cfg.get("rgb_only", True)),
			resize_short_edge=(
				int(image_cfg["resize_short_edge"])
				if image_cfg.get("resize_short_edge") is not None
				else None
			),
			center_crop=(
				int(image_cfg["center_crop"])
				if image_cfg.get("center_crop") is not None
				else None
			),
			interpolation=str(image_cfg.get("interpolation", "bicubic")),
		)

	def load_model(self) -> None:
		"""Load processor + LamRA model from Hugging Face/local checkpoint."""
		self._validate_checkpoint_layout(self.model_name_or_path)

		self.processor = self._load_processor()
		self._ensure_chat_template(self.processor, self.model_name_or_path)
		self.tokenizer = self.processor.tokenizer

		self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
			self.model_name_or_path,
			torch_dtype=self.dtype,
			low_cpu_mem_usage=self.low_cpu_mem_usage,
			trust_remote_code=self.trust_remote_code,
			local_files_only=self.local_files_only,
			revision=self.revision,
		).to(self.device)
		self.model.eval()

		self.emb_token_id = self._ensure_embed_token(self.emb_token)

	@staticmethod
	def _ensure_chat_template(processor: Qwen2_5_VLProcessor, model_name_or_path: str) -> None:
		"""Ensure chat template is attached to processor/tokenizer for apply_chat_template."""
		local_path = Path(model_name_or_path)
		if not local_path.exists() or not local_path.is_dir():
			return

		template_text: str | None = None
		chat_template_file = local_path / "chat_template.json"
		if chat_template_file.exists():
			with chat_template_file.open("r", encoding="utf-8") as file_obj:
				payload = json.load(file_obj)
				if isinstance(payload, dict):
					template_text = payload.get("chat_template")

		if not template_text:
			tokenizer_cfg_file = local_path / "tokenizer_config.json"
			if tokenizer_cfg_file.exists():
				with tokenizer_cfg_file.open("r", encoding="utf-8") as file_obj:
					payload = json.load(file_obj)
					if isinstance(payload, dict):
						template_text = payload.get("chat_template")

		if template_text:
			processor.chat_template = template_text
			processor.tokenizer.chat_template = template_text

	def _load_processor(self) -> Qwen2_5_VLProcessor:
		"""Load Qwen2.5-VL processor with robust local-checkpoint fallback.

		Primary path uses Qwen2_5_VLProcessor.from_pretrained.
		Fallback path manually builds processor with tokenizer + image processor + video processor,
		which is required for some checkpoint/transformers combinations.
		"""
		try:
			return Qwen2_5_VLProcessor.from_pretrained(
				self.model_name_or_path,
				trust_remote_code=self.trust_remote_code,
				local_files_only=self.local_files_only,
				revision=self.revision,
				use_fast=False,
			)
		except Exception as primary_exc:
			# Fallback: explicitly provide all processor subcomponents, including video processor.
			try:
				tokenizer = AutoTokenizer.from_pretrained(
					self.model_name_or_path,
					trust_remote_code=self.trust_remote_code,
					local_files_only=self.local_files_only,
					revision=self.revision,
				)
				image_processor = Qwen2VLImageProcessor.from_pretrained(
					self.model_name_or_path,
					trust_remote_code=self.trust_remote_code,
					local_files_only=self.local_files_only,
					revision=self.revision,
				)
				video_processor = Qwen2VLVideoProcessor.from_pretrained(
					self.model_name_or_path,
					trust_remote_code=self.trust_remote_code,
					local_files_only=self.local_files_only,
					revision=self.revision,
				)
				return Qwen2_5_VLProcessor(
					tokenizer=tokenizer,
					image_processor=image_processor,
					video_processor=video_processor,
				)
			except Exception as fallback_exc:
				raise RuntimeError(
					"Failed to load Qwen2.5-VL processor from local checkpoint. "
					f"Primary error: {primary_exc}. Fallback error: {fallback_exc}"
				) from fallback_exc

	@staticmethod
	def _validate_checkpoint_layout(model_name_or_path: str) -> None:
		"""Validate local HF checkpoint layout when a local directory is provided.

		Remote model IDs are accepted as-is and validated by transformers loaders.
		"""
		local_path = Path(model_name_or_path)
		if not local_path.exists():
			raise FileNotFoundError(
				"LamRA loading is configured for local checkpoints only, but path was not found: "
				f"{local_path}"
			)

		if not local_path.is_dir():
			raise ValueError(f"model_name_or_path must point to a directory, got: {local_path}")

		required_files = [
			"config.json",
			"preprocessor_config.json",
			"model.safetensors.index.json",
		]
		missing_required = [name for name in required_files if not (local_path / name).exists()]
		if missing_required:
			raise FileNotFoundError(
				"Local checkpoint is missing required Hugging Face files: "
				f"{missing_required}. Expected directory: {local_path}"
			)

		has_tokenizer = any(
			(local_path / name).exists()
			for name in ["tokenizer.json", "tokenizer_config.json", "vocab.json"]
		)
		if not has_tokenizer:
			raise FileNotFoundError(
				"Local checkpoint is missing tokenizer files (expected one of tokenizer.json, "
				"tokenizer_config.json, vocab.json)."
			)

		# Ensure sharded safetensors files referenced by index are physically present.
		shard_files = sorted(local_path.glob("model-*.safetensors"))
		if not shard_files:
			raise FileNotFoundError(
				"No sharded model safetensors found (expected files like model-00001-of-00004.safetensors)."
			)

	def score(self, query: RerankQuery, candidate: RerankCandidate) -> float:
		"""Return pointwise relevance score for one query-candidate pair.

		Score is computed as P(yes) from model next-token logits for a joint
		reference+text-edit+candidate input.
		"""
		if self.model is None or self.processor is None or self.emb_token_id is None:
			raise RuntimeError("LamRARanker is not initialized. Call load_model() first.")
		model_device = next(self.model.parameters()).device

		reference_image = load_image_item(
			query.reference_image,
			image_root=self.image_root,
			rgb_only=self.rgb_only,
			resize_short_edge=self.resize_short_edge,
			center_crop=self.center_crop,
			interpolation=self.interpolation,
		)
		candidate_image = load_image_item(
			candidate.image,
			image_root=self.image_root,
			rgb_only=self.rgb_only,
			resize_short_edge=self.resize_short_edge,
			center_crop=self.center_crop,
			interpolation=self.interpolation,
		)
		joint_messages = [
			build_pointwise_relevance_message(
				reference_image=reference_image,
				text_edit=query.text_edit,
				candidate_image=candidate_image,
			)
		]

		joint_inputs = process_messages_to_inputs(
			processor=self.processor,
			messages=joint_messages,
			device=self.device,
			move_to_device=False,
		)

		joint_inputs = joint_inputs.to(model_device)

		yes_token_id = self._get_single_token_id(" yes")
		no_token_id = self._get_single_token_id(" no")

		with torch.no_grad():
			logits = self.model(
				**joint_inputs,
				return_dict=True,
			).logits
			next_token_logits = logits[:, -1, :]
			yes_logit = next_token_logits[:, yes_token_id]
			no_logit = next_token_logits[:, no_token_id]
			binary_logits = torch.stack([no_logit, yes_logit], dim=1)
			prob_yes = torch.softmax(binary_logits, dim=1)[:, 1]
		return float(prob_yes.item())

	def score_batch(self, query: RerankQuery, candidates: Sequence[RerankCandidate]) -> list[float]:
		"""Return continuous relevance scores aligned with input candidate order."""
		return [self.score(query=query, candidate=candidate) for candidate in candidates]

	@staticmethod
	def _load_config_file(config_path: str) -> dict[str, Any]:
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

	@staticmethod
	def _resolve_device(device: str) -> torch.device:
		if device == "auto":
			return torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if device.startswith("cuda") and not torch.cuda.is_available():
			raise RuntimeError(
				"Configuration requested CUDA device, but CUDA is not available on this runtime. "
				"Set runtime.device=cpu or run on a CUDA-enabled machine."
			)
		return torch.device(device)

	@staticmethod
	def _resolve_dtype(dtype: str) -> torch.dtype:
		mapping = {
			"float32": torch.float32,
			"fp32": torch.float32,
			"float16": torch.float16,
			"fp16": torch.float16,
			"bfloat16": torch.bfloat16,
			"bf16": torch.bfloat16,
		}
		key = dtype.lower()
		if key not in mapping:
			supported = ", ".join(sorted(mapping.keys()))
			raise ValueError(f"Unsupported dtype '{dtype}'. Supported values: {supported}")
		return mapping[key]

	def _ensure_embed_token(self, emb_token: str) -> int:
		if self.tokenizer is None or self.model is None:
			raise RuntimeError("Tokenizer/model must be initialized before adding embedding token.")

		token_id = self.tokenizer.convert_tokens_to_ids(emb_token)
		if token_id is not None and token_id != self.tokenizer.unk_token_id:
			self.model.config.emb_token_ids = [int(token_id)]
			return int(token_id)

		num_added = self.tokenizer.add_tokens([emb_token])
		if num_added != 1:
			raise ValueError(f"Failed to add embedding token '{emb_token}'.")

		self.model.resize_token_embeddings(len(self.tokenizer))
		token_id = self.tokenizer.convert_tokens_to_ids(emb_token)
		if token_id is None:
			raise ValueError(f"Embedding token id not found for '{emb_token}'.")

		self.model.config.emb_token_ids = [int(token_id)]
		return int(token_id)

	def _get_single_token_id(self, token_text: str) -> int:
		if self.tokenizer is None:
			raise RuntimeError("Tokenizer is not initialized.")

		token_ids = self.tokenizer.encode(token_text, add_special_tokens=False)
		if len(token_ids) != 1:
			raise ValueError(
				f"Expected token_text '{token_text}' to map to one token, got ids={token_ids}."
			)
		return int(token_ids[0])



