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

	def __init__(
		self,
		model_name_or_path: str,
		device: str = "auto",
		dtype: str = "bfloat16",
		quantization_mode: str = "none",
		quantization_compute_dtype: str = "float16",
		quantization_double_quant: bool = True,
		quantization_cpu_offload: bool = False,
		trust_remote_code: bool = True,
		local_files_only: bool = True,
		revision: str = "main",
		low_cpu_mem_usage: bool = True,
		emb_token: str = "<emb>",
	) -> None:
		self.model_name_or_path = model_name_or_path
		self.device = self._resolve_device(device)
		print(f"DEBUG_LAMRA_INIT: resolved runtime device={self.device}", flush=True)
		self.dtype = self._resolve_dtype(dtype)
		self.quantization_mode = quantization_mode.lower()
		self.quantization_compute_dtype = self._resolve_dtype(quantization_compute_dtype)
		self.quantization_double_quant = quantization_double_quant
		self.quantization_cpu_offload = quantization_cpu_offload
		self.trust_remote_code = trust_remote_code
		self.local_files_only = local_files_only
		self.revision = revision
		self.low_cpu_mem_usage = low_cpu_mem_usage
		self.emb_token = emb_token

		valid_modes = {"none", "8bit", "4bit", "auto"}
		if self.quantization_mode not in valid_modes:
			raise ValueError(
				f"Unsupported quantization_mode '{quantization_mode}'. Supported: {sorted(valid_modes)}"
			)

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
			quantization_mode=str(runtime_cfg.get("quantization_mode", "none")),
			quantization_compute_dtype=str(runtime_cfg.get("quantization_compute_dtype", "float16")),
			quantization_double_quant=bool(runtime_cfg.get("quantization_double_quant", True)),
			quantization_cpu_offload=bool(runtime_cfg.get("quantization_cpu_offload", False)),
			trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
			local_files_only=bool(model_cfg.get("local_files_only", True)),
			revision=str(model_cfg.get("revision", "main")),
		)

	def load_model(self) -> None:
		"""Load processor + LamRA model from Hugging Face/local checkpoint."""
		self._validate_checkpoint_layout(self.model_name_or_path)

		self.processor = self._load_processor()
		self._ensure_chat_template(self.processor, self.model_name_or_path)
		self.tokenizer = self.processor.tokenizer

		base_model_kwargs: dict[str, Any] = {
			"low_cpu_mem_usage": self.low_cpu_mem_usage,
			"trust_remote_code": self.trust_remote_code,
			"local_files_only": self.local_files_only,
			"revision": self.revision,
		}

		if self.quantization_mode == "none":
			self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
				self.model_name_or_path,
				torch_dtype=self.dtype,
				**base_model_kwargs,
			).to(self.device)
		else:
			self.model = self._load_model_with_quantization(base_model_kwargs)
		self.model.eval()

		self.emb_token_id = self._ensure_embed_token(self.emb_token)

	def _load_model_with_quantization(
		self,
		base_model_kwargs: Mapping[str, Any],
	) -> Qwen2_5_VLForConditionalGeneration:
		"""Load model with 8bit/4bit quantization and fallback attempts."""
		try:
			from transformers import BitsAndBytesConfig
		except Exception as exc:
			raise RuntimeError(
				"Quantization mode requested, but BitsAndBytesConfig is unavailable. "
				"Install compatible bitsandbytes/transformers/accelerate packages."
			) from exc

		attempts = ["8bit", "4bit"] if self.quantization_mode == "auto" else [self.quantization_mode]
		load_errors: list[str] = []

		for attempt in attempts:
			try:
				if attempt == "8bit":
					quantization_config = BitsAndBytesConfig(
						load_in_8bit=True,
						llm_int8_enable_fp32_cpu_offload=self.quantization_cpu_offload,
					)
				elif attempt == "4bit":
					quantization_config = BitsAndBytesConfig(
						load_in_4bit=True,
						bnb_4bit_quant_type="nf4",
						bnb_4bit_compute_dtype=self.quantization_compute_dtype,
						bnb_4bit_use_double_quant=self.quantization_double_quant,
					)
				else:
					raise ValueError(f"Unsupported quantization attempt '{attempt}'.")

				device_map = self._resolve_quantized_device_map()
				model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
					self.model_name_or_path,
					quantization_config=quantization_config,
					device_map=device_map,
					torch_dtype=self.quantization_compute_dtype,
					**dict(base_model_kwargs),
				)
				print(
					f"DEBUG_LAMRA_INIT: quantized model load succeeded with mode={attempt}, device_map={device_map}",
					flush=True,
				)
				self.quantization_mode = attempt
				return model
			except Exception as exc:
				load_errors.append(f"{attempt}: {exc}")
				print(
					f"DEBUG_LAMRA_INIT: quantized load failed for mode={attempt}; trying next fallback if available",
					flush=True,
				)

		raise RuntimeError(
			"Failed to load quantized LamRA model. Attempt errors: " + " | ".join(load_errors)
		)

	def _resolve_quantized_device_map(self) -> Any:
		if self.device.type == "cuda":
			if self.quantization_cpu_offload:
				return "auto"
			return {"": int(self.device.index or 0)}
		if self.device.type == "cpu":
			return {"": "cpu"}
		return "auto"

	def _resolve_model_device(self) -> torch.device:
		if self.model is None:
			return self.device

		model_device = getattr(self.model, "device", None)
		if isinstance(model_device, torch.device):
			return model_device

		for param in self.model.parameters():
			if param.device.type != "meta":
				return param.device
		return self.device

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

		print("DEBUG_LAMRA_SCORE: 1 entering score(...)", flush=True)
		model_device = self._resolve_model_device()
		print(f"DEBUG_LAMRA_SCORE: model device={model_device}", flush=True)

		print("DEBUG_LAMRA_SCORE: 2 building joint prompt/message", flush=True)
		reference_image = load_image_item(query.reference_image)
		candidate_image = load_image_item(candidate.image)
		joint_messages = [
			build_pointwise_relevance_message(
				reference_image=reference_image,
				text_edit=query.text_edit,
				candidate_image=candidate_image,
			)
		]
		print("DEBUG_LAMRA_SCORE: 3 prompt/message built", flush=True)

		print("DEBUG_LAMRA_SCORE: 4 starting preprocessing", flush=True)
		joint_inputs = process_messages_to_inputs(
			processor=self.processor,
			messages=joint_messages,
			device=self.device,
			move_to_device=False,
		)
		print("DEBUG_LAMRA_SCORE: 5 preprocessing finished", flush=True)

		print("DEBUG_LAMRA_SCORE: 6 moving tensors to device", flush=True)
		joint_inputs = joint_inputs.to(model_device)
		input_ids = joint_inputs.get("input_ids")
		if input_ids is not None:
			print(
				f"DEBUG_LAMRA_SCORE: input tensor device={input_ids.device}, shape={tuple(input_ids.shape)}",
				flush=True,
			)
		else:
			print("DEBUG_LAMRA_SCORE: input_ids not found in model inputs", flush=True)
		print("DEBUG_LAMRA_SCORE: 7 tensors moved", flush=True)

		yes_token_id = self._get_single_token_id(" yes")
		no_token_id = self._get_single_token_id(" no")

		# Temporary pre-forward diagnostics for device placement and tensor layout.
		param_device = self._resolve_model_device()
		pixel_values = joint_inputs.get("pixel_values")
		shape_info = {
			key: tuple(value.shape)
			for key, value in joint_inputs.items()
			if hasattr(value, "shape")
		}
		print(f"DEBUG_LAMRA_SCORE: pre-forward model device={model_device}", flush=True)
		print(f"DEBUG_LAMRA_SCORE: pre-forward parameter device={param_device}", flush=True)
		print(
			f"DEBUG_LAMRA_SCORE: pre-forward input_ids device={input_ids.device if input_ids is not None else 'MISSING'}",
			flush=True,
		)
		print(
			f"DEBUG_LAMRA_SCORE: pre-forward pixel_values device={pixel_values.device if pixel_values is not None else 'MISSING'}",
			flush=True,
		)
		print(f"DEBUG_LAMRA_SCORE: pre-forward input tensor shapes={shape_info}", flush=True)

		with torch.no_grad():
			print("DEBUG_LAMRA_SCORE: 8 starting model forward pass", flush=True)
			logits = self.model(
				**joint_inputs,
				return_dict=True,
				use_cache=False,
			).logits
			print("DEBUG_LAMRA_SCORE: 9 model forward finished", flush=True)

			print("DEBUG_LAMRA_SCORE: 10 extracting yes/no logits", flush=True)
			next_token_logits = logits[:, -1, :]
			yes_logit = next_token_logits[:, yes_token_id]
			no_logit = next_token_logits[:, no_token_id]
			binary_logits = torch.stack([no_logit, yes_logit], dim=1)
			prob_yes = torch.softmax(binary_logits, dim=1)[:, 1]
			print("DEBUG_LAMRA_SCORE: 11 score computed", flush=True)

		print("DEBUG_LAMRA_SCORE: 12 returning score", flush=True)
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



