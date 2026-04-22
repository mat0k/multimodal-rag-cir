from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from src.rerankers.lamra_rank import LamRARanker


class Qwen3VLRanker(LamRARanker):
	"""Qwen3-VL reranker adapter using the common BaseReranker interface."""

	reranker_type = "qwen3vl"

	def load_model(self) -> None:
		"""Load processor + Qwen3-VL model from local checkpoint."""
		self._validate_checkpoint_layout(self.model_name_or_path)

		self.processor = self._load_processor()
		self._ensure_chat_template(self.processor, self.model_name_or_path)
		self.tokenizer = self.processor.tokenizer

		self.model = Qwen3VLForConditionalGeneration.from_pretrained(
			self.model_name_or_path,
			torch_dtype=self.dtype,
			low_cpu_mem_usage=self.low_cpu_mem_usage,
			trust_remote_code=self.trust_remote_code,
			local_files_only=self.local_files_only,
			revision=self.revision,
		).to(self.device)
		self.model.eval()

		self.emb_token_id = self._ensure_embed_token(self.emb_token)

	def _load_processor(self) -> Qwen3VLProcessor | Any:
		"""Load Qwen3-VL processor with AutoProcessor fallback for compatibility."""
		try:
			return Qwen3VLProcessor.from_pretrained(
				self.model_name_or_path,
				trust_remote_code=self.trust_remote_code,
				local_files_only=self.local_files_only,
				revision=self.revision,
				use_fast=False,
			)
		except Exception:
			processor = AutoProcessor.from_pretrained(
				self.model_name_or_path,
				trust_remote_code=self.trust_remote_code,
				local_files_only=self.local_files_only,
				revision=self.revision,
			)
			if not hasattr(processor, "tokenizer"):
				raise RuntimeError("Loaded processor does not expose tokenizer; expected Qwen3-VL processor.")
			return processor

	@staticmethod
	def _ensure_chat_template(processor: Any, model_name_or_path: str) -> None:
		"""Attach local chat template when provided as jinja/json metadata."""
		local_path = Path(model_name_or_path)
		if not local_path.exists() or not local_path.is_dir():
			return

		template_text: str | None = None

		jinja_file = local_path / "chat_template.jinja"
		if jinja_file.exists():
			template_text = jinja_file.read_text(encoding="utf-8")

		if not template_text:
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
			if hasattr(processor, "tokenizer"):
				processor.tokenizer.chat_template = template_text

	@staticmethod
	def _validate_checkpoint_layout(model_name_or_path: str) -> None:
		"""Validate local HF checkpoint layout for Qwen3-VL single or sharded weights."""
		local_path = Path(model_name_or_path)
		if not local_path.exists():
			raise FileNotFoundError(
				"Qwen3-VL loading is configured for local checkpoints only, but path was not found: "
				f"{local_path}"
			)

		if not local_path.is_dir():
			raise ValueError(f"model_name_or_path must point to a directory, got: {local_path}")

		required_files = [
			"config.json",
			"preprocessor_config.json",
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

		has_single_weight = (local_path / "model.safetensors").exists()
		has_sharded_index = (local_path / "model.safetensors.index.json").exists()
		has_shards = any(local_path.glob("model-*.safetensors"))

		if not has_single_weight and not (has_sharded_index and has_shards):
			raise FileNotFoundError(
				"No model weights found. Expected model.safetensors or sharded files referenced by "
				"model.safetensors.index.json."
			)
