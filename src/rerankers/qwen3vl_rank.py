from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.rerankers.backbones.lamra.preprocess import load_image_item, process_messages_to_inputs
from src.rerankers.base import BaseReranker, RerankCandidate, RerankQuery


class Qwen3VLReranker(BaseReranker):
	"""Qwen3-VL pointwise reranker wrapper using yes/no relevance scoring."""

	def __init__(
		self,
		model_name_or_path: str,
		device: str = "auto",
		dtype: str = "bfloat16",
		trust_remote_code: bool = True,
		local_files_only: bool = True,
		revision: str = "main",
		low_cpu_mem_usage: bool = True,
		instruction: str | None = None,
		candidate_batch_size: int = 8,
		fps: float = 1.0,
		max_frames: int = 64,
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
		self.instruction = instruction or (
			"Given a reference image and a text modification in the query, "
			"score candidate images by how well they match the edited target intent."
		)
		self.candidate_batch_size = max(1, int(candidate_batch_size))
		self.fps = float(fps)
		self.max_frames = int(max_frames)
		self.image_root = image_root
		self.rgb_only = rgb_only
		self.resize_short_edge = resize_short_edge
		self.center_crop = center_crop
		self.interpolation = interpolation

		self.model: Qwen3VLForConditionalGeneration | None = None
		self.processor: Any = None
		self.score_linear: torch.nn.Linear | None = None

		self.load_model()

	@classmethod
	def from_config(cls, config: Mapping[str, Any] | str) -> "Qwen3VLReranker":
		if isinstance(config, str):
			config_data = cls._load_config_file(config)
		else:
			config_data = dict(config)

		model_cfg = dict(config_data.get("model", {}))
		runtime_cfg = dict(config_data.get("runtime", {}))
		image_cfg = dict(config_data.get("image_io", {}))
		inference_cfg = dict(config_data.get("inference", {}))
		prompting_cfg = dict(config_data.get("prompting", {}))
		qwen_cfg = dict(config_data.get("qwen3vl", {}))

		model_name_or_path = (
			model_cfg.get("checkpoint_path")
			or model_cfg.get("model_name_or_path")
			or model_cfg.get("name")
		)
		if not model_name_or_path:
			raise ValueError(
				"Config must define model.checkpoint_path (or model.model_name_or_path/model.name)."
			)

		return cls(
			model_name_or_path=str(model_name_or_path),
			device=str(runtime_cfg.get("device", "auto")),
			dtype=str(runtime_cfg.get("dtype", "bfloat16")),
			trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
			local_files_only=bool(model_cfg.get("local_files_only", True)),
			revision=str(model_cfg.get("revision", "main")),
			instruction=str(prompting_cfg.get("instruction", "")).strip() or None,
			candidate_batch_size=int(inference_cfg.get("batch_size_candidates", 8)),
			fps=float(qwen_cfg.get("fps", 1.0)),
			max_frames=int(qwen_cfg.get("max_frames", 64)),
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
		self._validate_checkpoint_layout(self.model_name_or_path)

		self.processor = AutoProcessor.from_pretrained(
			self.model_name_or_path,
			trust_remote_code=self.trust_remote_code,
			local_files_only=self.local_files_only,
			revision=self.revision,
		)
		self.model = Qwen3VLForConditionalGeneration.from_pretrained(
			self.model_name_or_path,
			torch_dtype=self.dtype,
			low_cpu_mem_usage=self.low_cpu_mem_usage,
			trust_remote_code=self.trust_remote_code,
			local_files_only=self.local_files_only,
			revision=self.revision,
		).to(self.device)
		self.model.eval()

		yes_token_id = self._resolve_token_id("yes", " yes")
		no_token_id = self._resolve_token_id("no", " no")
		self.score_linear = self._build_binary_linear(yes_token_id=yes_token_id, no_token_id=no_token_id)

	def score(self, query: RerankQuery, candidate: RerankCandidate) -> float:
		return float(self.score_batch(query=query, candidates=[candidate])[0])

	def score_batch(self, query: RerankQuery, candidates: Sequence[RerankCandidate]) -> list[float]:
		if not candidates:
			return []

		scores: list[float] = []
		for start in range(0, len(candidates), self.candidate_batch_size):
			chunk = candidates[start:start + self.candidate_batch_size]
			pair_messages: list[list[dict[str, Any]]] = []
			for candidate in chunk:
				pair_messages.append(self._build_pair_message(query=query, candidate=candidate))
			scores.extend(self._score_messages(pair_messages))
		return scores

	def _build_pair_message(self, query: RerankQuery, candidate: RerankCandidate) -> list[dict[str, Any]]:
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

		content: list[dict[str, Any]] = [
			{"type": "text", "text": f"<Instruct>: {self.instruction}"},
			{"type": "text", "text": "<Query>:"},
			{"type": "image", "image": reference_image},
			{"type": "text", "text": f"Relative text edit: {query.text_edit}"},
			{"type": "text", "text": "\n<Document>:"},
			{"type": "image", "image": candidate_image},
		]

		return [
			{
				"role": "system",
				"content": [
					{
						"type": "text",
						"text": (
							"Judge whether the Document meets the requirements based on Query and Instruct. "
							"Answer only yes or no."
						),
					}
				],
			},
			{"role": "user", "content": content},
		]

	def _score_messages(self, pair_messages: Sequence[Sequence[dict[str, Any]]]) -> list[float]:
		if self.model is None or self.processor is None or self.score_linear is None:
			raise RuntimeError("Qwen3VLReranker is not initialized. Call load_model() first.")

		model_inputs = process_messages_to_inputs(
			processor=self.processor,
			messages=pair_messages,
			device=self.device,
			move_to_device=False,
		)
		model_inputs = model_inputs.to(next(self.model.parameters()).device)

		with torch.inference_mode():
			backbone_outputs = self.model.model(**model_inputs, return_dict=True)
			last_hidden = backbone_outputs.last_hidden_state[:, -1]
			scores = self.score_linear(last_hidden)
			prob_yes = torch.sigmoid(scores).squeeze(-1)
		return prob_yes.detach().float().cpu().tolist()

	def _build_binary_linear(self, yes_token_id: int, no_token_id: int) -> torch.nn.Linear:
		if self.model is None:
			raise RuntimeError("Model must be initialized before building score head.")

		lm_head = getattr(self.model, "lm_head", None)
		if lm_head is None:
			raise RuntimeError("Loaded Qwen3 model does not expose lm_head weights.")

		lm_head_weights = lm_head.weight.detach()
		weight_yes = lm_head_weights[yes_token_id]
		weight_no = lm_head_weights[no_token_id]

		dimension = int(weight_yes.shape[0])
		linear_layer = torch.nn.Linear(dimension, 1, bias=False)
		linear_layer = linear_layer.to(device=self.device, dtype=self.model.dtype)
		with torch.no_grad():
			linear_layer.weight[0] = (weight_yes - weight_no).to(self.model.dtype)
		linear_layer.eval()
		return linear_layer

	def _resolve_token_id(self, token: str, fallback_token_text: str) -> int:
		if self.processor is None:
			raise RuntimeError("Processor must be initialized before resolving token ids.")
		tokenizer = self.processor.tokenizer
		vocab = tokenizer.get_vocab()

		token_id = vocab.get(token)
		if token_id is not None:
			return int(token_id)

		fallback_ids = tokenizer.encode(fallback_token_text, add_special_tokens=False)
		if len(fallback_ids) != 1:
			raise ValueError(
				f"Expected fallback token text '{fallback_token_text}' to map to one token, got {fallback_ids}."
			)
		return int(fallback_ids[0])

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

	@staticmethod
	def _validate_checkpoint_layout(model_name_or_path: str) -> None:
		local_path = Path(model_name_or_path)
		if not local_path.exists():
			raise FileNotFoundError(
				"Qwen3-VL loading is configured for local checkpoints only, but path was not found: "
				f"{local_path}"
			)

		if not local_path.is_dir():
			raise ValueError(f"model_name_or_path must point to a directory, got: {local_path}")

		required_files = ["config.json", "preprocessor_config.json"]
		missing_required = [name for name in required_files if not (local_path / name).exists()]
		if missing_required:
			raise FileNotFoundError(
				"Local checkpoint is missing required Hugging Face files: "
				f"{missing_required}. Expected directory: {local_path}"
			)

		has_model_weights = (local_path / "model.safetensors").exists() or (local_path / "model.safetensors.index.json").exists()
		if not has_model_weights:
			raise FileNotFoundError(
				"Local checkpoint is missing model weights. Expected model.safetensors or model.safetensors.index.json."
			)


__all__ = ["Qwen3VLReranker"]
