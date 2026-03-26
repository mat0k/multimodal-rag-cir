from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

try:
	from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - environment dependent
	process_vision_info = None


Message = list[dict[str, Any]]


def _fallback_process_vision_info(messages: Sequence[Message]) -> tuple[list[Any | None], list[Any | None]]:
	"""Fallback parser for Qwen-style message content when qwen_vl_utils is unavailable.

	This fallback supports the message format used in this repository and extracts one
	image/video payload per chat sample, which is sufficient for current pointwise scoring.
	"""
	image_inputs: list[Any | None] = []
	video_inputs: list[Any | None] = []

	for sample_messages in messages:
		sample_images: list[Any] = []
		sample_videos: list[Any] = []

		for turn in sample_messages:
			content = turn.get("content", [])
			if not isinstance(content, list):
				continue

			for item in content:
				if not isinstance(item, dict):
					continue

				item_type = item.get("type")
				if item_type == "image" or "image" in item:
					image_value = item.get("image")
					if image_value is not None:
						sample_images.append(image_value)
				if item_type == "video" or "video" in item:
					video_value = item.get("video")
					if video_value is not None:
						sample_videos.append(video_value)

		image_inputs.append(sample_images if sample_images else None)
		video_inputs.append(sample_videos if sample_videos else None)

	return image_inputs, video_inputs


def load_image_item(
	image: str | Path | Image.Image,
	image_root: str | Path | None = None,
	rgb_only: bool = True,
) -> Image.Image:
	"""Load an image path/PIL input into a normalized PIL image object."""
	if isinstance(image, Image.Image):
		pil_image = image
	else:
		image_path = Path(image)
		if not image_path.is_absolute() and image_root is not None:
			image_path = Path(image_root) / image_path
		if not image_path.exists():
			raise FileNotFoundError(f"Image not found: {image_path}")
		pil_image = Image.open(image_path)

	if rgb_only and pil_image.mode != "RGB":
		pil_image = pil_image.convert("RGB")
	return pil_image


def build_query_message(
	reference_image: str | Path | Image.Image,
	text_edit: str,
	emb_token: str = "<emb>",
	query_instruction: str | None = None,
	image_root: str | Path | None = None,
) -> Message:
	"""Build a single chat-formatted query message from image + edit text."""
	query_instruction = query_instruction or (
		"Given the reference image and relative text edit below, "
		"produce a compact visual-intent representation."
	)
	normalized_image = load_image_item(reference_image, image_root=image_root)

	return [
		{
			"role": "user",
			"content": [
				{"type": "image", "image": normalized_image},
				{
					"type": "text",
					"text": (
						f"{query_instruction}\n"
						f"Relative text edit: {text_edit}\n"
						"Summarize intent in one word: "
					),
				},
			],
		},
		{
			"role": "assistant",
			"content": [{"type": "text", "text": f"{emb_token}."}],
		},
	]


def build_candidate_message(
	candidate_image: str | Path | Image.Image,
	emb_token: str = "<emb>",
	candidate_instruction: str | None = None,
	image_root: str | Path | None = None,
) -> Message:
	"""Build a single candidate-image message for embedding extraction."""
	candidate_instruction = candidate_instruction or "Produce a compact representation for this candidate image."
	normalized_image = load_image_item(candidate_image, image_root=image_root)

	return [
		{
			"role": "user",
			"content": [
				{"type": "image", "image": normalized_image},
				{"type": "text", "text": f"{candidate_instruction}\nSummarize in one word: "},
			],
		},
		{
			"role": "assistant",
			"content": [{"type": "text", "text": f"{emb_token}."}],
		},
	]


def build_pointwise_message(
	reference_image: str | Path | Image.Image,
	candidate_image: str | Path | Image.Image,
	text_edit: str,
	emb_token: str = "<emb>",
	pointwise_instruction: str | None = None,
	image_root: str | Path | None = None,
) -> Message:
	"""Build one message that includes reference image, candidate image, and edit text.

	Useful when using a pointwise/cross-attention scoring strategy.
	"""
	pointwise_instruction = pointwise_instruction or (
		"Given a reference image, a candidate image, and a relative text edit, "
		"estimate how well the candidate matches the edited reference."
	)

	reference_pil = load_image_item(reference_image, image_root=image_root)
	candidate_pil = load_image_item(candidate_image, image_root=image_root)

	return [
		{
			"role": "user",
			"content": [
				{"type": "text", "text": pointwise_instruction},
				{"type": "text", "text": "Reference image:"},
				{"type": "image", "image": reference_pil},
				{"type": "text", "text": f"Relative text edit: {text_edit}"},
				{"type": "text", "text": "Candidate image:"},
				{"type": "image", "image": candidate_pil},
				{"type": "text", "text": "Summarize final match in one word: "},
			],
		},
		{
			"role": "assistant",
			"content": [{"type": "text", "text": f"{emb_token}."}],
		},
	]


def build_query_message_batch(
	reference_images: Sequence[str | Path | Image.Image],
	text_edits: Sequence[str],
	emb_token: str = "<emb>",
	query_instruction: str | None = None,
	image_root: str | Path | None = None,
) -> list[Message]:
	"""Build batched query messages with one reference image + edit per item."""
	if len(reference_images) != len(text_edits):
		raise ValueError(
			"reference_images and text_edits must have the same length; "
			f"got {len(reference_images)} vs {len(text_edits)}"
		)

	return [
		build_query_message(
			reference_image=reference_image,
			text_edit=text_edit,
			emb_token=emb_token,
			query_instruction=query_instruction,
			image_root=image_root,
		)
		for reference_image, text_edit in zip(reference_images, text_edits)
	]


def build_candidate_message_batch(
	candidate_images: Sequence[str | Path | Image.Image],
	emb_token: str = "<emb>",
	candidate_instruction: str | None = None,
	image_root: str | Path | None = None,
) -> list[Message]:
	"""Build batched candidate-only messages."""
	return [
		build_candidate_message(
			candidate_image=candidate_image,
			emb_token=emb_token,
			candidate_instruction=candidate_instruction,
			image_root=image_root,
		)
		for candidate_image in candidate_images
	]


def apply_chat_templates(
	processor: Any,
	messages: Sequence[Message],
	add_generation_prompt: bool = True,
) -> list[str]:
	"""Render Qwen chat-template text for each message."""
	template_renderer = getattr(processor, "tokenizer", processor)
	return [
		template_renderer.apply_chat_template(
			msg,
			tokenize=False,
			add_generation_prompt=add_generation_prompt,
		)
		for msg in messages
	]


def process_messages_to_inputs(
	processor: Any,
	messages: Sequence[Message],
	device: torch.device | str,
) -> dict[str, torch.Tensor]:
	"""Convert chat messages into model-ready tensors and move to device."""
	texts = apply_chat_templates(processor=processor, messages=messages)
	if process_vision_info is None:
		image_inputs, video_inputs = _fallback_process_vision_info(messages)
	else:
		image_inputs, video_inputs = process_vision_info(messages)

	processor_kwargs: dict[str, Any] = {
		"text": texts,
		"padding": True,
		"return_tensors": "pt",
	}
	if any(images is not None for images in image_inputs):
		processor_kwargs["images"] = image_inputs
	if any(videos is not None for videos in video_inputs):
		processor_kwargs["videos"] = video_inputs

	model_inputs = processor(**processor_kwargs)
	return model_inputs.to(device)


__all__ = [
	"Message",
	"apply_chat_templates",
	"build_candidate_message",
	"build_candidate_message_batch",
	"build_pointwise_message",
	"build_query_message",
	"build_query_message_batch",
	"load_image_item",
	"process_messages_to_inputs",
]

