from __future__ import annotations

from typing import Any


Message = list[dict[str, Any]]


DEFAULT_SYSTEM_INSTRUCTION = (
	"You are a visual retrieval assistant. "
	"Judge whether the candidate image matches the reference image after applying the relative text edit."
)

DEFAULT_POINTWISE_PROMPT = (
	"Given a reference image, a relative text edit, and a candidate image, "
	"assess how well the candidate satisfies the edited reference intent."
)

DEFAULT_SUMMARY_REQUEST = "Summarize match quality in one word: "

DEFAULT_RELEVANCE_REQUEST = (
	"Does the candidate image satisfy the edited reference intent? "
	"Answer with one token: yes or no."
)


def build_pointwise_cir_message(
	reference_image: Any,
	text_edit: str,
	candidate_image: Any,
	emb_token: str = "<emb>",
	system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
	pointwise_prompt: str = DEFAULT_POINTWISE_PROMPT,
	summary_request: str = DEFAULT_SUMMARY_REQUEST,
) -> Message:
	"""Build a Qwen-compatible pointwise message for CIR reranking.

	Args:
		reference_image: Reference image object accepted by Qwen processor.
		text_edit: Relative text edit describing the transformation intent.
		candidate_image: Candidate image object accepted by Qwen processor.
		emb_token: Special token used later for embedding extraction.
		system_instruction: High-level behavior instruction.
		pointwise_prompt: Task framing for pointwise judgment.
		summary_request: Final text cue before assistant <emb> response.

	Returns:
		Chat-style message list compatible with Qwen apply_chat_template + process_vision_info.
	"""
	return [
		{
			"role": "system",
			"content": [{"type": "text", "text": system_instruction}],
		},
		{
			"role": "user",
			"content": [
				{"type": "text", "text": pointwise_prompt},
				{"type": "text", "text": "Reference image:"},
				{"type": "image", "image": reference_image},
				{"type": "text", "text": f"Relative text edit: {text_edit}"},
				{"type": "text", "text": "Candidate image:"},
				{"type": "image", "image": candidate_image},
				{"type": "text", "text": summary_request},
			],
		},
		{
			"role": "assistant",
			"content": [{"type": "text", "text": f"{emb_token}."}],
		},
	]


def build_pointwise_relevance_message(
	reference_image: Any,
	text_edit: str,
	candidate_image: Any,
	system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
	pointwise_prompt: str = DEFAULT_POINTWISE_PROMPT,
	relevance_request: str = DEFAULT_RELEVANCE_REQUEST,
) -> Message:
	"""Build a joint pointwise message for direct yes/no relevance scoring.

	This message intentionally leaves the assistant response empty so logits at the
	next token can be interpreted as relevance evidence.
	"""
	return [
		{
			"role": "system",
			"content": [{"type": "text", "text": system_instruction}],
		},
		{
			"role": "user",
			"content": [
				{"type": "text", "text": pointwise_prompt},
				{"type": "text", "text": "Reference image:"},
				{"type": "image", "image": reference_image},
				{"type": "text", "text": f"Relative text edit: {text_edit}"},
				{"type": "text", "text": "Candidate image:"},
				{"type": "image", "image": candidate_image},
				{"type": "text", "text": relevance_request},
			],
		},
	]


__all__ = [
	"DEFAULT_POINTWISE_PROMPT",
	"DEFAULT_SUMMARY_REQUEST",
	"DEFAULT_RELEVANCE_REQUEST",
	"DEFAULT_SYSTEM_INSTRUCTION",
	"Message",
	"build_pointwise_cir_message",
	"build_pointwise_relevance_message",
]

