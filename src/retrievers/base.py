from abc import abstractmethod
from typing import Any

import torch


class TwoEncoderVLM(torch.nn.Module):
    """Base contract for dual-encoder retrievers used in evaluation.

    Implementations must expose:
    - vision(images) -> object with .image_embeds
    - text(input_ids=..., attention_mask=...) -> object with .text_embeds
    - image_processor compatible with dataset loaders
    - tokenizer compatible with dataset loaders
    """

    vision: torch.nn.Module
    text: torch.nn.Module
    image_processor: Any
    tokenizer: Any

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> "TwoEncoderVLM":
        raise NotImplementedError
