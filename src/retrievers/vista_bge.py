from typing import Any

from src.retrievers.base import TwoEncoderVLM


class VistaBGERetriever(TwoEncoderVLM):
    """Scaffold for VISTA-BGE integration.

    Fill this class with the model-specific loading logic and expose:
    - self.vision
    - self.text
    - self.image_processor
    - self.tokenizer
    """

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> "VistaBGERetriever":
        raise NotImplementedError(
            "VistaBGERetriever.from_pretrained is not implemented yet. "
            "Add your VISTA-BGE loading code in src/retrievers/vista_bge.py."
        )
