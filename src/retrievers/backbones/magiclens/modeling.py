"""PyTorch port of MagicLens (Zhang et al., ICML'24).

Ported from the official JAX/Flax release (see magiclens_reference/model.py
for the source of truth this was built from). No pretrained weights are
loaded here — this only defines the architecture. Weight conversion from
the official checkpoint happens in a separate script (added in a later
step), and must pass a parity check against the original JAX model before
this is trusted.

Candidate/gallery images use the *same* fusion path as queries, with an
empty-string instruction — this matches the official eval protocol
(magiclens_reference/data_utils.py uses `tokenizer("")` for index images),
not a plain CLIP image embedding.
"""

from typing import List, Optional, Union

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.retrievers.backbones.magiclens.layers import (
    MLAttenTokenPoolingLayer,
    MLStackedTransformer,
)

MAGICLENS_CONFIGS = {
    "base": dict(
        embed_dim=512,
        ff_hidden_size=512 * 4,
        num_layers=4,
        num_heads=8,
        num_query_token=1,
        clip_model_name="ViT-B-16-quickgelu",
        image_resolution=224,
    ),
    "large": dict(
        embed_dim=768,
        ff_hidden_size=768 * 4,
        num_layers=4,
        num_heads=16,
        num_query_token=1,
        clip_model_name="ViT-L-14-quickgelu",
        image_resolution=224,
    ),
}


class MagicLens(nn.Module):
    """PyTorch MagicLens: CLIP backbone + custom multimodal fusion head.

    Exposes the same duck-typed contract the SP trainer expects from
    Visualized_BGE: encode_mm(images, texts), encode_image(images),
    .hidden_dim, .device.
    """

    def __init__(self, model_size: str = "base"):
        super().__init__()
        if model_size not in MAGICLENS_CONFIGS:
            raise ValueError(f"model_size must be one of {list(MAGICLENS_CONFIGS)}")
        cfg = MAGICLENS_CONFIGS[model_size]
        self.model_size = model_size
        self.hidden_dim = cfg["embed_dim"]
        self.image_resolution = cfg["image_resolution"]

        # CLIP backbone — architecture only, no pretrained weights (those
        # come from the converted MagicLens checkpoint, not stock CLIP).
        self.clip = open_clip.create_model(cfg["clip_model_name"], pretrained=None)
        self.tokenizer = open_clip.get_tokenizer(cfg["clip_model_name"])

        self.multimodal_encoder = MLStackedTransformer(
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            input_dim=cfg["embed_dim"],
            hidden_dim=cfg["ff_hidden_size"],
            use_bias=True,
            add_skip_connection=True,
            use_per_dim_scale=False,
        )
        self.contrastive_multimodal_pooler = MLAttenTokenPoolingLayer(
            input_dim=cfg["embed_dim"],
            query_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            num_query_tokens=cfg["num_query_token"],
            use_bias=True,
            use_per_dim_scale=True,
        )

        # Cached "empty instruction" tokenization for candidate/gallery
        # images (mirrors the official eval scripts' `tokenizer("")`).
        null_tokens = self.tokenizer([""])
        self.register_buffer("_null_text_tokens", null_tokens, persistent=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _normalize(self, embed: torch.Tensor) -> torch.Tensor:
        # MagicLens always normalizes in float32 regardless of compute dtype.
        embed = embed.float()
        norm = torch.sqrt((embed * embed).sum(dim=-1, keepdim=True) + 1e-12)
        return embed / norm

    def _clip_encode(self, images: torch.Tensor, text_tokens: torch.Tensor):
        """images: [B,3,H,W] preprocessed. text_tokens: [B,77] CLIP token ids."""
        image_embeds = self.clip.encode_image(images)
        text_embeds = self.clip.encode_text(text_tokens)
        return image_embeds, text_embeds

    def _fuse(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """[B,D], [B,D] -> fused, L2-normalized [B,D]."""
        img = image_embeds.unsqueeze(1)  # [B,1,D]
        txt = text_embeds.unsqueeze(1)   # [B,1,D]
        seq = torch.cat([img, txt], dim=1)  # [B,2,D]
        seq = self.multimodal_encoder(seq)
        pooled = self.contrastive_multimodal_pooler(seq)  # [B,1,D]
        pooled = pooled[:, 0]
        return self._normalize(pooled)

    def encode_mm(self, images: torch.Tensor, texts: Union[List[str], torch.Tensor]) -> torch.Tensor:
        """Query encoder: reference image + instruction text -> [B, hidden_dim].

        NOTE: takes raw strings (tokenized internally with CLIP's own
        tokenizer) or pre-tokenized ids. This does NOT yet accept VISTA's
        HF-style {input_ids, attention_mask} dict — the SP dataset/collator
        will need a per-backbone adapter for that. Flagged, not solved yet.
        """
        text_tokens = texts if torch.is_tensor(texts) else self.tokenizer(list(texts)).to(images.device)
        image_embeds, text_embeds = self._clip_encode(images, text_tokens)
        return self._fuse(image_embeds, text_embeds)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Candidate encoder: image only, fused with an empty-string
        instruction (matches the official MagicLens eval protocol)."""
        B = images.shape[0]
        text_tokens = self._null_text_tokens.expand(B, -1).to(images.device)
        image_embeds, text_embeds = self._clip_encode(images, text_tokens)
        return self._fuse(image_embeds, text_embeds)
