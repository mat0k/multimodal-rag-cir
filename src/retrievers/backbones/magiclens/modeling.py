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

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Union

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.retrievers.backbones.magiclens.layers import (
    MLAttenTokenPoolingLayer,
    MLStackedTransformer,
)

# Verified identical to Scenic's clip_model.IMAGE_MEAN/IMAGE_STD (max diff 2.4e-07).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# CLIP's context length. The trainer and every eval dataset builder hardcode 77;
# assert rather than assume, since a mismatch would silently mis-pad.
CONTEXT_LENGTH = 77


class MagicLensImagePreprocess:
    """Reproduces the official MagicLens evaluation preprocessing.

    Faithful to `magiclens_reference/data_utils.py::process_img` followed by
    `model.py::_preprocess_images`. Two details are unusual and load-bearing:

    1. Pixels are scaled by the image's OWN maximum value, not by 255. A frame
       whose brightest pixel is 200 is therefore brightened, not merely rescaled.
    2. The image is resized straight to a square, so the aspect ratio is
       squashed rather than cropped. The reference's `largest_square_crop` is a
       no-op in the eval path precisely because `process_img` already produced a
       square, so replicating the crop here would NOT match the official
       pipeline.

    Deviating from either would still produce plausible-looking embeddings while
    silently degrading retrieval, so both are kept exactly as upstream.
    """

    def __init__(self, size: int = 224, is_train: bool = False):
        self.size = size
        self.is_train = is_train
        self._mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        self._std = torch.tensor(CLIP_STD).view(3, 1, 1)
        self._random_crop = None
        if is_train:
            # Mild augmentation mirroring VISTA's train transform, applied before
            # the max-scaling so the scaling still reflects the pixels the model sees.
            from torchvision.transforms import InterpolationMode, RandomResizedCrop

            self._random_crop = RandomResizedCrop(
                size, scale=(0.9, 1.0), interpolation=InterpolationMode.BILINEAR, antialias=True
            )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if self._random_crop is not None:
            image = self._random_crop(image)

        array = torch.from_numpy(np.asarray(image).copy()).float()      # [H, W, 3]
        array = array / (array.max() + 1e-12)                           # per-image max
        tensor = array.permute(2, 0, 1).unsqueeze(0)                    # [1, 3, H, W]
        if tensor.shape[-2:] != (self.size, self.size):
            tensor = F.interpolate(
                tensor, size=(self.size, self.size),
                mode="bilinear", align_corners=False, antialias=True,
            )
        return ((tensor[0] - self._mean) / self._std)


class MagicLensTokenizer:
    """Adapts open_clip's tokenizer to the HuggingFace-style call the datasets use.

    The SP dataset and every eval dataset call the tokenizer as
    `tok(text, padding="max_length", max_length=77, truncation=True,
    return_tensors="pt")` and then index `["input_ids"]` / `["attention_mask"]`.
    open_clip instead returns a bare `[B, 77]` tensor and accepts no kwargs.
    """

    def __init__(self, tokenizer: Any, context_length: int = CONTEXT_LENGTH):
        self._tokenizer = tokenizer
        self.context_length = context_length
        self.model_max_length = context_length

    def __call__(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
        **_: Any,
    ) -> dict:
        # padding/truncation are accepted and ignored: open_clip always pads and
        # truncates to exactly context_length, which is the behaviour callers ask for.
        if max_length is not None and max_length != self.context_length:
            raise ValueError(
                f"MagicLens uses CLIP's fixed context length of {self.context_length}; "
                f"got max_length={max_length}."
            )
        if return_tensors != "pt":
            raise ValueError("MagicLensTokenizer supports return_tensors='pt' only.")

        input_ids = self._tokenizer([text] if isinstance(text, str) else list(text))
        # CLIP pads with id 0; every real token (including SOT/EOT) is non-zero.
        return {"input_ids": input_ids, "attention_mask": (input_ids != 0).long()}

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


@dataclass
class MagicLensOutput:
    """Mirrors `Visualized_BGE`'s EncoderOutput, so the trainer reads `.loss` the same way."""

    loss: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
    q_reps: Optional[torch.Tensor] = None
    c_reps: Optional[torch.Tensor] = None


class MagicLens(nn.Module):
    """PyTorch MagicLens: CLIP backbone + custom multimodal fusion head.

    Exposes the same duck-typed contract the SP trainer expects from
    Visualized_BGE: encode_mm(images, texts), encode_image(images),
    .hidden_dim, .device.
    """

    def __init__(self, model_size: str = "base", temperature: float = 0.02,
                 use_query_negatives: bool = True):
        super().__init__()
        if model_size not in MAGICLENS_CONFIGS:
            raise ValueError(f"model_size must be one of {list(MAGICLENS_CONFIGS)}")
        cfg = MAGICLENS_CONFIGS[model_size]
        self.model_size = model_size
        self.hidden_dim = cfg["embed_dim"]
        self.image_resolution = cfg["image_resolution"]
        self.temperature = temperature
        # Paper default. Set False only to reproduce the ablation in which the
        # query-image negative is removed (which degrades every benchmark).
        self.use_query_negatives = use_query_negatives
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

        # CLIP backbone — architecture only, no pretrained weights (those
        # come from the converted MagicLens checkpoint, not stock CLIP).
        self.clip = open_clip.create_model(cfg["clip_model_name"], pretrained=None)
        self.tokenizer = MagicLensTokenizer(open_clip.get_tokenizer(cfg["clip_model_name"]))

        # Named to match Visualized_BGE so the trainer and retriever wrappers can
        # use either backbone without dispatching on type.
        self.preprocess_train = MagicLensImagePreprocess(cfg["image_resolution"], is_train=True)
        self.preprocess_val = MagicLensImagePreprocess(cfg["image_resolution"], is_train=False)

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
        null_tokens = self.tokenizer([""])["input_ids"]
        self.register_buffer("_null_text_tokens", null_tokens, persistent=False)

        # Self-move to GPU, matching Visualized_BGE. The trainer never calls
        # .to(device) on the backbone -- it reads `backbone.device` and moves the
        # *batches* there -- so a backbone that stays on CPU silently trains on CPU
        # at roughly 1/24th the speed instead of failing.
        if torch.cuda.is_available():
            self.to(torch.device("cuda"))

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

    def encode_mm(
        self, images: torch.Tensor, texts: Union[List[str], torch.Tensor, Mapping[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Query encoder: reference image + instruction text -> [B, hidden_dim].

        Accepts raw strings, pre-tokenized CLIP ids, or the HF-style
        ``{"input_ids", "attention_mask"}`` dict that the trainer and the eval
        code pass, so this matches `Visualized_BGE.encode_mm`'s calling
        convention.

        ``attention_mask`` is deliberately ignored: CLIP locates the EOT token
        by ``argmax`` over the ids (EOT is 49407, the largest id in the
        vocabulary), so padding cannot affect pooling. This is correct, not a
        shortcut.
        """
        if isinstance(texts, Mapping):
            text_tokens = texts["input_ids"]
        elif torch.is_tensor(texts):
            text_tokens = texts
        else:
            text_tokens = self.tokenizer(list(texts))["input_ids"]
        text_tokens = text_tokens.to(images.device)
        image_embeds, text_embeds = self._clip_encode(images, text_tokens)
        return self._fuse(image_embeds, text_embeds)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Candidate encoder: image only, fused with an empty-string
        instruction (matches the official MagicLens eval protocol)."""
        B = images.shape[0]
        text_tokens = self._null_text_tokens.expand(B, -1).to(images.device)
        image_embeds, text_embeds = self._clip_encode(images, text_tokens)
        return self._fuse(image_embeds, text_embeds)

    def compute_similarity(self, q_reps: torch.Tensor, p_reps: torch.Tensor) -> torch.Tensor:
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(
        self,
        mm_it_query=None,
        image_candidate=None,
        task_type: Optional[str] = None,
        **_: Any,
    ) -> MagicLensOutput:
        """Contrastive training step, following MagicLens' own recipe.

        Signature matches `Visualized_BGE.forward` so the trainer can call either
        backbone identically, but the objective is MagicLens', not VISTA's.

        Direction is one-way, query -> candidates, per the paper: "our model is
        updated by contrasting the paired query-target against other targets in
        one training batch". This is *not* CLIP's symmetric image<->text loss;
        CLIP is a joint embedding trained for retrieval in both directions,
        whereas composed retrieval only ever ranks candidates given a query.

        When ``use_query_negatives`` is set (the default, and what the paper
        does), each reference image is *also* encoded with an empty instruction
        and appended to the candidate pool as an extra hard negative. The paper
        motivates this directly -- "the query image itself can be a challenging
        hard negative for the multimodal query" -- and its ablation shows that
        dropping it degrades every benchmark, with the model learning to "rank
        the query image itself higher than other images during retrieval".
        Note ``encode_image`` already encodes with an empty instruction, so it
        produces exactly the (image_q, "") embedding the paper describes.
        """
        if task_type != "edit_image":
            raise ValueError(
                f"MagicLens supports task_type='edit_image' only; got {task_type!r}."
            )

        ref_images, texts = mm_it_query
        query_reps = self.encode_mm(ref_images, texts)
        candi_reps = self.encode_image(image_candidate)

        if self.training:
            n_query = query_reps.size(0)
            # Target index is fixed against the TRUE candidates, before any extra
            # negatives are appended, so appending cannot shift the positive.
            target = torch.arange(n_query, device=query_reps.device, dtype=torch.long)
            target = target * (candi_reps.size(0) // n_query)

            pool = candi_reps
            if self.use_query_negatives:
                # The reference images themselves, as additional negatives.
                pool = torch.cat([candi_reps, self.encode_image(ref_images)], dim=0)

            scores = self.compute_similarity(query_reps, pool) / self.temperature
            scores = scores.view(n_query, -1)
            loss = self.cross_entropy(scores, target)
        else:
            scores = self.compute_similarity(query_reps, candi_reps)
            loss = None

        return MagicLensOutput(loss=loss, scores=scores, q_reps=query_reps, c_reps=candi_reps)
