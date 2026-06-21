"""
FashionIQ / CIRR contrastive (supervised) training dataset.

Reads the canonical benchmark triplets written by
scripts/precompute_teacher_scores_benchmarks.py and returns the same
VISTA-ready structure as LaSCo (ref image + target image + tokenized text),
so the existing contrastive trainer/collator/loss work unchanged.

Unlike BenchmarkDistill, this carries NO teacher scores and NO precomputed
negatives — the contrastive loss uses in-batch negatives. Filtering by
`sources` lets you train on a single dataset (FashionIQ-only or CIRR-only);
in-batch negatives are then automatically single-domain.

triplets.json entry format:
    {
        "uid":      int,
        "source":   "fashioniq_dress" | "fashioniq_shirt"
                    | "fashioniq_toptee" | "cirr",
        "ref_path": str,   # relative to source-specific images_dir
        "pos_path": str,   # target image, relative to source-specific images_dir
        "caption":  str,
    }
"""
import json
from pathlib import Path
from typing import Callable, Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset


class BenchmarkContrastive(Dataset):
    def __init__(
        self,
        triplets_path: str,
        fashioniq_images_dir: Optional[str] = None,
        cirr_images_dir: Optional[str] = None,
        sources: Optional[Sequence[str]] = None,
        image_transform: Optional[Callable] = None,
        caption_transform: Optional[Callable] = None,
        max_length_tokenizer: int = 77,
    ):
        super().__init__()
        self.name = "BenchmarkContrastive"
        # Only the dir(s) relevant to the filtered `sources` need to be set;
        # FashionIQ-only and CIRR-only runs each specify just their own dir.
        self.fiq_dir = Path(fashioniq_images_dir) if fashioniq_images_dir else None
        self.cirr_dir = Path(cirr_images_dir) if cirr_images_dir else None
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.max_length_tokenizer = max_length_tokenizer

        with open(triplets_path, "r") as f:
            all_triplets: list[dict] = json.load(f)

        if sources is not None:
            allowed = set(sources)
            triplets = [t for t in all_triplets if t["source"] in allowed]
        else:
            triplets = all_triplets

        # Some FashionIQ images referenced in the triplet list are absent on disk;
        # the teacher precompute silently skipped them, so drop them here too to
        # keep this dataset consistent with the distillation runs.
        kept = [t for t in triplets if self._exists(t["source"], t["ref_path"])
                and self._exists(t["source"], t["pos_path"])]
        n_dropped = len(triplets) - len(kept)
        if n_dropped:
            import logging
            logging.getLogger(__name__).warning(
                f"BenchmarkContrastive: dropped {n_dropped:,}/{len(triplets):,} "
                f"triplets with missing images."
            )
        self.triplets = kept

        if not self.triplets:
            raise ValueError(
                f"No triplets after filtering by sources={sources}. "
                f"Available sources: {sorted({t['source'] for t in all_triplets})}"
            )

    def _exists(self, source: str, relative_path: str) -> bool:
        base = self.cirr_dir if source == "cirr" else self.fiq_dir
        return base is not None and (base / relative_path).is_file()

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> dict:
        triplet = self.triplets[index]

        ref_image = self._load_image(triplet["source"], triplet["ref_path"])
        target_image = self._load_image(triplet["source"], triplet["pos_path"])

        text = triplet["caption"]
        transformed_text = text
        if self.caption_transform is not None:
            transformed_text = self.caption_transform(
                text,
                padding="max_length",
                max_length=self.max_length_tokenizer,
                truncation=True,
                return_tensors="pt",
            )

        def _get(tc, field):
            if hasattr(tc, "keys") and field in tc:
                return tc[field][0]
            return tc

        return {
            "qid": triplet["uid"],
            "ref_image": ref_image,
            "target_image": target_image,
            "input_ids": _get(transformed_text, "input_ids"),
            "attention_mask": _get(transformed_text, "attention_mask"),
        }

    def _load_image(self, source: str, relative_path: str):
        base = self.cirr_dir if source == "cirr" else self.fiq_dir
        if base is None:
            raise ValueError(
                f"No images_dir configured for source={source!r}. "
                "Set the matching fashioniq_images_dir / cirr_images_dir."
            )
        pil = Image.open(base / relative_path).convert("RGB")
        if self.image_transform is not None:
            return self.image_transform(pil, return_tensors="pt")["pixel_values"][0]
        return pil


def build_benchmark_contrastive_dataset(
    triplets_path: str,
    fashioniq_images_dir: Optional[str] = None,
    cirr_images_dir: Optional[str] = None,
    sources: Optional[Sequence[str]] = None,
    image_transform: Optional[Callable] = None,
    caption_transform: Optional[Callable] = None,
    max_length_tokenizer: int = 77,
) -> BenchmarkContrastive:
    return BenchmarkContrastive(
        triplets_path=triplets_path,
        fashioniq_images_dir=fashioniq_images_dir,
        cirr_images_dir=cirr_images_dir,
        sources=sources,
        image_transform=image_transform,
        caption_transform=caption_transform,
        max_length_tokenizer=max_length_tokenizer,
    )
