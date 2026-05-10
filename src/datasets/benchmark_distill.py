"""
FashionIQ + CIRR training split distillation dataset.

Loads pre-normalized triplets written by
scripts/precompute_teacher_scores_benchmarks.py together with
Qwen3-VL-2B teacher scores.  Returns the same VISTA-ready structure
as LaSCoDistill so the trainer, collator and loss are unchanged.

triplets.json — canonical format (list, uid == list index):
    [
      {
        "uid":      int,
        "source":   "fashioniq_dress" | "fashioniq_shirt"
                    | "fashioniq_toptee" | "cirr",
        "ref_path": str,   # relative to source-specific images_dir
        "pos_path": str,   # relative to source-specific images_dir
        "caption":  str,
      },
      ...
    ]

teacher_scores_qwen.json — keyed by uid (as string):
    {
        "<uid>": {
            "pos_score":   float,
            "neg_indices": [int, ...],   # indices into full triplets list
            "neg_scores":  [float, ...]
        },
        ...
    }
"""
import json
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class BenchmarkDistill(Dataset):
    """
    FashionIQ (dress + shirt + toptee) + CIRR training splits
    with pre-computed Qwen3-VL-2B teacher scores.

    Negative images are target images of other triplets, sampled
    across the combined gallery (cross-dataset negatives allowed).
    """

    def __init__(
        self,
        triplets_path: str,
        scores_path: str,
        fashioniq_images_dir: str,
        cirr_images_dir: str,
        image_transform: Optional[Callable] = None,
        caption_transform: Optional[Callable] = None,
        max_length_tokenizer: int = 77,
    ):
        super().__init__()
        self.name = "BenchmarkDistill"
        self.fiq_dir = Path(fashioniq_images_dir)
        self.cirr_dir = Path(cirr_images_dir)
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.max_length_tokenizer = max_length_tokenizer

        with open(triplets_path, "r") as f:
            all_triplets: list[dict] = json.load(f)

        with open(scores_path, "r") as f:
            raw: dict = json.load(f)

        self.teacher_scores: dict[int, dict] = {int(k): v for k, v in raw.items()}

        # Only keep triplets that were successfully scored (handles partial precompute)
        self.triplets = [t for t in all_triplets if t["uid"] in self.teacher_scores]

        # Full list kept for resolving negative indices (neg_indices point into all_triplets)
        self._all_triplets = all_triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> dict:
        triplet = self.triplets[index]
        uid = triplet["uid"]
        scores = self.teacher_scores[uid]

        ref_image = self._load_image(triplet["source"], triplet["ref_path"])
        pos_image = self._load_image(triplet["source"], triplet["pos_path"])

        neg_images = torch.stack([
            self._load_image(
                self._all_triplets[ni]["source"],
                self._all_triplets[ni]["pos_path"],  # target image of neg triplet = neg candidate
            )
            for ni in scores["neg_indices"]
        ])  # [K, C, H, W]

        text = triplet["caption"]
        if self.caption_transform is not None:
            transformed = self.caption_transform(
                text,
                padding="max_length",
                max_length=self.max_length_tokenizer,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = transformed["input_ids"][0]
            attention_mask = transformed["attention_mask"][0]
        else:
            input_ids = text
            attention_mask = text

        return {
            "qid": uid,
            "ref_image": ref_image,
            "pos_image": pos_image,
            "neg_images": neg_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "teacher_pos_score": torch.tensor(scores["pos_score"], dtype=torch.float32),
            "teacher_neg_scores": torch.tensor(scores["neg_scores"], dtype=torch.float32),
        }

    def _load_image(self, source: str, relative_path: str) -> torch.Tensor:
        base = self.cirr_dir if source == "cirr" else self.fiq_dir
        path = base / relative_path
        pil = Image.open(path).convert("RGB")
        if self.image_transform is not None:
            return self.image_transform(pil, return_tensors="pt")["pixel_values"][0]
        return pil


def build_benchmark_distill_dataset(
    triplets_path: str,
    scores_path: str,
    fashioniq_images_dir: str,
    cirr_images_dir: str,
    image_transform: Optional[Callable] = None,
    caption_transform: Optional[Callable] = None,
    max_length_tokenizer: int = 77,
) -> BenchmarkDistill:
    return BenchmarkDistill(
        triplets_path=triplets_path,
        scores_path=scores_path,
        fashioniq_images_dir=fashioniq_images_dir,
        cirr_images_dir=cirr_images_dir,
        image_transform=image_transform,
        caption_transform=caption_transform,
        max_length_tokenizer=max_length_tokenizer,
    )
