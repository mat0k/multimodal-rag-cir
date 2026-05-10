"""
LaSCo distillation dataset.

Loads a pre-sampled LaSCo subset together with Qwen3-VL-2B teacher scores
produced by scripts/precompute_teacher_scores.py.

Each sample returns the VISTA-ready inputs for one query plus its ground-truth
positive image and K pre-scored negative images, along with the teacher's
pairwise relevance scores used to compute MarginMSE.
"""
import json
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class LaSCoDistill(Dataset):
    """
    LaSCo subset dataset with pre-computed teacher scores.

    Annotation entry format (distill_subset.json, same as lasco_train.json):
        {
            "qid": int,
            "query-image":  [image_id, "train2014/COCO_train2014_XXXX.jpg"],
            "query-text":   str,
            "target-image": [image_id, "train2014/COCO_train2014_XXXX.jpg"]
        }

    Teacher scores format (teacher_scores_qwen.json):
        {
            "<qid>": {
                "pos_score":   float,
                "neg_qids":    [int, ...],   # length K
                "neg_scores":  [float, ...]  # length K
            },
            ...
        }
    """

    def __init__(
        self,
        subset_path: str,
        scores_path: str,
        images_dir: str,
        image_transform: Optional[Callable] = None,
        caption_transform: Optional[Callable] = None,
        max_length_tokenizer: int = 77,
    ):
        super().__init__()
        self.name = "LaSCoDistill"
        self.images_dir = Path(images_dir)
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.max_length_tokenizer = max_length_tokenizer

        with open(subset_path, "r") as f:
            all_triplets: list[dict] = json.load(f)

        with open(scores_path, "r") as f:
            teacher_scores_raw: dict = json.load(f)

        self.teacher_scores: dict[int, dict] = {
            int(k): v for k, v in teacher_scores_raw.items()
        }

        # Keep only triplets that have been scored (precompute may be partial)
        self.triplets = [t for t in all_triplets if t["qid"] in self.teacher_scores]

        # Build qid → triplet lookup for loading negative images
        self.by_qid: dict[int, dict] = {t["qid"]: t for t in all_triplets}

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> dict:
        triplet = self.triplets[index]
        qid = triplet["qid"]
        scores = self.teacher_scores[qid]

        ref_image = self._load_image(triplet["query-image"][1])
        pos_image = self._load_image(triplet["target-image"][1])

        neg_images = torch.stack([
            self._load_image(self.by_qid[neg_qid]["target-image"][1])
            for neg_qid in scores["neg_qids"]
        ])  # [K, C, H, W]

        text = triplet["query-text"]
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
            "qid": qid,
            "ref_image": ref_image,
            "pos_image": pos_image,
            "neg_images": neg_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "teacher_pos_score": torch.tensor(scores["pos_score"], dtype=torch.float32),
            "teacher_neg_scores": torch.tensor(scores["neg_scores"], dtype=torch.float32),
        }

    def _load_image(self, relative_path: str) -> torch.Tensor:
        path = self.images_dir / relative_path
        pil = Image.open(path).convert("RGB")
        if self.image_transform is not None:
            return self.image_transform(pil, return_tensors="pt")["pixel_values"][0]
        return pil


def build_lasco_distill_dataset(
    subset_path: str = "data/lasco/distill_subset.json",
    scores_path: str = "data/lasco/teacher_scores_qwen.json",
    images_dir: str = "data/lasco/images",
    image_transform: Optional[Callable] = None,
    caption_transform: Optional[Callable] = None,
    max_length_tokenizer: int = 77,
) -> LaSCoDistill:
    return LaSCoDistill(
        subset_path=subset_path,
        scores_path=scores_path,
        images_dir=images_dir,
        image_transform=image_transform,
        caption_transform=caption_transform,
        max_length_tokenizer=max_length_tokenizer,
    )
