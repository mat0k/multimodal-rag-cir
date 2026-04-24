import json
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class LaSCo(Dataset):
    """
    LaSCo (Large Scale Composed Image Retrieval) dataset.
    Triplet format: (ref_image, modification_text, target_image).
    Images are COCO 2014 train split.

    Annotation entry format:
        {
            "qid": int,
            "query-image": [image_id, "train2014/COCO_train2014_XXXX.jpg"],
            "query-text": str,
            "target-image": [image_id, "train2014/COCO_train2014_XXXX.jpg"]
        }
    """

    def __init__(
        self,
        annotations_path: str,
        images_dir: str,
        image_transform: Optional[Callable] = None,
        caption_transform: Optional[Callable] = None,
        max_length_tokenizer: int = 77,
    ):
        super().__init__()
        self.name = "LaSCo"
        self.images_dir = Path(images_dir)
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.max_length_tokenizer = max_length_tokenizer

        with open(annotations_path, "r") as f:
            self.triplets = json.load(f)

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> dict:
        triplet = self.triplets[index]

        ref_img_path = self.images_dir / triplet["query-image"][1]
        target_img_path = self.images_dir / triplet["target-image"][1]
        text = triplet["query-text"]

        ref_image = Image.open(ref_img_path).convert("RGB")
        target_image = Image.open(target_img_path).convert("RGB")

        if self.image_transform is not None:
            ref_image = self.image_transform(ref_image, return_tensors="pt")["pixel_values"][0]
            target_image = self.image_transform(target_image, return_tensors="pt")["pixel_values"][0]

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
            "qid": triplet["qid"],
            "ref_image": ref_image,
            "target_image": target_image,
            "input_ids": _get(transformed_text, "input_ids"),
            "attention_mask": _get(transformed_text, "attention_mask"),
        }


def build_lasco_dataset(
    annotations_path: str = "data/lasco/lasco_train.json",
    images_dir: str = "data/lasco/images",
    image_transform: Optional[Callable] = None,
    caption_transform: Optional[Callable] = None,
    max_length_tokenizer: int = 77,
) -> LaSCo:
    return LaSCo(
        annotations_path=annotations_path,
        images_dir=images_dir,
        image_transform=image_transform,
        caption_transform=caption_transform,
        max_length_tokenizer=max_length_tokenizer,
    )
