"""CIRCO dataset (Baldrati et al., 2023).

Composed image retrieval over ~123K unlabelled COCO images. Two properties make it
differ from CIRR/Fashion-IQ and shape this loader:

* **Multiple ground truths per query.** `gt_img_ids` lists every image a human judged
  correct, so the metric is mAP@k rather than Recall@k (see `src/evaluation/circo_eval.py`).
* **A ~50x larger gallery** (123,403 images vs CIRR's 2,297), which dominates runtime.

Splits: `val` carries ground truth and is scored locally; `test` withholds it and is
scored by the official server at https://circo.micc.unifi.it/.

Mirrors the `build_cirr_dataset` interface -- same `mode`, same transform arguments,
same item-dict shape -- so the existing model-agnostic eval path works unchanged.
"""

import json
import os
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class CIRCO(Dataset):
    """CIRCO in two modes: 'images' (the gallery) and 'triplets' (the queries)."""

    def __init__(
        self,
        dataset_path: str = "data/circo",
        split: str = "val",
        image_transform: Optional[Callable] = None,
        caption_transform: Optional[Callable] = None,
        max_length_tokenizer: int = 77,
        mode: str = "triplets",
        use_shared_concept: bool = True,
    ):
        if split not in ("val", "test"):
            raise ValueError(f"split must be 'val' or 'test', got {split!r}")
        if mode not in ("triplets", "images"):
            raise ValueError(f"mode must be 'triplets' or 'images', got {mode!r}")

        self.dataset_path = dataset_path
        self.split = split
        self.mode = mode
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.max_length_tokenizer = max_length_tokenizer
        self.use_shared_concept = use_shared_concept

        # The gallery is the full COCO unlabeled-2017 set, described by its own
        # info file; image ids are NOT the file names, hence the two maps below.
        img_info_path = os.path.join(
            dataset_path, "COCO2017_unlabeled", "annotations", "image_info_unlabeled2017.json"
        )
        with open(img_info_path) as f:
            imgs_info = json.load(f)["images"]

        images_dir = os.path.join(dataset_path, "COCO2017_unlabeled", "unlabeled2017")
        self.img_paths = [os.path.join(images_dir, i["file_name"]) for i in imgs_info]
        self.img_ids = [i["id"] for i in imgs_info]
        # Position of each image id in the gallery, so retrieved indices can be
        # mapped back to ids when scoring or writing a submission.
        self.img_id_to_index = {img_id: i for i, img_id in enumerate(self.img_ids)}

        with open(os.path.join(dataset_path, "annotations", f"{split}.json")) as f:
            self.annotations = json.load(f)

    def __len__(self) -> int:
        return len(self.img_paths) if self.mode == "images" else len(self.annotations)

    def _load_image(self, path: str):
        image = Image.open(path).convert("RGB")
        if self.image_transform is not None:
            return self.image_transform(image, return_tensors="pt")["pixel_values"][0]
        return image

    def _query_text(self, ann: dict) -> str:
        """Build the query string.

        CIRCO's `relative_caption` states only how the target DIFFERS from the
        reference ("shows two people and has a more colorful background") -- it never
        names the subject. `shared_concept` supplies that ("a girl with a traditional
        Chinese umbrella"). Over a 123K open-domain gallery the caption alone is close
        to unanswerable, so evaluations combine the two. This follows the official
        MagicLens protocol (magiclens_reference/data_utils.py::build_circo_dataset):

            f"find {shared_concept} but {relative_caption}"

        Dropping the concept costs roughly half the mAP, so this must match whatever
        protocol the numbers are being compared against.
        """
        if self.use_shared_concept and ann.get("shared_concept"):
            return f"find {ann['shared_concept']} but {ann['relative_caption']}"
        return ann["relative_caption"]

    @staticmethod
    def _field(tokenized, name):
        """Pull a field out of a tokenizer output, tolerating raw strings."""
        if hasattr(tokenized, "keys") and name in tokenized:
            return tokenized[name][0]
        return tokenized

    def __getitem__(self, index: int) -> dict:
        if self.mode == "images":
            return {
                "image_name": self.img_ids[index],   # int id, not a filename
                "image": self._load_image(self.img_paths[index]),
            }

        ann = self.annotations[index]
        reference = self._load_image(
            self.img_paths[self.img_id_to_index[ann["reference_img_id"]]]
        )

        caption = self._query_text(ann)
        transformed = caption
        if self.caption_transform is not None:
            transformed = self.caption_transform(
                caption,
                padding="max_length",
                max_length=self.max_length_tokenizer,
                truncation=True,
                return_tensors="pt",
            )

        item = {
            "query_id": ann["id"],
            "candidate": reference,
            "candidate_name": ann["reference_img_id"],
            "transformed_caption": self._field(transformed, "input_ids"),
            "attention_mask": self._field(transformed, "attention_mask"),
            "caption": caption,
        }

        if self.split == "val":
            # Padded to a fixed width so default collation can batch it; -1 marks
            # unused slots and is filtered out when scoring.
            gts = list(ann["gt_img_ids"])
            item["target_name"] = ann["target_img_id"]
            item["gt_img_ids"] = gts + [-1] * (self.MAX_GT - len(gts))
            item["n_gt"] = len(gts)
        return item

    # Largest gt_img_ids list in CIRCO; fixed so batches have uniform shape.
    MAX_GT = 23


def build_circo_dataset(
    split: str = "val",
    image_transform: Optional[Callable] = None,
    caption_transform: Optional[Callable] = None,
    max_length_tokenizer: int = 77,
    mode: str = "triplets",
    dataset_path: str = "data/circo",
    use_shared_concept: bool = True,
) -> CIRCO:
    return CIRCO(
        dataset_path=dataset_path,
        split=split,
        image_transform=image_transform,
        caption_transform=caption_transform,
        max_length_tokenizer=max_length_tokenizer,
        mode=mode,
        use_shared_concept=use_shared_concept,
    )


__all__ = ["CIRCO", "build_circo_dataset"]
