"""
LaSCo relation-based distillation dataset (group-matrix).

Unlike `lasco_distill.py` (per-query pool: 1 positive + K private negatives),
this dataset returns a whole GROUP of G queries that share a candidate pool —
the group's own G target images — together with the teacher's full G x G
relevance matrix produced by scripts/precompute_teacher_scores_relation.py.

One __getitem__ returns one group:
    qids            : [G]
    ref_images      : [G, C, H, W]   reference image of each query
    cand_images     : [G, C, H, W]   shared candidate pool (group targets)
    input_ids       : [G, T]         tokenized query text
    attention_mask  : [G, T]
    teacher_matrix  : [G, G]         matrix[i][j] = teacher P(yes)(q_i, c_j)

The diagonal teacher_matrix[i][i] is the true positive for query i; the
off-diagonals are shared in-group negatives. Because the pool is shared, the
matrix can be distilled row-wise (query -> candidates) AND column-wise
(candidate -> queries) — the latter being the relational signal that a
per-query pool cannot express.
"""
import json
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class LaSCoRelationDistill(Dataset):
    """LaSCo group-matrix dataset with pre-computed teacher relevance matrices.

    Subset format (relation_subset.json): a flat list of triplets, same schema
    as lasco_train.json. Group g is the contiguous slice [g*G : (g+1)*G].

    Matrix format (teacher_matrix_qwen.json):
        {
          "_meta": {"group_size": G, "n_groups": int, ...},
          "<group_id>": {"qids": [G], "matrix": [[G floats] x G]},
          ...
        }
    """

    def __init__(
        self,
        subset_path: str,
        matrix_path: str,
        images_dir: str,
        image_transform: Optional[Callable] = None,
        caption_transform: Optional[Callable] = None,
        max_length_tokenizer: int = 77,
    ):
        super().__init__()
        self.name = "LaSCoRelationDistill"
        self.images_dir = Path(images_dir)
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.max_length_tokenizer = max_length_tokenizer

        with open(subset_path, "r") as f:
            self.subset: list[dict] = json.load(f)

        with open(matrix_path, "r") as f:
            raw: dict = json.load(f)

        meta = raw.get("_meta", {})
        self.group_size: int = int(meta.get("group_size", 0))
        if self.group_size <= 0:
            raise ValueError(
                f"Could not read group_size from {matrix_path} _meta. "
                "Re-run the relation precompute script."
            )
        if len(self.subset) % self.group_size != 0:
            raise ValueError(
                f"Subset size {len(self.subset)} is not a multiple of "
                f"group_size {self.group_size}."
            )

        # Keep only groups that were actually scored (precompute may be partial).
        self.matrices: dict[int, dict] = {
            int(k): v for k, v in raw.items() if k != "_meta"
        }
        self.group_ids: list[int] = sorted(self.matrices.keys())

    def __len__(self) -> int:
        return len(self.group_ids)

    def __getitem__(self, index: int) -> dict:
        gid = self.group_ids[index]
        entry = self.matrices[gid]
        G = self.group_size

        group = self.subset[gid * G : (gid + 1) * G]

        ref_images = torch.stack([
            self._load_image(t["query-image"][1]) for t in group
        ])  # [G, C, H, W]
        cand_images = torch.stack([
            self._load_image(t["target-image"][1]) for t in group
        ])  # [G, C, H, W]  — shared pool (group targets)

        input_ids_list, attn_list = [], []
        for t in group:
            ids, mask = self._tokenize(t["query-text"])
            input_ids_list.append(ids)
            attn_list.append(mask)

        return {
            "qids": torch.tensor(entry["qids"], dtype=torch.long),
            "ref_images": ref_images,
            "cand_images": cand_images,
            "input_ids": torch.stack(input_ids_list),       # [G, T]
            "attention_mask": torch.stack(attn_list),        # [G, T]
            "teacher_matrix": torch.tensor(entry["matrix"], dtype=torch.float32),  # [G, G]
        }

    def _tokenize(self, text: str):
        if self.caption_transform is None:
            return text, text
        transformed = self.caption_transform(
            text,
            padding="max_length",
            max_length=self.max_length_tokenizer,
            truncation=True,
            return_tensors="pt",
        )
        return transformed["input_ids"][0], transformed["attention_mask"][0]

    def _load_image(self, relative_path: str) -> torch.Tensor:
        path = self.images_dir / relative_path
        pil = Image.open(path).convert("RGB")
        if self.image_transform is not None:
            return self.image_transform(pil, return_tensors="pt")["pixel_values"][0]
        return pil


def build_lasco_relation_distill_dataset(
    subset_path: str = "data/lasco/relation_subset.json",
    matrix_path: str = "data/lasco/teacher_matrix_qwen.json",
    images_dir: str = "data/lasco/images",
    image_transform: Optional[Callable] = None,
    caption_transform: Optional[Callable] = None,
    max_length_tokenizer: int = 77,
) -> LaSCoRelationDistill:
    return LaSCoRelationDistill(
        subset_path=subset_path,
        matrix_path=matrix_path,
        images_dir=images_dir,
        image_transform=image_transform,
        caption_transform=caption_transform,
        max_length_tokenizer=max_length_tokenizer,
    )
