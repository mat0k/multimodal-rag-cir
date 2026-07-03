"""
LaSCo Similarity-Preserving (SP) relation distillation dataset.

Relation-based KD (Tung & Mori, ICCV'19): distil the teacher's candidate x
candidate similarity STRUCTURE into the student. Each sample is one (query,
target) pair; the SP signal is built over the batch of target images.

__getitem__ returns:
    ref_image          reference image of the query          (transformed)
    target_image       the ground-truth target (a candidate)  (transformed)
    input_ids / attn   tokenized modification text
    teacher_target_emb cached LamRA-Ret embedding of the target (d_teacher,)
    target_id          relative image path (for CE duplicate-positive masking)
    qid                query id

Batching into semantically-related groups is done by ClusterBatchSampler using
`self.cluster_ids` (moderate k-means, ~0.34 within-cluster off-diag cosine).
Set cluster_labels_path=None for the random (floor) ablation.
"""
import json
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class LaSCoSPDistill(Dataset):
    def __init__(
        self,
        subset_path: str,
        teacher_cand_emb_path: str,
        images_dir: str,
        cluster_labels_path: Optional[str] = None,
        image_transform: Optional[Callable] = None,
        caption_transform: Optional[Callable] = None,
        max_length_tokenizer: int = 77,
    ):
        super().__init__()
        self.name = "LaSCoSPDistill"
        self.images_dir = Path(images_dir)
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.max_length_tokenizer = max_length_tokenizer

        with open(subset_path, "r") as f:
            self.subset: list[dict] = json.load(f)

        # Cached teacher target embeddings, aligned to subset order.
        self.teacher_emb: torch.Tensor = torch.load(teacher_cand_emb_path, map_location="cpu").float()
        self.teacher_emb = torch.nn.functional.normalize(self.teacher_emb, dim=-1)
        if self.teacher_emb.shape[0] != len(self.subset):
            raise ValueError(
                f"teacher emb rows {self.teacher_emb.shape[0]} != subset {len(self.subset)}"
            )

        # Cluster ids for batch sampling (None -> random floor ablation).
        self.cluster_ids: Optional[list[int]] = None
        if cluster_labels_path is not None:
            payload = json.load(open(cluster_labels_path))
            self.cluster_ids = payload["labels"]
            if len(self.cluster_ids) != len(self.subset):
                raise ValueError("cluster labels length != subset length")

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int) -> dict:
        t = self.subset[index]
        ref_image = self._load_image(t["query-image"][1])
        target_rel = t["target-image"][1]
        target_image = self._load_image(target_rel)
        input_ids, attn = self._tokenize(t["query-text"])
        return {
            "ref_image": ref_image,
            "target_image": target_image,
            "input_ids": input_ids,
            "attention_mask": attn,
            "teacher_target_emb": self.teacher_emb[index],   # [d_teacher]
            "target_id": target_rel,
            "qid": int(t["qid"]),
        }

    def _tokenize(self, text: str):
        if self.caption_transform is None:
            return text, text
        tr = self.caption_transform(
            text, padding="max_length", max_length=self.max_length_tokenizer,
            truncation=True, return_tensors="pt",
        )
        return tr["input_ids"][0], tr["attention_mask"][0]

    def _load_image(self, rel: str) -> torch.Tensor:
        pil = Image.open(self.images_dir / rel).convert("RGB")
        if self.image_transform is not None:
            return self.image_transform(pil, return_tensors="pt")["pixel_values"][0]
        return pil


def build_lasco_sp_distill_dataset(
    subset_path: str = "data/lasco/distill_subset.json",
    teacher_cand_emb_path: str = "data/lasco/lamra_ret_cand_emb.pt",
    images_dir: str = "data/lasco/images",
    cluster_labels_path: Optional[str] = None,
    image_transform: Optional[Callable] = None,
    caption_transform: Optional[Callable] = None,
    max_length_tokenizer: int = 77,
) -> LaSCoSPDistill:
    return LaSCoSPDistill(
        subset_path=subset_path,
        teacher_cand_emb_path=teacher_cand_emb_path,
        images_dir=images_dir,
        cluster_labels_path=cluster_labels_path,
        image_transform=image_transform,
        caption_transform=caption_transform,
        max_length_tokenizer=max_length_tokenizer,
    )
