"""
FashionIQ / CIRR relation- and feature-based distillation dataset (SUPERVISED setting).

The benchmark counterpart of src/datasets/lasco_sp_distill.py. Same item dict, same
collator (SPCollator), same losses (SP / RKD / CRD / feature) -- only the triplet
schema and the image roots differ:

    LaSCo      t["query-image"][1] / t["target-image"][1], one images_dir
    benchmarks t["ref_path"] / t["pos_path"] + t["source"], one root per source

Inputs come from scripts/precompute_bge_vl_benchmarks.py, which writes the triplets
file and the teacher embeddings already aligned: row i of `teacher_cand_emb_path`
is the target image of entry i. That alignment is asserted, not assumed -- a silent
off-by-one here would train every relation arm against the wrong teacher rows while
looking perfectly healthy.
"""
import json
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class BenchmarkSPDistill(Dataset):
    def __init__(
        self,
        triplets_path: str,
        teacher_cand_emb_path: str,
        fashioniq_images_dir: Optional[str] = None,
        cirr_images_dir: Optional[str] = None,
        cluster_labels_path: Optional[str] = None,
        image_transform: Optional[Callable] = None,
        caption_transform: Optional[Callable] = None,
        max_length_tokenizer: int = 77,
        teacher_query_emb_path: Optional[str] = None,
    ):
        super().__init__()
        self.name = "BenchmarkSPDistill"
        self.fiq_dir = Path(fashioniq_images_dir) if fashioniq_images_dir else None
        self.cirr_dir = Path(cirr_images_dir) if cirr_images_dir else None
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.max_length_tokenizer = max_length_tokenizer

        with open(triplets_path, "r") as f:
            self.triplets: list[dict] = json.load(f)

        # Target image identity, used by ClusterBatchSampler to keep a batch's
        # targets distinct. Prefixed by source: the same relative path can exist
        # under both image roots.
        self.target_ids: list[str] = [f"{t['source']}/{t['pos_path']}" for t in self.triplets]

        normalize = torch.nn.functional.normalize
        self.teacher_emb = normalize(
            torch.load(teacher_cand_emb_path, map_location="cpu").float(), dim=-1)
        if self.teacher_emb.shape[0] != len(self.triplets):
            raise ValueError(
                f"teacher emb rows {self.teacher_emb.shape[0]} != triplets "
                f"{len(self.triplets)} — regenerate with "
                f"scripts/precompute_bge_vl_benchmarks.py")

        # Feature-based KD only; None leaves SP/RKD/CRD behaviour unchanged.
        self.teacher_query_emb: Optional[torch.Tensor] = None
        if teacher_query_emb_path is not None:
            q = normalize(torch.load(teacher_query_emb_path, map_location="cpu").float(), dim=-1)
            if q.shape[0] != len(self.triplets):
                raise ValueError(
                    f"teacher query emb rows {q.shape[0]} != triplets {len(self.triplets)}")
            if q.shape[-1] != self.teacher_emb.shape[-1]:
                raise ValueError(
                    f"teacher query dim {q.shape[-1]} != candidate dim "
                    f"{self.teacher_emb.shape[-1]} — query and candidate embeddings must "
                    f"live in the SAME space or query-candidate cosine is meaningless.")
            self.teacher_query_emb = q

        # Cluster ids for batch sampling (None -> random floor, as in the LaSCo runs).
        self.cluster_ids: Optional[list[int]] = None
        if cluster_labels_path is not None:
            payload = json.load(open(cluster_labels_path))
            self.cluster_ids = payload["labels"]
            if len(self.cluster_ids) != len(self.triplets):
                raise ValueError("cluster labels length != triplets length")

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> dict:
        t = self.triplets[index]
        input_ids, attn = self._tokenize(t["caption"])
        item = {
            "ref_image": self._load_image(t["source"], t["ref_path"]),
            "target_image": self._load_image(t["source"], t["pos_path"]),
            "input_ids": input_ids,
            "attention_mask": attn,
            "teacher_target_emb": self.teacher_emb[index],      # [d_teacher]
            "target_id": self.target_ids[index],
            "qid": int(t["uid"]),
        }
        if self.teacher_query_emb is not None:
            item["teacher_query_emb"] = self.teacher_query_emb[index]
        return item

    def _tokenize(self, text: str):
        if self.caption_transform is None:
            return text, text
        tr = self.caption_transform(
            text, padding="max_length", max_length=self.max_length_tokenizer,
            truncation=True, return_tensors="pt",
        )
        return tr["input_ids"][0], tr["attention_mask"][0]

    def _load_image(self, source: str, relative_path: str):
        base = self.cirr_dir if source == "cirr" else self.fiq_dir
        if base is None:
            raise ValueError(
                f"No images_dir configured for source={source!r}. "
                "Set the matching fashioniq_images_dir / cirr_images_dir.")
        pil = Image.open(base / relative_path).convert("RGB")
        if self.image_transform is not None:
            return self.image_transform(pil, return_tensors="pt")["pixel_values"][0]
        return pil


def build_benchmark_sp_distill_dataset(
    triplets_path: str,
    teacher_cand_emb_path: str,
    fashioniq_images_dir: Optional[str] = None,
    cirr_images_dir: Optional[str] = None,
    cluster_labels_path: Optional[str] = None,
    image_transform: Optional[Callable] = None,
    caption_transform: Optional[Callable] = None,
    max_length_tokenizer: int = 77,
    teacher_query_emb_path: Optional[str] = None,
) -> BenchmarkSPDistill:
    return BenchmarkSPDistill(
        triplets_path=triplets_path,
        teacher_cand_emb_path=teacher_cand_emb_path,
        fashioniq_images_dir=fashioniq_images_dir,
        cirr_images_dir=cirr_images_dir,
        cluster_labels_path=cluster_labels_path,
        image_transform=image_transform,
        caption_transform=caption_transform,
        max_length_tokenizer=max_length_tokenizer,
        teacher_query_emb_path=teacher_query_emb_path,
    )


__all__ = ["BenchmarkSPDistill", "build_benchmark_sp_distill_dataset"]
