"""
LaSCo validation split evaluation.

Gallery : 39,826 images from lasco_val_corpus.json (COCO val2014)
Queries : 30,037 triplets from lasco_val.json
Metrics : Recall@1 / @5 / @10 / @50 / @100

Query embedding: encode_mm(ref_image, text)  — same mode as FashionIQ/CIRR training eval.
Gallery embedding: encode_image(image)        — L2-normalised by backbone.

Similarity is computed in chunks to avoid OOM on the MIG slice.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.retrievers.backbones.vista.modeling import Visualized_BGE
from src.retrievers.vista_retriever import VistaImageProcessor


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class _CorpusDataset(Dataset):
    """39,826 gallery images from lasco_val_corpus.json."""

    def __init__(self, corpus_path: str, images_dir: Path, image_transform):
        with open(corpus_path) as f:
            self._entries = json.load(f)
        self._images_dir = images_dir
        self._transform = image_transform

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self._entries[idx]
        # corpus path is "val/val2014/..." — images live at images_dir/val2014/...
        rel = entry["path"].removeprefix("val/")
        pil = Image.open(self._images_dir / rel).convert("RGB")
        return {
            "image": self._transform(pil, return_tensors="pt")["pixel_values"][0],
            "image_id": entry["id"],
        }


class _QueryDataset(Dataset):
    """30,037 query triplets from lasco_val.json."""

    def __init__(
        self,
        val_path: str,
        images_dir: Path,
        image_transform,
        caption_transform,
    ):
        with open(val_path) as f:
            self._triplets = json.load(f)
        self._images_dir = images_dir
        self._img_tf = image_transform
        self._cap_tf = caption_transform

    def __len__(self) -> int:
        return len(self._triplets)

    def __getitem__(self, idx: int) -> dict:
        t = self._triplets[idx]
        # query-image path is already "val2014/COCO_val2014_*.jpg"
        pil = Image.open(self._images_dir / t["query-image"][1]).convert("RGB")
        tok = self._cap_tf(
            t["query-text"],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "ref_image": self._img_tf(pil, return_tensors="pt")["pixel_values"][0],
            "input_ids": tok["input_ids"][0],
            "attention_mask": tok["attention_mask"][0],
            "target_image_id": t["target-image"][0],  # int — used for Recall@K
        }


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_lasco_val(
    backbone: Visualized_BGE,
    images_dir: str = "data/lasco/images",
    val_path: str = "data/lasco/lasco_val.json",
    corpus_path: str = "data/lasco/lasco_val_corpus.json",
    batch_size: int = 64,
    num_workers: int = 4,
    k_values: Optional[list[int]] = None,
) -> dict[str, float]:
    """
    Evaluate VISTA retrieval on the LaSCo validation split.

    Returns a flat dict of Recall@K metrics, e.g.:
        {"recall_at1": 12.3, "recall_at5": 28.7, ...}
    """
    if k_values is None:
        k_values = [1, 5, 10, 50, 100]

    device = backbone.device
    backbone.eval()

    image_tf = VistaImageProcessor(backbone.preprocess_val)
    caption_tf = backbone.tokenizer
    images_dir_path = Path(images_dir)

    # ── 1. Gallery embeddings ────────────────────────────────────────────
    corpus_ds = _CorpusDataset(corpus_path, images_dir_path, image_tf)
    corpus_loader = DataLoader(
        corpus_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    gallery_feats: list[torch.Tensor] = []
    gallery_ids: list[int] = []

    for batch in tqdm(corpus_loader, desc="Gallery embeddings"):
        feats = backbone.encode_image(batch["image"].to(device))  # [B, D], L2-norm
        gallery_feats.append(feats.cpu())
        gallery_ids.extend(batch["image_id"].tolist())

    gallery_matrix = torch.cat(gallery_feats, dim=0)  # [N, D]
    gallery_ids_np = np.array(gallery_ids, dtype=np.int64)

    # ── 2. Query embeddings ──────────────────────────────────────────────
    query_ds = _QueryDataset(val_path, images_dir_path, image_tf, caption_tf)
    query_loader = DataLoader(
        query_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    query_feats: list[torch.Tensor] = []
    target_ids: list[int] = []

    for batch in tqdm(query_loader, desc="Query embeddings"):
        texts = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        feats = backbone.encode_mm(batch["ref_image"].to(device), texts)  # [B, D]
        query_feats.append(feats.cpu())
        target_ids.extend(batch["target_image_id"].tolist())

    query_matrix = torch.cat(query_feats, dim=0)   # [Q, D]
    target_ids_np = np.array(target_ids, dtype=np.int64)

    # ── 3. Recall@K (chunked to avoid OOM on 30K × 40K matrix) ─────────
    hits = {k: 0 for k in k_values}
    chunk = 512
    gallery_gpu = gallery_matrix.to(device)

    for start in tqdm(range(0, len(query_matrix), chunk), desc="Recall@K"):
        q_chunk = query_matrix[start : start + chunk].to(device)          # [C, D]
        sims = torch.matmul(q_chunk, gallery_gpu.T)                       # [C, N]
        top_idx = torch.argsort(sims, dim=1, descending=True).cpu().numpy()  # [C, N]

        for qi in range(top_idx.shape[0]):
            global_qi = start + qi
            tgt = target_ids_np[global_qi]
            ranked = gallery_ids_np[top_idx[qi]]
            for k in k_values:
                if tgt in ranked[:k]:
                    hits[k] += 1

    total = len(target_ids_np)
    metrics = {f"recall_at{k}": round(100.0 * hits[k] / total, 4) for k in k_values}

    backbone.train()
    return metrics
