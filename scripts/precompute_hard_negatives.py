"""
Precompute retriever-mined hard negatives for re-ranker training.

For each query in data/lasco/distill_subset.json, runs the VISTA retriever
and saves the top-K most similar images that are NOT the true target.
These become the hard negatives for re-ranker fine-tuning.

Gallery: all unique images referenced in distill_subset.json (~50-80K images).

Output: data/lasco/hard_negatives/<name>.json
  {"<qid>": ["train2014/img1.jpg", "train2014/img2.jpg", ...], ...}
  Paths are relative to data/lasco/images/

Usage
-----
python scripts/precompute_hard_negatives.py \\
    --retriever_checkpoint results/training/vista_lasco_finetune_contrastive_v2/checkpoints/vista_epoch01.pth \\
    --top_k 15 \\
    --output data/lasco/hard_negatives/retriever_v2_ep1_k15.json \\
    --batch_size 256 \\
    --num_workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrievers.backbones.vista.modeling import Visualized_BGE
from src.retrievers.vista_retriever import VistaImageProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class _GalleryDataset(Dataset):
    """All unique images from the training subset (gallery for hard neg mining)."""

    def __init__(self, rel_paths: list[str], images_dir: Path, image_transform):
        self._paths = rel_paths
        self._images_dir = images_dir
        self._transform = image_transform

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx):
        rel = self._paths[idx]
        pil = Image.open(self._images_dir / rel).convert("RGB")
        return {
            "image": self._transform(pil, return_tensors="pt")["pixel_values"][0],
            "idx": idx,
        }


class _QueryDataset(Dataset):
    """Training queries from distill_subset.json."""

    def __init__(self, triplets: list[dict], images_dir: Path, image_transform, tokenize):
        self._triplets = triplets
        self._images_dir = images_dir
        self._img_tf = image_transform
        self._tokenize = tokenize

    def __len__(self):
        return len(self._triplets)

    def __getitem__(self, idx):
        t = self._triplets[idx]
        ref_rel = t["query-image"][1]
        pil = Image.open(self._images_dir / ref_rel).convert("RGB")
        tok = self._tokenize(
            t["query-text"],
            padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        )
        return {
            "ref_image": self._img_tf(pil, return_tensors="pt")["pixel_values"][0],
            "input_ids": tok["input_ids"][0],
            "attention_mask": tok["attention_mask"][0],
            "target_rel_path": t["target-image"][1],
            "qid": t["qid"],
        }


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

def _load_retriever(checkpoint: str | None) -> Visualized_BGE:
    backbone = Visualized_BGE(
        model_name_bge="BAAI/bge-base-en-v1.5",
        model_weight=str(PROJECT_ROOT / "models/Visualized_BGE/Visualized_base_en_v1.5.pth"),
        negatives_cross_device=False,
        from_pretrained=None,
    )
    if checkpoint is not None:
        ckpt = PROJECT_ROOT / checkpoint if not Path(checkpoint).is_absolute() else Path(checkpoint)
        state = torch.load(str(ckpt), map_location="cpu")
        backbone.load_state_dict(state)
        logger.info(f"Retriever checkpoint loaded: {ckpt}")
    else:
        logger.info("Retriever: zero-shot (base VISTA weights)")
    if torch.cuda.is_available():
        backbone = backbone.cuda()
    backbone.eval()
    return backbone


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    output_path = (
        PROJECT_ROOT / args.output
        if not Path(args.output).is_absolute()
        else Path(args.output)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"Output already exists: {output_path}  (delete to recompute)")
        return

    images_dir = PROJECT_ROOT / "data/lasco/images"
    subset_path = PROJECT_ROOT / "data/lasco/distill_subset.json"

    logger.info(f"Loading training subset: {subset_path}")
    with open(subset_path) as f:
        triplets: list[dict] = json.load(f)
    logger.info(f"  {len(triplets):,} triplets")

    # Build gallery from all unique images referenced in the subset
    gallery_paths: list[str] = []
    seen: set[str] = set()
    for t in triplets:
        for rel in (t["query-image"][1], t["target-image"][1]):
            if rel not in seen and (images_dir / rel).exists():
                seen.add(rel)
                gallery_paths.append(rel)
    path_to_idx: dict[str, int] = {p: i for i, p in enumerate(gallery_paths)}
    logger.info(f"Gallery: {len(gallery_paths):,} unique images")

    # Load retriever
    backbone = _load_retriever(args.retriever_checkpoint)
    device = next(backbone.parameters()).device
    image_tf = VistaImageProcessor(backbone.preprocess_val)
    tokenize = backbone.tokenizer

    # --- Encode gallery ---
    logger.info("Encoding gallery images …")
    gallery_ds = _GalleryDataset(gallery_paths, images_dir, image_tf)
    gallery_loader = DataLoader(
        gallery_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    t0 = time.time()
    gallery_feats = []
    with torch.no_grad():
        for batch in tqdm(gallery_loader, desc="Gallery"):
            feats = backbone.encode_image(batch["image"].to(device))
            gallery_feats.append(feats.cpu())
    gallery_matrix = torch.cat(gallery_feats, dim=0)   # [N, D]
    logger.info(f"  Done in {(time.time() - t0)/60:.1f} min — shape {tuple(gallery_matrix.shape)}")

    # --- Encode queries ---
    logger.info("Encoding training queries …")
    query_ds = _QueryDataset(triplets, images_dir, image_tf, tokenize)
    query_loader = DataLoader(
        query_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    t0 = time.time()
    query_feats: list[torch.Tensor] = []
    all_qids: list[int] = []
    all_target_rels: list[str] = []
    with torch.no_grad():
        for batch in tqdm(query_loader, desc="Queries"):
            texts = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            feats = backbone.encode_mm(batch["ref_image"].to(device), texts)
            query_feats.append(feats.cpu())
            all_qids.extend(batch["qid"].tolist())
            all_target_rels.extend(batch["target_rel_path"])
    query_matrix = torch.cat(query_feats, dim=0)   # [Q, D]
    logger.info(f"  Done in {(time.time() - t0)/60:.1f} min — shape {tuple(query_matrix.shape)}")

    del backbone
    torch.cuda.empty_cache()

    # --- Chunked similarity → top-K hard negatives per query ---
    top_k = args.top_k
    chunk = 256
    gallery_gpu = gallery_matrix.to(device)

    logger.info(f"Computing top-{top_k} hard negatives …")
    t0 = time.time()
    results: dict[str, list[str]] = {}

    for start in tqdm(range(0, len(query_matrix), chunk), desc="Top-K retrieval"):
        q_chunk = query_matrix[start : start + chunk].to(device)
        sims = torch.matmul(q_chunk, gallery_gpu.T)            # [chunk, N]
        top_indices = (
            torch.argsort(sims, dim=1, descending=True)[:, : top_k + 1].cpu().numpy()
        )

        for i in range(len(q_chunk)):
            qi = start + i
            tgt_idx = path_to_idx.get(all_target_rels[qi])
            hard_negs: list[str] = []
            for idx in top_indices[i]:
                if idx == tgt_idx:
                    continue
                hard_negs.append(gallery_paths[idx])
                if len(hard_negs) == top_k:
                    break
            results[str(all_qids[qi])] = hard_negs

    logger.info(f"  Done in {(time.time() - t0)/60:.1f} min")

    with open(output_path, "w") as f:
        json.dump(results, f)
    logger.info(f"Hard negatives saved → {output_path}  ({len(results):,} queries)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute retriever-mined hard negatives for re-ranker training."
    )
    parser.add_argument(
        "--retriever_checkpoint", type=str, default=None,
        help="Path to fine-tuned VISTA .pth checkpoint. Omit for zero-shot.",
    )
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of hard negatives to save per query (one per planned epoch).")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path (relative to project root or absolute).")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    main(parser.parse_args())
