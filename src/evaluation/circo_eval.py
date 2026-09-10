"""CIRCO evaluation: mAP@k over a multi-ground-truth gallery.

Why this is not a copy of `cirr_eval.py`: CIRCO annotates *several* correct images per
query, so Recall@k is the wrong metric -- it cannot distinguish "found one of five" from
"found all five". CIRCO's protocol is mean Average Precision at k, which rewards ranking
every ground truth highly.

Model-agnostic, like the other eval modules: it needs only `.vision`,
`.image_processor`, `.tokenizer` and native multimodal query encoding, so VISTA and
MagicLens (and any future `TwoEncoderVLM`) run through it unchanged.

    val  -> ground truth is local, scored here.
    test -> ground truth withheld; use scripts/generate_circo_test_submission.py.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.circo import build_circo_dataset
from src.retrievers.base import TwoEncoderVLM
from src.utils.tensor import make_normalized


def _module_device(module) -> torch.device:
    return next(module.parameters()).device


def _mm_device(model: TwoEncoderVLM) -> torch.device:
    backbone = getattr(model, "backbone", None)
    if isinstance(backbone, torch.nn.Module):
        return _module_device(backbone)
    return _module_device(model.vision)


def _encode_mm_query(model, images, input_ids, attention_mask) -> torch.Tensor:
    """Native multimodal query encoding, matching the CIRR/Fashion-IQ path."""
    if hasattr(model, "encode_query_mm"):
        return make_normalized(model.encode_query_mm(
            pixel_values=images, input_ids=input_ids, attention_mask=attention_mask))
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "encode_mm"):
        return make_normalized(backbone.encode_mm(
            images, {"input_ids": input_ids, "attention_mask": attention_mask}))
    raise ValueError("model must expose encode_query_mm(...) or backbone.encode_mm(...).")


@torch.no_grad()
def generate_circo_index_features(model, index_dataset, batch_size: int = 64,
                                  num_workers: int = 8, use_tqdm: bool = True):
    """Encode the gallery once. ~123K images -- the dominant cost of a CIRCO run."""
    model.eval()
    vision = model.vision
    device = _module_device(vision)
    loader = DataLoader(index_dataset, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True)

    feats, ids = [], []
    for batch in tqdm(loader, desc="CIRCO gallery", disable=not use_tqdm):
        images = batch["image"].to(device, non_blocking=True)
        feats.append(make_normalized(vision(images).image_embeds).cpu())
        ids.extend(batch["image_name"].tolist())
    return torch.vstack(feats), ids


@torch.no_grad()
def generate_circo_query_features(model, query_dataset, batch_size: int = 64,
                                  num_workers: int = 8, use_tqdm: bool = True):
    """Encode queries; returns features plus the per-query metadata needed to score."""
    model.eval()
    device = _mm_device(model)
    loader = DataLoader(query_dataset, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True)
    # Detected per batch rather than from `dataset.split`, so Subset/wrapper datasets
    # (and anything that does not forward attributes) still work.
    has_gt = None

    feats, query_ids, ref_ids, gts = [], [], [], []
    for batch in tqdm(loader, desc="CIRCO queries", disable=not use_tqdm):
        if has_gt is None:
            has_gt = "gt_img_ids" in batch
        q = _encode_mm_query(
            model,
            batch["candidate"].to(device, non_blocking=True),
            batch["transformed_caption"].to(device, non_blocking=True),
            batch["attention_mask"].to(device, non_blocking=True),
        )
        feats.append(q.cpu())
        query_ids.extend(batch["query_id"].tolist())
        ref_ids.extend(batch["candidate_name"].tolist())
        if has_gt:
            # gt_img_ids arrives as a list of per-slot tensors; transpose to per-query
            # lists and drop the -1 padding added by the dataset.
            padded = torch.stack(batch["gt_img_ids"], dim=1).tolist()
            gts.extend([[g for g in row if g != -1] for row in padded])

    return torch.vstack(feats), query_ids, ref_ids, (gts if has_gt else None)


def average_precision(ranked_ids, gt_ids) -> float:
    """AP for one query: precision at each hit, averaged over the ground-truth set.

    Divided by len(gt_ids) -- not by the number of hits found -- so failing to
    retrieve a ground truth is penalised rather than ignored.
    """
    if not gt_ids:
        return 0.0
    gt = set(gt_ids)
    hits = 0
    precisions = []
    for rank, img_id in enumerate(ranked_ids, start=1):
        if img_id in gt:
            hits += 1
            precisions.append(hits / rank)
    return float(np.sum(precisions) / len(gt)) if precisions else 0.0


def compute_circo_metrics(index_features, index_ids, query_features, ref_ids,
                          gt_lists, k_values=(5, 10, 25, 50), exclude_reference: bool = True):
    """mAP@k over the gallery. The reference image is removed from its own ranking,
    matching CIRCO's protocol (a query must not retrieve its own input)."""
    id_to_pos = {img_id: i for i, img_id in enumerate(index_ids)}
    index_ids_arr = np.asarray(index_ids)

    sims = query_features @ index_features.T          # both L2-normalised
    if exclude_reference:
        for row, ref in enumerate(ref_ids):
            pos = id_to_pos.get(ref)
            if pos is not None:
                sims[row, pos] = -float("inf")

    max_k = max(k_values)
    top_idx = sims.topk(max_k, dim=1).indices.cpu().numpy()

    aps = {k: [] for k in k_values}
    for row, gt in enumerate(gt_lists):
        ranked = index_ids_arr[top_idx[row]]
        for k in k_values:
            aps[k].append(average_precision(ranked[:k].tolist(), gt))

    metrics = {f"val_map_at{k}": float(np.mean(aps[k]) * 100) for k in k_values}
    metrics["val_num_queries"] = len(gt_lists)
    return metrics


@torch.no_grad()
def evaluate_circo(model, split: str = "val", batch_size: int = 64, num_workers: int = 8,
                   k_values=(5, 10, 25, 50), dataset_path: str = "data/circo",
                   use_tqdm: bool = True, return_index_tuple: bool = False):
    """Score a retriever on CIRCO val. Returns {val_map_at5, val_map_at10, ...}."""
    if split != "val":
        raise ValueError("evaluate_circo scores the val split; test has no local ground "
                         "truth -- use scripts/generate_circo_test_submission.py.")

    index_ds = build_circo_dataset(split=split, mode="images", dataset_path=dataset_path,
                                   image_transform=model.image_processor,
                                   caption_transform=model.tokenizer)
    query_ds = build_circo_dataset(split=split, mode="triplets", dataset_path=dataset_path,
                                   image_transform=model.image_processor,
                                   caption_transform=model.tokenizer)

    index_features, index_ids = generate_circo_index_features(
        model, index_ds, batch_size, num_workers, use_tqdm)
    query_features, _, ref_ids, gt_lists = generate_circo_query_features(
        model, query_ds, batch_size, num_workers, use_tqdm)

    metrics = compute_circo_metrics(index_features, index_ids, query_features,
                                    ref_ids, gt_lists, k_values)
    if return_index_tuple:
        return metrics, (index_features, index_ids)
    return metrics


__all__ = ["evaluate_circo", "compute_circo_metrics", "average_precision",
           "generate_circo_index_features", "generate_circo_query_features"]
