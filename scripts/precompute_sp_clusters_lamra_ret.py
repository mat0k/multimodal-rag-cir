"""
Calibrate + build cluster assignments for SP (candidate-candidate) relation
distillation batching. CPU-only, from the cached LamRA-Ret target embeddings.

We want MODERATE grouping (within-cluster off-diagonal cosine ~0.25-0.40):
related but not interchangeable, to avoid the false-negative trap of tight
clusters (LaSCo labels are incomplete). Random (~0.09) is the floor ablation,
tight (~0.52) is the upper diagnostic.

Modes
-----
--calibrate            sweep several k and report the exact mean within-cluster
                       off-diagonal cosine, so we can pick k for the moderate band.
--k K --out PATH       run spherical k-means at K clusters, save
                       {subset_index: cluster_id} to PATH (+ off-diag in _meta).

Exact within-cluster mean off-diagonal cosine for unit vectors v in a cluster:
    mean_offdiag = (|sum(v)|^2 - n) / (n * (n - 1))
(since sum_{i,j} <v_i, v_j> = |sum(v)|^2, and the n diagonal terms are 1).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def _spherical_kmeans(C: torch.Tensor, k: int, seed: int, iters: int = 25) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    centroids = C[torch.randperm(C.shape[0], generator=g)[:k]].clone()
    labels = torch.zeros(C.shape[0], dtype=torch.long)
    for _ in range(iters):
        labels = (C @ centroids.T).argmax(dim=1)
        for c in range(k):
            m = labels == c
            if m.any():
                centroids[c] = F.normalize(C[m].mean(0), dim=0)
    return labels


def _mean_within_offdiag(C: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Pair-weighted mean within-cluster off-diagonal cosine (exact)."""
    num, den = 0.0, 0.0
    for c in range(k):
        m = labels == c
        n = int(m.sum())
        if n < 2:
            continue
        s = C[m].sum(0)
        offdiag_sum = float(s @ s) - n           # |sum|^2 - n
        num += offdiag_sum
        den += n * (n - 1)
    return num / max(den, 1.0)


def main(args: argparse.Namespace) -> None:
    cand_cache = PROJECT_ROOT / args.cand_emb_cache
    C = F.normalize(torch.load(cand_cache, map_location="cpu").float(), dim=-1)
    logger.info(f"Loaded candidate embeddings {tuple(C.shape)} from {cand_cache}")

    if args.calibrate:
        logger.info("Calibration sweep (target MODERATE off-diag ~0.25-0.40):")
        logger.info(f"{'k':>6}  {'mean_within_offdiag_cos':>24}  {'avg_cluster_size':>16}")
        for k in args.sweep:
            labels = _spherical_kmeans(C, k, args.seed)
            off = _mean_within_offdiag(C, labels, k)
            logger.info(f"{k:>6}  {off:>24.3f}  {C.shape[0]/k:>16.0f}")
        logger.info("random floor (no clustering) off-diag ~0.09 for reference.")
        return

    labels = _spherical_kmeans(C, args.k, args.seed)
    off = _mean_within_offdiag(C, labels, args.k)
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"_meta": {"k": args.k, "seed": args.seed,
                         "mean_within_offdiag_cos": round(off, 4),
                         "n_items": int(C.shape[0])},
               "labels": labels.tolist()}
    json.dump(payload, open(out, "w"))
    logger.info(f"Saved {args.k}-cluster labels -> {out}  (within-cluster off-diag cos={off:.3f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Calibrate/build SP cluster labels (CPU).")
    p.add_argument("--cand_emb_cache", default="data/lasco/lamra_ret_cand_emb.pt")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--sweep", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
