"""
Pre-compute LamRA-Ret (RETRIEVER teacher) scores over the SAME 40K LaSCo subset
and the SAME negatives used for the Qwen / LamRA-Rank teachers, so the only thing
that changes versus those runs is the teacher itself. Drop-in for the CE+MSE
distillation trainer (src/datasets/lasco_distill.py).

LaSCo ONLY — unsupervised setting (we never train/distill on FashionIQ or CIRR).

Why this is cheap (unlike the reranker precompute, which did O(N*K) per-pair
forwards): a retriever teacher encodes each query and each candidate image ONCE,
then scores by cosine similarity (dot product of L2-normalized embeddings).

Score scheme (identical schema to teacher_scores_lamra.json):
    query   = composed(reference image, query-text)   -> Q[i]
    pos     = target image of triplet i               -> C[i]
    negs    = target images of K other subset triplets -> C[j]
    pos_score   = Q[i] . C[i]
    neg_scores  = [Q[i] . C[j] for j in negatives]

Output:
    data/lasco/teacher_scores_lamra_ret.json
        { "<qid>": {"pos_score": float, "neg_qids": [int...], "neg_scores": [float...]} }

Embeddings are cached to disk so the (expensive) encode pass is done only once.

Usage:
    python scripts/precompute_teacher_scores_lamra_ret.py \
        --config configs/distillation/precompute_lamra_ret.yaml
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrievers.lamra_ret_retriever import (  # noqa: E402
    DEFAULT_CIR_INSTRUCTION,
    LamRARetRetriever,
)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# --- subset + negatives: identical logic/seed to the reranker precompute -----
def _sample_subset(all_triplets: list[dict], size: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_triplets), size=min(size, len(all_triplets)), replace=False)
    return [all_triplets[i] for i in idx]


def _sample_negatives(triplet_idx: int, n: int, total: int, rng: np.random.Generator) -> list[int]:
    sample = rng.choice(total, size=n + 2, replace=False)
    return [int(j) for j in sample if j != triplet_idx][:n]


def _load_pil(images_dir: Path, rel: str) -> Image.Image:
    return Image.open(images_dir / rel).convert("RGB")


@torch.no_grad()
def _encode_all(retriever, subset, images_dir, batch_size, mode, cache_path: Path) -> torch.Tensor:
    """Encode every triplet's query (mode='query') or target image (mode='cand')."""
    if cache_path.exists():
        logger.info(f"[{mode}] loading cached embeddings from {cache_path}")
        return torch.load(cache_path)

    n = len(subset)
    embs = []
    t0 = time.time()
    for start in range(0, n, batch_size):
        chunk = subset[start : start + batch_size]
        if mode == "query":
            imgs = [_load_pil(images_dir, t["query-image"][1]) for t in chunk]
            caps = [t["query-text"] for t in chunk]
            e = retriever.encode_composed(imgs, caps)
        else:  # cand
            imgs = [_load_pil(images_dir, t["target-image"][1]) for t in chunk]
            e = retriever.encode_images(imgs)
        embs.append(e.cpu())

        done = min(start + batch_size, n)
        if (start // batch_size) % 50 == 0 or done == n:
            rate = (time.time() - t0) / max(done, 1)
            eta_h = (n - done) * rate / 3600
            logger.info(f"[{mode}] {done:,}/{n:,} encoded | {1/rate:.1f} it/s | ETA {eta_h:.2f}h")

    out = torch.cat(embs, dim=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache_path)
    logger.info(f"[{mode}] embeddings cached -> {cache_path}  shape={tuple(out.shape)}")
    return out


def main(args: argparse.Namespace) -> None:
    _setup_logging()
    cfg = yaml.safe_load(open(args.config))
    dc = cfg["data"]
    rc = cfg["retriever"]

    subset_size = int(dc["subset_size"])
    num_negatives = args.num_negatives or int(dc["num_negatives"])
    seed = int(dc["seed"])

    images_dir = PROJECT_ROOT / dc["images_dir"]
    subset_path = PROJECT_ROOT / dc["subset_path"]
    scores_output = PROJECT_ROOT / dc["scores_output"]
    q_cache = PROJECT_ROOT / dc["query_emb_cache"]
    c_cache = PROJECT_ROOT / dc["cand_emb_cache"]

    logger.info("=" * 60)
    logger.info(f"Experiment : {cfg['experiment_name']}")
    logger.info(f"Teacher    : LamRA-Ret-Qwen2.5VL-7B (retriever)")
    logger.info(f"Subset     : {subset_path}  (size {subset_size:,})")
    logger.info(f"Negatives  : K={num_negatives}  seed={seed}")
    logger.info(f"Scores out : {scores_output}")
    logger.info("=" * 60)

    # ── subset (reuse the exact 40K used for the other teachers) ────────────
    if subset_path.exists():
        subset = json.load(open(subset_path))
        logger.info(f"Loaded existing subset: {len(subset):,} triplets")
    else:
        all_triplets = json.load(open(PROJECT_ROOT / dc["lasco_annotations"]))
        subset = _sample_subset(all_triplets, subset_size, seed)
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(subset, open(subset_path, "w"))
        logger.info(f"Sampled new subset -> {subset_path}")

    total = len(subset)

    # ── identical negative sets (same seed+1 as reranker precompute) ────────
    rng_neg = np.random.default_rng(seed + 1)
    neg_indices_all = [_sample_negatives(i, num_negatives, total, rng_neg) for i in range(total)]

    # ── encode (once) ───────────────────────────────────────────────────────
    retriever = LamRARetRetriever(
        model_path=str(PROJECT_ROOT / rc["model_path"]),
        device="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation=rc.get("attn", "sdpa"),
        min_pixels=int(rc.get("min_pixels", 256 * 28 * 28)),
        max_pixels=int(rc.get("max_pixels", 1280 * 28 * 28)),
        cir_instruction=rc.get("instruction", DEFAULT_CIR_INSTRUCTION),
    )
    bs = int(rc.get("batch_size", 8))
    Q = _encode_all(retriever, subset, images_dir, bs, "query", q_cache)  # [N, D]
    C = _encode_all(retriever, subset, images_dir, bs, "cand", c_cache)   # [N, D]

    # ── score by dot product (embeddings are L2-normalized) ─────────────────
    logger.info("Computing cosine scores …")
    pos_all = (Q * C).sum(dim=-1)  # [N]
    scores: dict[str, dict] = {}
    for i, triplet in enumerate(subset):
        nidx = neg_indices_all[i]
        neg_scores = (Q[i : i + 1] @ C[nidx].T).squeeze(0).tolist()  # [K]
        scores[str(triplet["qid"])] = {
            "pos_score": float(pos_all[i]),
            "neg_qids": [int(subset[j]["qid"]) for j in nidx],
            "neg_scores": [float(s) for s in neg_scores],
        }

    scores_output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(scores, open(scores_output, "w"))
    # quick margin sanity (the diagnostic that mattered for the reranker teacher)
    margins = [s["pos_score"] - float(np.mean(s["neg_scores"])) for s in scores.values()]
    logger.info("=" * 60)
    logger.info(f"Done. {len(scores):,} triplets scored -> {scores_output}")
    logger.info(
        f"pos mean={float(pos_all.mean()):.4f}  "
        f"neg mean={float(np.mean([np.mean(s['neg_scores']) for s in scores.values()])):.4f}  "
        f"avg margin={float(np.mean(margins)):.4f}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute LamRA-Ret teacher scores on LaSCo.")
    parser.add_argument("--config", default="configs/distillation/precompute_lamra_ret.yaml")
    parser.add_argument("--num_negatives", type=int, default=None)
    main(parser.parse_args())
