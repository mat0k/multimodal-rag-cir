"""
Pre-compute Qwen3-VL-2B teacher scores over a sampled LaSCo subset.

Outputs
-------
  data/lasco/distill_subset.json      — 40K sampled triplets
  data/lasco/teacher_scores_qwen.json — per-triplet pos + K neg scores

Usage
-----
  # First run (or restart after interruption — resumes automatically):
  python scripts/precompute_teacher_scores.py \\
      --config configs/distillation/precompute_qwen.yaml

  # Override subset size or num negatives:
  python scripts/precompute_teacher_scores.py \\
      --config configs/distillation/precompute_qwen.yaml \\
      --subset_size 20000 --num_negatives 10
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rerankers.base import RerankCandidate, RerankQuery
from src.rerankers.qwen3vl_rank import Qwen3VLRanker


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logging.getLogger("transformers").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_reranker(cfg: dict) -> Qwen3VLRanker:
    rc = cfg["reranker"]
    img = rc.get("image_io", {})
    return Qwen3VLRanker(
        model_name_or_path=str(PROJECT_ROOT / rc["model_name_or_path"]),
        device=rc.get("device", "auto"),
        dtype=rc.get("dtype", "bfloat16"),
        trust_remote_code=bool(rc.get("trust_remote_code", True)),
        local_files_only=bool(rc.get("local_files_only", True)),
        revision=str(rc.get("revision", "main")),
        rgb_only=bool(img.get("rgb_only", True)),
        resize_short_edge=img.get("resize_short_edge"),
        center_crop=img.get("center_crop"),
        interpolation=str(img.get("interpolation", "bicubic")),
    )


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_subset(all_triplets: list[dict], size: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(all_triplets), size=min(size, len(all_triplets)), replace=False)
    return [all_triplets[i] for i in indices]


def _sample_negatives(
    triplet_idx: int,
    n: int,
    total: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample n negative indices from [0, total) excluding triplet_idx."""
    sample = rng.choice(total, size=n + 2, replace=False)
    filtered = [int(j) for j in sample if j != triplet_idx][:n]
    return filtered


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_triplet(
    reranker: Qwen3VLRanker,
    triplet: dict,
    neg_triplets: list[dict],
    images_dir: Path,
) -> dict:
    ref_path = str(images_dir / triplet["query-image"][1])
    pos_path = str(images_dir / triplet["target-image"][1])
    text = triplet["query-text"]

    query = RerankQuery(reference_image=ref_path, text_edit=text)
    pos_score = reranker.score(query, RerankCandidate(image=pos_path))

    neg_qids = []
    neg_scores = []
    for neg_t in neg_triplets:
        neg_path = str(images_dir / neg_t["target-image"][1])
        score = reranker.score(query, RerankCandidate(image=neg_path))
        neg_qids.append(neg_t["qid"])
        neg_scores.append(score)

    return {
        "pos_score": float(pos_score),
        "neg_qids": neg_qids,
        "neg_scores": neg_scores,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_existing_scores(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_scores(scores: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(scores, f)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    _setup_logging()
    cfg = _load_config(args.config)
    dc = cfg["data"]

    # CLI overrides
    subset_size = args.subset_size or int(dc["subset_size"])
    num_negatives = args.num_negatives or int(dc["num_negatives"])
    seed = int(dc["seed"])
    checkpoint_every = int(dc["checkpoint_every"])

    lasco_path = PROJECT_ROOT / dc["lasco_annotations"]
    images_dir = PROJECT_ROOT / dc["images_dir"]
    subset_output = PROJECT_ROOT / dc["subset_output"]
    scores_output = PROJECT_ROOT / dc["scores_output"]

    logger.info("=" * 60)
    logger.info(f"Experiment : {cfg['experiment_name']}")
    logger.info(f"Subset size: {subset_size:,}  |  Negatives/triplet: {num_negatives}")
    logger.info(f"Seed       : {seed}")
    logger.info(f"Scores out : {scores_output}")
    logger.info("=" * 60)

    # ── Load LaSCo ──────────────────────────────────────────────────────
    logger.info(f"Loading LaSCo annotations from {lasco_path} …")
    with open(lasco_path, "r") as f:
        all_triplets = json.load(f)
    logger.info(f"Total triplets: {len(all_triplets):,}")

    # ── Sample or load subset ───────────────────────────────────────────
    if subset_output.exists():
        logger.info(f"Loading existing subset from {subset_output}")
        with open(subset_output, "r") as f:
            subset = json.load(f)
        if len(subset) != subset_size:
            logger.warning(
                f"Existing subset has {len(subset):,} entries, expected {subset_size:,}. "
                "Using existing subset as-is."
            )
    else:
        logger.info(f"Sampling {subset_size:,} triplets (seed={seed}) …")
        subset = _sample_subset(all_triplets, subset_size, seed)
        subset_output.parent.mkdir(parents=True, exist_ok=True)
        with open(subset_output, "w") as f:
            json.dump(subset, f)
        logger.info(f"Subset saved → {subset_output}")

    total = len(subset)
    rng = np.random.default_rng(seed + 1)  # different seed for negatives

    # ── Load existing partial scores (resume support) ────────────────────
    scores: dict[str, dict] = _load_existing_scores(scores_output)
    already_done = len(scores)
    if already_done:
        logger.info(f"Resuming: {already_done:,} / {total:,} triplets already scored.")

    # ── Build per-triplet negative indices (deterministic) ───────────────
    # Pre-generate all negative indices so resuming produces identical sets.
    logger.info("Pre-generating negative sample indices …")
    neg_indices_all: list[list[int]] = []
    rng_neg = np.random.default_rng(seed + 1)
    for i in range(total):
        neg_indices_all.append(_sample_negatives(i, num_negatives, total, rng_neg))

    # ── Load reranker ────────────────────────────────────────────────────
    logger.info("Loading Qwen3-VL-2B reranker …")
    reranker = _build_reranker(cfg)
    logger.info("Reranker ready.")

    # ── Score ────────────────────────────────────────────────────────────
    scored_since_last_ckpt = 0
    t_start = time.time()

    for i, triplet in enumerate(subset):
        qid_str = str(triplet["qid"])
        if qid_str in scores:
            continue  # already scored — skip

        neg_triplets = [subset[j] for j in neg_indices_all[i]]

        try:
            result = _score_triplet(reranker, triplet, neg_triplets, images_dir)
        except Exception as exc:
            logger.warning(f"[{i+1}/{total}] qid={triplet['qid']} failed: {exc}")
            continue

        scores[qid_str] = result
        scored_since_last_ckpt += 1

        if scored_since_last_ckpt >= checkpoint_every:
            _save_scores(scores, scores_output)
            scored_since_last_ckpt = 0
            done = len(scores)
            elapsed = time.time() - t_start
            rate = elapsed / max(done - already_done, 1)
            eta_h = (total - done) * rate / 3600
            logger.info(
                f"Checkpoint | {done:,}/{total:,} scored | "
                f"rate={1/rate:.2f} triplets/s | ETA={eta_h:.1f}h"
            )

    # Final save
    _save_scores(scores, scores_output)
    elapsed_total = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"Done. {len(scores):,} triplets scored in {elapsed_total/3600:.2f}h")
    logger.info(f"Scores saved → {scores_output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute Qwen3-VL-2B teacher scores on a LaSCo subset."
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/distillation/precompute_qwen.yaml",
        help="Path to precompute config YAML.",
    )
    parser.add_argument("--subset_size", type=int, default=None, help="Override subset size.")
    parser.add_argument("--num_negatives", type=int, default=None, help="Override K negatives.")

    main(parser.parse_args())
