"""
Pre-compute Qwen3-VL-2B teacher scores using retriever-mined hard negatives.

Unlike precompute_teacher_scores.py (random negatives), this script uses the
hard negatives already mined by the retriever (data/lasco/hard_negatives/).
Each negative is a real top-K retrieved image that is NOT the ground-truth
target — making the list much harder and more informative for ListMLE training.

Outputs
-------
  data/lasco/teacher_scores_qwen_hard_neg.json — per-triplet pos + K hard neg scores

  Score entry format:
    "<qid>": {
        "pos_score":     float,
        "neg_img_paths": [str, ...],   # relative image paths (images_dir-relative)
        "neg_scores":    [float, ...]  # teacher score per hard negative
    }

Usage
-----
  python scripts/precompute_teacher_scores_hard_neg.py \\
      --config configs/distillation/precompute_qwen_hard_neg.yaml
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rerankers.base import RerankCandidate, RerankQuery
from src.rerankers.qwen3vl_rank import Qwen3VLRanker


def _setup_logging() -> None:
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logging.getLogger("transformers").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


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


def main(args: argparse.Namespace) -> None:
    _setup_logging()
    cfg = _load_config(args.config)
    dc = cfg["data"]

    checkpoint_every = int(dc.get("checkpoint_every", 500))
    images_dir = PROJECT_ROOT / dc["images_dir"]
    subset_path = PROJECT_ROOT / dc["subset_output"]
    hard_neg_path = PROJECT_ROOT / dc["hard_neg_path"]
    scores_output = PROJECT_ROOT / dc["scores_output"]

    logger.info("=" * 60)
    logger.info(f"Experiment : {cfg['experiment_name']}")
    logger.info(f"Hard negs  : {hard_neg_path}")
    logger.info(f"Scores out : {scores_output}")
    logger.info("=" * 60)

    # Load subset
    logger.info(f"Loading subset from {subset_path} …")
    with open(subset_path, "r") as f:
        subset: list[dict] = json.load(f)
    logger.info(f"Subset size: {len(subset):,}")

    # Load hard negatives: {qid_str: [img_path, ...]}
    logger.info(f"Loading hard negatives from {hard_neg_path} …")
    with open(hard_neg_path, "r") as f:
        hard_negs: dict = json.load(f)
    # Normalise keys to str for consistent lookup
    hard_negs = {str(k): v for k, v in hard_negs.items()}
    logger.info(f"Hard neg entries: {len(hard_negs):,}")

    # Resume support
    scores: dict = _load_existing_scores(scores_output)
    already_done = len(scores)
    if already_done:
        logger.info(f"Resuming: {already_done:,} / {len(subset):,} triplets already scored.")

    # Load reranker
    logger.info("Loading Qwen3-VL-2B reranker …")
    reranker = _build_reranker(cfg)
    logger.info("Reranker ready.")

    total = len(subset)
    done = already_done
    scored_since_last_ckpt = 0
    skipped = 0
    t0 = time.time()

    for triplet in subset:
        qid = triplet["qid"]
        qid_str = str(qid)

        if qid_str in scores:
            continue  # already scored

        neg_paths = hard_negs.get(qid_str)
        if not neg_paths:
            skipped += 1
            continue  # no hard negatives for this query

        ref_path = str(images_dir / triplet["query-image"][1])
        pos_path = str(images_dir / triplet["target-image"][1])
        text = triplet["query-text"]

        query = RerankQuery(reference_image=ref_path, text_edit=text)
        pos_score = reranker.score(query, RerankCandidate(image=pos_path))

        neg_scores = []
        for neg_rel_path in neg_paths:
            neg_full_path = str(images_dir / neg_rel_path)
            neg_scores.append(float(reranker.score(query, RerankCandidate(image=neg_full_path))))

        scores[qid_str] = {
            "pos_score": float(pos_score),
            "neg_img_paths": neg_paths,
            "neg_scores": neg_scores,
        }

        done += 1
        scored_since_last_ckpt += 1

        if scored_since_last_ckpt >= checkpoint_every:
            _save_scores(scores, scores_output)
            elapsed = time.time() - t0
            rate = elapsed / max(done - already_done, 1)
            eta_h = rate * (total - done) / 3600
            logger.info(
                f"Checkpoint | {done:,}/{total:,} scored | "
                f"rate={1/rate:.2f} triplets/s | ETA={eta_h:.1f}h"
            )
            scored_since_last_ckpt = 0

    _save_scores(scores, scores_output)
    logger.info(f"Done. {done:,} triplets scored, {skipped:,} skipped (no hard negs).")
    logger.info(f"Output: {scores_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args())
