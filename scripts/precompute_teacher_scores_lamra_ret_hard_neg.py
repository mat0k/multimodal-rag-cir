"""
Pre-compute LamRA-Ret (RETRIEVER teacher) scores using retriever-mined HARD
negatives (data/lasco/hard_negatives/retriever_v2_ep1_k15.json), mined by the
VISTA contrastive-v2 ep1 student. Same hard negatives as the prior Qwen hard-neg
distillation, so the only variable vs that run is the teacher.

LaSCo ONLY — unsupervised setting.

Retriever teacher -> encode once, score by cosine. We encode:
  * every composed query (reference image + query-text)
  * every UNIQUE candidate image (positive targets + all hard negatives)
then score by dot product of L2-normalized embeddings.

Output (schema matches teacher_scores_qwen_hard_neg.json -> uses neg_img_paths):
  data/lasco/teacher_scores_lamra_ret_hard_neg.json
    { "<qid>": {"pos_score": float,
                "neg_img_paths": [str, ...],   # K hard-negative relative paths
                "neg_scores":   [float, ...]} }

Usage:
  python scripts/precompute_teacher_scores_lamra_ret_hard_neg.py \
      --config configs/distillation/precompute_lamra_ret_hard_neg.yaml
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

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


def _load_pil(images_dir: Path, rel: str) -> Image.Image:
    return Image.open(images_dir / rel).convert("RGB")


@torch.no_grad()
def _encode_queries(retriever, subset, images_dir, bs, cache_path: Path) -> torch.Tensor:
    if cache_path.exists():
        logger.info(f"[query] loading cached embeddings {cache_path}")
        return torch.load(cache_path)
    n = len(subset)
    embs, t0 = [], time.time()
    for start in range(0, n, bs):
        chunk = subset[start : start + bs]
        imgs = [_load_pil(images_dir, t["query-image"][1]) for t in chunk]
        caps = [t["query-text"] for t in chunk]
        embs.append(retriever.encode_composed(imgs, caps).cpu())
        done = min(start + bs, n)
        if (start // bs) % 50 == 0 or done == n:
            rate = (time.time() - t0) / max(done, 1)
            logger.info(f"[query] {done:,}/{n:,} | ETA {(n-done)*rate/3600:.2f}h")
    out = torch.cat(embs, 0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache_path)
    return out


@torch.no_grad()
def _encode_unique_images(retriever, paths, images_dir, bs, cache_path: Path) -> dict:
    """Encode each unique relative path once -> {path: cpu tensor [D]}."""
    if cache_path.exists():
        logger.info(f"[cand] loading cached embeddings {cache_path}")
        return torch.load(cache_path)
    uniq = sorted(set(paths))
    logger.info(f"[cand] encoding {len(uniq):,} unique candidate images")
    emb_map, t0 = {}, time.time()
    for start in range(0, len(uniq), bs):
        chunk = uniq[start : start + bs]
        imgs = [_load_pil(images_dir, p) for p in chunk]
        e = retriever.encode_images(imgs).cpu()
        for p, row in zip(chunk, e):
            emb_map[p] = row
        done = min(start + bs, len(uniq))
        if (start // bs) % 50 == 0 or done == len(uniq):
            rate = (time.time() - t0) / max(done, 1)
            logger.info(f"[cand] {done:,}/{len(uniq):,} | ETA {(len(uniq)-done)*rate/3600:.2f}h")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb_map, cache_path)
    return emb_map


def main(args: argparse.Namespace) -> None:
    _setup_logging()
    cfg = yaml.safe_load(open(args.config))
    dc, rc = cfg["data"], cfg["retriever"]

    images_dir = PROJECT_ROOT / dc["images_dir"]
    subset = json.load(open(PROJECT_ROOT / dc["subset_path"]))
    hard_negs = json.load(open(PROJECT_ROOT / dc["hard_neg_path"]))  # {qid: [paths]}
    scores_output = PROJECT_ROOT / dc["scores_output"]
    q_cache = PROJECT_ROOT / dc["query_emb_cache"]
    c_cache = PROJECT_ROOT / dc["cand_emb_cache"]
    bs = int(rc.get("batch_size", 8))

    logger.info("=" * 60)
    logger.info(f"Experiment : {cfg['experiment_name']}")
    logger.info(f"Teacher    : LamRA-Ret-Qwen2.5VL-7B (retriever) | HARD negatives")
    logger.info(f"Subset     : {len(subset):,} triplets | hard-neg entries: {len(hard_negs):,}")
    logger.info(f"Scores out : {scores_output}")
    logger.info("=" * 60)

    retriever = LamRARetRetriever(
        model_path=str(PROJECT_ROOT / rc["model_path"]),
        device="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation=rc.get("attn", "sdpa"),
        min_pixels=int(rc.get("min_pixels", 256 * 28 * 28)),
        max_pixels=int(rc.get("max_pixels", 1280 * 28 * 28)),
        cir_instruction=rc.get("instruction", DEFAULT_CIR_INSTRUCTION),
    )

    # Queries (positional, one per subset triplet)
    Q = _encode_queries(retriever, subset, images_dir, bs, q_cache)  # [N, D]

    # Unique candidate images = positive targets + all hard negatives
    pos_paths = [t["target-image"][1] for t in subset]
    all_neg_paths = [p for t in subset for p in hard_negs.get(str(t["qid"]), [])]
    emb_map = _encode_unique_images(
        retriever, pos_paths + all_neg_paths, images_dir, bs, c_cache
    )

    logger.info("Computing cosine scores …")
    scores, skipped = {}, 0
    for i, t in enumerate(subset):
        qid = str(t["qid"])
        negs = hard_negs.get(qid)
        if not negs:
            skipped += 1
            continue
        q = Q[i]
        pos_path = t["target-image"][1]
        pos_score = float(torch.dot(q, emb_map[pos_path]))
        neg_scores = [float(torch.dot(q, emb_map[p])) for p in negs]
        scores[qid] = {
            "pos_score": pos_score,
            "neg_img_paths": list(negs),
            "neg_scores": neg_scores,
        }

    scores_output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(scores, open(scores_output, "w"))

    import numpy as np
    margins = [s["pos_score"] - float(np.mean(s["neg_scores"])) for s in scores.values()]
    logger.info("=" * 60)
    logger.info(f"Done. {len(scores):,} scored ({skipped} skipped) -> {scores_output}")
    logger.info(
        f"pos mean={float(np.mean([s['pos_score'] for s in scores.values()])):.4f}  "
        f"neg mean={float(np.mean([np.mean(s['neg_scores']) for s in scores.values()])):.4f}  "
        f"avg margin={float(np.mean(margins)):.4f}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LamRA-Ret teacher scores with hard negatives.")
    parser.add_argument("--config", default="configs/distillation/precompute_lamra_ret_hard_neg.yaml")
    main(parser.parse_args())
