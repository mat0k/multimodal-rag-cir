"""
Pre-compute Qwen3-VL-2B teacher scores over FashionIQ + CIRR training splits.

This is the benchmark counterpart of scripts/precompute_teacher_scores.py
(which operates on a sampled LaSCo subset).  Data loading and path resolution
differ; the Qwen scoring logic and checkpoint/resume pattern are the same.

Outputs
-------
  data/benchmarks_distill/triplets.json            — normalized triplets list
  data/benchmarks_distill/teacher_scores_qwen.json — per-triplet pos + K neg scores

Triplets are produced from:
  - FashionIQ train: dress + shirt + toptee  (~18K)
  - CIRR train                               (~28K)
  Combined total                              ~46K

Usage
-----
  # Default run (or resume after interruption — auto-resumes):
  python scripts/precompute_teacher_scores_benchmarks.py \\
      --config configs/distillation/precompute_benchmarks.yaml

  # Override K:
  python scripts/precompute_teacher_scores_benchmarks.py \\
      --config configs/distillation/precompute_benchmarks.yaml \\
      --num_negatives 10
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
    logging.basicConfig(
        level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / reranker
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
# Data loading + normalization
# ---------------------------------------------------------------------------

def _load_fashioniq_triplets(
    fashioniq_dir: Path,
    caption_mode: str,
    caption_sep: str,
) -> list[dict]:
    """
    Load FashionIQ train triplets from all three categories.

    caption_mode choices (logged to run config):
      "concat" — join both captions with caption_sep  (default, most common in literature)
      "first"  — use captions[0] only
      "second" — use captions[1] only

    Paths stored relative to fashioniq images dir:
      "{category}/{image_id}.jpg"
    """
    triplets = []
    for cat in ("dress", "shirt", "toptee"):
        ann_path = fashioniq_dir / "annotations" / f"cap.{cat}.train.json"
        with open(ann_path) as f:
            entries = json.load(f)

        for e in entries:
            if caption_mode == "concat":
                caption = caption_sep.join(e["captions"])
            elif caption_mode == "first":
                caption = e["captions"][0]
            else:  # "second"
                caption = e["captions"][1]

            triplets.append({
                "source": f"fashioniq_{cat}",
                "ref_path": f"{cat}/{e['candidate']}.jpg",  # query / reference image
                "pos_path": f"{cat}/{e['target']}.jpg",     # positive target image
                "caption": caption,
            })

    return triplets


def _load_cirr_triplets(cirr_dir: Path) -> list[dict]:
    """
    Load CIRR train triplets.

    Paths stored relative to cirr images dir:
      "train/{bucket}/{image_id}.png"
    """
    ann_path = cirr_dir / "annotations" / "captions" / "cap.rc2.train.json"
    split_path = cirr_dir / "annotations" / "image_splits" / "split.rc2.train.json"

    with open(ann_path) as f:
        entries = json.load(f)
    with open(split_path) as f:
        id_to_path: dict[str, str] = json.load(f)  # image_id → "./train/XX/image_id.png"

    def resolve(image_id: str) -> str:
        return id_to_path[image_id].lstrip("./")  # drop leading "./"

    triplets = []
    for e in entries:
        triplets.append({
            "source": "cirr",
            "ref_path": resolve(e["reference"]),
            "pos_path": resolve(e["target_hard"]),
            "caption": e["caption"],
        })

    return triplets


def _build_and_save_triplets(cfg: dict, triplets_output: Path) -> list[dict]:
    """Build normalized triplet list and save to disk (idempotent)."""
    if triplets_output.exists():
        logger.info(f"Loading existing triplets from {triplets_output}")
        with open(triplets_output) as f:
            return json.load(f)

    dc = cfg["data"]
    fashioniq_dir = PROJECT_ROOT / dc["fashioniq_dir"]
    cirr_dir = PROJECT_ROOT / dc["cirr_dir"]
    caption_mode = dc.get("fashioniq_caption_mode", "concat")
    caption_sep = dc.get("fashioniq_caption_separator", ", ")

    logger.info(f"FashionIQ caption_mode : {caption_mode!r}  sep={caption_sep!r}")

    fiq_triplets = _load_fashioniq_triplets(fashioniq_dir, caption_mode, caption_sep)
    cirr_triplets = _load_cirr_triplets(cirr_dir)
    combined = fiq_triplets + cirr_triplets

    # Assign sequential UIDs = list indices (stable across resume runs)
    for uid, t in enumerate(combined):
        t["uid"] = uid

    logger.info(f"FashionIQ train triplets : {len(fiq_triplets):,}")
    logger.info(f"CIRR     train triplets  : {len(cirr_triplets):,}")
    logger.info(f"Combined total           : {len(combined):,}")

    triplets_output.parent.mkdir(parents=True, exist_ok=True)
    with open(triplets_output, "w") as f:
        json.dump(combined, f)
    logger.info(f"Triplets saved → {triplets_output}")

    return combined


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def _sample_negatives(
    idx: int,
    n: int,
    total: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample n negative indices from [0, total) excluding idx."""
    sample = rng.choice(total, size=n + 2, replace=False)
    return [int(j) for j in sample if j != idx][:n]


def _sample_negatives_from_pool(
    idx: int,
    n: int,
    pool: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    """Sample n negative indices from `pool` (global indices) excluding idx.

    Used for single-domain negatives: pool is the set of triplet indices that
    share the same source as `idx`. Returned indices are still global indices
    into the full triplets list, so BenchmarkDistill resolves them unchanged.
    """
    sample = rng.choice(pool, size=min(n + 2, len(pool)), replace=False)
    return [int(j) for j in sample if j != idx][:n]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _resolve_path(triplet: dict, fashioniq_images_dir: Path, cirr_images_dir: Path) -> tuple[str, str]:
    """Return (ref_full_path, pos_full_path) for a triplet."""
    base = cirr_images_dir if triplet["source"] == "cirr" else fashioniq_images_dir
    return str(base / triplet["ref_path"]), str(base / triplet["pos_path"])


def _score_triplet(
    reranker: Qwen3VLRanker,
    triplet: dict,
    neg_triplets: list[dict],
    neg_indices: list[int],
    fashioniq_images_dir: Path,
    cirr_images_dir: Path,
) -> dict:
    ref_path, pos_path = _resolve_path(triplet, fashioniq_images_dir, cirr_images_dir)

    query = RerankQuery(reference_image=ref_path, text_edit=triplet["caption"])
    pos_score = reranker.score(query, RerankCandidate(image=pos_path))

    neg_scores = []
    for neg_t in neg_triplets:
        _, neg_img_path = _resolve_path(neg_t, fashioniq_images_dir, cirr_images_dir)
        score = reranker.score(query, RerankCandidate(image=neg_img_path))
        neg_scores.append(score)

    return {
        "pos_score": float(pos_score),
        "neg_indices": neg_indices,   # store indices (not qids) for cross-dataset negatives
        "neg_scores": neg_scores,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_existing_scores(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
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

    num_negatives = args.num_negatives or int(dc["num_negatives"])
    seed = int(dc["seed"])
    checkpoint_every = int(dc["checkpoint_every"])

    fashioniq_images_dir = PROJECT_ROOT / dc["fashioniq_dir"] / "images"
    cirr_images_dir = PROJECT_ROOT / dc["cirr_dir"] / "images"
    triplets_output = PROJECT_ROOT / dc["triplets_output"]
    scores_output = PROJECT_ROOT / dc["scores_output"]

    logger.info("=" * 60)
    logger.info(f"Experiment       : {cfg['experiment_name']}")
    logger.info(f"Negatives/triplet: {num_negatives}")
    logger.info(f"Seed             : {seed}")
    logger.info(f"Scores output    : {scores_output}")
    logger.info("=" * 60)

    # ── Build or load triplets ──────────────────────────────────────────────
    triplets = _build_and_save_triplets(cfg, triplets_output)
    total = len(triplets)

    # Optional: restrict which triplets are scored (e.g. FashionIQ-only) and
    # whether negatives are sampled within the same source (single-domain).
    restrict_sources = dc.get("restrict_sources")            # None or list[str]
    single_domain_negatives = bool(dc.get("single_domain_negatives", False))
    skip_missing_images = bool(dc.get("skip_missing_images", True))

    # Drop triplets whose ref OR pos image is absent on disk, from BOTH the
    # score set and the negative pools. Negatives use the pos image of other
    # triplets, so leaving missing-image triplets in the pool would cause many
    # otherwise-valid triplets to be skipped (any one bad negative skips the
    # whole triplet) and could crash the downstream BenchmarkDistill loader.
    def _img_ok(tp: dict) -> bool:
        base = cirr_images_dir if tp["source"] == "cirr" else fashioniq_images_dir
        return (base / tp["ref_path"]).is_file() and (base / tp["pos_path"]).is_file()

    if skip_missing_images:
        valid = {i for i in range(total) if _img_ok(triplets[i])}
        logger.info(
            f"Image existence filter: {len(valid):,}/{total:,} triplets have "
            f"both ref+pos images present"
        )
    else:
        valid = set(range(total))

    if restrict_sources:
        allowed = set(restrict_sources)
        score_indices = [
            i for i in range(total)
            if triplets[i]["source"] in allowed and i in valid
        ]
        logger.info(
            f"Restricting to sources={sorted(allowed)} → "
            f"{len(score_indices):,}/{total:,} triplets to score"
        )
    else:
        score_indices = [i for i in range(total) if i in valid]

    # ── Pre-generate negative index sets (deterministic) ────────────────────
    logger.info(
        f"Pre-generating negative sample indices "
        f"(single_domain={single_domain_negatives}) …"
    )
    rng_neg = np.random.default_rng(seed + 1)
    if single_domain_negatives:
        # Pool of global indices per source (valid images only)
        source_pools: dict[str, np.ndarray] = {}
        for src in {triplets[i]["source"] for i in score_indices}:
            source_pools[src] = np.array(
                [i for i in range(total) if triplets[i]["source"] == src and i in valid]
            )
        neg_indices_map: dict[int, list[int]] = {
            i: _sample_negatives_from_pool(
                i, num_negatives, source_pools[triplets[i]["source"]], rng_neg
            )
            for i in score_indices
        }
    else:
        neg_indices_map = {
            i: _sample_negatives(i, num_negatives, total, rng_neg)
            for i in score_indices
        }

    # ── Load existing partial scores (resume support) ───────────────────────
    scores: dict[str, dict] = _load_existing_scores(scores_output)
    already_done = len(scores)
    if already_done:
        logger.info(f"Resuming: {already_done:,} / {total:,} triplets already scored.")

    # ── Load reranker ────────────────────────────────────────────────────────
    logger.info("Loading Qwen3-VL-2B reranker …")
    reranker = _build_reranker(cfg)
    logger.info("Reranker ready.")

    # ── Score ────────────────────────────────────────────────────────────────
    scored_since_last_ckpt = 0
    t_start = time.time()

    for i in score_indices:
        triplet = triplets[i]
        uid_str = str(triplet["uid"])
        if uid_str in scores:
            continue  # already scored — skip

        neg_idx = neg_indices_map[i]
        neg_triplets = [triplets[j] for j in neg_idx]

        try:
            result = _score_triplet(
                reranker, triplet, neg_triplets, neg_idx,
                fashioniq_images_dir, cirr_images_dir,
            )
        except Exception as exc:
            logger.warning(f"[{i+1}/{total}] uid={triplet['uid']} ({triplet['source']}) failed: {exc}")
            continue

        scores[uid_str] = result
        scored_since_last_ckpt += 1

        if scored_since_last_ckpt >= checkpoint_every:
            _save_scores(scores, scores_output)
            scored_since_last_ckpt = 0
            done = len(scores)
            elapsed = time.time() - t_start
            rate = elapsed / max(done - already_done, 1)
            n_to_score = len(score_indices)
            eta_h = (n_to_score - done) * rate / 3600
            logger.info(
                f"Checkpoint | {done:,}/{n_to_score:,} scored | "
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
        description="Pre-compute Qwen3-VL-2B scores on FashionIQ + CIRR training splits."
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/distillation/precompute_benchmarks.yaml",
        help="Path to precompute config YAML.",
    )
    parser.add_argument(
        "--num_negatives", type=int, default=None,
        help="Override K negatives per triplet.",
    )
    main(parser.parse_args())
