"""
Pre-compute Qwen3-VL-2B teacher scores as GROUP MATRICES for relation-based
(structural) knowledge distillation.

Difference from precompute_teacher_scores.py
--------------------------------------------
The original script scores each query against its own private set of K random
negatives (per-query pool → row vector). This script instead partitions the
subset into fixed groups of `group_size` queries that SHARE a candidate pool
(the group's own target images), and scores the full G x G teacher matrix:

    matrix[i][j] = Qwen P(yes) for (query_i, candidate_j)   in group g

The diagonal matrix[i][i] is the true positive. Because the pool is shared
across the G queries, the matrix has a genuine relational structure that can be
distilled both row-wise (query → candidates) and column-wise (candidate →
queries) — which a per-query pool cannot express.

Outputs
-------
  data/lasco/relation_subset.json      — shuffled+truncated subset; group g is
                                          the contiguous slice [g*G : (g+1)*G]
  data/lasco/teacher_matrix_qwen.json  — {"<group_id>": {"qids": [G],
                                          "matrix": [G][G]}, ..., "_meta": {...}}

Usage
-----
  python scripts/precompute_teacher_scores_relation.py \\
      --config configs/distillation/precompute_qwen_relation.yaml

  # Override subset size or group size (must restart subset if group_size changes):
  python scripts/precompute_teacher_scores_relation.py \\
      --config configs/distillation/precompute_qwen_relation.yaml \\
      --subset_size 40000 --group_size 32
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
# Grouping
# ---------------------------------------------------------------------------

def _build_relation_subset(
    all_triplets: list[dict], size: int, group_size: int, seed: int
) -> list[dict]:
    """Sample `size` triplets, then truncate to a whole multiple of group_size.

    Group g is the contiguous slice subset[g*G : (g+1)*G]. The shuffle is
    deterministic in `seed`, so groups are reproducible and resume-safe.
    """
    rng = np.random.default_rng(seed)
    size = min(size, len(all_triplets))
    indices = rng.choice(len(all_triplets), size=size, replace=False)
    subset = [all_triplets[i] for i in indices]

    n_groups = len(subset) // group_size
    truncated = subset[: n_groups * group_size]
    if len(truncated) != len(subset):
        logger.info(
            f"Truncated subset {len(subset):,} → {len(truncated):,} "
            f"({n_groups} full groups of {group_size})"
        )
    return truncated


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_group(
    reranker: Qwen3VLRanker,
    group: list[dict],
    images_dir: Path,
) -> dict:
    """Score the full G x G teacher matrix for one group.

    matrix[i][j] = score(query_i, candidate_j) where candidate_j is the target
    image of group member j (shared pool). Diagonal is the true positive.
    """
    candidates = [
        RerankCandidate(image=str(images_dir / t["target-image"][1]))
        for t in group
    ]

    matrix: list[list[float]] = []
    for t in group:
        query = RerankQuery(
            reference_image=str(images_dir / t["query-image"][1]),
            text_edit=t["query-text"],
        )
        row = reranker.score_batch(query, candidates)
        matrix.append([float(s) for s in row])

    return {
        "qids": [int(t["qid"]) for t in group],
        "matrix": matrix,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_existing(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save(data: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    _setup_logging()
    cfg = _load_config(args.config)
    dc = cfg["data"]

    subset_size = args.subset_size or int(dc["subset_size"])
    group_size = args.group_size or int(dc["group_size"])
    seed = int(dc["seed"])
    checkpoint_every = int(dc["checkpoint_every"])

    lasco_path = PROJECT_ROOT / dc["lasco_annotations"]
    images_dir = PROJECT_ROOT / dc["images_dir"]
    subset_output = PROJECT_ROOT / dc["subset_output"]
    matrix_output = PROJECT_ROOT / dc["matrix_output"]

    logger.info("=" * 60)
    logger.info(f"Experiment : {cfg['experiment_name']}")
    logger.info(f"Subset size: {subset_size:,}  |  Group size: {group_size}")
    logger.info(f"Seed       : {seed}")
    logger.info(f"Matrix out : {matrix_output}")
    logger.info("=" * 60)

    # ── Load LaSCo ──────────────────────────────────────────────────────
    logger.info(f"Loading LaSCo annotations from {lasco_path} …")
    with open(lasco_path, "r") as f:
        all_triplets = json.load(f)
    logger.info(f"Total triplets: {len(all_triplets):,}")

    # ── Build or load the relation subset ───────────────────────────────
    if subset_output.exists():
        logger.info(f"Loading existing relation subset from {subset_output}")
        with open(subset_output, "r") as f:
            subset = json.load(f)
        if len(subset) % group_size != 0:
            raise ValueError(
                f"Existing subset size {len(subset):,} is not a multiple of "
                f"group_size={group_size}. Delete {subset_output} to rebuild, "
                "or pass the original --group_size."
            )
    elif bool(dc.get("groups_prebuilt", False)):
        raise FileNotFoundError(
            f"groups_prebuilt=true but {subset_output} is missing. Build HARD groups first:\n"
            f"  python scripts/build_relation_groups.py --config {args.config}\n"
            "(or set data.groups_prebuilt=false to fall back to random grouping)."
        )
    else:
        logger.info(f"Building RANDOM relation subset (size={subset_size:,}, G={group_size}, seed={seed}) …")
        subset = _build_relation_subset(all_triplets, subset_size, group_size, seed)
        subset_output.parent.mkdir(parents=True, exist_ok=True)
        with open(subset_output, "w") as f:
            json.dump(subset, f)
        logger.info(f"Relation subset saved → {subset_output} ({len(subset):,} triplets)")

    n_groups = len(subset) // group_size
    total_scores = n_groups * group_size * group_size
    logger.info(
        f"Groups: {n_groups:,}  |  Teacher forwards: {total_scores:,} "
        f"({n_groups} x {group_size}^2)"
    )

    # ── Resume ───────────────────────────────────────────────────────────
    data: dict = _load_existing(matrix_output)
    data.setdefault("_meta", {
        "group_size": group_size,
        "subset_size": len(subset),
        "n_groups": n_groups,
        "seed": seed,
        "score_kind": "qwen3vl_p_yes",
    })
    already_done = len([k for k in data if k != "_meta"])
    if already_done:
        logger.info(f"Resuming: {already_done:,} / {n_groups:,} groups already scored.")

    # ── Load reranker ────────────────────────────────────────────────────
    logger.info("Loading Qwen3-VL-2B reranker …")
    reranker = _build_reranker(cfg)
    logger.info("Reranker ready.")

    # ── Score group by group ─────────────────────────────────────────────
    scored_since_ckpt = 0
    t_start = time.time()

    for g in range(n_groups):
        gid = str(g)
        if gid in data:
            continue  # already scored

        group = subset[g * group_size : (g + 1) * group_size]

        try:
            data[gid] = _score_group(reranker, group, images_dir)
        except Exception as exc:  # noqa: BLE001 — keep going, log and skip
            logger.warning(f"[group {g+1}/{n_groups}] failed: {exc}")
            continue

        scored_since_ckpt += 1
        if scored_since_ckpt >= checkpoint_every:
            _save(data, matrix_output)
            scored_since_ckpt = 0
            done = len([k for k in data if k != "_meta"])
            elapsed = time.time() - t_start
            rate = elapsed / max(done - already_done, 1)   # sec per group
            eta_h = (n_groups - done) * rate / 3600
            logger.info(
                f"Checkpoint | {done:,}/{n_groups:,} groups | "
                f"{rate:.1f}s/group | ETA={eta_h:.1f}h"
            )

    _save(data, matrix_output)
    elapsed_total = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"Done. {len([k for k in data if k != '_meta']):,} groups scored "
                f"in {elapsed_total/3600:.2f}h")
    logger.info(f"Matrix saved → {matrix_output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute Qwen3-VL-2B group-matrix teacher scores (relation-based KD)."
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/distillation/precompute_qwen_relation.yaml",
        help="Path to precompute config YAML.",
    )
    parser.add_argument("--subset_size", type=int, default=None, help="Override subset size.")
    parser.add_argument("--group_size", type=int, default=None, help="Override group size G.")

    main(parser.parse_args())
