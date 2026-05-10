"""
Evaluate a VISTA checkpoint on the LaSCo validation split.

Supports zero-shot (base model) and any fine-tuned checkpoint.
Results are saved to results/lasco_val/<run_name>/metrics.json.

Usage
-----
# Zero-shot (base VISTA, no fine-tuning):
python scripts/eval_lasco_val.py --zero_shot --run_name zero_shot

# Fine-tuned checkpoint:
python scripts/eval_lasco_val.py \\
    --checkpoint results/training/vista_lasco_finetune_contrastive_v2/checkpoints/vista_epoch01.pth \\
    --run_name contrastive_v2_epoch01

# Override batch size or K values:
python scripts/eval_lasco_val.py --zero_shot --run_name zero_shot --batch_size 32 --k 1 5 10 50 100
"""

import argparse
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.lasco_eval import evaluate_lasco_val
from src.retrievers.backbones.vista.modeling import Visualized_BGE


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(output_dir: Path) -> None:
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "eval.log", encoding="utf-8"),
        ],
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_backbone(checkpoint: str | None) -> Visualized_BGE:
    backbone = Visualized_BGE(
        model_name_bge="BAAI/bge-base-en-v1.5",
        model_weight=str(PROJECT_ROOT / "models/Visualized_BGE/Visualized_base_en_v1.5.pth"),
        negatives_cross_device=False,
        from_pretrained=None,
    )
    if checkpoint is not None:
        ckpt_path = PROJECT_ROOT / checkpoint if not Path(checkpoint).is_absolute() else Path(checkpoint)
        state = torch.load(str(ckpt_path), map_location="cpu")
        backbone.load_state_dict(state)
        logger.info(f"Loaded checkpoint: {ckpt_path}")
    else:
        logger.info("Zero-shot mode: base VISTA weights, no fine-tuned checkpoint.")

    if torch.cuda.is_available():
        backbone = backbone.cuda()
    backbone.eval()
    return backbone


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def _save_result(
    output_dir: Path,
    run_name: str,
    checkpoint: str | None,
    metrics: dict,
    elapsed: float,
    k_values: list[int],
    args: argparse.Namespace,
) -> None:
    cuda_available = torch.cuda.is_available()
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "dataset": "lasco_val",
        "model": {
            "backbone": "VISTA — Visualized BGE-base-en-v1.5 + EVA02-CLIP-B-16",
            "checkpoint": checkpoint if checkpoint is not None else "zero_shot (base weights)",
        },
        "eval_settings": {
            "query_embedding_mode": "vista_mm",
            "gallery_size": 39826,
            "num_queries": 30037,
            "k_values": k_values,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "elapsed_seconds": round(elapsed, 1),
        },
        "metrics": metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metrics.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    run_name = args.run_name
    output_dir = PROJECT_ROOT / "results" / "lasco_val" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    _setup_logging(output_dir)

    checkpoint = None if args.zero_shot else args.checkpoint
    if not args.zero_shot and args.checkpoint is None:
        logger.error("Provide --checkpoint <path> or --zero_shot.")
        sys.exit(1)

    k_values = sorted(args.k)

    logger.info("=" * 60)
    logger.info(f"Run name   : {run_name}")
    logger.info(f"Checkpoint : {checkpoint or 'zero_shot'}")
    logger.info(f"K values   : {k_values}")
    logger.info(f"Output dir : {output_dir}")
    logger.info("=" * 60)

    backbone = _load_backbone(checkpoint)

    t0 = time.time()
    metrics = evaluate_lasco_val(
        backbone=backbone,
        images_dir=str(PROJECT_ROOT / "data/lasco/images"),
        val_path=str(PROJECT_ROOT / "data/lasco/lasco_val.json"),
        corpus_path=str(PROJECT_ROOT / "data/lasco/lasco_val_corpus.json"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        k_values=k_values,
    )
    elapsed = time.time() - t0

    # Print table
    logger.info("-" * 40)
    logger.info(f"LaSCo val results — {run_name}")
    for k in k_values:
        logger.info(f"  Recall@{k:<4d}: {metrics[f'recall_at{k}']:.2f}%")
    logger.info(f"  Elapsed: {elapsed/60:.1f} min")
    logger.info("-" * 40)

    _save_result(output_dir, run_name, checkpoint, metrics, elapsed, k_values, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VISTA on LaSCo val split.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zero_shot", action="store_true", help="Evaluate base model without fine-tuning.")
    group.add_argument("--checkpoint", type=str, help="Path to fine-tuned .pth checkpoint.")
    parser.add_argument("--run_name", type=str, required=True, help="Name for the output directory.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 50, 100],
                        help="K values for Recall@K.")
    main(parser.parse_args())
