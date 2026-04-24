"""
Entry point for retriever training (Setting 1: contrastive, Setting 2: distillation).

Usage:
    python scripts/train_retriever.py --config configs/training/lasco_finetune.yaml
"""

import argparse
import json
import logging
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from src.training.trainer import Trainer
from src.utils.io import save_to_json


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> None:
    log_path = output_dir / "train.log"
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    # suppress overly verbose third-party loggers
    for noisy in ("transformers", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Config + runtime snapshot
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_run_config(cfg: dict, args: argparse.Namespace) -> dict:
    cuda_available = torch.cuda.is_available()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": args.config,
        "experiment_name": cfg["experiment_name"],
        "training_mode": cfg["training_mode"],
        "seed": cfg["seed"],
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
            "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        },
        "model": cfg["model"],
        "training": cfg["training"],
        "data": cfg["data"],
        "distillation": cfg.get("distillation", {}),
        "evaluation": cfg["evaluation"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    # CLI overrides
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.run_name is not None:
        cfg["outputs"]["run_name"] = args.run_name

    # Build output directory
    run_name = cfg["outputs"]["run_name"]
    output_dir = Path(cfg["outputs"]["root_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)
    logger = logging.getLogger(__name__)

    set_seed(cfg["seed"])

    run_config = build_run_config(cfg, args)
    save_to_json(run_config, output_dir / "run_config.json")

    logger.info("=" * 60)
    logger.info(f"Experiment : {cfg['experiment_name']}")
    logger.info(f"Mode       : {cfg['training_mode']}")
    logger.info(f"Output dir : {output_dir}")
    logger.info(f"GPU        : {run_config['runtime']['gpu_name']}")
    logger.info("=" * 60)

    trainer = Trainer(cfg=cfg, output_dir=str(output_dir))
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VISTA retriever on LaSCo.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--run_name", type=str, default=None, help="Override output run name.")

    main(parser.parse_args())
