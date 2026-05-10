"""
Fine-tune Qwen3-VL-2B re-ranker on LaSCo training data.

Objective : binary cross-entropy on P(yes) next-token logit.
            Positive pair (query → ground-truth target) → label 1
            Negative pair (query → hard-negative image)  → label 0

Freeze strategy (no PEFT required):
  - Vision encoder fully frozen (ViT parameters)
  - Bottom `freeze_bottom_lm_layers` of 28 LM layers frozen
  - Top LM layers + lm_head remain trainable

Saved checkpoints are self-contained HuggingFace directories and can be
loaded directly by the existing Qwen3VLRanker inference class by updating
the model.checkpoint_path in the reranker config.

Usage
-----
python scripts/train_reranker.py --config configs/reranker/finetune_lasco.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.lasco_reranker import LaSCoReranker
from src.rerankers.backbones.lamra.preprocess import load_image_item, process_messages_to_inputs
from src.rerankers.backbones.lamra.prompts import build_pointwise_relevance_message


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(output_dir: Path) -> None:
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "train.log", encoding="utf-8"),
        ],
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_model_and_processor(
    checkpoint_path: str,
    dtype: str = "bfloat16",
) -> tuple[Qwen3VLForConditionalGeneration, Qwen3VLProcessor]:
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    processor = Qwen3VLProcessor.from_pretrained(
        checkpoint_path,
        use_fast=False,
        local_files_only=True,
    )
    return model, processor


def freeze_model(
    model: Qwen3VLForConditionalGeneration,
    freeze_vision: bool = True,
    freeze_bottom_lm_layers: int = 20,
) -> None:
    """Freeze vision encoder and bottom N LM layers in place."""
    if freeze_vision:
        for p in model.visual.parameters():
            p.requires_grad = False

    for p in model.model.embed_tokens.parameters():
        p.requires_grad = False

    total_layers = len(model.model.layers)
    n_freeze = min(freeze_bottom_lm_layers, total_layers)
    for i in range(n_freeze):
        for p in model.model.layers[i].parameters():
            p.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Freeze: vision={'yes' if freeze_vision else 'no'}, "
        f"bottom {n_freeze}/{total_layers} LM layers frozen. "
        f"Trainable: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.1f}%)"
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_p_yes(
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    ref_path: str,
    text: str,
    cand_path: str,
    device: torch.device,
    yes_id: int,
    no_id: int,
) -> torch.Tensor:
    """Return scalar P(yes) tensor for one (ref, text, cand) triple."""
    ref_img = load_image_item(ref_path)
    cand_img = load_image_item(cand_path)

    message = build_pointwise_relevance_message(
        reference_image=ref_img,
        text_edit=text,
        candidate_image=cand_img,
    )
    inputs = process_messages_to_inputs(processor, [message], device=device)

    logits = model(**inputs, return_dict=True).logits  # [1, seq_len, vocab]
    last_logits = logits[:, -1, :]                     # [1, vocab]
    binary_logits = torch.stack([last_logits[:, no_id], last_logits[:, yes_id]], dim=1)
    return torch.softmax(binary_logits, dim=1)[:, 1]   # [1]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    yes_id: int,
    no_id: int,
    grad_accum: int,
    epoch: int,
) -> dict[str, float]:
    model.train()
    # Keep frozen layers in eval mode to disable any internal dropout
    for p in model.parameters():
        if not p.requires_grad:
            pass  # grad is off, eval/train mode doesn't affect frozen weights in Qwen3-VL

    total_loss = 0.0
    n_pos = n_neg = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"[Epoch {epoch}] Training")
    for step, batch in enumerate(pbar, start=1):
        ref_path: str = batch["ref_path"][0]
        text: str = batch["text"][0]
        cand_path: str = batch["cand_path"][0]
        label: float = float(batch["label"].item())

        with torch.autocast("cuda", dtype=torch.bfloat16):
            p_yes = compute_p_yes(model, processor, ref_path, text, cand_path, device, yes_id, no_id)

        target = torch.tensor([[label]], dtype=torch.float32, device=device)
        loss = F.binary_cross_entropy(p_yes.unsqueeze(1), target) / grad_accum
        loss.backward()

        total_loss += loss.item() * grad_accum
        if label == 1.0:
            n_pos += 1
        else:
            n_neg += 1

        if step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()

        if step % 100 == 0:
            avg = total_loss / step
            pbar.set_postfix(loss=f"{avg:.4f}", pos=n_pos, neg=n_neg)

    # Final optimizer step if steps not divisible by grad_accum
    if len(dataloader) % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    return {"avg_loss": avg_loss, "n_pos": n_pos, "n_neg": n_neg}


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    output_dir: Path,
    epoch: int,
) -> Path:
    """Save full HuggingFace checkpoint usable by Qwen3VLRanker."""
    ckpt_dir = output_dir / "checkpoints" / f"epoch_{epoch:02d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    processor.save_pretrained(str(ckpt_dir))
    logger.info(f"Checkpoint saved → {ckpt_dir}")
    return ckpt_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _get_token_id(processor: Qwen3VLProcessor, text: str) -> int:
    ids = processor.tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise RuntimeError(f"Token '{text}' not found in tokenizer vocabulary.")
    return ids[0]


def main(cfg: dict) -> None:
    # Paths
    run_name: str = cfg["output"]["run_name"]
    output_dir = PROJECT_ROOT / cfg["output"].get("output_dir", "results/training") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir)

    # Dump config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    logger.info("=" * 60)
    logger.info(f" Re-ranker fine-tuning — {run_name}")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Model ---
    ckpt_path = str(PROJECT_ROOT / cfg["model"]["checkpoint_path"])
    logger.info(f"Loading model from {ckpt_path}")
    model, processor = load_model_and_processor(
        ckpt_path,
        dtype=cfg["model"].get("dtype", "bfloat16"),
    )
    model = model.to(device)

    train_cfg = cfg["training"]
    freeze_model(
        model,
        freeze_vision=train_cfg.get("freeze_vision", True),
        freeze_bottom_lm_layers=train_cfg.get("freeze_bottom_lm_layers", 20),
    )

    yes_id = _get_token_id(processor, " yes")
    no_id = _get_token_id(processor, " no")
    logger.info(f"yes_id={yes_id}, no_id={no_id}")

    # --- Dataset ---
    data_cfg = cfg["data"]
    dataset = LaSCoReranker(
        subset_path=str(PROJECT_ROOT / data_cfg["subset_path"]),
        scores_path=str(PROJECT_ROOT / data_cfg["scores_path"]),
        images_dir=str(PROJECT_ROOT / data_cfg["images_dir"]),
        neg_per_query=train_cfg.get("neg_per_query", 1),
        max_queries=train_cfg.get("max_queries", None),
        seed=train_cfg.get("seed", 42),
    )
    logger.info(f"Dataset: {len(dataset):,} pairs")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 2),
        pin_memory=False,
    )

    # --- Optimizer / scheduler ---
    lr = float(train_cfg.get("learning_rate", 2e-5))
    epochs = int(train_cfg.get("epochs", 3))
    grad_accum = int(train_cfg.get("grad_accum", 8))
    warmup_steps = int(train_cfg.get("warmup_steps", 100))

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    total_steps = math.ceil(len(dataloader) / grad_accum) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=lr * 0.1)

    logger.info(f"Epochs: {epochs}  |  Steps/epoch: {math.ceil(len(dataloader)/grad_accum)}  "
                f"|  LR: {lr}  |  grad_accum: {grad_accum}  |  warmup: {warmup_steps}")

    # --- Training loop ---
    train_log = []
    global_step = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        results = train_epoch(
            model, processor, dataloader, optimizer, device,
            yes_id, no_id, grad_accum, epoch,
        )
        elapsed = time.time() - t_epoch

        # Step scheduler after each effective optimizer step
        for _ in range(math.ceil(len(dataloader) / grad_accum)):
            global_step += 1
            if global_step > warmup_steps:
                scheduler.step()

        log_entry = {
            "epoch": epoch,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "train": results,
        }
        train_log.append(log_entry)

        logger.info(
            f"[Epoch {epoch}] loss={results['avg_loss']:.4f}  "
            f"pos={results['n_pos']}  neg={results['n_neg']}  "
            f"elapsed={elapsed/60:.1f}min"
        )

        with open(output_dir / "train_log.json", "w") as f:
            json.dump(train_log, f, indent=2)

        if train_cfg.get("save_every_epoch", True):
            save_checkpoint(model, processor, output_dir, epoch)

    total_elapsed = time.time() - t_start
    logger.info(f"Training done in {total_elapsed/3600:.1f}h")
    logger.info(f"Checkpoints: {output_dir / 'checkpoints'}")

    # --- Runtime info ---
    info = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "total_elapsed_seconds": round(total_elapsed, 1),
        },
        "train_log": train_log,
    }
    with open(output_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL-2B re-ranker on LaSCo.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
