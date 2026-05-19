"""
Fine-tune Qwen3-VL-2B re-ranker on LaSCo with LoRA (last layer only).

LoRA is applied exclusively to the last transformer layer (index 27/28):
  target modules : q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  rank           : r=16  (622K trainable params, 0.03% of 2.1B)

All other weights are frozen. After training, LoRA weights are merged back
into the base model and saved as a full HuggingFace checkpoint — fully
compatible with the existing Qwen3VLRanker inference class.

Training objective: binary cross-entropy on P(yes) next-token logit.
  Positive pair (query → ground-truth target) → label 1
  Negative pair (query → hard-negative image)  → label 0

Compare with: scripts/train_reranker.py (partial-freeze baseline, top-8 layers).

Usage
-----
python scripts/train_reranker_lora.py --config configs/reranker/finetune_lasco_lora.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
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
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    processor = Qwen3VLProcessor.from_pretrained(
        checkpoint_path,
        use_fast=False,
        local_files_only=True,
    )
    return model, processor


def apply_lora(
    model: Qwen3VLForConditionalGeneration,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    last_n_layers: int = 1,
) -> object:
    """Wrap model with LoRA adapters on the last N transformer layers only."""
    total_layers = len(model.model.language_model.layers)
    layers_to_transform = list(range(total_layers - last_n_layers, total_layers))

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=layers_to_transform,
        lora_dropout=lora_dropout,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_cfg)
    # Frozen embed_tokens breaks the gradient graph when PEFT is used.
    # This hook ensures embedding outputs retain requires_grad=True.
    peft_model.enable_input_require_grads()

    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    logger.info(
        f"LoRA applied to layers {layers_to_transform}  "
        f"(r={r}, alpha={lora_alpha}, dropout={lora_dropout})"
    )
    logger.info(
        f"Trainable: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.4f}%)"
    )
    return peft_model


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_p_yes(
    model,
    processor: Qwen3VLProcessor,
    ref_path: str,
    text: str,
    cand_path: str,
    device: torch.device,
    yes_id: int,
    no_id: int,
) -> torch.Tensor:
    ref_img = load_image_item(ref_path)
    cand_img = load_image_item(cand_path)
    message = build_pointwise_relevance_message(
        reference_image=ref_img,
        text_edit=text,
        candidate_image=cand_img,
    )
    inputs = process_messages_to_inputs(processor, [message], device=device)
    logits = model(**inputs, return_dict=True).logits
    last_logits = logits[:, -1, :]
    binary_logits = torch.stack([last_logits[:, no_id], last_logits[:, yes_id]], dim=1)
    return torch.softmax(binary_logits, dim=1)[:, 1]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model,
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
    total_loss = 0.0
    n_pos = n_neg = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"[Epoch {epoch}] LoRA training")
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
            pbar.set_postfix(loss=f"{total_loss/step:.4f}", pos=n_pos, neg=n_neg)

    if len(dataloader) % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad()

    return {"avg_loss": total_loss / len(dataloader), "n_pos": n_pos, "n_neg": n_neg}


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_adapter_checkpoint(model, processor: Qwen3VLProcessor, output_dir: Path, epoch: int) -> Path:
    """Save only the LoRA adapter weights — does NOT modify the training model.

    Calling merge_and_unload() mid-training destroys the PEFT model in-place,
    stripping LoRA adapters so subsequent epochs train nothing. Saving adapters
    only avoids this and is also much cheaper (a few MB vs several GB).

    Full merged HF checkpoints are produced by _merge_all_adapters() after
    the training loop finishes.
    """
    adapter_dir = output_dir / "adapters" / f"epoch_{epoch:02d}"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))   # saves adapter_model.safetensors + adapter_config.json
    processor.save_pretrained(str(adapter_dir))
    logger.info(f"Adapter checkpoint saved → {adapter_dir}")
    return adapter_dir


def _merge_all_adapters(
    base_ckpt: str,
    dtype: str,
    output_dir: Path,
    epochs: int,
    lora_cfg: dict,
) -> None:
    """Load base model + each epoch's adapter and save merged HF checkpoints.

    Called once after the training loop — at this point the GPU is free so
    we can load the base model fresh and merge without touching the training model.
    """
    from peft import PeftModel

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]

    for epoch in range(1, epochs + 1):
        adapter_dir = output_dir / "adapters" / f"epoch_{epoch:02d}"
        ckpt_dir = output_dir / "checkpoints" / f"epoch_{epoch:02d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if not adapter_dir.exists():
            logger.warning(f"Adapter dir missing, skipping epoch {epoch}: {adapter_dir}")
            continue

        logger.info(f"Merging epoch {epoch} adapters into full HF checkpoint …")
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_ckpt, dtype=torch_dtype, low_cpu_mem_usage=True, local_files_only=True,
        )
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(str(ckpt_dir))

        processor = Qwen3VLProcessor.from_pretrained(
            str(adapter_dir), use_fast=False, local_files_only=True,
        )
        processor.save_pretrained(str(ckpt_dir))

        del base_model, peft_model, merged
        torch.cuda.empty_cache()
        logger.info(f"Checkpoint saved → {ckpt_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _get_token_id(processor: Qwen3VLProcessor, text: str) -> int:
    ids = processor.tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise RuntimeError(f"Token '{text}' not found in tokenizer vocabulary.")
    return ids[0]


def main(cfg: dict) -> None:
    run_name: str = cfg["output"]["run_name"]
    output_dir = PROJECT_ROOT / cfg["output"].get("output_dir", "results/training") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    logger.info("=" * 60)
    logger.info(f" Re-ranker LoRA fine-tuning — {run_name}")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Model ---
    ckpt_path = str(PROJECT_ROOT / cfg["model"]["checkpoint_path"])
    logger.info(f"Loading base model from {ckpt_path}")
    model, processor = load_model_and_processor(ckpt_path, dtype=cfg["model"].get("dtype", "bfloat16"))
    model = model.to(device)

    lora_cfg = cfg.get("lora", {})
    model = apply_lora(
        model,
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        last_n_layers=int(lora_cfg.get("last_n_layers", 1)),
    )

    yes_id = _get_token_id(processor, " yes")
    no_id = _get_token_id(processor, " no")
    logger.info(f"yes_id={yes_id}, no_id={no_id}")

    # --- Dataset helpers ---
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    use_hard_neg = "hard_neg_path" in data_cfg
    base_seed = train_cfg.get("seed", 42)
    num_workers = train_cfg.get("num_workers", 2)

    def _build_dataset(neg_seed: int):
        if use_hard_neg:
            from src.datasets.lasco_reranker_hard_neg import LaSCoRerankerHardNeg
            return LaSCoRerankerHardNeg(
                subset_path=str(PROJECT_ROOT / data_cfg["subset_path"]),
                hard_neg_path=str(PROJECT_ROOT / data_cfg["hard_neg_path"]),
                images_dir=str(PROJECT_ROOT / data_cfg["images_dir"]),
                neg_per_query=train_cfg.get("neg_per_query", 1),
                max_queries=train_cfg.get("max_queries", None),
                seed=base_seed,
                neg_seed=neg_seed,
            )
        return LaSCoReranker(
            subset_path=str(PROJECT_ROOT / data_cfg["subset_path"]),
            scores_path=str(PROJECT_ROOT / data_cfg["scores_path"]),
            images_dir=str(PROJECT_ROOT / data_cfg["images_dir"]),
            neg_per_query=train_cfg.get("neg_per_query", 1),
            max_queries=train_cfg.get("max_queries", None),
            seed=neg_seed,
        )

    if use_hard_neg:
        logger.info("Using retriever-mined hard negatives — negatives rotated each epoch.")
    # Build once to get pair count for optimizer/scheduler setup
    n_pairs = len(_build_dataset(neg_seed=base_seed))
    logger.info(f"Pairs/epoch: {n_pairs:,}")

    # --- Optimizer / scheduler ---
    lr = float(train_cfg.get("learning_rate", 2e-4))
    epochs = int(train_cfg.get("epochs", 3))
    grad_accum = int(train_cfg.get("grad_accum", 8))
    warmup_steps = int(train_cfg.get("warmup_steps", 100))

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    steps_per_epoch = math.ceil(n_pairs / grad_accum)
    total_steps = steps_per_epoch * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=lr * 0.1)

    logger.info(f"Epochs: {epochs}  |  Steps/epoch: {steps_per_epoch}  "
                f"|  LR: {lr}  |  grad_accum: {grad_accum}  |  warmup: {warmup_steps}")

    # --- Training loop ---
    train_log = []
    global_step = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # Rebuild dataset each epoch: same query subset (base_seed), different neg (neg_seed=epoch)
        dataset = _build_dataset(neg_seed=epoch)
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=True,
            num_workers=num_workers, pin_memory=False,
        )

        t_epoch = time.time()
        results = train_epoch(
            model, processor, dataloader, optimizer, device,
            yes_id, no_id, grad_accum, epoch,
        )
        elapsed = time.time() - t_epoch

        for _ in range(steps_per_epoch):
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
            save_adapter_checkpoint(model, processor, output_dir, epoch)

    total_elapsed = time.time() - t_start
    logger.info(f"Training done in {total_elapsed/3600:.2f}h  ({total_elapsed:.0f}s)")

    # Merge adapters into full HF checkpoints (after training loop — GPU is free)
    logger.info("Merging LoRA adapters into full HF checkpoints …")
    _merge_all_adapters(
        base_ckpt=ckpt_path,
        dtype=cfg["model"].get("dtype", "bfloat16"),
        output_dir=output_dir,
        epochs=epochs,
        lora_cfg=lora_cfg,
    )

    info = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "method": "lora",
        "lora": lora_cfg,
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "total_elapsed_seconds": round(total_elapsed, 1),
            "total_elapsed_hours": round(total_elapsed / 3600, 3),
        },
        "train_log": train_log,
    }
    with open(output_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL-2B re-ranker with LoRA on LaSCo.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config_path = PROJECT_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
