"""
Retriever trainer — supports two training modes:
  - contrastive   : InfoNCE on LaSCo triplets (Setting 1)
  - distillation  : KD from Qwen3-VL-2B soft labels (Setting 2, placeholder)
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.datasets.lasco import build_lasco_dataset
from src.retrievers.backbones.vista.modeling import Visualized_BGE
from src.retrievers.vista_retriever import VistaImageProcessor
from src.training.collator import LaSCoCollator
from src.utils.io import save_to_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_backbone(cfg: dict) -> Visualized_BGE:
    m = cfg["model"]
    return Visualized_BGE(
        model_name_bge=m["model_name_or_path"],
        model_weight=m["checkpoint_path"],
        temperature=cfg["training"].get("temperature", 0.02),
        negatives_cross_device=False,
        from_pretrained=m.get("from_pretrained"),
    )


def _build_dataloader(backbone: Visualized_BGE, cfg: dict) -> DataLoader:
    tc = cfg["training"]
    dc = cfg["data"]["lasco"]

    image_transform = VistaImageProcessor(backbone.preprocess_train)

    def tokenize(text, **kwargs):
        return backbone.tokenizer(text, **kwargs)

    dataset = build_lasco_dataset(
        annotations_path=dc["annotations"],
        images_dir=dc["images_dir"],
        image_transform=image_transform,
        caption_transform=tokenize,
        max_length_tokenizer=77,
    )

    logger.info(f"LaSCo dataset loaded: {len(dataset):,} triplets")

    return DataLoader(
        dataset,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=tc["num_workers"],
        collate_fn=LaSCoCollator(),
        pin_memory=True,
        drop_last=True,
        persistent_workers=tc["num_workers"] > 0,
    )


def _build_optimizer_and_scheduler(
    backbone: Visualized_BGE,
    cfg: dict,
    total_steps: int,
) -> tuple[AdamW, Any]:
    tc = cfg["training"]
    warmup_steps = int(total_steps * tc["warmup_ratio"])

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, backbone.parameters()),
        lr=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(
        f"Optimizer: AdamW  lr={tc['learning_rate']}  wd={tc['weight_decay']}"
    )
    logger.info(
        f"Scheduler: cosine  total_steps={total_steps}  warmup_steps={warmup_steps}"
    )
    return optimizer, scheduler


def _run_eval(backbone: Visualized_BGE, cfg: dict) -> dict[str, float]:
    """Evaluate on FashionIQ val and CIRR val using the existing eval pipeline."""
    from src.evaluation.fashioniq_eval import evaluate_fashioniq
    from src.evaluation.cirr_eval import evaluate_cirr
    from src.retrievers.vista_retriever import VistaBGERetriever
    from src.utils.io import prepend_key_to_dict

    ec = cfg["evaluation"]
    backbone.eval()
    retriever = VistaBGERetriever(backbone)
    metrics: dict[str, float] = {}

    if "fashioniq" in ec["datasets"]:
        fiq = evaluate_fashioniq(
            model=retriever,
            query_embedding_mode=ec["query_embedding_mode"],
            eval_protocol=ec["fashioniq_eval_protocol"],
            fusion_type=ec["fusion_type"],
            batch_size=ec["batch_size"],
            num_workers=ec["num_workers"],
            tqdm=False,
            accelerator=None,
        )
        metrics.update(prepend_key_to_dict("fashioniq_", fiq))

    if "cirr" in ec["datasets"]:
        cirr = evaluate_cirr(
            model=retriever,
            query_embedding_mode=ec["query_embedding_mode"],
            fusion_type=ec["fusion_type"],
            batch_size=ec["batch_size"],
            num_workers=ec["num_workers"],
            tqdm=False,
            accelerator=None,
        )
        metrics.update(prepend_key_to_dict("cirr_", cirr))

    backbone.train()
    return metrics


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: dict, output_dir: str):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.training_mode: str = cfg["training_mode"]
        self.precision: str = cfg["training"].get("precision", "bf16")
        self.use_amp: bool = self.precision == "bf16"
        self.amp_dtype = torch.bfloat16 if self.use_amp else torch.float32

        self.grad_accum: int = cfg["training"]["gradient_accumulation_steps"]
        self.max_grad_norm: float = cfg["training"]["max_grad_norm"]
        self.log_every: int = cfg["training"]["log_every_n_steps"]
        self.eval_every: int = cfg["training"]["eval_every_n_epochs"]
        self.save_every: int = cfg["training"]["save_every_n_epochs"]

        self._epoch_log: list[dict] = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def train(self) -> None:
        logger.info(f"Training mode : {self.training_mode}")
        logger.info(f"Precision     : {self.precision}")
        logger.info(f"Output dir    : {self.output_dir}")

        if self.training_mode == "distillation":
            raise NotImplementedError(
                "Distillation training (Setting 2) is not yet implemented. "
                "Set training_mode: contrastive or implement _distillation_step()."
            )

        backbone = _build_backbone(self.cfg)
        freeze_info = self._freeze_layers(backbone)
        save_to_json(freeze_info, self.output_dir / "freeze_info.json")
        backbone.train()

        loader = _build_dataloader(backbone, self.cfg)
        steps_per_epoch = len(loader)
        total_steps = steps_per_epoch * self.cfg["training"]["epochs"] // self.grad_accum

        optimizer, scheduler = _build_optimizer_and_scheduler(
            backbone, self.cfg, total_steps
        )

        logger.info(
            f"Steps/epoch={steps_per_epoch}  "
            f"effective_batch={self.cfg['training']['batch_size'] * self.grad_accum}  "
            f"total_opt_steps={total_steps}"
        )

        best_metric: float = 0.0

        for epoch in range(1, self.cfg["training"]["epochs"] + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(
                backbone, loader, optimizer, scheduler, epoch, steps_per_epoch
            )
            elapsed = time.time() - t0

            eval_metrics: dict[str, float] = {}
            if epoch % self.eval_every == 0:
                logger.info(f"[Epoch {epoch}] Running evaluation …")
                eval_metrics = _run_eval(backbone, self.cfg)
                self._log_eval(epoch, eval_metrics)

                summary = eval_metrics.get(
                    "fashioniq_val_avg_recall_at10",
                    eval_metrics.get("cirr_val_summary_average", 0.0),
                )
                if summary > best_metric:
                    best_metric = summary
                    self._save_checkpoint(backbone, epoch, tag="best")
                    logger.info(f"[Epoch {epoch}] New best: {summary:.4f} → checkpoint saved")

            if epoch % self.save_every == 0:
                self._save_checkpoint(backbone, epoch, tag=f"epoch{epoch:02d}")

            self._append_epoch_log(epoch, elapsed, train_metrics, eval_metrics)

        self._save_checkpoint(backbone, self.cfg["training"]["epochs"], tag="final")
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        backbone: Visualized_BGE,
        loader: DataLoader,
        optimizer: AdamW,
        scheduler: Any,
        epoch: int,
        steps_per_epoch: int,
    ) -> dict[str, float]:
        device = backbone.device
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader, start=1):
            ref_images = batch.ref_images.to(device, non_blocking=True)
            target_images = batch.target_images.to(device, non_blocking=True)
            texts = {
                "input_ids": batch.input_ids.to(device, non_blocking=True),
                "attention_mask": batch.attention_mask.to(device, non_blocking=True),
            }

            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                output = backbone(
                    mm_it_query=(ref_images, texts),
                    image_candidate=target_images,
                    task_type="edit_image",
                )
                loss = output.loss / self.grad_accum

            loss.backward()
            total_loss += loss.item() * self.grad_accum

            if step % self.grad_accum == 0 or step == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(
                    backbone.parameters(), self.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % self.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"[Epoch {epoch:02d} | Step {step:05d}/{steps_per_epoch}] "
                    f"loss={loss.item() * self.grad_accum:.4f}  lr={lr:.2e}"
                )

        avg_loss = total_loss / steps_per_epoch
        logger.info(f"[Epoch {epoch:02d}] avg_loss={avg_loss:.4f}")
        return {"avg_loss": avg_loss}

    def _freeze_layers(self, backbone: Visualized_BGE) -> dict:
        """
        Freeze layers according to config freeze.strategy.
          - full            : train all parameters (v1 behaviour)
          - top_layers_only : freeze vision encoder + bottom BGE layers,
                              train top BGE layers + visual_proj only
        """
        freeze_cfg = self.cfg.get("freeze", {"strategy": "full"})
        strategy = freeze_cfg.get("strategy", "full")

        total_params = sum(p.numel() for p in backbone.parameters())

        if strategy == "full":
            trainable_params = total_params
            trainable_modules = ["all"]
            frozen_modules = []
        elif strategy == "top_layers_only":
            # Freeze everything first
            for param in backbone.parameters():
                param.requires_grad = False

            n_total = len(backbone.bge_encoder.layer)
            n_freeze = freeze_cfg.get("bge_freeze_bottom_n_layers", 9)
            n_train = n_total - n_freeze

            # Unfreeze top BGE transformer layers
            for i in range(n_freeze, n_total):
                for param in backbone.bge_encoder.layer[i].parameters():
                    param.requires_grad = True

            # Unfreeze visual_proj (vision-text bridge)
            for param in backbone.visual_proj.parameters():
                param.requires_grad = True

            trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            trainable_modules = [
                f"bge_encoder.layer[{n_freeze}:{n_total}]  ({n_train} layers)",
                "visual_proj",
            ]
            frozen_modules = [
                "bge_embeddings",
                f"bge_encoder.layer[0:{n_freeze}]  ({n_freeze} layers)",
                "model_visual  (EVA02-CLIP-B-16, full vision encoder)",
                "bge_pooler",
            ]
        else:
            raise ValueError(f"Unknown freeze strategy: {strategy!r}. Use 'full' or 'top_layers_only'.")

        info = {
            "strategy": strategy,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": total_params - trainable_params,
            "trainable_pct": round(100.0 * trainable_params / total_params, 2),
            "trainable_modules": trainable_modules,
            "frozen_modules": frozen_modules,
            "loss_function": "CrossEntropyLoss (InfoNCE with in-batch negatives)",
            "loss_defined_in": "src/retrievers/backbones/vista/modeling.py :: compute_loss()",
            "temperature": self.cfg["training"].get("temperature", 0.02),
        }

        logger.info(f"Freeze strategy   : {strategy}")
        logger.info(f"Trainable params  : {trainable_params:,} / {total_params:,}  ({info['trainable_pct']}%)")
        logger.info(f"Trainable modules : {trainable_modules}")
        return info

    def _save_checkpoint(
        self, backbone: Visualized_BGE, epoch: int, tag: str
    ) -> None:
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        path = ckpt_dir / f"vista_{tag}.pth"
        torch.save(backbone.state_dict(), path)
        logger.info(f"Checkpoint saved → {path}")

    def _log_eval(self, epoch: int, metrics: dict[str, float]) -> None:
        lines = [f"[Epoch {epoch:02d}] Evaluation results:"]
        for k, v in sorted(metrics.items()):
            lines.append(f"  {k}: {v:.4f}")
        logger.info("\n".join(lines))

    def _append_epoch_log(
        self,
        epoch: int,
        elapsed: float,
        train_metrics: dict,
        eval_metrics: dict,
    ) -> None:
        entry = {
            "epoch": epoch,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "train": train_metrics,
            "eval": eval_metrics,
        }
        self._epoch_log.append(entry)
        save_to_json(self._epoch_log, self.output_dir / "train_log.json")
