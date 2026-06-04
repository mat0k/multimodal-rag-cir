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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.datasets.lasco import build_lasco_dataset
from src.datasets.lasco_distill import build_lasco_distill_dataset
from src.datasets.benchmark_distill import build_benchmark_distill_dataset
from src.retrievers.backbones.vista.modeling import Visualized_BGE
from src.retrievers.vista_retriever import VistaImageProcessor
from src.training.collator import LaSCoCollator
from src.training.distill_collator import DistillCollator
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


def _build_distill_dataloader(backbone: Visualized_BGE, cfg: dict) -> DataLoader:
    tc = cfg["training"]
    data_cfg = cfg["data"]

    image_transform = VistaImageProcessor(backbone.preprocess_train)

    def tokenize(text, **kwargs):
        return backbone.tokenizer(text, **kwargs)

    if "lasco_distill" in data_cfg:
        dc = data_cfg["lasco_distill"]
        dataset = build_lasco_distill_dataset(
            subset_path=dc["subset_path"],
            scores_path=dc["scores_path"],
            images_dir=dc["images_dir"],
            image_transform=image_transform,
            caption_transform=tokenize,
            max_length_tokenizer=77,
        )
        logger.info(f"LaSCo distill dataset loaded: {len(dataset):,} triplets")

    elif "benchmark_distill" in data_cfg:
        dc = data_cfg["benchmark_distill"]
        dataset = build_benchmark_distill_dataset(
            triplets_path=dc["triplets_path"],
            scores_path=dc["scores_path"],
            fashioniq_images_dir=dc["fashioniq_images_dir"],
            cirr_images_dir=dc["cirr_images_dir"],
            image_transform=image_transform,
            caption_transform=tokenize,
            max_length_tokenizer=77,
        )
        logger.info(f"Benchmark distill dataset loaded: {len(dataset):,} triplets (FashionIQ + CIRR)")

    else:
        raise ValueError(
            "cfg['data'] must contain 'lasco_distill' or 'benchmark_distill' "
            f"for distillation mode. Got keys: {list(data_cfg.keys())}"
        )

    return DataLoader(
        dataset,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=tc["num_workers"],
        collate_fn=DistillCollator(),
        pin_memory=True,
        drop_last=True,
        persistent_workers=tc["num_workers"] > 0,
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

        if self.training_mode not in ("contrastive", "distillation"):
            raise ValueError(
                f"Unknown training_mode: {self.training_mode!r}. "
                "Use 'contrastive' or 'distillation'."
            )

        backbone = _build_backbone(self.cfg)
        freeze_info = self._freeze_layers(backbone)
        save_to_json(freeze_info, self.output_dir / "freeze_info.json")
        backbone.train()

        if self.training_mode == "distillation":
            loader = _build_distill_dataloader(backbone, self.cfg)
        else:
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
            if self.training_mode == "distillation":
                train_metrics = self._distillation_train_epoch(
                    backbone, loader, optimizer, scheduler, epoch, steps_per_epoch
                )
            else:
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

    def _distillation_train_epoch(
        self,
        backbone: Visualized_BGE,
        loader: DataLoader,
        optimizer: AdamW,
        scheduler: Any,
        epoch: int,
        steps_per_epoch: int,
    ) -> dict[str, float]:
        """Distillation training step — dispatches on distillation.loss from config.

        margin_mse : MSE on pairwise (pos_score − neg_score) margins.
        kl_div     : KL divergence on softmax distribution over pos + K negs (listwise).
        list_mle   : Plackett-Luce log-likelihood of teacher ranking permutation (listwise).
        """
        dist_cfg = self.cfg.get("distillation", {})
        loss_type = dist_cfg.get("loss", "margin_mse")
        kl_temp = float(dist_cfg.get("kl_temperature", 1.0))
        lambda_distill = float(dist_cfg.get("lambda_distill", 0.5))
        contrastive_temp = float(dist_cfg.get("contrastive_temperature", 0.02))
        distill_component = dist_cfg.get("distill_component", "margin_mse")

        valid_losses = ("margin_mse", "kl_div", "list_mle", "combined")
        if loss_type not in valid_losses:
            raise ValueError(f"Unknown distillation loss: {loss_type!r}. Use one of {valid_losses}.")

        device = backbone.device
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_distill_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader, start=1):
            ref_images = batch.ref_images.to(device, non_blocking=True)
            pos_images = batch.pos_images.to(device, non_blocking=True)
            neg_images = batch.neg_images.to(device, non_blocking=True)   # [B, K, C, H, W]
            texts = {
                "input_ids": batch.input_ids.to(device, non_blocking=True),
                "attention_mask": batch.attention_mask.to(device, non_blocking=True),
            }
            teacher_pos = batch.teacher_pos_scores.to(device, non_blocking=True)  # [B]
            teacher_neg = batch.teacher_neg_scores.to(device, non_blocking=True)  # [B, K]

            B, K = teacher_neg.shape

            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                q_emb = backbone.encode_mm(ref_images, texts)    # [B, dim]
                pos_emb = backbone.encode_image(pos_images)       # [B, dim]

                # Encode all K negatives in one batched forward pass
                neg_flat = neg_images.view(B * K, *neg_images.shape[2:])  # [B*K, C, H, W]
                neg_emb = backbone.encode_image(neg_flat).view(B, K, -1)  # [B, K, dim]

                # Cosine similarity (embeddings are L2-normalised by backbone)
                pos_scores = (q_emb * pos_emb).sum(-1)                    # [B]
                neg_scores = (q_emb.unsqueeze(1) * neg_emb).sum(-1)       # [B, K]

                if loss_type == "margin_mse":
                    student_margin = pos_scores.unsqueeze(1) - neg_scores  # [B, K]
                    teacher_margin = teacher_pos.unsqueeze(1) - teacher_neg
                    loss = F.mse_loss(student_margin, teacher_margin) / self.grad_accum

                elif loss_type == "kl_div":
                    all_student = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
                    all_teacher = torch.cat([teacher_pos.unsqueeze(1), teacher_neg], dim=1)
                    p_teacher = F.softmax(all_teacher / kl_temp, dim=-1)
                    log_p_student = F.log_softmax(all_student / kl_temp, dim=-1)
                    loss = F.kl_div(log_p_student, p_teacher, reduction="batchmean") / self.grad_accum

                elif loss_type == "list_mle":
                    all_student = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # [B, K+1]
                    all_teacher = torch.cat([teacher_pos.unsqueeze(1), teacher_neg], dim=1)  # [B, K+1]
                    # Sort items by teacher score descending to define target permutation
                    perm = torch.argsort(all_teacher, dim=-1, descending=True)
                    sorted_student = all_student.gather(dim=-1, index=perm)  # [B, K+1]
                    # Suffix log-sum-exp: logsumexp(sorted_student[i:]) for each position i
                    suffix_lse = torch.logcumsumexp(sorted_student.flip(dims=[-1]), dim=-1).flip(dims=[-1])
                    # Plackett-Luce NLL normalised by list length so scale is independent of K
                    K_plus_1 = sorted_student.shape[-1]
                    loss = (suffix_lse - sorted_student).sum(dim=-1).mean() / K_plus_1 / self.grad_accum

                else:  # combined
                    # Stack pos at index 0, then K negs → [B, K+1]
                    all_student = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

                    # Contrastive: CrossEntropy over K+1 candidates, pos always at index 0
                    labels = torch.zeros(B, dtype=torch.long, device=device)
                    loss_contrastive = F.cross_entropy(all_student / contrastive_temp, labels)

                    # Distillation component
                    if distill_component == "margin_mse":
                        student_margin = pos_scores.unsqueeze(1) - neg_scores
                        teacher_margin = teacher_pos.unsqueeze(1) - teacher_neg
                        loss_distill = F.mse_loss(student_margin, teacher_margin)
                    else:  # kl_div
                        all_teacher = torch.cat([teacher_pos.unsqueeze(1), teacher_neg], dim=1)
                        p_teacher = F.softmax(all_teacher / kl_temp, dim=-1)
                        log_p_student = F.log_softmax(all_student / kl_temp, dim=-1)
                        loss_distill = F.kl_div(log_p_student, p_teacher, reduction="batchmean")

                    loss = ((1 - lambda_distill) * loss_contrastive + lambda_distill * loss_distill) / self.grad_accum
                    total_contrastive_loss += loss_contrastive.item()
                    total_distill_loss += loss_distill.item()

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
                if loss_type == "combined":
                    logger.info(
                        f"[Epoch {epoch:02d} | Step {step:05d}/{steps_per_epoch}] "
                        f"combined_loss={loss.item() * self.grad_accum:.4f}  "
                        f"contrastive={loss_contrastive.item():.4f}  distill={loss_distill.item():.4f}  lr={lr:.2e}"
                    )
                else:
                    logger.info(
                        f"[Epoch {epoch:02d} | Step {step:05d}/{steps_per_epoch}] "
                        f"distill_loss={loss.item() * self.grad_accum:.4f}  lr={lr:.2e}"
                    )

        avg_loss = total_loss / steps_per_epoch
        result = {"avg_loss": avg_loss}
        if loss_type == "combined":
            result["avg_contrastive_loss"] = total_contrastive_loss / steps_per_epoch
            result["avg_distill_loss"] = total_distill_loss / steps_per_epoch
            logger.info(
                f"[Epoch {epoch:02d}] combined_avg_loss={avg_loss:.4f}  "
                f"contrastive={result['avg_contrastive_loss']:.4f}  distill={result['avg_distill_loss']:.4f}"
            )
        else:
            logger.info(f"[Epoch {epoch:02d}] distill_avg_loss={avg_loss:.4f}")
        return result

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
