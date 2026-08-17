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
from src.datasets.benchmark_contrastive import build_benchmark_contrastive_dataset
from src.datasets.lasco_distill import build_lasco_distill_dataset
from src.datasets.lasco_relation_distill import build_lasco_relation_distill_dataset
from src.datasets.lasco_sp_distill import build_lasco_sp_distill_dataset
from src.datasets.benchmark_distill import build_benchmark_distill_dataset
from src.retrievers.backbones.vista.modeling import Visualized_BGE
from src.retrievers.vista_retriever import VistaImageProcessor
from src.training.collator import LaSCoCollator
from src.training.distill_collator import DistillCollator
from src.training.relation_collator import RelationCollator
from src.training.cluster_batch_sampler import ClusterBatchSampler
from src.training.sp_collator import SPCollator
from src.utils.io import save_to_json

logger = logging.getLogger(__name__)


def _zscore(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Per-row (per-query) standardisation: zero mean, unit std along `dim`.

    Used to put teacher and student scores on a common scale before the
    distillation loss.  Teacher reranker scores live on a tiny, teacher-specific
    range (e.g. Qwen ~0.01-0.12) that is incomparable to the student's cosine
    similarities; standardising each query's K+1 scores keeps only the *ranking
    structure*, which is what we actually want to distil.
    """
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    return (x - mean) / (std + eps)


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

    if "lasco_sp_distill" in data_cfg:
        dc = data_cfg["lasco_sp_distill"]
        dataset = build_lasco_sp_distill_dataset(
            subset_path=dc["subset_path"],
            teacher_cand_emb_path=dc["teacher_cand_emb_path"],
            images_dir=dc["images_dir"],
            cluster_labels_path=dc.get("cluster_labels_path"),  # None -> random floor
            image_transform=image_transform,
            caption_transform=tokenize,
            max_length_tokenizer=77,
            # Feature-based KD only; None leaves SP/RKD/CRD behaviour unchanged.
            teacher_query_emb_path=dc.get("teacher_query_emb_path"),
        )
        sampler = ClusterBatchSampler(
            cluster_ids=dataset.cluster_ids,
            batch_size=tc["batch_size"],
            n_items=len(dataset),
            shuffle=True,
            seed=cfg.get("seed", 42),
            target_ids=dataset.target_ids,
            unique_per_batch=bool(dc.get("unique_per_batch", False)),
        )
        logger.info(
            f"LaSCo SP distill dataset loaded: {len(dataset):,} pairs | "
            f"{len(sampler):,} cluster-batches of size {tc['batch_size']} "
            f"({'random floor' if dataset.cluster_ids is None else 'clustered'})"
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=tc["num_workers"],
            collate_fn=SPCollator(),
            pin_memory=True,
            persistent_workers=tc["num_workers"] > 0,
        )

    if "lasco_relation_distill" in data_cfg:
        dc = data_cfg["lasco_relation_distill"]
        dataset = build_lasco_relation_distill_dataset(
            subset_path=dc["subset_path"],
            matrix_path=dc["matrix_path"],
            images_dir=dc["images_dir"],
            image_transform=image_transform,
            caption_transform=tokenize,
            max_length_tokenizer=77,
        )
        logger.info(
            f"LaSCo relation distill dataset loaded: {len(dataset):,} groups "
            f"of size {dataset.group_size}"
        )
        return DataLoader(
            dataset,
            batch_size=tc["batch_size"],          # = groups per step
            shuffle=True,
            num_workers=tc["num_workers"],
            collate_fn=RelationCollator(),
            pin_memory=True,
            drop_last=True,
            persistent_workers=tc["num_workers"] > 0,
        )

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
            fashioniq_images_dir=dc.get("fashioniq_images_dir"),
            cirr_images_dir=dc.get("cirr_images_dir"),
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
    data_cfg = cfg["data"]

    image_transform = VistaImageProcessor(backbone.preprocess_train)

    def tokenize(text, **kwargs):
        return backbone.tokenizer(text, **kwargs)

    if "benchmark_contrastive" in data_cfg:
        dc = data_cfg["benchmark_contrastive"]
        dataset = build_benchmark_contrastive_dataset(
            triplets_path=dc["triplets_path"],
            fashioniq_images_dir=dc.get("fashioniq_images_dir"),
            cirr_images_dir=dc.get("cirr_images_dir"),
            sources=dc.get("sources"),
            image_transform=image_transform,
            caption_transform=tokenize,
            max_length_tokenizer=77,
        )
        logger.info(
            f"Benchmark contrastive dataset loaded: {len(dataset):,} triplets "
            f"(sources={dc.get('sources')})"
        )
    else:
        dc = data_cfg["lasco"]
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
    extra_params: list | None = None,
    extra_lr: float | None = None,
) -> tuple[AdamW, Any]:
    tc = cfg["training"]
    warmup_steps = int(total_steps * tc["warmup_ratio"])

    base = list(filter(lambda p: p.requires_grad, backbone.parameters()))
    if extra_params:
        # Separate param group: freshly-init CRD projection heads need a much
        # higher lr than the pretrained backbone (1e-6 barely moves a new MLP).
        e_lr = extra_lr if extra_lr is not None else tc["learning_rate"]
        optimizer = AdamW(
            [
                {"params": base, "lr": tc["learning_rate"]},
                {"params": list(extra_params), "lr": e_lr},
            ],
            weight_decay=tc["weight_decay"], betas=(0.9, 0.999), eps=1e-8,
        )
        logger.info(f"Optimizer: 2 param groups — backbone lr={tc['learning_rate']}  projector lr={e_lr}")
    else:
        optimizer = AdamW(
            base,
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

        # CRD needs trainable projection heads (student/teacher -> shared space),
        # optimised alongside the backbone. Built only for loss: crd.
        extra_params = None
        extra_lr = None
        self.crd_projector = None
        if self.cfg.get("distillation", {}).get("loss") == "crd":
            from src.training.crd_projector import CRDProjector
            dcfg = self.cfg["distillation"]
            self.crd_projector = CRDProjector(
                student_dim=int(dcfg.get("student_dim", 768)),
                teacher_dim=int(dcfg.get("teacher_dim", 3584)),
                proj_dim=int(dcfg.get("projection_dim", 128)),
                student_head=dcfg.get("student_projector", "mlp"),
                teacher_head=dcfg.get("teacher_projector", "mlp"),
            ).to(backbone.device)
            self.crd_projector.train()
            extra_params = list(self.crd_projector.parameters())
            extra_lr = float(dcfg.get("projector_lr", 1.0e-3))
            logger.info(
                f"CRD projector built: proj_dim={dcfg.get('projection_dim', 128)}  "
                f"projector_lr={extra_lr}"
            )

        # Feature-based KD (FitNets/EmbedDistill-style embedding matching).
        # mode="up"  -> trainable Linear(student -> teacher), alignment in teacher space.
        # mode="none"-> teacher pre-projected offline to student dim; ZERO trainable
        #               projection, so the loss lands on the student's native
        #               retrieval embedding and absorption is impossible.
        self.feature_projector = None
        if self.cfg.get("distillation", {}).get("loss") == "feature":
            from src.training.feature_projector import FeatureProjector
            dcfg = self.cfg["distillation"]
            # Never hardcode the teacher dim: read it off the cache actually loaded.
            teacher_dim = int(loader.dataset.teacher_emb.shape[-1])
            student_dim = int(dcfg.get("student_dim", 768))
            mode = dcfg.get("feature_mode", "up")
            self.feature_projector = FeatureProjector(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                mode=mode,
                head=dcfg.get("feature_head", "linear"),
            ).to(backbone.device)
            self.feature_projector.train()
            n_proj = sum(p.numel() for p in self.feature_projector.parameters())
            if self.feature_projector.has_params:
                extra_params = list(self.feature_projector.parameters())
                extra_lr = float(dcfg.get("projector_lr", 1.0e-3))
            logger.info(
                f"Feature projector built: mode={mode} head={dcfg.get('feature_head','linear')} "
                f"student_dim={student_dim} teacher_dim={teacher_dim} (read from cache) "
                f"params={n_proj:,} projector_lr={extra_lr}"
            )
            if loader.dataset.teacher_query_emb is None:
                logger.warning(
                    "feature KD: no teacher_query_emb_path set — query-side matching "
                    "will be SKIPPED (candidate-side only)."
                )

        optimizer, scheduler = _build_optimizer_and_scheduler(
            backbone, self.cfg, total_steps, extra_params=extra_params, extra_lr=extra_lr
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

        if loss_type == "relation":
            return self._relation_train_epoch(
                backbone, loader, optimizer, scheduler, epoch, steps_per_epoch
            )

        if loss_type == "sp":
            return self._sp_train_epoch(
                backbone, loader, optimizer, scheduler, epoch, steps_per_epoch
            )

        if loss_type == "rkd":
            return self._rkd_train_epoch(
                backbone, loader, optimizer, scheduler, epoch, steps_per_epoch
            )

        if loss_type == "crd":
            return self._crd_train_epoch(
                backbone, loader, optimizer, scheduler, epoch, steps_per_epoch
            )

        if loss_type == "relation_joint":
            return self._relation_joint_train_epoch(
                backbone, loader, optimizer, scheduler, epoch, steps_per_epoch
            )

        if loss_type == "feature":
            return self._feature_train_epoch(
                backbone, loader, optimizer, scheduler, epoch, steps_per_epoch
            )

        kl_temp = float(dist_cfg.get("kl_temperature", 1.0))
        lambda_distill = float(dist_cfg.get("lambda_distill", 0.5))
        contrastive_temp = float(dist_cfg.get("contrastive_temperature", 0.02))
        distill_component = dist_cfg.get("distill_component", "margin_mse")
        # Per-query standardisation of teacher+student scores before the distill
        # loss. Off by default to keep older runs reproducible; the contrastive
        # CE term always uses raw cosine sims regardless of this flag.
        normalize_distill = bool(dist_cfg.get("normalize_scores", False))

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
                    all_student = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # [B, K+1]
                    all_teacher = torch.cat([teacher_pos.unsqueeze(1), teacher_neg], dim=1)
                    if normalize_distill:
                        all_student = _zscore(all_student)
                        all_teacher = _zscore(all_teacher)
                    student_margin = all_student[:, :1] - all_student[:, 1:]  # [B, K]
                    teacher_margin = all_teacher[:, :1] - all_teacher[:, 1:]
                    loss = F.mse_loss(student_margin, teacher_margin) / self.grad_accum

                elif loss_type == "kl_div":
                    all_student = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
                    all_teacher = torch.cat([teacher_pos.unsqueeze(1), teacher_neg], dim=1)
                    if normalize_distill:
                        all_student = _zscore(all_student)
                        all_teacher = _zscore(all_teacher)
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
                    #
                    #  Stack pos at index 0, then K negs → [B, K+1]
                    all_student = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
                    all_teacher = torch.cat([teacher_pos.unsqueeze(1), teacher_neg], dim=1)

                    # Contrastive: CrossEntropy over K+1 candidates, pos always at
                    # index 0. ALWAYS uses raw cosine sims — normalisation must not
                    # touch the retrieval objective.
                    labels = torch.zeros(B, dtype=torch.long, device=device)
                    loss_contrastive = F.cross_entropy(all_student / contrastive_temp, labels)

                    # Distillation component operates on (optionally) standardised
                    # scores so the teacher's ranking structure is on a usable scale.
                    s_vec = _zscore(all_student) if normalize_distill else all_student
                    t_vec = _zscore(all_teacher) if normalize_distill else all_teacher
                    if distill_component == "margin_mse":
                        student_margin = s_vec[:, :1] - s_vec[:, 1:]
                        teacher_margin = t_vec[:, :1] - t_vec[:, 1:]
                        loss_distill = F.mse_loss(student_margin, teacher_margin)
                    else:  # kl_div
                        p_teacher = F.softmax(t_vec / kl_temp, dim=-1)
                        log_p_student = F.log_softmax(s_vec / kl_temp, dim=-1)
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

    def _relation_train_epoch(
        self,
        backbone: Visualized_BGE,
        loader: DataLoader,
        optimizer: AdamW,
        scheduler: Any,
        epoch: int,
        steps_per_epoch: int,
    ) -> dict[str, float]:
        """Relation-based (group-matrix) distillation.

        For each group of G queries sharing a pool of G candidates, the student
        builds a G x G cosine-similarity matrix and is trained to reproduce the
        STRUCTURE of the teacher's G x G relevance matrix via:
          - row-wise KL  : per query,     distribution over candidates  (q -> c)
          - col-wise KL  : per candidate, distribution over queries      (c -> q)
          - contrastive  : cross-entropy with diagonal labels (in-group negatives)

        total = lambda_contrastive * CE + row_weight * KL_row + col_weight * KL_col

        Set col_weight=0 to recover a pure row-wise (response-style) listwise
        objective — the ablation that isolates the relational (column) signal.
        """
        dist_cfg = self.cfg.get("distillation", {})
        kl_temp = float(dist_cfg.get("kl_temperature", 1.0))
        row_weight = float(dist_cfg.get("row_weight", 1.0))
        col_weight = float(dist_cfg.get("col_weight", 1.0))
        lambda_contrastive = float(dist_cfg.get("lambda_contrastive", 0.2))
        contrastive_temp = float(dist_cfg.get("contrastive_temperature", 0.02))
        # Per-row/col standardisation of teacher+student before the KL terms.
        # The teacher matrix (qwen3vl_p_yes) is compressed to ~0.03-0.11, so its
        # softmax is ~uniform (entropy ratio 1.0) and the KD signal vanishes;
        # z-scoring restores usable structure. CE always uses raw cosine sims.
        normalize_distill = bool(dist_cfg.get("normalize_scores", False))

        device = backbone.device
        total_loss = total_row = total_col = total_ce = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader, start=1):
            B, G = batch.teacher_matrix.shape[:2]
            ref = batch.ref_images.to(device, non_blocking=True)       # [B, G, C, H, W]
            cand = batch.cand_images.to(device, non_blocking=True)     # [B, G, C, H, W]
            input_ids = batch.input_ids.to(device, non_blocking=True)  # [B, G, T]
            attention_mask = batch.attention_mask.to(device, non_blocking=True)
            teacher = batch.teacher_matrix.to(device, non_blocking=True)  # [B, G, G]

            C, H, W = ref.shape[2:]
            T = input_ids.shape[-1]
            ref_flat = ref.view(B * G, C, H, W)
            cand_flat = cand.view(B * G, C, H, W)
            texts = {
                "input_ids": input_ids.view(B * G, T),
                "attention_mask": attention_mask.view(B * G, T),
            }

            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                q_emb = backbone.encode_mm(ref_flat, texts).view(B, G, -1)  # [B, G, d]
                c_emb = backbone.encode_image(cand_flat).view(B, G, -1)     # [B, G, d]

                # Student G x G cosine-similarity matrix per group
                student = torch.bmm(q_emb, c_emb.transpose(1, 2))          # [B, G, G]

                # Raw cosine rows/cols (CE uses these; KL may use z-scored copies)
                s_rows_raw = student.reshape(B * G, G)
                s_cols_raw = student.transpose(1, 2).reshape(B * G, G)
                t_rows = teacher.reshape(B * G, G)
                t_cols = teacher.transpose(1, 2).reshape(B * G, G)

                if normalize_distill:
                    s_rows, t_rows = _zscore(s_rows_raw), _zscore(t_rows)
                    s_cols, t_cols = _zscore(s_cols_raw), _zscore(t_cols)
                else:
                    s_rows, s_cols = s_rows_raw, s_cols_raw

                # row-wise KL: per query over candidates
                kl_row = F.kl_div(
                    F.log_softmax(s_rows / kl_temp, dim=-1),
                    F.softmax(t_rows / kl_temp, dim=-1),
                    reduction="batchmean",
                )

                # col-wise KL: per candidate over queries (transpose first)
                kl_col = F.kl_div(
                    F.log_softmax(s_cols / kl_temp, dim=-1),
                    F.softmax(t_cols / kl_temp, dim=-1),
                    reduction="batchmean",
                )

                # contrastive CE: diagonal is the positive within each group.
                # Always on raw cosine sims — normalisation must not touch this.
                labels = torch.arange(G, device=device).repeat(B)         # [B*G]
                ce = F.cross_entropy(s_rows_raw / contrastive_temp, labels)

                loss = (
                    lambda_contrastive * ce
                    + row_weight * kl_row
                    + col_weight * kl_col
                ) / self.grad_accum

            loss.backward()
            total_loss += loss.item() * self.grad_accum
            total_row += kl_row.item()
            total_col += kl_col.item()
            total_ce += ce.item()

            if step % self.grad_accum == 0 or step == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % self.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"[Epoch {epoch:02d} | Step {step:05d}/{steps_per_epoch}] "
                    f"relation_loss={loss.item() * self.grad_accum:.4f}  "
                    f"row_kl={kl_row.item():.4f}  col_kl={kl_col.item():.4f}  "
                    f"ce={ce.item():.4f}  lr={lr:.2e}"
                )

        avg_loss = total_loss / steps_per_epoch
        result = {
            "avg_loss": avg_loss,
            "avg_row_kl": total_row / steps_per_epoch,
            "avg_col_kl": total_col / steps_per_epoch,
            "avg_contrastive_loss": total_ce / steps_per_epoch,
        }
        logger.info(
            f"[Epoch {epoch:02d}] relation_avg_loss={avg_loss:.4f}  "
            f"row_kl={result['avg_row_kl']:.4f}  col_kl={result['avg_col_kl']:.4f}  "
            f"ce={result['avg_contrastive_loss']:.4f}"
        )
        return result

    def _sp_train_epoch(
        self,
        backbone: Visualized_BGE,
        loader: DataLoader,
        optimizer: AdamW,
        scheduler: Any,
        epoch: int,
        steps_per_epoch: int,
    ) -> dict[str, float]:
        """Similarity-Preserving (SP) relation distillation (Tung & Mori, ICCV'19).

        Each batch is N (query, target) pairs from ONE semantic cluster. We distil
        the teacher's candidate x candidate similarity STRUCTURE into the student:

          A        = student target embeddings            [N, d_student]
          A_t      = teacher target embeddings (cached)    [N, d_teacher]
          G  = rownorm_L2(A  @ A^T)                         [N, N]
          G_t= rownorm_L2(A_t @ A_t^T)                      [N, N]
          L_SP = || G_t - G ||_F^2 / N^2

        Plus a supervised CE anchor on TRUE positives (never dropped):
          CE(query_i -> target_i), in-batch negatives, with known-duplicate
          positives masked out so we never push apart labelled-equivalent images.

          L = lambda_ce * CE + sp_weight * L_SP

        False-negative meter: per batch we log how many in-batch negatives have
        teacher-cosine > fn_threshold to the query's true target (watched against
        the recall curve as clustering tightens).
        """
        dist_cfg = self.cfg.get("distillation", {})
        sp_weight = float(dist_cfg.get("sp_weight", 1.0))
        lambda_ce = float(dist_cfg.get("lambda_contrastive", 1.0))
        contrastive_temp = float(dist_cfg.get("contrastive_temperature", 0.02))
        fn_threshold = float(dist_cfg.get("fn_threshold", 0.8))

        device = backbone.device
        total_loss = total_sp = total_ce = 0.0
        total_fn = 0.0  # avg fraction of in-batch negatives above fn_threshold
        optimizer.zero_grad()

        for step, batch in enumerate(loader, start=1):
            ref_images = batch.ref_images.to(device, non_blocking=True)       # [N, C, H, W]
            target_images = batch.target_images.to(device, non_blocking=True) # [N, C, H, W]
            texts = {
                "input_ids": batch.input_ids.to(device, non_blocking=True),
                "attention_mask": batch.attention_mask.to(device, non_blocking=True),
            }
            teacher_t = batch.teacher_target_emb.to(device, non_blocking=True)  # [N, d_teacher]
            N = teacher_t.shape[0]

            # duplicate-positive mask: target_j is also a positive for query_i.
            ids = batch.target_ids
            dup = torch.zeros(N, N, dtype=torch.bool, device=device)
            for i in range(N):
                for j in range(N):
                    if i != j and ids[i] == ids[j]:
                        dup[i, j] = True

            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                q_emb = backbone.encode_mm(ref_images, texts)      # [N, d_student] (L2-normed)
                c_emb = backbone.encode_image(target_images)        # [N, d_student] (L2-normed)

                # --- CE anchor on true positives (mask known-duplicate positives) ---
                logits = (q_emb @ c_emb.T) / contrastive_temp       # [N, N]
                logits = logits.masked_fill(dup, float("-inf"))
                labels = torch.arange(N, device=device)
                ce = F.cross_entropy(logits, labels)

                # --- SP: candidate-candidate structure (row-normalised Gram) ---
                G_s = F.normalize(c_emb @ c_emb.T, p=2, dim=1)       # [N, N]
                G_t = F.normalize(teacher_t @ teacher_t.T, p=2, dim=1)
                sp = ((G_t - G_s) ** 2).sum() / (N * N)

                loss = (lambda_ce * ce + sp_weight * sp) / self.grad_accum

            loss.backward()
            total_loss += loss.item() * self.grad_accum
            total_sp += sp.item()
            total_ce += ce.item()

            # false-negative meter (teacher cosine of negatives to the true target)
            with torch.no_grad():
                t_sim = teacher_t @ teacher_t.T                     # [N, N] teacher cosine
                offdiag = ~torch.eye(N, dtype=torch.bool, device=device)
                fn_frac = ((t_sim > fn_threshold) & offdiag).float().sum() / max(N * (N - 1), 1)
                total_fn += float(fn_frac)

            if step % self.grad_accum == 0 or step == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % self.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"[Epoch {epoch:02d} | Step {step:05d}/{steps_per_epoch}] "
                    f"sp_loss={loss.item() * self.grad_accum:.4f}  sp={sp.item():.4f}  "
                    f"ce={ce.item():.4f}  fn_frac={float(fn_frac):.3f}  lr={lr:.2e}"
                )

        result = {
            "avg_loss": total_loss / steps_per_epoch,
            "avg_sp": total_sp / steps_per_epoch,
            "avg_contrastive_loss": total_ce / steps_per_epoch,
            "avg_false_neg_frac": total_fn / steps_per_epoch,
        }
        logger.info(
            f"[Epoch {epoch:02d}] sp_avg_loss={result['avg_loss']:.4f}  "
            f"sp={result['avg_sp']:.4f}  ce={result['avg_contrastive_loss']:.4f}  "
            f"false_neg_frac={result['avg_false_neg_frac']:.3f}"
        )
        return result

    def _rkd_train_epoch(
        self,
        backbone: Visualized_BGE,
        loader: DataLoader,
        optimizer: AdamW,
        scheduler: Any,
        epoch: int,
        steps_per_epoch: int,
    ) -> dict[str, float]:
        """Relational Knowledge Distillation (Park et al., CVPR'19) — the canonical
        SECOND relation-based method, over the batch's target images.

        Distills the teacher's mutual relations between candidate images via:
          - distance-wise : match mean-normalised pairwise distances
          - angle-wise    : match triplet angles  cos<e_ij, e_kj>,  e_ij = norm(v_i - v_j)
        Both use smooth-L1 (Huber). Same supervised CE anchor on TRUE positives as SP.

          L = lambda_ce * CE + dist_weight * L_dist + angle_weight * L_angle

        Reuses the SP dataset/collator (target images + cached teacher embeddings).
        """
        dist_cfg = self.cfg.get("distillation", {})
        w_dist = float(dist_cfg.get("rkd_dist_weight", 25.0))
        w_angle = float(dist_cfg.get("rkd_angle_weight", 50.0))
        lambda_ce = float(dist_cfg.get("lambda_contrastive", 1.0))
        contrastive_temp = float(dist_cfg.get("contrastive_temperature", 0.02))

        def _rkd_distance(s: torch.Tensor, t: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
            ds, dt = torch.cdist(s, s), torch.cdist(t, t)
            ds = ds / ds[off].mean().clamp_min(1e-6)
            dt = dt / dt[off].mean().clamp_min(1e-6)
            return F.smooth_l1_loss(ds[off], dt[off])

        def _rkd_angle(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            def ang(x):
                e = F.normalize(x.unsqueeze(0) - x.unsqueeze(1), dim=-1)  # e[i,j]=norm(x_i-x_j) [N,N,d]
                return torch.einsum("ijd,kjd->ijk", e, e)                 # <e_ij, e_kj>
            return F.smooth_l1_loss(ang(s.float()), ang(t.float()))

        device = backbone.device
        total_loss = total_ce = total_d = total_a = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader, start=1):
            ref = batch.ref_images.to(device, non_blocking=True)
            tgt = batch.target_images.to(device, non_blocking=True)
            texts = {
                "input_ids": batch.input_ids.to(device, non_blocking=True),
                "attention_mask": batch.attention_mask.to(device, non_blocking=True),
            }
            teacher = batch.teacher_target_emb.to(device, non_blocking=True)  # [N, d_t]
            N = teacher.shape[0]
            off = ~torch.eye(N, dtype=torch.bool, device=device)

            ids = batch.target_ids
            dup = torch.zeros(N, N, dtype=torch.bool, device=device)
            for i in range(N):
                for j in range(N):
                    if i != j and ids[i] == ids[j]:
                        dup[i, j] = True

            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                q = backbone.encode_mm(ref, texts)     # [N, d_s]
                c = backbone.encode_image(tgt)          # [N, d_s]

                logits = (q @ c.T) / contrastive_temp
                logits = logits.masked_fill(dup, float("-inf"))
                ce = F.cross_entropy(logits, torch.arange(N, device=device))

                l_dist = _rkd_distance(c.float(), teacher.float(), off)
                l_angle = _rkd_angle(c, teacher)

                loss = (lambda_ce * ce + w_dist * l_dist + w_angle * l_angle) / self.grad_accum

            loss.backward()
            total_loss += loss.item() * self.grad_accum
            total_ce += ce.item()
            total_d += l_dist.item()
            total_a += l_angle.item()

            if step % self.grad_accum == 0 or step == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % self.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"[Epoch {epoch:02d} | Step {step:05d}/{steps_per_epoch}] "
                    f"rkd_loss={loss.item() * self.grad_accum:.4f}  ce={ce.item():.4f}  "
                    f"dist={l_dist.item():.4f}  angle={l_angle.item():.4f}  lr={lr:.2e}"
                )

        result = {
            "avg_loss": total_loss / steps_per_epoch,
            "avg_contrastive_loss": total_ce / steps_per_epoch,
            "avg_rkd_dist": total_d / steps_per_epoch,
            "avg_rkd_angle": total_a / steps_per_epoch,
        }
        logger.info(
            f"[Epoch {epoch:02d}] rkd_avg_loss={result['avg_loss']:.4f}  "
            f"ce={result['avg_contrastive_loss']:.4f}  dist={result['avg_rkd_dist']:.4f}  "
            f"angle={result['avg_rkd_angle']:.4f}"
        )
        return result

    def _crd_train_epoch(
        self,
        backbone: Visualized_BGE,
        loader: DataLoader,
        optimizer: AdamW,
        scheduler: Any,
        epoch: int,
        steps_per_epoch: int,
    ) -> dict[str, float]:
        """Contrastive Representation Distillation (Tian et al., ICLR'20).

        Cross-network per-instance alignment: the student's rep of a target image
        should agree with the TEACHER's rep of the SAME image (positive) and
        disagree with the teacher's reps of OTHER images (negatives) — an InfoNCE
        objective that maximises I(teacher; student). Both reps are pushed through
        trainable projection heads into a shared space (self.crd_projector).

        v1 uses in-batch negatives (N-1 per anchor); the frozen teacher buffer can
        extend this later. Same supervised CE anchor on TRUE positives as SP/RKD.

          L = lambda_ce * CE + crd_weight * InfoNCE(student_i <-> teacher_i)
        """
        dist_cfg = self.cfg.get("distillation", {})
        crd_weight = float(dist_cfg.get("crd_weight", 1.0))
        crd_temp = float(dist_cfg.get("crd_temperature", 0.07))
        lambda_ce = float(dist_cfg.get("lambda_contrastive", 1.0))
        contrastive_temp = float(dist_cfg.get("contrastive_temperature", 0.02))
        proj = self.crd_projector

        device = backbone.device
        total_loss = total_ce = total_crd = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader, start=1):
            ref = batch.ref_images.to(device, non_blocking=True)
            tgt = batch.target_images.to(device, non_blocking=True)
            texts = {
                "input_ids": batch.input_ids.to(device, non_blocking=True),
                "attention_mask": batch.attention_mask.to(device, non_blocking=True),
            }
            teacher = batch.teacher_target_emb.to(device, non_blocking=True)  # [N, d_t]
            N = teacher.shape[0]

            ids = batch.target_ids
            dup = torch.zeros(N, N, dtype=torch.bool, device=device)
            for i in range(N):
                for j in range(N):
                    if i != j and ids[i] == ids[j]:
                        dup[i, j] = True
            labels = torch.arange(N, device=device)

            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                q = backbone.encode_mm(ref, texts)     # [N, d_s]
                c = backbone.encode_image(tgt)          # [N, d_s]

                # CE anchor (retrieval): query_i -> target_i, dup positives masked.
                ce_logits = (q @ c.T) / contrastive_temp
                ce_logits = ce_logits.masked_fill(dup, float("-inf"))
                ce = F.cross_entropy(ce_logits, labels)

                # CRD InfoNCE: student_i rep vs teacher_j reps (positive = j==i).
                z_s = proj.project_student(c).float()          # [N, p]
                z_t = proj.project_teacher(teacher).float()    # [N, p]
                crd_logits = (z_s @ z_t.T) / crd_temp          # [N, N]
                crd_logits = crd_logits.masked_fill(dup, float("-inf"))
                crd = F.cross_entropy(crd_logits, labels)

                loss = (lambda_ce * ce + crd_weight * crd) / self.grad_accum

            loss.backward()
            total_loss += loss.item() * self.grad_accum
            total_ce += ce.item()
            total_crd += crd.item()

            if step % self.grad_accum == 0 or step == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % self.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"[Epoch {epoch:02d} | Step {step:05d}/{steps_per_epoch}] "
                    f"crd_loss={loss.item() * self.grad_accum:.4f}  ce={ce.item():.4f}  "
                    f"crd={crd.item():.4f}  lr={lr:.2e}"
                )

        result = {
            "avg_loss": total_loss / steps_per_epoch,
            "avg_contrastive_loss": total_ce / steps_per_epoch,
            "avg_crd": total_crd / steps_per_epoch,
        }
        logger.info(
            f"[Epoch {epoch:02d}] crd_avg_loss={result['avg_loss']:.4f}  "
            f"ce={result['avg_contrastive_loss']:.4f}  crd={result['avg_crd']:.4f}"
        )
        return result

    def _feature_train_epoch(
        self,
        backbone: Visualized_BGE,
        loader: DataLoader,
        optimizer: AdamW,
        scheduler: Any,
        epoch: int,
        steps_per_epoch: int,
    ) -> dict[str, float]:
        """FEATURE-based distillation — embedding matching (FitNets / EmbedDistill).

        Unlike SP/RKD (which match *relative* structure and are invariant to any
        rotation of the space), this pins the student's embedding to the teacher's
        ABSOLUTE coordinates — a strictly stronger constraint.

          L = lambda_ce * CE  +  feat_weight * ( ||q_hat - q_t||^2 + ||c_hat - c_t||^2 )

        where ||.||^2 is summed over dims and averaged over the batch, so with both
        sides L2-normalized it equals 2 - 2*cos in [0, 4] — dimension-independent,
        which keeps feat_weight sane (see feature_projector.squared_l2).

        Two modes, opposed by design to diagnose projector ABSORPTION:
          up   : q_hat = normalize(W_q @ q_s) in TEACHER space (trainable head).
          none : q_hat = q_s, the student's NATIVE retrieval embedding, matched
                 against an offline-projected teacher. No trainable projection,
                 so the head cannot absorb the alignment.

        Logged per step: the two loss terms AND the mean student-teacher cosine,
        so absorption (feat loss falling while recall stays flat) is visible early.
        """
        from src.training.feature_projector import squared_l2

        dist_cfg = self.cfg.get("distillation", {})
        feat_weight = float(dist_cfg.get("feat_weight", 1.0))
        lambda_ce = float(dist_cfg.get("lambda_contrastive", 1.0))
        contrastive_temp = float(dist_cfg.get("contrastive_temperature", 0.02))
        align_query = bool(dist_cfg.get("align_query", True))
        align_cand = bool(dist_cfg.get("align_candidate", True))
        proj = self.feature_projector

        device = backbone.device
        total_loss = total_ce = total_feat = 0.0
        total_cos_q = total_cos_c = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader, start=1):
            ref = batch.ref_images.to(device, non_blocking=True)
            tgt = batch.target_images.to(device, non_blocking=True)
            texts = {
                "input_ids": batch.input_ids.to(device, non_blocking=True),
                "attention_mask": batch.attention_mask.to(device, non_blocking=True),
            }
            t_cand = batch.teacher_target_emb.to(device, non_blocking=True)   # [N, d_t]
            t_query = (
                batch.teacher_query_emb.to(device, non_blocking=True)
                if batch.teacher_query_emb is not None else None
            )
            N = t_cand.shape[0]

            ids = batch.target_ids
            dup = torch.zeros(N, N, dtype=torch.bool, device=device)
            for i in range(N):
                for j in range(N):
                    if i != j and ids[i] == ids[j]:
                        dup[i, j] = True
            labels = torch.arange(N, device=device)

            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                q = backbone.encode_mm(ref, texts)      # [N, d_s] (L2-normed)
                c = backbone.encode_image(tgt)          # [N, d_s]

                # CE anchor on TRUE positives — kept in every arm, never dropped.
                ce_logits = (q @ c.T) / contrastive_temp
                ce_logits = ce_logits.masked_fill(dup, float("-inf"))
                ce = F.cross_entropy(ce_logits, labels)

                # Embedding matching. In mode="none" project_* is identity, so the
                # loss lands on exactly the embedding retrieval uses.
                feat = q.new_zeros(())
                cos_q = cos_c = 0.0
                if align_cand:
                    c_hat = proj.project_candidate(c).float()
                    tc = F.normalize(t_cand.float(), dim=-1)
                    feat = feat + squared_l2(c_hat, tc)
                    cos_c = float((c_hat * tc).sum(-1).mean())
                if align_query and t_query is not None:
                    q_hat = proj.project_query(q).float()
                    tq = F.normalize(t_query.float(), dim=-1)
                    feat = feat + squared_l2(q_hat, tq)
                    cos_q = float((q_hat * tq).sum(-1).mean())

                loss = (lambda_ce * ce + feat_weight * feat) / self.grad_accum

            loss.backward()
            total_loss += loss.item() * self.grad_accum
            total_ce += ce.item()
            total_feat += float(feat)
            total_cos_q += cos_q
            total_cos_c += cos_c

            if step % self.grad_accum == 0 or step == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % self.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"[Epoch {epoch:02d} | Step {step:05d}/{steps_per_epoch}] "
                    f"feat_loss={loss.item() * self.grad_accum:.4f}  ce={ce.item():.4f}  "
                    f"feat={float(feat):.4f}  cos_q={cos_q:.4f}  cos_c={cos_c:.4f}  lr={lr:.2e}"
                )

        result = {
            "avg_loss": total_loss / steps_per_epoch,
            "avg_contrastive_loss": total_ce / steps_per_epoch,
            "avg_feat": total_feat / steps_per_epoch,
            "avg_cos_query": total_cos_q / steps_per_epoch,
            "avg_cos_candidate": total_cos_c / steps_per_epoch,
        }
        logger.info(
            f"[Epoch {epoch:02d}] feat_avg_loss={result['avg_loss']:.4f}  "
            f"ce={result['avg_contrastive_loss']:.4f}  feat={result['avg_feat']:.4f}  "
            f"cos_q={result['avg_cos_query']:.4f}  cos_c={result['avg_cos_candidate']:.4f}"
        )
        return result

    def _relation_joint_train_epoch(
        self,
        backbone: Visualized_BGE,
        loader: DataLoader,
        optimizer: AdamW,
        scheduler: Any,
        epoch: int,
        steps_per_epoch: int,
    ) -> dict[str, float]:
        """Joint relation distillation: SP + RKD (distance + angle), sharing one
        batch of target images and the same CE anchor on true positives. Tests
        whether two complementary relational signals stack.

          L = lambda_ce*CE + sp_weight*SP
              + rkd_dist_weight*RKD_dist + rkd_angle_weight*RKD_angle

        SP  = ||rownorm(G_t) - rownorm(G_s)||_F^2 / N^2   (pairwise similarity)
        RKD = smooth_l1 on mean-normalised pairwise distances + triplet angles.
        Set any weight to 0 to drop that term. Reuses the SP dataset/collator.
        """
        dc = self.cfg.get("distillation", {})
        sp_weight = float(dc.get("sp_weight", 3000.0))
        w_dist = float(dc.get("rkd_dist_weight", 500.0))
        w_angle = float(dc.get("rkd_angle_weight", 500.0))
        lambda_ce = float(dc.get("lambda_contrastive", 1.0))
        contrastive_temp = float(dc.get("contrastive_temperature", 0.02))

        def _rkd_distance(s, t, off):
            ds, dt = torch.cdist(s, s), torch.cdist(t, t)
            ds = ds / ds[off].mean().clamp_min(1e-6)
            dt = dt / dt[off].mean().clamp_min(1e-6)
            return F.smooth_l1_loss(ds[off], dt[off])

        def _rkd_angle(s, t):
            def ang(x):
                e = F.normalize(x.unsqueeze(0) - x.unsqueeze(1), dim=-1)
                return torch.einsum("ijd,kjd->ijk", e, e)
            return F.smooth_l1_loss(ang(s.float()), ang(t.float()))

        device = backbone.device
        total_loss = total_ce = total_sp = total_d = total_a = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader, start=1):
            ref = batch.ref_images.to(device, non_blocking=True)
            tgt = batch.target_images.to(device, non_blocking=True)
            texts = {
                "input_ids": batch.input_ids.to(device, non_blocking=True),
                "attention_mask": batch.attention_mask.to(device, non_blocking=True),
            }
            teacher = batch.teacher_target_emb.to(device, non_blocking=True)  # [N, d_t]
            N = teacher.shape[0]
            off = ~torch.eye(N, dtype=torch.bool, device=device)

            ids = batch.target_ids
            dup = torch.zeros(N, N, dtype=torch.bool, device=device)
            for i in range(N):
                for j in range(N):
                    if i != j and ids[i] == ids[j]:
                        dup[i, j] = True

            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                q = backbone.encode_mm(ref, texts)      # [N, d_s]
                c = backbone.encode_image(tgt)           # [N, d_s]

                logits = ((q @ c.T) / contrastive_temp).masked_fill(dup, float("-inf"))
                ce = F.cross_entropy(logits, torch.arange(N, device=device))

                G_s = F.normalize(c @ c.T, p=2, dim=1)
                G_t = F.normalize(teacher @ teacher.T, p=2, dim=1)
                sp = ((G_t - G_s) ** 2).sum() / (N * N)

                l_dist = _rkd_distance(c.float(), teacher.float(), off)
                l_angle = _rkd_angle(c, teacher)

                loss = (
                    lambda_ce * ce + sp_weight * sp
                    + w_dist * l_dist + w_angle * l_angle
                ) / self.grad_accum

            loss.backward()
            total_loss += loss.item() * self.grad_accum
            total_ce += ce.item(); total_sp += sp.item()
            total_d += l_dist.item(); total_a += l_angle.item()

            if step % self.grad_accum == 0 or step == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), self.max_grad_norm)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

            if step % self.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"[Epoch {epoch:02d} | Step {step:05d}/{steps_per_epoch}] "
                    f"joint_loss={loss.item() * self.grad_accum:.4f}  ce={ce.item():.4f}  "
                    f"sp={sp.item():.5f}  dist={l_dist.item():.4f}  angle={l_angle.item():.4f}  lr={lr:.2e}"
                )

        result = {
            "avg_loss": total_loss / steps_per_epoch,
            "avg_contrastive_loss": total_ce / steps_per_epoch,
            "avg_sp": total_sp / steps_per_epoch,
            "avg_rkd_dist": total_d / steps_per_epoch,
            "avg_rkd_angle": total_a / steps_per_epoch,
        }
        logger.info(
            f"[Epoch {epoch:02d}] joint_avg_loss={result['avg_loss']:.4f}  "
            f"ce={result['avg_contrastive_loss']:.4f}  sp={result['avg_sp']:.5f}  "
            f"dist={result['avg_rkd_dist']:.4f}  angle={result['avg_rkd_angle']:.4f}"
        )
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
