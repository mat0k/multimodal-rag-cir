# Retriever Training — Experiment Log

## Setting 1 — Contrastive Fine-tuning on LaSCo

**Idea:** Fine-tune VISTA on automatically generated CIR triplets (LaSCo) using InfoNCE loss with in-batch negatives. No human-labeled CIR data used (unsupervised).

**Config:** `configs/training/lasco_finetune.yaml`
**Script:** `scripts/train_retriever.py`
**Job:** `jobs/training/train_retriever_a100.sbatch`

| # | Run name | Epochs | Batch (eff.) | LR | Temp | FashionIQ R@10 | CIRR Summary | Notes |
|---|----------|--------|--------------|----|------|----------------|--------------|-------|
| — | baseline (pre-trained VISTA, no fine-tuning) | — | — | — | — | TBD (need vista_mm full-gallery run) | TBD | zero-shot reference |
| 1 | vista_lasco_finetune_contrastive | 10 | 128 | 2e-5 | 0.02 | 9.15 (ep10) / **11.12 (ep1-best)** | 35.09 (ep10) / **38.65 (ep1-best)** | bf16, A100 20GB MIG; full model trained → catastrophic forgetting; best epoch not saved (bug fixed) |
| 2 | vista_lasco_finetune_contrastive_v2 | 5 | 128 | **1e-6** | 0.02 | — | — | **partial freeze**: only top 3 BGE layers + visual_proj trained (~10% params); fixes forgetting |

---

## Setting 2 — Distillation from Qwen3-VL-2B

**Idea:** Use Qwen3-VL-2B reranker as teacher. Pre-compute soft relevance scores over LaSCo, then train VISTA to match them (KL divergence loss). Produces a fast retriever that approximates the reranker's ranking quality.

**Status:** Not yet implemented. Placeholder in trainer (`training_mode: distillation`).

| # | Run name | Epochs | Batch (eff.) | LR | α | Temp KD | FashionIQ R@10 | CIRR R@5 | Notes |
|---|----------|--------|--------------|----|---|---------|----------------|----------|-------|
| — | — | — | — | — | — | — | — | — | — |

---

## Notes

- All evaluations: FashionIQ **val** split + CIRR **val** split (no test leakage during training)
- Final model tested on FashionIQ val + CIRR test for thesis results
- Checkpoints saved to `results/training/<run_name>/checkpoints/`
- Full log (loss per step, eval per epoch) in `results/training/<run_name>/train_log.json`
