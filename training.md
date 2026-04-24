# Retriever Training — Experiment Log

## Setting 1 — Contrastive Fine-tuning on LaSCo

**Idea:** Fine-tune VISTA on automatically generated CIR triplets (LaSCo) using InfoNCE loss with in-batch negatives. No human-labeled CIR data used (unsupervised).

**Config:** `configs/training/lasco_finetune.yaml`
**Script:** `scripts/train_retriever.py`
**Job:** `jobs/training/train_retriever_a100.sbatch`

| # | Run name | Epochs | Batch (eff.) | LR | Temp | FashionIQ R@10 | CIRR R@5 | Notes |
|---|----------|--------|--------------|----|------|----------------|----------|-------|
| — | baseline (pre-trained VISTA, no fine-tuning) | — | — | — | — | — | — | zero-shot reference |

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
