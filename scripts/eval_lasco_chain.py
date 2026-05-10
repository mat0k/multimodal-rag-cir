"""
Two-stage chain evaluation on LaSCo val split.

Stage A : VISTA retriever retrieves top-M candidates per query (full gallery).
Stage B : Qwen3-VL-2B re-ranker scores the top-N candidates and re-orders them.
Metric  : Recall@K on LaSCo val (30,037 queries, 39,826 gallery images).

Usage
-----
# Retriever-tuned (contrastive v2 ep1) + zero-shot Qwen, pool-32:
python scripts/eval_lasco_chain.py \\
    --retriever_checkpoint results/training/vista_lasco_finetune_contrastive_v2/checkpoints/vista_epoch01.pth \\
    --top_m 32 \\
    --rerank_top_n 32 \\
    --run_name lasco_retriever_tuned_ranker_not_tuned_pool32

# Zero-shot retriever + zero-shot Qwen (baseline):
python scripts/eval_lasco_chain.py \\
    --zero_shot_retriever \\
    --top_m 32 \\
    --rerank_top_n 32 \\
    --run_name lasco_retriever_not_tuned_ranker_not_tuned_pool32
"""

import argparse
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rerankers.base import RerankCandidate, RerankQuery
from src.rerankers.factory import build_reranker_from_config
from src.retrievers.backbones.vista.modeling import Visualized_BGE
from src.retrievers.vista_retriever import VistaImageProcessor


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
# Datasets (Stage A)
# ---------------------------------------------------------------------------

class _CorpusDataset(Dataset):
    def __init__(self, corpus_path: str, images_dir: Path, image_transform):
        with open(corpus_path) as f:
            self._entries = json.load(f)
        self._images_dir = images_dir
        self._transform = image_transform

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        entry = self._entries[idx]
        rel = entry["path"].removeprefix("val/")
        pil = Image.open(self._images_dir / rel).convert("RGB")
        return {
            "image": self._transform(pil, return_tensors="pt")["pixel_values"][0],
            "image_id": entry["id"],
            "image_path": str(self._images_dir / rel),
        }


class _QueryDataset(Dataset):
    def __init__(self, val_path: str, images_dir: Path, image_transform, caption_transform):
        with open(val_path) as f:
            self._triplets = json.load(f)
        self._images_dir = images_dir
        self._img_tf = image_transform
        self._cap_tf = caption_transform

    def __len__(self):
        return len(self._triplets)

    def __getitem__(self, idx):
        t = self._triplets[idx]
        img_rel = t["query-image"][1]
        pil = Image.open(self._images_dir / img_rel).convert("RGB")
        tok = self._cap_tf(
            t["query-text"],
            padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        )
        return {
            "ref_image": self._img_tf(pil, return_tensors="pt")["pixel_values"][0],
            "input_ids": tok["input_ids"][0],
            "attention_mask": tok["attention_mask"][0],
            "target_image_id": t["target-image"][0],
            "query_image_path": str(self._images_dir / img_rel),
            "query_text": t["query-text"],
        }


# ---------------------------------------------------------------------------
# Stage A — VISTA retrieval
# ---------------------------------------------------------------------------

def _load_retriever(checkpoint: str | None) -> Visualized_BGE:
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
        logger.info(f"Retriever checkpoint loaded: {ckpt_path}")
    else:
        logger.info("Retriever: zero-shot (base VISTA weights).")
    if torch.cuda.is_available():
        backbone = backbone.cuda()
    backbone.eval()
    return backbone


@torch.no_grad()
def _stage_a_retrieve(
    backbone: Visualized_BGE,
    images_dir: Path,
    val_path: str,
    corpus_path: str,
    top_m: int,
    batch_size: int,
    num_workers: int,
) -> tuple[list[dict], list[int], list[str], np.ndarray]:
    """
    Returns:
        corpus_entries  : list of {id, path} dicts (length N)
        target_ids      : list of int ground-truth image IDs (length Q)
        query_img_paths : list of ref image path strings (length Q)
        query_texts     : list of caption strings (length Q)
        top_indices     : np.ndarray [Q, top_m] — indices into corpus_entries
    """
    device = backbone.device
    image_tf = VistaImageProcessor(backbone.preprocess_val)
    tokenize = backbone.tokenizer

    # Gallery
    with open(corpus_path) as f:
        corpus_entries = json.load(f)

    corpus_ds = _CorpusDataset(corpus_path, images_dir, image_tf)
    corpus_loader = DataLoader(
        corpus_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    gallery_feats, gallery_paths = [], []
    for batch in tqdm(corpus_loader, desc="[Stage A] Gallery embeddings"):
        feats = backbone.encode_image(batch["image"].to(device))
        gallery_feats.append(feats.cpu())
        gallery_paths.extend(batch["image_path"])
    gallery_matrix = torch.cat(gallery_feats, dim=0)   # [N, D]
    gallery_ids_np = np.array([e["id"] for e in corpus_entries], dtype=np.int64)

    # Queries
    query_ds = _QueryDataset(val_path, images_dir, image_tf, tokenize)
    query_loader = DataLoader(
        query_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    query_feats, target_ids, query_img_paths, query_texts = [], [], [], []
    for batch in tqdm(query_loader, desc="[Stage A] Query embeddings"):
        texts = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        feats = backbone.encode_mm(batch["ref_image"].to(device), texts)
        query_feats.append(feats.cpu())
        target_ids.extend(batch["target_image_id"].tolist())
        query_img_paths.extend(batch["query_image_path"])
        query_texts.extend(batch["query_text"])
    query_matrix = torch.cat(query_feats, dim=0)   # [Q, D]

    # Chunked similarity → top-M indices
    chunk = 256
    gallery_gpu = gallery_matrix.to(device)
    top_indices = np.empty((len(query_matrix), top_m), dtype=np.int64)

    for start in tqdm(range(0, len(query_matrix), chunk), desc="[Stage A] Top-M retrieval"):
        q_chunk = query_matrix[start: start + chunk].to(device)
        sims = torch.matmul(q_chunk, gallery_gpu.T)
        top_idx = torch.argsort(sims, dim=1, descending=True)[:, :top_m].cpu().numpy()
        top_indices[start: start + len(q_chunk)] = top_idx

    return corpus_entries, target_ids, query_img_paths, query_texts, gallery_ids_np, gallery_paths, top_indices


# ---------------------------------------------------------------------------
# Stage B — Qwen re-ranking
# ---------------------------------------------------------------------------

def _stage_b_rerank(
    reranker,
    corpus_entries: list[dict],
    gallery_paths: list[str],
    target_ids: list[int],
    gallery_ids_np: np.ndarray,
    query_img_paths: list[str],
    query_texts: list[str],
    top_indices: np.ndarray,
    rerank_top_n: int,
    k_values: list[int],
) -> dict[str, float]:
    Q = len(target_ids)
    hits = {k: 0 for k in k_values}

    for qi in tqdm(range(Q), desc="[Stage B] Qwen re-ranking"):
        cand_indices = top_indices[qi, :rerank_top_n]   # [N] corpus indices
        cand_ids = gallery_ids_np[cand_indices]
        cand_paths = [gallery_paths[ci] for ci in cand_indices]

        query = RerankQuery(
            reference_image=query_img_paths[qi],
            text_edit=query_texts[qi],
        )
        candidates = [
            RerankCandidate(image=p, candidate_id=str(cand_ids[i]))
            for i, p in enumerate(cand_paths)
        ]

        scores = reranker.score_batch(query, candidates)
        ranked_order = np.argsort(scores)[::-1]
        ranked_ids = cand_ids[ranked_order]

        tgt = target_ids[qi]
        for k in k_values:
            if tgt in ranked_ids[:k]:
                hits[k] += 1

    return {f"recall_at{k}": round(100.0 * hits[k] / Q, 4) for k in k_values}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    output_dir = PROJECT_ROOT / "results" / "lasco_chain" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir)

    retriever_ckpt = None if args.zero_shot_retriever else args.retriever_checkpoint
    k_values = sorted(args.k)
    images_dir = PROJECT_ROOT / "data/lasco/images"
    val_path = str(PROJECT_ROOT / "data/lasco/lasco_val.json")
    corpus_path = str(PROJECT_ROOT / "data/lasco/lasco_val_corpus.json")

    logger.info("=" * 60)
    logger.info(f"Run name         : {args.run_name}")
    logger.info(f"Retriever        : {retriever_ckpt or 'zero_shot'}")
    logger.info(f"Reranker config  : {args.reranker_config}")
    logger.info(f"top_m / rerank_n : {args.top_m} / {args.rerank_top_n}")
    logger.info(f"K values         : {k_values}")
    logger.info(f"Output dir       : {output_dir}")
    logger.info("=" * 60)

    # Stage A
    t0 = time.time()
    backbone = _load_retriever(retriever_ckpt)
    corpus_entries, target_ids, query_img_paths, query_texts, gallery_ids_np, gallery_paths, top_indices = \
        _stage_a_retrieve(
            backbone, images_dir, val_path, corpus_path,
            top_m=args.top_m, batch_size=args.batch_size, num_workers=args.num_workers,
        )
    elapsed_a = time.time() - t0
    logger.info(f"Stage A done in {elapsed_a/60:.1f} min")

    # Free retriever GPU memory before loading Qwen
    del backbone
    torch.cuda.empty_cache()

    # Stage B
    t1 = time.time()
    reranker = build_reranker_from_config(args.reranker_config)
    reranked_metrics = _stage_b_rerank(
        reranker=reranker,
        corpus_entries=corpus_entries,
        gallery_paths=gallery_paths,
        target_ids=target_ids,
        gallery_ids_np=gallery_ids_np,
        query_img_paths=query_img_paths,
        query_texts=query_texts,
        top_indices=top_indices,
        rerank_top_n=args.rerank_top_n,
        k_values=k_values,
    )
    elapsed_b = time.time() - t1
    logger.info(f"Stage B done in {elapsed_b/60:.1f} min")

    # Log results
    logger.info("-" * 40)
    logger.info(f"LaSCo chain results — {args.run_name}")
    for k in k_values:
        logger.info(f"  Recall@{k:<4d}: {reranked_metrics[f'recall_at{k}']:.2f}%")
    logger.info(f"  Stage A elapsed : {elapsed_a/60:.1f} min")
    logger.info(f"  Stage B elapsed : {elapsed_b/60:.1f} min")
    logger.info("-" * 40)

    # Save
    cuda_available = torch.cuda.is_available()
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "dataset": "lasco_val",
        "pipeline": {
            "stage_a_retriever": retriever_ckpt or "zero_shot (base VISTA weights)",
            "stage_b_reranker": args.reranker_config,
            "top_m": args.top_m,
            "rerank_top_n": args.rerank_top_n,
            "k_values": k_values,
            "batch_size_retriever": args.batch_size,
            "num_workers": args.num_workers,
        },
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "stage_a_elapsed_seconds": round(elapsed_a, 1),
            "stage_b_elapsed_seconds": round(elapsed_b, 1),
            "total_elapsed_seconds": round(elapsed_a + elapsed_b, 1),
        },
        "metrics": reranked_metrics,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved → {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stage chain eval on LaSCo val split.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zero_shot_retriever", action="store_true",
                       help="Use base VISTA weights (no fine-tuning) for Stage A.")
    group.add_argument("--retriever_checkpoint", type=str,
                       help="Path to fine-tuned VISTA .pth checkpoint for Stage A.")
    parser.add_argument("--reranker_config", type=str,
                        default="configs/reranker/qwen3vl_reranker_2b.yaml",
                        help="Path to Qwen reranker config.")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Output folder name under results/lasco_chain/.")
    parser.add_argument("--top_m", type=int, default=32,
                        help="Number of candidates retrieved by Stage A.")
    parser.add_argument("--rerank_top_n", type=int, default=32,
                        help="Number of Stage A candidates to re-rank in Stage B.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for VISTA encoding.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 50, 100],
                        help="K values for Recall@K.")
    main(parser.parse_args())
