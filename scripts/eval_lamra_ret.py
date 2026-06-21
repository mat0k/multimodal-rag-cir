"""
Zero-shot evaluation of the LamRA-Ret retriever on FashionIQ and CIRR (each
alone), to decide whether it is a strong enough TEACHER for retriever ->
retriever distillation versus the VISTA student.

It REUSES the existing dataset readers and metric functions:
  - src/datasets/{fashioniq,cirr}.py        (raw PIL + raw caption, no transforms)
  - src/evaluation/fashioniq_eval.compute_fashioniq_metrics
  - src/evaluation/cirr_eval.compute_cirr_metrics
and only swaps in LamRA-native encoding (src/retrievers/lamra_ret_retriever.py).

Nothing in the VISTA path is touched. Results are written as JSON to
results/lamra_ret_zeroshot/.

Usage (full run is GPU-heavy -> submit via jobs/retriever/eval_lamra_ret_a100.sbatch):
  python scripts/eval_lamra_ret.py --dataset both --batch-size 8
  python scripts/eval_lamra_ret.py --dataset cirr --limit-queries 50 --limit-index 500   # smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.datasets.cirr import build_cirr_dataset
from src.datasets.fashioniq import build_fashioniq_dataset
from src.evaluation.cirr_eval import compute_cirr_metrics
from src.evaluation.fashioniq_eval import compute_fashioniq_metrics
from src.retrievers.lamra_ret_retriever import LamRARetRetriever


def _save(metrics: dict, meta: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"metrics": metrics, "meta": meta}, indent=2))
    print(f"[saved] {out_path}")


def _chunks(n: int, size: int):
    for start in range(0, n, size):
        yield start, min(start + size, n)


@torch.no_grad()
def _encode_index(retriever, ds, batch_size, limit, desc):
    """Encode a gallery (mode='images') -> (features[N,D], names, classes)."""
    n = len(ds) if limit is None else min(limit, len(ds))
    feats, names, classes = [], [], []
    for start, end in tqdm(list(_chunks(n, batch_size)), desc=desc):
        imgs, bnames, bcls = [], [], []
        for i in range(start, end):
            s = ds[i]
            imgs.append(s["image"])
            bnames.append(s["image_name"])
            bcls.append(s.get("class"))
        feats.append(retriever.encode_images(imgs).cpu())
        names.extend(bnames)
        classes.extend(bcls)
    return torch.cat(feats, dim=0), names, classes


# ---------------------------------------------------------------------------
# FashionIQ
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_fashioniq(retriever, batch_size, limit_q, limit_i):
    index_ds = build_fashioniq_dataset(
        split="val", mode="images", image_transform=None, caption_transform=None
    )
    triplet_ds = build_fashioniq_dataset(
        split="val", mode="triplets", image_transform=None, caption_transform=None,
        caption_joiner=" ",
    )

    index_features, index_names, index_classes = _encode_index(
        retriever, index_ds, batch_size, limit_i, "FashionIQ gallery"
    )

    n = len(triplet_ds) if limit_q is None else min(limit_q, len(triplet_ds))
    q_feats, ref_names, tgt_names, classes = [], [], [], []
    for start, end in tqdm(list(_chunks(n, batch_size)), desc="FashionIQ queries"):
        imgs, caps = [], []
        for i in range(start, end):
            s = triplet_ds[i]
            imgs.append(s["candidate"])
            caps.append(s["transformed_caption"])  # raw joined caption (no tokenizer)
            ref_names.append(s["candidate_name"])
            tgt_names.append(s["target_name"])
            classes.append(s["class"])
        q_feats.append(retriever.encode_composed(imgs, caps).cpu())
    predicted_features = torch.cat(q_feats, dim=0)

    return compute_fashioniq_metrics(
        index_features=index_features,
        index_names=index_names,
        index_classes=index_classes,
        predicted_features=predicted_features,
        reference_names=ref_names,
        target_names=tgt_names,
        triplet_classes=classes,
        k_values=[5, 10, 50],
        metric_prefix="val",
    )


# ---------------------------------------------------------------------------
# CIRR
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_cirr(retriever, batch_size, limit_q, limit_i):
    index_ds = build_cirr_dataset(
        split="val", mode="images", image_transform=None, caption_transform=None
    )
    triplet_ds = build_cirr_dataset(
        split="val", mode="triplets", image_transform=None, caption_transform=None
    )

    index_features, index_names, _ = _encode_index(
        retriever, index_ds, batch_size, limit_i, "CIRR gallery"
    )

    n = len(triplet_ds) if limit_q is None else min(limit_q, len(triplet_ds))
    q_feats, ref_names, tgt_names, group_members, pair_ids = [], [], [], [], []
    for start, end in tqdm(list(_chunks(n, batch_size)), desc="CIRR queries"):
        imgs, caps = [], []
        for i in range(start, end):
            s = triplet_ds[i]
            imgs.append(s["reference"])
            caps.append(s["transformed_caption"])  # raw caption (no tokenizer)
            ref_names.append(s["reference_name"])
            tgt_names.append(s["target_name"])
            group_members.append(list(s["group_members"]))
            pair_ids.append(s["pair_id"])
        q_feats.append(retriever.encode_composed(imgs, caps).cpu())
    predicted_features = torch.cat(q_feats, dim=0)

    # Subset metrics require the full gallery; skip them only when galleries are
    # truncated for a smoke test.
    skip_subset = limit_i is not None
    raw = compute_cirr_metrics(
        index_features=index_features,
        index_names=index_names,
        predicted_features=predicted_features,
        reference_names=ref_names,
        target_names=tgt_names,
        group_members=group_members,
        pair_ids=pair_ids,
        k_values=[1, 5, 10, 50],
        k_values_subset=[1, 2, 3],
        skip_subset_metrics=skip_subset,
        return_type="metrics",
    )

    metrics = {f"val_global_recall_at{k}": float(raw[f"recall_at{k}"]) for k in [1, 5, 10, 50]}
    if not skip_subset:
        for k in [1, 2, 3]:
            metrics[f"val_subset_recall_at{k}"] = float(raw[f"subset_recall_at{k}"])
        metrics["val_summary_average"] = float(
            np.mean([metrics["val_global_recall_at5"], metrics["val_subset_recall_at1"]])
        )
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Zero-shot LamRA-Ret eval on FashionIQ / CIRR.")
    ap.add_argument("--model-path", default="models/LamRA-Ret-Qwen2.5VL-7b")
    ap.add_argument("--dataset", choices=["fashioniq", "cirr", "both"], default="both")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--instruction", default=None, help="Override the composed-CIR instruction.")
    ap.add_argument("--output-dir", default="results/lamra_ret_zeroshot")
    ap.add_argument("--limit-queries", type=int, default=None, help="Smoke test: cap #queries.")
    ap.add_argument("--limit-index", type=int, default=None, help="Smoke test: cap #gallery imgs.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    kwargs = dict(
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation=args.attn,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    if args.instruction is not None:
        kwargs["cir_instruction"] = args.instruction

    print(f"Loading LamRA-Ret from {args.model_path} ...")
    retriever = LamRARetRetriever(**kwargs)
    print(f"Loaded. <emb> id={retriever.emb_token_id}  instruction={retriever.cir_instruction!r}")

    meta_base = {
        "model_path": args.model_path,
        "instruction": retriever.cir_instruction,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "batch_size": args.batch_size,
        "attn": args.attn,
        "limit_queries": args.limit_queries,
        "limit_index": args.limit_index,
    }

    if args.dataset in ("fashioniq", "both"):
        t0 = time.time()
        m = eval_fashioniq(retriever, args.batch_size, args.limit_queries, args.limit_index)
        print("FashionIQ:", json.dumps(m, indent=2))
        _save(m, {**meta_base, "dataset": "fashioniq", "elapsed_s": round(time.time() - t0, 1)},
              out_dir / "fashioniq.json")

    if args.dataset in ("cirr", "both"):
        t0 = time.time()
        m = eval_cirr(retriever, args.batch_size, args.limit_queries, args.limit_index)
        print("CIRR:", json.dumps(m, indent=2))
        _save(m, {**meta_base, "dataset": "cirr", "elapsed_s": round(time.time() - t0, 1)},
              out_dir / "cirr.json")


if __name__ == "__main__":
    main()
