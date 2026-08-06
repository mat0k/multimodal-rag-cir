"""
Zero-shot evaluation of BGE-VL-MLLM-S1 on FashionIQ + CIRR — the gating check
for whether it is a strong enough CLEAN teacher (MegaPairs-only, never trained
on FashionIQ/CIRR) for retriever -> retriever distillation.

Reuses the SAME eval functions as the LamRA-Ret gate (scripts/eval_lamra_ret.py),
so the protocol is byte-identical and the numbers are directly comparable to:
    LamRA-Ret zero-shot : FIQ 53.07 / CIRR 77.79
    VISTA student       : FIQ 25.71 / CIRR 53.97   <- the bar to clear

Output: results/bge_vl/bge_vl_zeroshot/zeroshot_fiq_cirr.json

Usage:
  python scripts/eval_bge_vl.py --dataset both --batch-size 4
  python scripts/eval_bge_vl.py --dataset cirr --limit-queries 50 --limit-index 500   # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the identical eval protocol used for the LamRA-Ret gate.
from scripts.eval_lamra_ret import _save, eval_cirr, eval_fashioniq  # noqa: E402
from src.retrievers.bge_vl_retriever import BGEVLRetriever  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Zero-shot BGE-VL-MLLM-S1 eval on FashionIQ / CIRR.")
    ap.add_argument("--model-path", default="models/BGE-VL-MLLM-S1")
    ap.add_argument("--dataset", choices=["fashioniq", "cirr", "both"], default="both")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--instruction", default=None, help="Override the composed-CIR instruction.")
    ap.add_argument("--output-dir", default="results/bge_vl/bge_vl_zeroshot")
    ap.add_argument("--limit-queries", type=int, default=None)
    ap.add_argument("--limit-index", type=int, default=None)
    args = ap.parse_args()

    # Guard: this phase requires the CLEAN S1 model, never S2 (MMEB fine-tuned).
    if "S2" in args.model_path:
        raise SystemExit(
            "REFUSING: BGE-VL-MLLM-S2 is fine-tuned on the MMEB training set "
            "(which contains CIR benchmarks) and invalidates the unsupervised claim. Use S1."
        )

    kwargs = dict(
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16 if args.dtype == "float16" else torch.bfloat16,
    )
    if args.instruction is not None:
        kwargs["cir_instruction"] = args.instruction

    print(f"Loading BGE-VL from {args.model_path} ...")
    retriever = BGEVLRetriever(**kwargs)
    print(f"Loaded. instruction={retriever.cir_instruction!r}")

    meta = {
        "model_path": args.model_path,
        "instruction": retriever.cir_instruction,
        "image_size": retriever.image_size,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "limit_queries": args.limit_queries,
        "limit_index": args.limit_index,
        "teacher_training_data": "MegaPairs only (S1) — NOT trained on FashionIQ/CIRR",
        "datasets": [],
    }
    results: dict[str, dict] = {}

    if args.dataset in ("fashioniq", "both"):
        t0 = time.time()
        m = eval_fashioniq(retriever, args.batch_size, args.limit_queries, args.limit_index)
        print("FashionIQ:", json.dumps(m, indent=2))
        results["fashioniq"] = {"metrics": m, "elapsed_s": round(time.time() - t0, 1)}
        meta["datasets"].append("fashioniq")

    if args.dataset in ("cirr", "both"):
        t0 = time.time()
        m = eval_cirr(retriever, args.batch_size, args.limit_queries, args.limit_index)
        print("CIRR:", json.dumps(m, indent=2))
        results["cirr"] = {"metrics": m, "elapsed_s": round(time.time() - t0, 1)}
        meta["datasets"].append("cirr")

    _save(results, meta, Path(args.output_dir) / "zeroshot_fiq_cirr.json")


if __name__ == "__main__":
    main()
