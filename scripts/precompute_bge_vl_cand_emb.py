"""
Pre-compute BGE-VL-MLLM-S1 (CLEAN teacher) candidate embeddings over the 40K
LaSCo subset target images, aligned to data/lasco/distill_subset.json order.

This is all the relation-based methods (SP / RKD / CRD) need — they use only the
candidate (target) embeddings; the CE anchor uses ground-truth labels, not the
teacher. Output is a drop-in replacement for data/lasco/lamra_ret_cand_emb.pt.

Teacher = BGE-VL-MLLM-S1 (MegaPairs only, NOT trained on FashionIQ/CIRR).
Refuses S2 (MMEB fine-tuned).

Output: data/lasco/bge_vl_cand_emb.pt   [N, 4096] L2-normalized

Usage:
  python scripts/precompute_bge_vl_cand_emb.py --batch-size 8
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrievers.bge_vl_retriever import BGEVLRetriever  # noqa: E402


def main(a):
    if "S2" in a.model_path:
        raise SystemExit("REFUSING: BGE-VL-MLLM-S2 is MMEB-fine-tuned (contaminated). Use S1.")

    subset = json.load(open(PROJECT_ROOT / a.subset_path))
    images_dir = PROJECT_ROOT / a.images_dir
    out = PROJECT_ROOT / a.out
    if out.exists() and not a.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to redo.")

    retriever = BGEVLRetriever(
        model_path=str(PROJECT_ROOT / a.model_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
    )

    embs = []
    n = len(subset)
    t0 = time.time()
    for s in range(0, n, a.batch_size):
        chunk = subset[s : s + a.batch_size]
        imgs = [Image.open(images_dir / t["target-image"][1]).convert("RGB") for t in chunk]
        embs.append(retriever.encode_images(imgs).cpu())
        done = min(s + a.batch_size, n)
        if (s // a.batch_size) % 50 == 0 or done == n:
            rate = (time.time() - t0) / max(done, 1)
            print(f"  {done:,}/{n:,} encoded | {1/rate:.1f} img/s | ETA {(n-done)*rate/3600:.2f}h", flush=True)

    emb = torch.nn.functional.normalize(torch.cat(embs, 0), dim=-1)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, out)
    print(f"saved {tuple(emb.shape)} -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="models/BGE-VL-MLLM-S1")
    p.add_argument("--subset_path", default="data/lasco/distill_subset.json")
    p.add_argument("--images_dir", default="data/lasco/images")
    p.add_argument("--out", default="data/lasco/bge_vl_cand_emb.pt")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    main(p.parse_args())
