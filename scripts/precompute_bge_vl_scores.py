"""
Pre-compute BGE-VL-MLLM-S1 response-based teacher SCORES for the 40K LaSCo subset
(for MSE / CE+MSE distillation). Reuses the already-cached candidate embeddings
(data/lasco/bge_vl_cand_emb.pt) and only encodes the composed QUERIES, then scores
by cosine (dot product of L2-normalized embeddings).

Random negatives, K=15, seed+1 — IDENTICAL scheme to the LamRA-Ret random-neg
response run, so the two teachers are directly comparable.

Teacher = BGE-VL-MLLM-S1 (MegaPairs only). Refuses S2.

Output (drop-in for src/datasets/lasco_distill.py):
  data/lasco/teacher_scores_bge_vl.json
    {"<qid>": {"pos_score": float, "neg_qids": [int...], "neg_scores": [float...]}}
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrievers.bge_vl_retriever import BGEVLRetriever  # noqa: E402


def _sample_negatives(i, n, total, rng):
    sample = rng.choice(total, size=n + 2, replace=False)
    return [int(j) for j in sample if j != i][:n]


@torch.no_grad()
def _encode_queries(retriever, subset, images_dir, bs, cache):
    if cache.exists():
        print(f"[query] loading cache {cache}", flush=True)
        return torch.load(cache, map_location="cpu")
    n = len(subset)
    embs, t0 = [], time.time()
    for s in range(0, n, bs):
        chunk = subset[s : s + bs]
        imgs = [Image.open(images_dir / t["query-image"][1]).convert("RGB") for t in chunk]
        caps = [t["query-text"] for t in chunk]
        embs.append(retriever.encode_composed(imgs, caps).cpu())
        done = min(s + bs, n)
        if (s // bs) % 50 == 0 or done == n:
            rate = (time.time() - t0) / max(done, 1)
            print(f"[query] {done:,}/{n:,} | {1/rate:.1f} it/s | ETA {(n-done)*rate/3600:.2f}h", flush=True)
    out = torch.cat(embs, 0)
    torch.save(out, cache)
    return out


def main(a):
    if "S2" in a.model_path:
        raise SystemExit("REFUSING: BGE-VL-MLLM-S2 is MMEB-fine-tuned (contaminated). Use S1.")

    subset = json.load(open(PROJECT_ROOT / a.subset_path))
    images_dir = PROJECT_ROOT / a.images_dir
    C = torch.nn.functional.normalize(
        torch.load(PROJECT_ROOT / a.cand_emb, map_location="cpu").float(), dim=-1
    )
    assert C.shape[0] == len(subset), f"cand {C.shape[0]} != subset {len(subset)}"

    total = len(subset)
    rng = np.random.default_rng(a.seed + 1)
    neg_idx = [_sample_negatives(i, a.num_negatives, total, rng) for i in range(total)]

    retriever = BGEVLRetriever(
        model_path=str(PROJECT_ROOT / a.model_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
    )
    Q = torch.nn.functional.normalize(
        _encode_queries(retriever, subset, images_dir, a.batch_size,
                        PROJECT_ROOT / a.query_emb).float(), dim=-1)

    print("Scoring …", flush=True)
    pos_all = (Q * C).sum(-1)
    scores = {}
    for i, t in enumerate(subset):
        nidx = neg_idx[i]
        ns = (Q[i:i+1] @ C[nidx].T).squeeze(0).tolist()
        scores[str(t["qid"])] = {
            "pos_score": float(pos_all[i]),
            "neg_qids": [int(subset[j]["qid"]) for j in nidx],
            "neg_scores": [float(x) for x in ns],
        }
    out = PROJECT_ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(scores, open(out, "w"))
    margins = [s["pos_score"] - float(np.mean(s["neg_scores"])) for s in scores.values()]
    print(f"Done. {len(scores):,} scored -> {out}")
    print(f"pos mean={float(pos_all.mean()):.4f}  neg mean={float(np.mean([np.mean(s['neg_scores']) for s in scores.values()])):.4f}  avg margin={float(np.mean(margins)):.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="models/BGE-VL-MLLM-S1")
    p.add_argument("--subset_path", default="data/lasco/distill_subset.json")
    p.add_argument("--images_dir", default="data/lasco/images")
    p.add_argument("--cand_emb", default="data/lasco/bge_vl_cand_emb.pt")
    p.add_argument("--query_emb", default="data/lasco/bge_vl_query_emb.pt")
    p.add_argument("--out", default="data/lasco/teacher_scores_bge_vl.json")
    p.add_argument("--num_negatives", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    main(p.parse_args())
