"""
Pre-compute BGE-VL-MLLM-S1 (CLEAN teacher) artifacts over ONE benchmark training
split -- FashionIQ or CIRR -- for SUPERVISED (in-domain) distillation.

Why this exists: data/benchmarks_distill/ currently holds Qwen3-VL scores only, and
Qwen is trained on M-BEIR, which contains CIRR and FashionIQ. Any distillation result
from it is contaminated. This is the benchmark counterpart of the two LaSCo scripts
(precompute_bge_vl_cand_emb.py + precompute_bge_vl_scores.py), merged into one pass
because the query embeddings are needed by feature-based KD anyway.

Writes four aligned files per dataset (row i of every tensor == entry i of the json):

  triplets_<ds>.json                 filtered triplets, uid == list index
  bge_vl_cand_emb_<ds>.pt   [N,4096] target-image embeddings   (SP / RKD / CRD / feature)
  bge_vl_query_emb_<ds>.pt  [N,4096] composed-query embeddings (feature)
  teacher_scores_bge_vl_<ds>.json    pos + K negative scores   (CE+MSE)

Triplets whose images are missing on disk are dropped BEFORE encoding, so the row
alignment holds and the arms train on exactly the set the supervised contrastive
baseline used. `neg_indices` index into triplets_<ds>.json (what BenchmarkDistill
expects), NOT into the original combined triplets file.

Teacher = BGE-VL-MLLM-S1 (MegaPairs only). Refuses S2 (MMEB fine-tuned).

Usage:
  python scripts/precompute_bge_vl_benchmarks.py --dataset fashioniq --batch-size 8
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

SOURCES = {
    "fashioniq": ("fashioniq_dress", "fashioniq_shirt", "fashioniq_toptee"),
    "cirr": ("cirr",),
}


def _image_root(dataset: str, args) -> Path:
    return PROJECT_ROOT / (args.fashioniq_images_dir if dataset == "fashioniq"
                           else args.cirr_images_dir)


def _select_triplets(dataset: str, args) -> list[dict]:
    """Filter the combined triplets file to one dataset, dropping missing images."""
    allt = json.load(open(PROJECT_ROOT / args.triplets_path))
    wanted = set(SOURCES[dataset])
    root = _image_root(dataset, args)

    kept, dropped = [], 0
    for t in allt:
        if t["source"] not in wanted:
            continue
        if not ((root / t["ref_path"]).is_file() and (root / t["pos_path"]).is_file()):
            dropped += 1
            continue
        # uid is re-indexed to the position in THIS list: the datasets and the
        # embedding rows all address entries by that index.
        kept.append({**t, "orig_uid": t["uid"], "uid": len(kept)})

    print(f"[{dataset}] {len(kept):,} triplets kept, {dropped:,} dropped (missing images)",
          flush=True)
    return kept


@torch.no_grad()
def _encode(retriever, triplets, root, batch_size, cache: Path, kind: str) -> torch.Tensor:
    """Encode targets ('cand') or composed queries ('query'); cached per file."""
    if cache.exists():
        emb = torch.load(cache, map_location="cpu")
        if emb.shape[0] != len(triplets):
            raise SystemExit(
                f"{cache} has {emb.shape[0]} rows but {len(triplets)} triplets were "
                f"selected — stale cache, delete it or pass --overwrite.")
        print(f"[{kind}] loaded cache {cache.name} {tuple(emb.shape)}", flush=True)
        return emb

    n, embs, t0 = len(triplets), [], time.time()
    for s in range(0, n, batch_size):
        chunk = triplets[s : s + batch_size]
        if kind == "cand":
            imgs = [Image.open(root / t["pos_path"]).convert("RGB") for t in chunk]
            out = retriever.encode_images(imgs)
        else:
            imgs = [Image.open(root / t["ref_path"]).convert("RGB") for t in chunk]
            out = retriever.encode_composed(imgs, [t["caption"] for t in chunk])
        embs.append(out.cpu())
        done = min(s + batch_size, n)
        if (s // batch_size) % 50 == 0 or done == n:
            rate = (time.time() - t0) / max(done, 1)
            print(f"[{kind}] {done:,}/{n:,} | {1/rate:.1f} it/s | "
                  f"ETA {(n-done)*rate/3600:.2f}h", flush=True)

    emb = torch.cat(embs, 0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, cache)
    print(f"[{kind}] saved {tuple(emb.shape)} -> {cache}", flush=True)
    return emb


def _sample_negatives(i: int, k: int, total: int, rng) -> list[int]:
    """k random negatives != i, drawn within this dataset only."""
    sample = rng.choice(total, size=k + 2, replace=False)
    return [int(j) for j in sample if j != i][:k]


def main(a):
    if "S2" in a.model_path:
        raise SystemExit("REFUSING: BGE-VL-MLLM-S2 is MMEB-fine-tuned (contaminated). Use S1.")

    ds = a.dataset
    out_dir = PROJECT_ROOT / a.out_dir
    triplets_out = out_dir / f"triplets_{ds}.json"
    cand_out = out_dir / f"bge_vl_cand_emb_{ds}.pt"
    query_out = out_dir / f"bge_vl_query_emb_{ds}.pt"
    scores_out = out_dir / f"teacher_scores_bge_vl_{ds}.json"

    if scores_out.exists() and not a.overwrite:
        raise SystemExit(f"{scores_out} exists; pass --overwrite to redo.")
    if a.overwrite:
        for p in (cand_out, query_out):
            p.unlink(missing_ok=True)

    triplets = _select_triplets(ds, a)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(triplets, open(triplets_out, "w"))
    print(f"wrote {triplets_out}", flush=True)

    root = _image_root(ds, a)
    retriever = BGEVLRetriever(
        model_path=str(PROJECT_ROOT / a.model_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
    )

    norm = torch.nn.functional.normalize
    C = norm(_encode(retriever, triplets, root, a.batch_size, cand_out, "cand").float(), dim=-1)
    Q = norm(_encode(retriever, triplets, root, a.batch_size, query_out, "query").float(), dim=-1)

    print("Scoring …", flush=True)
    total = len(triplets)
    rng = np.random.default_rng(a.seed + 1)
    pos_all = (Q * C).sum(-1)
    scores = {}
    for i in range(total):
        nidx = _sample_negatives(i, a.num_negatives, total, rng)
        ns = (Q[i : i + 1] @ C[nidx].T).squeeze(0).tolist()
        scores[str(i)] = {
            "pos_score": float(pos_all[i]),
            "neg_indices": nidx,                       # positions in triplets_<ds>.json
            "neg_scores": [float(x) for x in ns],
        }
    json.dump(scores, open(scores_out, "w"))

    neg_mean = float(np.mean([np.mean(s["neg_scores"]) for s in scores.values()]))
    print(f"Done. {len(scores):,} scored -> {scores_out}")
    print(f"pos mean={float(pos_all.mean()):.4f}  neg mean={neg_mean:.4f}  "
          f"avg margin={float(pos_all.mean()) - neg_mean:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(SOURCES), required=True)
    p.add_argument("--model-path", default="models/BGE-VL-MLLM-S1")
    p.add_argument("--triplets_path", default="data/benchmarks_distill/triplets.json")
    p.add_argument("--fashioniq_images_dir", default="data/fashioniq/images")
    p.add_argument("--cirr_images_dir", default="data/cirr/images")
    p.add_argument("--out_dir", default="data/benchmarks_distill")
    p.add_argument("--num_negatives", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    main(p.parse_args())
