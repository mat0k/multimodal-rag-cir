"""Generate CIRCO **test** submission files for the official evaluation server.

CIRCO withholds test ground truth (`test.json` has no `gt_img_ids`), so test-split
numbers can only be obtained by submitting ranked predictions to

    https://circo.micc.unifi.it/

Reuses the exact val protocol from src/evaluation/circo_eval.py -- same encoders, same
L2-normalised cosine ranking, same reference-image removal -- switching only the split
and asking for ranked ids instead of metrics.

Submission format: JSON mapping each query id to its top-50 ranked image ids.

    python scripts/generate_circo_test_submission.py --only magiclens_sp_ce0.1_ep1

Writes the file; submitting is a manual step. Like CIRR, the server limits test
submissions, so submit final models only -- do not iterate against returned scores.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.circo import build_circo_dataset  # noqa: E402
from src.evaluation.circo_eval import (  # noqa: E402
    generate_circo_index_features,
    generate_circo_query_features,
)
from scripts.run_circo_eval import MODELS, build_model  # noqa: E402

TOP_K = 50


def generate_for_model(entry: dict, args, device: str) -> dict:
    print(f"\n{'='*70}\n{entry['name']}\n  {entry['checkpoint']}\n{'='*70}", flush=True)
    model = build_model(entry, device)

    index_ds = build_circo_dataset(split="test", mode="images", dataset_path=args.dataset_path,
                                   image_transform=model.image_processor,
                                   caption_transform=model.tokenizer)
    query_ds = build_circo_dataset(split="test", mode="triplets", dataset_path=args.dataset_path,
                                   image_transform=model.image_processor,
                                   caption_transform=model.tokenizer)
    print(f"  gallery={len(index_ds):,} images | queries={len(query_ds):,}", flush=True)

    # The gallery is split-independent (both splits index the same COCO images), so a
    # val cache is reused when present.
    t0 = time.time()
    cache = PROJECT_ROOT / args.cache_dir / f"{entry['short']}_index.pt"
    if cache.exists() and not args.no_cache:
        print(f"  loading cached gallery: {cache.name}", flush=True)
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        index_features, index_ids = payload["features"], payload["ids"]
    else:
        index_features, index_ids = generate_circo_index_features(
            model, index_ds, args.batch_size, args.num_workers, args.tqdm)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": index_features, "ids": index_ids}, cache)
    print(f"  gallery ready ({time.time()-t0:.0f}s)", flush=True)

    query_features, query_ids, ref_ids, _ = generate_circo_query_features(
        model, query_ds, args.batch_size, args.num_workers, args.tqdm)

    id_to_pos = {img_id: i for i, img_id in enumerate(index_ids)}
    index_ids_arr = np.asarray(index_ids)

    sims = query_features @ index_features.T
    # Same protocol as val: a query must not retrieve its own reference image.
    for row, ref in enumerate(ref_ids):
        pos = id_to_pos.get(ref)
        if pos is not None:
            sims[row, pos] = -float("inf")

    top_idx = sims.topk(TOP_K, dim=1).indices.cpu().numpy()
    payload = {str(qid): [int(i) for i in index_ids_arr[top_idx[row]]]
               for row, qid in enumerate(query_ids)}

    out_dir = PROJECT_ROOT / args.out_dir / entry["short"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{entry['short']}_circo_test_top{TOP_K}.json"
    with open(path, "w") as f:
        json.dump(payload, f)
    size_mb = path.stat().st_size / 1024 ** 2
    print(f"  wrote {path.relative_to(PROJECT_ROOT)} ({size_mb:.2f} MB)", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"name": entry["name"], "short": entry["short"], "checkpoint": entry["checkpoint"],
            "queries": len(query_ids), "gallery": len(index_ids),
            "file": str(path.relative_to(PROJECT_ROOT)), "size_mb": round(size_mb, 2)}


def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA — this will be very slow.", flush=True)

    selected = MODELS
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        selected = [e for e in MODELS if e["short"] in wanted]
        missing = wanted - {e["short"] for e in selected}
        if missing:
            raise SystemExit(f"unknown model(s): {sorted(missing)}; "
                             f"available: {sorted(e['short'] for e in MODELS)}")

    results = [generate_for_model(e, args, device) for e in selected]

    out_dir = PROJECT_ROOT / args.out_dir
    manifest_path = out_dir / "submission_manifest.json"
    manifest = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "split": "test",
                "server": "https://circo.micc.unifi.it/", "top_k": TOP_K,
                "protocol": "identical to the val eval in src/evaluation/circo_eval.py",
                "models": results}
    if manifest_path.exists():
        try:
            previous = json.load(open(manifest_path))
            keep = [m for m in previous.get("models", [])
                    if m.get("short") not in {r["short"] for r in results}]
            manifest["models"] = keep + results
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  WARNING: could not merge manifest ({exc})")
    json.dump(manifest, open(manifest_path, "w"), indent=2)

    print(f"\n{'='*70}\nSUBMIT TO https://circo.micc.unifi.it/\n{'='*70}")
    for r in results:
        print(f"  {r['name']}\n     {r['file']}  ({r['size_mb']} MB)")
    print(f"\nManifest -> {manifest_path.relative_to(PROJECT_ROOT)}")
    print("\nNOTE: the server limits test submissions. Submit final models only.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--dataset-path", default="data/circo")
    p.add_argument("--out-dir", default="results/circo_test_submissions")
    p.add_argument("--cache-dir", default="results/circo/cache")
    p.add_argument("--only", default=None, help="comma-separated model short names")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--tqdm", action="store_true")
    main(p.parse_args())
