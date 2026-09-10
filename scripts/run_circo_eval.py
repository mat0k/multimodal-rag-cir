"""Evaluate a retriever on CIRCO **val** (mAP@k, scored locally).

One script for every student: it builds models through the same registry used by the
CIRR submission script, so VISTA and MagicLens arms differ only by a name on the
command line.

    python scripts/run_circo_eval.py --only magiclens_sp_ce0.1_ep1
    python scripts/run_circo_eval.py                 # all registered models

CIRCO's gallery is ~123K images (CIRR's is 2,297), so a single arm takes far longer
than a CIRR run. The gallery encoding is cached per model and reused on re-runs.
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


def _rel(path: Path) -> str:
    """Project-relative when possible; absolute paths are reported as-is."""
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)

from src.datasets.circo import build_circo_dataset  # noqa: E402
from src.evaluation.circo_eval import (  # noqa: E402
    compute_circo_metrics,
    generate_circo_index_features,
    generate_circo_query_features,
)

BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Same shape as generate_cirr_test_submission.py's registry: adding an arm is one entry.
MODELS: list[dict] = [
    {
        # Off-the-shelf reference: what you get by downloading VISTA.
        "name": "VISTA zero-shot",
        "short": "vista_zeroshot",
        "checkpoint": "models/Visualized_BGE/Visualized_base_en_v1.5.pth",
        "out_root": "results/bge_vl/circo_vista_zeroshot",
    },
    {
        # The checkpoint SP actually starts from, so the CIRCO delta is measured
        # against the same reference as the CIRR/Fashion-IQ deltas.
        "name": "VISTA fine-tuned (SP baseline)",
        "short": "baseline_vista_contrastive_v2_ep1",
        "checkpoint": "results/training/vista_lasco_finetune_contrastive_v2/checkpoints/vista_epoch01.pth",
        "out_root": "results/bge_vl/circo_vista_contrastive_v2_ep1",
    },
    {
        "name": "VISTA + SP distillation (ours)",
        "short": "distilled_bge_vl_sp_w3000_ep7",
        "checkpoint": "results/bge_vl/distill_bge_vl_sp_w3000/checkpoints/vista_epoch07.pth",
        "out_root": "results/bge_vl/circo_vista_sp_w3000_ep7",
    },
    {
        "name": "MagicLens-B zero-shot (baseline)",
        "short": "magiclens_base_zeroshot",
        "checkpoint": "models/magiclens/magic_lens_clip_base.pt",
        "out_root": "results/magiclens/circo_magiclens_zeroshot",
        "type": "magiclens",
        "model_size": "base",
    },
    {
        "name": "MagicLens-B + SP distillation (ours)",
        "short": "magiclens_sp_ce0.1_ep1",
        "checkpoint": "results/magiclens/distill_magiclens_bge_vl_sp_w3000_ce0.1/checkpoints/magiclens_epoch01.pth",
        "out_root": "results/magiclens/circo_magiclens_sp_ce0.1_ep1",
        "type": "magiclens",
        "model_size": "base",
    },
]


def build_model(entry: dict, device: str):
    """Construct a retriever. The eval path is model-agnostic; only this differs."""
    ckpt = PROJECT_ROOT / entry["checkpoint"]
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")

    model_type = str(entry.get("type", "vista")).strip().lower()
    if model_type == "magiclens":
        # Lazy: MagicLens needs open_clip, the VISTA path does not.
        from src.retrievers.magiclens_retriever import MagicLensRetriever

        model = MagicLensRetriever.from_pretrained(
            entry.get("model_size", "base"), checkpoint_path=str(ckpt))
        backbone = model.backbone
    elif model_type == "vista":
        from src.retrievers.backbones.vista.modeling import Visualized_BGE
        from src.retrievers.vista_retriever import VistaBGERetriever

        backbone = Visualized_BGE(model_name_bge=BGE_MODEL_NAME, model_weight=str(ckpt),
                                  negatives_cross_device=False)
        model = VistaBGERetriever(backbone)
    else:
        raise ValueError(f"unsupported type {model_type!r}; use 'vista' or 'magiclens'")

    backbone.eval()
    backbone.to(device)
    return model


TOP_K = 50  # ranked ids per query in a test submission


def _write_test_submission(model, entry, args, index_features, index_ids, device) -> dict:
    """Rank the test queries against the already-encoded gallery and write the JSON."""
    query_ds = build_circo_dataset(split="test", mode="triplets", dataset_path=args.dataset_path,
                                   image_transform=model.image_processor,
                                   caption_transform=model.tokenizer)
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

    sub_dir = PROJECT_ROOT / args.submission_dir / entry["short"]
    sub_dir.mkdir(parents=True, exist_ok=True)
    path = sub_dir / f"{entry['short']}_circo_test_top{TOP_K}.json"
    with open(path, "w") as f:
        json.dump(payload, f)
    size_mb = path.stat().st_size / 1024 ** 2
    print(f"  test submission -> {_rel(path)} ({size_mb:.2f} MB)", flush=True)
    return {"file": _rel(path), "queries": len(query_ids),
            "size_mb": round(size_mb, 2)}


def evaluate(entry: dict, args, device: str) -> dict:
    """One pass per model: encode the gallery once, then score val and/or write test.

    val and test index the SAME 123K COCO images, so sharing one encoding across both
    is what makes doing them together nearly free compared with two separate jobs.
    """
    print(f"\n{'='*70}\n{entry['name']}\n  {entry['checkpoint']}\n{'='*70}", flush=True)
    model = build_model(entry, device)
    out_root = PROJECT_ROOT / entry.get("out_root", f"{args.out_dir}/{entry['short']}")
    out_root.mkdir(parents=True, exist_ok=True)

    index_ds = build_circo_dataset(split="val", mode="images", dataset_path=args.dataset_path,
                                   image_transform=model.image_processor,
                                   caption_transform=model.tokenizer)
    print(f"  gallery={len(index_ds):,} images", flush=True)

    # Encoding 123K images dominates runtime; cache it next to that model's results.
    cache = out_root / "gallery_index.pt"
    t0 = time.time()
    if cache.exists() and not args.no_cache:
        print(f"  loading cached gallery: {cache.name}", flush=True)
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        index_features, index_ids = payload["features"], payload["ids"]
    else:
        index_features, index_ids = generate_circo_index_features(
            model, index_ds, args.batch_size, args.num_workers, args.tqdm)
        torch.save({"features": index_features, "ids": index_ids}, cache)
    print(f"  gallery ready ({time.time()-t0:.0f}s)", flush=True)

    record = {"name": entry["name"], "short": entry["short"],
              "checkpoint": entry["checkpoint"], "out_root": _rel(out_root)}

    if args.split in ("val", "both"):
        query_ds = build_circo_dataset(split="val", mode="triplets", dataset_path=args.dataset_path,
                                       image_transform=model.image_processor,
                                       caption_transform=model.tokenizer)
        qf, _, ref_ids, gt_lists = generate_circo_query_features(
            model, query_ds, args.batch_size, args.num_workers, args.tqdm)
        metrics = compute_circo_metrics(index_features, index_ids, qf, ref_ids,
                                        gt_lists, tuple(args.k_values))
        record.update(metrics)
        for k in args.k_values:
            print(f"  val mAP@{k:<3} {metrics[f'val_map_at{k}']:.2f}", flush=True)
        json.dump(record, open(out_root / "circo_val_metrics.json", "w"), indent=2)

    if args.split in ("test", "both"):
        record["test_submission"] = _write_test_submission(
            model, entry, args, index_features, index_ids, device)

    record["elapsed_seconds"] = round(time.time() - t0, 1)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return record


def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA — a 123K-image gallery will be extremely slow.", flush=True)

    selected = MODELS
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        selected = [e for e in MODELS if e["short"] in wanted]
        missing = wanted - {e["short"] for e in selected}
        if missing:
            raise SystemExit(f"unknown model(s): {sorted(missing)}; "
                             f"available: {sorted(e['short'] for e in MODELS)}")

    results = [evaluate(e, args, device) for e in selected]

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge with any previous results so a partial re-run does not erase other arms.
    path = out_dir / "circo_val_results.json"
    if path.exists():
        try:
            previous = json.load(open(path))
            keep = [m for m in previous if m.get("short") not in {r["short"] for r in results}]
            results = keep + results
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  WARNING: could not merge existing results ({exc})")
    json.dump(results, open(path, "w"), indent=2)

    if args.split in ("val", "both"):
        print(f"\n{'model':<40}" + "".join(f"{'mAP@'+str(k):>10}" for k in args.k_values))
        for r in results:
            if f"val_map_at{args.k_values[0]}" in r:
                print(f"{r['name'][:39]:<40}" +
                      "".join(f"{r[f'val_map_at{k}']:>10.2f}" for k in args.k_values))
    if args.split in ("test", "both"):
        print(f"\nSUBMIT TO https://circo.micc.unifi.it/")
        for r in results:
            if "test_submission" in r:
                print(f"   {r['name'][:44]:<46}{r['test_submission']['file']}")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 25, 50])
    p.add_argument("--dataset-path", default="data/circo")
    p.add_argument("--split", choices=["val", "test", "both"], default="both",
                   help="val scores locally; test writes a submission; both shares one "
                        "gallery encoding and is what you normally want.")
    p.add_argument("--submission-dir", default="results/circo_submissions")
    p.add_argument("--out-dir", default="results/circo",
                   help="fallback root for arms without an explicit out_root")
    p.add_argument("--only", default=None, help="comma-separated model short names")
    p.add_argument("--no-cache", action="store_true", help="re-encode the gallery")
    p.add_argument("--tqdm", action="store_true")
    main(p.parse_args())
