"""
Generate CIRR **test1** submission files for the official evaluation server.

CIRR withholds test1 ground truth (`cap.rc2.test1.json` has no `target_hard`),
so test-split numbers can only be obtained by submitting ranked predictions to
    https://cirr.cecs.anu.edu.au/        (backup: https://cirr.junjie.au/)

This reuses the EXACT val-eval protocol already in src/evaluation/cirr_eval.py
(same encoders, same L2-normalised cosine ranking, same reference-image removal),
switching only the split to 'test1' and asking for ranked NAMES instead of metrics.
compute_cirr_metrics(..., return_type='names') is target-free: every assertion that
needs ground truth is guarded behind return_type == 'metrics'.

Two files per model, because the server takes ONE metric per submission:
    *_recall_top50.json        {"version":"rc2","metric":"recall",        "<pair_id>":[50 names]}
    *_recall_subset_top3.json  {"version":"rc2","metric":"recall_subset", "<pair_id>":[3 names]}

Both arms must be submitted so the paper compares baseline vs. ours on the SAME
split — a test-split number is not comparable against a val-split baseline.

Adding a future better model = one more entry in MODELS. No code changes.

Usage:
  python scripts/generate_cirr_test_submission.py
  python scripts/generate_cirr_test_submission.py --batch-size 64
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

from src.datasets.cirr import build_cirr_dataset  # noqa: E402
from src.evaluation.cirr_eval import (  # noqa: E402
    compute_cirr_metrics,
    generate_cirr_index_features,
    generate_cirr_predicted_features,
)
from src.retrievers.backbones.vista.modeling import Visualized_BGE  # noqa: E402
from src.retrievers.vista_retriever import VistaBGERetriever  # noqa: E402

# --------------------------------------------------------------------------
# Models to generate submissions for. Same list shape as benchmark_efficiency.py.
# --------------------------------------------------------------------------
MODELS: list[dict] = [
    {
        "name": "VISTA fine-tuned (baseline)",
        "short": "baseline_vista_contrastive_v2_ep1",
        "checkpoint": "results/training/vista_lasco_finetune_contrastive_v2/checkpoints/vista_epoch01.pth",
    },
    {
        "name": "VISTA + SP distillation (ours)",
        "short": "distilled_bge_vl_sp_w3000_ep7",
        "checkpoint": "results/bge_vl/distill_bge_vl_sp_w3000/checkpoints/vista_epoch07.pth",
    },
    {
        # Zero-shot, straight from the converted weights: this is the row the
        # MagicLens paper reports (Table 13, MagicLens-B / CLIP-B, 166M params),
        # so submitting it tests whether the port reproduces the published
        # test-split numbers -- R@1 27.0, R@5 58.0, R@10 70.9, R@50 91.1,
        # subset R@1 66.7, R@2 83.9, R@3 92.4.
        "name": "MagicLens-B zero-shot (ported)",
        "short": "magiclens_base_zeroshot",
        "checkpoint": "models/magiclens/magic_lens_clip_base.pt",
        "type": "magiclens",
        "model_size": "base",
    },
]

BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DATASET_VERSION = "rc2"
MAX_UPLOAD_MB = 5.0  # server limit


def _write_submission(path: Path, metric: str, ranked: dict) -> float:
    """Write one submission file; returns its size in MB."""
    payload = {"version": DATASET_VERSION, "metric": metric}
    for pair_id, names in ranked.items():
        # json turns int keys into strings automatically; be explicit anyway.
        payload[str(pair_id)] = list(names)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)
    return path.stat().st_size / 1024**2


def generate_for_model(entry: dict, args, device: str) -> dict:
    ckpt = PROJECT_ROOT / entry["checkpoint"]
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")

    print(f"\n{'='*70}\n{entry['name']}\n  {entry['checkpoint']}\n{'='*70}", flush=True)

    model_type = str(entry.get("type", "vista")).strip().lower()
    if model_type == "magiclens":
        # Imported lazily: MagicLens needs open_clip, which the VISTA path does not.
        from src.retrievers.magiclens_retriever import MagicLensRetriever

        model = MagicLensRetriever.from_pretrained(
            entry.get("model_size", "base"), checkpoint_path=str(ckpt)
        )
        backbone = model.backbone
    elif model_type == "vista":
        backbone = Visualized_BGE(
            model_name_bge=BGE_MODEL_NAME,
            model_weight=str(ckpt),
            negatives_cross_device=False,
        )
        model = VistaBGERetriever(backbone)
    else:
        raise ValueError(f"Unsupported model type {model_type!r}; use 'vista' or 'magiclens'.")

    backbone.eval()
    backbone.to(device)

    # ---- test1 gallery + queries (dataset already supports split='test1') ----
    index_ds = build_cirr_dataset(
        split="test1", mode="images",
        image_transform=model.image_processor, caption_transform=model.tokenizer,
        max_length_tokenizer=77,
    )
    triplet_ds = build_cirr_dataset(
        split="test1", mode="triplets",
        image_transform=model.image_processor, caption_transform=model.tokenizer,
        max_length_tokenizer=77,
    )
    print(f"  gallery={len(index_ds):,} images | queries={len(triplet_ds):,}", flush=True)

    t0 = time.time()
    with torch.no_grad():
        index_features, index_names = generate_cirr_index_features(
            clip_model=model, index_dataset=index_ds,
            batch_size=args.batch_size, num_workers=args.num_workers, use_tqdm=False,
        )
        print(f"  index encoded {tuple(index_features.shape)} ({time.time()-t0:.0f}s)", flush=True)

        # skip_targets=True -> target_names comes back empty (test1 has no labels).
        preds, ref_names, target_names, group_members, pair_ids = generate_cirr_predicted_features(
            clip_model=model, triplet_dataset=triplet_ds,
            query_embedding_mode=args.query_embedding_mode, fusion_type=args.fusion_type,
            batch_size=args.batch_size, num_workers=args.num_workers, use_tqdm=False,
            skip_targets=True,
        )
    print(f"  queries encoded {tuple(preds.shape)} ({time.time()-t0:.0f}s)", flush=True)
    assert not target_names, "expected no targets for test1 (skip_targets=True)"

    # ---- rank; return_type='names' skips every ground-truth assertion ----
    ranked = compute_cirr_metrics(
        index_features=index_features, index_names=index_names,
        predicted_features=preds, reference_names=ref_names,
        target_names=target_names,          # empty — unused when return_type='names'
        group_members=group_members, pair_ids=pair_ids,
        k_values=[args.top_k], k_values_subset=[args.subset_top_k],
        return_type="names",
    )

    top_full = ranked[f"top_{args.top_k}"]
    top_subset = ranked[f"subset_top_{args.subset_top_k}"]

    # ---- sanity: shape + no reference leakage ----
    assert len(top_full) == len(triplet_ds), f"{len(top_full)} != {len(triplet_ds)} queries"
    assert len(top_subset) == len(triplet_ds)
    bad = [p for p, names in top_full.items() if len(names) != args.top_k]
    assert not bad, f"{len(bad)} queries lack {args.top_k} candidates"
    ref_by_pair = {
        (pid.item() if hasattr(pid, "item") else pid): r for pid, r in zip(pair_ids, ref_names)
    }
    leaked = [p for p, names in top_full.items() if ref_by_pair.get(p) in names]
    assert not leaked, f"reference image leaked into ranking for {len(leaked)} queries"

    # ---- write both files ----
    out_dir = PROJECT_ROOT / args.out_dir / entry["short"]
    f_full = out_dir / f"{entry['short']}_recall_top{args.top_k}.json"
    f_sub = out_dir / f"{entry['short']}_recall_subset_top{args.subset_top_k}.json"
    mb_full = _write_submission(f_full, "recall", top_full)
    mb_sub = _write_submission(f_sub, "recall_subset", top_subset)

    for f, mb in ((f_full, mb_full), (f_sub, mb_sub)):
        flag = "  ** EXCEEDS 5MB SERVER LIMIT **" if mb > MAX_UPLOAD_MB else ""
        print(f"  -> {f.relative_to(PROJECT_ROOT)}  ({mb:.2f} MB){flag}", flush=True)

    del backbone, model, index_features, preds
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "name": entry["name"], "short": entry["short"], "checkpoint": entry["checkpoint"],
        "gallery_size": len(index_ds), "num_queries": len(triplet_ds),
        "files": {
            "recall": {"path": str(f_full.relative_to(PROJECT_ROOT)), "size_mb": round(mb_full, 2),
                       "within_limit": mb_full <= MAX_UPLOAD_MB},
            "recall_subset": {"path": str(f_sub.relative_to(PROJECT_ROOT)), "size_mb": round(mb_sub, 2),
                              "within_limit": mb_sub <= MAX_UPLOAD_MB},
        },
    }


def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA unavailable — this will be very slow.", flush=True)

    selected = MODELS
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        selected = [e for e in MODELS if e["short"] in wanted]
        missing = wanted - {e["short"] for e in selected}
        if missing:
            raise SystemExit(f"unknown model short name(s): {sorted(missing)}; "
                             f"available: {sorted(e['short'] for e in MODELS)}")

    results = [generate_for_model(e, args, device) for e in selected]

    out_dir = PROJECT_ROOT / args.out_dir
    manifest = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": "test1",
        "dataset_version": DATASET_VERSION,
        "server": "https://cirr.cecs.anu.edu.au/  (backup https://cirr.junjie.au/)",
        "protocol": {
            "query_embedding_mode": args.query_embedding_mode,
            "fusion_type": args.fusion_type,
            "top_k": args.top_k, "subset_top_k": args.subset_top_k,
            "note": "identical to the val-split eval protocol in src/evaluation/cirr_eval.py",
        },
        "models": results,
    }

    # Merge into any existing manifest rather than clobbering it, so a partial
    # re-run (--only) does not erase the record of previously generated models.
    manifest_path = out_dir / "submission_manifest.json"
    if manifest_path.exists():
        try:
            previous = json.load(open(manifest_path))
            kept = [m for m in previous.get("models", [])
                    if m.get("short") not in {r.get("short") for r in results}]
            manifest["models"] = kept + results
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  WARNING: could not merge existing manifest ({exc}); writing fresh.")

    json.dump(manifest, open(manifest_path, "w"), indent=2)

    print(f"\n{'='*70}\nSUBMIT THESE {2*len(results)} FILES to https://cirr.cecs.anu.edu.au/\n{'='*70}")
    for r in results:
        print(f"\n{r['name']}")
        for metric, info in r["files"].items():
            print(f"   [{metric:13s}] {info['path']}  ({info['size_mb']} MB)")
    print(f"\nManifest -> {(out_dir / 'submission_manifest.json').relative_to(PROJECT_ROOT)}")
    print("\nNOTE: the server caps test-split submissions to discourage tuning on test.\n"
          "      Submit only final models; do not iterate against the returned scores.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate CIRR test1 submission files.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--query-embedding-mode", default="vista_mm")
    p.add_argument("--fusion-type", default="sum")
    p.add_argument("--top-k", type=int, default=50, help="server wants top-50 for 'recall'")
    p.add_argument("--subset-top-k", type=int, default=3, help="server wants top-3 for 'recall_subset'")
    p.add_argument("--out-dir", default="results/cirr_test_submissions")
    p.add_argument("--only", default=None,
                   help="Comma-separated model short names to generate (default: all). "
                        "Use this to avoid regenerating models already submitted.")
    main(p.parse_args())
