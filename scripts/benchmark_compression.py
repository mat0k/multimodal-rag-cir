"""
Embedding-compression benchmark: the efficiency axis that CAN differ between two
models sharing one architecture.

scripts/benchmark_efficiency.py showed every architecture-bound cost (params,
FLOPs, latency, memory, index bytes) is bit-identical across our arms — the same
forward pass with different weights. The only thing weights can change is the
GEOMETRY of the embedding space, so the only efficiency question left is:

    how much accuracy survives when the index is made cheaper?

Three measurements, all off one cached encode pass:
  1. PCA / dimension truncation  768 -> 512/384/256/128/64   (index shrinks pro rata)
  2. Quantization                fp32 -> fp16 / int8 / binary (2x / 4x / 32x shrink)
  3. Effective rank + spectrum   how many dims are actually used -> EXPLAINS 1 and 2

Stage 1 (GPU) encodes each benchmark once per model and caches embeddings + all
metadata. Stage 2 (CPU) runs every sweep off that cache, reusing the SAME recall
functions as the val eval, so numbers are directly comparable to reported results.

ANN (HNSW/IVF) is deliberately excluded: the galleries are 2,297 (CIRR) and 6,346
(FashionIQ largest class) vectors, where exact search already takes ~0.1 ms. ANN
targets millions of vectors; at this scale it would measure index-build noise.

Adding a future checkpoint = one more entry in MODELS. No code changes.

Usage:
  python scripts/benchmark_compression.py                 # encode (if needed) + sweep
  python scripts/benchmark_compression.py --sweep-only    # reuse cache, CPU only
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

# --------------------------------------------------------------------------
# Arms under test. Baseline = LaSCo contrastive fine-tune (contrastive_v2 ep1).
# Ours     = SP relation distillation from BGE-VL-MLLM-S1, epoch 7 (its best
#            epoch on BOTH FIQ 27.46 and CIRR 56.55; the checkpoint submitted to
#            the CIRR test server).
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
    # --- MagicLens arm: the second student, same question ---------------------
    # Zero-shot vs SP-distilled. Contrastive fine-tuning is deliberately not an arm
    # here: it never beat zero-shot for MagicLens, so zero-shot IS the baseline.
    {
        "name": "MagicLens-B zero-shot (baseline)",
        "short": "magiclens_base_zeroshot",
        "checkpoint": "models/magiclens/magic_lens_clip_base.pt",
        "type": "magiclens",
        "model_size": "base",
    },
    {
        "name": "MagicLens-B + SP distillation (ours)",
        "short": "magiclens_sp_ce0.1_ep1",
        # Epoch 1 is the genuine best (27.52 / 66.18); both SP runs peak there and
        # decay after. NOT magiclens_best.pth, which selects among trained epochs
        # only and never compares against the untrained start.
        "checkpoint": "results/magiclens/distill_magiclens_bge_vl_sp_w3000_ce0.1/checkpoints/magiclens_epoch01.pth",
        "type": "magiclens",
        "model_size": "base",
    },
]

BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
TRUNC_DIMS = [512, 384, 256, 128, 64]
PRECISIONS = ["fp16", "int8", "binary"]


def _norm(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)


# ---------------------------------------------------------------- stage 1: encode
def encode_model(entry: dict, args, device: str) -> dict:
    from src.datasets.cirr import build_cirr_dataset
    from src.datasets.fashioniq import build_fashioniq_dataset
    from src.evaluation.cirr_eval import (
        generate_cirr_index_features, generate_cirr_predicted_features)
    from src.evaluation.fashioniq_eval import (
        generate_fashioniq_index_features, generate_fashioniq_predicted_features)
    from src.retrievers.backbones.vista.modeling import Visualized_BGE
    from src.retrievers.vista_retriever import VistaBGERetriever

    ckpt = PROJECT_ROOT / entry["checkpoint"]
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    print(f"\n{'='*70}\nENCODING  {entry['name']}\n  {entry['checkpoint']}\n{'='*70}", flush=True)

    model_type = str(entry.get("type", "vista")).strip().lower()
    if model_type == "magiclens":
        # Imported lazily: MagicLens needs open_clip, which the VISTA path does not.
        from src.retrievers.magiclens_retriever import MagicLensRetriever

        model = MagicLensRetriever.from_pretrained(entry.get("model_size", "base"),
                                                   checkpoint_path=str(ckpt))
        backbone = model.backbone
    elif model_type == "vista":
        backbone = Visualized_BGE(model_name_bge=BGE_MODEL_NAME, model_weight=str(ckpt),
                                  negatives_cross_device=False)
        model = VistaBGERetriever(backbone)
    else:
        raise ValueError(f"Unsupported model type {model_type!r}; use 'vista' or 'magiclens'.")

    backbone.eval(); backbone.to(device)
    t0 = time.time()

    # ---- CIRR val ----
    ci = build_cirr_dataset(split="val", mode="images", image_transform=model.image_processor,
                            caption_transform=model.tokenizer, max_length_tokenizer=77)
    ct = build_cirr_dataset(split="val", mode="triplets", image_transform=model.image_processor,
                            caption_transform=model.tokenizer, max_length_tokenizer=77)
    with torch.no_grad():
        c_idx, c_names = generate_cirr_index_features(
            clip_model=model, index_dataset=ci, batch_size=args.batch_size,
            num_workers=args.num_workers, use_tqdm=False)
        c_pred, c_ref, c_tgt, c_grp, c_pair = generate_cirr_predicted_features(
            clip_model=model, triplet_dataset=ct, query_embedding_mode="vista_mm",
            fusion_type="sum", batch_size=args.batch_size, num_workers=args.num_workers,
            use_tqdm=False)
    print(f"  CIRR  index={tuple(c_idx.shape)} queries={tuple(c_pred.shape)} ({time.time()-t0:.0f}s)", flush=True)

    # ---- FashionIQ val (all 3 classes in one pass; metrics split per class) ----
    fi = build_fashioniq_dataset(split="val", mode="images", image_transform=model.image_processor,
                                 caption_transform=model.tokenizer, max_length_tokenizer=77)
    ft = build_fashioniq_dataset(split="val", mode="triplets", image_transform=model.image_processor,
                                 caption_transform=model.tokenizer, max_length_tokenizer=77,
                                 reverse_caption_order=False)
    with torch.no_grad():
        f_idx, f_names, f_icls = generate_fashioniq_index_features(
            clip_model=model, index_dataset=fi, batch_size=args.batch_size,
            num_workers=args.num_workers, use_tqdm=False)
        f_pred, f_ref, f_tgt, f_tcls = generate_fashioniq_predicted_features(
            clip_model=model, triplet_dataset=ft, query_embedding_mode="vista_mm",
            fusion_type="sum", batch_size=args.batch_size, num_workers=args.num_workers,
            use_tqdm=False)
    print(f"  FIQ   index={tuple(f_idx.shape)} queries={tuple(f_pred.shape)} ({time.time()-t0:.0f}s)", flush=True)

    payload = {
        "cirr": {"index": c_idx.cpu(), "index_names": c_names, "pred": c_pred.cpu(),
                 "ref": c_ref, "tgt": c_tgt, "grp": c_grp,
                 "pair": [int(p) for p in c_pair]},
        "fiq": {"index": f_idx.cpu(), "index_names": f_names, "index_classes": f_icls,
                "pred": f_pred.cpu(), "ref": f_ref, "tgt": f_tgt, "classes": f_tcls},
    }
    cache = PROJECT_ROOT / args.out_dir / "cache" / f"{entry['short']}.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    print(f"  cached -> {cache.relative_to(PROJECT_ROOT)}", flush=True)

    del backbone, model
    if device == "cuda":
        torch.cuda.empty_cache()
    return payload


# ---------------------------------------------------------------- recall helpers
def cirr_recall(d: dict, index: torch.Tensor, pred: torch.Tensor) -> dict:
    from src.evaluation.cirr_eval import compute_cirr_metrics
    m = compute_cirr_metrics(
        index_features=index, index_names=d["index_names"], predicted_features=pred,
        reference_names=d["ref"], target_names=d["tgt"], group_members=d["grp"],
        pair_ids=d["pair"], k_values=[1, 5, 10, 50], k_values_subset=[1, 2, 3])
    m["summary_average"] = (m["recall_at5"] + m["subset_recall_at1"]) / 2
    return m


def fiq_recall(d: dict, index: torch.Tensor, pred: torch.Tensor) -> dict:
    from src.evaluation.fashioniq_eval import compute_fashioniq_metrics
    return compute_fashioniq_metrics(
        index_features=index, index_names=d["index_names"], index_classes=d["index_classes"],
        predicted_features=pred, reference_names=d["ref"], target_names=d["tgt"],
        triplet_classes=d["classes"], k_values=[5, 10, 50], metric_prefix="val")


def score(payload: dict, tf) -> dict:
    """Apply transform tf(index, pred) -> (index', pred') and score both benchmarks."""
    c, f = payload["cirr"], payload["fiq"]
    ci, cp = tf(c["index"].float(), c["pred"].float())
    fi, fp = tf(f["index"].float(), f["pred"].float())
    cm, fm = cirr_recall(c, ci, cp), fiq_recall(f, fi, fp)
    return {
        "cirr_summary_avg": round(cm["summary_average"], 3),
        "cirr_R@1": round(cm["recall_at1"], 3),
        "cirr_R@5": round(cm["recall_at5"], 3),
        "cirr_R@10": round(cm["recall_at10"], 3),
        "cirr_Rsub@1": round(cm["subset_recall_at1"], 3),
        "fiq_avg_R@10": round(fm["val_avg_recall_at10"], 3),
        "fiq_avg_R@50": round(fm["val_avg_recall_at50"], 3),
    }


# ---------------------------------------------------------------- transforms
def make_pca(dim: int):
    """Fit an uncentered top-`dim` SVD basis on the INDEX (what you actually store),
    apply the same basis to queries. Uncentered because centering distorts the
    cosine geometry retrieval depends on."""
    def tf(index, pred):
        X = _norm(index)
        G = X.T @ X
        ev, V = torch.linalg.eigh(G)
        V = V[:, torch.argsort(ev, descending=True)[:dim]].contiguous()
        return _norm(X @ V), _norm(_norm(pred) @ V)
    return tf


def make_quant(kind: str):
    """Simulated quantization: store at reduced precision, score as usual.
    Applied to index AND queries so the whole pipeline is at that precision."""
    def q(x):
        x = _norm(x)
        if kind == "fp16":
            return _norm(x.half().float())
        if kind == "int8":
            s = x.abs().amax(dim=0, keepdim=True).clamp(min=1e-8) / 127.0
            return _norm(torch.round(x / s).clamp(-127, 127) * s)
        if kind == "binary":
            # sign vectors: cosine ranking on +-1 is monotonic in Hamming distance
            return _norm(torch.sign(x).where(x != 0, torch.ones_like(x)))
        raise ValueError(kind)
    return lambda index, pred: (q(index), q(pred))


def spectrum_stats(index: torch.Tensor) -> dict:
    """Effective rank + energy curve of the gallery — the MECHANISM behind the
    truncation/quantization results, not an efficiency metric itself."""
    X = _norm(index.float())
    ev = torch.linalg.eigvalsh(X.T @ X).flip(0).clamp(min=0)
    tot = ev.sum()
    p = (ev / tot)
    p = p[p > 0]
    erank = float(torch.exp(-(p * p.log()).sum()))
    cum = torch.cumsum(ev, 0) / tot
    return {
        "effective_rank": round(erank, 1),
        "ambient_dim": int(ev.numel()),
        "energy_at": {str(k): round(float(cum[k - 1]) * 100, 2) for k in TRUNC_DIMS},
        "dims_for_95pct": int((cum < 0.95).sum()) + 1,
        "dims_for_99pct": int((cum < 0.99).sum()) + 1,
    }


# ---------------------------------------------------------------- report
def render(results: list[dict], env: dict) -> str:
    L = [
        "# Embedding-Compression Benchmark\n",
        f"_Generated {env['timestamp']} · {env['device']} · torch {env['torch']}_\n",
        "Companion to `results/efficiency/efficiency_report.md`, where every "
        "architecture-bound cost came out **bit-identical** across these two arms "
        "(same forward pass, different weights). Weights can only change the *geometry* "
        "of the embedding space, so the remaining efficiency question is how much "
        "accuracy survives making the index cheaper. Metrics use the same recall "
        "functions as the val eval, so they are directly comparable to reported numbers.\n",
        "\nANN (HNSW/IVF) is excluded by design: CIRR val is 2,297 vectors and the largest "
        "FashionIQ class 6,346, where exact search already runs in ~0.1 ms. ANN targets "
        "millions of vectors; here it would measure index-build noise.\n",
    ]
    names = [r["name"] for r in results]
    hdr = "| Setting | Index size | " + " | ".join(names) + " |"
    sep = "|---|---|" + "---|" * len(names)

    # Read the real embedding width off the results rather than assuming 768:
    # VISTA is 768-d but MagicLens is 512-d, and hardcoding the former silently
    # misreports every index size and compression ratio for the latter.
    dim = results[0].get("dim", 768)

    def block(title, key, fmt="{:.2f}"):
        L.append(f"\n## {title}\n")
        L.append(hdr); L.append(sep)
        base = results[0]["full"]["_index_mb"]
        L.append(f"| **fp32, {dim}-d (uncompressed)** | " + f"{base:.2f} MB | " +
                 " | ".join(fmt.format(r["full"][key]) for r in results) + " |")
        for d in TRUNC_DIMS:
            if d >= dim:
                continue  # not a truncation for this model
            mb = base * d / dim
            L.append(f"| PCA {d}-d | {mb:.2f} MB ({dim/d:.1f}× smaller) | " +
                     " | ".join(fmt.format(r["pca"][str(d)][key]) for r in results) + " |")
        for p in PRECISIONS:
            factor = {"fp16": 2, "int8": 4, "binary": 32}[p]
            L.append(f"| {p}, {dim}-d | {base/factor:.2f} MB ({factor}× smaller) | " +
                     " | ".join(fmt.format(r["quant"][p][key]) for r in results) + " |")

    block("CIRR val — summary average (R@5 + R_sub@1)/2", "cirr_summary_avg")
    block("FashionIQ val — average Recall@10", "fiq_avg_R@10")

    L.append("\n## Embedding geometry (mechanism)\n")
    L.append("| Metric | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    for bench, label in (("cirr", "CIRR"), ("fiq", "FashionIQ")):
        L.append(f"| **{label} gallery** | " + " | ".join("" for _ in results) + " |")
        L.append(f"| effective rank (of {dim}) | " +
                 " | ".join(str(r["spectrum"][bench]["effective_rank"]) for r in results) + " |")
        L.append(f"| dims for 95% energy | " +
                 " | ".join(str(r["spectrum"][bench]["dims_for_95pct"]) for r in results) + " |")
        for d in (256, 128):
            L.append(f"| energy in top-{d} dims (%) | " +
                     " | ".join(str(r["spectrum"][bench]["energy_at"][str(d)]) for r in results) + " |")
    L.append("\n_Higher effective rank = information spread over more dimensions = more lost "
             "when truncating. This is the explanation for the tables above, not a cost metric._\n")

    # ---- Retention: how much of its OWN score each model keeps under compression.
    # Absolute scores already say who is better; this isolates whose embedding space
    # is more robust to being squeezed, independent of the starting accuracy.
    def _settings():
        for d in TRUNC_DIMS:
            if d < dim:
                yield f"PCA {d}-d", lambda r, d=d: r["pca"][str(d)]
        for p in PRECISIONS:
            yield p, lambda r, p=p: r["quant"][p]

    L.append("\n## Retention — % of each model's own uncompressed score\n")
    L.append("| Setting | " + " | ".join(
        f"{r['name']} — CIRR | {r['name']} — FIQ" for r in results) + " |")
    L.append("|---" * (2 * len(results) + 1) + "|")
    for label, get in _settings():
        cells = []
        for r in results:
            for key in ("cirr_summary_avg", "fiq_avg_R@10"):
                full = r["full"][key]
                cells.append(f"{100.0 * get(r)[key] / full:.1f}%" if full else "n/a")
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    # ---- Iso-accuracy: how far the improved model can be compressed while still
    # beating the baseline's UNCOMPRESSED score. The strongest efficiency claim
    # available here -- accuracy gain and index savings at the same time.
    if len(results) >= 2:
        base_m, ours = results[0], results[-1]
        L.append(f"\n## Iso-accuracy — {ours['name']} compressed vs. "
                 f"{base_m['name']} uncompressed\n")
        L.append("| Benchmark | Baseline (fp32) | Smallest setting still beating it | Score | Index saving |")
        L.append("|---|---|---|---|---|")
        base_mb = results[0]["full"]["_index_mb"]
        for key, label in (("cirr_summary_avg", "CIRR summary"),
                           ("fiq_avg_R@10", "FashionIQ avg R@10")):
            target = base_m["full"][key]
            best = None  # smallest index that still clears the baseline
            for lbl, get in _settings():
                score_ = get(ours)[key]
                if score_ <= target:
                    continue
                if lbl.startswith("PCA"):
                    d = int(lbl.split()[1].rstrip("-d")); mb = base_mb * d / dim
                else:
                    mb = base_mb / {"fp16": 2, "int8": 4, "binary": 32}[lbl]
                if best is None or mb < best[2]:
                    best = (lbl, score_, mb)
            if best:
                L.append(f"| {label} | {target:.2f} | {best[0]} | {best[1]:.2f} | "
                         f"{base_mb:.2f} → {best[2]:.2f} MB ({base_mb/best[2]:.1f}× smaller) |")
            else:
                L.append(f"| {label} | {target:.2f} | none | — | — |")
        L.append("")
    return "\n".join(L)


def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    selected = MODELS
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        selected = [e for e in MODELS if e["short"] in wanted]
        missing = wanted - {e["short"] for e in selected}
        if missing:
            raise SystemExit(f"unknown model short name(s): {sorted(missing)}; "
                             f"available: {sorted(e['short'] for e in MODELS)}")

    for entry in selected:
        cache = out_dir / "cache" / f"{entry['short']}.pt"
        if args.sweep_only or cache.exists():
            if not cache.exists():
                raise SystemExit(f"--sweep-only but cache missing: {cache}")
            print(f"\nloading cache {cache.relative_to(PROJECT_ROOT)}", flush=True)
            payload = torch.load(cache, map_location="cpu", weights_only=False)
        else:
            payload = encode_model(entry, args, device)

        print(f"\n{'='*70}\nSWEEPING  {entry['name']}\n{'='*70}", flush=True)
        r = {"name": entry["name"], "short": entry["short"], "checkpoint": entry["checkpoint"]}

        idx_mb = payload["cirr"]["index"].numel() * 4 / 1024**2
        r["full"] = score(payload, lambda i, p: (_norm(i), _norm(p)))
        r["full"]["_index_mb"] = round(idx_mb, 2)
        # Real embedding width, so the report never assumes 768.
        r["dim"] = int(payload["cirr"]["index"].shape[-1])
        print(f"  fp32/{r['dim']}  CIRR {r['full']['cirr_summary_avg']}  FIQ {r['full']['fiq_avg_R@10']}", flush=True)

        r["pca"] = {}
        for d in TRUNC_DIMS:
            r["pca"][str(d)] = score(payload, make_pca(d))
            print(f"  PCA {d:4d}  CIRR {r['pca'][str(d)]['cirr_summary_avg']}  "
                  f"FIQ {r['pca'][str(d)]['fiq_avg_R@10']}", flush=True)

        r["quant"] = {}
        for p in PRECISIONS:
            r["quant"][p] = score(payload, make_quant(p))
            print(f"  {p:8s}  CIRR {r['quant'][p]['cirr_summary_avg']}  "
                  f"FIQ {r['quant'][p]['fiq_avg_R@10']}", flush=True)

        r["spectrum"] = {"cirr": spectrum_stats(payload["cirr"]["index"]),
                         "fiq": spectrum_stats(payload["fiq"]["index"])}
        print(f"  eff.rank  CIRR {r['spectrum']['cirr']['effective_rank']}  "
              f"FIQ {r['spectrum']['fiq']['effective_rank']}", flush=True)
        results.append(r)

    env = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           "device": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
           "torch": torch.__version__, "trunc_dims": TRUNC_DIMS, "precisions": PRECISIONS}
    json.dump({"environment": env, "models": results},
              open(out_dir / "compression_results.json", "w"), indent=2)
    report = render(results, env)
    (out_dir / "compression_report.md").write_text(report)

    # flat CSV for the thesis
    rows = ["model,dim,setting,index_mb,cirr_summary_avg,cirr_R@1,fiq_avg_R@10,fiq_avg_R@50"]
    for r in results:
        base = r["full"]["_index_mb"]
        rows.append(f"{r['short']},{r['dim']},fp32_{r['dim']},{base:.2f},{r['full']['cirr_summary_avg']},"
                    f"{r['full']['cirr_R@1']},{r['full']['fiq_avg_R@10']},{r['full']['fiq_avg_R@50']}")
        for d in TRUNC_DIMS:
            s = r["pca"][str(d)]
            rows.append(f"{r['short']},{r['dim']},pca_{d},{base*d/r['dim']:.2f},{s['cirr_summary_avg']},"
                        f"{s['cirr_R@1']},{s['fiq_avg_R@10']},{s['fiq_avg_R@50']}")
        for p in PRECISIONS:
            s = r["quant"][p]
            fct = {"fp16": 2, "int8": 4, "binary": 32}[p]
            rows.append(f"{r['short']},{r['dim']},{p}_{r['dim']},{base/fct:.2f},{s['cirr_summary_avg']},"
                        f"{s['cirr_R@1']},{s['fiq_avg_R@10']},{s['fiq_avg_R@50']}")
    (out_dir / "compression_results.csv").write_text("\n".join(rows) + "\n")

    print("\n" + report)
    print(f"\nJSON   -> {(out_dir/'compression_results.json').relative_to(PROJECT_ROOT)}")
    print(f"CSV    -> {(out_dir/'compression_results.csv').relative_to(PROJECT_ROOT)}")
    print(f"Report -> {(out_dir/'compression_report.md').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Embedding-compression benchmark.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--sweep-only", action="store_true", help="reuse cached embeddings (CPU only)")
    p.add_argument("--out-dir", default="results/compression")
    p.add_argument("--only", default=None,
                   help="Comma-separated model short names (default: all). Use this to run "
                        "one arm pair without recomputing the others.")
    main(p.parse_args())
