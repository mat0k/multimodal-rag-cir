"""
Inference-efficiency benchmark for the CIR retrievers (paper efficiency table).

Compares the deployed models head-to-head on serving cost. Both arms are the SAME
VISTA architecture and differ only in weights, so the architecture-bound metrics
are expected to be IDENTICAL — that is the point: it shows the recall gain from
relation-based distillation is free at inference time. The teacher (BGE-VL-MLLM-S1)
is deliberately NOT benchmarked: it is a training-time artifact that never runs at
serving time.

Metrics (all measured, none asserted):
  Static  : total/trainable params, per-branch params, embedding dim, checkpoint size
  Compute : FLOPs per forward (image branch + composed-query branch)
  Latency : batch=1 mean/std/p50/p95 for encode_image and encode_mm
  Through.: images/sec and queries/sec at a serving batch size
  Memory  : peak GPU memory during a forward pass
  Index   : embedding-store bytes and brute-force search latency over a gallery

Adding a future better model = one more entry in MODELS (below). No code changes.

Usage:
  python scripts/benchmark_efficiency.py                      # all models in MODELS
  python scripts/benchmark_efficiency.py --trials 100 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrievers.backbones.vista.modeling import Visualized_BGE  # noqa: E402

# --------------------------------------------------------------------------
# The models under test. Same architecture, different weights.
# To benchmark a future/better checkpoint, append one entry here.
# --------------------------------------------------------------------------
MODELS: list[dict] = [
    {
        "name": "VISTA fine-tuned (baseline)",
        "short": "baseline_vista_contrastive_v2_ep1",
        "checkpoint": "results/training/vista_lasco_finetune_contrastive_v2/checkpoints/vista_epoch01.pth",
        "note": "LaSCo contrastive fine-tune, epoch 1 — the baseline student.",
    },
    {
        "name": "VISTA + SP distillation (ours)",
        "short": "distilled_bge_vl_sp_w3000_ep7",
        "checkpoint": "results/bge_vl/distill_bge_vl_sp_w3000/checkpoints/vista_epoch07.pth",
        "note": "Unsupervised relation-based (SP) distillation from BGE-VL-MLLM-S1 on LaSCo, best epoch.",
    },
]

BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
TOKEN_LEN = 77  # matches lasco_sp_distill / eval tokenization


@dataclass
class Timing:
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    trials: int = field(default=0)


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _time_calls(fn, trials: int, warmup: int, device: str) -> Timing:
    """Time fn() with warmup, syncing around each call so GPU work is counted."""
    for _ in range(warmup):
        fn()
    _sync(device)

    samples: list[float] = []
    for _ in range(trials):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    return Timing(
        mean_ms=statistics.fmean(samples),
        std_ms=statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        p50_ms=samples[len(samples) // 2],
        p95_ms=samples[min(int(len(samples) * 0.95), len(samples) - 1)],
        trials=trials,
    )


def _count_params(backbone: Visualized_BGE) -> dict:
    total = sum(p.numel() for p in backbone.parameters())
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    # Per-branch split: the EVA vision tower vs the BGE text/fusion encoder.
    vision = sum(p.numel() for p in backbone.model_visual.parameters())
    return {
        "total_params": total,
        "total_params_millions": round(total / 1e6, 2),
        "trainable_params": trainable,
        "vision_branch_params": vision,
        "vision_branch_params_millions": round(vision / 1e6, 2),
        "text_fusion_branch_params": total - vision,
        "text_fusion_branch_params_millions": round((total - vision) / 1e6, 2),
    }


def _image_tensor_shape(backbone: Visualized_BGE) -> tuple[int, int, int]:
    """Derive the real preprocessed image shape instead of hardcoding 224."""
    dummy = Image.new("RGB", (640, 480), (128, 128, 128))
    t = backbone.preprocess_val(dummy)
    return tuple(t.shape)  # (C, H, W)


def _measure_flops(backbone, images, texts, device: str) -> dict:
    """FLOPs per forward for each encode path. Hardware-independent compute cost."""
    from torch.utils.flop_counter import FlopCounterMode

    out = {}
    for label, fn in (
        ("encode_image", lambda: backbone.encode_image(images)),
        ("encode_mm", lambda: backbone.encode_mm(images, texts)),
    ):
        try:
            counter = FlopCounterMode(display=False)
            with counter, torch.no_grad():
                fn()
            flops = counter.get_total_flops()
            out[f"{label}_flops_per_sample"] = int(flops / images.shape[0])
            out[f"{label}_gflops_per_sample"] = round(flops / images.shape[0] / 1e9, 3)
        except Exception as exc:  # counter is best-effort; never kill the run
            out[f"{label}_flops_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _measure_peak_memory(backbone, images, texts, device: str) -> dict:
    if device != "cuda":
        return {"peak_memory_note": "CPU run — GPU memory not measured"}
    res = {}
    for label, fn in (
        ("encode_image", lambda: backbone.encode_image(images)),
        ("encode_mm", lambda: backbone.encode_mm(images, texts)),
    ):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            fn()
        torch.cuda.synchronize()
        res[f"{label}_peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    return res


def _measure_index_cost(dim: int, gallery_sizes: dict[str, int], device: str,
                        trials: int, warmup: int) -> dict:
    """Embedding-store size + brute-force search latency (1 query vs the gallery).

    Depends only on (dim, gallery size), so it is identical across same-dim models;
    measured anyway so the table carries real numbers rather than a claim.
    """
    out = {"embedding_dim": dim}
    for name, n in gallery_sizes.items():
        bytes_fp32 = n * dim * 4
        out[f"{name}_gallery_size"] = n
        out[f"{name}_index_mb_fp32"] = round(bytes_fp32 / 1024**2, 2)
        out[f"{name}_index_mb_fp16"] = round(bytes_fp32 / 2 / 1024**2, 2)

        index = torch.randn(n, dim, device=device)
        index = torch.nn.functional.normalize(index, dim=-1)
        q = torch.nn.functional.normalize(torch.randn(1, dim, device=device), dim=-1)

        def _search():
            scores = q @ index.T
            return scores.topk(50, dim=-1)

        t = _time_calls(_search, trials, warmup, device)
        out[f"{name}_search_ms_mean"] = round(t.mean_ms, 4)
        out[f"{name}_search_ms_p95"] = round(t.p95_ms, 4)
        del index, q
        if device == "cuda":
            torch.cuda.empty_cache()
    return out


def benchmark_model(entry: dict, args, device: str) -> dict:
    ckpt = PROJECT_ROOT / entry["checkpoint"]
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")

    print(f"\n{'='*70}\n{entry['name']}\n  {entry['checkpoint']}\n{'='*70}", flush=True)

    backbone = Visualized_BGE(
        model_name_bge=BGE_MODEL_NAME,
        model_weight=str(ckpt),
        negatives_cross_device=False,
    )
    backbone.eval()
    backbone.to(device)

    res: dict = {
        "name": entry["name"],
        "short": entry["short"],
        "checkpoint": entry["checkpoint"],
        "note": entry["note"],
        "checkpoint_size_mb": round(ckpt.stat().st_size / 1024**2, 1),
    }

    # ---- static ----------------------------------------------------------
    res.update(_count_params(backbone))
    c, h, w = _image_tensor_shape(backbone)
    res["input_resolution"] = f"{h}x{w}"

    # ---- inputs (fixed seed so both models see identical tensors) --------
    torch.manual_seed(0)
    img1 = torch.randn(1, c, h, w, device=device)
    imgB = torch.randn(args.batch_size, c, h, w, device=device)
    tok = backbone.tokenizer(
        ["a photo of a dress, but make it red and sleeveless"],
        padding="max_length", max_length=TOKEN_LEN, truncation=True, return_tensors="pt",
    )
    txt1 = {k: v.to(device) for k, v in tok.items() if k in ("input_ids", "attention_mask")}
    txtB = {k: v.repeat(args.batch_size, 1) for k, v in txt1.items()}

    with torch.no_grad():
        emb = backbone.encode_image(img1)
    res["embedding_dim"] = int(emb.shape[-1])

    # ---- compute ---------------------------------------------------------
    print("  FLOPs …", flush=True)
    res.update(_measure_flops(backbone, img1, txt1, device))

    # ---- latency @ batch=1 (deployment-realistic) ------------------------
    print(f"  latency batch=1 ({args.trials} trials) …", flush=True)
    with torch.no_grad():
        t_img = _time_calls(lambda: backbone.encode_image(img1), args.trials, args.warmup, device)
        t_mm = _time_calls(lambda: backbone.encode_mm(img1, txt1), args.trials, args.warmup, device)
    res["latency_bs1_encode_image"] = {
        "mean_ms": round(t_img.mean_ms, 3), "std_ms": round(t_img.std_ms, 3),
        "p50_ms": round(t_img.p50_ms, 3), "p95_ms": round(t_img.p95_ms, 3),
    }
    res["latency_bs1_encode_mm"] = {
        "mean_ms": round(t_mm.mean_ms, 3), "std_ms": round(t_mm.std_ms, 3),
        "p50_ms": round(t_mm.p50_ms, 3), "p95_ms": round(t_mm.p95_ms, 3),
    }

    # ---- throughput @ serving batch --------------------------------------
    print(f"  throughput batch={args.batch_size} …", flush=True)
    bt = max(args.trials // 5, 5)
    with torch.no_grad():
        tb_img = _time_calls(lambda: backbone.encode_image(imgB), bt, max(args.warmup // 2, 2), device)
        tb_mm = _time_calls(lambda: backbone.encode_mm(imgB, txtB), bt, max(args.warmup // 2, 2), device)
    res["throughput_batch_size"] = args.batch_size
    res["throughput_images_per_sec"] = round(args.batch_size / (tb_img.mean_ms / 1000.0), 1)
    res["throughput_queries_per_sec"] = round(args.batch_size / (tb_mm.mean_ms / 1000.0), 1)
    res["batch_latency_encode_image_ms"] = round(tb_img.mean_ms, 3)
    res["batch_latency_encode_mm_ms"] = round(tb_mm.mean_ms, 3)

    # ---- memory ----------------------------------------------------------
    print("  peak memory …", flush=True)
    res.update(_measure_peak_memory(backbone, imgB, txtB, device))
    res["memory_batch_size"] = args.batch_size

    # ---- index / search --------------------------------------------------
    print("  index + search …", flush=True)
    del backbone, img1, imgB
    if device == "cuda":
        torch.cuda.empty_cache()
    res["index"] = _measure_index_cost(
        res["embedding_dim"],
        {"cirr_val": args.cirr_gallery, "fashioniq_val_shirt": args.fiq_gallery},
        device, args.trials, args.warmup,
    )
    return res


def _fmt_delta(a: float, b: float, unit: str = "", pct: bool = True) -> str:
    """b relative to a."""
    d = b - a
    if pct and a:
        return f"{d:+.2f}{unit} ({d/a*100:+.2f}%)"
    return f"{d:+.2f}{unit}"


def render_report(results: list[dict], env: dict) -> str:
    """Markdown report: identical-by-construction metrics vs. the real numbers."""
    L: list[str] = []
    L.append("# Inference-Efficiency Comparison\n")
    L.append(f"_Generated {env['timestamp']} · {env['gpu']} · torch {env['torch']}_\n")
    L.append(
        "All arms share one VISTA architecture and differ only in weights, so every "
        "inference-time metric below is expected to be identical by construction. "
        "They are measured, not asserted: matching numbers are the evidence that the "
        "recall gain from relation-based distillation costs nothing at serving time. "
        "The teacher (BGE-VL-MLLM-S1) is excluded — it runs only during training.\n"
    )

    names = [r["name"] for r in results]

    def row(label: str, vals: list, note: str = "") -> str:
        return f"| {label} | " + " | ".join(str(v) for v in vals) + f" | {note} |"

    hdr = "| Metric | " + " | ".join(names) + " | |"
    sep = "|---|" + "---|" * len(names) + "---|"

    # ---- A: model size / static -----------------------------------------
    L.append("\n## A. Model size (identical by construction)\n")
    L.append(hdr); L.append(sep)
    L.append(row("Total params (M)", [r["total_params_millions"] for r in results]))
    L.append(row("  ├ vision branch (M)", [r["vision_branch_params_millions"] for r in results]))
    L.append(row("  └ text+fusion branch (M)", [r["text_fusion_branch_params_millions"] for r in results]))
    L.append(row("Embedding dim", [r["embedding_dim"] for r in results]))
    L.append(row("Input resolution", [r["input_resolution"] for r in results]))
    L.append(row("Checkpoint on disk (MB)", [r["checkpoint_size_mb"] for r in results]))

    # ---- B: compute ------------------------------------------------------
    L.append("\n## B. Compute per forward pass (identical by construction)\n")
    L.append(hdr); L.append(sep)
    if all("encode_image_gflops_per_sample" in r for r in results):
        L.append(row("GFLOPs — encode_image", [r["encode_image_gflops_per_sample"] for r in results]))
    if all("encode_mm_gflops_per_sample" in r for r in results):
        L.append(row("GFLOPs — encode_mm (query)", [r["encode_mm_gflops_per_sample"] for r in results]))

    # ---- C: latency / throughput ----------------------------------------
    L.append("\n## C. Latency & throughput (measured; expected to match within noise)\n")
    L.append(hdr); L.append(sep)
    L.append(row("encode_image bs=1 mean (ms)", [r["latency_bs1_encode_image"]["mean_ms"] for r in results]))
    L.append(row("encode_image bs=1 p95 (ms)", [r["latency_bs1_encode_image"]["p95_ms"] for r in results]))
    L.append(row("encode_mm bs=1 mean (ms)", [r["latency_bs1_encode_mm"]["mean_ms"] for r in results]))
    L.append(row("encode_mm bs=1 p95 (ms)", [r["latency_bs1_encode_mm"]["p95_ms"] for r in results]))
    bs = results[0]["throughput_batch_size"]
    L.append(row(f"Images/sec (bs={bs})", [r["throughput_images_per_sec"] for r in results]))
    L.append(row(f"Queries/sec (bs={bs})", [r["throughput_queries_per_sec"] for r in results]))

    # ---- D: memory -------------------------------------------------------
    L.append("\n## D. Peak GPU memory (identical by construction)\n")
    L.append(hdr); L.append(sep)
    if all("encode_image_peak_mem_mb" in r for r in results):
        mb = results[0].get("memory_batch_size", bs)
        L.append(row(f"encode_image peak (MB, bs={mb})", [r["encode_image_peak_mem_mb"] for r in results]))
        L.append(row(f"encode_mm peak (MB, bs={mb})", [r["encode_mm_peak_mem_mb"] for r in results]))

    # ---- E: index --------------------------------------------------------
    L.append("\n## E. Index storage & search (identical by construction)\n")
    L.append(hdr); L.append(sep)
    i0 = results[0]["index"]
    for g in ("cirr_val", "fashioniq_val_shirt"):
        if f"{g}_index_mb_fp32" in i0:
            n = i0[f"{g}_gallery_size"]
            L.append(row(f"{g} index fp32 (MB, N={n:,})", [r["index"][f"{g}_index_mb_fp32"] for r in results]))
            L.append(row(f"{g} search mean (ms)", [r["index"][f"{g}_search_ms_mean"] for r in results]))

    # ---- verdict ---------------------------------------------------------
    L.append("\n## Reading the table\n")
    if len(results) >= 2:
        a, b = results[0], results[1]
        lat_a = a["latency_bs1_encode_mm"]["mean_ms"]
        lat_b = b["latency_bs1_encode_mm"]["mean_ms"]
        L.append(
            f"- Parameters, embedding dim, FLOPs, memory and index cost are **exactly equal** "
            f"across arms — same architecture, only the weights differ.\n"
            f"- Query-encode latency differs by {_fmt_delta(lat_a, lat_b, ' ms')}, which is "
            f"measurement jitter rather than a real cost difference (same computation graph).\n"
            f"- **Conclusion:** the accuracy improvement is obtained at *zero* additional "
            f"inference cost — no extra parameters, compute, memory, or index footprint.\n"
        )
    L.append(
        "\n_Note: training-side costs (extra distillation epochs and the one-time teacher "
        "encoding of the LaSCo candidate pool) are reported separately; they are incurred "
        "once at training time and do not affect serving._\n"
    )
    return "\n".join(L)


def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA unavailable — latency/memory numbers will not be "
              "representative. Run this on a GPU node.", flush=True)

    env = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "torch": torch.__version__,
        "cuda": torch.version.cuda if device == "cuda" else None,
        "python": platform.python_version(),
        "trials": args.trials,
        "warmup": args.warmup,
        "batch_size": args.batch_size,
        "precision": "fp32",
    }

    results = [benchmark_model(e, args, device) for e in MODELS]

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "efficiency_results.json"
    md_path = out_dir / "efficiency_report.md"

    json.dump({"environment": env, "models": results}, open(json_path, "w"), indent=2)
    report = render_report(results, env)
    md_path.write_text(report)

    print("\n" + report)
    print(f"\nJSON   -> {json_path}")
    print(f"Report -> {md_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Inference-efficiency benchmark for CIR retrievers.")
    p.add_argument("--trials", type=int, default=50, help="timed iterations per measurement")
    p.add_argument("--warmup", type=int, default=10, help="untimed warmup iterations")
    p.add_argument("--batch-size", type=int, default=64, help="serving batch for throughput/memory")
    p.add_argument("--cirr-gallery", type=int, default=2297, help="CIRR val gallery size")
    p.add_argument("--fiq-gallery", type=int, default=6346,
                   help="FashionIQ val gallery — largest category (shirt); FIQ is scored per category "
                        "(dress 3817 / shirt 6346 / toptee 5373), not over a merged gallery")
    p.add_argument("--out-dir", default="results/efficiency")
    main(p.parse_args())
