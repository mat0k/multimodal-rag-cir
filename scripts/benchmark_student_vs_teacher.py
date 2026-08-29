"""Inference-cost comparison: deployable students vs. the distillation teacher.

Motivation figure for the paper. Distillation is worth doing only if the teacher is
genuinely too expensive to serve, so this quantifies that gap on one GPU under one
protocol: VISTA and MagicLens (the students) against BGE-VL-MLLM-S1 (the teacher).

Deliberately excludes compression/quantisation — those compare two weightings of one
architecture and are covered by `benchmark_compression.py`. Here the architectures
differ, so the interesting axes are size, compute, latency, memory, and index cost.

    python scripts/benchmark_student_vs_teacher.py --out-dir results/efficiency/student_vs_teacher

## A note on fairness

The three models do not share an input contract. VISTA and MagicLens take pre-processed
tensors; BGE-VL takes PIL images and raw strings and pre-processes internally. Timing each
model end-to-end therefore charges BGE-VL for pre-processing that the students are not
charged for. We measure it that way regardless, for two reasons: it is what a deployment
actually pays, and the gap is large enough (7B vs ~0.2B parameters) that pre-processing is
far below the noise floor. The asymmetry is recorded in the report rather than hidden.

FLOPs are omitted for BGE-VL: the counter cannot trace through its custom `trust_remote_code`
generation path. Parameter count and measured latency cover the same ground.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOKEN_LEN = 77
PROMPT = "a photo of a dress, but make it red and sleeveless"

MODELS: list[dict] = [
    {
        "name": "VISTA (student)",
        "short": "vista_zeroshot",
        "type": "vista",
        "checkpoint": "models/Visualized_BGE/Visualized_base_en_v1.5.pth",
        "role": "student",
        "note": "Off-the-shelf Visualized-BGE: BGE-base-en-v1.5 + EVA02-CLIP-B-16.",
    },
    {
        "name": "MagicLens-B (student)",
        "short": "magiclens_base_zeroshot",
        "type": "magiclens",
        "checkpoint": "models/magiclens/magic_lens_clip_base.pt",
        "model_size": "base",
        "role": "student",
        "note": "Our PyTorch port of MagicLens-B (CLIP-B/16), zero-shot.",
    },
    {
        "name": "BGE-VL-MLLM-S1 (teacher)",
        "short": "bge_vl_mllm_s1",
        "type": "bge_vl",
        "checkpoint": "models/BGE-VL-MLLM-S1",
        "role": "teacher",
        "note": "The distillation teacher. Training-time only; never served.",
    },
]


def _count_flops(paths: dict, allow_grad: bool = True) -> dict:
    """GFLOPs per sample for each encode path.

    Deliberately NOT run under `torch.no_grad()`: FlopCounterMode attributes ops via
    autograd, and without a graph it raises "Expected gradient function to be set".
    VISTA happened to tolerate it, MagicLens did not — so grad stays enabled here and
    memory/latency measurements keep their own `no_grad`.
    """
    from torch.utils.flop_counter import FlopCounterMode

    out = {}
    for label, make in paths.items():
        result = None
        # `allow_grad=False` for very large models. A grad-enabled forward on the 7B
        # teacher fits, but leaves the GPU so full that the latency measurement which
        # follows it OOMs -- the FLOPs pass poisons the rest of the run. Students are
        # small enough that grad is free, and they are the ones whose FLOPs we need.
        modes = (True, False) if allow_grad else (False,)
        for grad_enabled in modes:
            fn = counter = None
            try:
                fn = make(1)
                counter = FlopCounterMode(display=False)
                if grad_enabled:
                    with counter:
                        fn()
                else:
                    with counter, torch.no_grad():
                        fn()
                result = round(counter.get_total_flops() / 1e9, 3)
            except Exception as exc:
                out[f"{label}_flops_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
            finally:
                # Release the graph and activations before returning. Without this
                # the FLOPs pass starves the latency/memory measurements that follow
                # it, which is what OOM-ed the 7B teacher on the previous attempt.
                del fn, counter
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if result is not None:
                break
        if result is not None:
            out[f"{label}_gflops_per_sample"] = result
            out.pop(f"{label}_flops_error", None)
    return out


@dataclass
class Timing:
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _time_calls(fn, trials: int, warmup: int, device: str) -> Timing:
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return Timing(
        mean_ms=statistics.mean(samples),
        std_ms=statistics.stdev(samples) if len(samples) > 1 else 0.0,
        p50_ms=samples[len(samples) // 2],
        p95_ms=samples[min(int(len(samples) * 0.95), len(samples) - 1)],
    )


# --------------------------------------------------------------------------
# Adapters: one uniform surface over three different input contracts.
# Each exposes encode_image_fn / encode_mm_fn returning zero-argument callables,
# so the timing code never has to know what a given model eats.
# --------------------------------------------------------------------------
class _StudentAdapter:
    """VISTA and MagicLens: pre-processed tensors in, embeddings out."""

    def __init__(self, backbone, device: str, vision_module):
        self.backbone = backbone
        self.device = device
        self._vision = vision_module
        dummy = Image.new("RGB", (640, 480), (128, 128, 128))
        self.image_shape = tuple(backbone.preprocess_val(dummy).shape)  # (C, H, W)

    def params(self) -> dict:
        total = sum(p.numel() for p in self.backbone.parameters())
        vision = sum(p.numel() for p in self._vision.parameters()) if self._vision else 0
        return {"total": total, "vision": vision, "other": total - vision}

    def _inputs(self, batch: int):
        c, h, w = self.image_shape
        images = torch.randn(batch, c, h, w, device=self.device)
        tok = self.backbone.tokenizer([PROMPT] * batch, padding="max_length",
                                      max_length=TOKEN_LEN, truncation=True,
                                      return_tensors="pt")
        texts = {k: v.to(self.device) for k, v in tok.items()
                 if k in ("input_ids", "attention_mask")}
        return images, texts

    def encode_image_fn(self, batch: int):
        images, _ = self._inputs(batch)
        return lambda: self.backbone.encode_image(images)

    def encode_mm_fn(self, batch: int):
        images, texts = self._inputs(batch)
        return lambda: self.backbone.encode_mm(images, texts)

    def embedding_dim(self) -> int:
        images, _ = self._inputs(1)
        with torch.no_grad():
            return int(self.backbone.encode_image(images).shape[-1])

    def flops(self, device: str) -> dict:
        return _count_flops({"encode_image": self.encode_image_fn,
                             "encode_mm": self.encode_mm_fn})


class _BGEVLAdapter:
    """BGE-VL: PIL images and raw strings in, embeddings out."""

    def __init__(self, retriever, device: str):
        self.retriever = retriever
        self.device = device
        self.image_shape = (3, 224, 224)  # nominal; it resizes internally

    def params(self) -> dict:
        total = sum(p.numel() for p in self.retriever.model.parameters())
        return {"total": total, "vision": 0, "other": total}

    @staticmethod
    def _pil(batch: int):
        return [Image.new("RGB", (448, 448), (128, 128, 128)) for _ in range(batch)]

    def encode_image_fn(self, batch: int):
        images = self._pil(batch)
        return lambda: self.retriever.encode_images(images)

    def encode_mm_fn(self, batch: int):
        images = self._pil(batch)
        texts = [PROMPT] * batch
        return lambda: self.retriever.encode_composed(images, texts)

    def embedding_dim(self) -> int:
        with torch.no_grad():
            return int(self.retriever.encode_images(self._pil(1)).shape[-1])

    def flops(self, device: str) -> dict:
        # Not attempted at all. Counting FLOPs on a 7B model needs an extra forward
        # whose allocations the caching allocator does not fully return, and the
        # latency measurement that follows then OOMs on a 40 GB partition -- observed
        # across three separate runs, with and without grad enabled. Parameter count
        # (7,566M vs 166M) and measured latency cover the same ground, so this metric
        # is dropped for the teacher rather than allowed to break the whole benchmark.
        return {"flops_note": "not measured — counting on a 7B model exhausts the GPU "
                              "and breaks the subsequent latency measurement"}


def build_adapter(entry: dict, device: str):
    kind = entry["type"]
    ckpt = PROJECT_ROOT / entry["checkpoint"]
    if not ckpt.exists():
        raise FileNotFoundError(f"missing: {ckpt}")

    if kind == "vista":
        from src.retrievers.backbones.vista.modeling import Visualized_BGE

        backbone = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5",
                                  model_weight=str(ckpt), negatives_cross_device=False)
        backbone.eval().to(device)
        return _StudentAdapter(backbone, device, backbone.model_visual)

    if kind == "magiclens":
        from src.retrievers.magiclens_retriever import MagicLensRetriever

        model = MagicLensRetriever.from_pretrained(entry.get("model_size", "base"),
                                                   checkpoint_path=str(ckpt))
        model.backbone.eval().to(device)
        return _StudentAdapter(model.backbone, device, model.backbone.clip.visual)

    if kind == "bge_vl":
        from src.retrievers.bge_vl_retriever import BGEVLRetriever

        return _BGEVLAdapter(BGEVLRetriever(str(ckpt), device=device), device)

    raise ValueError(f"unknown type {kind!r}")


def _dir_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1024**2
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2


def benchmark(entry: dict, args, device: str) -> dict:
    print(f"\n{'='*70}\n{entry['name']}\n  {entry['checkpoint']}\n{'='*70}", flush=True)
    adapter = build_adapter(entry, device)

    res = {k: entry[k] for k in ("name", "short", "type", "role", "note", "checkpoint")}
    res["checkpoint_size_mb"] = round(_dir_size_mb(PROJECT_ROOT / entry["checkpoint"]), 1)

    p = adapter.params()
    res["total_params"] = p["total"]
    res["total_params_millions"] = round(p["total"] / 1e6, 2)
    if p["vision"]:
        res["vision_branch_params_millions"] = round(p["vision"] / 1e6, 2)
        res["text_fusion_branch_params_millions"] = round(p["other"] / 1e6, 2)

    with torch.no_grad():
        res["embedding_dim"] = adapter.embedding_dim()
    res["input_resolution"] = f"{adapter.image_shape[1]}x{adapter.image_shape[2]}"

    print("  FLOPs …", flush=True)
    res.update(adapter.flops(device))

    print(f"  latency batch=1 ({args.trials} trials) …", flush=True)
    with torch.no_grad():
        ti = _time_calls(adapter.encode_image_fn(1), args.trials, args.warmup, device)
        tm = _time_calls(adapter.encode_mm_fn(1), args.trials, args.warmup, device)
    res["latency_bs1_encode_image"] = {"mean_ms": round(ti.mean_ms, 3), "p95_ms": round(ti.p95_ms, 3)}
    res["latency_bs1_encode_mm"] = {"mean_ms": round(tm.mean_ms, 3), "p95_ms": round(tm.p95_ms, 3)}

    print(f"  throughput batch={args.batch_size} …", flush=True)
    bt = max(args.trials // 5, 5)
    with torch.no_grad():
        bi = _time_calls(adapter.encode_image_fn(args.batch_size), bt, max(args.warmup // 2, 2), device)
        bm = _time_calls(adapter.encode_mm_fn(args.batch_size), bt, max(args.warmup // 2, 2), device)
    res["throughput_batch_size"] = args.batch_size
    res["throughput_images_per_sec"] = round(args.batch_size / (bi.mean_ms / 1000.0), 2)
    res["throughput_queries_per_sec"] = round(args.batch_size / (bm.mean_ms / 1000.0), 2)

    if device == "cuda":
        print("  peak memory …", flush=True)
        for label, make in (("encode_image", adapter.encode_image_fn),
                            ("encode_mm", adapter.encode_mm_fn)):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                make(args.batch_size)()
            torch.cuda.synchronize()
            res[f"{label}_peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

    # Index cost: what it takes to store and search an encoded gallery.
    dim = res["embedding_dim"]
    res["index"] = {}
    for name, n in (("cirr_val", args.cirr_gallery), ("fashioniq_val", args.fiq_gallery)):
        res["index"][name] = {
            "vectors": n,
            "fp32_index_mb": round(n * dim * 4 / 1024**2, 2),
        }
    # Wall-clock to encode each gallery once, at the measured throughput.
    ips = res["throughput_images_per_sec"]
    for name, n in (("cirr_val", args.cirr_gallery), ("fashioniq_val", args.fiq_gallery)):
        res["index"][name]["encode_gallery_seconds"] = round(n / ips, 1)

    del adapter
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


def render_report(results: list[dict], env: dict) -> str:
    students = [r for r in results if r["role"] == "student"]
    teachers = [r for r in results if r["role"] == "teacher"]
    cols = results
    head = "| Metric | " + " | ".join(r["name"] for r in cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"

    L = [
        "# Students vs. Teacher — Inference Cost",
        "",
        f"_Generated {env['generated']} · {env['gpu']} · torch {env['torch']}_",
        "",
        "Motivation for distillation: the teacher is the accuracy target, but it is not "
        "servable. This quantifies the gap on one GPU under one protocol.",
        "",
        "> **Measurement asymmetry, stated up front.** VISTA and MagicLens receive "
        "pre-processed tensors; BGE-VL receives PIL images and pre-processes internally, so "
        "its timings include work the students are not charged for. The parameter gap is "
        "~30x, far larger than that overhead, but the numbers are not perfectly like-for-like.",
        "",
        "## A. Model size",
        "", head, sep,
        "| Total params (M) | " + " | ".join(f"{r['total_params_millions']:,}" for r in cols) + " |",
        "| Embedding dim | " + " | ".join(str(r["embedding_dim"]) for r in cols) + " |",
        "| Checkpoint on disk (MB) | " + " | ".join(f"{r['checkpoint_size_mb']:,}" for r in cols) + " |",
        "| Input resolution | " + " | ".join(r["input_resolution"] for r in cols) + " |",
        "",
        "## B. Compute per forward pass",
        "", head, sep,
        "| GFLOPs — encode_image | " + " | ".join(
            str(r.get("encode_image_gflops_per_sample", "n/a")) for r in cols) + " |",
        "| GFLOPs — encode_mm | " + " | ".join(
            str(r.get("encode_mm_gflops_per_sample", "n/a")) for r in cols) + " |",
        "",
        "## C. Latency and throughput",
        "", head, sep,
        "| encode_image bs=1 mean (ms) | " + " | ".join(
            str(r["latency_bs1_encode_image"]["mean_ms"]) for r in cols) + " |",
        "| encode_mm bs=1 mean (ms) | " + " | ".join(
            str(r["latency_bs1_encode_mm"]["mean_ms"]) for r in cols) + " |",
        "| Images/sec (bs=%d) | " % results[0]["throughput_batch_size"] + " | ".join(
            str(r["throughput_images_per_sec"]) for r in cols) + " |",
        "| Queries/sec (bs=%d) | " % results[0]["throughput_batch_size"] + " | ".join(
            str(r["throughput_queries_per_sec"]) for r in cols) + " |",
        "",
        "## D. Peak GPU memory",
        "", head, sep,
        "| encode_image peak (MB) | " + " | ".join(
            str(r.get("encode_image_peak_mem_mb", "n/a")) for r in cols) + " |",
        "| encode_mm peak (MB) | " + " | ".join(
            str(r.get("encode_mm_peak_mem_mb", "n/a")) for r in cols) + " |",
        "",
        "## E. Index storage and gallery-encoding time",
        "", head, sep,
    ]
    for gal, label in (("cirr_val", "CIRR val"), ("fashioniq_val", "FashionIQ val")):
        n = results[0]["index"][gal]["vectors"]
        L.append(f"| {label} index, fp32 (MB) — {n:,} vectors | " + " | ".join(
            str(r["index"][gal]["fp32_index_mb"]) for r in cols) + " |")
        L.append(f"| {label} encode time (s) | " + " | ".join(
            str(r["index"][gal]["encode_gallery_seconds"]) for r in cols) + " |")

    if students and teachers:
        t = teachers[0]
        L += ["", "## F. The gap that motivates distillation", ""]
        L.append("| Student | × smaller (params) | × faster (queries/sec) | × smaller index |")
        L.append("|---|---|---|---|")
        for s in students:
            L.append(
                f"| {s['name']} | "
                f"{t['total_params'] / s['total_params']:.1f}× | "
                f"{s['throughput_queries_per_sec'] / t['throughput_queries_per_sec']:.1f}× | "
                f"{t['embedding_dim'] / s['embedding_dim']:.1f}× |")
        L += ["",
              "The teacher is used only during training — its scores and embeddings are "
              "precomputed offline. Distillation transfers what it knows into a model that "
              "costs a fraction as much to serve.",
              ""]
    return "\n".join(L)


def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA — timings will not be meaningful.", flush=True)

    selected = MODELS
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        selected = [e for e in MODELS if e["short"] in wanted]
        missing = wanted - {e["short"] for e in selected}
        if missing:
            raise SystemExit(f"unknown: {sorted(missing)}; have {[e['short'] for e in MODELS]}")

    results = [benchmark(e, args, device) for e in selected]

    env = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "trials": args.trials,
        "batch_size": args.batch_size,
    }
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"environment": env, "models": results},
              open(out_dir / "student_vs_teacher_results.json", "w"), indent=2)
    (out_dir / "student_vs_teacher_report.md").write_text(render_report(results, env))
    print(f"\nWrote {out_dir}/student_vs_teacher_report.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16,
                   help="serving batch; kept modest so the 7B teacher fits alongside")
    p.add_argument("--cirr-gallery", type=int, default=2297)
    p.add_argument("--fiq-gallery", type=int, default=6346)
    p.add_argument("--out-dir", default="results/efficiency/student_vs_teacher")
    p.add_argument("--only", default=None, help="comma-separated model short names")
    main(p.parse_args())
