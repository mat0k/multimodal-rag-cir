"""Measure GFLOPs for the BGE-VL-MLLM-S1 teacher, in isolation.

Split out of benchmark_student_vs_teacher.py deliberately. Counting FLOPs needs an
extra forward pass, and on a 7B model that pass leaves the GPU too full for anything
that follows -- it OOM-ed the latency measurement across three separate runs. Here
nothing follows it: the process measures, writes, and exits, so the allocator never
has to give the memory back.

    python scripts/benchmark_teacher_flops.py --out results/efficiency/student_vs_teacher/teacher_flops.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROMPT = "a photo of a dress, but make it red and sleeveless"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="models/BGE-VL-MLLM-S1")
    ap.add_argument("--out", default="results/efficiency/student_vs_teacher/teacher_flops.json")
    args = ap.parse_args()

    from torch.utils.flop_counter import FlopCounterMode

    from src.retrievers.bge_vl_retriever import BGEVLRetriever

    device = "cuda" if torch.cuda.is_available() else "cpu"
    retriever = BGEVLRetriever(str(PROJECT_ROOT / args.model_path), device=device)
    images = [Image.new("RGB", (448, 448), (128, 128, 128))]

    out = {"model": "BGE-VL-MLLM-S1", "device": device,
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}

    # One path at a time, freeing in between: even isolated, two 7B forwards back to
    # back can exhaust the partition.
    for label, fn in (("encode_image", lambda: retriever.encode_images(images)),
                      ("encode_mm", lambda: retriever.encode_composed(images, [PROMPT]))):
        try:
            counter = FlopCounterMode(display=False)
            with counter, torch.no_grad():
                fn()
            out[f"{label}_gflops_per_sample"] = round(counter.get_total_flops() / 1e9, 3)
            print(f"{label}: {out[f'{label}_gflops_per_sample']} GFLOPs", flush=True)
        except Exception as exc:
            out[f"{label}_flops_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            print(f"{label}: FAILED {type(exc).__name__}", flush=True)
        finally:
            if device == "cuda":
                torch.cuda.empty_cache()

    dest = PROJECT_ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
