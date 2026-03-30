import argparse
from datetime import datetime, timezone
import importlib
import json
import os
import platform
import sys
from typing import Any

import torch

from src.evaluation.cirr_eval import evaluate_cirr, generate_cirr_test_submission
from src.evaluation.fashioniq_eval import evaluate_fashioniq
from src.retrievers.base import TwoEncoderVLM
from src.utils.io import prepend_key_to_dict, save_records_to_csv, save_to_csv, save_to_json


FASHIONIQ_STUDY_SETTINGS = {"default", "caption_ablation_v1"}


def resolve_fashioniq_study_setting(
    study_setting: str,
    caption_joiner_mode: str,
    caption_order_mode: str,
) -> tuple[str, str]:
    if study_setting not in FASHIONIQ_STUDY_SETTINGS:
        supported = ", ".join(sorted(FASHIONIQ_STUDY_SETTINGS))
        raise ValueError(f"Unsupported fashioniq_study_setting '{study_setting}'. Supported: {supported}")

    if study_setting == "caption_ablation_v1":
        # Preset used for the small ablation study.
        return "and", "both"

    return caption_joiner_mode, caption_order_mode


def resolve_caption_joiner_override(mode: str) -> str | None:
    if mode == "protocol_default":
        return None
    if mode == "space":
        return " "
    if mode == "and":
        return " and "
    raise ValueError(f"Unsupported fashioniq_caption_joiner_mode '{mode}'.")


def describe_fashioniq_text_composition(
    eval_protocol: str,
    caption_joiner_mode: str,
    caption_order_mode: str,
) -> str:
    joiner = resolve_caption_joiner_override(caption_joiner_mode)
    if joiner is None:
        joiner = " and " if eval_protocol == "original_split" else " "

    joiner_name = "space" if joiner == " " else "' and '"
    if caption_order_mode == "both":
        return f"caption1 + {joiner_name} + caption2 and reversed order average"
    return f"caption1 + {joiner_name} + caption2"


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            return torch.device("cuda")
        print("No GPU available, using CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_retriever(
    retriever_module: str,
    retriever_class: str,
    model_name_or_path: str,
    device: torch.device,
    init_kwargs: dict[str, Any],
) -> TwoEncoderVLM:
    try:
        module = importlib.import_module(retriever_module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Could not import retriever module '{retriever_module}'. "
            "Create this module under src/retrievers before running evaluation."
        ) from exc

    if not hasattr(module, retriever_class):
        raise AttributeError(
            f"Retriever class '{retriever_class}' not found in module '{retriever_module}'."
        )

    retriever_cls = getattr(module, retriever_class)

    if hasattr(retriever_cls, "from_pretrained"):
        model = retriever_cls.from_pretrained(model_name_or_path, **init_kwargs)
    else:
        model = retriever_cls(model_name_or_path=model_name_or_path, **init_kwargs)

    if hasattr(model, "to"):
        model = model.to(device)

    return model


def test_model(
    model: TwoEncoderVLM,
    datasets: list[str],
    query_embedding_mode: str = "vista_mm",
    fashioniq_eval_protocol: str = "val_split",
    fashioniq_study_setting: str = "default",
    fashioniq_caption_joiner_mode: str = "protocol_default",
    fashioniq_caption_order_mode: str = "original",
    fusion_type: str = "sum",
    batch_size: int = 64,
    num_workers: int = 4,
    use_tqdm: bool = True,
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    if "fashioniq" in datasets:
        effective_caption_joiner_mode, effective_caption_order_mode = resolve_fashioniq_study_setting(
            fashioniq_study_setting,
            fashioniq_caption_joiner_mode,
            fashioniq_caption_order_mode,
        )

        caption_joiner_override = resolve_caption_joiner_override(
            effective_caption_joiner_mode,
        )

        fashioniq_metrics = evaluate_fashioniq(
            model=model,
            query_embedding_mode=query_embedding_mode,
            eval_protocol=fashioniq_eval_protocol,
            caption_joiner_override=caption_joiner_override,
            caption_order_mode=effective_caption_order_mode,
            fusion_type=fusion_type,
            batch_size=batch_size,
            num_workers=num_workers,
            tqdm=use_tqdm,
            accelerator=None,
        )

        fashioniq_prefix = "original" if fashioniq_eval_protocol == "original_split" else "val"
        if "latency_seconds" in fashioniq_metrics:
            fashioniq_metrics[f"{fashioniq_prefix}_latency_seconds"] = fashioniq_metrics.pop("latency_seconds")

        metrics.update(prepend_key_to_dict("fashioniq_", fashioniq_metrics))

    if "cirr" in datasets:
        cirr_metrics = evaluate_cirr(
            model=model,
            query_embedding_mode=query_embedding_mode,
            fusion_type=fusion_type,
            batch_size=batch_size,
            num_workers=num_workers,
            tqdm=use_tqdm,
            accelerator=None,
        )

        if "latency_seconds" in cirr_metrics:
            cirr_metrics["val_latency_seconds"] = cirr_metrics.pop("latency_seconds")

        metrics.update(prepend_key_to_dict("cirr_", cirr_metrics))

    return metrics


def build_runtime_context(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    requested_device = args.device
    resolved_device = str(device)

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_device_index: int | None = None
    cuda_device_name: str | None = None

    if device.type == "cuda" and cuda_available:
        cuda_device_index = device.index if device.index is not None else int(torch.cuda.current_device())
        cuda_device_name = str(torch.cuda.get_device_name(cuda_device_index))

    effective_caption_joiner_mode, effective_caption_order_mode = resolve_fashioniq_study_setting(
        args.fashioniq_study_setting,
        args.fashioniq_caption_joiner_mode,
        args.fashioniq_caption_order_mode,
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "device": {
            "requested": requested_device,
            "resolved": resolved_device,
            "type": device.type,
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
            "cuda_device_index": cuda_device_index,
            "cuda_device_name": cuda_device_name,
        },
        "evaluation": {
            "datasets": list(args.datasets),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "tqdm": bool(args.tqdm),
            "query_embedding_mode": args.query_embedding_mode,
            "fusion_type": args.fusion_type,
            "fusion_type_effective": args.fusion_type if args.query_embedding_mode == "legacy_fusion" else "not_used_in_vista_mm",
            "fashioniq_eval_protocol": args.fashioniq_eval_protocol,
            "fashioniq_study_setting": args.fashioniq_study_setting,
            "fashioniq_caption_joiner_mode": args.fashioniq_caption_joiner_mode,
            "fashioniq_caption_order_mode": args.fashioniq_caption_order_mode,
            "fashioniq_caption_joiner_mode_effective": effective_caption_joiner_mode,
            "fashioniq_caption_order_mode_effective": effective_caption_order_mode,
            "fashioniq_text_composition_effective": describe_fashioniq_text_composition(
                args.fashioniq_eval_protocol,
                effective_caption_joiner_mode,
                effective_caption_order_mode,
            ),
            "skip_submission": list(args.skip_submission),
            "dataloader_pin_memory": True,
        },
        "retriever": {
            "module": args.retriever_module,
            "class": args.retriever_class,
            "model_name_or_path": args.model_name_or_path,
            "checkpoint_path": args.checkpoint_path,
        },
    }


def build_structured_metrics(
    metrics: dict[str, float],
    query_embedding_mode: str,
    runtime_context: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    runtime_summary = {
        "device": runtime_context["device"]["resolved"],
        "gpu_name": runtime_context["device"]["cuda_device_name"],
        "batch_size": runtime_context["evaluation"]["batch_size"],
        "num_workers": runtime_context["evaluation"]["num_workers"],
        "tqdm": runtime_context["evaluation"]["tqdm"],
    }

    for full_metric_name, value in metrics.items():
        dataset = "unknown"
        split = "unknown"
        metric = full_metric_name
        protocol = {
            "name": "unknown",
            "split": split,
            "candidate_pool": "unknown",
            "reference_removed": True,
            "gallery_category_specific": False,
            "gallery_full_corpus": False,
            "text_composition": "unknown",
            "query_embedding_mode": query_embedding_mode,
            "runtime": runtime_summary,
            "score_type": "dataset_level",
        }

        if full_metric_name.startswith("fashioniq_val_"):
            dataset = "fashioniq"
            split = "val"
            metric = full_metric_name.removeprefix("fashioniq_val_")
            protocol["name"] = "val_split"
            protocol["split"] = split
            protocol["candidate_pool"] = "category-specific validation gallery (dress/shirt/toptee), reference removed"
            protocol["gallery_category_specific"] = True
            protocol["text_composition"] = runtime_context["evaluation"]["fashioniq_text_composition_effective"]

            if metric.endswith("avg_recall_at5") or metric.endswith("avg_recall_at10") or metric.endswith("avg_recall_at50"):
                protocol["score_type"] = "macro_average"
            elif metric == "latency_seconds":
                protocol["score_type"] = "system_latency"

        elif full_metric_name.startswith("fashioniq_original_"):
            dataset = "fashioniq"
            split = "test"
            metric = full_metric_name.removeprefix("fashioniq_original_")
            protocol["name"] = "original_split"
            protocol["split"] = split
            protocol["candidate_pool"] = "category-specific official test gallery (dress/shirt/toptee), reference removed"
            protocol["gallery_category_specific"] = True
            protocol["text_composition"] = runtime_context["evaluation"]["fashioniq_text_composition_effective"]

            if metric.endswith("avg_recall_at5") or metric.endswith("avg_recall_at10") or metric.endswith("avg_recall_at50"):
                protocol["score_type"] = "macro_average"
            elif metric == "latency_seconds":
                protocol["score_type"] = "system_latency"

        elif full_metric_name.startswith("cirr_val_"):
            dataset = "cirr"
            split = "val"
            metric = full_metric_name.removeprefix("cirr_val_")
            protocol["name"] = "benchmark_val"
            protocol["split"] = split
            protocol["text_composition"] = "single-caption"

            if metric.startswith("subset_recall_at"):
                protocol["candidate_pool"] = "query-group subset gallery, reference removed"
                protocol["score_type"] = "dataset_level"
            elif metric.startswith("global_recall_at"):
                protocol["candidate_pool"] = "full validation gallery, reference removed"
                protocol["gallery_full_corpus"] = True
                protocol["score_type"] = "dataset_level"
            elif metric == "summary_average":
                protocol["candidate_pool"] = "combined global(full-corpus) and subset(query-group) validation metrics"
                protocol["score_type"] = "macro_average"
            elif metric == "latency_seconds":
                protocol["candidate_pool"] = "mixed global and subset validation pipelines"
                protocol["score_type"] = "system_latency"

        records.append(
            {
                "dataset": dataset,
                "split": split,
                "metric": metric,
                "value": value,
                "protocol": protocol,
            }
        )

    return records


def build_protocol_validation_summary(
    datasets: list[str],
    query_embedding_mode: str,
    runtime_context: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "validated": True,
        "query_embedding_mode": query_embedding_mode,
        "runtime": runtime_context,
        "query_embedding_modes": {
            "default": "vista_mm",
            "supported": ["vista_mm", "legacy_fusion"],
            "vista_mm": "native VISTA multimodal query encoding (encode_mm)",
            "legacy_fusion": "separate vision/text encoders + manual fusion",
        },
        "datasets": {},
    }

    if "fashioniq" in datasets:
        summary["datasets"]["fashioniq"] = {
            "supported_protocols": {
                "val_split": {
                    "split": "val",
                    "candidate_pool": "category-specific gallery (dress/shirt/toptee)",
                    "reference_removed": True,
                    "text_composition": "protocol_default: caption1 + space + caption2",
                },
                "original_split": {
                    "split": "test",
                    "candidate_pool": "category-specific official test gallery (dress/shirt/toptee)",
                    "reference_removed": True,
                    "text_composition": "protocol_default: caption1 + ' and ' + caption2",
                },
            },
            "active_ablation": {
                "study_setting": runtime_context["evaluation"]["fashioniq_study_setting"],
                "caption_joiner_mode": runtime_context["evaluation"]["fashioniq_caption_joiner_mode"],
                "caption_order_mode": runtime_context["evaluation"]["fashioniq_caption_order_mode"],
                "caption_joiner_mode_effective": runtime_context["evaluation"]["fashioniq_caption_joiner_mode_effective"],
                "caption_order_mode_effective": runtime_context["evaluation"]["fashioniq_caption_order_mode_effective"],
                "text_composition_effective": runtime_context["evaluation"]["fashioniq_text_composition_effective"],
            },
        }

    if "cirr" in datasets:
        summary["datasets"]["cirr"] = {
            "split": "val",
            "global_candidate_pool": "full validation gallery",
            "subset_candidate_pool": "query group members",
            "reference_removed": True,
            "benchmark_metrics": {
                "global": ["R@1", "R@5", "R@10", "R@50"],
                "subset": ["R@1", "R@2", "R@3"],
                "summary": "(global R@5 + subset R@1)/2",
            },
        }

    return summary


def write_ablation_study_note(output_path: str, args: argparse.Namespace, runtime_context: dict[str, Any]) -> None:
    note_lines = [
        "Ablation Study Note",
        "",
        "Running only retriever on these settings:",
        f"- model: {args.model_name_or_path}",
        f"- test sets: {', '.join(args.datasets)}",
        f"- FashionIQ protocol: {args.fashioniq_eval_protocol}",
        f"- encoder type: {args.query_embedding_mode}",
        "",
        "High-level important details:",
        f"- FashionIQ text composition: {runtime_context['evaluation']['fashioniq_text_composition_effective']}",
        f"- fusion_type argument: {args.fusion_type} (effective: {runtime_context['evaluation']['fusion_type_effective']})",
        "- CIRR and FashionIQ both remove reference image from ranking before scoring",
        "- Outputs in this run folder include metrics.csv, metrics_structured.json, evaluation_runtime.json, and evaluation_protocol.json",
    ]
    note_path = os.path.join(output_path, "ablation_study_note.txt")
    with open(note_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(note_lines) + "\n")


def main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    runtime_context = build_runtime_context(args=args, device=device)

    init_kwargs: dict[str, Any] = json.loads(args.retriever_init_kwargs)
    if args.checkpoint_path:
        init_kwargs["checkpoint_path"] = args.checkpoint_path

    model = load_retriever(
        retriever_module=args.retriever_module,
        retriever_class=args.retriever_class,
        model_name_or_path=args.model_name_or_path,
        device=device,
        init_kwargs=init_kwargs,
    )

    run_name = args.run_name or f"{args.retriever_class}-zero-shot-{args.fusion_type}"
    output_path = os.path.join(args.output_path, run_name)
    os.makedirs(output_path, exist_ok=True)

    metrics = test_model(
        model=model,
        datasets=args.datasets,
        query_embedding_mode=args.query_embedding_mode,
        fashioniq_eval_protocol=args.fashioniq_eval_protocol,
        fashioniq_study_setting=args.fashioniq_study_setting,
        fashioniq_caption_joiner_mode=args.fashioniq_caption_joiner_mode,
        fashioniq_caption_order_mode=args.fashioniq_caption_order_mode,
        fusion_type=args.fusion_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_tqdm=args.tqdm,
    )
    save_to_csv(metrics, os.path.join(output_path, "metrics.csv"))

    structured_metrics = build_structured_metrics(
        metrics,
        query_embedding_mode=args.query_embedding_mode,
        runtime_context=runtime_context,
    )
    save_records_to_csv(
        structured_metrics,
        os.path.join(output_path, "metrics_structured.csv"),
        fieldnames=["dataset", "split", "metric", "value", "protocol"],
    )
    save_to_json(structured_metrics, os.path.join(output_path, "metrics_structured.json"))
    save_to_json(runtime_context, os.path.join(output_path, "evaluation_runtime.json"))

    protocol_validation = build_protocol_validation_summary(
        args.datasets,
        query_embedding_mode=args.query_embedding_mode,
        runtime_context=runtime_context,
    )
    save_to_json(protocol_validation, os.path.join(output_path, "evaluation_protocol.json"))

    run_config = vars(args).copy()
    run_config["resolved_device"] = str(device)
    run_config["runtime_metadata_file"] = "evaluation_runtime.json"
    run_config["fusion_type_effective"] = runtime_context["evaluation"]["fusion_type_effective"]
    run_config["fashioniq_text_composition_effective"] = runtime_context["evaluation"]["fashioniq_text_composition_effective"]
    with open(os.path.join(output_path, "run_config.json"), "w", encoding="utf-8") as file_obj:
        json.dump(run_config, file_obj, indent=2)

    write_ablation_study_note(output_path=output_path, args=args, runtime_context=runtime_context)

    if "cirr" in args.datasets and "cirr" not in args.skip_submission:
        cirr_test_sub = generate_cirr_test_submission(
            model=model,
            query_embedding_mode=args.query_embedding_mode,
            fusion_type=args.fusion_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            tqdm=args.tqdm,
        )
        cirr_recall_dict = cirr_test_sub["top_50"]
        cirr_subset_recall_dict = cirr_test_sub["subset_top_3"]
        cirr_recall_dict.update({"version": "rc2", "metric": "recall"})
        cirr_subset_recall_dict.update({"version": "rc2", "metric": "recall_subset"})

        with open(os.path.join(output_path, "cirr_test_submission.json"), "w", encoding="utf-8") as file_obj:
            json.dump(cirr_recall_dict, file_obj)
        with open(os.path.join(output_path, "cirr_subset_test_submission.json"), "w", encoding="utf-8") as file_obj:
            json.dump(cirr_subset_recall_dict, file_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run zero-shot CIR baselines.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Base model name or local path.")
    parser.add_argument("--retriever_module", type=str, default="src.retrievers.vista_retriever", help="Python module containing retriever class.")
    parser.add_argument("--retriever_class", type=str, default="VistaBGERetriever", help="Retriever class name inside retriever module.")
    parser.add_argument("--retriever_init_kwargs", type=str, default="{}", help="Extra retriever kwargs in JSON format.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Retriever checkpoint path. Required unless model_weight is provided in --retriever_init_kwargs.",
    )
    parser.add_argument("--datasets", nargs="+", default=["cirr", "fashioniq"], choices=["cirr", "fashioniq"], help="Datasets to evaluate.")
    parser.add_argument("--skip_submission", nargs="*", default=[], choices=["cirr"], help="Skip test submission generation per dataset.")
    parser.add_argument(
        "--fashioniq_eval_protocol",
        type=str,
        default="val_split",
        choices=["val_split", "original_split"],
        help="FashionIQ evaluation protocol: val_split (internal baseline) or original_split (benchmark-comparable).",
    )
    parser.add_argument(
        "--fashioniq_study_setting",
        type=str,
        default="default",
        choices=["default", "caption_ablation_v1"],
        help="Optional preset study setting. 'default' preserves old behavior; 'caption_ablation_v1' applies and-joiner + both-order averaging.",
    )
    parser.add_argument(
        "--fashioniq_caption_joiner_mode",
        type=str,
        default="protocol_default",
        choices=["protocol_default", "space", "and"],
        help="FashionIQ caption joiner ablation: protocol default, force space, or force 'and'.",
    )
    parser.add_argument(
        "--fashioniq_caption_order_mode",
        type=str,
        default="original",
        choices=["original", "both"],
        help="FashionIQ caption order ablation: use original caption order only, or average original+reversed order.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument(
        "--query_embedding_mode",
        type=str,
        default="vista_mm",
        choices=["vista_mm", "legacy_fusion"],
        help="Query embedding path: vista_mm (native multimodal encoding) or legacy_fusion (manual image-text fusion).",
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        default="sum",
        help="Image-text fusion strategy used only when --query_embedding_mode=legacy_fusion.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--tqdm", action="store_true", help="Enable progress bars.")
    parser.add_argument("--output_path", type=str, default="results", help="Directory where outputs are saved.")
    parser.add_argument("--run_name", type=str, default="", help="Optional run folder name override.")

    main(parser.parse_args())


