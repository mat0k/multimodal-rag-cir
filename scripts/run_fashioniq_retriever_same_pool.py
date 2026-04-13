import argparse
from datetime import datetime, timezone
import importlib
import json
import os
import platform
import random
import sys
import time
from typing import Any, Sequence

import torch

from src.datasets.fashioniq import build_fashioniq_dataset
from src.evaluation.fashioniq_eval import (
    generate_fashioniq_index_features,
    generate_fashioniq_predicted_features,
)
from src.evaluation.fashioniq_rerank_eval import FashionIQStandaloneQuery, build_sampled_candidate_pool
from src.retrievers.base import TwoEncoderVLM
from src.utils.io import prepend_key_to_dict, save_records_to_csv, save_to_csv, save_to_json


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


def resolve_caption_joiner(eval_protocol: str) -> str:
    if eval_protocol == "original_split":
        return " and "
    return " "


def resolve_k_values(k_values: Sequence[int]) -> tuple[int, ...]:
    resolved = set(int(k) for k in k_values)
    resolved.update({1, 5, 10})
    return tuple(sorted(resolved))


def evaluate_fashioniq_retriever_same_pool(
    model: TwoEncoderVLM,
    split: str,
    eval_protocol: str,
    query_embedding_mode: str,
    fusion_type: str,
    num_random_distractors: int,
    seed: int,
    k_values: Sequence[int],
    batch_size: int,
    num_workers: int,
    use_tqdm: bool,
    max_queries: int | None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if split == "test":
        raise ValueError("This fair-pool evaluation requires split with targets. Use split='val' or 'train'.")

    metric_prefix = "original" if eval_protocol == "original_split" else "val"
    caption_joiner = resolve_caption_joiner(eval_protocol)
    resolved_k_values = resolve_k_values(k_values)

    eval_start_time = time.perf_counter()

    index_dataset = build_fashioniq_dataset(
        split=split,
        mode="images",
        image_transform=model.image_processor,
        caption_transform=model.tokenizer,
        max_length_tokenizer=77,
        caption_joiner=caption_joiner,
    )
    triplet_dataset = build_fashioniq_dataset(
        split=split,
        mode="triplets",
        image_transform=model.image_processor,
        caption_transform=model.tokenizer,
        max_length_tokenizer=77,
        caption_joiner=caption_joiner,
    )

    index_features, index_names, index_classes = generate_fashioniq_index_features(
        clip_model=model,
        index_dataset=index_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tqdm=use_tqdm,
        accelerator=None,
    )

    predicted_features, reference_names, target_names, triplet_classes = generate_fashioniq_predicted_features(
        clip_model=model,
        triplet_dataset=triplet_dataset,
        query_embedding_mode=query_embedding_mode,
        fusion_type=fusion_type,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tqdm=use_tqdm,
        accelerator=None,
    )

    if any(target_name is None for target_name in target_names):
        raise ValueError("Found missing target_name in query set; cannot compute recall metrics.")

    class_to_global_indices: dict[str, list[int]] = {cls: [] for cls in index_dataset.classes}
    for global_idx, cls in enumerate(index_classes):
        class_to_global_indices[cls].append(global_idx)

    class_to_features: dict[str, torch.Tensor] = {}
    class_to_name_to_local_idx: dict[str, dict[str, int]] = {}
    for cls in index_dataset.classes:
        global_indices = class_to_global_indices[cls]
        class_to_features[cls] = index_features[global_indices]
        class_names = [index_names[idx] for idx in global_indices]
        class_to_name_to_local_idx[cls] = {name: local_idx for local_idx, name in enumerate(class_names)}

    class_to_candidate_ids = {cls: list(index_dataset.images[cls]) for cls in index_dataset.classes}

    total_queries = len(triplet_classes)
    if max_queries is not None:
        total_queries = min(total_queries, max_queries)

    rng = random.Random(seed)

    hits: dict[str, dict[int, int]] = {cls: {k: 0 for k in resolved_k_values} for cls in index_dataset.classes}
    eligible: dict[str, dict[int, int]] = {cls: {k: 0 for k in resolved_k_values} for cls in index_dataset.classes}

    pool_sizes: list[int] = []
    class_query_counts: dict[str, int] = {cls: 0 for cls in index_dataset.classes}
    class_scored_pairs: dict[str, int] = {cls: 0 for cls in index_dataset.classes}
    class_latency_seconds: dict[str, float] = {cls: 0.0 for cls in index_dataset.classes}
    total_scored_pairs = 0

    pool_records: list[dict[str, Any]] = []

    for i in range(total_queries):
        query_start_time = time.perf_counter()

        cls = str(triplet_classes[i])
        reference_name = str(reference_names[i])
        target_name = str(target_names[i])

        pool_query = FashionIQStandaloneQuery(
            query_class=cls,
            reference_name=reference_name,
            target_name=target_name,
            text_edit="",
            reference_image=None,
        )

        candidate_ids = build_sampled_candidate_pool(
            query=pool_query,
            class_to_candidate_ids=class_to_candidate_ids,
            rng=rng,
            num_random_distractors=num_random_distractors,
            include_target=True,
            remove_reference=True,
        )

        if target_name not in candidate_ids:
            raise ValueError(f"Target '{target_name}' missing from sampled pool for class '{cls}'.")

        name_to_local_idx = class_to_name_to_local_idx[cls]
        local_candidate_indices = [name_to_local_idx[candidate_id] for candidate_id in candidate_ids]
        candidate_features = class_to_features[cls][local_candidate_indices]

        query_feature = predicted_features[i]
        similarity_scores = torch.matmul(candidate_features, query_feature)
        sorted_indices = torch.argsort(similarity_scores, dim=0, descending=True).cpu().tolist()
        ranked_ids = [candidate_ids[idx] for idx in sorted_indices]

        query_elapsed = time.perf_counter() - query_start_time

        pool_sizes.append(len(ranked_ids))
        total_scored_pairs += len(ranked_ids)
        class_query_counts[cls] += 1
        class_scored_pairs[cls] += len(ranked_ids)
        class_latency_seconds[cls] += query_elapsed

        for k in resolved_k_values:
            if len(ranked_ids) < k:
                continue
            eligible[cls][k] += 1
            if target_name in ranked_ids[:k]:
                hits[cls][k] += 1

        pool_records.append(
            {
                "query_index": i,
                "class": cls,
                "reference_name": reference_name,
                "target_name": target_name,
                "candidate_pool": candidate_ids,
                "pool_size": len(candidate_ids),
            }
        )

    elapsed_seconds = time.perf_counter() - eval_start_time
    scored_queries = len(pool_sizes)

    metrics: dict[str, float] = {
        "num_queries": float(total_queries),
        "avg_candidate_pool_size": float(sum(pool_sizes) / max(1, len(pool_sizes))),
        "num_random_distractors": float(num_random_distractors),
        "scored_pairs": float(total_scored_pairs),
        "latency_seconds": float(elapsed_seconds),
        "latency_seconds_per_query": float(elapsed_seconds / max(1, scored_queries)),
        "latency_seconds_per_scored_pair": float(elapsed_seconds / max(1, total_scored_pairs)),
    }

    for cls in index_dataset.classes:
        metrics[f"{metric_prefix}_{cls}_latency_seconds"] = float(class_latency_seconds[cls])
        metrics[f"{metric_prefix}_{cls}_latency_seconds_per_query"] = float(
            class_latency_seconds[cls] / max(1, class_query_counts[cls])
        )
        metrics[f"{metric_prefix}_{cls}_scored_pairs"] = float(class_scored_pairs[cls])

        for k in resolved_k_values:
            metric_name = f"{metric_prefix}_{cls}_recall_at{k}"
            if eligible[cls][k] == 0:
                metrics[metric_name] = float("nan")
            else:
                metrics[metric_name] = float((hits[cls][k] / eligible[cls][k]) * 100.0)

    for k in resolved_k_values:
        class_metric_names = [f"{metric_prefix}_{cls}_recall_at{k}" for cls in index_dataset.classes]
        valid_values = [metrics[name] for name in class_metric_names if metrics[name] == metrics[name]]
        if valid_values:
            metrics[f"{metric_prefix}_avg_recall_at{k}"] = float(sum(valid_values) / len(valid_values))
        else:
            metrics[f"{metric_prefix}_avg_recall_at{k}"] = float("nan")

    return metrics, pool_records


def build_prefixed_metrics(metrics: dict[str, float], eval_protocol: str) -> dict[str, float]:
    protocol_prefix = "original" if eval_protocol == "original_split" else "val"

    latency_keys = {
        "latency_seconds",
        "latency_seconds_per_query",
        "latency_seconds_per_scored_pair",
    }
    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        if key in latency_keys:
            normalized[f"{protocol_prefix}_{key}"] = value
        else:
            normalized[key] = value

    return prepend_key_to_dict("fashioniq_", normalized)


def build_runtime_context(args: argparse.Namespace, device: torch.device, run_output_dir: str) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_device_index: int | None = None
    cuda_device_name: str | None = None

    if device.type == "cuda" and cuda_available:
        cuda_device_index = device.index if device.index is not None else int(torch.cuda.current_device())
        cuda_device_name = str(torch.cuda.get_device_name(cuda_device_index))

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
            "requested": args.device,
            "resolved": str(device),
            "type": device.type,
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
            "cuda_device_index": cuda_device_index,
            "cuda_device_name": cuda_device_name,
        },
        "run": {
            "dataset": "fashioniq",
            "config_path": "",
            "output_dir": run_output_dir,
            "script": "scripts/run_fashioniq_retriever_same_pool.py",
        },
        "retriever": {
            "module": args.retriever_module,
            "class": args.retriever_class,
            "model_name_or_path": args.model_name_or_path,
            "checkpoint_path": args.checkpoint_path,
            "query_embedding_mode": args.query_embedding_mode,
            "fusion_type": args.fusion_type,
            "retriever_init_kwargs": args.retriever_init_kwargs,
        },
        "candidate_pool": {
            "mode": "sampled_target_plus_random_negatives",
            "split": args.split,
            "eval_protocol": args.eval_protocol,
            "num_random_negatives": args.num_random_distractors,
            "configured_pool_size": args.num_random_distractors + 1,
            "seed": args.seed,
            "k_values": list(args.k_values),
        },
    }


def build_structured_metrics(
    prefixed_metrics: dict[str, float],
    runtime_context: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    runtime_summary = {
        "device": runtime_context["device"]["resolved"],
        "gpu_name": runtime_context["device"]["cuda_device_name"],
        "query_embedding_mode": runtime_context["retriever"]["query_embedding_mode"],
        "fusion_type": runtime_context["retriever"]["fusion_type"],
    }

    for full_metric_name, value in prefixed_metrics.items():
        metric = full_metric_name.removeprefix("fashioniq_") if full_metric_name.startswith("fashioniq_") else full_metric_name
        split = runtime_context["candidate_pool"]["split"]

        protocol = {
            "name": "fair_same_pool_retriever",
            "split": split,
            "candidate_pool": "class-specific sampled pool from split gallery, reference removed",
            "reference_removed": True,
            "runtime": runtime_summary,
            "score_type": "dataset_level",
        }

        if "avg_recall_at" in metric:
            protocol["score_type"] = "macro_average"
        elif metric.endswith("latency_seconds"):
            protocol["score_type"] = "system_latency"
        elif metric.endswith("latency_seconds_per_query") or metric.endswith("latency_seconds_per_scored_pair"):
            protocol["score_type"] = "efficiency"
        elif metric.endswith("scored_pairs") or metric in {
            "num_queries",
            "avg_candidate_pool_size",
            "num_random_distractors",
            "scored_pairs",
        }:
            protocol["score_type"] = "run_stat"

        records.append(
            {
                "dataset": "fashioniq",
                "split": split,
                "metric": metric,
                "value": value,
                "protocol": protocol,
            }
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FashionIQ retriever evaluation on the exact same sampled pool protocol as standalone LamRA reranker."
    )
    parser.add_argument("--model_name_or_path", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--retriever_module", type=str, default="src.retrievers.vista_retriever")
    parser.add_argument("--retriever_class", type=str, default="VistaBGERetriever")
    parser.add_argument("--retriever_init_kwargs", type=str, default="{}")
    parser.add_argument("--checkpoint_path", type=str, default="models/Visualized_BGE/Visualized_base_en_v1.5.pth")

    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--eval_protocol", type=str, default="val_split", choices=["val_split", "original_split"])
    parser.add_argument("--num_random_distractors", type=int, default=31)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 50])
    parser.add_argument("--max_queries", type=int, default=0, help="0 means full query set.")

    parser.add_argument("--query_embedding_mode", type=str, default="vista_mm", choices=["vista_mm", "legacy_fusion"])
    parser.add_argument("--fusion_type", type=str, default="sum")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tqdm", action="store_true")

    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--save_pool_records", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)

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

    max_queries = None if args.max_queries <= 0 else int(args.max_queries)

    metrics_raw, pool_records = evaluate_fashioniq_retriever_same_pool(
        model=model,
        split=args.split,
        eval_protocol=args.eval_protocol,
        query_embedding_mode=args.query_embedding_mode,
        fusion_type=args.fusion_type,
        num_random_distractors=args.num_random_distractors,
        seed=args.seed,
        k_values=tuple(args.k_values),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_tqdm=args.tqdm,
        max_queries=max_queries,
    )

    metrics = build_prefixed_metrics(metrics_raw, eval_protocol=args.eval_protocol)

    run_name = args.run_name or (
        f"{args.retriever_class}-fashioniq-same-pool-"
        f"n{args.num_random_distractors + 1}-seed{args.seed}"
    )
    output_dir = os.path.join(args.output_path, run_name)
    os.makedirs(output_dir, exist_ok=True)

    runtime_context = build_runtime_context(args=args, device=device, run_output_dir=output_dir)
    structured_metrics = build_structured_metrics(prefixed_metrics=metrics, runtime_context=runtime_context)

    save_to_csv(metrics, os.path.join(output_dir, "metrics.csv"))
    save_to_json(metrics, os.path.join(output_dir, "metrics.json"))
    save_to_json(metrics_raw, os.path.join(output_dir, "metrics_raw.json"))
    save_records_to_csv(
        structured_metrics,
        os.path.join(output_dir, "metrics_structured.csv"),
        fieldnames=["dataset", "split", "metric", "value", "protocol"],
    )
    save_to_json(structured_metrics, os.path.join(output_dir, "metrics_structured.json"))
    save_to_json(runtime_context, os.path.join(output_dir, "evaluation_runtime.json"))

    protocol = {
        "validated": True,
        "mode": "fair_same_pool_retriever",
        "dataset": "fashioniq",
        "split": args.split,
        "eval_protocol": args.eval_protocol,
        "candidate_pool": {
            "type": "target_plus_random_negatives",
            "num_random_negatives": args.num_random_distractors,
            "configured_pool_size": args.num_random_distractors + 1,
            "seed": args.seed,
            "reference_removed": True,
        },
    }
    save_to_json(protocol, os.path.join(output_dir, "evaluation_protocol.json"))

    run_config = vars(args).copy()
    run_config["resolved_device"] = str(device)
    run_config["output_dir"] = output_dir
    save_to_json(run_config, os.path.join(output_dir, "run_config.json"))

    if args.save_pool_records:
        save_to_json(pool_records, os.path.join(output_dir, "candidate_pools.json"))

    print("==== FashionIQ Same-Pool Retriever Summary ====")
    print(f"output_dir: {output_dir}")
    for key in sorted(metrics.keys()):
        if key.endswith("avg_recall_at1") or key.endswith("avg_recall_at5") or key.endswith("avg_recall_at10"):
            print(f"{key}: {metrics[key]:.4f}")
    if "fashioniq_val_latency_seconds" in metrics:
        print(f"fashioniq_val_latency_seconds: {metrics['fashioniq_val_latency_seconds']:.4f}")


if __name__ == "__main__":
    main()
