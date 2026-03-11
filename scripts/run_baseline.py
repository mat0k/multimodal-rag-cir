import argparse
import importlib
import json
import os
from typing import Any

import torch

from src.evaluation.cirr_eval import evaluate_cirr, generate_cirr_test_submission
from src.evaluation.fashioniq_eval import evaluate_fashioniq
from src.retrievers.base import TwoEncoderVLM
from src.utils.io import prepend_key_to_dict, save_to_csv


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
    fusion_type: str = "sum",
    batch_size: int = 64,
    num_workers: int = 4,
    use_tqdm: bool = True,
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    if "fashioniq" in datasets:
        fashioniq_metrics = evaluate_fashioniq(
            model=model,
            fusion_type=fusion_type,
            batch_size=batch_size,
            num_workers=num_workers,
            tqdm=use_tqdm,
            accelerator=None,
        )
        metrics.update(prepend_key_to_dict("fashioniq_", fashioniq_metrics))

    if "cirr" in datasets:
        cirr_metrics = evaluate_cirr(
            model=model,
            fusion_type=fusion_type,
            batch_size=batch_size,
            num_workers=num_workers,
            tqdm=use_tqdm,
            accelerator=None,
        )
        metrics.update(prepend_key_to_dict("cirr_", cirr_metrics))

    return metrics


def main(args: argparse.Namespace) -> None:
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

    run_name = args.run_name or f"{args.retriever_class}-zero-shot-{args.fusion_type}"
    output_path = os.path.join(args.output_path, run_name)
    os.makedirs(output_path, exist_ok=True)

    metrics = test_model(
        model=model,
        datasets=args.datasets,
        fusion_type=args.fusion_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_tqdm=args.tqdm,
    )
    save_to_csv(metrics, os.path.join(output_path, "metrics.csv"))

    with open(os.path.join(output_path, "run_config.json"), "w", encoding="utf-8") as file_obj:
        json.dump(vars(args), file_obj, indent=2)

    if "cirr" in args.datasets and "cirr" not in args.skip_submission:
        cirr_test_sub = generate_cirr_test_submission(
            model=model,
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
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--fusion_type", type=str, default="sum", help="Image-text feature fusion strategy.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--tqdm", action="store_true", help="Enable progress bars.")
    parser.add_argument("--output_path", type=str, default="results", help="Directory where outputs are saved.")
    parser.add_argument("--run_name", type=str, default="", help="Optional run folder name override.")

    main(parser.parse_args())


