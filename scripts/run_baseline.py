from xml.parsers.expat import model
from evals.circo_eval import evaluate_circo, generate_circo_test_submission
from evals.cirr_eval import evaluate_cirr, generate_cirr_test_submission
from evals.fashioniq_eval import evaluate_fashioniq
from evals.ma_cir_eval import evaluate_macir
from evals.metrics import plot_sim_distributions
from evals.simat_eval import evaluate_simat
from evals.mscoco_eval import eval_mscoco
from models import TwoEncoderVLM, AutoModel, AutoConfig
from utils.dict import prepend_key_to_dict, save_to_csv
import torch
import json
import os


def test_model(
		model: TwoEncoderVLM, 
        fusion_type: str = "sum",
		batch_size: int = 64,
		num_workers: int = 4,
		tqdm: bool = True,
        skip_metrics: list[str] = [],
    ) -> dict[str, float]:
    """
    Evaluate the model on custom tasks and return metrics.

    Args:
        trainer (Trainer): The Hugging Face Trainer with the model to evaluate.
    Returns:
        dict: Dictionary of evaluation metrics.
    """
	
    metrics = {}
    cached_data = {}

    if "fashioniq" not in skip_metrics:
        fashioniq_metrics = evaluate_fashioniq(
            model=model,
            fusion_type=fusion_type,
            batch_size=batch_size,
            num_workers=num_workers,
            tqdm=tqdm,
            accelerator=None,
        )
        metrics.update(prepend_key_to_dict("fashioniq_", fashioniq_metrics))

    if "simat" not in skip_metrics:
        simat_metrics = evaluate_simat(
            model=model,
            batch_size=batch_size,
            num_workers=num_workers,
            tqdm=tqdm,
            accelerator=None,
            split='test',
        )
        metrics.update(prepend_key_to_dict("simat_", simat_metrics))

    if "circo" not in skip_metrics:
        circo_metrics, circo_index_tuple = evaluate_circo(
            model=model,
            fusion_type=fusion_type,
            batch_size=batch_size,
            num_workers=num_workers,
            tqdm=tqdm,
            accelerator=None,
            return_index_tuple=True
        )
        metrics.update(prepend_key_to_dict("circo_", circo_metrics))
        cached_data['circo_index_tuple'] = circo_index_tuple

    if "cirr" not in skip_metrics:
        cirr_metrics = evaluate_cirr(
            model=model,
            fusion_type=fusion_type,
            batch_size=batch_size,
            num_workers=num_workers,
            tqdm=tqdm,
            accelerator=None,
        )
        metrics.update(prepend_key_to_dict("cirr_", cirr_metrics))

    if "macir" not in skip_metrics:
        ma_cir_metrics = evaluate_macir(
            model=model,
            eval_level="full_splits",
            split="",
            batch_size=batch_size,
            num_workers=num_workers,
            tqdm=tqdm,
            fusion_type=fusion_type,
            accelerator=None
        )
        metrics.update(prepend_key_to_dict("macir_", ma_cir_metrics))
        ma_cir_metrics = evaluate_macir(
            model=model,
            eval_level="full",
            split="",
            batch_size=batch_size,
            num_workers=num_workers,
            tqdm=tqdm,
            fusion_type=fusion_type,
            accelerator=None
        )
        metrics.update(prepend_key_to_dict("macir_", ma_cir_metrics))

    if "mscoco" not in skip_metrics:
        mscoco_metrics, mscoco_sim_distributions = eval_mscoco(
            model=model,
            batch_size=batch_size,
            num_workers=num_workers,
            tqdm=tqdm,
            accelerator=None,
            split='val',
        )
        metrics.update(prepend_key_to_dict("mscoco_", mscoco_metrics))
        cached_data['mscoco_sim_distributions'] = mscoco_sim_distributions
    return metrics, cached_data

def main(args):
    checkpoint_path = args.checkpoint_path
    model_config_path = os.path.relpath(os.path.join(checkpoint_path, os.pardir))

    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model config path does not exist: {model_config_path}")
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        device = torch.device("cuda")
    else:
        print("No GPU available, using CPU.")
        device = torch.device("cpu")

    dirname_ck_component = os.path.basename(checkpoint_path.rstrip('/')) if not args.zero_shot else "zero-shot"
    dirname_config_component = os.path.basename(model_config_path.rstrip('/'))
    newdirname = dirname_config_component + "-" + dirname_ck_component + '-' + args.fusion_type
    output_path = os.path.join(args.output_path, newdirname)
    os.makedirs(output_path, exist_ok=True)

    #load model
    print("Loading model config from:", model_config_path)    
    config = AutoConfig.from_pretrained(model_config_path)
    model = AutoModel.from_config(config)

    if args.zero_shot:
        print("Zero-shot model specified; skipping loading of adapter or full model weights.")

    elif not args.no_peft:
        adapter_path = os.path.join(checkpoint_path, "lora_adapter")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"PEFT adapter path does not exist: {adapter_path}")
        print("Loading PEFT adapter from:", adapter_path)
        model = model.apply_peft_from_pretrained(adapter_path)
        
    else:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint path does not exist: {checkpoint_path}")
        print("Loading full model weights from:", checkpoint_path)
        model.from_pretrained(checkpoint_path)

    #compute and save metrics to file
    model.to(device)
    test_metrics, cached_data = test_model(
        model=model,
        fusion_type=args.fusion_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tqdm=args.tqdm,
        skip_metrics=args.skip_metrics,
    )

    save_to_csv(test_metrics, os.path.join(output_path, "metrics.csv"))


    #generate and save submissions to file
    

    if "cirr" not in args.skip_submission:
        cirr_test_sub = generate_cirr_test_submission(
            model=model,
            fusion_type=args.fusion_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            tqdm=args.tqdm,
        )

        cirr_recall_dict, cirr_subset_recall_dict = cirr_test_sub['top_50'], cirr_test_sub['subset_top_3']
        cirr_recall_dict.update({"version": "rc2", "metric": "recall"})
        cirr_subset_recall_dict.update({"version": "rc2", "metric": "recall_subset"})

        with open(os.path.join(output_path, "cirr_test_submission.json"), 'w') as f:
            json.dump(cirr_recall_dict, f, indent=None)

        with open(os.path.join(output_path, "cirr_subset_test_submission.json"), 'w') as f:
            json.dump(cirr_subset_recall_dict, f, indent=None)

    if "circo" not in args.skip_submission:
        circo_test_sub = generate_circo_test_submission(
            model=model,
            fusion_type=args.fusion_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            tqdm=args.tqdm,
            index_tuple=cached_data['circo_index_tuple'] if 'circo_index_tuple' in cached_data else None,
        )

        with open(os.path.join(output_path, "circo_test_submission.json"), 'w') as f:
            json.dump(circo_test_sub, f, indent=None)

    if "mscoco_sim_distributions" in cached_data:
        pos_sim, rnd_sim = cached_data['mscoco_sim_distributions']
        if args.save_sim_distributions:
            torch.save(pos_sim, os.path.join(output_path, "mscoco_pos_sims.pt"))
            torch.save(rnd_sim, os.path.join(output_path, "mscoco_rnd_sims.pt"))
        plot_sim_distributions(
            pos_similarities=pos_sim,
            rnd_similarities=rnd_sim,
            save_path=os.path.join(output_path, "mscoco_sim_distribution.svg"),
        )




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test model.")
    # Add arguments as needed
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint. Model config will be loaded from the parent directory.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--fusion_type", type=str, default="sum", help="Fusion type for evaluation.")
    parser.add_argument("--tqdm", action='store_true', help="Use tqdm for progress bars.")
    parser.add_argument("--output_path", type=str, default="results/", help="Path to save the evaluation metrics.")
    parser.add_argument("--no_peft", action='store_true', help="Do not use PEFT model.")
    parser.add_argument("--skip_metrics", nargs='*', default=[], help="List of metrics to skip during evaluation.")
    parser.add_argument("--skip_submission", nargs='*', default=[], help="List of metrics to skip during evaluation.")
    parser.add_argument("--save_sim_distributions", action='store_true', help="Save similarity distributions.")
    parser.add_argument("--zero-shot", action='store_true', help="Indicates if the model is zero-shot. Adapter or full model weights are ignored if set. Model config is still loaded from checkpoint path parent directory.")
    args = parser.parse_args()
    main(args)


