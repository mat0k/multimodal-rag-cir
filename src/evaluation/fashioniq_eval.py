import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from typing import Literal, Optional

from src.datasets.fashioniq import FashionIQ, build_fashioniq_dataset
from src.fusion import fusion
from src.retrievers.base import TwoEncoderVLM
from src.utils.decorators import timed_metric
from src.utils.tensor import make_normalized


QUERY_EMBEDDING_MODES = {"legacy_fusion", "vista_mm"}


FASHIONIQ_EVAL_PROTOCOLS = {
    "val_split": {
        "split": "val",
        "metric_prefix": "val",
        "caption_joiner": " ",
        "candidate_pool": "category-specific validation gallery (dress/shirt/toptee), reference removed",
    },
    "original_split": {
        # True original FashionIQ split uses the official test split annotations and gallery.
        "split": "test",
        "metric_prefix": "original",
        "caption_joiner": " and ",
        "candidate_pool": "category-specific official test gallery (dress/shirt/toptee), reference removed",
    },
}


def resolve_fashioniq_eval_protocol(eval_protocol: str) -> dict[str, str]:
    if eval_protocol not in FASHIONIQ_EVAL_PROTOCOLS:
        supported = ", ".join(sorted(FASHIONIQ_EVAL_PROTOCOLS.keys()))
        raise ValueError(f"Unsupported FashionIQ eval protocol '{eval_protocol}'. Supported: {supported}")
    return FASHIONIQ_EVAL_PROTOCOLS[eval_protocol].copy()


def _get_module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _resolve_query_embedding_mode(query_embedding_mode: str) -> str:
    if query_embedding_mode not in QUERY_EMBEDDING_MODES:
        supported = ", ".join(sorted(QUERY_EMBEDDING_MODES))
        raise ValueError(
            f"Unsupported query_embedding_mode '{query_embedding_mode}'. Supported: {supported}"
        )
    return query_embedding_mode


def _get_mm_device(model: TwoEncoderVLM) -> torch.device:
    backbone = getattr(model, "backbone", None)
    if isinstance(backbone, torch.nn.Module):
        return _get_module_device(backbone)
    return _get_module_device(model.vision)


def _encode_mm_query(
    model: TwoEncoderVLM,
    images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if hasattr(model, "encode_query_mm"):
        query_features = model.encode_query_mm(
            pixel_values=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return make_normalized(query_features)

    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "encode_mm"):
        tokenized = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return make_normalized(backbone.encode_mm(images, tokenized))

    raise ValueError(
        "query_embedding_mode='vista_mm' requires either model.encode_query_mm(...) "
        "or model.backbone.encode_mm(...)."
    )

def compute_fashioniq_metrics(
    index_features: torch.Tensor, # (N, D) tensor of index features
    index_names: list, # list of length N with the names of the index images
    index_classes: list, # list of length N with the classes of the index features
    predicted_features: torch.Tensor, # (M, D) tensor of predicted features for the M triplets
    reference_names: list, # list of length M with the names of the reference images for each triplet
    target_names: list, # list of length M with the names of the target images for each triplet
    triplet_classes: list, # list of length M with the classes of the triplets
    k_values: Optional[list] = [5, 10, 50],
    metric_prefix: str = "val",
):
    """Compute FashionIQ recall metrics grouped by class for the given index and predicted features."""
    metrics = {}
    expected_classes = {"dress", "shirt", "toptee"}
    unique_classes = sorted(set(triplet_classes))

    if len(reference_names) != len(target_names):
        raise ValueError("FashionIQ evaluation expects one target per query; found mismatch.")

    if not expected_classes.issubset(set(unique_classes)):
        raise ValueError(
            "FashionIQ evaluation is expected to contain dress/shirt/toptee queries; "
            f"found classes: {sorted(set(unique_classes))}"
        )

    for cls in unique_classes:
        cls_index_indices = [i for i, c in enumerate(index_classes) if c == cls]
        cls_index_features = index_features[cls_index_indices]
        cls_index_names = [index_names[i] for i in cls_index_indices]

        cls_indices = [i for i, c in enumerate(triplet_classes) if c == cls]
        cls_predicted_features = predicted_features[cls_indices]
        cls_reference_names = [reference_names[i] for i in cls_indices]
        cls_target_names = [target_names[i] for i in cls_indices]

        if cls_index_features.shape[0] <= max(k_values):
            raise ValueError(
                f"Class '{cls}' gallery has {cls_index_features.shape[0]} items, "
                f"which is not enough to compute max recall@{max(k_values)}."
            )

        # Compute similarity scores between predicted features and index features
        similarity_scores = torch.matmul(cls_predicted_features, cls_index_features.T)
        sorted_indices = torch.argsort(similarity_scores, dim=1, descending=True).cpu()
        # sorted_index_names = [[cls_index_names[i] for i in sorted_indices[j]] for j in range(len(cls_predicted_features))]
        sorted_index_names = np.array(cls_index_names)[sorted_indices]

        #remove reference names from sorted index names
        reference_mask = torch.tensor(
            sorted_index_names != np.repeat(np.array(cls_reference_names), len(cls_index_names)).reshape(
                len(cls_predicted_features), -1
            )
        )

        # print("reference mask", reference_mask.shape, reference_mask.sum())
        # print("sorted index names", sorted_index_names.shape)
        # print("\ncls reference names", len(cls_reference_names), cls_reference_names[:10])
        # print("\ncls target names", len(cls_target_names), cls_target_names[:10])
        # print("\ncls index names", len(cls_index_names), cls_index_names[:10])

        sorted_index_names = sorted_index_names[reference_mask].reshape(
            sorted_index_names.shape[0], sorted_index_names.shape[1] - 1
        )

        targets_np = np.array(cls_target_names).reshape(-1, 1)
        reference_np = np.array(cls_reference_names).reshape(-1, 1)

        # Audit checks to ensure protocol correctness.
        if np.any(sorted_index_names == reference_np):
            raise ValueError(f"Reference removal failed for class '{cls}'.")
        if not np.all(np.any(sorted_index_names == targets_np, axis=1)):
            raise ValueError(f"Some targets are not present in class '{cls}' candidate gallery.")

        # Compute R@K for each K in k_values
        for k in k_values:
            top_k_names = sorted_index_names[:, :k]
            hits = np.any(top_k_names == targets_np, axis=1)
            recall_at_k = np.mean(hits) * 100
            metrics[f"{metric_prefix}_{cls}_recall_at{k}"] = float(recall_at_k)

    # compute averages across classes
    for k in k_values:
        avg_recall_at_k = np.mean([metrics[f"{metric_prefix}_{cls}_recall_at{k}"] for cls in unique_classes])
        metrics[f"{metric_prefix}_avg_recall_at{k}"] = float(avg_recall_at_k)

    return metrics
    

@torch.no_grad()
def generate_fashioniq_index_features(
    clip_model :TwoEncoderVLM,
    index_dataset: FashionIQ,
    batch_size: int = 64,
    num_workers: int = 4,
    use_tqdm: bool = False,
    accelerator=None
):
    dataloader = DataLoader(
        index_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    all_image_features = []
    all_image_names = []
    all_classes = []

    clip_model.eval()
    vision_encoder = clip_model.vision
    vision_device = _get_module_device(vision_encoder)

    for batch in tqdm(dataloader, disable=not use_tqdm, desc="Generating FashionIQ index features"):
        images = batch['image'].to(vision_device)

        image_features = vision_encoder(images).image_embeds

        all_image_features.append(image_features)
        all_image_names.extend(batch['image_name'])
        all_classes.extend(batch['class'])
    all_image_features = torch.vstack(all_image_features)
    all_image_features = make_normalized(all_image_features)

    return all_image_features, all_image_names, all_classes

@torch.no_grad()
def generate_fashioniq_triplet_features(
    clip_model :TwoEncoderVLM,
    triplet_dataset: FashionIQ,
    batch_size: int = 64,
    num_workers: int = 4,
    use_tqdm: bool = False,
    accelerator=None,
    skip_targets: bool = False
):
    dataloader = DataLoader(
        triplet_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    all_image_features = []
    all_text_features = []
    all_reference_names = []
    all_target_names = []
    all_classes = []

    clip_model.eval()
    text_encoder = clip_model.text
    vision_encoder = clip_model.vision
    text_device = _get_module_device(text_encoder)
    vision_device = _get_module_device(vision_encoder)

    for batch in tqdm(dataloader, disable=not use_tqdm, desc="Generating FashionIQ triplet features"):
        reference_images = batch['candidate'].to(vision_device)
        reference_names = batch['candidate_name']
        relative_captions = batch['transformed_caption'].to(text_device)
        attention_masks = batch['attention_mask'].to(text_device)

        if skip_targets:
            target_names = []
        else:
            raw_target_names = batch.get('target_name')
            if raw_target_names is None:
                target_names = [None] * len(reference_names)
            else:
                target_names = [name if name else None for name in raw_target_names]

        reference_features = vision_encoder(reference_images).image_embeds
        caption_features = text_encoder(
            input_ids=relative_captions,
            attention_mask=attention_masks
        ).text_embeds
        
        all_image_features.append(reference_features)
        all_text_features.append(caption_features)
        all_reference_names.extend(reference_names)
        all_target_names.extend(target_names)
        all_classes.extend(batch['class'])

    all_image_features = torch.vstack(all_image_features)
    all_text_features = torch.vstack(all_text_features)
    return all_image_features, all_text_features, all_reference_names, all_target_names, all_classes


@torch.no_grad()
def generate_fashioniq_mm_query_features(
    clip_model: TwoEncoderVLM,
    triplet_dataset: FashionIQ,
    batch_size: int = 64,
    num_workers: int = 4,
    use_tqdm: bool = False,
    accelerator=None,
    skip_targets: bool = False,
):
    dataloader = DataLoader(
        triplet_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_predicted_features = []
    all_reference_names = []
    all_target_names = []
    all_classes = []

    clip_model.eval()
    mm_device = _get_mm_device(clip_model)

    for batch in tqdm(dataloader, disable=not use_tqdm, desc="Generating FashionIQ multimodal query features"):
        reference_images = batch['candidate'].to(mm_device)
        reference_names = batch['candidate_name']
        relative_captions = batch['transformed_caption'].to(mm_device)
        attention_masks = batch['attention_mask'].to(mm_device)

        if skip_targets:
            target_names = []
        else:
            raw_target_names = batch.get('target_name')
            if raw_target_names is None:
                target_names = [None] * len(reference_names)
            else:
                target_names = [name if name else None for name in raw_target_names]

        predicted_features = _encode_mm_query(
            model=clip_model,
            images=reference_images,
            input_ids=relative_captions,
            attention_mask=attention_masks,
        )

        all_predicted_features.append(predicted_features)
        all_reference_names.extend(reference_names)
        all_target_names.extend(target_names)
        all_classes.extend(batch['class'])

    all_predictions = torch.vstack(all_predicted_features)
    return all_predictions, all_reference_names, all_target_names, all_classes


@torch.no_grad()
def generate_fashioniq_predicted_features(
    clip_model: TwoEncoderVLM,
    triplet_dataset: FashionIQ,
    query_embedding_mode: Literal["legacy_fusion", "vista_mm"] = "vista_mm",
    fusion_type: str = "sum",
    batch_size: int = 64,
    num_workers: int = 4,
    use_tqdm: bool = False,
    accelerator=None,
    skip_targets: bool = False,
):
    resolved_mode = _resolve_query_embedding_mode(query_embedding_mode)

    if resolved_mode == "vista_mm":
        return generate_fashioniq_mm_query_features(
            clip_model=clip_model,
            triplet_dataset=triplet_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            use_tqdm=use_tqdm,
            accelerator=accelerator,
            skip_targets=skip_targets,
        )

    image_features, text_features, reference_names, target_names, triplet_classes = generate_fashioniq_triplet_features(
        clip_model=clip_model,
        triplet_dataset=triplet_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tqdm=use_tqdm,
        accelerator=accelerator,
        skip_targets=skip_targets,
    )

    predicted_features = fusion(
        image_features=image_features,
        text_features=text_features,
        fusion_type=fusion_type,
        alpha=0.6,
    )
    predicted_features = make_normalized(predicted_features)
    return predicted_features, reference_names, target_names, triplet_classes

@timed_metric
def evaluate_fashioniq(
    model: TwoEncoderVLM,
    query_embedding_mode: Literal["legacy_fusion", "vista_mm"] = "vista_mm",
    fusion_type: str = 'sum',
    batch_size: int = 64,
    num_workers: int = 4,
    tqdm : bool = False,
    accelerator=None,
    eval_protocol: str = "val_split",
):
    protocol_config = resolve_fashioniq_eval_protocol(eval_protocol)

    fashioniq_index = build_fashioniq_dataset(
        split=protocol_config["split"],
        mode='images',
        image_transform=model.image_processor,
        caption_transform=model.tokenizer,
        max_length_tokenizer=77,
        caption_joiner=protocol_config["caption_joiner"],
    )

    fashioniq_triplets = build_fashioniq_dataset(
        split=protocol_config["split"],
        mode='triplets',
        image_transform=model.image_processor,
        caption_transform=model.tokenizer,
        max_length_tokenizer=77,
        caption_joiner=protocol_config["caption_joiner"],
    )

    index_features, index_names, index_classes  = generate_fashioniq_index_features(
        clip_model=model,
        index_dataset=fashioniq_index,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tqdm=tqdm,
        accelerator=accelerator
    )

    predicted_features, reference_names, target_names, triplet_classes = generate_fashioniq_predicted_features(
        clip_model=model,
        triplet_dataset=fashioniq_triplets,
        query_embedding_mode=query_embedding_mode,
        fusion_type=fusion_type,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tqdm=tqdm,
        accelerator=accelerator
    )

    if any(target_name is None for target_name in target_names):
        if protocol_config["split"] == "test":
            # Official test split has no targets; run in submission-style mode.
            return {}
        raise ValueError(
            "FashionIQ evaluation requires target ids for metric computation, but at least one target is missing. "
            f"Protocol '{eval_protocol}' is configured with split '{protocol_config['split']}'."
        )

    metrics = compute_fashioniq_metrics(
        index_features=index_features,
        index_names=index_names,
        index_classes = index_classes,
        predicted_features=predicted_features,
        reference_names=reference_names,
        target_names=target_names,
        triplet_classes=triplet_classes,
        k_values = [5, 10, 50],
        metric_prefix=protocol_config["metric_prefix"],
    )

    return metrics

def fashioniq_test_alpha(
    model: TwoEncoderVLM,
    alphas: list[int],
    batch_size: int = 64,
    num_workers: int = 4,
    use_tqdm: bool = False,
):
    fashioniq_index = build_fashioniq_dataset(
        split='val',
        mode='images',
        image_transform=model.image_processor,
        caption_transform=model.tokenizer,
        max_length_tokenizer=77
    )

    fashioniq_triplets = build_fashioniq_dataset(
        split='val',
        mode='triplets',
        image_transform=model.image_processor,
        caption_transform=model.tokenizer,
        max_length_tokenizer=77
    )

    index_features, index_names, index_classes  = generate_fashioniq_index_features(
        clip_model=model,
        index_dataset=fashioniq_index,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tqdm=use_tqdm,
    )

    image_features, text_features, reference_names, target_names, triplet_classes = generate_fashioniq_triplet_features(
        clip_model=model,
        triplet_dataset=fashioniq_triplets,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tqdm=use_tqdm,
    )

    alpha_scores = {}
    for alpha in alphas:
        predicted_features = fusion(
            image_features=image_features,
            text_features=text_features,
            fusion_type="slerp",
            alpha=alpha
        )

        metrics = compute_fashioniq_metrics(
            index_features=index_features,
            index_names=index_names,
            index_classes = index_classes,
            predicted_features=predicted_features,
            reference_names=reference_names,
            target_names=target_names,
            triplet_classes=triplet_classes,
            k_values = [5, 10, 50],
        )
        alpha_scores[alpha] = metrics

    return alpha_scores