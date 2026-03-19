import numpy as np
import torch

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from typing import Literal, Optional, Tuple
from torch.utils.data import Dataset

from src.datasets.cirr import build_cirr_dataset
from src.fusion import fusion
from src.retrievers.base import TwoEncoderVLM
from src.utils.decorators import timed_metric
from src.utils.tensor import make_normalized

DEBUG = False
QUERY_EMBEDDING_MODES = {"legacy_fusion", "vista_mm"}


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

def compute_recall(top_k_retrieved, targets_np):
    """
    return recall given top-k retrieved names and target names
    """
    correct_retrievals = np.any(top_k_retrieved == targets_np, axis=1)
    recall = np.mean(correct_retrievals) * 100.0
    return recall

def compute_names(top_k_retrieved, pair_ids):
    """
    return a dict mapping pair_id to list of retrieved names
    """
    names_dict = {}
    for i, pair_id in enumerate(pair_ids):
        pair_id_value = pair_id.item() if hasattr(pair_id, "item") else pair_id
        names_dict[pair_id_value] = top_k_retrieved[i].tolist()
    return names_dict

def compute_cirr_metrics(
    index_features: torch.Tensor,
    index_names: list,
    predicted_features: torch.Tensor,
    reference_names: list,
    target_names: list,
    group_members: list,
    pair_ids: list,
    k_values: Optional[list] = [1, 5, 10, 50],
    k_values_subset: Optional[list] = [1, 2, 3],
    skip_subset_metrics: bool = False,
    return_type:str = 'metrics'
):
    """
    Generate CIRR evaluation metrics.
    Args:
        index_features (torch.Tensor): Index features of shape (M, D).
        index_names (list): List of index image names of length M.
        predicted_features (torch.Tensor): Predicted features of shape (N, D).
        reference_names (list): List of reference image names of length N.
        target_names (list): List of target image names of length N.
        group_members (list): List of lists of group members for each target of length N.
        pair_ids (list): List of pair IDs of length N.
        k_values (list, optional): List of k values for Recall@K. Defaults to [1, 5, 10, 50].
        k_values_subset (list, optional): List of k values for subset Recall@K. Defaults to [1, 2, 3].
        skip_subset_metrics (bool, optional): Whether to skip subset metrics. Defaults to False.
        return_type (str, optional): Type of return value ('metrics' or 'names'). Defaults to 'metrics'.
    Returns:
        dict: Dictionary containing evaluation metrics.
    """

    if DEBUG:
        print(f"index_features shape: {index_features.shape}")
        print(f"predicted_features shape: {predicted_features.shape}")
        print(f"Number of index names: {len(index_names)}")
        print(f"Number of reference names: {len(reference_names)}")
        print(f"Number of target names: {len(target_names)}")
        print(f"Number of group members: {len(group_members)}")
        print(f"Number of pair ids: {len(pair_ids)}")

    similarities = predicted_features @ index_features.T  # (N, M)
    #sorted indices of database images (row-wise) for each query
    sorted_indices = torch.argsort(similarities, dim=1, descending=True).cpu()  # (N, M)

    sorted_index_names = np.array(index_names)[sorted_indices]  # (N, M)

    # Remove the reference (original) image itself from the ranking (we do not want the reference image to count as a retrieval candidate)
    # reference_names: (N,)
    # sorted_index_names_matrix: (N, M)
    # Build a mask that is False where the candidate equals the reference image,
    reference_mask = torch.tensor(
        sorted_index_names
        != np.repeat(np.array(reference_names), len(index_names)).reshape(
            len(pair_ids), -1
        )
    )
    if DEBUG:
        print(f"sorted_index_names shape: {sorted_index_names.shape}")
        print(f"reference_mask shape: {reference_mask.shape}")
        print(f"Number of True in reference_mask (should be N*(M-1)): {torch.sum(reference_mask).item()}")

    # Apply the mask and reshape back to (N, M-1).
    # Now each row corresponds to candidates excluding the exact reference image.
    sorted_index_names = sorted_index_names[reference_mask].reshape(
        sorted_index_names.shape[0], sorted_index_names.shape[1] - 1
    )

    #returned data structure
    output = {}

    # ---- compute recall@k -------

    # convert target names to numpy array and reshape for broadcasting
    targets_np = np.array(target_names).reshape(-1, 1)  # (N, 1)

    for k in k_values:
        top_k_retrieved = sorted_index_names[:, :k]
        if return_type == 'names':
            output[f"top_{k}"] = compute_names(top_k_retrieved, pair_ids)
        elif return_type == 'metrics':
            output[f"recall_at{k}"] = compute_recall(top_k_retrieved, targets_np)

    # Audit global protocol correctness.
    reference_np = np.array(reference_names).reshape(-1, 1)
    if np.any(sorted_index_names == reference_np):
        raise ValueError("CIRR reference-image removal failed: reference still appears in ranking.")
    if return_type == 'metrics' and not np.all(np.any(sorted_index_names == targets_np, axis=1)):
        raise ValueError("CIRR target not found in full candidate gallery for one or more queries.")

    # ---- compute subset recall@k ------
    if not skip_subset_metrics:
        max_subset_k = max(k_values_subset)
        subset_candidates: list[np.ndarray] = []

        for i, members in enumerate(group_members):
            member_set = set(members)
            subset_candidate_names = np.array([name for name in sorted_index_names[i] if name in member_set])

            if len(subset_candidate_names) < max_subset_k:
                pair_id = pair_ids[i].item() if hasattr(pair_ids[i], "item") else pair_ids[i]
                raise ValueError(
                    f"Subset gallery has only {len(subset_candidate_names)} candidates for pair_id={pair_id}; "
                    f"expected at least {max_subset_k}."
                )

            if return_type == 'metrics' and target_names[i] not in member_set:
                pair_id = pair_ids[i].item() if hasattr(pair_ids[i], "item") else pair_ids[i]
                raise ValueError(f"Target for pair_id={pair_id} is missing from CIRR subset member list.")

            subset_candidates.append(subset_candidate_names)

        for k in k_values_subset:
            if return_type == 'names':
                subset_top_k = np.array([candidates[:k] for candidates in subset_candidates])
                output[f"subset_top_{k}"] = compute_names(subset_top_k, pair_ids)
            elif return_type == 'metrics':
                subset_hits = []
                for i, candidates in enumerate(subset_candidates):
                    subset_hits.append(target_names[i] in candidates[:k])
                output[f"subset_recall_at{k}"] = float(np.mean(subset_hits) * 100.0)

    return output
    

@torch.no_grad()
def generate_cirr_index_features(
    clip_model :TwoEncoderVLM,
    index_dataset: Dataset,
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

    clip_model.eval()
    vision_encoder = clip_model.vision
    vision_device = _get_module_device(vision_encoder)

    for batch in tqdm(dataloader, disable=not use_tqdm, desc="Generating CIRR index features"):
        images = batch['image'].to(vision_device)

        image_features = vision_encoder(images).image_embeds

        all_image_features.append(image_features)
        all_image_names.extend(batch['image_name'])

    all_image_features = torch.vstack(all_image_features)
    all_image_features = make_normalized(all_image_features)

    return all_image_features, all_image_names

# @torch.no_grad()
# def generate_cirr_predictions(
#     clip_model :TwoEncoderVLM,
#     triplet_dataset: Dataset,
#     fusion_type: str,
#     batch_size: int = 64,
#     num_workers: int = 4,
#     use_tqdm: bool = False,
#     accelerator=None,
#     skip_targets: bool = False
# ):
#     dataloader = DataLoader(
#         triplet_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#     )
#     all_predicted_features = []
#     all_reference_names = []
#     all_target_names = []
#     all_group_members = []
#     all_pair_ids = []

#     clip_model.eval()
#     text_encoder = clip_model.text
#     vision_encoder = clip_model.vision

#     for batch in tqdm(dataloader, disable=not use_tqdm, desc="Generating CIRR predictions"):
#         reference_images = batch['reference'].to(vision_encoder.device)
#         reference_names = batch['reference_name']
#         group_members = batch['group_members']
#         pair_ids = batch['pair_id']
#         relative_captions = batch['transformed_caption'].to(text_encoder.device)
#         attention_masks = batch['attention_mask'].to(text_encoder.device)

#         if skip_targets:
#             target_names = []
#         else:
#             target_names = batch['target_name']

#         # batch size is returned as (G, B) where G is the number of groups and B is the number of triplets per group
#         # we need to switch to (B, G) for proper processing
#         group_members_reshaped = []
#         for i in range(len(group_members[0])):
#             group_members_reshaped.append([group_members[j][i] for j in range(len(group_members))])


#         reference_features = vision_encoder(reference_images).image_embeds
#         caption_features = text_encoder(
#             input_ids=relative_captions,
#             attention_mask=attention_masks
#         ).text_embeds

#         predicted_features = fusion(
#             image_features=reference_features,
#             text_features=caption_features,
#             fusion_type=fusion_type
#         )
        
#         all_predicted_features.append(predicted_features)
#         all_reference_names.extend(reference_names)
#         all_target_names.extend(target_names)
#         all_group_members.extend(group_members_reshaped)
#         all_pair_ids.extend(pair_ids)

#     all_predictions = torch.vstack(all_predicted_features)
#     return all_predictions, all_reference_names, all_target_names, all_group_members, all_pair_ids

@torch.no_grad()
def generate_cirr_triplet_features(
    clip_model :TwoEncoderVLM,
    triplet_dataset: Dataset,
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
    all_group_members = []
    all_pair_ids = []

    clip_model.eval()
    text_encoder = clip_model.text
    vision_encoder = clip_model.vision
    text_device = _get_module_device(text_encoder)
    vision_device = _get_module_device(vision_encoder)

    for batch in tqdm(dataloader, disable=not use_tqdm, desc="Generating CIRR triplet features"):
        reference_images = batch['reference'].to(vision_device)
        reference_names = batch['reference_name']
        group_members = batch['group_members']
        pair_ids = batch['pair_id']
        relative_captions = batch['transformed_caption'].to(text_device)
        attention_masks = batch['attention_mask'].to(text_device)

        if skip_targets:
            target_names = []
        else:
            target_names = batch['target_name']

        # batch size is returned as (G, B) where G is the number of groups and B is the number of triplets per group
        # we need to switch to (B, G) for proper processing
        group_members_reshaped = []
        for i in range(len(group_members[0])):
            group_members_reshaped.append([group_members[j][i] for j in range(len(group_members))])


        reference_features = vision_encoder(reference_images).image_embeds
        caption_features = text_encoder(
            input_ids=relative_captions,
            attention_mask=attention_masks
        ).text_embeds
        
        all_image_features.append(reference_features)
        all_text_features.append(caption_features)
        all_reference_names.extend(reference_names)
        all_target_names.extend(target_names)
        all_group_members.extend(group_members_reshaped)
        all_pair_ids.extend(pair_ids)

    all_image_features = torch.vstack(all_image_features)
    all_text_features = torch.vstack(all_text_features)
    return all_image_features, all_text_features, all_reference_names, all_target_names, all_group_members, all_pair_ids


@torch.no_grad()
def generate_cirr_mm_query_features(
    clip_model: TwoEncoderVLM,
    triplet_dataset: Dataset,
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
    all_group_members = []
    all_pair_ids = []

    clip_model.eval()
    mm_device = _get_mm_device(clip_model)

    for batch in tqdm(dataloader, disable=not use_tqdm, desc="Generating CIRR multimodal query features"):
        reference_images = batch['reference'].to(mm_device)
        reference_names = batch['reference_name']
        group_members = batch['group_members']
        pair_ids = batch['pair_id']
        relative_captions = batch['transformed_caption'].to(mm_device)
        attention_masks = batch['attention_mask'].to(mm_device)

        if skip_targets:
            target_names = []
        else:
            target_names = batch['target_name']

        group_members_reshaped = []
        for i in range(len(group_members[0])):
            group_members_reshaped.append([group_members[j][i] for j in range(len(group_members))])

        predicted_features = _encode_mm_query(
            model=clip_model,
            images=reference_images,
            input_ids=relative_captions,
            attention_mask=attention_masks,
        )

        all_predicted_features.append(predicted_features)
        all_reference_names.extend(reference_names)
        all_target_names.extend(target_names)
        all_group_members.extend(group_members_reshaped)
        all_pair_ids.extend(pair_ids)

    all_predictions = torch.vstack(all_predicted_features)
    return all_predictions, all_reference_names, all_target_names, all_group_members, all_pair_ids


@torch.no_grad()
def generate_cirr_predicted_features(
    clip_model: TwoEncoderVLM,
    triplet_dataset: Dataset,
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
        return generate_cirr_mm_query_features(
            clip_model=clip_model,
            triplet_dataset=triplet_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            use_tqdm=use_tqdm,
            accelerator=accelerator,
            skip_targets=skip_targets,
        )

    image_features, text_features, reference_names, target_names, group_members, pair_ids = generate_cirr_triplet_features(
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
        alpha=0.7,
    )
    predicted_features = make_normalized(predicted_features)

    return predicted_features, reference_names, target_names, group_members, pair_ids

@timed_metric
def evaluate_cirr(
    model: TwoEncoderVLM,
    query_embedding_mode: Literal["legacy_fusion", "vista_mm"] = "vista_mm",
    fusion_type: str = 'sum',
    batch_size: int = 64,
    num_workers: int = 4,
    tqdm : bool = False,
    accelerator=None,
    skip_subset_metrics: bool = False,
    index_tuple: Tuple[torch.Tensor, list[int]] = None,
    return_index_tuple: bool = False,
):
    if index_tuple is None:
        cirr_index = build_cirr_dataset(
            split='val',
            mode='images',
            image_transform=model.image_processor,
            caption_transform=model.tokenizer,
            max_length_tokenizer=77
        )

    cirr_triplets = build_cirr_dataset(
        split='val',
        mode='triplets',
        image_transform=model.image_processor,
        caption_transform=model.tokenizer,
        max_length_tokenizer=77
    )

    if index_tuple is None:
        index_features, index_names = generate_cirr_index_features(
            clip_model=model,
            index_dataset=cirr_index,
            batch_size=batch_size,
            num_workers=num_workers,
            use_tqdm=tqdm,
            accelerator=accelerator
        )
    else:
        index_features, index_names = index_tuple

    # predicted_features, reference_names, target_names, group_members, pair_ids = generate_cirr_predictions(
    #     clip_model=model,
    #     triplet_dataset=cirr_triplets,
    #     fusion_type=fusion_type,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     use_tqdm=tqdm,
    #     accelerator=accelerator
    # )

    predicted_features, reference_names, target_names, group_members, pair_ids = generate_cirr_predicted_features(
        clip_model=model,
        triplet_dataset=cirr_triplets,
        query_embedding_mode=query_embedding_mode,
        fusion_type=fusion_type,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tqdm=tqdm,
        accelerator=accelerator,
    )

    raw_metrics = compute_cirr_metrics(
        index_features=index_features,
        index_names=index_names,
        predicted_features=predicted_features,
        reference_names=reference_names,
        target_names=target_names,
        group_members=group_members,
        pair_ids=pair_ids,
        skip_subset_metrics=skip_subset_metrics,
        return_type='metrics',
        k_values = [1,5,10,50],
        k_values_subset = [1,2,3],
    )

    metrics: dict[str, float] = {}

    for k in [1, 5, 10, 50]:
        key = f"recall_at{k}"
        if key in raw_metrics:
            metrics[f"val_global_recall_at{k}"] = float(raw_metrics[key])

    for k in [1, 2, 3]:
        key = f"subset_recall_at{k}"
        if key in raw_metrics:
            metrics[f"val_subset_recall_at{k}"] = float(raw_metrics[key])

    if "val_global_recall_at5" in metrics and "val_subset_recall_at1" in metrics:
        metrics["val_summary_average"] = float(
            np.mean([metrics["val_global_recall_at5"], metrics["val_subset_recall_at1"]])
        )

    if return_index_tuple:
        return metrics, (index_features, index_names)
    return metrics


def generate_cirr_test_submission(
    model: TwoEncoderVLM,
    query_embedding_mode: Literal["legacy_fusion", "vista_mm"] = "vista_mm",
    fusion_type: str = 'sum',
    batch_size: int = 64,
    num_workers: int = 4,
    tqdm : bool = False,
    accelerator=None,
    index_tuple: Tuple[torch.Tensor, list[int]] = None,
    return_index_tuple: bool = False,
):
    """
    Similar to evaluate_cirr but for test set. 
    It return a dict with two keys: 'subset_top_3' and 'top_50', each containing a dict of pair_id -> list[retrieved names].
    """
    if index_tuple is None:
        cirr_index = build_cirr_dataset(
            split='test1',
            mode='images',
            image_transform=model.image_processor,
            caption_transform=model.tokenizer,
            max_length_tokenizer=77
        )

    cirr_triplets = build_cirr_dataset(
        split='test1',
        mode='triplets',
        image_transform=model.image_processor,
        caption_transform=model.tokenizer,
        max_length_tokenizer=77
    )

    if index_tuple is None:
        index_features, index_names = generate_cirr_index_features(
            clip_model=model,
            index_dataset=cirr_index,
            batch_size=batch_size,
            num_workers=num_workers,
            use_tqdm=tqdm,
            accelerator=accelerator
        )
    else:
        index_features, index_names = index_tuple

    # predicted_features, reference_names, target_names, group_members, pair_ids = generate_cirr_predictions(
    #     clip_model=model,
    #     triplet_dataset=cirr_triplets,
    #     fusion_type=fusion_type,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     use_tqdm=tqdm,
    #     accelerator=accelerator,
    #     skip_targets=True
    # )

    predicted_features, reference_names, target_names, group_members, pair_ids = generate_cirr_predicted_features(
        clip_model=model,
        triplet_dataset=cirr_triplets,
        query_embedding_mode=query_embedding_mode,
        fusion_type=fusion_type,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tqdm=tqdm,
        accelerator=accelerator,
        skip_targets=True
    )

    submission = compute_cirr_metrics(
        index_features=index_features,
        index_names=index_names,
        predicted_features=predicted_features,
        reference_names=reference_names,
        target_names=target_names,
        group_members=group_members,
        pair_ids=pair_ids,
        return_type='names',
        k_values=[50],  # retrieve top-50 for test submission
        k_values_subset=[3],  # retrieve top-3 for test subset test submission
    )

    if return_index_tuple:
        return submission, (index_features, index_names)
    return submission   


def cirr_test_alpha(
    model: TwoEncoderVLM,
    alphas: list[int],
    batch_size: int = 64,
    num_workers: int = 4,
    use_tqdm: bool = False,
):
    cirr_index = build_cirr_dataset(
        split='val',
        mode='images',
        image_transform=model.image_processor,
        caption_transform=model.tokenizer,
        max_length_tokenizer=77
    )

    cirr_triplets = build_cirr_dataset(
        split='val',
        mode='triplets',
        image_transform=model.image_processor,
        caption_transform=model.tokenizer,
        max_length_tokenizer=77
    )

    index_features, index_names = generate_cirr_index_features(
        clip_model=model,
        index_dataset=cirr_index,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tqdm=use_tqdm,
    )

    image_features, text_features, reference_names, target_names, group_members, pair_ids = generate_cirr_triplet_features(
        clip_model=model,
        triplet_dataset=cirr_triplets,
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
        metrics = compute_cirr_metrics(
            index_features=index_features,
            index_names=index_names,
            predicted_features=predicted_features,
            reference_names=reference_names,
            target_names=target_names,
            group_members=group_members,
            pair_ids=pair_ids,
            skip_subset_metrics=False,
            return_type='metrics',
            k_values = [1,5,10,50],
            k_values_subset = [1,2,3],
        )
        alpha_scores[alpha] = metrics

    return alpha_scores