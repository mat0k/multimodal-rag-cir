import numpy as np
import torch

from evals import metrics
from models import TwoEncoderVLM
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from typing import Optional, Tuple
from torch.utils.data import Dataset

from custom_datasets.cirr import build_cirr_dataset
from utils.decorators import timed_metric
from utils.tensor import make_normalized
from fusion import fusion

DEBUG = False

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
        names_dict[pair_id.item()] = top_k_retrieved[i].tolist()
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

    # ---- compute subset recall@k ------
    if not skip_subset_metrics:
        #we need to exclude non-group members from the ranking
        # previous implementation:

        # length of group members might vary, so we need to build a mask for each query
        # max_k = max(k_values_subset)
        # subset_candidates = []
        # for i, members in enumerate(group_members):
        #     member_set = set(members)
        #     mask = np.array([name in member_set for name in sorted_index_names[i]]) #(M-1,)
        #     subset_candidate_names = sorted_index_names[i][mask]

        #     assert len(subset_candidate_names) >= max_k, f"Number of subset candidates ({len(subset_candidate_names)}) is not enough for max_k ({max_k}) for pair_id {pair_ids[i]}"
        #     subset_candidates.append(subset_candidate_names[:max_k])

        # sorted_index_names_subset = np.array(subset_candidates)  # (N, max_k)

        # convert group members to numpy array and reshape for broadcasting
        group_members = np.array(group_members).reshape(len(pair_ids), 1, -1)  # (N, 1, G)
        # compute mask to select only group members from sorted_index_names
        subset_mask = np.any(sorted_index_names[:, :, np.newaxis] == group_members, axis=2) # (N, M-1)
        # apply mask and reshape
        sorted_index_names_subset = sorted_index_names[subset_mask].reshape(sorted_index_names.shape[0], -1) # (N, G)

        if DEBUG:
            print(f"sorted_index_names shape: {sorted_index_names.shape}")
            print(f"pair_ids length: {len(pair_ids)}")
            print(f"group_members length: {len(group_members)}. Width of first element: {len(group_members[0])}")
            print(f"sorted_index_names_subset shape: {sorted_index_names_subset.shape}")


        for k in k_values_subset:
            top_k_retrieved = sorted_index_names_subset[:, :k]
            if return_type == 'names':
                output[f"subset_top_{k}"] = compute_names(top_k_retrieved, pair_ids)
            elif return_type == 'metrics':
                output[f"subset_recall_at{k}"] = compute_recall(top_k_retrieved, targets_np)

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

    for batch in tqdm(dataloader, disable=not use_tqdm, desc="Generating CIRR index features"):
        images = batch['image'].to(vision_encoder.device)

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

    for batch in tqdm(dataloader, disable=not use_tqdm, desc="Generating CIRR triplet features"):
        reference_images = batch['reference'].to(vision_encoder.device)
        reference_names = batch['reference_name']
        group_members = batch['group_members']
        pair_ids = batch['pair_id']
        relative_captions = batch['transformed_caption'].to(text_encoder.device)
        attention_masks = batch['attention_mask'].to(text_encoder.device)

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

@timed_metric
def evaluate_cirr(
    model: TwoEncoderVLM,
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

    image_features, text_features, reference_names, target_names, group_members, pair_ids = generate_cirr_triplet_features(
        clip_model=model,
        triplet_dataset=cirr_triplets,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tqdm=tqdm,
    )

    predicted_features = fusion(
        image_features=image_features,
        text_features=text_features,
        fusion_type=fusion_type,
        alpha=0.7
    )

    metrics = compute_cirr_metrics(
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

    if return_index_tuple:
        return metrics, (index_features, index_names)
    return metrics


def generate_cirr_test_submission(
    model: TwoEncoderVLM,
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

    image_features, text_features, reference_names, target_names, group_members, pair_ids = generate_cirr_triplet_features(
        clip_model=model,
        triplet_dataset=cirr_triplets,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tqdm=tqdm,
        skip_targets=True
    )

    predicted_features = fusion(
        image_features=image_features,
        text_features=text_features,
        fusion_type=fusion_type,
        alpha=0.7
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