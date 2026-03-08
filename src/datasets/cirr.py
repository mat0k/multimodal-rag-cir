import os
import json
from collections import OrderedDict, defaultdict
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import itertools
from typing import Callable, Optional, Tuple

from torch.utils.data import Dataset


class CIRR(Dataset):
    '''
    Args:
        images_dirpath (str): Directory where images are stored.
        annotations_dirpath (str): Directory where annotations are stored.
        split (str): Dataset split, one of ['train', 'val', 'test1'].
        image_transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        caption_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        max_length_tokenizer (int): The maximum length required by some text tokenizers,
        mode (str): Mode of operation, one of ['triplets', 'images']. If 'triplets', returns triplets of images and captions.
            If 'images', returns only images.
    '''

    def __init__(
        self,
        images_dirpath: str,
        annotations_dirpath: str,
        split: str = 'train',
        image_transform: Optional[Callable] = None,
        caption_transform: Optional[Callable] = None,
        max_length_tokenizer: int = 77,
        mode = 'triplets'  # 'triplets' or 'images'
    ):
        super(CIRR, self).__init__()

        if split == 'test':
            split = 'test1'

        assert split in ['train', 'val', 'test1'], f"split must be one of ['train', 'val', 'test1'], found {split} instead."
        assert mode in ['triplets', 'images'], f"mode must be one of ['triplets', 'images'], found {mode} instead."

        self.name = 'CIRR'
        self.split = split
        self.images_dirpath = images_dirpath
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.max_length_tokenizer = max_length_tokenizer
        self.mode = mode

        #load annotated triplets
        with open(os.path.join(annotations_dirpath, "captions", f"cap.rc2.{split}.json"), 'r') as f:
            self.triplets = json.load(f)

        # get image mapping
        with open(os.path.join(annotations_dirpath, "image_splits", f"split.rc2.{split}.json"), 'r') as f:
            self.name_to_relpath = json.load(f)


    def __getitem__(self, index: int):
        """
        Args:
            index (int): index in [0, self.__len__())

        Returns:
            If mode is 'triplets' and split is 'test1':
                tuple: Tuple (pair_id, reference_name, transformed_caption, caption, group_members).
            If mode is 'triplets' and split is not 'test1':
                tuple: Tuple (reference, target, tranformed_query, text_query).
            If mode is 'images':
                tuple: Tuple (image_name, image), where image_name is the name of the image
        """
        if self.mode == 'triplets':
            triplet = self.triplets[index]

            # Load reference image
            reference_name = triplet['reference']
            reference = Image.open(os.path.join(self.images_dirpath, self.name_to_relpath[reference_name])).convert('RGB')
            if self.image_transform is not None:
                reference = self.image_transform(reference, return_tensors='pt')['pixel_values'][0]

            # Load target image (not available for test split)
            target = None
            if self.split != 'test1':
                target_name = triplet['target_hard']
                target = Image.open(os.path.join(self.images_dirpath, self.name_to_relpath[target_name])).convert('RGB')
                if self.image_transform is not None:
                    target = self.image_transform(target, return_tensors='pt')['pixel_values'][0]

            #load caption
            caption = triplet['caption']
            transformed_caption = caption
            if self.caption_transform is not None:
                transformed_caption = self.caption_transform(
                    caption,
                    padding='max_length',
                    max_length=self.max_length_tokenizer,
                    truncation=True,
                    return_tensors='pt')
                
            # load other info
            pair_id = triplet['pairid']
            group_members = triplet['img_set']['members']
            
            if self.split == 'test1':
                    return {
                    'pair_id': pair_id,
                    'reference_name': reference_name,
                    'reference': reference,
                    'transformed_caption': transformed_caption["input_ids"][0],
                    'attention_mask': transformed_caption["attention_mask"][0],
                    'caption': caption,
                    'group_members': group_members
                }

            return {
                'pair_id': pair_id,
                'reference_name': reference_name,
                'reference': reference,
                'target': target,
                'target_name': target_name,
                'transformed_caption': transformed_caption["input_ids"][0],
                'attention_mask': transformed_caption["attention_mask"][0],
                'caption': caption,
                'group_members': group_members,
            }
        
        elif self.mode == 'images':
            image_name = list(self.name_to_relpath.keys())[index]
            image = Image.open(os.path.join(self.images_dirpath, self.name_to_relpath[image_name])).convert('RGB')
            if self.image_transform is not None:
                image = self.image_transform(image, return_tensors='pt')['pixel_values'][0]
            return {
                'image_name': image_name,
                'image': image
            }
        
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Use 'triplets' or 'images'.")


    def __len__(self) -> int:
        if self.mode == 'triplets':
            return len(self.triplets)
        else:
            return len(self.name_to_relpath)


def build_cirr_dataset(
    split: str = 'train',
    image_transform: Optional[Callable] = None,
    caption_transform: Optional[Callable] = None,
    max_length_tokenizer: int = 77,
    mode = 'triplets'  # 'triplets' or 'images'
) -> CIRR:
    dataset = CIRR(
        images_dirpath="data/cirr/images",
        annotations_dirpath="data/cirr/annotations",
        split=split,
        image_transform=image_transform,
        caption_transform=caption_transform,
        max_length_tokenizer=max_length_tokenizer,
        mode=mode
    )
    return dataset