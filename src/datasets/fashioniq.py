import os
import json
from collections import OrderedDict, defaultdict
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import itertools
from typing import Callable, Literal, Optional, Tuple

from torch.utils.data import Dataset


class FashionIQ(Dataset):
    '''
    Args:
        root (string): Root directory where images are downloaded to.
        annotations_file (string): Path to annotation file.
        image_transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        caption_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        max_length_tokenizer (int): The maximum length required by some text tokenizers,
            used to truncate captions if necessary.
        mode (str): Whether to return triplets of (candidate, caption, target) or just images. Options are 'triplets' or 'images'. 

    '''

    def __init__(
        self,
        images_path: str,
        annotations_path: str,
        logs_path: str,
        split: Literal['train', 'val', 'test'] = 'val',
        image_transform: Optional[Callable] = None,
        caption_transform: Optional[Callable] = None,
        max_length_tokenizer: int = 77,
        mode: Literal['triplets', 'images'] = 'triplets',
        caption_joiner: str = ' ',
        reverse_caption_order: bool = False,
    ):
        super(FashionIQ, self).__init__()

        assert split in ['train', 'val', 'test'], f"split must be one of ['train', 'val', 'test'], found {split} instead."

        self.name = 'FashionIQ'
        self.split = split
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.max_length_tokenizer = max_length_tokenizer
        self.caption_joiner = caption_joiner
        self.reverse_caption_order = reverse_caption_order

        self.classes = ['dress', 'shirt', 'toptee']
        self.mode = mode

        self.annotations = {cls: [] for cls in self.classes}
        self.image_paths = {cls: None for cls in self.classes}
        self.images = {cls: [] for cls in self.classes}

        for cls in self.classes:
            #load annotations
            ann = os.path.join(annotations_path, f"cap.{cls}.{self.split}.json")
            if not os.path.exists(ann):
                raise FileNotFoundError(f"Annotation file {ann} does not exist.")
            with open(ann, 'r') as f:
                annotations = json.load(f)
            #load image paths
            self.image_paths[cls] = os.path.join(images_path, cls)
            #load missing files
            log_path = os.path.join(logs_path, f"missing_{cls}.log")
            if not os.path.exists(log_path):
                print( Warning(f"Missing file log {log_path} does not exist."))
                missing_files = []
            else:
                with open(log_path, 'r') as f:
                    missing_files = f.read().splitlines()
            # remove missing files from annotations
            for ann in annotations:
                candidate_path = self.get_image_path(cls, ann['candidate'])
                has_target = 'target' in ann
                target_exists = True
                target_not_missing = True

                if has_target:
                    target_path = self.get_image_path(cls, ann['target'])
                    target_exists = os.path.exists(target_path)
                    target_not_missing = ann['target'] not in missing_files

                if ann['candidate'] not in missing_files and os.path.exists(candidate_path) and target_exists and target_not_missing:
                    self.annotations[cls].append(ann)
            # save all images for the class and split
            img_file = os.path.join(annotations_path, f"split.{cls}.{self.split}.json")
            with open(img_file, 'r') as f:
                images_raw = json.load(f)
            for img in images_raw:
                img_path = self.get_image_path(cls, img)
                if img not in missing_files and os.path.exists(img_path):
                    self.images[cls].append(img)

        if self.mode == 'triplets':
            self.lengths = {cls: len(self.annotations[cls]) for cls in self.classes}
        elif self.mode == 'images':
            self.lengths = {cls: len(self.images[cls]) for cls in self.classes}

    def get_class_index(self, index:int) -> Tuple[str, int]:
        """
        Get the class and index of a triplet for a given global index in the dataset.
        """
        cumulative_length = 0
        for cls, length in self.lengths.items():
            cumulative_length += length
            if index < cumulative_length:
                return cls, index - (cumulative_length - length)
            
    def get_image_path(self, cls: str, image_name: str) -> str:
        """
        Get the full path of an image given its class and name.
        """
        return os.path.join(self.image_paths[cls], image_name + '.jpg')


    def __getitem__(self, index: int):
        """
        Args:
            index (int): index in [0, self.__len__())

        Returns:
            If mode is 'triplets':
                dict: A dictionary with keys 'class', 'candidate', 'candidate_name', 'target', 'target_name', 'transformed_caption', 'attention_mask'.
            If mode is 'images':
                dict: A dictionary with keys 'class', 'image', 'image_name'.
        """

        cls, local_index = self.get_class_index(index)
        if self.mode == 'triplets':
            triplet = self.annotations[cls][local_index]

            candidate_path = self.get_image_path(cls, triplet['candidate'])
            candidate = Image.open(candidate_path).convert('RGB')
            target_name = triplet.get('target')
            target = None

            if target_name is not None:
                target_path = self.get_image_path(cls, target_name)
                target = Image.open(target_path).convert('RGB')

            if self.image_transform is not None:
                candidate = self.image_transform(candidate, return_tensors='pt')['pixel_values'][0]
                if target is not None:
                    target = self.image_transform(target, return_tensors='pt')['pixel_values'][0]

            # join all captions into one string
            captions_list = list(triplet["captions"])
            if self.reverse_caption_order:
                captions_list = list(reversed(captions_list))

            captions = self.caption_joiner.join(captions_list)
            transformed_captions = captions

            if self.caption_transform is not None:
                transformed_captions = self.caption_transform(
                    captions,
                    padding='max_length',
                    max_length=self.max_length_tokenizer,
                    truncation=True,
                    return_tensors='pt')

            def get_caption_field(tc, field):
                # Handle tokenizer outputs such as dict or BatchEncoding.
                if hasattr(tc, "keys") and field in tc:
                    return tc[field][0]
                # fallback: return as-is (string or tensor)
                return tc

            sample = {
                'class': cls,
                'candidate': candidate,
                'candidate_name': triplet["candidate"],
                'target_name': target_name if target_name is not None else "",
                'transformed_caption': get_caption_field(transformed_captions, "input_ids"),
                'attention_mask': get_caption_field(transformed_captions, "attention_mask"),
            }

            if target is not None:
                sample['target'] = target

            return sample
        elif self.mode == 'images':
            image_name = self.images[cls][local_index]
            image_path = self.get_image_path(cls, image_name)
            image = Image.open(image_path).convert('RGB')

            if self.image_transform is not None:
                image = self.image_transform(image, return_tensors='pt')['pixel_values'][0]

            return {
                'class': cls,
                'image': image,
                'image_name': image_name,
            }


    def __len__(self) -> int:
        return sum(self.lengths.values())


def build_fashioniq_dataset(
    split: Literal['train', 'val', 'test'] = 'val',
    image_transform: Optional[Callable] = None,
    caption_transform: Optional[Callable] = None,
    max_length_tokenizer: int = 77,
    mode: Literal['triplets', 'images'] = 'triplets',  # 'triplets' or 'images',
    caption_joiner: str = ' ',
    reverse_caption_order: bool = False,
):
    return FashionIQ(
        images_path="data/fashioniq/images",
        annotations_path="data/fashioniq/annotations",
        logs_path="data/fashioniq/logs",
        split=split,
        image_transform=image_transform,
        caption_transform=caption_transform,
        max_length_tokenizer=max_length_tokenizer,
        mode=mode,
        caption_joiner=caption_joiner,
        reverse_caption_order=reverse_caption_order,
    )