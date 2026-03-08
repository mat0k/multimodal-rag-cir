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
        split: Literal['train', 'val'] = 'val',
        image_transform: Optional[Callable] = None,
        caption_transform: Optional[Callable] = None,
        max_length_tokenizer: int = 77,
        mode: Literal['triplets', 'images'] = 'triplets'
    ):
        super(FashionIQ, self).__init__()

        assert split in ['train', 'val'], f"split must be one of ['train', 'val'], found {split} instead."

        self.name = 'FashionIQ'
        self.split = split
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.max_length_tokenizer = max_length_tokenizer

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
                target_path = self.get_image_path(cls, ann['target'])
                if ann['candidate'] not in missing_files and ann['target'] not in missing_files and os.path.exists(candidate_path) and os.path.exists(target_path):
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
            target_path = self.get_image_path(cls, triplet['target'])
            candidate = Image.open(candidate_path).convert('RGB')
            target = Image.open(target_path).convert('RGB')

            if self.image_transform is not None:
                candidate = self.image_transform(candidate, return_tensors='pt')['pixel_values'][0]
                target = self.image_transform(target, return_tensors='pt')['pixel_values'][0]

            #join all captions into one string separated by [SEP] token
            captions = " ".join(triplet["captions"])
                
            if self.caption_transform is not None:
                transformed_captions = self.caption_transform(
                    captions,
                    padding='max_length',
                    max_length=self.max_length_tokenizer,
                    truncation=True,
                    return_tensors='pt')

            return {
                'class': cls,
                'candidate': candidate,
                'candidate_name': triplet["candidate"],
                'target': target,
                'target_name': triplet["target"],
                'transformed_caption': transformed_captions["input_ids"][0],
                'attention_mask': transformed_captions["attention_mask"][0],
            }
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
    split: Literal['train', 'val'] = 'val',
    image_transform: Optional[Callable] = None,
    caption_transform: Optional[Callable] = None,
    max_length_tokenizer: int = 77,
    mode: Literal['triplets', 'images'] = 'triplets',  # 'triplets' or 'images',
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
    )