"""
@author haoyu
based to official implementation of  Diffusion-DPO at
https://github.com/SalesforceAIResearch/DiffusionDPO/blob/main/train.py

先实现再迭代优化
"""

import os
import random
import json
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import glob
import pandas as pd
import yaml

from datasets import load_dataset
from torchvision import transforms
import io
from PIL import Image


# def preprocess_train(examples):
#     # Preprocessing the datasets.
#     train_transforms = transforms.Compose(
#         [
#             transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
#             transforms.RandomCrop(resolution) if args.random_crop else transforms.CenterCrop(resolution),
#             transforms.Lambda(lambda x: x) if args.no_hflip else transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5]),
#         ]
#     )
#     all_pixel_values = []
#     for col_name in ['jpg_0', 'jpg_1']:
#         images = [Image.open(io.BytesIO(im_bytes)).convert("RGB")
#                     for im_bytes in examples[col_name]]
#         pixel_values = [train_transforms(image) for image in images]
#         all_pixel_values.append(pixel_values)
#     # Double on channel dim, jpg_y then jpg_w
#     im_tup_iterator = zip(*all_pixel_values)
#     combined_pixel_values = []
#     for im_tup, label_0 in zip(im_tup_iterator, examples['label_0']):
#         combined_im = torch.cat(im_tup, dim=0) # no batch dim
#         combined_pixel_values.append(combined_im)
#     examples["pixel_values"] = combined_pixel_values
#     return examples


class PicAPic(Dataset):
    def __init__(
        self,
        data_root,
        resolution,
        subset_split="all",
        dreamlike_pairs_only=False,
        max_train_samples=None,
        random_crop=False,
        no_hflip=False,
        seed=None,
    ):
        self.data_root = data_root
        self.resolution = resolution
        self.subset_split = subset_split
        assert self.subset_split in ["train", "test", "all"]
        if subset_split == "all":
            data_files = None
        else:
            data_files = {}
            data_files[self.subset_split] = os.path.join(self.data_root, "**")

        dataset = load_dataset(cache_dir="/home/haoyu/research/dataset/picapic_v2/data")

        # Preprocessing the datasets.
        # eliminate no-decisions (0.5-0.5 labels)
        orig_len = dataset[self.subset_split].num_rows
        not_split_idx = [
            i
            for i, label_0 in enumerate(dataset[self.subset_split]["label_0"])
            if label_0 in (0, 1)
        ]
        dataset[self.subset_split] = dataset[self.subset_split].select(not_split_idx)
        new_len = dataset[self.subset_split].num_rows
        print(
            f"Eliminated {orig_len - new_len}/{orig_len} split decisions for Pick-a-pic"
        )

        # Below if if want to train on just the Dreamlike vs dreamlike pairs
        if dreamlike_pairs_only:
            orig_len = dataset[self.subset_split].num_rows
            dream_like_idx = [
                i
                for i, (m0, m1) in enumerate(
                    zip(
                        dataset[self.subset_split]["model_0"],
                        dataset[self.subset_split]["model_1"],
                    )
                )
                if (("dream" in m0) and ("dream" in m1))
            ]
            dataset[self.subset_split] = dataset[self.subset_split].select(
                dream_like_idx
            )
            new_len = dataset[self.subset_split].num_rows
            print(
                f"Eliminated {orig_len - new_len}/{orig_len} non-dreamlike gens for Pick-a-pic"
            )

        if max_train_samples is not None:
            dataset[self.subset_split] = (
                dataset[self.subset_split]
                .shuffle(seed=seed)
                .select(range(max_train_samples))
            )
        # Set the training transforms
        print("Ignoring image_column variable, reading from jpg_0 and jpg_1")
        self.train_dataset = dataset[
            self.subset_split
        ]  # .with_transform(preprocess_train)
        # Preprocessing the datasets.
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(
                    resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.RandomCrop(resolution)
                if random_crop
                else transforms.CenterCrop(resolution),
                transforms.Lambda(lambda x: x)
                if no_hflip
                else transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def preprocess_train(self, examples):
            all_pixel_values = []
            for col_name in ["jpg_0", "jpg_1"]:
                images = [
                    Image.open(io.BytesIO(im_bytes)).convert("RGB")
                    for im_bytes in examples[col_name]
                ]
                pixel_values = [self.train_transforms(image) for image in images]
                all_pixel_values.append(pixel_values)
            # Double on channel dim, jpg_y then jpg_w
            im_tup_iterator = zip(*all_pixel_values)
            combined_pixel_values = []
            for im_tup, label_0 in zip(im_tup_iterator, examples["label_0"]):
                if (
                    label_0 == 0
                ):  # don't want to flip things if using choice_model for AI feedback
                    im_tup = im_tup[::-1]
                combined_im = torch.cat(im_tup, dim=0)  # no batch dim
                combined_pixel_values.append(combined_im)
            examples["pixel_values"] = combined_pixel_values
            return examples

    def __getitem__(self, index):
        example = self.train_dataset[index]
        # y_w and y_l were concatenated along channel dimension
        # If using AIF then we haven't ranked yet so do so now
        # Only implemented for BS=1 (assert-protected)
        # a video has 1 frame is a video
        return {"video": example["pixel_values"], "caption": example["caption"]}


if __name__ == "__main__":
    PicAPic(data_root="/home/haoyu/research/dataset/picapic_v2/data", resolution=512)
