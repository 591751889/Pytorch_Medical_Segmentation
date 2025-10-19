from glob import glob
from os.path import dirname, join, basename, isfile
import sys

sys.path.append('./')
import csv
import torch
from medpy.io import load
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
import torchio as tio
from torchio import AFFINE, DATA
import torchio
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
from torchio.data import UniformSampler
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)
from pathlib import Path

from hparam import hparams as hp


class MedData_train(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):

        if hp.mode == '3d':
            patch_size = hp.patch_size
        elif hp.mode == '2d':
            patch_size = hp.patch_size
        else:
            raise Exception('no such kind of mode!')

        queue_length = 5
        samples_per_volume = 5

        self.subjects = []

        if (hp.in_class == 1) and (hp.out_class == 1) or (hp.in_class == 3) and (hp.out_class == 1):

            images_dir = Path(images_dir)

            self.image_paths = sorted(images_dir.glob(hp.fold_arch))
            labels_dir = Path(labels_dir)
            self.label_paths = sorted(labels_dir.glob(hp.fold_arch))

            for (image_path, label_path) in zip(self.image_paths, self.label_paths):
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    label=tio.LabelMap(label_path),
                )
                self.subjects.append(subject)

        self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)

        self.queue_dataset = Queue(
            self.training_set,
            queue_length,
            samples_per_volume,
            UniformSampler(patch_size),
        )

    def transform(self):

        if hp.mode == '3d':
            if hp.aug:
                training_transform = Compose([
                    # ToCanonical(),
                    CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                    # RandomMotion(),
                    RandomBiasField(),
                    ZNormalization(),
                    RandomNoise(),
                    RandomFlip(axes=(0,)),
                    OneOf({
                        RandomAffine(): 0.8,
                        RandomElasticDeformation(): 0.2,
                    }), ])
            else:
                training_transform = Compose([
                    CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                    ZNormalization(),
                ])
        elif hp.mode == '2d':
            if hp.aug:
                training_transform = Compose([
                    CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                    # RandomMotion(),
                    RandomBiasField(),
                    ZNormalization(),
                    RandomNoise(),
                    RandomFlip(axes=(0,)),
                    OneOf({
                        RandomAffine(): 0.8,
                        RandomElasticDeformation(): 0.2,
                    }), ])
            else:
                training_transform = Compose([
                    CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                    ZNormalization(),
                ])

        else:
            raise Exception('no such kind of mode!')

        return training_transform
import torch
from pathlib import Path
import torchio as tio
from torchio.data import Queue, UniformSampler

class MedData_val(torch.utils.data.Dataset):
    """
    Validation-time dataset using torchio.
    Applies only deterministic preprocessing (no augmentation).
    """
    def __init__(self, images_dir: str, labels_dir: str):
        # ➊ 确定 patch / crop 尺寸
        if hp.mode in {'2d', '3d'}:
            patch_size = hp.patch_size
        else:
            raise ValueError(f"Unsupported mode {hp.mode}")

        # ➋ 读取影像与标签路径
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        image_paths = sorted(images_dir.glob(hp.fold_arch))
        label_paths = sorted(labels_dir.glob(hp.fold_arch))

        self.subjects = [
            tio.Subject(
                source=tio.ScalarImage(img_p),
                label=tio.LabelMap(lbl_p),
            )
            for img_p, lbl_p in zip(image_paths, label_paths)
        ]

        # ➌ 定义仅包含确定性操作的 transform
        self.transforms = self._build_transform()

        # ➍ 构建 SubjectsDataset 和 Queue（按需取 patch）
        val_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)

        self.queue_dataset = Queue(
            val_set,
            1,          # 评估时通常只需保持 1
            samples_per_volume=1,     # 若想多 patch 评估可调大
            sampler=UniformSampler(patch_size),
            shuffle_subjects=False,   # 保持顺序一致
            num_workers=0
        )

    def _build_transform(self):
        """仅包含确定性预处理；无随机增强"""
        common_ops = [
            CropOrPad(hp.crop_or_pad_size, padding_mode='reflect'),
            ZNormalization(),
        ]
        return Compose(common_ops)

    def __len__(self):
        return len(self.queue_dataset)

    def __getitem__(self, idx):
        return self.queue_dataset[idx]


class MedData_test(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):

        self.subjects = []

        if (hp.in_class == 1) and (hp.out_class == 1):

            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(hp.fold_arch))
            labels_dir = Path(labels_dir)
            self.label_paths = sorted(labels_dir.glob(hp.fold_arch))

            for (image_path, label_path) in zip(self.image_paths, self.label_paths):
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    label=tio.LabelMap(label_path),
                )
                self.subjects.append(subject)


        # self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=None)


