# -*- coding: utf-8 -*-
"""PyTorch Dataset for the (already augmented/prepared) LGG MRI segmentation data."""

import os
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MRISegmentationDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, img_size: int = 256):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.tif")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.tif")))
        self.img_size = img_size

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(
                f"Mismatch between number of images ({len(self.image_paths)}) "
                f"and masks ({len(self.mask_paths)}) in {image_dir} / {mask_dir}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Replicate grayscale to 3 channels to match the VGG16 encoder's expected input.
        img = np.repeat(np.expand_dims(img, axis=0), 3, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
