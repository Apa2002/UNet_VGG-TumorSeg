# -*- coding: utf-8 -*-
"""
Data preparation & augmentation for LGG MRI Segmentation.

- Splits the dataset into train/val/test (80/10/10)
- Applies Albumentations (3x augmentation on the train split only)
- Saves paired images/masks into ./augmented/<split>/{images,masks}

Run:
    python src/data_prep.py --base-dir /path/to/lgg-mri-segmentation/kaggle_3m --out-dir augmented
"""

import argparse
import os
from glob import glob

import albumentations as A
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_augmentation_pipeline() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ElasticTransform(p=0.2),
    ])


def collect_pairs(base_dir: str):
    """Masks follow '*_mask.tif'; the matching image has the same name without '_mask'."""
    mask_paths = sorted(glob(os.path.join(base_dir, "*", "*_mask.tif")))
    image_paths = [p.replace("_mask", "") for p in mask_paths]
    return image_paths, mask_paths


def save_pairs(images, masks, subset, out_dir, augment, augment_data=False):
    """
    Reads grayscale images & masks, applies optional augmentation,
    and saves them under <out_dir>/<subset>/{images,masks}.
    - For training, if augment_data=True, creates 3 copies per pair (1 original-style + augmented).
    - For val/test, keeps a single copy without augmentation.
    """
    img_out = os.path.join(out_dir, subset, "images")
    mask_out = os.path.join(out_dir, subset, "masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    for i, (img_path, mask_path) in tqdm(
        enumerate(zip(images, masks)), total=len(images), desc=subset.upper()
    ):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"Skipped (could not read): {img_path} or {mask_path}")
            continue

        num_copies = 3 if augment_data else 1

        for j in range(num_copies):
            if augment_data:
                transformed = augment(image=img, mask=mask)
                aug_img, aug_mask = transformed["image"], transformed["mask"]
            else:
                aug_img, aug_mask = img, mask

            cv2.imwrite(os.path.join(img_out, f"img_{i}_{j}.tif"), aug_img)
            cv2.imwrite(os.path.join(mask_out, f"mask_{i}_{j}.tif"), aug_mask)


def main():
    parser = argparse.ArgumentParser(description="Prepare & augment the LGG MRI dataset")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/kaggle/input/lgg-mri-segmentation/lgg-mri-segmentation/kaggle_3m",
        help="Path to the kaggle_3m folder of the LGG dataset",
    )
    parser.add_argument(
        "--out-dir", type=str, default="augmented", help="Where to write the prepared dataset"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image_paths, mask_paths = collect_pairs(args.base_dir)

    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=args.seed
    )
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs, temp_masks, test_size=0.5, random_state=args.seed
    )

    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

    augment = get_augmentation_pipeline()

    save_pairs(train_imgs, train_masks, "train", args.out_dir, augment, augment_data=True)
    save_pairs(val_imgs, val_masks, "val", args.out_dir, augment, augment_data=False)
    save_pairs(test_imgs, test_masks, "test", args.out_dir, augment, augment_data=False)

    print("Augmented images and masks saved successfully.")


if __name__ == "__main__":
    main()
