# -*- coding: utf-8 -*-
"""
Load a saved checkpoint and evaluate it on the test split.

Run:
    python src/evaluate.py --data-dir augmented --checkpoint results/best_model.pth --out-dir results
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from dataset import MRISegmentationDataset
from model import UNetVGG16
from train import evaluate_test_set


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test set")
    parser.add_argument("--data-dir", type=str, default="augmented")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = MRISegmentationDataset(
        os.path.join(args.data_dir, "test", "images"), os.path.join(args.data_dir, "test", "masks")
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = UNetVGG16()
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    evaluate_test_set(model, test_loader, device, args.out_dir)


if __name__ == "__main__":
    main()
