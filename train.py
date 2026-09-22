# -*- coding: utf-8 -*-
"""
Training loop for U-Net + VGG16 encoder on the (prepared) LGG MRI dataset.

Run:
    python src/train.py --data-dir augmented --epochs 30 --batch-size 4 --out-dir results
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MRISegmentationDataset
from metrics import calculate_dice, calculate_iou
from model import DiceLoss, UNetVGG16
from utils import set_seed, show_prediction


def get_loaders(data_dir: str, batch_size: int):
    train_dataset = MRISegmentationDataset(
        os.path.join(data_dir, "train", "images"), os.path.join(data_dir, "train", "masks")
    )
    val_dataset = MRISegmentationDataset(
        os.path.join(data_dir, "val", "images"), os.path.join(data_dir, "val", "masks")
    )
    test_dataset = MRISegmentationDataset(
        os.path.join(data_dir, "test", "images"), os.path.join(data_dir, "test", "masks")
    )

    print("Train:", len(train_dataset))
    print("Val:", len(val_dataset))
    print("Test:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, val_loader, test_loader


def plot_history(history: dict, out_dir: str):
    epochs = range(1, len(history["train_loss"]) + 1)

    for metric in ["loss", "dice", "iou"]:
        plt.figure()
        plt.plot(epochs, history[f"train_{metric}"], label=f"Train {metric.capitalize()}")
        plt.plot(epochs, history[f"val_{metric}"], label=f"Val {metric.capitalize()}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.title(f"{metric.capitalize()} over Epochs")
        plt.savefig(os.path.join(out_dir, f"{metric}_curve.png"), bbox_inches="tight")
        plt.close()


def evaluate_test_set(model, test_loader, device, out_dir: str, n_examples: int = 10):
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    model.eval()

    counter = 0
    for images, masks in test_loader:
        preds = model(images.to(device))
        show_prediction(
            images[0].cpu(),
            preds[0].cpu(),
            masks[0].cpu(),
            save_path=os.path.join(out_dir, "predictions", f"test_{counter}.png"),
        )
        counter += 1
        if counter == n_examples:
            break

    test_dice, test_iou, test_ccr = [], [], []
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            test_dice.append(calculate_dice(preds, masks))
            test_iou.append(calculate_iou(preds, masks))

            pred_mask = (torch.sigmoid(preds) > 0.5).float()
            true_mask = (masks > 0.5).float()
            correct = ((true_mask == 0) & (pred_mask == 0)).sum().item() + (
                (true_mask == 1) & (pred_mask == 1)
            ).sum().item()
            test_ccr.append(correct / true_mask.numel())

    import numpy as np

    print("\n=== Test Results ===")
    print(f"Test Dice Coefficient (mean): {np.mean(test_dice):.4f}")
    print(f"Test IoU Score (mean): {np.mean(test_iou):.4f}")
    print(f"Test CCR (mean): {np.mean(test_ccr):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train U-Net + VGG16 for MRI tumor segmentation")
    parser.add_argument("--data-dir", type=str, default="augmented")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    model = UNetVGG16()
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = DiceLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_iou": [], "val_iou": [],
    }
    best_val_dice = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss, dice_epoch, iou_epoch = 0.0, 0.0, 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            dice_epoch += calculate_dice(preds, masks)
            iou_epoch += calculate_iou(preds, masks)

        history["train_loss"].append(total_loss / len(train_loader))
        history["train_dice"].append(dice_epoch / len(train_loader))
        history["train_iou"].append(iou_epoch / len(train_loader))

        model.eval()
        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                loss = criterion(preds, masks)
                val_loss += loss.item()
                val_dice += calculate_dice(preds, masks)
                val_iou += calculate_iou(preds, masks)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {history['train_loss'][-1]:.4f} | "
            f"Train Dice: {history['train_dice'][-1]:.4f} | "
            f"Train IoU: {history['train_iou'][-1]:.4f}"
        )
        print(
            f"                 | Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pth"))

    torch.save(model.state_dict(), os.path.join(args.out_dir, "last_model.pth"))
    plot_history(history, args.out_dir)
    evaluate_test_set(model, test_loader, device, args.out_dir)


if __name__ == "__main__":
    main()
