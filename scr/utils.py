# -*- coding: utf-8 -*-
"""Small helpers for visualizing predictions and setting seeds."""

import random

import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def show_prediction(img, pred_mask, true_mask, save_path: str = None):
    """
    img: tensor (3, H, W)
    pred_mask, true_mask: tensor (1, H, W)
    """
    img = img.permute(1, 2, 0).cpu().numpy()
    pred_mask = torch.sigmoid(pred_mask).squeeze().cpu().detach().numpy()
    true_mask = true_mask.squeeze().cpu().numpy()
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Input Image")
    axs[1].imshow(true_mask, cmap="gray")
    axs[1].set_title("Ground Truth Mask")
    axs[2].imshow(pred_mask, cmap="gray")
    axs[2].set_title("Predicted Mask")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
