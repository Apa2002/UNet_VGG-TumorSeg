# -*- coding: utf-8 -*-
"""Segmentation metrics: IoU and Dice coefficient."""

import torch


def calculate_iou(preds, masks, threshold: float = 0.5) -> float:
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = ((preds + masks) >= 1).float().sum(dim=(1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def calculate_dice(preds, masks, threshold: float = 0.5) -> float:
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    total = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (total + 1e-6)
    return dice.mean().item()
