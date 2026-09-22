# -*- coding: utf-8 -*-
"""U-Net with a pretrained VGG16 encoder, and the Dice loss used to train it."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )


class UNetVGG16(nn.Module):
    """
    Encoder: VGG16 convolutional feature extractor (ImageNet-pretrained, frozen).
    Decoder: a stack of transposed-conv upsampling blocks back to the input resolution.
    Output: single-channel logits (apply sigmoid for a 0-1 tumor probability map).
    """

    def __init__(self, pretrained: bool = True, freeze_encoder: bool = True):
        super().__init__()
        vgg = models.vgg16(weights="IMAGENET1K_V1" if pretrained else None)
        self.encoder = vgg.features

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        return x
