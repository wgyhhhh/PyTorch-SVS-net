"""Losses and metrics used by the PyTorch SVS-Net scripts."""

from __future__ import annotations

import torch
from torch import nn


def dice_coefficient(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    y_true = y_true.reshape(y_true.shape[0], -1)
    intersection = (y_pred * y_true).sum(dim=1)
    denominator = y_pred.sum(dim=1) + y_true.sum(dim=1)
    return ((2.0 * intersection + eps) / (denominator + eps)).mean()


class DiceLoss(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return 1.0 - dice_coefficient(y_pred, y_true)


class NegativeDiceLoss(nn.Module):
    """Alternative Dice objective that directly maximizes the Dice score."""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return -dice_coefficient(y_pred, y_true)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.clamp(self.eps, 1.0 - self.eps)
        pt1 = torch.where(y_true == 1, y_pred, torch.ones_like(y_pred))
        pt0 = torch.where(y_true == 0, y_pred, torch.zeros_like(y_pred))
        loss_pos = self.alpha * (1.0 - pt1).pow(self.gamma) * pt1.log()
        loss_neg = (1.0 - self.alpha) * pt0.pow(self.gamma) * (1.0 - pt0).log()
        return -(loss_pos + loss_neg).mean()


def binary_scores(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> dict[str, float]:
    pred = (y_pred >= threshold).float()
    true = (y_true >= threshold).float()
    tp = (pred * true).sum()
    fp = (pred * (1.0 - true)).sum()
    fn = ((1.0 - pred) * true).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    dice = dice_coefficient(pred, true, eps=eps)
    return {
        "precision": float(precision.detach().cpu()),
        "recall": float(recall.detach().cpu()),
        "f1": float(f1.detach().cpu()),
        "dice": float(dice.detach().cpu()),
    }
