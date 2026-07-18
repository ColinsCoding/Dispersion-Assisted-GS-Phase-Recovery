"""Inference and evaluation helpers.

Purpose:
    Batch prediction and a dependency-free confusion matrix for the ML stage.

Limitations:
    - Confusion matrix assumes integer labels in [0, n_classes).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

__all__ = ["predict", "confusion_matrix"]


def predict(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Return predicted class indices for a batch of feature vectors."""
    model.eval()
    with torch.no_grad():
        return model(features).argmax(dim=1)


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int) -> np.ndarray:
    """Row = true class, column = predicted class."""
    yt = y_true.cpu().numpy().astype(int)
    yp = y_pred.cpu().numpy().astype(int)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(yt, yp):
        matrix[t, p] += 1
    return matrix
