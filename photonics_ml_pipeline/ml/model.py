"""Feed-forward classifier consuming physics-derived features.

Purpose:
    A small, fully-connected network that maps standardized feature vectors to class
    logits. Deliberately simple: the physics does the heavy lifting via the features.

Assumptions:
    - Input already standardized (zero mean, unit variance per feature).
Limitations:
    - MLP only; swap in a 1-D CNN for raw-field input if desired.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

__all__ = ["FeatureMLP"]


class FeatureMLP(nn.Module):
    """Multilayer perceptron for feature-vector classification."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], n_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of feature vectors."""
        return self.net(x)
