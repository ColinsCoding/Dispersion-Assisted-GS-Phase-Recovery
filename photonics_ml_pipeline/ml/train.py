"""Training loop for the feature classifier.

Purpose:
    Deterministic train/validation split, Adam optimization, and per-epoch metric
    tracking. Returns the trained model plus a history for the timing/accuracy plots.

Assumptions:
    - CPU training is sufficient (tiny model, tabular features) -- no GPU dependence.
Limitations:
    - Single holdout split (no k-fold).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from ml.model import FeatureMLP

__all__ = ["TrainResult", "train_model"]


@dataclass
class TrainResult:
    """Outcome of a training run."""

    model: FeatureMLP
    history: dict[str, list[float]] = field(default_factory=dict)
    val_accuracy: float = 0.0


def _accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb).argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
    return correct / total if total else 0.0


def train_model(
    dataset: Dataset,
    input_dim: int,
    n_classes: int,
    hidden_dims: tuple[int, ...] = (32, 16),
    epochs: int = 60,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    val_fraction: float = 0.25,
    seed: int = 0,
) -> TrainResult:
    """Train a `FeatureMLP` and return the model with its metric history."""
    torch.manual_seed(seed)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = FeatureMLP(input_dim, hidden_dims, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_accuracy": []}
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * xb.shape[0]
        history["train_loss"].append(epoch_loss / n_train)
        history["val_accuracy"].append(_accuracy(model, val_loader))

    return TrainResult(model=model, history=history, val_accuracy=history["val_accuracy"][-1])
