"""Matplotlib figure builders for every pipeline stage.

Purpose:
    Single-responsibility functions that each return a `matplotlib.figure.Figure`, so
    callers decide whether to show or save. Uses the non-interactive 'Agg' backend by
    default for headless/CI use.

Limitations:
    - Static figures; animation helpers can be layered on top of `plot_optical_field`.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = [
    "plot_optical_field",
    "plot_training_history",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_timing",
]


def plot_optical_field(z_um: np.ndarray, width_um: np.ndarray, title: str = "Gaussian beam width") -> plt.Figure:
    """Plot beam width w(z) versus axial position."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(z_um, width_um, color="#4C78A8")
    ax.plot(z_um, -width_um, color="#4C78A8")
    ax.fill_between(z_um, -width_um, width_um, alpha=0.15, color="#4C78A8")
    ax.set_xlabel("z (um)")
    ax.set_ylabel("w(z) (um)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_training_history(history: dict[str, list[float]]) -> plt.Figure:
    """Plot training loss and validation accuracy over epochs."""
    fig, ax1 = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], color="#E45756", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="#E45756")
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_accuracy"], color="#4C78A8", label="val accuracy")
    ax2.set_ylabel("val accuracy", color="#4C78A8")
    ax1.set_title("Training history")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str] | None = None) -> plt.Figure:
    """Render a confusion matrix as an annotated heatmap."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    n = matrix.shape[0]
    ticks = class_names or [str(i) for i in range(n)]
    ax.set_xticks(range(n), ticks)
    ax.set_yticks(range(n), ticks)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_feature_importance(names: list[str], importances: np.ndarray) -> plt.Figure:
    """Horizontal bar chart of feature importances."""
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([names[i] for i in order], importances[order], color="#54A24B")
    ax.set_xlabel("importance")
    ax.set_title("Feature importance")
    fig.tight_layout()
    return fig


def plot_timing(labels: list[str], times_s: list[float]) -> plt.Figure:
    """Bar chart of per-backend timings on a log scale."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, times_s, color="#B279A2")
    ax.set_yscale("log")
    ax.set_ylabel("time (s, log scale)")
    ax.set_title("Benchmark timing")
    fig.tight_layout()
    return fig
