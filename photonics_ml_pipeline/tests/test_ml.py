"""ML tests: dataset integrity, training convergence, inference utilities."""
from __future__ import annotations

import torch

from ml.dataset import BeamFeatureDataset
from ml.inference import confusion_matrix, predict
from ml.train import train_model


def _small_dataset() -> BeamFeatureDataset:
    return BeamFeatureDataset(n_samples=300, n_transverse=128, seed=0)


def test_dataset_shapes_and_classes() -> None:
    ds = _small_dataset()
    x, y = ds[0]
    assert x.shape == (ds.feature_dim,)
    assert y.dtype == torch.long
    assert ds.n_classes == 3


def test_training_reaches_high_accuracy() -> None:
    ds = _small_dataset()
    result = train_model(ds, input_dim=ds.feature_dim, n_classes=ds.n_classes, epochs=40, seed=0)
    assert result.val_accuracy > 0.85


def test_confusion_matrix_shape_and_total() -> None:
    ds = _small_dataset()
    result = train_model(ds, input_dim=ds.feature_dim, n_classes=ds.n_classes, epochs=20, seed=0)
    features = torch.stack([ds[i][0] for i in range(len(ds))])
    labels = torch.stack([ds[i][1] for i in range(len(ds))])
    preds = predict(result.model, features)
    cm = confusion_matrix(labels, preds, ds.n_classes)
    assert cm.shape == (3, 3)
    assert int(cm.sum()) == len(ds)
