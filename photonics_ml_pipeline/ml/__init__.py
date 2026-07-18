"""Machine-learning stage (PyTorch): datasets, model, training, inference."""
from ml.dataset import BeamFeatureDataset
from ml.inference import confusion_matrix, predict
from ml.model import FeatureMLP
from ml.train import TrainResult, train_model

__all__ = [
    "BeamFeatureDataset",
    "FeatureMLP",
    "train_model",
    "TrainResult",
    "predict",
    "confusion_matrix",
]
