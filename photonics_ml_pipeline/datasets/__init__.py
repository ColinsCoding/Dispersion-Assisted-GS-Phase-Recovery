"""Datasets package.

Synthetic samples are generated on the fly by `ml.dataset.BeamFeatureDataset`; this
directory holds any cached or externally provided data files. Kept as a package so the
path is importable and discoverable.
"""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

__all__ = ["DATA_DIR"]
