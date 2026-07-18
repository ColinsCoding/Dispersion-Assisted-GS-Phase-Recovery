"""End-to-end pipeline example.

Runs the full workflow and writes figures + generated C to `output_dir`:
    Physics -> SymPy -> numerical verification -> features -> ML -> generated C -> docs.

Usage:
    python examples/run_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from c_codegen.generator import generate_c, write_c
from config import Config, get_logger
from feature_extraction.features import feature_names
from ml.dataset import BeamFeatureDataset
from ml.inference import confusion_matrix, predict
from ml.train import train_model
from physics.gaussian_beam import GaussianBeam
from physics.symbolic import gaussian_beam_width
from visualization.plots import (
    plot_confusion_matrix,
    plot_optical_field,
    plot_training_history,
)

LOGGER = get_logger("pipeline")


def main(config: Config | None = None) -> float:
    """Execute the pipeline; return the validation accuracy."""
    config = config or Config()
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Physics + SymPy
    beam = GaussianBeam(config.beam.wavelength_um, config.beam.waist_um)
    z = np.linspace(config.beam.z_min_um, config.beam.z_max_um, config.beam.n_axial)
    width = beam.width_um(z)
    LOGGER.info("Rayleigh range = %.1f um", beam.rayleigh_range_um)
    plot_optical_field(z, width).savefig(out / "beam_width.png", dpi=120)

    # 2. Feature extraction + ML
    dataset = BeamFeatureDataset(
        n_samples=config.ml.n_samples,
        wavelength_um=config.beam.wavelength_um,
        half_width_um=config.beam.transverse_half_width_um,
        n_transverse=config.beam.n_transverse,
        seed=config.ml.seed,
    )
    result = train_model(
        dataset,
        input_dim=dataset.feature_dim,
        n_classes=dataset.n_classes,
        hidden_dims=config.ml.hidden_dims,
        epochs=config.ml.epochs,
        learning_rate=config.ml.learning_rate,
        batch_size=config.ml.batch_size,
        val_fraction=config.ml.val_fraction,
        seed=config.ml.seed,
    )
    LOGGER.info("Validation accuracy = %.3f", result.val_accuracy)
    plot_training_history(result.history).savefig(out / "training_history.png", dpi=120)

    features = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = torch.stack([dataset[i][1] for i in range(len(dataset))])
    cm = confusion_matrix(labels, predict(result.model, features), dataset.n_classes)
    plot_confusion_matrix(cm, list(map(str, range(dataset.n_classes)))).savefig(out / "confusion.png", dpi=120)

    # 3. Generated C from the finalized symbolic model
    codegen = generate_c(gaussian_beam_width())
    paths = write_c(codegen, src_dir="src", include_dir="include")
    LOGGER.info("Generated C: %s", paths["source"])

    LOGGER.info("Features: %s", ", ".join(feature_names()))
    LOGGER.info("Figures written to %s", out.resolve())
    return result.val_accuracy


if __name__ == "__main__":
    main()
