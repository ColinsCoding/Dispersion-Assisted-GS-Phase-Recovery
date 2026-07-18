"""Typed configuration for the photonics-ML pipeline.

Purpose:
    Central, immutable-by-default configuration objects (dataclasses) plus a small
    logging helper. Avoids global mutable state -- every component receives an explicit
    config instance.

References:
    - Saleh & Teich, *Fundamentals of Photonics*, Ch. 3 (Gaussian beams).
Assumptions:
    - All spatial quantities are in micrometers (um) unless noted.
Limitations:
    - YAML loading is optional; if PyYAML is absent, `from_yaml` raises a clear error.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:  # optional dependency
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

__all__ = ["BeamConfig", "MLConfig", "Config", "get_logger"]


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a single stream handler (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@dataclass(frozen=True)
class BeamConfig:
    """Parameters for the Gaussian-beam physics stage."""

    wavelength_um: float = 1.55
    waist_um: float = 10.0
    z_min_um: float = -2000.0
    z_max_um: float = 2000.0
    n_axial: int = 256
    n_transverse: int = 256
    transverse_half_width_um: float = 60.0


@dataclass(frozen=True)
class MLConfig:
    """Parameters for the machine-learning stage."""

    hidden_dims: tuple[int, ...] = (32, 16)
    n_classes: int = 3
    epochs: int = 60
    learning_rate: float = 1e-3
    batch_size: int = 32
    n_samples: int = 900
    val_fraction: float = 0.25
    seed: int = 0


@dataclass(frozen=True)
class Config:
    """Top-level configuration aggregating every stage."""

    beam: BeamConfig = field(default_factory=BeamConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    output_dir: str = "outputs"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain nested dictionary."""
        return asdict(self)

    def to_json(self, path: str | Path) -> Path:
        """Write the configuration to a JSON file and return its path."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Construct a Config from a nested dictionary (missing keys use defaults)."""
        beam = BeamConfig(**data.get("beam", {}))
        ml_data = dict(data.get("ml", {}))
        if "hidden_dims" in ml_data:
            ml_data["hidden_dims"] = tuple(ml_data["hidden_dims"])
        ml = MLConfig(**ml_data)
        return cls(beam=beam, ml=ml, output_dir=data.get("output_dir", "outputs"))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load a Config from a YAML file (requires PyYAML)."""
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is not installed; cannot read YAML config.")
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)
