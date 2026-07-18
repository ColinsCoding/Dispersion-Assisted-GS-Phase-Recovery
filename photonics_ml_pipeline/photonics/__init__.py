"""Photonics stage: dispersive propagation (the project's core operator)."""
from photonics.dispersion import (
    apply_dispersion,
    group_delay,
    transfer_function,
)

__all__ = ["transfer_function", "apply_dispersion", "group_delay"]
