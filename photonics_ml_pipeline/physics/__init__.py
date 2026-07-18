"""Physics stage: symbolic derivations (SymPy) and their numeric realizations."""
from physics.gaussian_beam import GaussianBeam
from physics.symbolic import (
    SymbolicExpression,
    gaussian_beam_width,
    gouy_phase,
    rayleigh_range,
)

__all__ = [
    "GaussianBeam",
    "SymbolicExpression",
    "gaussian_beam_width",
    "gouy_phase",
    "rayleigh_range",
]
