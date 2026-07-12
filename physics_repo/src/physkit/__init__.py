"""physkit: reusable numerics for the physics_repo notebooks.

The package is the shared spine of the repository. Later notebooks import earlier results from here
instead of redefining them, so the chapters compose rather than repeat.

Submodules
----------
constants : CODATA physical constants in SI, plus a tidy table.
units     : object-oriented dimensional analysis (Dimension, Quantity).
linalg    : eigenproblem and orthogonalization helpers used from chapter 06 onward.
plotting  : a single publishable Matplotlib style.

PyTorch is optional throughout the repository. Call `optional_torch()` to obtain the module if it is
importable, or None otherwise, and branch on the result.
"""

from . import constants, units, linalg, plotting

__all__ = ["constants", "units", "linalg", "plotting", "optional_torch"]
__version__ = "0.1.0"


def optional_torch():
    """Return the imported ``torch`` module, or ``None`` if it is not available.

    Notebooks use this so a missing or non-loadable PyTorch never breaks execution; the NumPy path
    remains authoritative and PyTorch is only a cross-check or accelerator when present.
    """
    try:
        import torch  # noqa: F401
        return torch
    except Exception:
        return None
