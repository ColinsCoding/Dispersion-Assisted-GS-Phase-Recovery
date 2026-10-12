"""Mirror symmetry of images: reflection, even/odd parts, symmetry score.

Physics background
------------------
Reflection is a *parity* operator ``P``: applying it twice gives the original
image back (``P P = 1``).  Any operator with that property has only two
eigenvalues, +1 and -1, and every image splits uniquely into

    I_even = (I + I_R) / 2      (eigenvalue +1, unchanged by the mirror)
    I_odd  = (I - I_R) / 2      (eigenvalue -1, sign-flipped by the mirror)

with ``I = I_even + I_odd``.  The two parts are orthogonal, so their energies
add like the sides of a right triangle:

    ||I||^2 = ||I_even||^2 + ||I_odd||^2 .

This is the same mathematics as even/odd functions in Fourier analysis and
parity eigenstates in quantum mechanics.

Axis naming used in this package
--------------------------------
``"horizontal"``  mirror left <-> right   (x -> -x, flips the column order)
``"vertical"``    mirror top  <-> bottom  (y -> -y, flips the row order)
"""

from __future__ import annotations

import numpy as np

from .io import require_2d_float

# Map the axis name to the NumPy array axis that gets reversed.
AXES = {"horizontal": 1, "vertical": 0}


def _array_axis(axis: str) -> int:
    try:
        return AXES[axis]
    except KeyError:
        raise ValueError(
            f"axis must be one of {sorted(AXES)}, got {axis!r}."
        ) from None


def reflect(image: np.ndarray, axis: str = "horizontal") -> np.ndarray:
    """Return the mirror image ``I_R``."""
    arr = require_2d_float(image)
    return np.flip(arr, axis=_array_axis(axis))


def even_odd_decompose(
    image: np.ndarray, axis: str = "horizontal"
) -> tuple[np.ndarray, np.ndarray]:
    """Split an image into its even and odd parts under reflection.

    Returns
    -------
    (even, odd) : two float64 arrays with ``even + odd == image``.
        ``even`` equals its own mirror image; ``odd`` changes sign under the
        mirror and can therefore be negative.
    """
    arr = require_2d_float(image)
    mirrored = reflect(arr, axis)
    even = 0.5 * (arr + mirrored)
    odd = 0.5 * (arr - mirrored)
    return even, odd


def symmetry_score(
    image: np.ndarray,
    axis: str = "horizontal",
    eps: float = 1e-12,
    remove_mean: bool = False,
) -> float:
    """Symmetry score ``S = 1 - ||I - I_R||_2 / (||I||_2 + eps)``.

    ``S = 1`` means perfectly mirror-symmetric.  Smaller is less symmetric.

    Range
    -----
    For a non-negative image the smallest possible value is ``1 - sqrt(2)``
    (about -0.41), reached when the image and its mirror do not overlap at
    all.  The score is therefore *not* a probability.

    Parameters
    ----------
    remove_mean : bool
        Subtract the mean brightness first.  A bright, low-contrast surface
        has a huge constant (DC) term that is trivially symmetric and pushes
        ``S`` towards 1 even when the *pattern* is asymmetric.  Removing the
        mean makes the score respond to the pattern only.
    """
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}.")
    arr = require_2d_float(image)
    if remove_mean:
        arr = arr - arr.mean()
    difference = arr - reflect(arr, axis)
    return float(1.0 - np.linalg.norm(difference) / (np.linalg.norm(arr) + eps))


def odd_energy_fraction(
    image: np.ndarray, axis: str = "horizontal", eps: float = 1e-12
) -> float:
    """Fraction of the image energy carried by the odd part, in [0, 1].

    ``||I_odd||^2 / ||I||^2``.  Zero for a symmetric image.
    """
    arr = require_2d_float(image)
    _, odd = even_odd_decompose(arr, axis)
    return float(np.sum(odd**2) / (np.sum(arr**2) + eps))


def analyze_symmetry(image: np.ndarray, eps: float = 1e-12) -> dict:
    """Compute all symmetry numbers for both mirror axes.

    Returns a JSON-friendly dictionary::

        {"horizontal": {"score": ..., "score_mean_removed": ...,
                        "odd_energy_fraction": ...},
         "vertical":   {...}}
    """
    arr = require_2d_float(image)
    report = {}
    for axis in AXES:
        report[axis] = {
            "score": symmetry_score(arr, axis, eps),
            "score_mean_removed": symmetry_score(arr, axis, eps, remove_mean=True),
            "odd_energy_fraction": odd_energy_fraction(arr, axis, eps),
        }
    return report
