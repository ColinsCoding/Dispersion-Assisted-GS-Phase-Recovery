"""Image input/output, preprocessing, and deterministic synthetic test images.

Conventions used everywhere in this package
-------------------------------------------
* An image is a 2-D ``numpy.ndarray`` of ``float64`` with values in [0, 1].
  0 is black, 1 is white.  Cracks are assumed to be *darker* than the surface.
* Array index ``[row, col]`` = ``[y, x]`` with row 0 at the TOP of the picture.
* A mask is a 2-D ``bool`` array, ``True`` on crack pixels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from . import DEFAULT_SEED


# --------------------------------------------------------------------------
# Validation helper
# --------------------------------------------------------------------------
def require_2d_float(image: np.ndarray, name: str = "image") -> np.ndarray:
    """Return ``image`` as a finite 2-D float64 array or raise a clear error."""
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D grayscale array, got shape {arr.shape}. "
            "Convert colour images with crack_photonics.io.load_image()."
        )
    if arr.size == 0:
        raise ValueError(f"{name} is empty (shape {arr.shape}).")
    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


# --------------------------------------------------------------------------
# Loading and saving
# --------------------------------------------------------------------------
def load_image(path: str | Path) -> np.ndarray:
    """Load an image file as grayscale float64 in [0, 1].

    Colour images are converted to luminance.  8-bit and 16-bit files are
    scaled by their full integer range (255 or 65535), so no contrast
    stretching happens here -- that is the job of :func:`preprocess`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file exists but cannot be decoded as an image.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {path}\n"
            "  Put your own image there, or run with --make-example to create "
            "the built-in synthetic test image."
        )

    # np.fromfile + imdecode (instead of cv2.imread) also works for paths
    # that contain non-ASCII characters.
    raw_bytes = np.fromfile(str(path), dtype=np.uint8)
    raw = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(
            f"Could not decode {path} as an image. "
            "Supported formats include PNG, JPEG, TIFF and BMP."
        )

    if raw.ndim == 3:
        if raw.shape[2] == 4:  # drop the alpha channel
            raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    if raw.dtype == np.uint8:
        scale = 255.0
    elif raw.dtype == np.uint16:
        scale = 65535.0
    elif np.issubdtype(raw.dtype, np.floating):
        scale = 1.0
    else:
        raise ValueError(f"Unsupported pixel type {raw.dtype} in {path}.")

    return np.clip(raw.astype(np.float64) / scale, 0.0, 1.0)


def save_image(path: str | Path, image: np.ndarray) -> Path:
    """Save a [0, 1] float image or a boolean mask as an 8-bit PNG.

    Values outside [0, 1] are clipped.  Parent folders are created.
    """
    path = Path(path)
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"Can only save 2-D images, got shape {arr.shape}.")
    arr8 = (np.clip(arr.astype(np.float64), 0.0, 1.0) * 255.0).round().astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", arr8)
    if not ok:
        raise OSError(f"PNG encoding failed for {path}.")
    buffer.tofile(str(path))
    return path


def signed_to_display(image: np.ndarray) -> np.ndarray:
    """Map a signed image (for example an odd component) into [0, 1].

    Zero maps to mid-gray 0.5; the largest magnitude maps to 0 or 1.
    This is for *viewing only* -- the mapping is not invertible without
    knowing the scale factor, so do not use the PNG for further arithmetic.
    """
    arr = require_2d_float(image, "signed image")
    peak = float(np.max(np.abs(arr)))
    if peak == 0.0:
        return np.full_like(arr, 0.5)
    return 0.5 + 0.5 * arr / peak


def _json_default(obj):
    """Convert NumPy scalars/arrays so ``json`` can serialise them."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable.")


def save_json(path: str | Path, data: dict) -> Path:
    """Write ``data`` as indented JSON.  Parent folders are created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_json_default)
        handle.write("\n")
    return path


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def preprocess(image: np.ndarray, denoise_sigma: float = 1.0) -> np.ndarray:
    """Normalise contrast to [0, 1] and apply a light Gaussian denoise.

    Steps
    -----
    1. Min-max normalisation, so later thresholds do not depend on exposure.
    2. Gaussian blur of standard deviation ``denoise_sigma`` pixels
       (0 disables it).  The blur uses mirrored borders, so a mirror-symmetric
       input stays mirror-symmetric.

    Raises
    ------
    ValueError
        If the image is constant (no contrast to analyse) or the sigma is
        negative.
    """
    arr = require_2d_float(image)
    if denoise_sigma < 0:
        raise ValueError(f"denoise_sigma must be >= 0, got {denoise_sigma}.")

    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        raise ValueError(
            "Image is constant (max == min); there is no contrast to analyse."
        )
    out = (arr - lo) / (hi - lo)
    if denoise_sigma > 0:
        out = ndimage.gaussian_filter(out, sigma=denoise_sigma, mode="reflect")
    return out


# --------------------------------------------------------------------------
# Deterministic synthetic images
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SyntheticCrack:
    """A synthetic crack image together with its known ground truth."""

    image: np.ndarray          # float64 in [0, 1]
    mask: np.ndarray           # bool, True on crack pixels
    centerline_xy: np.ndarray  # (n, 2) array of (x, y) points along the crack
    true_length_px: float      # arc length of the centreline
    true_width_px: float       # nominal crack width measured along a column
    seed: int


def _surface_texture(shape: tuple[int, int], rng: np.random.Generator,
                     noise_std: float) -> np.ndarray:
    """Concrete-like surface: mid-gray + smooth blotches + fine sensor noise."""
    blotches = ndimage.gaussian_filter(rng.standard_normal(shape), sigma=6.0)
    blotches /= blotches.std()
    fine = rng.standard_normal(shape)
    return 0.65 + 0.04 * blotches + noise_std * fine


def _check_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2 or min(shape) < 32:
        raise ValueError(f"shape must be (rows, cols) with both >= 32, got {shape}.")
    return int(shape[0]), int(shape[1])


def generate_synthetic_crack(
    shape: tuple[int, int] = (256, 256),
    seed: int = DEFAULT_SEED,
    crack_width: float = 4.0,
    crack_depth: float = 0.5,
    noise_std: float = 0.03,
) -> SyntheticCrack:
    """Create a reproducible gray surface with one dark, wandering crack.

    The crack centreline is a smoothed random walk that drifts diagonally
    across the middle 80 % of the image, so the picture is clearly *not*
    mirror-symmetric.  The same ``seed`` always gives the same image.

    Parameters
    ----------
    shape : (rows, cols)
    seed : int
        Seed for ``numpy.random.default_rng``.
    crack_width : float
        Nominal crack width in pixels, measured along a column.
    crack_depth : float
        Fractional darkening at the crack centre (0 = invisible, 1 = black).
    noise_std : float
        Standard deviation of the fine pixel noise.

    Notes
    -----
    This is a *test pattern*, not a model of real fracture.  Real cracks
    branch, change width, collect dirt, and sit on stained, shadowed,
    textured surfaces.
    """
    rows, cols = _check_shape(shape)
    if crack_width <= 0:
        raise ValueError(f"crack_width must be > 0, got {crack_width}.")
    if not 0 < crack_depth <= 1:
        raise ValueError(f"crack_depth must be in (0, 1], got {crack_depth}.")
    if noise_std < 0:
        raise ValueError(f"noise_std must be >= 0, got {noise_std}.")

    rng = np.random.default_rng(seed)
    surface = _surface_texture((rows, cols), rng, noise_std)

    # Centreline y(x): diagonal drift plus a smoothed random walk.
    x = np.arange(cols, dtype=np.float64)
    walk = np.cumsum(rng.normal(0.0, 1.0, size=cols))
    walk = ndimage.gaussian_filter1d(walk, sigma=5.0, mode="nearest")
    walk *= 0.04 * rows / max(np.abs(walk).max(), 1e-12)
    y_center = 0.35 * rows + 0.30 * rows * x / cols + walk

    # The crack only spans the middle 80 % of the columns.
    x0, x1 = int(0.10 * cols), int(0.90 * cols)
    inside = (x >= x0) & (x < x1)

    # Distance of every pixel from the centreline, measured along its column.
    row_index = np.arange(rows, dtype=np.float64)[:, None]
    distance = np.abs(row_index - y_center[None, :])

    mask = (distance <= crack_width / 2.0) & inside[None, :]

    # Smooth dark valley: Gaussian profile whose full width at half maximum
    # equals crack_width.
    profile_sigma = crack_width / 2.355
    profile = np.exp(-0.5 * (distance / profile_sigma) ** 2) * inside[None, :]
    image = np.clip(surface * (1.0 - crack_depth * profile), 0.0, 1.0)

    xs, ys = x[inside], y_center[inside]
    true_length = float(np.sum(np.hypot(np.diff(xs), np.diff(ys)))) + 1.0

    return SyntheticCrack(
        image=image,
        mask=mask,
        centerline_xy=np.column_stack([xs, ys]),
        true_length_px=true_length,
        true_width_px=float(crack_width),
        seed=int(seed),
    )


def generate_symmetric_image(
    shape: tuple[int, int] = (256, 256),
    seed: int = DEFAULT_SEED,
    axis: str = "horizontal",
    noise_std: float = 0.03,
) -> np.ndarray:
    """Create a reproducible crack-free surface that is exactly mirror-symmetric.

    The random surface texture ``T`` is replaced by its even part
    ``(T + T_R) / 2``, which is symmetric by construction.

    ``axis`` follows :mod:`crack_photonics.symmetry`:
    ``"horizontal"`` mirrors left <-> right, ``"vertical"`` mirrors top <-> bottom.
    """
    rows, cols = _check_shape(shape)
    if axis not in ("horizontal", "vertical"):
        raise ValueError(f"axis must be 'horizontal' or 'vertical', got {axis!r}.")
    rng = np.random.default_rng(seed)
    surface = _surface_texture((rows, cols), rng, noise_std)
    flip_axis = 1 if axis == "horizontal" else 0
    symmetric = 0.5 * (surface + np.flip(surface, axis=flip_axis))
    return np.clip(symmetric, 0.0, 1.0)
