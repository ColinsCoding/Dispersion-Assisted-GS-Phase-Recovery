"""Crack segmentation and civil-engineering style crack metrics.

All lengths are reported in PIXELS unless a ``pixel_size_mm`` scale is
supplied.  A pixel is not a unit of length: to report millimetres you must
photograph a scale reference (ruler, crack-width gauge card) in the same
plane as the crack and measure how many pixels one millimetre covers.

Angle convention
----------------
``orientation_deg`` is in [0, 180), measured counter-clockwise from the
picture's +x axis (pointing right) as the picture is normally displayed.
0 deg = horizontal crack, 90 deg = vertical crack, 45 deg = rising towards
the upper right.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from .io import require_2d_float

_SQRT2 = float(np.sqrt(2.0))


def _require_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError("mask is empty.")
    if arr.dtype != np.bool_:
        raise TypeError(
            f"mask must have dtype bool, got {arr.dtype}. "
            "Use (array > threshold) to create one."
        )
    return arr


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------
def segment_crack(
    image: np.ndarray,
    background_sigma: float = 12.0,
    threshold_sigma: float = 4.0,
    min_contrast: float = 0.08,
    min_component_area: int = 30,
) -> np.ndarray:
    """Find pixels that are clearly darker than their local surroundings.

    Method (a classic "dark residual" threshold, no learning involved)
    ------------------------------------------------------------------
    1. Estimate the slowly varying surface brightness with a wide Gaussian
       blur (``background_sigma`` pixels).  This tolerates uneven lighting.
    2. ``residual = background - image`` is positive where a pixel is darker
       than its neighbourhood.
    3. Keep pixels whose residual exceeds BOTH
       ``median + threshold_sigma * robust_std`` (statistically unusual) and
       ``min_contrast`` (physically meaningful darkening on the 0-1 scale).
    4. Close one-pixel gaps and delete connected blobs smaller than
       ``min_component_area`` pixels.

    The input should come from :func:`crack_photonics.io.preprocess`.

    Returns
    -------
    bool array, ``True`` on crack pixels.  An all-``False`` mask is a valid
    answer ("nothing found"), not an error.
    """
    arr = require_2d_float(image)
    if background_sigma <= 0:
        raise ValueError(f"background_sigma must be > 0, got {background_sigma}.")
    if threshold_sigma <= 0:
        raise ValueError(f"threshold_sigma must be > 0, got {threshold_sigma}.")
    if min_contrast < 0:
        raise ValueError(f"min_contrast must be >= 0, got {min_contrast}.")
    if min_component_area < 0:
        raise ValueError(f"min_component_area must be >= 0, got {min_component_area}.")

    background = ndimage.gaussian_filter(arr, sigma=background_sigma, mode="reflect")
    residual = background - arr

    median = float(np.median(residual))
    robust_std = 1.4826 * float(np.median(np.abs(residual - median)))
    threshold = max(median + threshold_sigma * robust_std, min_contrast)
    mask = residual > threshold

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_component_area  # label 0 = background
    return keep[labels]


# --------------------------------------------------------------------------
# Skeleton (centreline) and length
# --------------------------------------------------------------------------
def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Thin a mask to a one-pixel-wide centreline (Zhang-Suen algorithm).

    The algorithm repeatedly peels boundary pixels that can be removed
    without breaking connectivity, alternating between two sub-passes, until
    nothing changes.  Each pass is vectorised with NumPy; the number of
    passes is about half the widest crack width in pixels.

    Reference: T. Y. Zhang and C. Y. Suen, "A fast parallel algorithm for
    thinning digital patterns," Communications of the ACM, 1984.
    """
    img = np.pad(_require_mask(mask), 1)  # False border so every pixel has 8 neighbours
    center = img[1:-1, 1:-1]              # view: edits below change ``img``

    while True:
        changed = False
        for sub_pass in (0, 1):
            # Neighbours P2..P9, clockwise starting from the pixel above.
            p2, p3 = img[:-2, 1:-1], img[:-2, 2:]
            p4, p5 = img[1:-1, 2:], img[2:, 2:]
            p6, p7 = img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]

            neighbours = sum(p.astype(np.uint8) for p in ring[:8])
            # Number of False -> True transitions walking once around the ring.
            transitions = sum(
                (~ring[i] & ring[i + 1]).astype(np.uint8) for i in range(8)
            )
            if sub_pass == 0:
                side = ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                side = ~(p2 & p4 & p8) & ~(p2 & p6 & p8)

            removable = (
                center & (neighbours >= 2) & (neighbours <= 6)
                & (transitions == 1) & side
            )
            if removable.any():
                center[removable] = False
                changed = True
        if not changed:
            break
    return center.copy()


def skeleton_length(skeleton: np.ndarray) -> float:
    """Length of a skeleton in pixels (chain-code estimate).

    * A step between horizontally or vertically adjacent pixels counts 1.
    * A diagonal step counts sqrt(2).
    * Thinning leaves small "staircase" corners where three pixels of a 2x2
      block are set.  Walking around such a corner costs 2, but the crack
      really cuts across it, so each corner is counted as one diagonal step
      (sqrt(2)) instead.
    * One pixel is added per connected piece, because a run of n pixel
      centres spans n - 1 steps but covers n pixels.

    Known bias: a path made only of 0/45/90 degree steps is longer than the
    straight line it approximates unless that line is itself at 0, 45 or 90
    degrees.  The overestimate peaks at about 8 % near 22.5 degrees.
    """
    sk = _require_mask(skeleton)
    if not sk.any():
        return 0.0

    horizontal = np.count_nonzero(sk[:, :-1] & sk[:, 1:])
    vertical = np.count_nonzero(sk[:-1, :] & sk[1:, :])

    # Every 2x2 block of the skeleton:  a b
    #                                   c d
    a, b = sk[:-1, :-1], sk[:-1, 1:]
    c, d = sk[1:, :-1], sk[1:, 1:]
    diagonal = np.count_nonzero(a & d & ~b & ~c) + np.count_nonzero(b & c & ~a & ~d)
    filled = a.astype(np.uint8) + b + c + d
    corners = np.count_nonzero(filled == 3)

    pieces = cv2.connectedComponents(sk.astype(np.uint8), connectivity=8)[0] - 1
    steps = horizontal + vertical + _SQRT2 * diagonal - (2.0 - _SQRT2) * corners
    return float(steps + pieces)


# --------------------------------------------------------------------------
# Individual metrics
# --------------------------------------------------------------------------
def crack_area_percent(mask: np.ndarray) -> float:
    """Percentage of image pixels flagged as crack: ``100 * crack / total``."""
    m = _require_mask(mask)
    return 100.0 * float(np.count_nonzero(m)) / float(m.size)


def count_components(mask: np.ndarray) -> int:
    """Number of separate crack pieces (8-connected: diagonals touch)."""
    m = _require_mask(mask)
    return int(cv2.connectedComponents(m.astype(np.uint8), connectivity=8)[0] - 1)


def orientation_angle(mask: np.ndarray) -> tuple[float | None, float]:
    """Dominant crack direction from principal component analysis (PCA).

    The crack pixels are treated as a cloud of (x, y) points.  The
    eigenvector of their 2x2 covariance matrix with the largest eigenvalue
    is the direction in which the cloud is longest.

    Returns
    -------
    (angle_deg, anisotropy)
        ``angle_deg`` in [0, 180), or ``None`` if there are fewer than two
        crack pixels.  ``anisotropy = 1 - small_eigenvalue / large_eigenvalue``
        is 1 for a perfect line and 0 for a round blob, where the angle is
        meaningless.
    """
    m = _require_mask(mask)
    rows, cols = np.nonzero(m)
    if rows.size < 2:
        return None, 0.0

    x = cols.astype(np.float64)
    y = -rows.astype(np.float64)  # flip so +y points up, as on a displayed picture
    covariance = np.cov(np.vstack([x, y]))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)  # ascending order
    major = eigenvectors[:, 1]

    angle = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    if angle >= 180.0 - 1e-9:  # guard against 179.999... for a horizontal line
        angle = 0.0
    large = float(eigenvalues[1])
    anisotropy = 1.0 - float(eigenvalues[0]) / large if large > 0 else 0.0
    return angle, anisotropy


def confidence_score(
    mask: np.ndarray,
    image: np.ndarray,
    edge_mask: np.ndarray,
    length_px: float,
    mean_width_px: float,
) -> dict:
    """Heuristic 0-1 score for "this mask looks like a crack".

    THIS IS NOT A CALIBRATED PROBABILITY.  It is a transparent weighted sum
    of three hand-chosen cues, each in [0, 1]:

    contrast (weight 0.4)
        How much darker the mask is than the rest of the image.
        A 30 % relative darkening or more scores 1.
    edge_support (weight 0.3)
        Fraction of mask pixels lying within 2 pixels of an edge found by
        the independent PST-style detector.
    elongation (weight 0.3)
        Cracks are long and thin.  ``length / width`` of 1 scores 0,
        10 or more scores 1.

    An empty mask scores 0.
    """
    m = _require_mask(mask)
    img = require_2d_float(image)
    edges = _require_mask(edge_mask)
    if not (m.shape == img.shape == edges.shape):
        raise ValueError(
            f"Shape mismatch: mask {m.shape}, image {img.shape}, edges {edges.shape}."
        )

    parts = {"contrast": 0.0, "edge_support": 0.0, "elongation": 0.0}
    if m.any() and not m.all():
        background_mean = float(img[~m].mean())
        crack_mean = float(img[m].mean())
        relative_darkening = (background_mean - crack_mean) / max(background_mean, 1e-12)
        parts["contrast"] = float(np.clip(relative_darkening / 0.30, 0.0, 1.0))

        near_edge = cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        parts["edge_support"] = float(near_edge[m].mean())

        if mean_width_px > 0:
            aspect = length_px / mean_width_px
            parts["elongation"] = float(np.clip((aspect - 1.0) / 9.0, 0.0, 1.0))

    score = 0.4 * parts["contrast"] + 0.3 * parts["edge_support"] + 0.3 * parts["elongation"]
    return {"confidence": float(score), "confidence_components": parts}


# --------------------------------------------------------------------------
# Everything together
# --------------------------------------------------------------------------
def compute_metrics(
    mask: np.ndarray,
    image: np.ndarray,
    edge_mask: np.ndarray,
    pixel_size_mm: float | None = None,
) -> dict:
    """Compute all crack metrics for one mask.

    Parameters
    ----------
    mask : bool array, crack pixels
    image : preprocessed float image the mask was computed from
    edge_mask : bool array from :func:`crack_photonics.pst_edges.detect_edges`
    pixel_size_mm : float or None
        Millimetres per pixel from a scale reference.  When ``None`` the
        millimetre fields are ``null`` -- the code will not guess a scale.

    Returns
    -------
    JSON-friendly dict with keys

    ``crack_area_percent``   crack pixels / all pixels * 100
    ``crack_pixel_count``    number of crack pixels
    ``crack_length_px``      skeleton (centreline) length
    ``mean_width_px``        area / length
    ``max_width_px``         from the distance transform: 2 * max_distance - 1
    ``orientation_deg``      PCA direction in [0, 180) or null
    ``orientation_anisotropy``  1 = line-like, 0 = blob-like
    ``connected_components`` number of separate pieces
    ``confidence``           heuristic 0-1 score (see :func:`confidence_score`)
    """
    m = _require_mask(mask)
    if pixel_size_mm is not None and pixel_size_mm <= 0:
        raise ValueError(f"pixel_size_mm must be > 0, got {pixel_size_mm}.")

    area_px = int(np.count_nonzero(m))
    skeleton = skeletonize(m)
    length_px = skeleton_length(skeleton)
    mean_width_px = area_px / length_px if length_px > 0 else 0.0

    if area_px:
        # Distance from each crack pixel to the nearest non-crack pixel.
        # On the centreline of a w-pixel-wide strip this is (w + 1) / 2.
        distance = ndimage.distance_transform_edt(m)
        max_width_px = 2.0 * float(distance[skeleton].max()) - 1.0
    else:
        max_width_px = 0.0

    angle, anisotropy = orientation_angle(m)

    metrics = {
        "crack_area_percent": crack_area_percent(m),
        "crack_pixel_count": area_px,
        "crack_length_px": length_px,
        "mean_width_px": mean_width_px,
        "max_width_px": max_width_px,
        "orientation_deg": angle,
        "orientation_anisotropy": anisotropy,
        "connected_components": count_components(m),
    }
    metrics.update(confidence_score(m, image, edge_mask, length_px, mean_width_px))

    metrics["pixel_size_mm"] = pixel_size_mm
    if pixel_size_mm is None:
        metrics["crack_length_mm"] = None
        metrics["mean_width_mm"] = None
        metrics["max_width_mm"] = None
        metrics["units_note"] = (
            "No scale reference given: lengths are in pixels. "
            "Pass --pixel-size-mm to convert."
        )
    else:
        metrics["crack_length_mm"] = length_px * pixel_size_mm
        metrics["mean_width_mm"] = mean_width_px * pixel_size_mm
        metrics["max_width_mm"] = max_width_px * pixel_size_mm
        metrics["units_note"] = f"Converted with {pixel_size_mm} mm per pixel."
    return metrics
