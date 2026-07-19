"""Unit tests for the ROS-independent PST core (NumPy only)."""
from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ros2_phycv.pst_core import PstParams, PstResult, phase_stretch_transform


def _test_image() -> np.ndarray:
    m = n = 128
    yy, xx = np.mgrid[0:m, 0:n]
    disk = 0.5 * (1 - np.tanh((np.sqrt((xx - 64) ** 2 + (yy - 64) ** 2) - 30) / 1.5))
    return 0.2 + 0.8 * disk


def test_output_types_and_shapes() -> None:
    img = _test_image()
    result = phase_stretch_transform(img)
    assert isinstance(result, PstResult)
    assert result.phase.shape == img.shape
    assert result.edges.shape == img.shape
    assert result.edges.dtype == bool


def test_detects_edges_on_a_disk() -> None:
    img = _test_image()
    result = phase_stretch_transform(img, PstParams(strength=4.0, warp=15.0, sigma_lpf=0.2))
    # the boundary of the disk should carry a stronger phase response than the interior
    gy, gx = np.gradient(img)
    boundary = np.sqrt(gx**2 + gy**2) > 0.05
    resp = np.abs(result.phase)
    assert resp[boundary].mean() > 3.0 * resp[~boundary].mean()


def test_rejects_non_2d_input() -> None:
    try:
        phase_stretch_transform(np.zeros((4, 4, 3)))
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-2D input")


def test_flat_image_produces_no_edges() -> None:
    result = phase_stretch_transform(np.full((64, 64), 0.5))
    assert result.edges.sum() == 0
