"""Unit tests for the ROM-quantized PST path (NumPy only)."""
from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ros2_phycv.pst_core import PstParams, phase_stretch_transform
from ros2_phycv.pst_rom import build_radial_rom, phase_stretch_transform_rom


def _disk(m: int = 128) -> np.ndarray:
    yy, xx = np.mgrid[0:m, 0:m]
    disk = 0.5 * (1 - np.tanh((np.sqrt((xx - m // 2) ** 2 + (yy - m // 2) ** 2) - 30) / 1.5))
    return 0.2 + 0.8 * disk


def test_rom_tables_are_quantized_and_bounded() -> None:
    rom = build_radial_rom(PstParams(), n_bins=256, n_bits=8)
    assert rom.re.shape == (256,) and rom.im.shape == (256,)
    assert rom.re.min() >= -rom.scale and rom.re.max() <= rom.scale       # fits signed 8-bit
    assert rom.n_bins == 256 and rom.scale == 127


def test_rom_is_deterministic_and_size_independent() -> None:
    a = build_radial_rom(PstParams())
    b = build_radial_rom(PstParams())
    assert np.array_equal(a.re, b.re) and np.array_equal(a.im, b.im)      # same ROM every time
    # the same ROM reconstructs kernels of different frame sizes
    assert a.kernel((128, 128)).shape == (128, 128)
    assert a.kernel((200, 256)).shape == (200, 256)


def test_rom_kernel_magnitude_bounded() -> None:
    rom = build_radial_rom(PstParams())
    # independent int rounding of Re and Im can push the magnitude up to ~0.5 LSB over 1
    assert np.all(np.abs(rom.kernel((96, 96))) <= 1.0 + np.sqrt(0.5) / rom.scale + 1e-9)


def test_rom_pst_detects_edges() -> None:
    img = _disk()
    result = phase_stretch_transform_rom(img, PstParams(strength=4.0, warp=15.0, sigma_lpf=0.2))
    gy, gx = np.gradient(img)
    boundary = np.sqrt(gx**2 + gy**2) > 0.05
    resp = np.abs(result.phase)
    assert resp[boundary].mean() > 3.0 * resp[~boundary].mean()


def test_rom_edge_map_matches_float() -> None:
    # the published output is the binary edge map; the 8-bit ROM must reproduce it closely
    img = _disk()
    params = PstParams(strength=4.0, warp=15.0, sigma_lpf=0.2)
    edges_float = phase_stretch_transform(img, params).edges
    edges_rom = phase_stretch_transform_rom(img, params).edges
    agreement = float((edges_float == edges_rom).mean())
    assert agreement > 0.95                                               # >95% pixel agreement (measured ~99.95%)
