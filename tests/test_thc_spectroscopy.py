"""Tests for dgs/thc_spectroscopy.py."""
import numpy as np
import sympy as sp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.thc_spectroscopy import (
    thc_absorption_spectrum, beer_lambert_inversion,
    simulate_tsdft_measurement, spatial_hash_3d,
    voxelize_for_printing, thc_measurement_pipeline,
    thc_spectroscopy_sympy_5,
)


# ── Beer-Lambert ──────────────────────────────────────────────────────────────

def test_absorption_peak_location():
    lam = np.linspace(1500, 1900, 1000)
    spec = thc_absorption_spectrum(lam, concentration_mgmL=10.0)
    # Peak should be near 1680 nm
    peak_lam = lam[np.argmax(spec["A_thc"])]
    assert 1650 < peak_lam < 1720


def test_absorption_scales_linearly():
    lam = np.linspace(1600, 1750, 200)
    spec1 = thc_absorption_spectrum(lam, concentration_mgmL=5.0)
    spec2 = thc_absorption_spectrum(lam, concentration_mgmL=10.0)
    ratio = spec2["A_thc"] / (spec1["A_thc"] + 1e-12)
    np.testing.assert_allclose(ratio[ratio > 0.01], 2.0, rtol=0.01)


def test_transmittance_range():
    lam = np.linspace(1500, 1900, 500)
    spec = thc_absorption_spectrum(lam, concentration_mgmL=20.0)
    T = spec["transmittance"]
    assert T.min() >= 0.0
    assert T.max() <= 1.0 + 1e-10


def test_beer_lambert_inversion_exact():
    lam = np.linspace(1500, 1900, 500)
    C_true = 10.0
    spec = thc_absorption_spectrum(lam, C_true, path_cm=1.0)
    inv = beer_lambert_inversion(spec["A_thc"], lam, path_cm=1.0)
    # Least-squares should recover concentration to within 20%
    assert abs(inv["C_mgmL_lstsq"] - C_true) / C_true < 0.2


def test_beer_lambert_path_length():
    lam = np.linspace(1600, 1800, 300)
    # Doubling path doubles absorbance
    s1 = thc_absorption_spectrum(lam, 10.0, path_cm=1.0)
    s2 = thc_absorption_spectrum(lam, 10.0, path_cm=2.0)
    np.testing.assert_allclose(s2["A_thc"], 2 * s1["A_thc"], rtol=0.001)


# ── TS-DFT simulation ─────────────────────────────────────────────────────────

def test_tsdft_spectrum_shape():
    meas = simulate_tsdft_measurement(concentration_mgmL=10.0)
    assert len(meas["wavelength_nm"]) == len(meas["I_ref"])
    assert len(meas["A_measured"]) == len(meas["wavelength_nm"])


def test_tsdft_positive_ref():
    meas = simulate_tsdft_measurement(concentration_mgmL=10.0)
    assert np.all(meas["I_ref"] > 0)
    assert np.all(meas["I_sample"] > 0)


def test_tsdft_dispersion_parameter():
    meas = simulate_tsdft_measurement(L_fiber_km=10.0)
    # dt/dlam should be positive for anomalous dispersion fiber
    # SMF-28 at 1700nm: D ~ 17 ps/(nm*km) -> dt_dlam ~ 170 ps/nm for 10 km
    assert meas["dt_dlam_ps_nm"] > 0


# ── Spatial hash ──────────────────────────────────────────────────────────────

def test_spatial_hash_all_points():
    pts = np.random.default_rng(42).standard_normal((100, 3))
    h = spatial_hash_3d(pts, cell_size=1.0)
    total_in_buckets = sum(len(v) for v in h["table"].values())
    assert total_in_buckets == 100


def test_spatial_hash_deterministic():
    pts = np.random.default_rng(0).standard_normal((50, 3))
    h1 = spatial_hash_3d(pts)
    h2 = spatial_hash_3d(pts)
    np.testing.assert_array_equal(h1["hashes"], h2["hashes"])


def test_spatial_hash_same_cell():
    # Two points in the same voxel
    pts = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
    h = spatial_hash_3d(pts, cell_size=1.0)
    assert h["n_occupied_voxels"] == 1   # both in (0,0,0) voxel


def test_spatial_hash_different_cells():
    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    h = spatial_hash_3d(pts, cell_size=1.0)
    assert h["n_occupied_voxels"] == 2


# ── Full pipeline ─────────────────────────────────────────────────────────────

def test_pipeline_accuracy():
    pipe = thc_measurement_pipeline(sample_thc_pct=20.0)
    # Should measure within 10% of true value
    assert abs(pipe["measured_thc_pct"] - 20.0) < 10.0


def test_pipeline_report_text():
    pipe = thc_measurement_pipeline(sample_thc_pct=15.0)
    assert "THC" in pipe["report"]
    assert "Beer-Lambert" in pipe["report"]


def test_pipeline_voxels_nonzero():
    pipe = thc_measurement_pipeline(sample_thc_pct=20.0)
    assert pipe["print_voxels"] > 0


def test_pipeline_spectra_per_sec():
    pipe = thc_measurement_pipeline()
    assert pipe["spectra_per_sec"] == 1e9   # 1 GHz rep rate


# ── SymPy ─────────────────────────────────────────────────────────────────────

def test_sympy_5_count():
    eqs = thc_spectroscopy_sympy_5()
    assert len(eqs) == 5


def test_sympy_5_all_eq():
    eqs = thc_spectroscopy_sympy_5()
    for k, v in eqs.items():
        assert isinstance(v, sp.Basic)
