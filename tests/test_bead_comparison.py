"""Tests for projects.seals.morphology.bead_comparison -- Part 1+2 of the
SEALS morphology research spec (two-bead reproduction + diameter sweep).
All traces are SIMULATED (Mie forward model), not real instrument data."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from projects.seals.morphology.bead_comparison import (
    bead_trace, extract_features, compare_two_beads, diameter_sweep,
    BEAD_A_DIAMETER_M, BEAD_B_DIAMETER_M,
)


def test_bead_trace_rejects_nonpositive_diameter():
    with pytest.raises(ValueError, match="positive"):
        bead_trace(0.0)
    with pytest.raises(ValueError, match="positive"):
        bead_trace(-1e-6)


def test_extract_features_on_synthetic_two_lobe_trace():
    """Sanity-check the feature extractor on a hand-built trace with a
    KNOWN answer, independent of the Mie model -- two clean Gaussian lobes
    at known angles/heights."""
    theta = np.linspace(-30, 30, 601)
    I = (np.exp(-((theta + 10) ** 2) / 8) + 2.0 * np.exp(-((theta - 10) ** 2) / 8))
    feats = extract_features(theta, I)

    assert feats["n_lobes"] == 2
    assert feats["lobe_spacing_deg"] == pytest.approx(20.0, abs=0.5)
    assert feats["peak_intensity"] == pytest.approx(2.0, abs=0.01)
    assert feats["integrated_intensity"] > 0            # a real physical area
    assert -10 < feats["centroid_deg"] < 10              # pulled toward the taller (right) lobe
    assert feats["variance_deg2"] > 0


def test_extract_features_rejects_negative_intensity():
    theta = np.array([0.0, 1.0, 2.0])
    I = np.array([1.0, -0.5, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        extract_features(theta, I)


def test_extract_features_is_order_independent_for_integrated_and_spacing():
    """The SEALS grating maps wavelength to DECREASING angle (see
    seals_to_tdgsa.seals_intensity_trace) -- integrated_intensity and
    lobe_spacing_deg must come out the same (positive) sign regardless of
    whether theta_deg is given ascending or descending. This is the
    regression test for the sign-flip bug found while building this module:
    np.trapezoid/np.diff on a DECREASING x-array silently negate physically
    non-negative quantities."""
    theta_asc = np.linspace(-30, 30, 601)
    I = np.exp(-((theta_asc + 10) ** 2) / 8) + 2.0 * np.exp(-((theta_asc - 10) ** 2) / 8)

    feats_asc = extract_features(theta_asc, I)
    feats_desc = extract_features(theta_asc[::-1], I[::-1])

    assert feats_asc["integrated_intensity"] > 0
    assert feats_desc["integrated_intensity"] > 0
    assert feats_asc["integrated_intensity"] == pytest.approx(feats_desc["integrated_intensity"])
    assert feats_asc["lobe_spacing_deg"] > 0
    assert feats_desc["lobe_spacing_deg"] > 0
    assert feats_asc["lobe_spacing_deg"] == pytest.approx(feats_desc["lobe_spacing_deg"])


def test_compare_two_beads_uses_paper_diameters_by_default():
    r = compare_two_beads()
    assert r["diameter_a_um"] == pytest.approx(7.32)
    assert r["diameter_b_um"] == pytest.approx(9.94)
    assert BEAD_A_DIAMETER_M == pytest.approx(7.32e-6)
    assert BEAD_B_DIAMETER_M == pytest.approx(9.94e-6)


def test_compare_two_beads_shapes_and_physical_positivity():
    r = compare_two_beads()
    N = len(r["theta_deg"])
    for key in ("lamvec", "I_a", "I_b", "I_a_norm", "I_b_norm", "diff_norm"):
        assert r[key].shape == (N,)
    assert np.all(r["I_a"] >= 0) and np.all(r["I_b"] >= 0)
    assert r["I_a_norm"].max() == pytest.approx(1.0)
    assert r["I_b_norm"].max() == pytest.approx(1.0)
    assert r["features_a"]["integrated_intensity"] > 0
    assert r["features_b"]["integrated_intensity"] > 0


def test_compare_two_beads_different_diameters_give_different_traces():
    """Not a trivial/degenerate comparison -- the two bead sizes must
    actually produce different scattering profiles."""
    r = compare_two_beads()
    assert not np.allclose(r["I_a_norm"], r["I_b_norm"])
    assert np.abs(r["diff_norm"]).max() > 0.05


def test_diameter_sweep_rejects_too_few_diameters():
    with pytest.raises(ValueError, match="at least 2"):
        diameter_sweep(diameters_um=(9.94,))


def test_diameter_sweep_returns_consistent_shapes_and_positive_features():
    theta_deg, traces, df = diameter_sweep(diameters_um=(6, 9.94, 12))
    assert set(traces.keys()) == {6, 9.94, 12}
    for I in traces.values():
        assert I.shape == theta_deg.shape
        assert np.all(I >= 0)
    assert len(df) == 3
    assert list(df["diameter_um"]) == [6, 9.94, 12]
    assert (df["integrated_intensity"] > 0).all()
    assert (df["lobe_spacing_deg"] > 0).all()


def test_diameter_sweep_peak_intensity_grows_with_diameter():
    """Physical sanity check: for these bead sizes (5-12um, well above the
    ~1.5um wavelength), larger particles scatter more strongly in the
    forward direction -- peak intensity should trend upward with diameter,
    not be flat or inverted. A real (if coarse) check that the Mie model
    is behaving physically, not just self-consistently."""
    _, _, df = diameter_sweep(diameters_um=(5, 6, 7.32, 8, 9.94, 11, 12))
    peaks = df.sort_values("diameter_um")["peak_intensity"].to_numpy()
    assert peaks[-1] > peaks[0] * 5, \
        f"expected peak intensity to grow substantially from smallest to largest bead, got {peaks}"
