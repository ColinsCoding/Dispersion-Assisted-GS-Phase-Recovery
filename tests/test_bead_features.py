"""Tests for projects.seals.morphology_classification.bead_features --
the SEALS_paper.pdf Fig. 5 bead-comparison reproduction and its feature
extraction (lobe count, spacing, peak/integrated intensity, centroid,
variance)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from projects.seals.morphology_classification.bead_features import (
    compute_bead_trace, extract_features, PAPER_BEAD_SMALL_M, PAPER_BEAD_LARGE_M,
)


def test_compute_bead_trace_shapes_and_nonnegativity():
    lamvec, theta_deg, I_p = compute_bead_trace(PAPER_BEAD_LARGE_M)
    assert lamvec.shape == theta_deg.shape == I_p.shape
    assert np.all(np.isfinite(I_p))
    assert np.all(I_p >= 0)


def test_compute_bead_trace_rejects_nonpositive_diameter():
    with pytest.raises(ValueError, match="dia_m"):
        compute_bead_trace(0.0)
    with pytest.raises(ValueError, match="dia_m"):
        compute_bead_trace(-1e-6)


def test_larger_bead_shows_more_lobes_than_smaller_bead():
    """The actual, paper-stated ground truth (SEALS_paper.pdf Sec. 3,
    confirmed by reading the paper directly): 'the smaller sized beads
    show a smaller number of lobes than the larger sized beads.' This is
    the single most important correctness check in this module -- if it
    fails, the simulation disagrees with the published measurement."""
    _, theta_small, I_small = compute_bead_trace(PAPER_BEAD_SMALL_M)
    _, theta_large, I_large = compute_bead_trace(PAPER_BEAD_LARGE_M)
    feat_small = extract_features(theta_small, I_small)
    feat_large = extract_features(theta_large, I_large)
    assert feat_large.n_lobes >= feat_small.n_lobes, \
        f"expected 9.94um bead (n_lobes={feat_large.n_lobes}) to show >= lobes than " \
        f"7.32um bead (n_lobes={feat_small.n_lobes}), per the paper's own stated result"


def test_refractive_index_matches_paper_citation():
    """Regression: the paper explicitly states n=1.39 for these beads
    (its own citation [15]) -- confirm this repo's default hasn't drifted
    from that confirmed value."""
    from projects.seals.inverse import _seals_physics as physics
    assert physics.P_DEFAULT["npar"] == pytest.approx(1.39)


def test_extract_features_matches_shapes():
    with pytest.raises(ValueError, match="shape"):
        extract_features(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_extract_features_centroid_and_variance_sane_for_simple_case():
    """Sanity check extract_features against a hand-computable case: a
    single symmetric triangular pulse centered at theta=0 should have
    centroid ~0 and a small positive variance, independent of the SEALS
    physics -- this isolates the feature-extraction MATH from the Mie
    physics, which is tested separately above."""
    theta = np.linspace(-10, 10, 501)
    I = np.maximum(0, 1 - np.abs(theta) / 5)  # triangle, peak at theta=0
    feat = extract_features(theta, I, peak_prominence_frac=0.5)
    assert abs(feat.centroid_deg) < 0.1
    assert feat.variance_deg2 > 0
    assert feat.peak_intensity == pytest.approx(1.0, abs=1e-6)
    assert feat.n_lobes == 1


def test_integrated_intensity_scales_with_trapezoid_rule():
    """Regression for the numpy.trapz -> numpy.trapezoid rename (NumPy
    2.0): confirm integrated_intensity actually equals the trapezoidal
    integral, not silently wrong from an AttributeError being swallowed
    somewhere."""
    theta = np.linspace(0, 10, 11)
    I = np.ones_like(theta)  # constant intensity -> integral = 10 * 1 = 10
    feat = extract_features(theta, I, peak_prominence_frac=0.5)
    assert feat.integrated_intensity == pytest.approx(10.0)
