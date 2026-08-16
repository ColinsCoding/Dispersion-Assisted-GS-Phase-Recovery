"""Tests for projects.seals.inverse.noise_robustness -- does the N-plane
classical GS fix (which resolves the STRUCTURAL 2-plane ambiguity, see
test_seals_to_tdgsa.py's test_more_measurement_diversity_dramatically_
improves_recovery) stay accurate once realistic measurement noise is added?
Answer: no -- noise is a separate, independent limitation from measurement
diversity, and this file pins that down as a tested fact."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from projects.seals.inverse.noise_robustness import n_plane_recovery_at_noise, sweep_noise_robustness


def test_noiseless_n_plane_recovery_matches_known_result():
    """Regression: noiseless N=3-plane recovery should reproduce the
    ~0.0014 rad result already established in test_seals_to_tdgsa.py."""
    r = n_plane_recovery_at_noise(noise_std=0.0)
    assert r["rms_vs_truth"] < 0.01


def test_accuracy_degrades_with_noise():
    """The actual finding: RMS-vs-truth grows roughly monotonically as
    measurement noise increases, even though the STRUCTURAL ambiguity
    (2 vs. 3 planes) is already resolved -- noise is a separate,
    independent limitation from measurement diversity."""
    results = sweep_noise_robustness(noise_levels=(0.0, 0.05, 0.3, 1.5))
    rms_vals = [r["rms_vs_truth"] for r in results]
    assert rms_vals == sorted(rms_vals), \
        f"expected RMS to grow monotonically with noise, got {rms_vals}"
    assert rms_vals[0] < 0.01           # noiseless: essentially exact
    assert rms_vals[-1] > 0.5           # heavy (150%) noise: back to badly wrong


def test_realistic_noise_still_reasonably_accurate():
    """At the ~5% noise level used elsewhere in this package
    (inverse_scattering.synthesize_measurement's convention), N=3-plane
    recovery should still be MUCH better than the 2-plane structural floor
    (~0.5 rad) -- noise degrades precision, but doesn't erase the benefit
    of the extra measurement plane."""
    r = n_plane_recovery_at_noise(noise_std=0.05)
    assert r["rms_vs_truth"] < 0.1


def test_n_plane_gs_still_fits_its_own_noisy_measurements():
    """Even under noise, classical GS's alternating-projection step still
    converges to near-zero SELF-consistency error (it fits whatever
    measurements it's given, noisy or not) -- distinguishing 'GS didn't
    converge' from 'GS converged to something noise-corrupted', the same
    distinction test_bridge_demo_runs_and_gs_fits_its_own_measurements makes
    for the 2-plane case. This is WHY GS has no overfitting failure mode
    here (contrast seals_to_tdgsa.demonstrate_autograd_overfitting): its
    hard amplitude projection reaches this fixed point almost immediately
    regardless of noise or n_iter, so there's no notion of "training too
    long" to overfit with."""
    r = n_plane_recovery_at_noise(noise_std=0.3)
    assert r["gs_final_error"] < 1e-6


def test_reproducible_given_same_seed():
    r1 = n_plane_recovery_at_noise(noise_std=0.3, seed=5)
    r2 = n_plane_recovery_at_noise(noise_std=0.3, seed=5)
    assert r1["rms_vs_truth"] == r2["rms_vs_truth"]
    np.testing.assert_array_equal(r1["phi_recovered"], r2["phi_recovered"])


def test_rejects_negative_noise():
    import pytest
    with pytest.raises(ValueError, match="noise_std"):
        n_plane_recovery_at_noise(noise_std=-0.1)
