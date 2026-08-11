"""Tests for projects.seals.inverse.inverse_scattering, plus a SEALS regression
guard for the underlying (unmodified) forward model."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from projects.seals.inverse import _seals_physics as physics
from projects.seals.inverse import inverse_scattering as inv


def test_seals_regression_default_parameters_unchanged():
    """
    Locks in the validated default-parameter SEALS/Mie outputs so a future
    change to _seals_physics.py (or an accidental edit to seals_stable.py)
    would be caught here. These reference values were independently
    cross-checked against a real saved MATLAB workspace (test1.mat) in a
    separate validation pass; this repo does not carry that .mat file, so
    the check here is a numeric-literal snapshot rather than a live MATLAB
    comparison, per this task's "if MATLAB reference arrays exist, do not
    break those comparisons" -- none exist in this repo, so this is the
    regression guard for it going forward.
    """
    p = physics.P_DEFAULT
    lamvec = np.linspace(p['lam1'], p['lam2'], p['N_lam'])
    lam0 = 0.5 * (p['lam1'] + p['lam2'])

    y, theta, valid = physics.seals(p['d'], p['D'], p['a'], p['dcorr'], p['P'], p['NA'], lamvec)
    theta_shifted = theta + p['mangle']
    theta_rad = np.deg2rad(theta_shifted)
    sigma_s, I_p, I_s, an, bn, T_p, T_s = physics.mie(
        p['npar'], p['nmed'], p['dia'], lam0, theta_rad, p['r'])
    I_tot = I_p + I_s

    assert np.isclose(y[0], 0.005227542221385073, rtol=1e-9)
    assert y[-1] == 0.0
    assert np.isclose(theta_shifted[0], 56.500605150906715, rtol=1e-9)
    assert np.isclose(theta_shifted[-1], -25.76104631226272, rtol=1e-9)
    assert np.isclose(sigma_s, 1.6400057034569075e-10, rtol=1e-9)
    assert np.isclose(I_tot[0], 9.425557256866724e-10, rtol=1e-6)
    assert np.isclose(I_tot.max(), 5.402732414999209e-07, rtol=1e-6)
    assert int(I_tot.argmax()) == 363


def test_diameter_recovery_from_close_start():
    """Model-based inverse scattering: recover a known diameter from a
    synthetic noisy intensity spectrum, starting near (not at) the truth."""
    p = physics.P_DEFAULT
    lamvec = np.linspace(p['lam1'], p['lam2'], p['N_lam'])
    lam0 = 0.5 * (p['lam1'] + p['lam2'])
    y, theta, _ = physics.seals(p['d'], p['D'], p['a'], p['dcorr'], p['P'], p['NA'], lamvec)
    theta_rad = np.deg2rad(theta + p['mangle'])

    true_diameter = p['dia']
    I_meas = inv.synthesize_measurement(true_diameter, p['npar'], p['nmed'], lam0,
                                         theta_rad, p['r'], noise_std=0.05, seed=42)

    bounds = (true_diameter * 0.9, true_diameter * 1.1)
    result = inv.estimate_diameter(I_meas, p['npar'], p['nmed'], lam0, theta_rad, p['r'], bounds)

    rel_error = abs(result.diameter - true_diameter) / true_diameter
    assert rel_error < 0.05   # within 5% given 5% measurement noise and a search confined near truth

    # regression guard: scipy's default xatol=1e-5 for method='bounded' is in meters, the
    # same units as `dia` -- for this bracket (~1e-6 m wide) that default is LARGER than
    # the whole search interval, so minimize_scalar silently "converges" after a single
    # evaluation without actually searching. estimate_diameter must override xatol; this
    # assertion catches a regression back to that silent-unit-mismatch trap.
    assert result.n_evals > 5

    # step 6: the recovered diameter's predicted phase comes from the SAME
    # validated Mie model, not a separate phase-retrieval optimization
    assert result.predicted_fields.phase_p.shape == theta_rad.shape
    assert np.all(np.isfinite(result.predicted_fields.phase_p))


def test_log_intensity_loss_is_zero_at_true_parameters():
    """Sanity check on the loss function itself: noiseless measurement at
    the true diameter should give exactly zero loss."""
    p = physics.P_DEFAULT
    lamvec = np.linspace(p['lam1'], p['lam2'], p['N_lam'])
    lam0 = 0.5 * (p['lam1'] + p['lam2'])
    y, theta, _ = physics.seals(p['d'], p['D'], p['a'], p['dcorr'], p['P'], p['NA'], lamvec)
    theta_rad = np.deg2rad(theta + p['mangle'])

    sigma_s, I_p, I_s, an, bn, T_p, T_s = physics.mie(
        p['npar'], p['nmed'], p['dia'], lam0, theta_rad, p['r'])
    I_noiseless = I_p + I_s

    loss = inv.log_intensity_loss(p['dia'], p['npar'], p['nmed'], lam0, theta_rad, p['r'], I_noiseless)
    assert loss < 1e-20
