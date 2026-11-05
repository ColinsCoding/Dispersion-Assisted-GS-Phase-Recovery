"""Tests for dgs/phase_shifting.py: N-step phase-shifting interferometry vs dispersion-assisted GS."""
import numpy as np
import pytest

from dgs import phase_shifting as P
from dgs import gs_core as gs

PHI = gs.make_qpsk_measurements()["phi_true"]        # the README's test phase


@pytest.mark.parametrize("n", [3, 4, 5, 8])
def test_noiseless_recovery_is_exact(n):
    amp = 0.5 + 0.4 * np.cos(np.linspace(0, 6, PHI.size))      # works with non-constant amplitude too
    phi, a = P.psi_retrieve(P.psi_frames(amp * np.exp(1j * PHI), R=1.3, n_steps=n), R=1.3)
    assert P.wrapped_rms(phi, PHI) < 1e-12                   # no offset or sign removed
    assert np.allclose(a, amp, atol=1e-12)


def test_four_step_is_the_atan2_formula():
    I = P.psi_frames(np.exp(1j * PHI))
    assert np.allclose(P.psi_retrieve(I)[0], np.arctan2(I[1] - I[3], I[0] - I[2]), atol=1e-12)


@pytest.mark.parametrize("snr_db", [15.0, 25.0, 35.0])
def test_noise_matches_prediction(snr_db):
    r = P.compare_with_gs(snr_db=snr_db)
    assert abs(r["psi_rms"] / r["psi_predicted"] - 1) < 0.1


def test_readme_numbers():
    r = P.compare_with_gs(snr_db=25.0)
    assert abs(r["gs_rms"] - 0.061) < 0.0015                  # the README's GS result
    assert abs(r["psi_rms"] - 0.020) < 0.0015
    assert r["psi_rms"] < r["gs_rms"]


@pytest.mark.parametrize("eps", [0.01, 0.05, 0.1])
def test_step_miscalibration_is_offset_plus_half_eps_ripple(eps):
    phi = P.psi_retrieve(P.psi_frames(np.exp(1j * PHI), step_error_rad=eps))[0]
    d = np.angle(np.exp(1j * (phi - PHI)))
    off = np.angle(np.mean(np.exp(1j * d)))
    ripple = np.abs(np.angle(np.exp(1j * (d - off)))).max()
    assert abs(off) > 1.4 * eps                              # mostly a constant offset
    assert abs(ripple / (eps / 2) - 1) < 0.1                 # plus a ripple of peak ~eps/2


def test_validation():
    with pytest.raises(ValueError):
        P.phase_steps(2)
    with pytest.raises(ValueError):
        P.psi_frames(np.ones(8), R=0)
    with pytest.raises(ValueError):
        P.psi_retrieve(np.ones(8))
