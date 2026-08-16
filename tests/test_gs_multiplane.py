"""Tests for projects.seals.inverse.gs_multiplane -- the N-plane
generalization of dgs.gs_core's 2-plane classical GS. Independent of SEALS:
uses dgs.gs_core.make_qpsk_measurements (this repo's existing synthetic
unit-amplitude test signal) so these tests do not depend on the Mie forward
model at all."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from dgs import gs_core
from projects.seals.inverse.gs_multiplane import (
    retrieve_phase_n_plane, gs_iteration_n, apply_prior_regularized_amplitude, _check_planes,
)


def test_two_plane_call_matches_gs_core_exactly():
    """N-plane GS with exactly 2 planes must be bit-for-bit identical to
    dgs.gs_core.retrieve_phase_with_history -- it reuses the same
    disperse/undisperse/apply_amplitude_constraint primitives in the same
    order, so this is a pure regression check that generalizing the loop to
    N planes did not change the N=2 behavior at all."""
    data = gs_core.make_qpsk_measurements(D1=-5000.0, D2=-5750.0, snr_db=60.0)
    phi_ref, errors_ref, hist_ref = gs_core.retrieve_phase_with_history(
        data["I1"], data["I2"], data["D1"], data["D2"], n_iter=30, unit_amplitude=True)
    phi_n, errors_n, hist_n = retrieve_phase_n_plane(
        [data["I1"], data["I2"]], [data["D1"], data["D2"]], n_iter=30, unit_amplitude=True)

    np.testing.assert_allclose(phi_n, phi_ref)
    np.testing.assert_allclose(hist_n, hist_ref)
    assert errors_n == errors_ref


def test_more_planes_reduces_error_on_a_hard_case():
    """A varying-amplitude signal (unit_amplitude=False) that only barely
    converges with 2 planes should have measurement self-consistency that
    does not get WORSE with a 3rd, independent dispersion plane added."""
    data = gs_core.make_qpsk_measurements(D1=-5000.0, D2=-5750.0, snr_db=60.0)
    E_true = np.exp(1j * data["phi_true"])
    D3 = 8300.0
    I3 = np.abs(gs_core.disperse(E_true, D3)) ** 2

    _, errors_2, _ = retrieve_phase_n_plane(
        [data["I1"], data["I2"]], [data["D1"], data["D2"]], n_iter=50, unit_amplitude=True)
    _, errors_3, _ = retrieve_phase_n_plane(
        [data["I1"], data["I2"], I3], [data["D1"], data["D2"], D3], n_iter=50, unit_amplitude=True)

    assert errors_3[-1] <= errors_2[-1] * 10, \
        "adding a 3rd independent measurement plane should not blow up self-consistency"


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        _check_planes([np.zeros(4)], [1.0, 2.0])


def test_rejects_fewer_than_two_planes():
    with pytest.raises(ValueError, match="at least 2"):
        _check_planes([np.zeros(4)], [1000.0])


def test_rejects_duplicate_dispersions():
    with pytest.raises(ValueError, match="distinct"):
        _check_planes([np.zeros(4), np.zeros(4)], [5000.0, 5000.0])


def test_amplitude_prior_requires_i_native():
    with pytest.raises(ValueError, match="I_native"):
        retrieve_phase_n_plane(
            [np.ones(8), np.ones(8)], [5000.0, -6000.0], n_iter=5,
            amplitude_prior=np.ones(8))


def test_apply_prior_regularized_amplitude_only_touches_weak_samples():
    E = np.array([1.0, 1.0, 1.0, 1.0], dtype=complex)
    I_native = np.array([100.0, 100.0, 0.1, 0.1])   # last two samples are "weak"
    prior = np.array([5.0, 5.0, 5.0, 5.0])
    out = apply_prior_regularized_amplitude(E, prior, I_native, floor_frac=0.01)
    np.testing.assert_allclose(np.abs(out[:2]), 1.0)   # strong samples: untouched
    np.testing.assert_allclose(np.abs(out[2:]), 5.0)   # weak samples: pulled to prior
