"""Tests for dgs.qm_phase_retrieval (1D QM = dispersion-assisted phase retrieval)."""
import warnings

import numpy as np
import pytest

from dgs import qm_phase_retrieval as q
from dgs import diffraction_dispersion as dd


def test_time_dispersion_dictionary_roundtrip():
    assert np.isclose(q.time_from_D(q.D_from_time(3.7, dx=0.2), dx=0.2), 3.7)
    assert np.isclose(q.D_from_time(1.0), -2 * np.pi)


def test_free_evolution_matches_duality_module_and_closed_form():
    x = (np.arange(2048) - 1024) * 0.1
    p0 = q.gaussian_packet(x, 1.0, k0=2.0)
    pt = q.free_evolve(p0, 3.0, dx=0.1)
    assert np.allclose(pt, dd.quadratic_phase_propagate(p0, 0.1, -3.0), atol=1e-12)   # p = -hbar t/m
    nrm, mean, width = q.density_moments(x, pt, 0.1)
    assert np.isclose(nrm, 1) and np.isclose(mean, 6.0) and np.isclose(width, q.packet_width(1.0, 3.0))


def test_time_reversal_identity():
    psi = q.phase_only_state(256)
    assert q.time_reversal_check(psi, 500.0) < 1e-12


def test_pauli_retrieval_ideal():
    psi = q.phase_only_state(512)
    I1, I2 = q.pauli_measurements(psi, 800.0, 920.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec, err = q.pauli_retrieve(I1, I2, 800.0, 920.0, n_iter=150)
    assert q.phase_rms_error(psi, rec) < 1e-6
    assert err[-1] < 1e-6 * err[0]


def test_global_phase_and_twin():
    psi = q.phase_only_state(512)
    I1, I2 = q.pauli_measurements(psi, 800.0, 920.0)
    assert q.measurement_mismatch(np.exp(0.7j) * psi, I1, I2, 800.0, 920.0) < 1e-12   # gauge: invisible
    assert q.measurement_mismatch(np.conj(psi), I1, I2, 800.0, 920.0) > 0.1          # twin: rejected
    assert q.measurement_mismatch(np.conj(psi), I1, I2, -800.0, -920.0) < 1e-12      # ...fits at -t


def test_adc_bits_improve_accuracy():
    res = q.adc_bit_sweep(bits_list=(4, 8, 12), n=256, n_iter=120)
    errs = [r[1] for r in res]
    assert errs[0] > errs[1] > errs[2]
    assert 6 < errs[0] / errs[1] < 40          # ~4x per 2 bits -> ~16x per 4 bits


def test_quantize():
    I = np.linspace(0, 1, 11)
    assert np.allclose(q.quantize(I, None), I)
    Q = q.quantize(I, 1)
    assert set(np.round(Q, 12)) <= {0.0, 1.0}


def test_qubit_two_bases_resolve_sign():
    for phi in (-2.5, -1.0, 0.3, 2.0):
        s = q.qubit_state(np.pi / 2, phi)
        assert np.isclose(q.basis_probability0(s, "Z"), 0.5)                 # Z: blind to phase
        px, py = q.basis_probability0(s, "X"), q.basis_probability0(s, "Y")
        assert np.isclose(abs(q.phase_from_x_only(px)[0]), abs(phi))          # X: |phi| only
        assert np.isclose(q.phase_from_xy(px, py), phi)                      # X + Y: phi


def test_flop_model():
    assert q.gs_flops_per_iteration(1024) == 4 * 5 * 1024 * 10 + (24 + 20 + 12) * 1024


@pytest.mark.parametrize("call", [
    lambda: q.D_from_time(1.0, m=0),
    lambda: q.gaussian_packet(np.zeros(4), 0.0),
    lambda: q.quantize(np.ones(3), 0),
    lambda: q.pauli_measurements(np.ones(8), 1.0, 1.0),
    lambda: q.basis_probability0(np.array([1, 0]), "W"),
    lambda: q.gs_flops_per_iteration(1),
])
def test_validation(call):
    with pytest.raises(ValueError):
        call()
