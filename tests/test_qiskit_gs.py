"""Tests for dgs/qiskit_gs.py: the Qiskit dispersion circuit and Gerchberg-Saxton on its measurements."""
import numpy as np
import pytest

pytest.importorskip('qiskit')
from qiskit.quantum_info import Operator   # noqa: E402

from dgs import gs_core                     # noqa: E402
from dgs import qiskit_gs as Q              # noqa: E402

N_QUBITS = 7
D1, D2 = -1000.0, -1150.0


@pytest.fixture(scope='module')
def setup():
    t, E, phi = Q.smooth_phase_field(N_QUBITS, n_harmonics=8, seed=0)
    c1, c2 = Q.dispersion_circuit(N_QUBITS, D1), Q.dispersion_circuit(N_QUBITS, D2)
    return E, phi, c1, c2, Operator(c1).data, Operator(c2).data


def test_circuit_equals_gs_core_dispersion(setup):
    E, _, _, _, U1, U2 = setup
    assert np.allclose(U1 @ E, gs_core.disperse(E, D1), atol=1e-12)
    assert np.allclose(U2 @ E, gs_core.disperse(E, D2), atol=1e-12)


def test_circuit_is_unitary(setup):
    *_, U1, _ = setup
    assert np.allclose(U1.conj().T @ U1, np.eye(len(U1)), atol=1e-12)


def test_exact_measurement_is_detector_intensity(setup):
    E, _, c1, *_ = setup
    assert np.allclose(Q.measure_intensity(E, c1), np.abs(gs_core.disperse(E, D1))**2, atol=1e-12)


def test_sampled_measurement_keeps_energy(setup):
    E, _, c1, *_ = setup
    I = Q.measure_intensity(E, c1, shots=20_000, seed=3)
    assert np.all(I >= 0) and abs(I.sum() - np.vdot(E, E).real) < 1e-9


def test_recovery_exact_and_matches_gs_core(setup):
    E, phi, c1, c2, U1, U2 = setup
    I1, I2 = Q.measure_intensity(E, c1), Q.measure_intensity(E, c2)
    Er, _ = Q.gs_retrieve(I1, I2, U1, U2, n_iter=100)
    assert Q.align_rms(np.angle(Er), phi)[0] < 1e-3
    phc, _ = gs_core.retrieve_phase(I1, I2, D1, D2, n_iter=100)
    assert np.allclose(np.angle(Er), phc, atol=1e-6)


def test_recovery_from_shots(setup):
    E, phi, c1, c2, U1, U2 = setup
    I1 = Q.measure_intensity(E, c1, shots=100_000, seed=1)
    I2 = Q.measure_intensity(E, c2, shots=100_000, seed=2)
    Er, _ = Q.gs_retrieve(I1, I2, U1, U2, n_iter=100)
    assert Q.align_rms(np.angle(Er), phi)[0] < 0.05


def test_too_little_dispersion_fails(setup):
    E, phi, *_ = setup
    c1, c2 = Q.dispersion_circuit(N_QUBITS, -200.0), Q.dispersion_circuit(N_QUBITS, -230.0)
    Er, _ = Q.gs_retrieve(Q.measure_intensity(E, c1), Q.measure_intensity(E, c2), Operator(c1).data, Operator(c2).data)
    assert Q.align_rms(np.angle(Er), phi)[0] > 0.5      # the edge case: weak dispersion, ill-posed
