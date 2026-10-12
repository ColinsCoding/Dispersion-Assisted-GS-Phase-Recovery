"""Tests for dgs.qiskit_phase_flip."""
import numpy as np
import pytest

pytest.importorskip("qiskit")
from dgs import qiskit_phase_flip as pf
from dgs import quantum_error_correction as q


def test_no_noise_is_identity():
    for s in pf.PAULI_STATES.values():
        assert pf.average_fidelity(pf.dephasing(0.0)) == pytest.approx(1.0)
        rho = pf.protected_state(s, pf.dephasing(0.0))
        assert np.allclose(rho.data, np.outer(s.data, s.data.conj()), atol=1e-12)


@pytest.mark.parametrize("qubit", [0, 1, 2])
def test_every_single_phase_flip_is_corrected(qubit):
    from qiskit.quantum_info import DensityMatrix, Operator, partial_trace, state_fidelity
    Z = Operator(np.diag([1.0, -1.0]))
    for s in pf.PAULI_STATES.values():
        dm = pf._prepare(s).evolve(pf.encoder()).evolve(Z, qargs=[qubit]).evolve(pf.syndrome_and_correct())
        assert state_fidelity(partial_trace(dm, [1, 2, 3, 4]), s) == pytest.approx(1.0)


@pytest.mark.parametrize("p", [0.01, 0.1, 0.3, 0.5, 0.7])
def test_average_fidelity_matches_closed_form(p):
    assert pf.average_fidelity(pf.dephasing(p), encoded=True) == pytest.approx(pf.average_fidelity_theory(p, True), abs=1e-12)
    assert pf.average_fidelity(pf.dephasing(p), encoded=False) == pytest.approx(pf.average_fidelity_theory(p, False), abs=1e-12)
    # same p_L as the NumPy module
    assert 1 - 1.5 * (1 - pf.average_fidelity(pf.dephasing(p))) == pytest.approx(1 - q.logical_error(p, 3), abs=1e-12)


def test_code_helps_below_one_half_only():
    assert pf.average_fidelity(pf.dephasing(0.2)) > pf.average_fidelity(pf.dephasing(0.2), encoded=False)
    assert pf.average_fidelity(pf.dephasing(0.7)) < pf.average_fidelity(pf.dephasing(0.7), encoded=False)


def test_coherent_phase_suppressed_to_fourth_order():
    th = np.array([0.02, 0.04, 0.08])
    bare = np.array([1 - pf.average_fidelity(pf.coherent_phase(t), encoded=False) for t in th])
    coded = np.array([1 - pf.average_fidelity(pf.coherent_phase(t), encoded=True) for t in th])
    assert np.polyfit(np.log(th), np.log(bare), 1)[0] == pytest.approx(2, abs=0.05)
    assert np.polyfit(np.log(th), np.log(coded), 1)[0] == pytest.approx(4, abs=0.1)


def test_wilson_interval_and_shots():
    lo, hi = pf.wilson_interval(0, 100)
    assert lo == 0.0 and 0 < hi < 0.05
    est, (lo, hi) = pf.logical_error_rate_from_shots(0.1, 20000, rng=1)
    assert lo <= q.logical_error(0.1, 3) <= hi
    with pytest.raises(ValueError):
        pf.dephasing(1.5)
