"""Tests for dgs.quantum_error_correction."""
import numpy as np
import pytest
import sympy as sp

from dgs import quantum_error_correction as q


def test_syndrome_table_and_correction():
    t = q.syndrome_table()
    assert t == {"none": (1, 1), "X on qubit 0": (-1, 1), "X on qubit 1": (-1, -1), "X on qubit 2": (1, -1)}
    for qubit in range(3):
        assert q.correction_for(t[f"X on qubit {qubit}"]) == qubit


def test_logical_error_formula_and_monte_carlo():
    p = sp.symbols("p", positive=True)
    assert q.logical_error_symbolic(3) == 3 * p**2 - 2 * p**3
    assert q.code_capacity_crossover() == [sp.Rational(1, 2)]
    r = [q.run_bit_flip(0.6, 0.8, 0.1, rng=i) for i in range(6000)]
    assert np.mean([e for _, e in r]) == pytest.approx(q.logical_error(0.1, 3), abs=0.006)
    assert 1 - np.mean([f for f, _ in r]) < np.mean([e for _, e in r])     # fidelity under-counts
    with pytest.raises(ValueError):
        q.logical_error_symbolic(4)


def test_bigger_codes_better_below_threshold():
    assert q.logical_error(0.05, 7) < q.logical_error(0.05, 5) < q.logical_error(0.05, 3) < 0.05
    assert q.logical_error(0.6, 7) > q.logical_error(0.6, 3)


def test_phase_flip_and_hamming():
    a, b = q.hadamard_conjugation_check()
    assert a == sp.zeros(2) and b == sp.zeros(2)
    assert [q.hamming_syndrome(np.eye(7, dtype=int)[j]) for j in range(7)] == list(range(1, 8))
    cw = q.hamming_codewords()
    assert len(cw) == 16 and q.min_distance(cw) == 3
    assert not q.steane_check_commutation().any()


def test_matching_decoder():
    rng = np.random.default_rng(0)
    # single data error in one round is always corrected
    for _ in range(50):
        assert not q.logical_failure(5, 5, 0.0, 0.0, rng)
    # low noise: rare failures; bigger code fails less
    lo3 = np.mean([q.logical_failure(3, 3, 0.03, 0.03, rng) for _ in range(600)])
    lo7 = np.mean([q.logical_failure(7, 7, 0.03, 0.03, rng) for _ in range(600)])
    assert lo7 < lo3 < 0.1


def test_overhead_and_heat():
    d = q.surface_code_distance(1e-3, 1e-12)
    assert d % 2 == 1 and q.surface_code_qubits(d) == 2 * d * d - 1
    Qh, Qc, eta = q.carnot_cycle_symbolic()
    Th, Tc = sp.symbols("T_h T_c", positive=True)
    assert sp.simplify(eta - (1 - Tc / Th)) == 0
    assert sp.simplify(q.carnot_cop_symbolic() - Tc / (Th - Tc)) == 0
    assert q.wall_power_for_cooling(1e-6, 0.02) == pytest.approx(1e-6 * 299.98 / 0.02)
    assert q.qec_landauer_heat(1, 3, 1.0, 1.0) == pytest.approx(8 * q.K_B * np.log(2))
