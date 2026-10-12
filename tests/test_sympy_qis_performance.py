"""Tests for dgs/sympy_qis_performance.py."""
import numpy as np
import pytest
import sympy as sp

from dgs.sympy_qis_performance import (
    a, b, c, circuit_full_state, circuit_state_vector, numeric_state, purity_expression,
    cse_savings, check_qis_units, gates_within_coherence, cnot_logic, toffoli_logic,
    logic_to_permutation, toffoli_is_nand, inputs_for_toffoli_output, time_call,
)

TH = sp.symbols("t0:8", real=True)
VALS = np.linspace(0.3, 1.1, 8)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_state_vector_equals_full_matrix(n):
    f_full = sp.lambdify(TH[:n], circuit_full_state(n, TH), "numpy")
    f_vec = sp.lambdify(TH[:n], circuit_state_vector(n, TH), "numpy")
    v1 = np.array(f_full(*VALS[:n]), float).ravel()
    v2 = np.array(f_vec(*VALS[:n]), float).ravel()
    assert np.allclose(v1, v2) and np.allclose(v2, numeric_state(n, VALS))
    assert np.linalg.norm(v2) == pytest.approx(1.0)


def test_state_vector_method_is_faster():
    t_full, _ = time_call(circuit_full_state, 5, TH)
    t_vec, _ = time_call(circuit_state_vector, 5, TH)
    assert t_vec < t_full / 5


def test_cse_shrinks_expression():
    before, after = cse_savings(purity_expression(4, TH))
    assert after < before / 2


def test_units():
    u = check_qis_units()
    assert u["Rabi d*E/hbar"] == "T^-1" and u["Omega*t"] == "1"
    assert gates_within_coherence(100e-6, 50e-9) == pytest.approx(2000)
    with pytest.raises(ValueError):
        gates_within_coherence(0, 1)


def test_reversible_logic():
    P = logic_to_permutation(cnot_logic(), (a, b))
    assert np.array_equal(P, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    T = logic_to_permutation(toffoli_logic(), (a, b, c))
    assert np.array_equal(T @ T, np.eye(8, dtype=int))          # self-inverse: reversible
    assert toffoli_is_nand()
    assert inputs_for_toffoli_output(True) == [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
