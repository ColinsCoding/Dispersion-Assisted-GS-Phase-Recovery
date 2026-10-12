"""Tests for dgs/opamp_inverting.py (Nilsson & Riedel Probs. 5.1-5.2)."""
import numpy as np
import pytest
import sympy as sp

from dgs.opamp_inverting import (
    ideal_solution, finite_gain_solution, output_voltage, is_saturated, linear_input_range,
    output_current, PROBLEM_5_2_SOURCES,
)


def test_ideal_gain():
    vo, (vs, R1, Rf) = ideal_solution()
    assert sp.simplify(vo + Rf / R1 * vs) == 0
    assert float(output_voltage(2.0)) == pytest.approx(-6.0)


def test_finite_gain_limits_to_ideal_and_virtual_short():
    vo_A, vo_inf, vn_A, (vs, R1, Rf, A) = finite_gain_solution()
    assert sp.simplify(vo_inf + Rf / R1 * vs) == 0
    assert sp.limit(vn_A, A, sp.oo) == 0                  # v_n -> v_p = 0: the virtual short
    val = float(vo_A.subs({vs: 2, R1: 4e3, Rf: 12e3, A: 1e5}))
    assert val == pytest.approx(-6.0, rel=1e-3) and val > -6.0


def test_problem_5_2():
    vo = output_voltage(PROBLEM_5_2_SOURCES)
    assert np.allclose(vo, [15.0, 10.5, 3.75, -7.2, -13.5, -15.0])
    assert list(is_saturated(PROBLEM_5_2_SOURCES)) == [True, False, False, False, False, True]
    assert linear_input_range() == (-5.0, 5.0)


def test_output_current_and_validation():
    assert float(output_current(2.0)) == pytest.approx(-6 / 6e3 - 6 / 12e3)
    with pytest.raises(ValueError):
        output_voltage(1.0, R1=0)
