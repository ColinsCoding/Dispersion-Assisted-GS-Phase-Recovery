"""Tests for dgs/planck_quantization_taylor.py."""
import numpy as np
import pytest
import sympy as sp

from dgs.planck_quantization_taylor import (
    x, mean_energy_symbolic, mean_energy_numeric, taylor_low_frequency, large_x_ratio_limit,
    taylor_error, wien_temperature, nearest_option, full_adder_truth_table, ripple_add,
)


def test_integer_multiples_give_planck():
    Z, S1, E = mean_energy_symbolic()
    assert sp.simplify(E - x / (sp.exp(x) - 1)) == 0
    for xv in (0.1, 1.0, 4.0):
        assert mean_energy_numeric(xv) == pytest.approx(xv / np.expm1(xv), rel=1e-10)


def test_taylor_and_limits():
    s = taylor_low_frequency(6).removeO()
    assert sp.simplify(s - (1 - x / 2 + x ** 2 / 12 - x ** 4 / 720)) == 0
    assert large_x_ratio_limit() == 1
    errs = [taylor_error(0.5, o) for o in (2, 4, 6)]
    assert errs[0] > errs[1] > errs[2]                       # more terms, smaller error
    assert taylor_error(0.5, 6) < taylor_error(3.0, 6)       # worse far from the centre


def test_red_hot_iron():
    T = wien_temperature(720)
    assert T == pytest.approx(4025, abs=1)
    assert nearest_option(T, [4000, 1000, 5000, 2000, 3000]) == 4000
    with pytest.raises(ValueError):
        wien_temperature(0)


def test_full_adder():
    rows = full_adder_truth_table()
    assert len(rows) == 8 and all(r[5] == r[6] for r in rows)   # a+b+c = 2 cout + s
    assert ripple_add(200, 100) == 300 and ripple_add(255, 255) == 510
    with pytest.raises(ValueError):
        ripple_add(256, 0)
