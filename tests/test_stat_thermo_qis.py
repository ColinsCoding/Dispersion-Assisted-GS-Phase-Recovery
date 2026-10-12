"""Tests for dgs.stat_thermo_qis."""
import numpy as np
import pytest
import sympy as sp

from dgs import stat_thermo_qis as s
from dgs.stat_thermo_qis import eps, T, k


def test_two_level_thermodynamics():
    th = s.two_level_symbolic()
    assert sp.simplify(th["U"] - eps / (sp.exp(eps / (k * T)) + 1)) == 0
    assert s.gibbs_equals_shannon_two_level() == 0
    assert s.fluctuation_dissipation_two_level() == 0
    assert s.schottky_peak() == pytest.approx(0.4168, abs=1e-3)


def test_thermal_qubit_and_density_matrix():
    rho, bz, d = s.thermal_density_matrix_symbolic()
    b = sp.symbols("beta", positive=True)
    assert sp.simplify(bz + sp.tanh(b * eps / 2)) == 0 and d == 0
    q = s.thermal_qubit(5e9, 0.02)
    assert q["p_excited"] == pytest.approx(6.16e-6, rel=0.01)
    hot = s.thermal_qubit(5e9, 300)
    assert hot["entropy_bits"] == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(ValueError):
        s.thermal_qubit(0, 1)


def test_oscillator():
    Z, U, nbar = s.oscillator_symbolic()
    b, w, h = sp.symbols("beta omega hbar", positive=True)
    assert sp.simplify((Z - 1 / (2 * sp.sinh(b * h * w / 2))).rewrite(sp.exp)) == 0
    assert sp.simplify((nbar - 1 / (sp.exp(b * h * w) - 1)).rewrite(sp.exp)) == 0
    assert s.bose_occupation(5e9, 300) == pytest.approx(s.K_B * 300 / (s.H * 5e9) - 0.5, rel=1e-3)


def test_box_counting():
    assert s.count_states_loop(500) == s.count_states_vectorized(500)
    d = s.degeneracies(20)
    assert d[3] == 1 and d[6] == 3 and d[14] == 6
    N, g, E = s.octant_count_symbolic()
    exact = s.count_states_vectorized(40000)
    approx = float(N.subs(E, 40000) - 3 * sp.pi / 8 * 40000)      # with surface correction
    assert abs(approx - exact) / exact < 1e-3


def test_ideal_gas():
    assert s.translational_Z_symbolic()[1] == 0
    assert s.sackur_tetrode_symbolic() == 0
    assert s.sackur_tetrode_molar(39.948) == pytest.approx(154.8, abs=0.2)
    assert s.sackur_tetrode_molar(4.0026) == pytest.approx(126.2, abs=0.2)
    norm, mean, rms, vp = s.maxwell_speed_symbolic()
    m = sp.symbols("m", positive=True)
    assert norm == 1
    assert sp.simplify(rms - sp.sqrt(3 * k * T / m)) == 0
    assert sp.simplify(vp - sp.sqrt(2 * k * T / m)) == 0


def test_information_is_physical():
    assert s.bit_entropy_symbolic() == k * sp.log(2)
    assert sp.simplify(s.szilard_work_symbolic() - k * T * sp.log(2)) == 0
    assert s.landauer_limit(300) == pytest.approx(2.871e-21, rel=1e-3)
    assert s.required_temperature(5e9, 1e-3) == pytest.approx(0.0347, rel=0.01)
    with pytest.raises(ValueError):
        s.landauer_limit(0)
