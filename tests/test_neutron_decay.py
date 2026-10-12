"""Tests for dgs/neutron_decay.py."""
import numpy as np
import pytest
import sympy as sp

from dgs.neutron_decay import (
    symbolic_q_value, delta_m, q_value_ev, q_value_units, nearest_option,
    significant_figures_lost, max_electron_kinetic_ev, max_proton_kinetic_ev,
    common_mistakes, beta_spectrum_allowed, endpoint_sum_identity,
)


def test_symbolic_solution():
    K, (mn, mp, me, c) = symbolic_q_value()
    assert sp.simplify(K - (mn - mp - me) * c ** 2) == 0


def test_q_value_and_answer():
    assert float(q_value_ev()) == pytest.approx(782_313, rel=1e-5)
    assert float(q_value_units()[0].args[0]) == pytest.approx(float(q_value_ev()), rel=1e-12)
    assert nearest_option(q_value_ev()) == "1 MeV"
    assert delta_m() == sp.Rational("1.3946e-30")


def test_cancellation_costs_three_digits():
    assert 3.0 < significant_figures_lost() < 3.2


def test_endpoints():
    assert float(max_proton_kinetic_ev()) == pytest.approx(751.1, abs=0.2)
    assert float(max_electron_kinetic_ev()) < float(q_value_ev())
    assert endpoint_sum_identity() == 0
    assert max_electron_kinetic_ev() + max_proton_kinetic_ev() == q_value_ev()


def test_mistakes_map_to_wrong_options():
    m = common_mistakes()
    assert m["correct: (m_n - m_p - m_e) c^2"][1] == "1 MeV"
    for k in ("exponent slip: Delta_m = 1.3946e-33 kg",
              "divided by 1.6e-16 (J per keV) but read it as eV",
              "reported proton recoil only (T_p,max)"):
        assert m[k][1] == "1 keV"


def test_spectrum_shape():
    Q = float(q_value_ev())
    T = np.linspace(0, Q, 2001)
    S = beta_spectrum_allowed(T)
    assert S[0] == 0 and S[-1] == 0 and np.all(S >= 0)
    assert beta_spectrum_allowed([Q * 1.1])[0] == 0
    mean_T = np.sum(T * S) / np.sum(S)
    assert 0.25 * Q < mean_T < 0.5 * Q


def test_validation():
    with pytest.raises(ValueError):
        nearest_option(0)
