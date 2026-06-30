"""Tests for dgs/physical_chemistry.py."""
import numpy as np
import sympy as sp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.physical_chemistry import (
    arrhenius, arrhenius_two_temp, first_order_kinetics,
    gibbs_free_energy, vant_hoff, clausius_clapeyron,
    boltzmann_distribution, michaelis_menten,
    water_phase, physical_chemistry_sympy_5,
    R_GAS, kB,
)


# ── Arrhenius ─────────────────────────────────────────────────────────────────

def test_arrhenius_zero_Ea():
    res = arrhenius(1e6, 0.0, 300)
    assert abs(res["k"] - 1e6) < 1e-6


def test_arrhenius_higher_T_faster():
    k1 = arrhenius(1e13, 50, 298)["k"]
    k2 = arrhenius(1e13, 50, 350)["k"]
    assert k2 > k1


def test_arrhenius_two_temp_ratio():
    res = arrhenius_two_temp(1e13, 50, 298, 308)
    # Q10 rule: ~2x per 10 degrees for biological reactions
    assert 1.5 < res["ratio_k2_k1"] < 3.5


def test_first_order_half_life():
    k = 0.1
    res = first_order_kinetics(k, C0=1.0, t_arr=[np.log(2)/k])
    assert abs(res["C"][0] - 0.5) < 1e-10


def test_first_order_decay():
    res = first_order_kinetics(0.2, 10.0, np.linspace(0, 20, 100))
    assert np.all(np.diff(res["C"]) <= 0)


# ── Thermodynamics ────────────────────────────────────────────────────────────

def test_gibbs_spontaneous_exothermic():
    g = gibbs_free_energy(-100.0, 50.0, 298)
    assert bool(g["spontaneous"]) is True
    assert g["dG_kJ_mol"] < 0


def test_gibbs_K_eq_unity_at_dG_zero():
    # dG = 0 -> K_eq = 1
    # dH = T*dS -> T_crossover: solve
    g = gibbs_free_energy(0.0, 0.0, 298)
    assert abs(g["K_eq"] - 1.0) < 1e-10


def test_vant_hoff_endothermic():
    # Endothermic: higher T increases K
    res = vant_hoff(K1=0.01, T1_K=300, T2_K=400, delta_H_kJ_mol=50)
    assert res["K2"] > res["K1"]


def test_clausius_clapeyron_water():
    # Water at 373K, 1 atm -> 355K should be below 1 atm
    cc = clausius_clapeyron(373.15, 101325, 355.0, 40.7)
    assert cc["P2_atm"] < 1.0


def test_clausius_clapeyron_higher_T_higher_P():
    cc1 = clausius_clapeyron(373.15, 101325, 380, 40.7)
    cc2 = clausius_clapeyron(373.15, 101325, 400, 40.7)
    assert cc2["P2_Pa"] > cc1["P2_Pa"]


# ── Boltzmann ─────────────────────────────────────────────────────────────────

def test_boltzmann_populations_sum_to_one():
    E = np.array([0, 0.01, 0.05]) * 1.6e-19
    b = boltzmann_distribution(E, 300)
    assert abs(b["populations"].sum() - 1.0) < 1e-10


def test_boltzmann_ground_state_most_populated():
    E = np.array([0, 0.1, 0.2]) * 1.6e-19
    b = boltzmann_distribution(E, 300)
    assert b["populations"][0] > b["populations"][1] > b["populations"][2]


def test_boltzmann_high_T_equal_populations():
    # Very high T -> all states equally probable
    E = np.array([0.0, 1e-30, 2e-30])
    b = boltzmann_distribution(E, 1e10)
    np.testing.assert_allclose(b["populations"], 1/3, atol=1e-4)


# ── Michaelis-Menten ──────────────────────────────────────────────────────────

def test_mm_half_vmax_at_Km():
    mm = michaelis_menten(Vmax=10.0, Km=2.0, S_arr=[2.0])
    assert abs(mm["v"][0] - 5.0) < 1e-10


def test_mm_saturates_at_high_S():
    mm = michaelis_menten(Vmax=10.0, Km=2.0, S_arr=[1000.0])
    assert mm["v"][0] > 9.9


def test_mm_zero_rate_at_zero_S():
    mm = michaelis_menten(Vmax=10.0, Km=2.0, S_arr=[0.0])
    assert abs(mm["v"][0]) < 1e-10


def test_mm_efficiency_between_0_and_1():
    mm = michaelis_menten(Vmax=5.0, Km=1.0, S_arr=np.linspace(0.01, 100, 50))
    assert np.all(mm["efficiency"] >= 0)
    assert np.all(mm["efficiency"] <= 1)


# ── Phase ─────────────────────────────────────────────────────────────────────

def test_water_liquid_at_room():
    w = water_phase(25, 1.0)
    assert "liquid" in w["phase"]


def test_water_ice_below_zero():
    w = water_phase(-10, 1.0)
    assert "solid" in w["phase"]


# ── SymPy ─────────────────────────────────────────────────────────────────────

def test_sympy_5_count():
    eqs = physical_chemistry_sympy_5()
    assert len(eqs) == 5


def test_sympy_arrhenius_form():
    eqs = physical_chemistry_sympy_5()
    eq = eqs["Arrhenius"]
    assert isinstance(eq, sp.Eq)
    syms = {str(s) for s in eq.rhs.free_symbols}
    assert "A" in syms and "T" in syms


def test_sympy_gibbs_form():
    eqs = physical_chemistry_sympy_5()
    eq = eqs["Gibbs_free_energy"]
    assert isinstance(eq, sp.Eq)
    syms = {str(s) for s in eq.rhs.free_symbols}
    assert "Delta_H" in syms and "T" in syms
