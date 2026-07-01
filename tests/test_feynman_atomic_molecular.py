import numpy as np
import pytest
import sympy as sp
from dgs.feynman_atomic_molecular import (
    ideal_gas, maxwell_boltzmann_speed,
    bohr_hydrogen, hydrogen_spectral_line, hydrogen_series,
    hydrogen_molecule_binding, london_dispersion,
    planck_spectrum, selection_rule_check,
    feynman_atomic_sympy_5,
    k_B, E_rydberg,
)


# ── ideal gas / kinetic theory ────────────────────────────────────────

def test_ideal_gas_pv_equals_nkt():
    N = 6.022e23
    T = 300
    V = 0.0224  # 1 mole at STP ~22.4 L
    r = ideal_gas(N, T, V_m3=V)
    assert r["PV_product_J"] == pytest.approx(r["NkT"], rel=0.01)

def test_ideal_gas_compute_volume():
    N = 1e22
    T = 300
    P = 101325  # 1 atm
    r = ideal_gas(N, T, P_Pa=P)
    assert r["V_m3"] > 0

def test_ideal_gas_invalid_temp():
    with pytest.raises(ValueError):
        ideal_gas(1e23, 0, V_m3=0.001)


# ── Maxwell-Boltzmann ─────────────────────────────────────────────────

def test_mb_rms_gt_avg_gt_mp():
    m_H = 1.67e-27
    r = maxwell_boltzmann_speed(300, m_H, 1000)
    assert r["v_rms_ms"] > r["v_avg_ms"] > r["v_mp_ms"]

def test_mb_higher_T_faster():
    m = 1.67e-27
    r300 = maxwell_boltzmann_speed(300, m, 1000)
    r600 = maxwell_boltzmann_speed(600, m, 1000)
    assert r600["v_rms_ms"] > r300["v_rms_ms"]

def test_mb_invalid():
    with pytest.raises(ValueError):
        maxwell_boltzmann_speed(0, 1e-26, 100)


# ── Bohr hydrogen ─────────────────────────────────────────────────────

def test_bohr_ground_state_energy():
    r = bohr_hydrogen(1)
    assert r["E_n_eV"] == pytest.approx(-13.606, abs=0.01)

def test_bohr_n2_energy():
    r = bohr_hydrogen(2)
    assert r["E_n_eV"] == pytest.approx(-13.606 / 4, abs=0.01)

def test_bohr_radius_n1():
    r = bohr_hydrogen(1)
    assert r["r_n_angstrom"] == pytest.approx(0.529, abs=0.001)

def test_bohr_energy_decreases_with_n():
    energies = [bohr_hydrogen(n)["E_n_eV"] for n in range(1, 6)]
    for i in range(len(energies)-1):
        assert energies[i] < energies[i+1]   # less negative -> higher energy

def test_bohr_invalid_n():
    with pytest.raises(ValueError):
        bohr_hydrogen(0)

def test_bohr_float_n_invalid():
    with pytest.raises((ValueError, TypeError)):
        bohr_hydrogen(1.5)


# ── hydrogen spectral lines ───────────────────────────────────────────

def test_lyman_alpha():
    r = hydrogen_spectral_line(2, 1)
    assert r["lambda_nm"] == pytest.approx(121.6, abs=0.5)

def test_balmer_alpha_656():
    r = hydrogen_spectral_line(3, 2)
    assert r["lambda_nm"] == pytest.approx(656.3, abs=1.0)

def test_balmer_visible():
    r = hydrogen_spectral_line(3, 2)
    assert r["visible"] is True

def test_lyman_uv():
    r = hydrogen_spectral_line(2, 1)
    assert r["visible"] is False

def test_spectral_line_energy_conservation():
    r = hydrogen_spectral_line(3, 2)
    E_diff = abs(bohr_hydrogen(3)["E_n_eV"] - bohr_hydrogen(2)["E_n_eV"])
    assert r["photon_energy_eV"] == pytest.approx(E_diff, rel=1e-4)

def test_invalid_spectral_transition():
    with pytest.raises(ValueError):
        hydrogen_spectral_line(1, 2)   # lower > upper is invalid


# ── molecular binding ─────────────────────────────────────────────────

def test_h2_binding_energy_positive():
    hm = hydrogen_molecule_binding()
    assert hm["binding_energy_eV"] > 0

def test_h2_dissociation_lt_binding():
    hm = hydrogen_molecule_binding()
    assert hm["dissociation_energy_eV"] < hm["binding_energy_eV"]

def test_h2_bond_length_reasonable():
    hm = hydrogen_molecule_binding()
    assert 0.5 < hm["bond_length_angstrom"] < 1.5


# ── London dispersion ─────────────────────────────────────────────────

def test_london_r6_dependence():
    U1 = london_dispersion(1.5, 1.5, 15.0, 15.0, 3.0)
    U2 = london_dispersion(1.5, 1.5, 15.0, 15.0, 6.0)  # 2x distance
    # |U| scales as r^-6: (6/3)^6 = 64x weaker
    assert abs(U1["U_eV"]) > abs(U2["U_eV"]) * 30

def test_london_energy_negative():
    U = london_dispersion(1.5, 1.5, 15.0, 15.0, 4.0)
    assert U["U_eV"] < 0   # attractive

def test_london_invalid():
    with pytest.raises(ValueError):
        london_dispersion(0, 1.5, 15.0, 15.0, 3.0)


# ── Planck spectrum ───────────────────────────────────────────────────

def test_planck_positive():
    r = planck_spectrum(5778, 550)
    assert r["B_W_per_m2_sr_nm"] > 0

def test_planck_wien_displacement():
    r = planck_spectrum(5778, 550)
    assert 480 < r["lambda_peak_nm"] < 520   # sun peak ~500 nm

def test_planck_hotter_brighter():
    r3000 = planck_spectrum(3000, 550)
    r6000 = planck_spectrum(6000, 550)
    assert r6000["B_W_per_m2_sr_nm"] > r3000["B_W_per_m2_sr_nm"]

def test_planck_invalid_temp():
    with pytest.raises(ValueError):
        planck_spectrum(0, 550)


# ── selection rules ───────────────────────────────────────────────────

def test_selection_p_to_s_allowed():
    r = selection_rule_check(1, 0, 0)   # 2p -> 1s
    assert r["E1_allowed"] is True

def test_selection_s_to_s_forbidden():
    r = selection_rule_check(0, 0, 0)   # 2s -> 1s
    assert r["E1_allowed"] is False

def test_selection_d_to_p_allowed():
    r = selection_rule_check(2, 1, 0)   # 3d -> 2p
    assert r["E1_allowed"] is True


# ── SymPy 5 ──────────────────────────────────────────────────────────

def test_feynman_sympy_5_count():
    eqs = feynman_atomic_sympy_5()
    assert len(eqs) == 5

def test_feynman_sympy_5_types():
    for k, eq in feynman_atomic_sympy_5().items():
        assert isinstance(eq, sp.Basic), f"{k} not SymPy"
