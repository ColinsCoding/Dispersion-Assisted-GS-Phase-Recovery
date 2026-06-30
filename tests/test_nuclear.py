"""Tests for dgs/nuclear.py"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.nuclear import (
    bethe_weizsacker, binding_energy_per_nucleon,
    half_life_to_decay_constant, activity, carbon_dating_age,
    neutron_cross_section_estimate, moderation_ratio, fission_q_value,
    bethe_bloch_stopping, dose_gray, dose_sievert,
    irradiation_energy_deposited, Q_E, N_AVOGADRO,
)


def test_fe56_peak():
    # Fe-56 should have B/A near 8.8 MeV/nucleon
    bpa = binding_energy_per_nucleon(26, 56)
    assert 8.5 < bpa < 9.1, f"Fe-56 B/A={bpa}"


def test_heavier_less_bound_than_fe():
    bpa_fe = binding_energy_per_nucleon(26, 56)
    bpa_u  = binding_energy_per_nucleon(92, 235)
    assert bpa_u < bpa_fe, "U-235 should be less bound per nucleon than Fe-56"


def test_decay_constant():
    # C-14 t_half = 5730 yr; lambda = ln2/5730 per year
    t_half_yr = 5730.0
    lam = half_life_to_decay_constant(t_half_yr)
    assert abs(lam - np.log(2)/5730) < 1e-15


def test_activity_half_life():
    # After one half-life, activity should be halved
    N0, t_half = 1e20, 3600.0   # 1 hour half-life
    A0 = activity(N0, t_half, 0)
    A1 = activity(N0, t_half, t_half)
    assert abs(A1/A0 - 0.5) < 1e-10


def test_carbon_dating_half_life():
    # 50% C-14 remaining = exactly one half-life = 5730 years
    age = carbon_dating_age(0.5)
    assert abs(age - 5730) < 1, f"age={age}"


def test_hydrogen_perfect_moderation():
    # H (A=1) retains 0% of neutron energy per collision
    assert moderation_ratio(1) == 0.0


def test_uranium_poor_moderator():
    # U-238 retains >98% of energy per collision
    assert moderation_ratio(238) > 0.98


def test_fission_q_positive():
    # U-236 -> Ba-141 + Kr-92 should release energy (Q > 0)
    Q = fission_q_value(92, 236, 56, 141, 36, 92)
    assert Q > 100, f"Q={Q} MeV, expected ~150-200 MeV"


def test_bethe_bloch_positive():
    sp = bethe_bloch_stopping(z=2, beta=0.05, Z_target=7, A_target=14)
    assert sp > 0, f"stopping power={sp}"


def test_dose_gray():
    assert abs(dose_gray(1.0, 1.0) - 1.0) < 1e-15
    assert abs(dose_gray(10.0, 2.0) - 5.0) < 1e-15


def test_dose_sievert_alpha():
    # Alpha quality factor = 20
    assert abs(dose_sievert(1.0, 20) - 20.0) < 1e-15


def test_irradiation_energy():
    # 3 kGy * 1 kg = 3000 J
    r = irradiation_energy_deposited(dose_kGy=3.0, mass_kg=1.0)
    assert abs(r['energy_J'] - 3000.0) < 1e-8


def test_u235_coal_equiv():
    # 1 kg U-235 fission ~ 2000 tons coal
    Q_MeV = fission_q_value(92, 236, 56, 141, 36, 92)
    N_atoms = (1000.0 / 235.0) * N_AVOGADRO
    E_J = N_atoms * Q_MeV * 1e6 * Q_E
    coal_equiv = E_J / 30e9   # 30 GJ / ton
    assert 1000 < coal_equiv < 5000, f"coal_equiv={coal_equiv}"
