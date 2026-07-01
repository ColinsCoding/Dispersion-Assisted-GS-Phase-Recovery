import numpy as np
import pytest
import sympy as sp
from dgs.cellular_biophysics import (
    nernst_potential, goldman_potential,
    hh_steady_state, hodgkin_huxley,
    optogenetic_current, membrane_rc,
    cellular_biophysics_sympy_5,
    ION_CONCENTRATIONS, HH_PARAMS,
)


# ── Nernst potential ──────────────────────────────────────────────────

def test_nernst_potassium_negative():
    r = nernst_potential("K")
    assert r["E_mV"] < -60   # K+ resting potential is negative

def test_nernst_sodium_positive():
    r = nernst_potential("Na")
    assert r["E_mV"] > 50   # Na+ drives depolarization

def test_nernst_custom_ion():
    r = nernst_potential(conc_in_mM=140, conc_out_mM=5, z=+1)
    assert r["E_mV"] < 0

def test_nernst_symmetric_is_zero():
    r = nernst_potential(conc_in_mM=100, conc_out_mM=100, z=+1)
    assert r["E_mV"] == pytest.approx(0.0, abs=0.01)

def test_nernst_invalid_ion():
    with pytest.raises(ValueError):
        nernst_potential("Xe")

def test_nernst_invalid_concentration():
    with pytest.raises(ValueError):
        nernst_potential(conc_in_mM=-1, conc_out_mM=100, z=1)

def test_nernst_zero_valence_invalid():
    with pytest.raises(ValueError):
        nernst_potential(conc_in_mM=100, conc_out_mM=10, z=0)


# ── Goldman potential ─────────────────────────────────────────────────

def test_goldman_resting_negative():
    g = goldman_potential()
    assert -80 < g["V_rest_mV"] < -50   # physiological range

def test_goldman_high_sodium_depolarizes():
    # increase Na+ permeability -> depolarize (less negative)
    g_rest = goldman_potential()
    g_depol = goldman_potential({"K": 1.0, "Na": 10.0, "Cl": 0.45})
    assert g_depol["V_rest_mV"] > g_rest["V_rest_mV"]

def test_goldman_only_potassium_equals_nernst():
    # if only K+ is permeable, Goldman = Nernst for K
    g = goldman_potential({"K": 1.0, "Na": 0.0, "Cl": 0.0})
    e_k = nernst_potential("K")
    assert abs(g["V_rest_mV"] - e_k["E_mV"]) < 1.0


# ── HH steady state ───────────────────────────────────────────────────

def test_hh_steady_state_gating_range():
    ss = hh_steady_state(-65.0)
    for key in ["m_inf", "h_inf", "n_inf"]:
        assert 0 <= ss[key] <= 1, f"{key} out of [0,1]"

def test_hh_steady_state_depolarized():
    ss_dep = hh_steady_state(0.0)   # depolarized
    ss_rest = hh_steady_state(-65.0)
    assert ss_dep["m_inf"] > ss_rest["m_inf"]   # m increases with depolarization
    assert ss_dep["h_inf"] < ss_rest["h_inf"]   # h decreases (inactivation)

def test_hh_open_probability_at_rest_low():
    ss = hh_steady_state(-65.0)
    assert ss["g_Na_frac"] < 0.1   # Na+ channels mostly closed at rest
    assert ss["g_K_frac"] < 0.2    # K+ channels also mostly closed


# ── Hodgkin-Huxley simulation ─────────────────────────────────────────

def test_hh_no_current_no_spike():
    r = hodgkin_huxley(0.0, t_end_ms=20.0)
    assert r["n_spikes"] == 0
    assert np.max(r["V_mV"]) < -50   # stays near rest

def test_hh_suprathreshold_generates_spike():
    r = hodgkin_huxley(10.0, t_end_ms=50.0)
    assert r["n_spikes"] >= 1

def test_hh_action_potential_peak():
    r = hodgkin_huxley(10.0, t_end_ms=50.0)
    peak = np.max(r["V_mV"])
    assert peak > 20.0   # must exceed +20 mV (typically reaches ~+40 mV)

def test_hh_repetitive_firing_at_high_current():
    r = hodgkin_huxley(20.0, t_end_ms=100.0)
    assert r["n_spikes"] >= 2   # multiple spikes

def test_hh_invalid_dt():
    with pytest.raises(ValueError):
        hodgkin_huxley(10.0, dt_ms=0)


# ── optogenetics ──────────────────────────────────────────────────────

def test_chr2_on_peak_wavelength():
    r = optogenetic_current(1.0, wavelength_nm=470, channel="ChR2")
    assert r["I_uA_cm2"] > 0
    assert r["activation"] > 0.5   # at 1 mW/mm^2 (> P50=0.5)

def test_chr2_off_peak_attenuated():
    r_on  = optogenetic_current(1.0, wavelength_nm=470, channel="ChR2")
    r_off = optogenetic_current(1.0, wavelength_nm=600, channel="ChR2")
    assert abs(r_off["I_uA_cm2"]) < abs(r_on["I_uA_cm2"])

def test_nphr_hyperpolarizes():
    r = optogenetic_current(1.0, wavelength_nm=580, channel="NpHR")
    assert r["I_uA_cm2"] < 0   # inhibitory current

def test_optogenetic_invalid_channel():
    with pytest.raises(ValueError):
        optogenetic_current(1.0, channel="RandomOpsin")

def test_optogenetic_zero_power():
    r = optogenetic_current(0.0, channel="ChR2")
    assert r["I_uA_cm2"] == pytest.approx(0.0, abs=1e-6)


# ── passive RC membrane ───────────────────────────────────────────────

def test_membrane_rc_tau_positive():
    rc = membrane_rc()
    assert rc["tau_ms"] > 0

def test_membrane_rc_steady_state():
    g = HH_PARAMS["g_L"]
    I = 1.0
    V_rest = HH_PARAMS["V_rest"]
    rc = membrane_rc(g_total_mS_cm2=g, I_ext_uA_cm2=I, V_rest_mV=V_rest)
    V_final = rc["V_mV"][-1]
    expected = V_rest + I / g
    assert V_final == pytest.approx(expected, abs=0.5)

def test_membrane_rc_zero_current_stays_at_rest():
    rc = membrane_rc(I_ext_uA_cm2=0.0)
    assert np.allclose(rc["V_mV"], rc["V_mV"][0], atol=1e-6)


# ── SymPy 5 ──────────────────────────────────────────────────────────

def test_cellular_sympy_5_count():
    eqs = cellular_biophysics_sympy_5()
    assert len(eqs) == 5

def test_cellular_sympy_5_types():
    for k, eq in cellular_biophysics_sympy_5().items():
        assert isinstance(eq, sp.Basic), f"{k} not SymPy"
