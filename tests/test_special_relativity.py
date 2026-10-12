"""Tests for dgs/special_relativity.py."""
import numpy as np
import sympy as sp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.special_relativity import (
    lorentz_factor, lorentz_transform, time_dilation, length_contraction,
    relativity_of_simultaneity, four_vector_boost, relativistic_energy,
    energy_momentum_relation, relativistic_doppler, velocity_addition,
    phase_velocity, group_velocity, gvd_from_dispersion,
    smf28_dispersion, sr_sympy_5, C_SI,
)


# ── Lorentz factor ────────────────────────────────────────────────────────────

def test_gamma_at_rest():
    lf = lorentz_factor(0.0)
    assert abs(lf["gamma"] - 1.0) < 1e-10


def test_gamma_at_09c():
    lf = lorentz_factor(0.9 * C_SI)
    assert abs(lf["gamma"] - 1.0/np.sqrt(1 - 0.81)) < 1e-8


def test_gamma_superluminal_raises():
    try:
        lorentz_factor(1.1 * C_SI)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ── Kinematics ────────────────────────────────────────────────────────────────

def test_lorentz_transform_at_rest():
    res = lorentz_transform(x=1.0, t=0.0, v=0.0)
    assert abs(res["x_prime"] - 1.0) < 1e-10
    assert abs(res["t_prime"] - 0.0) < 1e-10


def test_time_dilation_gamma():
    res = time_dilation(tau0=1.0, v=0.6 * C_SI)
    g = 1.0 / np.sqrt(1 - 0.36)
    assert abs(res["t_lab"] - g) < 1e-8


def test_length_contraction_gamma():
    res = length_contraction(L0=1.0, v=0.6 * C_SI)
    g = 1.0 / np.sqrt(1 - 0.36)
    assert abs(res["L_lab"] - 1.0/g) < 1e-8


def test_simultaneity_agrees_at_rest():
    # v=0: no boost, both events stay simultaneous
    res = relativity_of_simultaneity(x1=-500.0, x2=500.0, t=0.0, v=0.0)
    assert abs(res["dt_prime"]) < 1e-10
    assert res["simultaneous_in_S_prime"]


def test_simultaneity_breaks_under_boost():
    # classic Einstein train: same-time strikes at x=-500 and x=+500 are
    # NOT simultaneous to an observer moving at 0.6c
    res = relativity_of_simultaneity(x1=-500.0, x2=500.0, t=0.0, v=0.6 * C_SI)
    assert not res["simultaneous_in_S_prime"]
    assert abs(res["dt_prime"]) > 1e-9


def test_simultaneity_matches_closed_form():
    res = relativity_of_simultaneity(x1=-500.0, x2=500.0, t=0.0, v=0.6 * C_SI)
    assert abs(res["dt_prime"] - res["dt_prime_formula_check"]) < 1e-15


def test_simultaneity_reverses_sign_with_boost_direction():
    # the SAME two events order differently depending on which way the
    # observer moves -- the defining feature of relativity of simultaneity
    pos = relativity_of_simultaneity(x1=-500.0, x2=500.0, t=0.0, v=0.6 * C_SI)
    neg = relativity_of_simultaneity(x1=-500.0, x2=500.0, t=0.0, v=-0.6 * C_SI)
    assert pos["dt_prime"] * neg["dt_prime"] < 0


def test_simultaneity_identical_position_is_frame_independent():
    # two events at the SAME place are simultaneous in every frame
    res = relativity_of_simultaneity(x1=100.0, x2=100.0, t=0.0, v=0.6 * C_SI)
    assert abs(res["dt_prime"]) < 1e-10


def test_four_vector_invariant_preserved():
    res = four_vector_boost([10.0, 5.0, 0.0, 0.0], v=0.5 * C_SI)
    assert abs(res["invariant_prime"] - res["invariant_orig"]) < 1e-6


# ── Energy / momentum ─────────────────────────────────────────────────────────

def test_energy_rest_frame():
    me = 9.109e-31
    res = relativistic_energy(me, 0.0)
    assert abs(res["KE_J"]) < 1e-40   # no KE at rest
    assert abs(res["E_rest_J"] - me * C_SI**2) < 1e-20


def test_energy_momentum_consistency():
    me = 9.109e-31
    res = relativistic_energy(me, 0.5 * C_SI)
    assert res["energy_momentum_error"] < 1e-20


def test_energy_momentum_photon():
    # photon: m=0, E = pc
    p = 1e-27
    res = energy_momentum_relation(0.0, p)
    assert abs(res["E_J"] - p * C_SI) < 1e-30


# ── Doppler ───────────────────────────────────────────────────────────────────

def test_doppler_approaching_blueshift():
    res = relativistic_doppler(1e14, v=0.1 * C_SI, approaching=True)
    assert res["f_obs"] > res["f0"]


def test_doppler_receding_redshift():
    res = relativistic_doppler(1e14, v=0.1 * C_SI, approaching=False)
    assert res["f_obs"] < res["f0"]


def test_doppler_symmetry():
    fa = relativistic_doppler(1e14, 0.3 * C_SI, approaching=True)["f_obs"]
    fr = relativistic_doppler(1e14, 0.3 * C_SI, approaching=False)["f_obs"]
    assert abs(fa * fr - 1e14**2) < 1e16   # f_a * f_r = f0^2


# ── Velocity addition ─────────────────────────────────────────────────────────

def test_velocity_addition_subluminal():
    res = velocity_addition(0.9 * C_SI, 0.9 * C_SI)
    assert abs(res["u_ms"]) < C_SI


def test_velocity_addition_classical_limit():
    # At low speed, u ~ v1 + v2
    res = velocity_addition(100.0, 200.0)
    assert abs(res["u_ms"] - 300.0) < 0.01


def test_velocity_addition_c_plus_anything_is_c():
    # postulate 2: adding ANY subluminal speed to light still gives light --
    # this used to crash (beta hit exactly 1.0, lorentz_factor raised) since
    # gamma is genuinely infinite/undefined for a lightlike result, not an error
    res = velocity_addition(C_SI, 20.0)
    assert res["u_ms"] == C_SI
    assert res["gamma"] == float("inf")
    assert res["beta"] == 1.0


# ── Phase and group velocity ──────────────────────────────────────────────────

def test_phase_velocity_linear():
    omega = np.array([1.0, 2.0, 3.0])
    k     = np.array([0.5, 1.0, 1.5])
    res = phase_velocity(omega, k)
    np.testing.assert_allclose(res["v_p"], 2.0)


def test_group_velocity_linear_dispersion():
    # omega = 2*k -> v_g = d(omega)/dk = 2 everywhere
    k     = np.linspace(1, 10, 200)
    omega = 2 * k
    res = group_velocity(omega, k)
    # v_g = 1/(dk/domega) = 1/(1/2) = 2
    np.testing.assert_allclose(res["v_g_mean"], 2.0, rtol=1e-3)


def test_gvd_smf28_recovery():
    om0   = 2 * np.pi * 3e8 / 1550e-9
    omega = np.linspace(om0 - 2*np.pi*5e12, om0 + 2*np.pi*5e12, 4096)
    smf   = smf28_dispersion(omega)
    gvd   = gvd_from_dispersion(omega, smf["k"])
    # Should recover -22e-27 s^2/m within 1%
    assert abs(gvd["beta2_mean"] - smf["beta2_theory"]) / abs(smf["beta2_theory"]) < 0.01


def test_gvd_tsdft_note_present():
    om0   = 2 * np.pi * 3e8 / 1550e-9
    omega = np.linspace(om0 - 1e12, om0 + 1e12, 100)
    smf   = smf28_dispersion(omega)
    gvd   = gvd_from_dispersion(omega, smf["k"])
    assert "branch cuts" in gvd["tsdft_note"]


# ── SymPy ─────────────────────────────────────────────────────────────────────

def test_sympy_5_count():
    eqs = sr_sympy_5()
    assert len(eqs) == 5


def test_sympy_lorentz_factor_form():
    eqs = sr_sympy_5()
    eq = eqs["Lorentz_factor"]
    assert isinstance(eq, sp.Eq)
    syms = {str(s) for s in eq.rhs.free_symbols}
    assert "v" in syms and "c" in syms


def test_sympy_energy_momentum():
    eqs = sr_sympy_5()
    eq = eqs["Energy_momentum"]
    assert isinstance(eq, sp.Eq)
    # E^2 = (pc)^2 + (mc^2)^2 -> RHS has m, p, c
    syms = {str(s) for s in eq.rhs.free_symbols}
    assert "m" in syms
