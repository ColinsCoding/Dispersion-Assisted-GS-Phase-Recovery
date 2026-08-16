"""Tests for dgs/special_relativity.py."""
import numpy as np
import sympy as sp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.special_relativity import (
    lorentz_factor, lorentz_transform, time_dilation, length_contraction,
    four_vector_boost, relativistic_energy, energy_momentum_relation,
    relativistic_doppler, velocity_addition,
    phase_velocity, group_velocity, gvd_from_dispersion,
    smf28_dispersion, sr_sympy_5, C_SI,
    derive_lorentz_transformation_symbolic, verify_lorentz_transformation_derivation,
    verify_galilean_limit_recovered,
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


def test_derive_lorentz_transformation_symbolic():
    """gamma must be uniquely determined by self-consistency (forward
    transform composed with the v->-v inverse must be the identity), and
    match 1/sqrt(1-v^2/c^2) -- not assumed, solved."""
    gamma_solved, x_prime, t_prime, invariant_residual = derive_lorentz_transformation_symbolic()
    assert invariant_residual == 0
    for v_val, c_val in [(0.1, 1.0), (0.5, 1.0), (0.9, 1.0)]:
        v_s, c_s = sp.symbols("v c", positive=True)
        expected = 1 / sp.sqrt(1 - v_s**2 / c_s**2)
        diff = float((gamma_solved - expected).subs({v_s: v_val, c_s: c_val}))
        assert abs(diff) < 1e-9


def test_verify_lorentz_transformation_derivation():
    assert verify_lorentz_transformation_derivation() is True


def test_verify_galilean_limit_recovered():
    """As c -> infinity, the relativistic transform must reduce EXACTLY
    to Feynman's Joe/Moe Galilean transformation: x'=x-vt, t'=t."""
    assert verify_galilean_limit_recovered() is True


def test_derived_transform_matches_existing_lorentz_transform():
    """The derivation's closed-form x', t' must match the module's own
    (already-implemented, already-used-elsewhere) lorentz_transform()
    numerically, at a real relativistic speed -- not just agree with
    itself."""
    _, x_prime_sym, t_prime_sym, _ = derive_lorentz_transformation_symbolic()
    x_s, t_s, v_s, c_s = sp.symbols("x t v c", positive=True)

    x_val, t_val, v_val = 1000.0, 2.0, 0.6 * C_SI
    result = lorentz_transform(x=x_val, t=t_val, v=v_val)
    x_prime_num = float(x_prime_sym.subs({x_s: x_val, t_s: t_val, v_s: v_val, c_s: C_SI}))
    t_prime_num = float(t_prime_sym.subs({x_s: x_val, t_s: t_val, v_s: v_val, c_s: C_SI}))

    assert abs(result["x_prime"] - x_prime_num) < 1e-6
    assert abs(result["t_prime"] - t_prime_num) < 1e-15
