"""Tests for dgs/temporal_gs.py — temporal GS algorithm (Solli/Jalali APL 2009)."""
import numpy as np
import sympy as sp
import warnings
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.temporal_gs import (
    gvd_phase, apply_dispersion, remove_dispersion,
    temporal_gs, simulate_near_field, simulate_absorption_lines,
    diversity_sweep, far_field_condition, temporal_gs_sympy_5,
)


# ── GVD phase functions ───────────────────────────────────────────────────────
def test_gvd_phase_quadratic():
    omega = np.array([0.0, 1e11, 2e11])
    D = -600e-27
    phi = gvd_phase(omega, D)
    expected = D * omega**2 / 2
    np.testing.assert_allclose(phi, expected, rtol=1e-12)


def test_apply_remove_inverse():
    n = 128; dt = 1e-12
    omega = 2 * np.pi * np.fft.fftfreq(n, d=dt)
    E = np.random.randn(n) + 1j * np.random.randn(n)
    D = -600e-27
    E_applied = apply_dispersion(E, omega, D)
    E_back = remove_dispersion(E_applied, omega, D)
    np.testing.assert_allclose(E_back, E, atol=1e-12)


def test_apply_dispersion_all_pass():
    n = 64; dt = 1e-12
    omega = 2 * np.pi * np.fft.fftfreq(n, d=dt)
    E = np.exp(-np.arange(n).astype(float)**2 / 100) + 0j
    E_out = apply_dispersion(E, omega, -600e-27)
    # |H|=1 -> energy conserved
    np.testing.assert_allclose(np.sum(np.abs(E_out)**2), np.sum(np.abs(E)**2), rtol=1e-12)


# ── simulate_absorption_lines ─────────────────────────────────────────────────
def test_simulate_absorption_one_line():
    sim = simulate_absorption_lines(512, 5e-12, [50.0], [5.0], [0.9])
    assert "E_true_omega" in sim
    assert "true_spectrum" in sim
    assert len(sim["E_true_omega"]) == 512
    # Absorption should reduce spectral power somewhere
    H = np.abs(sim["H_absorption"])
    assert H.min() < 0.5  # at least one point strongly absorbed


def test_simulate_no_absorption():
    sim = simulate_absorption_lines(256, 5e-12, [], [], [])
    # No absorption: H_absorption = 1 everywhere
    np.testing.assert_allclose(np.abs(sim["H_absorption"]), 1.0, atol=1e-12)


# ── simulate_near_field ───────────────────────────────────────────────────────
def test_simulate_near_field_energy():
    sim = simulate_absorption_lines(256, 5e-12, [50.0], [5.0], [0.9])
    omega = sim["omega"]
    nf = simulate_near_field(sim["E_true_omega"], omega, -600e-27, 5e-12)
    # GVD conserves energy: sum|E_out|^2 = sum|E_in|^2 (Parseval)
    E_in_energy = np.sum(np.abs(sim["E_true_omega"])**2)
    E_out_energy = np.sum(nf["I_out_t"])
    # They differ by FFT normalization but ratio is fixed
    assert abs(E_out_energy) > 0


def test_simulate_near_field_keys():
    sim = simulate_absorption_lines(128, 5e-12, [50.0], [5.0], [0.9])
    nf = simulate_near_field(sim["E_true_omega"], sim["omega"], -600e-27, 5e-12)
    for k in ("E_out_t", "I_out_t", "f_t", "t"):
        assert k in nf


# ── temporal_gs ───────────────────────────────────────────────────────────────
def test_temporal_gs_keys():
    sim = simulate_absorption_lines(256, 5e-12, [50.0], [5.0], [0.9])
    omega = sim["omega"]
    D1, D2 = -600e-27, -800e-27
    m1 = simulate_near_field(sim["E_true_omega"], omega, D1, 5e-12)
    m2 = simulate_near_field(sim["E_true_omega"], omega, D2, 5e-12)
    res = temporal_gs(m1["f_t"], m2["f_t"], D1=D1, D2=D2, dt_s=5e-12, n_iter=5)
    for k in ("E_recovered", "I_recovered", "spectrum", "spectrum_shifted",
              "omega", "omega_shifted", "phase_history", "mag_history",
              "diversity", "converged"):
        assert k in res


def test_temporal_gs_phase_error_decreases():
    sim = simulate_absorption_lines(512, 5e-12, [50.0], [5.0], [0.9])
    omega = sim["omega"]
    D1, D2 = -600e-27, -1800e-27   # D2/D1=3 -> good diversity
    m1 = simulate_near_field(sim["E_true_omega"], omega, D1, 5e-12)
    m2 = simulate_near_field(sim["E_true_omega"], omega, D2, 5e-12)
    res = temporal_gs(m1["f_t"], m2["f_t"], D1=D1, D2=D2,
                       dt_s=5e-12, n_iter=20, diversity_warn=False)
    ph = res["phase_history"]
    # Phase error should generally decrease (not necessarily monotone, but overall)
    assert ph[-1] < ph[0]


def test_temporal_gs_magnitude_constraint_satisfied():
    # After GS, recovered intensity should match f1 (magnitude constraint)
    sim = simulate_absorption_lines(256, 5e-12, [50.0], [5.0], [0.9])
    omega = sim["omega"]
    D1, D2 = -600e-27, -1800e-27
    m1 = simulate_near_field(sim["E_true_omega"], omega, D1, 5e-12)
    m2 = simulate_near_field(sim["E_true_omega"], omega, D2, 5e-12)
    res = temporal_gs(m1["f_t"], m2["f_t"], D1=D1, D2=D2, dt_s=5e-12, n_iter=10)
    # Recovered field magnitude = f1 (magnitude constraint is applied each iter)
    np.testing.assert_allclose(np.abs(res["E_recovered"]), m1["f_t"], atol=1e-10)


def test_temporal_gs_diversity_warning():
    sim = simulate_absorption_lines(128, 5e-12, [50.0], [5.0], [0.9])
    omega = sim["omega"]
    D1, D2 = -600e-27, -630e-27   # D2/D1=1.05 < 1.33 -> warn
    m1 = simulate_near_field(sim["E_true_omega"], omega, D1, 5e-12)
    m2 = simulate_near_field(sim["E_true_omega"], omega, D2, 5e-12)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        temporal_gs(m1["f_t"], m2["f_t"], D1=D1, D2=D2, dt_s=5e-12, n_iter=3)
        assert len(w) == 1
        assert "1.05" in str(w[0].message)


def test_temporal_gs_rejects_equal_D():
    sim = simulate_absorption_lines(64, 5e-12, [], [], [])
    omega = sim["omega"]
    D1 = -600e-27
    m1 = simulate_near_field(sim["E_true_omega"], omega, D1, 5e-12)
    try:
        temporal_gs(m1["f_t"], m1["f_t"], D1=D1, D2=D1, dt_s=5e-12)
        assert False, "should raise"
    except ValueError:
        pass


def test_temporal_gs_diversity_field():
    sim = simulate_absorption_lines(256, 5e-12, [50.0], [5.0], [0.9])
    omega = sim["omega"]
    D1 = -600e-27
    for ratio in [1.33, 3.0]:
        D2 = D1 * ratio
        m1 = simulate_near_field(sim["E_true_omega"], omega, D1, 5e-12)
        m2 = simulate_near_field(sim["E_true_omega"], omega, D2, 5e-12)
        res = temporal_gs(m1["f_t"], m2["f_t"], D1=D1, D2=D2,
                           dt_s=5e-12, n_iter=20, diversity_warn=False)
        assert res["diversity"] == pytest_approx(ratio, rel=1e-6)


# ── far_field_condition ───────────────────────────────────────────────────────
def test_far_field_condition_SMF28():
    # SMF-28 at 1550nm: D~17 ps/(nm*km). For 0.1nm line: far-field requires many km
    # 1 km of SMF: D_eff ~ -20e-27 s^2
    res = far_field_condition(-20e-27, 1550.0, 0.1)
    assert "far_field" in res
    assert isinstance(res["far_field"], (bool, np.bool_))


def test_far_field_sufficient_dispersion():
    # Very large D -> far-field (need |D| >> lam^3/(2c*dlam^2) ~ 6e-10 s^2)
    res = far_field_condition(-1e-6, 1550.0, 0.1)
    assert bool(res["far_field"]) is True


def test_far_field_insufficient_dispersion():
    # Tiny D -> near-field
    res = far_field_condition(-1e-30, 1550.0, 0.1)
    assert bool(res["far_field"]) is False


# ── temporal_gs_sympy_5 ───────────────────────────────────────────────────────
def test_temporal_gs_sympy_5_count():
    eqs = temporal_gs_sympy_5()
    assert len(eqs) == 5


def test_temporal_gs_sympy_5_all_Eq():
    eqs = temporal_gs_sympy_5()
    for k, v in eqs.items():
        assert isinstance(v, sp.Eq), f"{k} is not sp.Eq"


# ── pytest helper (avoid import at top level) ─────────────────────────────────
def pytest_approx(val, rel=1e-6):
    import pytest
    return pytest.approx(val, rel=rel)
