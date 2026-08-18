import numpy as np
import sympy as sp
import pytest

from griffiths.conservation_laws import (
    maxwell_stress_tensor, verify_stress_tensor_symmetric, momentum_density,
    radiation_pressure,
)
from griffiths.retarded_potentials import (
    retarded_time_symbolic, retarded_potential_formula,
    lienard_wiechert_potentials, coulomb_potential,
)
from griffiths.radiation import (
    abraham_lorentz_force, verify_abraham_lorentz_does_work_matching_larmor,
    magnetic_dipole_average_power, electric_vs_magnetic_dipole_power_ratio,
    dipole_average_power,
)


# ── Ch.8 conservation laws ──────────────────────────────────────────────

def test_stress_tensor_symmetric_symbolic():
    Ex, Ey, Ez, Bx, By, Bz = sp.symbols('Ex Ey Ez Bx By Bz', real=True)
    assert verify_stress_tensor_symmetric([Ex, Ey, Ez], [Bx, By, Bz])


def test_stress_tensor_rejects_wrong_shape():
    with pytest.raises(ValueError):
        maxwell_stress_tensor([1, 2], [1, 2, 3])


def test_stress_tensor_pure_E_diagonal_sign():
    # pure E along x: T_xx should be +eps0*Ex^2/2, T_yy = T_zz = -eps0*Ex^2/2
    eps0 = sp.Symbol('epsilon_0', positive=True)
    Ex = sp.Symbol('Ex', positive=True)
    T = maxwell_stress_tensor([Ex, 0, 0], [0, 0, 0])
    assert sp.simplify(T[0, 0] - eps0 * Ex**2 / 2) == 0
    assert sp.simplify(T[1, 1] + eps0 * Ex**2 / 2) == 0
    assert sp.simplify(T[2, 2] + eps0 * Ex**2 / 2) == 0


def test_momentum_density_direction():
    # E along x, B along y -> E x B along z -> g along z
    Ex = sp.Symbol('Ex', positive=True)
    By = sp.Symbol('By', positive=True)
    g = momentum_density([Ex, 0, 0], [0, By, 0])
    assert g[0] == 0 and g[1] == 0
    assert sp.simplify(g[2]) != 0


def test_radiation_pressure_reflecting_is_double_absorbing():
    I = 1361.0
    P_abs = radiation_pressure(I, absorbed=True)
    P_refl = radiation_pressure(I, absorbed=False)
    assert P_refl == pytest.approx(2 * P_abs)


def test_radiation_pressure_matches_known_solar_value():
    # known real-world value: solar radiation pressure at Earth ~4.5 uPa (absorbing)
    P = radiation_pressure(1361.0, absorbed=True)
    assert P == pytest.approx(4.5e-6, rel=0.05)


def test_radiation_pressure_rejects_negative_intensity():
    with pytest.raises(ValueError):
        radiation_pressure(-1.0)


# ── Ch.10 retarded potentials ────────────────────────────────────────────

def test_retarded_time_formula_structure():
    eq = retarded_time_symbolic()
    assert isinstance(eq, sp.Eq)


def test_retarded_potential_formula_is_integral_eq():
    eq = retarded_potential_formula()
    assert isinstance(eq, sp.Eq)
    assert eq.rhs.has(sp.Integral)


def test_lienard_wiechert_static_charge_matches_coulomb_exactly():
    q = 1e-9
    r_field = np.array([1.0, 0.0, 0.0])

    def w_static(t):
        return np.array([0.0, 0.0, 0.0])

    result = lienard_wiechert_potentials(q, w_static, r_field, t_eval=0.0)
    V_coulomb = coulomb_potential(q, 1.0)
    assert result["V"] == pytest.approx(V_coulomb, rel=1e-6)
    np.testing.assert_allclose(result["A"], [0.0, 0.0, 0.0], atol=1e-15)


def test_lienard_wiechert_retarded_time_matches_analytic_for_constant_velocity():
    # for a charge moving at constant v away from the field point, the retarded
    # time has a closed form: solve |r - v*t_r| = c*(t_eval - t_r) for 1D motion
    q = 1e-9
    v0 = 1e6
    r_field = np.array([1.0, 0.0, 0.0])
    c_num = 299792458.0

    def w_moving(t):
        return np.array([v0 * t, 0.0, 0.0])

    result = lienard_wiechert_potentials(q, w_moving, r_field, t_eval=0.0)
    # analytic: |1 - v0*t_r| = c*(0-t_r) = -c*t_r (t_r<0, so -c*t_r>0)
    # for small |v0*t_r| << 1: t_r ~= -1/(c - v0)  (charge approaching from behind... )
    # solve exactly: 1 - v0*t_r = -c*t_r  (assuming 1-v0*t_r>0 near t_r~0)
    #  -> 1 = v0*t_r - c*t_r = t_r*(v0-c)  -> t_r = 1/(v0-c) = -1/(c-v0)
    t_r_analytic = 1.0 / (v0 - c_num)
    assert result["t_r"] == pytest.approx(t_r_analytic, rel=1e-6)


def test_lienard_wiechert_moving_charge_has_nonzero_A():
    q = 1e-9
    v0 = 1e6
    r_field = np.array([1.0, 0.0, 0.0])

    def w_moving(t):
        return np.array([v0 * t, 0.0, 0.0])

    result = lienard_wiechert_potentials(q, w_moving, r_field, t_eval=0.0)
    assert abs(result["A"][0]) > 1e-15


def test_coulomb_potential_rejects_nonpositive_r():
    with pytest.raises(ValueError):
        coulomb_potential(1e-9, 0.0)


def test_lienard_wiechert_rejects_small_n_search():
    with pytest.raises(ValueError):
        lienard_wiechert_potentials(1e-9, lambda t: np.zeros(3), np.array([1.0, 0, 0]),
                                     t_eval=0.0, n_search=5)


# ── Ch.11 radiation reaction and magnetic dipole ─────────────────────────

def test_abraham_lorentz_force_matches_larmor_average_power():
    q, a0, omega, t = sp.symbols('q a0 omega t', positive=True)
    result = verify_abraham_lorentz_does_work_matching_larmor(q, a0, omega, t)
    assert result["match"] is True


def test_abraham_lorentz_force_scales_with_jerk():
    q = sp.Symbol('q', positive=True)
    jerk = sp.Symbol('jerk')
    F = abraham_lorentz_force(q, jerk)
    assert F.has(jerk)
    assert sp.simplify(F.subs(jerk, 0)) == 0


def test_magnetic_dipole_power_positive_for_real_inputs():
    m0, omega = sp.symbols('m0 omega', positive=True)
    P = magnetic_dipole_average_power(m0, omega)
    P_num = float(P.subs({m0: 1.0, omega: 1e9,
                           sp.Symbol('mu_0', positive=True): 4 * np.pi * 1e-7,
                           sp.Symbol('c', positive=True): 3e8}))
    assert P_num > 0


def test_electric_vs_magnetic_dipole_ratio_is_c_squared():
    ratio = electric_vs_magnetic_dipole_power_ratio()
    c = sp.Symbol('c', positive=True)
    assert sp.simplify(ratio - c**2) == 0
