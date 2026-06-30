"""Tests for dgs/math_methods.py — Mathematical Methods of Physics."""
import numpy as np
import sympy as sp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.math_methods import (
    # Calculus
    taylor_series_sympy, partial_derivatives_sympy,
    double_integral_sympy, greens_theorem_check, calculus_sympy_5,
    # Linear algebra
    eigenvalue_decomposition, svd_analysis, gram_schmidt,
    matrix_exponential, linalg_sympy_5,
    # ODEs
    ode_second_order_sympy, harmonic_oscillator_ivp,
    frobenius_bessel_sympy, ode_sympy_5,
    # Mechanics
    lagrangian_particle_sympy, phase_space_trajectory, mechanics_sympy_5,
    # Complex analysis
    cauchy_riemann_check, residue_theorem,
    cauchy_integral_formula, partial_fractions_complex,
    complex_analysis_sympy_5,
    # Number theory
    modular_arithmetic, softmax_and_extrema,
    chinese_remainder_theorem, number_theory_sympy_5,
    all_math_methods_sympy,
)


# ── Calculus ─────────────────────────────────────────────────────────────────
def test_taylor_exp():
    x = sp.Symbol("x")
    res = taylor_series_sympy(sp.exp(x), x, 0, n_terms=4)
    assert "series" in res
    assert "poly" in res


def test_partial_derivatives_gradient():
    x, y = sp.symbols("x y")
    f = x**2 + y**2
    res = partial_derivatives_sympy(f, [x, y])
    assert sp.simplify(res["gradient"][0] - 2*x) == 0
    assert sp.simplify(res["gradient"][1] - 2*y) == 0
    assert sp.simplify(res["laplacian"] - 4) == 0


def test_partial_derivatives_hessian():
    x, y = sp.symbols("x y")
    f = x**2 * y + y**3
    res = partial_derivatives_sympy(f, [x, y])
    # d^2f/dx^2 = 2y, d^2f/dy^2 = 6y
    assert sp.simplify(res["hessian"][0, 0] - 2*y) == 0
    assert sp.simplify(res["hessian"][1, 1] - 6*y) == 0


def test_double_integral():
    x, y = sp.symbols("x y")
    # int_0^1 int_0^1 x*y dx dy = 1/4
    res = double_integral_sympy(x*y, x, (0, 1), y, (0, 1))
    assert sp.simplify(res["result"] - sp.Rational(1, 4)) == 0


def test_greens_theorem_exact():
    x, y = sp.symbols("x y")
    # For P = x, Q = y: curl = dQ/dx - dP/dy = 0 (exact)
    res = greens_theorem_check(x, y, x, y)
    assert res["exact"] is True
    assert sp.simplify(res["curl"]) == 0


def test_greens_theorem_non_exact():
    x, y = sp.symbols("x y")
    # P = -y, Q = x: curl = 1+1 = 2 (rotation, not exact)
    res = greens_theorem_check(-y, x, x, y)
    assert res["exact"] is False
    assert sp.simplify(res["curl"] - 2) == 0


def test_calculus_sympy_5():
    eqs = calculus_sympy_5()
    assert len(eqs) == 5
    for k, v in eqs.items():
        assert isinstance(v, sp.Eq)


# ── Linear algebra ────────────────────────────────────────────────────────────
def test_eigenvalue_symmetric():
    A = np.array([[4., 2.], [2., 3.]])
    res = eigenvalue_decomposition(A)
    assert res["hermitian"] is True
    # Trace = sum eigenvalues
    assert abs(np.sum(res["eigenvalues"]) - np.trace(A)) < 1e-8
    # Det = product eigenvalues
    assert abs(np.prod(res["eigenvalues"]) - np.linalg.det(A)) < 1e-8
    assert res["trace_check"]


def test_eigenvalue_non_symmetric():
    A = np.array([[1., 2.], [0., 3.]])
    res = eigenvalue_decomposition(A)
    assert res["hermitian"] is False
    assert set(np.round(np.real(res["eigenvalues"]), 6)) == {1.0, 3.0}


def test_svd_reconstruction():
    A = np.random.randn(4, 3)
    res = svd_analysis(A)
    assert res["recon_error"] < 1e-10
    assert res["rank"] == 3


def test_svd_rank_deficient():
    A = np.zeros((4, 3)); A[0, 0] = 1.0; A[1, 1] = 2.0
    res = svd_analysis(A)
    assert res["rank"] == 2


def test_gram_schmidt_orthonormal():
    V = np.array([[1., 1.], [1., 0.], [0., 1.]])
    Q = gram_schmidt(V)
    # Q^T Q = I
    np.testing.assert_allclose(Q.T @ Q, np.eye(2), atol=1e-10)
    # Column norms = 1
    np.testing.assert_allclose(np.linalg.norm(Q, axis=0), np.ones(2), atol=1e-10)


def test_matrix_exponential_identity():
    # exp(0) = I
    A = np.zeros((3, 3))
    expm = matrix_exponential(A, t=1.0)
    np.testing.assert_allclose(expm, np.eye(3), atol=1e-10)


def test_matrix_exponential_diagonal():
    A = np.diag([1.0, 2.0, 3.0])
    expm = matrix_exponential(A, t=1.0)
    expected = np.diag(np.exp([1., 2., 3.]))
    np.testing.assert_allclose(expm, expected, atol=1e-10)


def test_linalg_sympy_5():
    eqs = linalg_sympy_5()
    assert len(eqs) == 5
    for k, v in eqs.items():
        assert isinstance(v, sp.Basic)


# ── ODEs ──────────────────────────────────────────────────────────────────────
def test_ode_harmonic_oscillator_underdamped():
    res = harmonic_oscillator_ivp(10.0, 0.1, (0, 5))
    assert res["regime"] == "underdamped"
    # Energy should decay (not conserved with damping)
    assert res["E"][-1] < res["E"][0]


def test_ode_harmonic_oscillator_critical():
    res = harmonic_oscillator_ivp(10.0, 1.0, (0, 5))
    assert res["regime"] == "critically_damped"
    assert res["x"][0] == pytest_approx(1.0)


def test_ode_harmonic_oscillator_undamped_energy():
    # zeta=0: no damping -> use underdamped formula with zeta tiny
    res = harmonic_oscillator_ivp(1.0, 1e-10, (0, 2*np.pi), x0=1.0, v0=0.0)
    # Energy should be approximately conserved
    assert abs(res["E"][-1] - res["E"][0]) / res["E"][0] < 0.01


def test_ode_q_factor():
    res = harmonic_oscillator_ivp(10.0, 0.5, (0, 5))
    assert abs(res["Q_factor"] - 1.0) < 1e-10


def test_frobenius_bessel_nu0():
    res = frobenius_bessel_sympy(0)
    # Indicial roots should be 0, 0 (double root for nu=0)
    assert 0 in res["indicial_roots"]
    assert "J_nu_series" in res


def test_frobenius_bessel_nu1():
    res = frobenius_bessel_sympy(1)
    # Indicial roots: ±1
    roots = set(res["indicial_roots"])
    assert sp.Integer(1) in roots or sp.Integer(-1) in roots


def test_ode_sympy_5():
    eqs = ode_sympy_5()
    assert len(eqs) == 5
    for k, v in eqs.items():
        assert isinstance(v, sp.Basic)


# ── Mechanics ────────────────────────────────────────────────────────────────
def test_lagrangian_SHO():
    x, m = sp.Symbol("x"), sp.Symbol("m", positive=True)
    k = sp.Symbol("k", positive=True)
    V = sp.Rational(1, 2) * k * x**2
    res = lagrangian_particle_sympy(V, [x], m)
    assert "L" in res
    assert "EOM" in res
    assert len(res["EOM"]) == 1


def test_phase_space_energy_conserved():
    traj = phase_space_trajectory(1.0, 1.0, 0.5)
    assert traj["H_check"] is True


def test_phase_space_ellipse():
    # For SHO: x(t)=A*cos, p(t)=-mw*A*sin -> ellipse in phase space
    traj = phase_space_trajectory(2.0, 1.0, 0.0)
    # x^2/(2E/omega^2) + p^2/(2E) = 1
    E = traj["E"]
    omega = traj["omega0"]
    m = 1.0
    invariant = 0.5*traj["p"]**2/m + 0.5*m*omega**2*traj["x"]**2
    np.testing.assert_allclose(invariant, E, rtol=1e-9)


def test_mechanics_sympy_5():
    eqs = mechanics_sympy_5()
    assert len(eqs) == 5


# ── Complex analysis ──────────────────────────────────────────────────────────
def test_cauchy_riemann_analytic():
    z = sp.Symbol("z")
    x, y = sp.symbols("x y")
    # f(z) = z^2 = (x+iy)^2 = x^2-y^2 + 2ixy -> analytic
    res = cauchy_riemann_check(z**2, z, x, y)
    assert res["analytic"] is True


def test_cauchy_riemann_non_analytic():
    z = sp.Symbol("z")
    x, y = sp.symbols("x y")
    # f = z_bar = x - iy -> not analytic (du/dx=1, dv/dy=-1 -> CR fails)
    res = cauchy_riemann_check(sp.conjugate(z), z, x, y)
    assert res["analytic"] is False


def test_residue_1_over_z():
    z = sp.Symbol("z")
    res = residue_theorem(1/z, z, [0])
    assert sp.simplify(res["residues"][0] - 1) == 0
    # Contour integral = 2*pi*i
    assert sp.simplify(res["contour_integral"] - 2*sp.pi*sp.I) == 0


def test_residue_1_over_z2_plus_1():
    z = sp.Symbol("z")
    res = residue_theorem(1/(z**2 + 1), z, [sp.I, -sp.I])
    assert sp.simplify(res["residues"][sp.I] + sp.I/2) == 0
    assert sp.simplify(res["residues"][-sp.I] - sp.I/2) == 0


def test_cauchy_integral_formula():
    z = sp.Symbol("z")
    # int_C z^2/(z-1) dz, z0=1: result = 2*pi*i * f(1) = 2*pi*i
    result = cauchy_integral_formula(z**2, z, 1, n=0)
    assert sp.simplify(result - 2*sp.pi*sp.I) == 0


def test_cauchy_integral_derivative():
    z = sp.Symbol("z")
    # int_C z^3/(z-0)^2 dz = 2*pi*i/1! * d/dz[z^3] at z=0 = 0
    result = cauchy_integral_formula(z**3, z, 0, n=1)
    assert sp.simplify(result) == 0


def test_complex_analysis_sympy_5():
    eqs = complex_analysis_sympy_5()
    assert len(eqs) == 5
    for k, v in eqs.items():
        assert isinstance(v, sp.Eq)


# ── Number theory ─────────────────────────────────────────────────────────────
def test_modular_arith_basic():
    res = modular_arithmetic(7, 5, 11)
    assert res["a_mod_m"] == 7
    assert res["mul"] == (7*5) % 11
    assert res["pow_b"] == pow(7, 5, 11)


def test_modular_inverse():
    res = modular_arithmetic(7, 1, 11)
    inv = res["a_inverse"]
    assert (7 * inv) % 11 == 1


def test_modular_no_inverse():
    # gcd(6, 9) = 3 != 1 -> no inverse
    res = modular_arithmetic(6, 1, 9)
    assert res["a_inverse"] is None
    assert res["coprime"] is False


def test_softmax_sums_to_one():
    x = np.array([1.0, 2.0, 3.0, 0.5])
    res = softmax_and_extrema(x)
    assert abs(res["softmax"].sum() - 1.0) < 1e-12


def test_softmax_argmax():
    x = np.array([1.0, 5.0, 2.0])
    res = softmax_and_extrema(x)
    assert res["argmax"] == 1
    assert res["argmin"] == 0


def test_softmax_cold_temperature():
    # T->0: softmax concentrates at max
    x = np.array([1.0, 3.0, 2.0])
    res = softmax_and_extrema(x, temperature=0.01)
    assert res["softmax"][1] > 0.99


def test_crt_basic():
    # x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)
    res = chinese_remainder_theorem([2, 3, 2], [3, 5, 7])
    assert res["verify"] is True
    x = res["x"]
    assert x % 3 == 2 and x % 5 == 3 and x % 7 == 2


def test_crt_single():
    res = chinese_remainder_theorem([1], [7])
    assert res["x"] == 1


def test_number_theory_sympy_5():
    eqs = number_theory_sympy_5()
    assert len(eqs) == 5
    for k, v in eqs.items():
        assert isinstance(v, sp.Basic)


# ── All sections ──────────────────────────────────────────────────────────────
def test_all_math_methods_sections():
    all_eqs = all_math_methods_sympy()
    assert len(all_eqs) == 6
    for section, eqs in all_eqs.items():
        assert len(eqs) == 5, f"{section} should have 5 equations"


# ── Helper ────────────────────────────────────────────────────────────────────
def pytest_approx(val, rel=1e-6):
    import pytest
    return pytest.approx(val, rel=rel)
