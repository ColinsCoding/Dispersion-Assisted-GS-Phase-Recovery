import sympy as sp
from dgs.gauge_invariance import (
    x, y, z, t, c, gauge_transform, E_field, B_field,
    verify_gauge_invariance, coulomb_gauge_condition, lorenz_gauge_condition,
    gauge_invariance_sympy_5,
)


def test_gauge_transform_shifts_V_and_A():
    V = sp.Symbol('V0')
    A = sp.Matrix([0, 0, 0])
    lam = x * y * t
    V_new, A_new = gauge_transform(V, A, lam)
    assert sp.simplify(V_new - (V - y * x)) == 0
    assert A_new == sp.Matrix([y * t, x * t, 0])


def test_E_field_static_potential():
    V = x**2
    A = sp.Matrix([0, 0, 0])
    E = E_field(V, A)
    assert E == sp.Matrix([-2 * x, 0, 0])


def test_B_field_uniform_A_is_zero():
    A = sp.Matrix([y, 0, 0])
    B = B_field(A)
    assert sp.simplify(B[2] - (-1)) == 0


def test_verify_gauge_invariance_plane_wave():
    k_, omega = sp.symbols('k omega', positive=True)
    V0 = sp.cos(k_ * x - omega * t)
    A0 = sp.Matrix([sp.sin(k_ * x - omega * t), 0, 0])
    lam = x * y * t
    res = verify_gauge_invariance(V0, A0, lam)
    assert res["E_invariant"] is True
    assert res["B_invariant"] is True


def test_verify_gauge_invariance_static_case():
    V0 = x**2 + y**2
    A0 = sp.Matrix([0, z, 0])
    lam = x**2 * y
    res = verify_gauge_invariance(V0, A0, lam)
    assert res["E_invariant"] is True
    assert res["B_invariant"] is True


def test_coulomb_gauge_condition_is_div_zero():
    eq = coulomb_gauge_condition()
    assert isinstance(eq, sp.Eq)
    assert eq.rhs == 0


def test_lorenz_gauge_condition_has_c_squared():
    eq = lorenz_gauge_condition()
    assert isinstance(eq, sp.Eq)
    assert c in eq.lhs.free_symbols


def test_gauge_invariance_sympy_5_count_and_type():
    eqs = gauge_invariance_sympy_5()
    assert len(eqs) == 5
    for eq in eqs.values():
        assert isinstance(eq, sp.Eq)
