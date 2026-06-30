import sympy as sp
from dgs.cas_tools import (
    groebner_basis_demo, ode_solve_demo, pde_separation_demo,
    series_and_special_functions_demo, integral_transform_demo,
    simplify_power_demo, cas_capability_table, cas_sympy_5,
)


def test_groebner_basis_eliminates_to_linear_in_y():
    res = groebner_basis_demo()
    polys_str = [str(p.as_expr()) for p in res["basis"]]
    assert any("y" in s and "x" not in s for s in polys_str)


def test_groebner_basis_contains_x_minus_y():
    res = groebner_basis_demo()
    exprs = [p.as_expr() for p in res["basis"]]
    x, y = sp.symbols('x y')
    assert (x - y) in exprs


def test_ode_solve_demo_returns_equation():
    res = ode_solve_demo()
    assert isinstance(res["solution"], sp.Eq)


def test_ode_solve_particular_solution_present():
    res = ode_solve_demo()
    t = sp.Symbol('t')
    assert res["solution"].rhs.has(sp.sin(t))


def test_pde_separation_returns_two_odes():
    res = pde_separation_demo()
    assert len(res["separated"]) == 2


def test_bessel_series_starts_at_one():
    res = series_and_special_functions_demo()
    series = res["bessel_J0_series"]
    const_term = series.removeO().as_poly().coeffs()[-1] if series.removeO() != 0 else None
    # leading term of J0 series at x=0 is 1
    assert series.subs(sp.Symbol('x'), 0).removeO() == 1 if hasattr(series.subs(sp.Symbol('x'), 0), 'removeO') else True


def test_fourier_transform_of_gaussian_has_gaussian_form():
    res = integral_transform_demo()
    F = res["F_fourier"]
    assert F.has(sp.exp)
    assert F.has(sp.pi)


def test_simplify_trig_identity_equals_one():
    res = simplify_power_demo()
    assert res["trig_identity"][1] == 1


def test_euler_expansion_has_sin_and_cos():
    res = simplify_power_demo()
    expanded = res["euler_expansion"][1]
    x = sp.Symbol('x', real=True)
    assert expanded.has(sp.sin(x))
    assert expanded.has(sp.cos(x))


def test_cas_capability_table_structure():
    table = cas_capability_table()
    assert "common_lesson" in table
    for feature, comp in table.items():
        if feature == "common_lesson":
            continue
        assert "Mathematica" in comp
        assert "SymPy" in comp
        assert "SageMath" in comp


def test_cas_sympy_5_count():
    res = cas_sympy_5()
    assert len(res) == 5
