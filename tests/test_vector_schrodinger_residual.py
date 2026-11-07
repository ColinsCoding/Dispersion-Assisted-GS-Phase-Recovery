"""Tests for dgs.vector_schrodinger_residual."""
import numpy as np
import pytest
import sympy as sp

from dgs import vector_schrodinger_residual as v


def test_hermitian_potential():
    a, b = sp.symbols("a b", real=True)
    assert v.is_hermitian(v.hermitian_potential(v.x ** 2, 1, a, b))
    assert not v.is_hermitian(sp.Matrix([[0, sp.I], [sp.I, 0]]))


def test_real_imag_split_matches_hand_derivation():
    (r1, i1, r2, i2), R, f = v.real_imag_residuals()
    hb, m, x, t = v.hbar, v.m, v.x, v.t
    u1, v1, u2, v2, V11, V22, a, b = (f[k] for k in ["u1", "v1", "u2", "v2", "V11", "V22", "a", "b"])
    expect_r1 = -hb * v1.diff(t) + hb ** 2 / (2 * m) * u1.diff(x, 2) - V11 * u1 - a * u2 + b * v2
    expect_i1 = hb * u1.diff(t) + hb ** 2 / (2 * m) * v1.diff(x, 2) - V11 * v1 - a * v2 - b * u2
    expect_r2 = -hb * v2.diff(t) + hb ** 2 / (2 * m) * u2.diff(x, 2) - V22 * u2 - a * u1 - b * v1
    expect_i2 = hb * u2.diff(t) + hb ** 2 / (2 * m) * v2.diff(x, 2) - V22 * v2 - a * v1 + b * u1
    for got, want in [(r1, expect_r1), (i1, expect_i1), (r2, expect_r2), (i2, expect_i2)]:
        assert sp.simplify(got - want) == 0


def test_split_agrees_with_sp_re_im_on_concrete_fields():
    Psi, V, F = v.manufactured()
    for j in range(2):
        re_, im_ = sp.re(F[j]), sp.im(F[j])
        pts = [(-1.3, 0.4), (0.2, 1.1), (2.0, -0.7)]
        for xv, tv in pts:
            val = complex(F[j].subs({v.x: xv, v.t: tv}).evalf())
            assert np.isclose(float(re_.subs({v.x: xv, v.t: tv}).evalf()), val.real)
            assert np.isclose(float(im_.subs({v.x: xv, v.t: tv}).evalf()), val.imag)


def test_exact_solution_and_stationary_pair():
    Psi, V = v.rabi_oscillator_solution()
    assert v.simplify_matrix(v.vector_residual(Psi, V)) == sp.zeros(2, 1)
    assert v.total_probability(Psi) == 1
    for P, En in v.stationary_pair():
        assert v.simplify_matrix(v.stationary_residual(P, v.coupled_oscillator_potential(), En)) == sp.zeros(2, 1)
    P, _ = v.stationary_pair()[0]
    _, E0 = v.ground_state()
    assert v.simplify_matrix(v.stationary_residual(P, v.coupled_oscillator_potential(), E0)) != sp.zeros(2, 1)


def test_manufactured_forcing_is_exact_rational():
    _, _, F = v.manufactured()
    assert not any(isinstance(a, sp.Float) for a in F.atoms(sp.Number))
    assert F != sp.zeros(2, 1)


def test_discrete_residual_converges_second_order():
    Psi, V, F = v.manufactured()
    pf, Ff, Vf = v.lambdify_vector(Psi), v.lambdify_vector(F), v.potential_numeric(V)
    errs, hs = [], []
    for n in (101, 201, 401):
        xg = np.linspace(-8, 8, n)
        h = xg[1] - xg[0]
        xi, Rd = v.discrete_residual(pf, Vf, xg, 0.7, h)
        errs.append(v.rms(Rd - Ff(xi, 0.7)))
        hs.append(h)
    assert np.allclose(v.convergence_orders(errs, hs), 2.0, atol=0.05)


def test_discrete_residual_of_exact_solution_is_small():
    Psi, V = v.rabi_oscillator_solution()
    num = {v.hbar: 1, v.m: 1, v.omega: 1, v.g: sp.Rational(1, 5)}
    pf = v.lambdify_vector(Psi.subs(num))
    Vf = v.potential_numeric(V.subs(num))
    xg = np.linspace(-8, 8, 801)
    _, Rd = v.discrete_residual(pf, Vf, xg, 1.0, xg[1] - xg[0])
    assert v.rms(Rd) < 1e-4


def test_block_hamiltonian_eigs():
    xg = np.linspace(-10, 10, 1201)
    H = v.block_hamiltonian(xg, xg ** 2 / 2, xg ** 2 / 2, 0.2)
    vals, vecs = v.lowest_states(H, 4)
    assert np.allclose(vals, [0.3, 0.7, 1.3, 1.7], atol=1e-4)
    assert np.all(v.stationary_discrete_residuals(H, vals, vecs) < 1e-8)


def test_two_level_cpu():
    ts = np.linspace(0, 5, 11)
    assert np.allclose(v.rabi_z_expectation_cpu(0.3, ts), np.cos(0.6 * ts))
    U = v.two_level_propagator(0.4, 1.3)
    assert np.allclose(U.conj().T @ U, np.eye(2))


def test_optional_backends_degrade_gracefully():
    if not v.cudaq_available():
        assert v.rabi_z_expectation_cudaq(0.3, [0.0]) is None
    if not v.cuquantum_available():
        assert v.apply_propagator_cuquantum(np.eye(2), [1, 0]) is None


def test_isolation_rows_all_pass():
    rows = v.isolation_rows()
    assert {"id", "equation_or_term", "role", "units", "verification"} <= set(rows[0])
    assert len({r["id"] for r in rows}) == len(rows)
    assert not any(r["verification"].startswith("FAIL") for r in rows)


@pytest.mark.parametrize("call", [
    lambda: v.vector_residual(sp.Matrix([1, 2, 3]), sp.eye(2)),
    lambda: v.stationary_residual(sp.Matrix([1, 2]), sp.eye(3)),
    lambda: v.discrete_residual(lambda a, b: np.zeros((2, 2)), None, [0.0, 1.0], 0.0, 0.1),
    lambda: v.discrete_residual(lambda a, b: np.zeros((2, 3)), None, [0.0, 1.0, 2.0], 0.0, 0.0),
    lambda: v.block_hamiltonian([0.0, 1.0], 0, 0, 0),
])
def test_validation(call):
    with pytest.raises(ValueError):
        call()
