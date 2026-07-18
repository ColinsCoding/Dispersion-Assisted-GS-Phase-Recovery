"""Symbolic-correctness tests (SymPy)."""
from __future__ import annotations

import numpy as np
import sympy as sp

from physics.symbolic import gaussian_beam_width, gouy_phase, jacobian, rayleigh_range


def test_rayleigh_range_formula() -> None:
    zr = rayleigh_range()
    w0, lam = zr.symbols
    assert sp.simplify(zr.expr - sp.pi * w0**2 / lam) == 0


def test_beam_width_gradient_matches_hand_derivative() -> None:
    bw = gaussian_beam_width()
    z, w0, zR = bw.symbols
    grad = bw.gradient()
    # d/dz [w0 sqrt(1+(z/zR)^2)] = w0 z / (zR^2 sqrt(1+(z/zR)^2))
    expected = w0 * z / (zR**2 * sp.sqrt(1 + (z / zR) ** 2))
    assert sp.simplify(grad[0] - expected) == 0


def test_hessian_is_symmetric() -> None:
    bw = gaussian_beam_width()
    hess = bw.hessian()
    assert sp.simplify(hess - hess.T) == sp.zeros(*hess.shape)


def test_lambdify_matches_subs() -> None:
    bw = gaussian_beam_width()
    f = bw.lambdify("numpy")
    z_vals = np.linspace(-500, 500, 11)
    numeric = f(z_vals, 10.0, 200.0)
    symbolic = [bw.evaluate(z=float(zv), w0=10.0, zR=200.0) for zv in z_vals]
    assert np.allclose(numeric, symbolic)


def test_jacobian_shape_and_values() -> None:
    x, y = sp.symbols("x y")
    jac = jacobian([x**2 * y, sp.sin(x + y)], [x, y])
    assert jac.shape == (2, 2)
    assert sp.simplify(jac[0, 0] - 2 * x * y) == 0


def test_gouy_phase_endpoints() -> None:
    gp = gouy_phase()
    assert abs(gp.evaluate(z=1e9, zR=1.0) - np.pi / 2) < 1e-6
