"""Tests for dgs/dimensions_inner_product.py (+ opt-in MATLAB twin: RUN_MATLAB=1)."""
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest
import sympy as sp

from dgs.dimensions_inner_product import (
    d_sym, ball_volume_symbolic, ball_volume, ball_volume_mc, peak_dimension,
    thomson_field_d, inner, norm, cauchy_schwarz_holds, fourier_inner_symbolic,
)


def test_symbolic_volume_formula():
    S, V = ball_volume_symbolic()
    assert sp.simplify(V - sp.pi ** (d_sym / 2) / sp.gamma(d_sym / 2 + 1)) == 0
    assert sp.simplify(S.subs(d_sym, 3) - 4 * sp.pi) == 0              # sphere area 4 pi r^2


@pytest.mark.parametrize("d,exact", [(1, 2.0), (2, np.pi), (3, 4 * np.pi / 3), (4, np.pi ** 2 / 2)])
def test_known_volumes_and_monte_carlo(d, exact):
    assert ball_volume(d) == pytest.approx(exact)
    n = 400_000
    p = exact / 2 ** d
    tol = 5 * np.sqrt(p * (1 - p) / n) * 2 ** d
    assert abs(ball_volume_mc(d, n=n) - exact) <= tol + 1e-12        # d = 1 is exact (p = 1)


def test_fractional_dimension_interpolates():
    assert ball_volume(1) < ball_volume(1.5) < ball_volume(2)
    assert peak_dimension() == pytest.approx(5.2569, abs=1e-3)
    with pytest.raises(ValueError):
        ball_volume_mc(2.5)


def test_gauss_law_every_dimension():
    Q, eps, R, r = sp.symbols("Q epsilon R r", positive=True)
    ins, out, _ = thomson_field_d(3)
    assert sp.simplify(ins - Q * r / (4 * sp.pi * eps * R ** 3)) == 0
    assert sp.simplify(out - Q / (4 * sp.pi * eps * r ** 2)) == 0
    for d in (1, 2, 2.5, 4):
        ins_d, out_d, _ = thomson_field_d(d)
        assert sp.simplify(sp.diff(ins_d, r, 2)) == 0                   # linear in r
        assert sp.simplify(ins_d.subs(r, R) - out_d.subs(r, R)) == 0    # continuous at surface


def test_inner_product_properties():
    rng = np.random.default_rng(0)
    for n in range(1, 8):
        u = rng.normal(size=n) + 1j * rng.normal(size=n)
        v = rng.normal(size=n) + 1j * rng.normal(size=n)
        assert inner(u, v) == pytest.approx(np.vdot(u, v))
        assert inner(u, v) == pytest.approx(np.conj(inner(v, u)))          # conjugate symmetry
        assert inner(u, u).imag == pytest.approx(0) and inner(u, u).real > 0
        assert cauchy_schwarz_holds(u, v)
    u, v = [1 + 2j, 3 - 1j], [2j, 1]
    assert inner(u, v) == 7 + 3j and np.dot(u, v) == -1 + 1j               # dot does NOT conjugate
    with pytest.raises(ValueError):
        inner([1], [1, 2])


def test_fourier_orthogonality():
    for k in range(-2, 3):
        for m in range(-2, 3):
            assert fourier_inner_symbolic(k, m) == (2 * sp.pi if k == m else 0)


@pytest.mark.skipif(os.environ.get("RUN_MATLAB") != "1", reason="set RUN_MATLAB=1 (MATLAB startup is slow)")
def test_matlab_twin():
    exe = shutil.which("matlab") or r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"
    root = pathlib.Path(__file__).resolve().parent.parent
    out = subprocess.run([exe, "-batch", "run('learn/matlab/dimensions_inner_product.m')"], cwd=root,
                         capture_output=True, text=True, timeout=600).stdout
    assert "MATLAB checks passed" in out and "d = 5.2569" in out
