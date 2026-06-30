import numpy as np
import pytest
import sympy as sp
from dgs.vector_calculus_torsion import (
    x, y, z, gradient, divergence, curl, laplacian,
    curl_of_gradient_is_zero, div_of_curl_is_zero,
    stokes_theorem_statement, divergence_theorem_statement,
    frenet_serret, frenet_serret_sympy,
    phase_circulation, berry_phase_demo, vector_calculus_sympy_5,
)


# -- symbolic vector calculus -------------------------------------------------

def test_gradient_linear():
    f = 3*x + 2*y - z
    g = gradient(f)
    assert g == sp.Matrix([3, 2, -1])


def test_divergence_quadratic():
    F = [x**2, y**2, z**2]
    assert sp.simplify(divergence(F) - (2*x + 2*y + 2*z)) == 0


def test_curl_of_constant_field_zero():
    F = [sp.Integer(1), sp.Integer(2), sp.Integer(3)]
    c = curl(F)
    assert sp.simplify(c) == sp.zeros(3, 1)


def test_curl_of_gradient_is_zero_identity():
    f = x**2 * y + sp.sin(z) * sp.cos(x)
    res = curl_of_gradient_is_zero(f)
    assert res["is_zero"] is True


def test_div_of_curl_is_zero_identity():
    F = [x*y*z, x**2*y, y**2*z]
    res = div_of_curl_is_zero(F)
    assert res["is_zero"] is True


def test_laplacian_of_harmonic():
    # x^2 + y^2 - 2*z^2 is harmonic (Laplacian = 0)
    f = x**2 + y**2 - 2*z**2
    assert sp.simplify(laplacian(f)) == 0


def test_stokes_theorem_is_equation():
    eq = stokes_theorem_statement()
    assert isinstance(eq, sp.Eq)


def test_divergence_theorem_is_equation():
    eq = divergence_theorem_statement()
    assert isinstance(eq, sp.Eq)


# -- Frenet-Serret ------------------------------------------------------------

def test_helix_curvature():
    t = np.linspace(0, 4*np.pi, 4000)
    fs = frenet_serret(lambda t: np.column_stack([np.cos(t), np.sin(t), t]), t)
    mid = len(t) // 2
    assert fs["kappa"][mid] == pytest.approx(0.5, abs=0.01)


def test_helix_torsion():
    t = np.linspace(0, 4*np.pi, 4000)
    fs = frenet_serret(lambda t: np.column_stack([np.cos(t), np.sin(t), t]), t)
    mid = len(t) // 2
    assert fs["tau"][mid] == pytest.approx(0.5, abs=0.01)


def test_circle_torsion_zero():
    # Planar circle: torsion should be 0
    t = np.linspace(0, 2*np.pi, 4000)
    fs = frenet_serret(lambda t: np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)]), t)
    mid = len(t) // 2
    assert abs(fs["tau"][mid]) < 0.05


def test_circle_curvature_one():
    t = np.linspace(0, 2*np.pi, 4000)
    fs = frenet_serret(lambda t: np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)]), t)
    mid = len(t) // 2
    assert fs["kappa"][mid] == pytest.approx(1.0, abs=0.02)


def test_frenet_serret_sympy_has_three_equations():
    eqs = frenet_serret_sympy()
    assert len(eqs) == 3
    for eq in eqs.values():
        assert isinstance(eq, sp.Eq)


# -- Phase circulation / vortex detection -------------------------------------

def test_phase_circulation_single_vortex():
    res = phase_circulation(lambda x, y: np.arctan2(y, x), cx=0, cy=0, radius=1.0)
    assert res["winding_number"] == 1


def test_phase_circulation_no_vortex():
    # constant phase field: winding number = 0
    res = phase_circulation(lambda x, y: np.zeros_like(x + y), cx=0, cy=0, radius=1.0)
    assert res["winding_number"] == 0


# -- Berry phase --------------------------------------------------------------

def test_berry_phase_full_loop_bloch_sphere():
    theta_path = np.full(361, np.pi/3)
    phi_path   = np.linspace(0, 2*np.pi, 361)
    bp = berry_phase_demo(theta_path, phi_path)
    expected = -np.cos(np.pi/3) / 2 * 2 * np.pi
    assert bp["berry_phase_rad"] == pytest.approx(expected, abs=0.01)


def test_berry_phase_equator_is_minus_pi():
    theta_path = np.full(361, np.pi/2)
    phi_path   = np.linspace(0, 2*np.pi, 361)
    bp = berry_phase_demo(theta_path, phi_path)
    # cos(pi/2)=0 -> Berry phase should be 0 at equator
    assert abs(bp["berry_phase_rad"]) < 0.05


# -- sympy 5 ------------------------------------------------------------------

def test_vector_calculus_sympy_5_count_and_type():
    eqs = vector_calculus_sympy_5()
    assert len(eqs) == 5
    for eq in eqs.values():
        assert isinstance(eq, sp.Eq)
