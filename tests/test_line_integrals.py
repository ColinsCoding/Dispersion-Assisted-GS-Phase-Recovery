"""Tests for dgs/line_integrals.py"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.line_integrals import (
    scalar_line_integral, vector_line_integral,
    verify_greens, verify_stokes, verify_divergence,
    lorentz_work, biot_savart_segment,
)

PI = np.pi


def test_arc_length_circle():
    r = lambda t: np.array([np.cos(2*PI*t), np.sin(2*PI*t), 0.0])
    val = scalar_line_integral(lambda xyz: 1.0, r, n_pts=128)
    assert abs(val - 2*PI) < 1e-4, f"arc length={val}"


def test_conservative_field_work():
    # grad(x^2+y^2) from (0,0) to (1,1) = phi(1,1)-phi(0,0) = 2
    F = lambda r: np.array([2*r[0], 2*r[1], 0.0])
    r = lambda t: np.array([t, t**2, 0.0])
    val = vector_line_integral(F, r)
    assert abs(val - 2.0) < 1e-4, f"work={val}"


def test_circulation_rotation_field():
    # int_C (-y,x,0).dr over unit circle = 2*pi (curl=2, area=pi -> 2*pi via Stokes)
    F = lambda r: np.array([-r[1], r[0], 0.0])
    r = lambda t: np.array([np.cos(2*PI*t), np.sin(2*PI*t), 0.0])
    val = vector_line_integral(F, r)
    assert abs(val - 2*PI) < 1e-3, f"circulation={val}"


def test_greens_theorem():
    F = lambda x, y: (-y, x)
    bdry = lambda t: (np.cos(2*PI*t), np.sin(2*PI*t))
    g = verify_greens(F, bdry, domain_samples=800)
    assert g['relative_error'] < 0.05, f"Green's rel_err={g['relative_error']}"


def test_stokes_theorem():
    F = lambda r: np.array([r[1], -r[0], r[2]])
    r_eq = lambda t: np.array([np.cos(2*PI*t), np.sin(2*PI*t), 0.0])
    r_h  = lambda u, v: np.array([
        np.sin(PI/2*u)*np.cos(2*PI*v),
        np.sin(PI/2*u)*np.sin(2*PI*v),
        np.cos(PI/2*u),
    ])
    s = verify_stokes(F, r_eq, r_h, n_line_pts=256, n_surf_pts=20)
    assert s['relative_error'] < 0.05, f"Stokes rel_err={s['relative_error']}"


def test_divergence_theorem():
    F = lambda r: r.copy()   # F=r, div F=3, int_V 3dV = 4*pi
    d = verify_divergence(F, radius=1.0, n_surf=40, n_vol=20)
    assert d['relative_error'] < 0.05, f"Divergence rel_err={d['relative_error']}"


def test_magnetic_force_does_no_work():
    E_zero = lambda r, t: np.zeros(3)
    B_z    = lambda r, t: np.array([0.0, 0.0, 1.0])
    r_c = lambda t: np.array([np.cos(2*PI*t), np.sin(2*PI*t), 0.0])
    v_c = lambda t: np.array([-np.sin(2*PI*t), np.cos(2*PI*t), 0.0]) * 2*PI
    W = lorentz_work(E_zero, B_z, r_c, v_c, q=1.0)
    assert abs(W) < 1e-8, f"magnetic work={W}"


def test_biot_savart_long_wire():
    B = biot_savart_segment(
        r_obs=np.array([1.0, 0.0, 0.0]),
        r_start=np.array([0.0, 0.0, -1000.0]),
        r_end  =np.array([0.0, 0.0,  1000.0]),
        I=1.0,
    )
    B_theory = 2e-7
    assert abs(np.linalg.norm(B) - B_theory) / B_theory < 0.01, f"|B|={np.linalg.norm(B)}"
