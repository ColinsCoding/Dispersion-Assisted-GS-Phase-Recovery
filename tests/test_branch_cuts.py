"""Tests for dgs/branch_cuts.py — log branch cuts, phase unwrapping, GVD."""
import numpy as np
import sympy as sp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.branch_cuts import (
    complex_log, principal_log, branch_cut_demo, sqrt_branch_cut,
    unwrap_phase, phase_nyquist_check, gvd_phase_unwrapped,
    gs_phase_unwrap_and_gvd, riemann_sheet_log,
    loop_around_branch_point, branch_cuts_sympy_5,
)


# ── Complex log ───────────────────────────────────────────────────────────────

def test_log_positive_real():
    z = np.array([np.e + 0j])
    log_z = complex_log(z)
    assert abs(log_z[0] - 1.0) < 1e-10


def test_log_branch_difference():
    z = np.array([-1.0 + 0j])
    log0 = complex_log(z, branch=0)
    log1 = complex_log(z, branch=1)
    assert abs(np.imag(log1[0]) - np.imag(log0[0]) - 2*np.pi) < 1e-10


def test_principal_log_on_positive():
    z = np.array([4.0 + 0j])
    log_z = principal_log(z)
    assert abs(np.real(log_z[0]) - np.log(4)) < 1e-10
    assert abs(np.imag(log_z[0])) < 1e-10


def test_principal_log_imaginary_arg():
    # Log(i) = i*pi/2
    z = np.array([1j])
    log_z = principal_log(z)
    assert abs(np.imag(log_z[0]) - np.pi/2) < 1e-10


def test_branch_cut_jump():
    res = branch_cut_demo()
    # Jump across (-inf,0] should be ~2*pi
    assert abs(res["mean_jump"] - 2*np.pi) < 1e-4


def test_sqrt_branch_cut_imaginary():
    res = sqrt_branch_cut()
    # sqrt just above cut at x=-1: should be ~i
    idx = np.argmin(np.abs(res["x"] + 1))
    assert abs(np.real(res["sqrt_above"][idx])) < 0.01
    assert abs(np.imag(res["sqrt_above"][idx]) - 1.0) < 0.01


# ── Phase unwrapping ──────────────────────────────────────────────────────────

def test_unwrap_phase_no_jump():
    phi = np.linspace(0, np.pi/2, 100)
    unwrapped = unwrap_phase(phi)
    np.testing.assert_allclose(unwrapped, phi, atol=1e-10)


def test_unwrap_phase_removes_jump():
    # Wrapped: 0 -> pi -> 0 (with 2*pi jump at pi)
    phi = np.array([0.0, np.pi/2, np.pi, -np.pi + 0.1, 0.0])
    unwrapped = unwrap_phase(phi)
    # Should be monotone increasing (no jump)
    assert np.all(np.diff(unwrapped) >= -0.2)


def test_phase_nyquist_ok():
    # Short fiber, narrow BW: should be OK
    res = phase_nyquist_check(-22e-27, 100.0, 2*np.pi*500e9, 2048)
    assert bool(res["nyquist_ok"]) is True


def test_phase_nyquist_violated():
    # Long fiber, wide BW: should fail
    res = phase_nyquist_check(-22e-27, 10e3, 2*np.pi*5e12, 2048)
    assert bool(res["nyquist_ok"]) is False
    assert res["n_pts_min"] > 2048


def test_gvd_unwrap_beta2_recovery():
    # Narrow BW + short fiber: Nyquist OK -> correct recovery
    omega = np.linspace(-2*np.pi*500e9, 2*np.pi*500e9, 4096)
    beta2_true = -22e-27
    res = gvd_phase_unwrapped(omega, beta2_true, L_m=100.0)
    assert bool(res["nyquist"]["nyquist_ok"]) is True
    assert res["beta2_error_pct"] < 1.0


def test_gvd_unwrap_crossings():
    omega = np.linspace(-2*np.pi*500e9, 2*np.pi*500e9, 4096)
    res = gvd_phase_unwrapped(omega, -22e-27, L_m=100.0)
    # At least 1 crossing (phi spans ~10 rad)
    assert res["n_branch_cut_crossings"] >= 1


def test_gs_phase_unwrap_clean():
    # Clean chirped Gaussian: no nulls, no pi slips
    omega = np.linspace(-2*np.pi*1e12, 2*np.pi*1e12, 1024)
    beta2, L = -22e-27, 100.0
    E = np.exp(-0.5*(omega*50e-15)**2) * np.exp(1j * 0.5*beta2*L*omega**2)
    res = gs_phase_unwrap_and_gvd(E, omega, beta2_nominal=beta2)
    assert len(res["pi_slips"]) == 0
    assert "phi_unwrapped" in res


def test_gs_phase_unwrap_gvd_estimate():
    # gs_phase_unwrap_and_gvd returns d^2(phi)/d(omega^2) = beta2*L.
    # Verify by checking the group delay slope directly.
    omega = np.linspace(-2*np.pi*500e9, 2*np.pi*500e9, 4096)
    beta2, L = -22e-27, 100.0
    domega = omega[1] - omega[0]
    E = np.exp(-0.5*(omega*100e-15)**2) * np.exp(1j * 0.5*beta2*L*omega**2)
    res = gs_phase_unwrap_and_gvd(E, omega)
    # gvd_estimated = beta2*L (second deriv of phase, not normalized by L)
    gvd_beta2_L = res["gvd_estimated"]
    beta2_recovered = gvd_beta2_L / L
    assert abs(beta2_recovered - beta2) / abs(beta2) < 0.05


# ── Riemann surface ───────────────────────────────────────────────────────────

def test_riemann_sheet_offset():
    z = np.array([1.0 + 0j])
    log0 = riemann_sheet_log(z, sheet=0)
    log2 = riemann_sheet_log(z, sheet=2)
    assert abs(np.imag(log2[0]) - np.imag(log0[0]) - 4*np.pi) < 1e-10


def test_loop_winding_number():
    res = loop_around_branch_point()
    assert res["winding_number"] == 1


def test_loop_total_winding():
    res = loop_around_branch_point()
    assert abs(res["total_winding"] - 2*np.pi) < 0.1


# ── SymPy ─────────────────────────────────────────────────────────────────────

def test_sympy_5_count():
    eqs = branch_cuts_sympy_5()
    assert len(eqs) == 5


def test_sympy_5_all_basic():
    eqs = branch_cuts_sympy_5()
    for k, v in eqs.items():
        assert isinstance(v, sp.Basic)
