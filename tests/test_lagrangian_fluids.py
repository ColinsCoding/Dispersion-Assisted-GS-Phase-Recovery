"""Smoke tests for dgs/lagrangian_fluids.py (and the torch twin, if torch is present)."""
import numpy as np
import pytest
import sympy as sp

from dgs import lagrangian_fluids as L


def test_symbolic_identities():
    assert L.momentum_conservation_identity() == 0
    assert L.cole_hopf_identity() == 0
    assert L.taylor_shock()[1] == 0
    assert L.lagrangian_burgers_check()[0] == 0
    assert L.four_velocity_derivative_check() == 0
    assert L.relativistic_dust_reduces_to_burgers() == (1, 0)
    assert L.madelung_check() == (0, 0)


def test_breaking_time_sine():
    assert L.breaking_time(sp.sin(2 * sp.pi * L.x)) == pytest.approx(1 / (2 * np.pi), rel=1e-9)
    assert L.breaking_time(sp.Integer(1) + 0 * L.x) is None


def test_cole_hopf_exact_initial_condition():
    xg = np.linspace(0, 1, 64, endpoint=False)
    assert np.max(np.abs(L.cole_hopf_exact(xg, 0.0, 0.02) - np.sin(2 * np.pi * xg))) < 1e-6


@pytest.mark.parametrize("viscosity,tol", [(0.05, 2e-5), (0.02, 6e-5)])
def test_fd_matches_exact(viscosity, tol):
    xg, u, _ = L.solve_burgers(512, viscosity, 0.5)
    assert np.max(np.abs(u - L.cole_hopf_exact(xg, 0.5, viscosity))) < tol


def test_fd_second_order():
    e = []
    for n in (128, 256):
        xg, u, _ = L.solve_burgers(n, 0.05, 0.2)
        e.append(np.max(np.abs(u - L.cole_hopf_exact(xg, 0.2, 0.05))))
    assert 3.0 < e[0] / e[1] < 5.0


def test_womersley():
    assert L.womersley_ode_residual(5.0) < 1e-3
    r = np.linspace(0, 1, 11)
    assert np.allclose(L.womersley_profile(r, 0.02, 0.0), 1 - r**2, atol=1e-4)
    Re, a = L.reynolds_womersley(*L.VESSELS["capillary"], L.FLUIDS["blood 37 C (high shear)"])
    assert Re < 0.01 and a < 0.05


def test_stokes_settling():
    v, Re = L.stokes_settling(10e-6)
    assert v == pytest.approx(3.0e-3, rel=0.02)
    assert Re < 0.01
    with pytest.raises(ValueError):
        L.stokes_settling(-1e-6)


def test_bounds():
    with pytest.raises(ValueError):
        L.solve_burgers(8, 0.01, 0.5)
    with pytest.raises(ValueError):
        L.cole_hopf_exact(np.zeros(4), 0.1, 0.0)
    with pytest.raises(ValueError):
        L.reynolds_womersley(1e-3, 0.1, -1.0)


def test_torch_twin():
    torch = pytest.importorskip("torch")
    from dgs.torch.burgers_batched import solve_batched
    xg, u, steps, _, _ = solve_batched([0.05, 0.05], n=128, t_end=0.2, device=torch.device("cpu"))
    xr, ur, _ = L.solve_burgers(128, 0.05, 0.2, dt=0.2 / steps)
    assert np.max(np.abs(u[0] - ur)) < 1e-12
