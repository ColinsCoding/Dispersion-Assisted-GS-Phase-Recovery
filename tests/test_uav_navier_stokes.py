"""Tests for dgs/uav_navier_stokes.py."""
import numpy as np
import pytest
import sympy as sp

from dgs.uav_navier_stokes import (
    x, y, t, nu, rho,
    ns_residuals, nondimensional_ns, reynolds,
    couette, plane_poiseuille, stokes_first_problem, taylor_green, blasius,
    actuator_disk_derivation, hover_power, forward_flight_power,
    energy_per_mile_wh, fit_surrogate, check_analogy,
)


@pytest.mark.parametrize("sol", [couette, plane_poiseuille,
                                 stokes_first_problem, taylor_green])
def test_exact_solutions_have_zero_residual(sol):
    assert ns_residuals(*sol()) == (0, 0, 0)


def test_wrong_field_has_nonzero_residual():
    # Taylor-Green with the wrong decay rate is NOT a solution
    F = sp.exp(-nu * t)
    r = ns_residuals(sp.sin(x) * sp.cos(y) * F, -sp.cos(x) * sp.sin(y) * F,
                     rho / 4 * (sp.cos(2 * x) + sp.cos(2 * y)) * F ** 2)
    assert r[0] == 0 and r[1] != 0


def test_nondimensional_only_reynolds_survives():
    expr, Re = nondimensional_ns()
    assert expr.free_symbols == {sp.Symbol("x_s", real=True),
                                 sp.Symbol("y_s", real=True),
                                 sp.Symbol("t_s", real=True), Re}


def test_reynolds():
    assert reynolds(10.0, 0.15) == pytest.approx(1e5)
    with pytest.raises(ValueError):
        reynolds(1.0, 0.0)


def test_blasius_textbook_constants():
    b = blasius()
    assert b["fpp0"] == pytest.approx(0.33206, abs=1e-5)
    assert b["delta99"] == pytest.approx(4.91, abs=0.01)
    assert b["fp"][-1] == pytest.approx(1.0, abs=1e-6)


def test_actuator_disk():
    d = actuator_disk_derivation()
    T, A = sp.symbols("T A", positive=True)
    assert sp.simplify(d["w"] - 2 * d["v_i"]) == 0
    assert sp.simplify(d["P"] - T ** sp.Rational(3, 2) / sp.sqrt(2 * rho * A)) == 0


def test_hover_power_scaling():
    p1 = hover_power(5, 0.38, figure_of_merit=1.0)
    T, A = 5 * 9.81 / 4, np.pi * 0.19 ** 2
    assert p1 == pytest.approx(4 * T ** 1.5 / np.sqrt(2 * 1.225 * A))
    # P ~ m^1.5 and ~ 1/D at fixed mass
    assert hover_power(10, 0.38) / hover_power(5, 0.38) == pytest.approx(2 ** 1.5)
    assert hover_power(5, 0.76) / hover_power(5, 0.38) == pytest.approx(0.5)


def test_forward_flight_reduces_to_hover_and_has_bucket():
    assert forward_flight_power(0, 5, 0.38) == pytest.approx(hover_power(5, 0.38))
    V = np.linspace(0, 30, 61)
    P = np.array([forward_flight_power(v, 5, 0.38) for v in V])
    k = np.argmin(P)
    assert 0 < k < len(V) - 1          # minimum-power speed is interior
    assert P[k] < 0.8 * P[0]


def test_energy_per_mile():
    assert energy_per_mile_wh(3600.0, 1609.344) == pytest.approx(1.0)
    assert energy_per_mile_wh(0, 1, overhead_s=3600, overhead_power_w=5) == pytest.approx(5)


def test_surrogate_data_fit_vs_physics():
    plain = fit_surrogate(lam_div=0.0)
    pinn = fit_surrogate(lam_div=100.0)
    assert plain["data_rmse"] < 0.01
    # continuity penalty kills the divergence without hurting the data fit
    assert pinn["continuity_rms"] < plain["continuity_rms"] / 100
    assert pinn["data_rmse"] < 1.5 * plain["data_rmse"]
    # ... but cannot fix momentum, which is nonlinear in the coefficients
    assert pinn["momentum_rms"] > 0.5 * plain["momentum_rms"]


def test_surrogate_validation():
    with pytest.raises(ValueError):
        fit_surrogate(deg=0)
    with pytest.raises(ValueError):
        fit_surrogate(lam_div=-1)


def test_dispersion_is_imaginary_viscosity():
    r_diff, r_disp, same = check_analogy()
    assert r_diff == 0 and r_disp == 0 and same == 0
