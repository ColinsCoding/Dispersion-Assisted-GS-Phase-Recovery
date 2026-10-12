"""Tests for dgs.symmetry_phase_limits."""
import numpy as np
import pytest
import sympy as sp

from dgs import symmetry_phase_limits as s


def test_symmetries():
    dE, dB = s.em_gauge_check()
    assert dE == sp.zeros(3, 1) and dB == sp.zeros(3, 1)
    assert s.charge_conservation_from_maxwell() == 0
    assert s.global_phase_check() == 0
    assert s.local_gauge_schrodinger_check() == 0
    assert s.global_only_breaks_without_A() != 0
    free, pot = s.noether_commutators()
    assert free == 0
    f, V = sp.Function("f")(s.x), sp.Function("V")(s.x)
    assert sp.simplify(pot - sp.I * s.hbar * f * sp.diff(V, s.x)) == 0


def test_phase():
    vp, vg, diff = s.phase_group_velocity_symbolic()
    k, c = sp.symbols("k c", positive=True)
    assert sp.simplify(diff.subs(sp.Function("n")(k), 1.5).doit()) == 0
    assert np.allclose(s.relative_phase_interference([0, np.pi]), [1, 0])
    b = s.qubit_global_vs_relative_symbolic()
    assert not b.has(sp.Symbol("gamma", real=True))


def test_limits():
    r, th, p, eps0 = sp.symbols("r theta p epsilon_0", positive=True)
    assert sp.simplify(s.dipole_limit_symbolic() - p * sp.cos(th) / (4 * sp.pi * eps0 * r**2)) == 0
    area, sift, sift_lim, peak = s.gaussian_delta_limit_symbolic()
    assert area == 1 and sift_lim == 1 and peak == sp.oo
    assert s.correspondence_limit_symbolic()[1] == 0
    lim, U = s.infinitesimal_rotation_limit()
    th2 = sp.symbols("theta", real=True)
    assert sp.simplify(lim - sp.exp(-sp.I * th2)) == 0
    errs = [s.rotation_power_numeric(np.pi / 2, n)[1] for n in (100, 1000)]
    assert errs[1] < errs[0] / 5


def test_check_limits_tool():
    g, l, T = sp.symbols("g l theta_0", positive=True)
    # large-amplitude pendulum correction factor 1 + theta0^2/16 -> 1 as theta0 -> 0
    res = s.check_limits(1 + T**2 / 16, T, [(0, 1)])
    assert res[0][2]


def test_discovery_latent_dimension():
    X, x0, v, tt = s.projectile_observations()
    assert s.latent_dimension(X)[0] == 2
    Xn, *_ = s.projectile_observations(noise=0.01)
    assert s.latent_dimension(Xn, 1e-2)[0] == 2
    pred, a, b = s.predict_from_two_numbers(X, tt, 1.5)
    assert np.allclose(pred, x0 + v * 1.5 - 0.5 * 9.81 * 1.5**2)
