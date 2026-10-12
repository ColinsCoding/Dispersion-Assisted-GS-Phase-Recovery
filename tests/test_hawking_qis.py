"""Tests for dgs.hawking_qis: Hawking/Bekenstein SymPy derivations and qubit models."""
import numpy as np
import pytest
import sympy as sp

from dgs import hawking_qis as h
from dgs.hawking_qis import Gs, Ms, cs, hb, kB


def test_surface_gravity_and_temperature():
    assert sp.simplify(h.surface_gravity_symbolic() - cs**4 / (4 * Gs * Ms)) == 0
    T = h.hawking_temperature_symbolic()
    assert sp.simplify(T - hb * cs**3 / (8 * sp.pi * Gs * Ms * kB)) == 0
    # Hawking = Unruh with a = kappa
    a = sp.symbols("a", positive=True)
    assert sp.simplify(h.unruh_temperature_symbolic().subs(a, h.surface_gravity_symbolic()) - T) == 0


def test_euclidean_regularity_gives_period():
    rho, coeff, beta = h.euclidean_period_symbolic()
    fp = sp.Symbol("fprime", positive=True)
    assert sp.simplify(beta - 4 * sp.pi / (cs * fp)) == 0


def test_first_law_gives_bekenstein_hawking():
    S, bh, diff = h.entropy_from_first_law()
    assert diff == 0


def test_evaporation_formulas_match_numerics():
    dMdt, life = h.evaporation_symbolic()
    M0 = sp.Symbol("M_0", positive=True)
    assert sp.simplify(life - 5120 * sp.pi * Gs**2 * M0**3 / (cs**4 * hb)) == 0
    # Stefan-Boltzmann numerically
    M = 1e12
    P = h.SIGMA_SB * 4 * np.pi * h.schwarzschild_radius(M) ** 2 * h.hawking_temperature(M) ** 4
    assert P == pytest.approx(h.evaporation_power(M), rel=1e-6)


def test_reference_numbers():
    assert h.hawking_temperature(h.M_SUN) == pytest.approx(6.17e-8, rel=0.01)
    assert h.entropy_bits(h.M_SUN) == pytest.approx(1.5e77, rel=0.02)
    assert h.lifetime_s(h.M_SUN) / h.YEAR == pytest.approx(2.1e67, rel=0.02)
    assert h.mass_for_temperature(h.hawking_temperature(3.0e22)) == pytest.approx(3.0e22)
    assert h.lifetime_s(h.mass_for_lifetime(1e17)) == pytest.approx(1e17)
    assert h.unruh_acceleration_for(1.0) == pytest.approx(2.47e20, rel=0.01)
    with pytest.raises(ValueError):
        h.hawking_temperature(0)


def test_entanglement_entropy_basics():
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    assert h.entanglement_entropy_bits(bell, 1, 2) == pytest.approx(1)
    assert h.entanglement_entropy_bits(np.kron([1, 0], [0, 1]), 1, 2) == pytest.approx(0)
    U = h.haar_unitary(8, rng=1)
    assert np.allclose(U.conj().T @ U, np.eye(8))


def test_page_curve_turns_over():
    k, sim, page, hawking = h.page_curve(10, n_samples=10)
    assert np.allclose(sim, page, atol=0.05)
    assert np.argmax(sim) == 5 and sim[-1] == pytest.approx(0, abs=1e-9)
    assert hawking[-1] == 10                      # thermal guess never comes back down
    assert h.page_entropy_bits(5, 5) == pytest.approx(5 - 1 / (2 * np.log(2)), abs=0.01)


def test_hayden_preskill_diary_escapes_quickly():
    k, I = h.hayden_preskill(n_bh=5, n_emit_max=4)
    assert I[0] == pytest.approx(0, abs=1e-9)    # before emission: no info outside
    assert I[3] > 1.9 and np.all(np.diff(I) > -1e-9)


def test_timeline_sorted_and_in_range():
    years = [y for y, _, _ in h.QIS_TIMELINE]
    assert years == sorted(years) and 2002 <= years[0] and years[-1] <= 2026
