"""Tests for dgs/probability_ce.py: symbolic derivations, physical numbers, Monte Carlo agreement."""
import math

import pytest
import sympy as sp

from dgs import probability_ce as P


def test_binomial_limit_is_poisson():
    _, limit = P.binomial_to_poisson()
    k, lam = sp.symbols("k lambda", positive=True)
    assert sp.simplify(limit - lam**k * sp.exp(-lam) / sp.factorial(k)) == 0
    mean, var = P.poisson_moments()
    assert sp.simplify(mean - sp.Symbol("lambda", positive=True)) == 0 and sp.simplify(var - mean) == 0


def test_physical_numbers():
    # 1 uW at 10 Gb/s, 1550 nm: about 780 photons per bit
    assert P.photons_per_bit(1e-6, 10e9, 1550e-9) == pytest.approx(780, rel=0.01)
    assert P.shot_noise_snr(100) == 10
    # 1 kOhm, 300 K, 1 MHz: about 4.07 uV rms
    assert P.johnson_sigma(300, 1e3, 1e6) == pytest.approx(4.07e-6, rel=0.01)
    with pytest.raises(ValueError):
        P.johnson_sigma(-1, 1e3, 1e6)


def test_ber_derivation_matches_numeric():
    _, closed = P.ber_derivation()
    A, s = sp.Symbol("A", positive=True), sp.Symbol("sigma", positive=True)
    assert float(closed.subs({A: 1, s: 0.4})) == pytest.approx(P.ber(1, 0.4), rel=1e-12)
    assert P.ber(1, 0.4) == pytest.approx(6.21e-3, rel=0.01)


def test_packets_retries_memoryless_fermi():
    exact, bound = P.packet_error(1e-4, 1500)
    assert exact <= bound and exact == pytest.approx(1 - math.exp(-0.15), rel=1e-3)
    with pytest.raises(ValueError):
        P.packet_error(1.5, 10)
    assert sp.simplify(P.expected_transmissions() - 1 / (1 - sp.Symbol("q", positive=True))) == 0
    assert P.memoryless() == 0 and P.fermi_symmetry() == 0
    assert P.fermi(0.5, 0.5, 300) == pytest.approx(0.5)
    assert P.fermi(0.6, 0.5, 300) + P.fermi(0.4, 0.5, 300) == pytest.approx(1.0)


def test_monte_carlo_agrees():
    for name, (formula, sim) in P.monte_carlo().items():
        if name.startswith("Markov"):
            assert sim <= formula
        else:
            assert sim == pytest.approx(formula, rel=0.05), name
