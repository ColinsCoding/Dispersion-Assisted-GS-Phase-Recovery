"""Tests for dgs/quantum_receivers.py."""
import numpy as np
import pytest
import sympy as sp

from dgs.quantum_receivers import (
    homodyne_error, heterodyne_error, kennedy_error, displaced_counting_error,
    optimized_kennedy, optimized_kennedy_condition, overlap, helstrom_error,
    kennedy_helstrom_ratio_limit, coherent_state, helstrom_error_fock,
    helstrom_measurement_check, mc_homodyne, mc_heterodyne, mc_displaced_counting,
    displaced_counting_error_imperfect, dolinar_error, mc_dolinar,
    squeezed_homodyne_error, optimal_squeezing,
)


def _mc_close(mc, p, trials):
    return abs(mc - p) < 4 * np.sqrt(p * (1 - p) / trials) + 1e-12


@pytest.mark.parametrize("N", [0.2, 1.0, 2.0])
def test_monte_carlo_matches_closed_forms(N):
    T = 400_000
    assert _mc_close(mc_homodyne(N, T, seed=1), homodyne_error(N), T)
    assert _mc_close(mc_heterodyne(N, T, seed=2), heterodyne_error(N), T)
    assert _mc_close(mc_displaced_counting(N, np.sqrt(N), T, seed=3), kennedy_error(N), T)


def test_coherent_state_normalised_and_overlap():
    for a in (0.0, 0.7, 2.0):
        v = coherent_state(a)
        assert np.vdot(v, v).real == pytest.approx(1.0, abs=1e-12)
    a = 1.3
    assert np.vdot(coherent_state(a), coherent_state(-a)).real == pytest.approx(overlap(a ** 2), rel=1e-10)


@pytest.mark.parametrize("N", [0.1, 1.0, 4.0])
def test_helstrom_two_ways(N):
    assert helstrom_error_fock(N) == pytest.approx(helstrom_error(N), rel=1e-8)
    err, closed = helstrom_measurement_check(N)
    assert err == pytest.approx(closed, rel=1e-8)


def test_ordering_of_receivers():
    for N in (0.5, 1.0, 2.0, 3.0):
        H = helstrom_error(N)
        assert H < optimized_kennedy(N)[1] <= kennedy_error(N) + 1e-15
        assert homodyne_error(N) < heterodyne_error(N)
        assert H < homodyne_error(N)
    # Kennedy beats homodyne only above a crossover photon number
    assert kennedy_error(0.1) > homodyne_error(0.1)
    assert kennedy_error(1.0) < homodyne_error(1.0)


def test_kennedy_ratio_limit_is_two():
    assert kennedy_helstrom_ratio_limit() == 2


def test_optimized_kennedy_stationarity():
    N = 1.0
    b, P = optimized_kennedy(N)
    assert b > np.sqrt(N)                                # over-displace
    cond = optimized_kennedy_condition()
    a_, b_ = sp.symbols("alpha beta", positive=True)
    assert abs(float(cond.subs({a_: np.sqrt(N), b_: b}))) < 1e-8
    assert P == pytest.approx(float(displaced_counting_error(N, b)))


def test_dolinar_one_slice_is_optimized_kennedy_and_improves():
    N = 1.0
    assert dolinar_error(N, 1) == pytest.approx(optimized_kennedy(N)[1], rel=1e-8)
    errs = [dolinar_error(N, M) for M in (1, 2, 4, 8)]
    assert all(e2 < e1 for e1, e2 in zip(errs, errs[1:]))
    assert errs[-1] < 1.25 * helstrom_error(N)
    assert errs[-1] > helstrom_error(N)                 # never beats the bound


def test_dolinar_monte_carlo():
    N, M, T = 0.5, 6, 60_000
    assert _mc_close(mc_dolinar(N, M, trials=T, seed=4), dolinar_error(N, M), T)


def test_imperfect_detector():
    N = 1.0
    b = np.sqrt(N)
    assert displaced_counting_error_imperfect(N, b) == pytest.approx(kennedy_error(N))
    worse = displaced_counting_error_imperfect(N, b, eta=0.8, dark=0.01)
    assert worse > kennedy_error(N)
    T = 400_000
    assert _mc_close(mc_displaced_counting(N, b, T, dark=0.01, eta=0.8, seed=5), worse, T)


def test_squeezing_helps_homodyne_at_fixed_energy():
    N = 1.0
    assert squeezed_homodyne_error(N, 0.0) == pytest.approx(homodyne_error(N))
    r, P = optimal_squeezing(N)
    assert r > 0 and P < homodyne_error(N)
    rs = np.linspace(0, np.arcsinh(1.0) - 1e-6, 400)
    assert P <= np.nanmin([squeezed_homodyne_error(N, x) for x in rs]) + 1e-12


def test_validation():
    with pytest.raises(ValueError):
        homodyne_error(-1)
    with pytest.raises(ValueError):
        dolinar_error(1.0, 0)
    with pytest.raises(ValueError):
        mc_displaced_counting(1.0, 1.0, eta=0)
