"""Tests for dgs/complex_eigen_bridge.py."""
import numpy as np
import pytest
from scipy import special

from dgs.complex_eigen_bridge import (
    rlc_state_matrix, rlc_poles, rlc_impulse_current_residues, rlc_impulse_current_expm,
    bessel_j_contour, bessel_i_from_j, bessel_zeros_scipy, bessel_zeros_matrix,
    count_zeros_argument_principle, corral_energies_eV, corral_density,
    clamped_plate_lambdas, clamped_plate_frequencies, machining_cost,
)

L_, C_ = 1e-3, 1e-6


@pytest.mark.parametrize("R", [10.0, 2 * np.sqrt(L_ / C_), 200.0])
def test_poles_are_eigenvalues(R):
    ev = np.sort_complex(np.linalg.eigvals(rlc_state_matrix(R, L_, C_)))
    assert np.allclose(ev, np.sort_complex(rlc_poles(R, L_, C_)), rtol=1e-6)


@pytest.mark.parametrize("R", [10.0, 63.0, 2 * np.sqrt(L_ / C_), 64.0, 200.0])
def test_residue_contour_equals_matrix_exponential(R):
    t = np.array([0.0, 1e-4, 3e-4, 1e-3, 3e-3])
    a = rlc_impulse_current_residues(R, L_, C_, t)
    b = rlc_impulse_current_expm(R, L_, C_, t)
    assert np.allclose(a, b, rtol=1e-9, atol=1e-9 * np.max(np.abs(b)))
    assert a[0] == pytest.approx(1 / L_)            # i(0+) = 1/L for a voltage impulse


def test_bessel_contour_exponential_convergence():
    exact = special.jv(3, 5.0)
    errs = [abs(bessel_j_contour(3, 5.0, n) - exact) for n in (12, 16, 24, 32)]
    assert errs[-1] < 1e-14
    assert errs[2] < errs[1] / 1e5                 # exponential, not algebraic
    for n_ord, x in ((0, 2.4), (1, 7.0), (5, 3.3)):
        assert bessel_j_contour(n_ord, x, 64) == pytest.approx(special.jv(n_ord, x), abs=1e-14)


def test_modified_bessel_is_j_on_imaginary_axis():
    x = np.linspace(0.1, 6, 25)
    for n in range(4):
        assert np.allclose(bessel_i_from_j(n, x), special.iv(n, x), rtol=1e-12)


def test_three_ways_to_bessel_zeros():
    z = bessel_zeros_scipy(0, 5)
    assert np.allclose(bessel_zeros_matrix(0, 5), z, rtol=1e-5)
    assert count_zeros_argument_principle(0, 8.0, 6.0) == int(np.sum(np.abs(z - 8.0) < 6.0))


def test_corral_energies():
    E = corral_energies_eV(7.13e-9)
    assert [(m, n) for _, m, n in E[:4]] == [(0, 1), (1, 1), (2, 1), (0, 2)]
    E01 = E[0][0]
    assert E01 == pytest.approx(0.0114, rel=0.01)
    # E ~ 1/a^2
    assert corral_energies_eV(2 * 7.13e-9)[0][0] == pytest.approx(E01 / 4)


def test_corral_density_vanishes_at_wall():
    assert corral_density(0, 1, np.array([1.0]), np.array([0.0]))[0] == pytest.approx(0, abs=1e-25)


def test_clamped_plate_tabulated_eigenvalues():
    assert np.allclose(clamped_plate_lambdas(0, 3) ** 2, [10.2158, 39.771, 89.104], rtol=1e-4)
    assert np.allclose(clamped_plate_lambdas(1, 2) ** 2, [21.26, 60.82], rtol=1e-3)


def test_plate_frequency_scaling():
    f1 = clamped_plate_frequencies(0.05, 0.002)[0]
    assert clamped_plate_frequencies(0.05, 0.004)[0] == pytest.approx(2 * f1)   # f ~ h
    assert clamped_plate_frequencies(0.10, 0.002)[0] == pytest.approx(f1 / 4)   # f ~ 1/a^2


def test_machining_cost():
    c = machining_cost(1e-4, 5e-5, mrr_cm3_per_min=10, shop_rate_per_hr=100,
                       al_price_per_kg=10, setup_hr=0.5, finishing_factor=1.0)
    assert c["material"] == pytest.approx(1e-4 * 2700 * 10)
    assert c["machine_hours"] == pytest.approx(0.5 + 50 / 10 / 60)
    assert c["total"] == pytest.approx(c["material"] + c["machine"])
    with pytest.raises(ValueError):
        machining_cost(1e-4, 2e-4)


def test_validation():
    with pytest.raises(ValueError):
        rlc_state_matrix(0, 1, 1)
    with pytest.raises(ValueError):
        bessel_j_contour(0, 1.0, 2)
    with pytest.raises(ValueError):
        corral_energies_eV(-1)
