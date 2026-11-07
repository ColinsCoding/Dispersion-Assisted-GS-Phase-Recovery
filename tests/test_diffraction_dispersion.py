"""Tests for dgs.diffraction_dispersion (space-time duality)."""
import numpy as np
import pytest

from dgs import diffraction_dispersion as dd
from dgs.dispersive_fourier import gvd_transfer_function


def _grid(n=4096, span=200.0):
    d = span / n
    return (np.arange(n) - n // 2) * d, d


def test_gaussian_fft_matches_exact():
    x, d = _grid()
    u0 = dd.gaussian_field(x, 1.0)
    for p in (0.5, 3.0, -7.0):
        assert np.allclose(dd.quadratic_phase_propagate(u0, d, p), dd.gaussian_exact(x, 1.0, p), atol=1e-9)


def test_width_law_and_characteristic_length():
    x, d = _grid()
    s0 = 1.5
    ps = np.linspace(0, 20, 9)
    u = dd.quadratic_phase_propagate(dd.gaussian_field(x, s0), d, ps)
    assert u.shape == (9, x.size)
    assert np.allclose(np.sqrt(2) * dd.rms_width(x, u), dd.gaussian_width(s0, ps), rtol=1e-6)
    assert np.isclose(dd.gaussian_width(s0, dd.characteristic_p(s0)), np.sqrt(2) * s0)


def test_energy_conserved():
    x, d = _grid()
    u0 = dd.gaussian_field(x, 1.0) * (1 + 0.3 * x)
    u = dd.quadratic_phase_propagate(u0, d, 12.0)
    assert np.isclose(np.sum(np.abs(u) ** 2), np.sum(np.abs(u0) ** 2))


def test_same_transfer_function_as_gvd_module():
    nu = np.linspace(-5, 5, 101)
    beta2, L = -2.17e-26, 1000.0
    assert np.allclose(dd.transfer_function(nu * 1e12, dd.dispersion_p(beta2, L)),
                       gvd_transfer_function(nu * 1e12, beta2, L))


def test_far_field_is_fourier_transform():
    x, d = _grid(8192, 4000.0)
    u0 = dd.gaussian_field(x - 2, 0.7) + 0.5 * dd.gaussian_field(x + 3, 0.4)   # asymmetric
    p = 400.0
    I = np.abs(dd.quadratic_phase_propagate(u0, d, p)) ** 2
    pred = dd.far_field_prediction(u0, d, x, p)
    assert dd.fresnel_number(5.0, p) < 0.1
    assert np.max(np.abs(I - pred)) < 0.05 * I.max()


def test_talbot_revival_and_half_shift():
    n, period = 1024, 8.0
    d = 32 * period / n
    x = np.arange(n) * d
    u0 = (np.abs(((x % period) - period / 2)) < 1.0).astype(complex)
    pT = dd.talbot_p(period)
    assert np.allclose(dd.quadratic_phase_propagate(u0, d, pT), u0, atol=1e-9)
    shifted = np.roll(u0, int(round(period / 2 / d)))
    assert np.allclose(dd.quadratic_phase_propagate(u0, d, pT / 2), shifted, atol=1e-9)


def test_talbot_lengths():
    assert np.isclose(dd.talbot_length_diffraction(100e-6, 1550e-9), 2 * (100e-6) ** 2 / 1550e-9)
    assert np.isclose(dd.talbot_length_dispersion(100e-12, -2.17e-26), (100e-12) ** 2 / (np.pi * 2.17e-26))


def test_duality_mapping():
    assert np.isclose(dd.diffraction_p(2.0, 1e-6), -2.0 / (2 * np.pi / 1e-6))
    assert np.isclose(dd.dispersion_p(-2e-26, 10.0), -2e-25)


def test_direct_loop_vectorized_and_fft_agree():
    n = 256
    x = (np.arange(n) - n // 2) * 0.1
    u0 = dd.gaussian_field(x, 1.0)
    p = 0.3
    loop = dd.fresnel_direct_loop(u0, x, p)
    vec = dd.fresnel_direct_vectorized(u0, x, p)
    exact = dd.gaussian_exact(x, 1.0, p)
    assert np.allclose(loop, vec, atol=1e-10)
    assert np.allclose(vec, exact, atol=2e-3)


def test_sympy_results():
    r = dd.sympy_duality()
    S = r["symbols"]
    assert r["residual"] == 0
    assert r["nu_star"] == -S["x"] / S["p"]
    assert r["z_R"] == S["k"] * S["s0"] ** 2
    assert r["L_D"] == S["T0"] ** 2 / S["beta2"]
    assert (r["z_T"] - 2 * S["Lam"] ** 2 / S["lam"]).equals(0)


@pytest.mark.parametrize("call", [
    lambda: dd.gaussian_field(0.0, 0.0),
    lambda: dd.quadratic_phase_propagate([1.0], 0.1, 1.0),
    lambda: dd.quadratic_phase_propagate(np.ones(8), -1.0, 1.0),
    lambda: dd.quadratic_phase_propagate(np.ones(8), 0.1, np.ones((2, 2))),
    lambda: dd.talbot_p(0.0),
    lambda: dd.diffraction_p(1.0, 0.0),
    lambda: dd.dispersion_p(1e-26, -1.0),
    lambda: dd.fresnel_kernel(1.0, 0.0),
])
def test_validation(call):
    with pytest.raises(ValueError):
        call()
