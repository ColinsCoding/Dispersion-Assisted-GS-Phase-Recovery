"""Tests for dgs/pulse_moments_group_delay.py."""
import numpy as np
import pytest

from dgs.gs_core import disperse
from dgs.pulse_moments_group_delay import (
    chirped_pulse, group_delay_fft, group_delay_logdiff_fd, group_delay_unwrap,
    wrapped_error, temporal_moments, spectral_moments, predicted_sigma2,
    chirp_from_two_traces, quadratic_from_three_traces, chirp_estimate_spread,
    silica_index, group_index_exact, group_index_fd,
    material_dispersion_ps_nm_km, group_velocity,
)

N = 8192
NU = np.fft.fftfreq(N)


def test_delay_sign_convention():
    E = chirped_pulse(T0=20, nu0=0.05)
    for D in (1000, -1000, 3000):
        m, _ = temporal_moments(np.abs(disperse(E, D)) ** 2)
        assert m - N / 2 == pytest.approx(-D * 0.05, abs=1e-6)


@pytest.mark.parametrize("shape,chirp,cubic", [("gauss", 0, 0), ("gauss", 3, 0),
                                               ("sech", 3, 1), ("sech", -2, 4)])
def test_moment_identity_any_shape(shape, chirp, cubic):
    E = chirped_pulse(chirp=chirp, cubic=cubic, shape=shape)
    sm = spectral_moments(E)
    _, s0 = temporal_moments(np.abs(E) ** 2)
    for D in (-5000, -1200, 0, 900, 4000):
        m, s2 = temporal_moments(np.abs(disperse(E, D)) ** 2)
        # exact for band-limited fields; the sech+cubic case leaks ~1e-4 of its
        # spectrum to Nyquist, costing ~1e-7 relative
        assert s2 == pytest.approx(predicted_sigma2(D, s0, sm["cov_nu_tau"], sm["var_nu"]),
                                   rel=1e-6)
        assert m == pytest.approx(sm["mean_tau"] - D * sm["mean_nu"], rel=1e-9)


def test_two_traces_recover_chirp_exactly():
    E = chirped_pulse(chirp=3.0, cubic=1.0, shape="sech")
    sm = spectral_moments(E)
    I1, I2 = (np.abs(disperse(E, D)) ** 2 for D in (-5000, -5750))
    s0, cov = chirp_from_two_traces(I1, I2, -5000, -5750, sm["var_nu"])
    assert cov == pytest.approx(sm["cov_nu_tau"], rel=1e-8)
    assert s0 == pytest.approx(temporal_moments(np.abs(E) ** 2)[1], rel=1e-8)


def test_gaussian_chirp_slope_analytic():
    C, T0 = 3.0, 40.0
    E = chirped_pulse(T0=T0, chirp=C)
    sm = spectral_moments(E)
    b = sm["cov_nu_tau"] / sm["var_nu"]
    assert b == pytest.approx(2 * np.pi * T0 ** 2 * C / (1 + C ** 2), rel=1e-6)
    # minimum width at D = b (the compressing dispersion)
    Ds = np.linspace(b - 500, b + 500, 201)
    widths = [temporal_moments(np.abs(disperse(E, D)) ** 2)[1] for D in Ds]
    assert Ds[int(np.argmin(widths))] == pytest.approx(b, abs=5.0)


def test_three_traces_without_spectrum():
    E = chirped_pulse(chirp=-2.0, shape="sech")
    sm = spectral_moments(E)
    Ds = [-5000, -5750, -3000]
    s0, cov, var_nu = quadratic_from_three_traces(
        [np.abs(disperse(E, D)) ** 2 for D in Ds], Ds)
    assert var_nu == pytest.approx(sm["var_nu"], rel=1e-6)
    assert cov == pytest.approx(sm["cov_nu_tau"], rel=1e-6)


def test_group_delay_estimators_pure_gvd():
    G = chirped_pulse(T0=40)
    for D in (3000, 80000):
        Ed = disperse(G, D)
        Ak = np.abs(np.fft.fft(Ed))
        w = Ak > 1e-3 * Ak.max()
        truth = N / 2 - D * NU[w]
        assert np.max(np.abs(wrapped_error(group_delay_fft(Ed)[w], truth, N))) < 1e-6
        assert np.max(np.abs(wrapped_error(group_delay_unwrap(Ed, t_ref=N / 2)[w], truth, N))) < 1e-6
    # finite-difference log-derivative degrades with D
    e_small = np.max(np.abs(wrapped_error(group_delay_logdiff_fd(disperse(G, 3000), N / 2)[w],
                                          N / 2 - 3000 * NU[w], N)))
    e_big = np.max(np.abs(wrapped_error(group_delay_logdiff_fd(disperse(G, 80000), N / 2)[w],
                                        N / 2 - 80000 * NU[w], N)))
    assert e_big > 100 * e_small


def test_unwrap_fails_without_reference():
    G = chirped_pulse(T0=40)                     # centred: pi phase per bin
    Ed = disperse(G, 3000)
    Ak = np.abs(np.fft.fft(Ed))
    w = Ak > 1e-3 * Ak.max()
    err = wrapped_error(group_delay_unwrap(Ed)[w], N / 2 - 3000 * NU[w], N)
    assert np.max(np.abs(err)) > 1000


def test_noise_spread_scales_with_diversity():
    E = chirped_pulse(chirp=3.0, cubic=1.0, shape="sech")
    s_small, _, _ = chirp_estimate_spread(E, -5000, -5750, n_trials=60, seed=1)
    s_big, _, _ = chirp_estimate_spread(E, -5000, -10000, n_trials=60, seed=1)
    assert 3 < s_small / s_big < 15             # ~ |dD| ratio 5000/750 = 6.7


def test_threshold_trades_variance_for_bias():
    E = chirped_pulse(chirp=3.0, cubic=1.0, shape="sech")
    s_raw, _, true = chirp_estimate_spread(E, -5000, -10000, n_trials=60, seed=2)
    s_th, m_th, _ = chirp_estimate_spread(E, -5000, -10000, n_trials=60, threshold=5, seed=2)
    assert s_th < s_raw / 10
    assert abs(m_th - true) > 5 * s_th           # biased


def test_silica_numbers():
    assert float(silica_index(1.55)) == pytest.approx(1.444, abs=1e-3)
    assert float(group_index_exact(1.55)) == pytest.approx(1.4626, abs=2e-4)
    assert float(material_dispersion_ps_nm_km(1.55)) == pytest.approx(21.9, abs=0.3)
    zdw = np.linspace(1.2, 1.35, 3001)
    z = zdw[np.argmin(np.abs(material_dispersion_ps_nm_km(zdw)))]
    assert 1.26 < z < 1.29
    assert float(group_velocity(1.55)) == pytest.approx(299_792_458 / 1.4626, rel=2e-4)


def test_finite_difference_u_curve():
    err = lambda h: abs(float(group_index_fd(1.55, h) - group_index_exact(1.55)))
    assert err(1e-5) < err(1e-1) and err(1e-5) < err(1e-9)


def test_validation():
    with pytest.raises(ValueError):
        chirped_pulse(shape="square")
    with pytest.raises(ValueError):
        chirp_from_two_traces(np.ones(4), np.ones(4), 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        temporal_moments(-np.ones(4))
    with pytest.raises(ValueError):
        group_index_fd(1.55, 0.0)
