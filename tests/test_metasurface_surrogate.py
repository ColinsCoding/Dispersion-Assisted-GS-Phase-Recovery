"""Tests for dgs/metasurface_surrogate.py: the solver is checked against closed forms first."""
import numpy as np
import pytest

from dgs import metasurface_surrogate as ms


def test_energy_conservation():
    d = [80, 120, 60, 150]
    wl = np.linspace(500, 900, 25)
    t, r = ms.tmm(d, wl)
    assert np.allclose(np.abs(t) ** 2 + np.abs(r) ** 2, 1.0, atol=1e-12)


def test_single_slab_matches_airy():
    n, d, wl = 2.0, 150.0, np.linspace(500, 900, 30)
    T = ms.transmission([d], wl, n_layers=[n])
    delta = 2 * np.pi * n * d / wl
    r12 = (1 - n) / (1 + n)
    expected = (1 - r12 ** 2) ** 2 / (1 - 2 * r12 ** 2 * np.cos(2 * delta) + r12 ** 4)
    assert np.allclose(T, expected, atol=1e-10)


def test_half_wave_slab_is_transparent():
    n, wl0 = 2.0, 600.0
    d = wl0 / (2 * n)                  # half-wave: slab disappears
    assert ms.transmission([d], [wl0], n_layers=[n])[0] == pytest.approx(1.0, abs=1e-12)


def test_quarter_wave_mirror_reflects():
    wl0 = 700.0
    d = [wl0 / (4 * ms.N_HI), wl0 / (4 * ms.N_LO)] * 6
    assert ms.transmission(d, [wl0])[0] < 0.05


def test_group_delay_positive_through_slab():
    tau = ms.group_delay([200.0], np.linspace(600, 800, 50), n_layers=[2.0])
    # order of magnitude: optical path n*d/c = 1.3 fs, enhanced by internal reflections
    assert 0.5 * (2.0 * 200e-9 / ms.C0) < np.mean(tau) < 5 * (2.0 * 200e-9 / ms.C0)


@pytest.mark.parametrize('bad', [[], [-1.0], [[1.0]]])
def test_bad_thickness_rejected(bad):
    with pytest.raises(ValueError):
        ms.tmm(bad, [600.0])


def test_surrogate_learns_and_runs_faster_than_solver():
    pytest.importorskip('torch')
    import time
    X, Y, wl = ms.make_dataset(900, n_layers=4, wavelengths_nm=np.linspace(550, 850, 32))
    s = ms.Surrogate(X.shape[1], Y.shape[1], hidden=96, depth=3).fit(X[:700], Y[:700], epochs=250)
    m = s.evaluate(X[700:], Y[700:])
    assert m['rmse'] < 0.08
    baseline = float(np.sqrt(np.mean((Y[700:] - Y[:700].mean(0)) ** 2)))
    assert m['rmse'] < 0.6 * baseline          # clearly better than predicting the mean
    t0 = time.perf_counter(); [ms.transmission(x, wl) for x in X[700:]]; t_solver = time.perf_counter() - t0
    t0 = time.perf_counter(); s.predict(X[700:]); t_net = time.perf_counter() - t0
    assert t_net < t_solver
