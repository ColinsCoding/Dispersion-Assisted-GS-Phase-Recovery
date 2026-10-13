"""Tests for dgs/receivers.py: the fiber model and the three receivers."""
import numpy as np
import pytest

from dgs import receivers as R

TAPS = np.array([-2500, -5000, -10000, -20000, -40000.0])/(2*np.pi*R.FS**2)
L = 20e3
B2L = R.BETA2*L
A_LOSS = np.sqrt(10**(-R.ALPHA_DB_KM*L/1000/10))


def frames(n, seed=0):
    rng = np.random.default_rng(seed)
    return [list(rng.integers(0, 4, 6)) for _ in range(n)], rng


def test_dispersion_is_all_pass_and_undoes():
    rng = np.random.default_rng(1)
    E = rng.standard_normal(R.N) + 1j*rng.standard_normal(R.N)
    D = R.disperse(E, B2L)
    assert np.isclose(np.sum(np.abs(D)**2), np.sum(np.abs(E)**2))
    assert np.allclose(R.disperse(D, -B2L), E)


def test_fiber_loss_matches_db_per_km():
    E = R.transmit([0, 1, 2, 3, 0, 1], 1e-3)
    out = R.fiber(E, L, nonlinear=True)
    assert np.isclose(np.mean(np.abs(out)**2)/np.mean(np.abs(E)**2), A_LOSS**2, rtol=1e-6)   # Kerr keeps power


def test_transmitter_is_constant_envelope_and_periodic():
    E = R.transmit([0, 3, 1, 2, 2, 0], 1e-3)
    assert np.allclose(np.abs(E)**2, 1e-3)
    ph = R.smooth_phase(R.PILOT + [0, 3, 1, 2, 2, 0])
    assert abs(np.angle(np.exp(1j*(ph[0] - ph[-1])))) < 0.1          # wraps smoothly back to the start


@pytest.mark.parametrize('name', ['coherent', 'kk', 'gs'])
def test_noise_free_link_decodes(name):
    data_list, rng = frames(10)
    for data in data_list:
        want = R.PILOT + data
        if name == 'coherent':
            got = R.rx_coherent(R.fiber(R.transmit(data, 1e-3), L, nonlinear=False), B2L, R.RS)
        elif name == 'kk':
            E, ca, sa = R.kk_tx(data, 1e-3, 10, R.RS)
            got = R.rx_kk(R.fiber(E, L, nonlinear=False), B2L, ca*A_LOSS, sa*A_LOSS, R.RS, R.RS)
        else:
            E_rx = R.fiber(R.transmit(data, 1e-3), L, nonlinear=False)
            got, _, _ = R.rx_gs(R.gs_branches(E_rx, TAPS, 0.0, rng, R.RS), TAPS, B2L, rng)
        assert list(got) == want


def test_kk_signal_is_minimum_phase():
    E, ca, sa = R.kk_tx([0, 1, 2, 3, 3, 2], 1e-3, 10, R.RS)
    assert np.max(np.abs(E - ca)) < ca                                  # carrier stronger than the signal
