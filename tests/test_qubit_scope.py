import math

import numpy as np
import pytest

from dgs import qubit_scope as q

TWO_PI = 2 * math.pi


@pytest.mark.parametrize("Delta", [0.0, TWO_PI * 0.3e6, TWO_PI * 2e6])
def test_rabi_closed_form_matches_schrodinger(Delta):
    t = np.linspace(0, 3e-6, 61)
    Om = TWO_PI * 1e6
    assert np.allclose(q.rabi_probability(t, Om, Delta), q.simulate_rabi(t, Om, Delta), atol=1e-12)


def test_pi_pulse_and_detuned_contrast():
    Om = TWO_PI * 1e6
    assert q.rabi_probability(math.pi / Om, Om) == pytest.approx(1.0)
    assert q.rabi_probability(np.linspace(0, 1e-5, 2001), Om, Om).max() == pytest.approx(0.5, abs=1e-4)


@pytest.mark.parametrize("T2", [math.inf, 1.5e-6])
def test_ramsey_closed_form_matches_density_matrix(T2):
    tau = np.linspace(0, 4e-6, 41)
    D = TWO_PI * 1.2e6
    assert np.allclose(q.ramsey_probability(tau, D, T2), q.simulate_ramsey(tau, D, T2), atol=1e-12)


def test_rabi_experiment_fit_recovers_omega():
    d = np.linspace(0, 2e-6, 41)
    r = q.rabi_experiment(d, amplitude=0.5, kappa=TWO_PI * 4e6, n_shots=800, rng=1)
    Om, c0, c1 = q.fit_rabi(r["durations"], r["v_mean"])
    assert Om == pytest.approx(r["Omega"], rel=0.02)
    assert c1 == pytest.approx(1e-3, rel=0.15)


def test_voltages():
    f = q.drive_voltage(0.3, 100e-9, 50e6, t_start=20e-9)
    assert f(10e-9) == 0 and f(20e-9) == pytest.approx(0.3) and f(130e-9) == 0
    shots = q.single_shots(1.0, 2000, v0=0, v1=1e-3, sigma=1e-4, rng=0)
    assert shots.mean() == pytest.approx(1e-3, rel=0.01)
    tr = q.readout_trace(1e-3, 0, 1e-6, tau_ring=50e-9)
    assert tr(5e-7) == pytest.approx(1e-3, rel=1e-3)
    with pytest.raises(ValueError):
        q.single_shots(1.5, 10)
    with pytest.raises(ValueError):
        q.rabi_probability(1e-6, -1.0)
