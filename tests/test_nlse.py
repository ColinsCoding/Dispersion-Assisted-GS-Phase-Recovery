import numpy as np
from dgs import nlse


def test_gaussian_pulse_peak():
    t = np.linspace(-5, 5, 101)
    a = nlse.gaussian_pulse(t, t0=1.0)
    assert np.argmax(np.abs(a)) == 50


def test_nlse_propagate_conserves_energy_linear():
    t = np.linspace(-8, 8, 256)
    a0 = nlse.gaussian_pulse(t, t0=1.0, chirp=0.5)
    a1 = nlse.nlse_propagate(a0, t, z=5.0, beta2=-1.0, gamma=0.0, n_steps=100)
    e0 = np.sum(np.abs(a0) ** 2)
    e1 = np.sum(np.abs(a1) ** 2)
    assert np.isclose(e0, e1, rtol=1e-2)


def test_make_nlse_measurements_shapes():
    out = nlse.make_nlse_measurements(n_points=128, seed=0)
    assert out["I1"].shape == (128,)
    assert out["I2"].shape == (128,)
    assert np.all(out["I1"] >= 0)
    assert np.all(out["I2"] >= 0)


def test_make_nlse_measurements_diversity():
    out = nlse.make_nlse_measurements(n_points=128, z1=3.0, z2=7.0, seed=1)
    corr = np.corrcoef(out["I1"], out["I2"])[0, 1]
    assert corr < 0.999
