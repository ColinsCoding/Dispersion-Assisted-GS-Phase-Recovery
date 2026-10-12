"""Smoke tests for dgs/scinet_discovery.py."""
import numpy as np
import pytest
import sympy as sp

pytest.importorskip("torch")
from dgs import scinet_discovery as S


def test_symbolic():
    ratio, transverse = S.doppler_symbolic()
    beta = sp.Symbol("beta", real=True)
    assert sp.simplify(transverse - sp.sqrt(1 - beta**2)) == 0
    assert S.damped_oscillator_symbolic()[1] == 0


def test_data_matches_formulas():
    d = S.make_data("uniform motion", n=50, seed=3)
    p = d["params"]
    assert np.allclose(d["a"][:, 0], p[:, 0] + p[:, 1] * d["q"][:, 0], atol=1e-6)
    d = S.make_data("relativistic Doppler", n=50, seed=3)
    b, th = d["params"][:, 0], d["q"][:, 0]
    assert np.allclose(d["a"][:, 0], np.sqrt(1 - b**2) / (1 - b * np.cos(th)), atol=1e-5)
    d = S.make_data("damped oscillator", n=5, seed=3)
    assert d["obs"].shape == (5, 50) and np.allclose(d["obs"][:, 0], 1.0)


def test_bounds():
    with pytest.raises(ValueError):
        S.make_data("pendulum")
    with pytest.raises(ValueError):
        S.make_data("uniform motion", n=0)
    with pytest.raises(ValueError):
        S.train("uniform motion", steps=0)


def test_short_training_finds_beta():
    model, _, _ = S.train("relativistic Doppler", steps=800, n=4000)
    r = S.latent_report(model, S.make_data("relativistic Doppler", n=1000, seed=7))
    assert r["r2"][0] > 0.95
    assert r["mse"] < 0.01
