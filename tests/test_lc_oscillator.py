"""Tests for dgs/lc_oscillator.py: LC = harmonic oscillator, quantum levels, THz limits, torch residuals."""
import math

import numpy as np
import pytest

from dgs import lc_oscillator as O

# A transmon-like resonator: L = 10 nH, C = 100 fF -> f0 = 5.03 GHz, Z = 316 ohm
L, C = 10e-9, 100e-15


def test_params_match_hand_values():
    p = O.lc_params(L, C)
    assert p["f0"] == pytest.approx(5.033e9, rel=1e-3)
    assert p["Z"] == pytest.approx(316.2, rel=1e-3)
    with pytest.raises(ValueError):
        O.lc_params(-1, C)


def test_quantum_levels_equally_spaced_and_fridge_scale():
    q = O.quantum_levels(L, C, 3)
    d = np.diff(q["E_J"])
    assert np.allclose(d, d[0]) and d[0] == pytest.approx(O.HBAR * O.lc_params(L, C)["omega0"])
    assert q["E_J"][0] == pytest.approx(d[0] / 2)                 # zero-point energy
    assert q["E_K"][1] - q["E_K"][0] == pytest.approx(0.2416, rel=1e-3)   # 5 GHz ~ 0.24 K
    assert O.thermal_occupation(L, C, 0.010) < 1e-9               # 10 mK: ground state
    assert O.thermal_occupation(L, C, 300) > 1000                 # room temperature: classical


def test_zero_point_at_uncertainty_limit():
    z = O.zero_point(L, C)
    assert z["product_over_hbar"] == pytest.approx(0.5)
    assert 1 < z["dQ_electrons"] < 10                             # a few electrons of charge noise


@pytest.mark.parametrize("R, conserved", [(0.0, True), (20.0, False)])
def test_leapfrog_energy(R, conserved):
    s = O.simulate_lc(L, C, Q0=1e-15, periods=200, R=R)
    if conserved:
        assert np.max(np.abs(s["E"] / s["E"][0] - 1)) < 1e-3     # bounded, no drift
        w = O.lc_params(L, C)["omega0"]                          # and it rings at 1/sqrt(LC)
        assert np.allclose(s["Q"][:400], 1e-15 * np.cos(w * s["t"][:400]), atol=1e-17)
    else:
        tau = L / R                                              # energy decays as exp(-R t / L)
        ratio = s["E"][-200:].mean() / s["E"][:200].mean()
        assert ratio == pytest.approx(math.exp(-(s["t"][-100] - s["t"][100]) / tau), rel=0.02)


def test_lumped_limit_at_terahertz():
    assert O.wavelength(1e12) == pytest.approx(299.8e-6, rel=1e-3)
    assert O.lumped_ok(1e9, 1e-3)["lumped_ok"]                     # 1 mm part at 1 GHz: fine
    assert not O.lumped_ok(1e12, 1e-3)["lumped_ok"]                # same part at 1 THz: a waveguide
    assert O.average_power(2.0, 0.5, math.pi / 2) == pytest.approx(0)
    ts = O.time_stretch(20e9, 50, 1e-9)
    assert ts["effective_bandwidth_Hz"] == pytest.approx(1e12) and ts["captured_window_s"] == pytest.approx(2e-11)


def test_costs_are_sane():
    for name, lo, hi in O.COSTS:
        assert 0 <= lo <= hi, name


@pytest.mark.parametrize("prefer", ["cpu", "cuda"])
def test_torch_residual_zero_for_true_solution(prefer):
    torch = pytest.importorskip("torch")
    dev = O.torch_device(prefer)
    w = O.lc_params(L, C)["omega0"]
    t = np.linspace(0, 4 * 2 * math.pi / w, 300)
    good = O.lc_residual(lambda tt: torch.cos(w * tt), L, C, t, dev)
    bad = O.lc_residual(lambda tt: torch.cos(1.1 * w * tt), L, C, t, dev)
    assert good.abs().max().item() < 1e-9
    assert bad.abs().max().item() == pytest.approx(1.1**2 - 1, rel=1e-6)


def test_fit_omega_recovers_resonance():
    pytest.importorskip("torch")
    r = O.fit_omega(L, C)
    assert r["omega_fit"] == pytest.approx(r["omega_true"], rel=1e-3)
    assert r["loss"][-1] < 1e-5 * r["loss"][0]
