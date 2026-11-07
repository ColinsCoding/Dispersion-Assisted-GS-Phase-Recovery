"""Tests for dgs.paul_trap (RF quadrupole ion trap) and dgs.scope_bench (simulated oscilloscope)."""
import math

import numpy as np
import pytest
import sympy as sp

from dgs import paul_trap as pt
from dgs.scope_bench import Scope


# ---------------------------------------------------------------- SymPy derivation
def test_potential_solves_laplace():
    assert pt.derive_mathieu()["laplacian"] == 0


def test_mathieu_parameters():
    d = pt.derive_mathieu()
    e, m, r0, U, V, Om = sp.symbols("e m r_0 U V Omega", positive=True)
    assert sp.simplify(d["x"]["a"] - 4 * e * U / (m * r0 ** 2 * Om ** 2)) == 0
    assert sp.simplify(d["x"]["q"] - 2 * e * V / (m * r0 ** 2 * Om ** 2)) == 0
    assert sp.simplify(d["y"]["a"] + d["x"]["a"]) == 0 and sp.simplify(d["y"]["q"] + d["x"]["q"]) == 0


# ---------------------------------------------------------------- Floquet stability
def test_first_stability_edge():
    assert pt.q_max() == pytest.approx(0.908046, abs=2e-4)


def test_secular_frequency_small_q():
    assert pt.floquet_beta(0.0, 0.1) == pytest.approx(0.1 / math.sqrt(2), rel=0.005)


def test_unstable_beyond_edge():
    assert math.isnan(pt.floquet_beta(0.0, 0.95))


# ---------------------------------------------------------------- a 40Ca+ qubit trap
def test_calcium_trap_numbers():
    p = pt.trap_parameters()
    assert 0.3 < p["q"] < 0.45
    assert 2.0 < p["secular_MHz"] < 3.5
    assert p["lamb_dicke_eta"] < 0.1                  # Lamb-Dicke regime
    assert p["pseudopotential_depth_eV"] == pytest.approx(p["q"] * 300 / 8, rel=1e-9)   # D = qV/8


def test_micromotion_at_rf_frequency():
    xi, u, _ = pt.ion_trajectory(0.0, 0.3, n_periods=60, steps_per_period=64)
    spec = np.abs(np.fft.rfft(u - u.mean()))
    freqs = np.fft.rfftfreq(u.size, d=(xi[1] - xi[0]) / math.pi)      # in units of the RF frequency
    sec = freqs[np.argmax(spec)]
    beta = pt.floquet_beta(0.0, 0.3)
    assert sec == pytest.approx(beta / 2, abs=0.01)                  # slow secular line at beta*Omega/2
    side = spec[(freqs > 1 - beta / 2 - 0.02) & (freqs < 1 - beta / 2 + 0.02)].max()
    assert side > 0.02 * spec.max()                                   # micromotion sideband near Omega


# ---------------------------------------------------------------- stability map: CPU, CUDA, Floquet
def test_cpu_map_matches_floquet():
    a = np.linspace(-0.3, 0.3, 9)
    q = np.linspace(0.05, 1.0, 11)
    assert np.mean(pt.stability_map_cpu(a, q) == pt.stability_map_floquet(a, q)) > 0.97


@pytest.mark.skipif(not pt.gpu_available(), reason="no CUDA GPU / CuPy")
def test_gpu_map_matches_cpu():
    a = np.linspace(-0.3, 0.3, 40)
    q = np.linspace(0.0, 1.0, 50)
    assert np.array_equal(pt.stability_map_gpu(a, q), pt.stability_map_cpu(a, q))


# ---------------------------------------------------------------- scope bench
def _sine(f, amp=1.0):
    return lambda t: amp * np.sin(2 * np.pi * f * t)


def test_scope_measures_frequency_and_vpp():
    s = Scope(bandwidth=500e6, sample_rate=2e9, bits=10)
    s.connect(1, "rf", _sine(20e6, 3.0), volts_per_div=1.0)
    m = s.measure(s.acquire(time_per_div=50e-9), 1)
    assert m["frequency"] == pytest.approx(20e6, rel=0.01)
    assert m["vpp"] == pytest.approx(6.0, rel=0.02)


def test_scope_aliases_below_nyquist():
    s = Scope(bandwidth=2e9, sample_rate=100e6, bits=10)   # 70 MHz tone, 100 MS/s -> appears at 30 MHz
    s.connect(1, "tone", _sine(70e6), volts_per_div=0.5)
    m = s.measure(s.acquire(time_per_div=200e-9), 1)
    assert m["frequency"] == pytest.approx(30e6, rel=0.03)


def test_scope_bandwidth_attenuates():
    s = Scope(bandwidth=20e6, sample_rate=2e9, bits=12)    # single pole: |H| = 1/sqrt(1 + (f/BW)^2)
    s.connect(1, "fast", _sine(100e6), volts_per_div=0.5)
    vpp = s.measure(s.acquire(time_per_div=50e-9), 1)["vpp"]
    assert vpp == pytest.approx(2 / math.sqrt(1 + 25), rel=0.08)


def test_trigger_puts_rising_edge_at_zero():
    s = Scope(bandwidth=1e9, sample_rate=5e9, bits=12)
    s.connect(1, "rf", _sine(20e6), volts_per_div=0.5)
    s.trigger(1, 0.0, "rising")
    shot = s.acquire(time_per_div=20e-9)
    i0 = np.argmin(np.abs(shot["t"]))
    assert abs(shot["ch"][1][i0]) < 0.05 and shot["ch"][1][i0 + 5] > shot["ch"][1][i0]


def test_adc_quantisation_and_clipping():
    s = Scope(bandwidth=1e9, sample_rate=2e9, bits=3)
    s.connect(1, "big", _sine(10e6, 10.0), volts_per_div=1.0)        # 20 Vpp into a +-4 V screen
    v = s.acquire(time_per_div=50e-9)["ch"][1]
    assert v.max() <= 4.0 + 1e-9 and len(np.unique(np.round(v, 6))) <= 2 ** 3 + 1
