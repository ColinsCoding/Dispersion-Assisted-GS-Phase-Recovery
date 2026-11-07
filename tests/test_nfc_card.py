"""Tests for dgs/nfc_card.py (contactless card: coupling, tuning, voltage vs time, load modulation)."""
import math

import numpy as np
import pytest
import sympy as sp

from dgs import nfc_card as nc


def test_sympy_dipole_limit():
    assert nc.dipole_limit_check()


def test_sympy_neumann_matches_scipy():
    D = nc.derivation()
    s = D["symbols"]
    f = sp.lambdify((s["a"], s["b"], s["d"], s["mu0"]), D["M_exact"], "mpmath")
    for d in (0.01, 0.03, 0.1):
        assert float(f(0.035, 0.03, d, nc.MU0)) == pytest.approx(nc.mutual_inductance(0.035, 0.03, d), rel=1e-9)


def test_far_field_is_dipole():
    a, b, d = 0.035, 0.03, 0.5
    dipole = nc.MU0 * math.pi * a ** 2 * b ** 2 / (2 * (a ** 2 + d ** 2) ** 1.5)
    assert nc.mutual_inductance(a, b, d) == pytest.approx(dipole, rel=0.01)


def test_tuning_hits_13_56_MHz():
    C = nc.tuning_capacitor()
    assert 1 / (2 * math.pi * math.sqrt(nc.CARD["L"] * C)) == pytest.approx(13.56e6, rel=1e-12)
    assert 30e-12 < C < 80e-12                     # tens of pF, as on real cards


def test_iso_constants():
    assert nc.F_SUB == pytest.approx(847.5e3)
    assert nc.BIT_RATE == pytest.approx(105.94e3, rel=1e-3)


def test_field_in_iso_range_at_typical_distance():
    h_rms = nc.reader_field(0.03) / math.sqrt(2)
    assert nc.H_ISO[0] <= h_rms <= nc.H_ISO[1]


def test_linear_simulation_matches_phasor():
    # far enough that the regulator never clamps: the time-domain steady state equals the phasor solution
    sim = nc.simulate(d=0.12, bits=(1,))
    t = sim["t"]
    env = nc.envelope(sim["V_card"], t)
    steady = env[(t > 2.5e-6) & (t < 3.8e-6)].mean()
    assert steady == pytest.approx(nc.card_voltage_phasor(0.12), rel=0.03)


def test_regulator_clamps_close_to_the_reader():
    sim = nc.simulate(d=0.01, bits=(1,))
    assert nc.card_voltage_phasor(0.01) > 30          # unregulated it would be tens of volts
    assert np.abs(sim["V_card"]).max() < 6.0


def test_reader_pause_drains_the_card():
    sim = nc.simulate(d=0.03, bits=(0,))
    t = sim["t"]
    env = nc.envelope(sim["V_card"], t)
    assert env[(t > 5.8e-6) & (t < 6.0e-6)].max() < 0.3 * env[(t > 2e-6) & (t < 3.5e-6)].mean()


def test_load_modulation_round_trip():
    bits = (1, 0, 0, 1, 1, 0, 1)
    for d in (0.02, 0.05, 0.08):
        assert nc.decode(nc.simulate(d=d, bits=bits)) == list(bits)


def test_subcarrier_sidebands():
    sim = nc.simulate(d=0.03, bits=(1, 1, 0, 1))
    t, x = sim["t"], sim["V_reflected"]
    sel = t > sim["t_answer"]
    F = np.abs(np.fft.rfft(x[sel]))
    fr = np.fft.rfftfreq(sel.sum(), t[1] - t[0])
    peak = lambda f0: F[np.abs(fr - f0) < 60e3].max()
    side = min(peak(nc.FC - nc.F_SUB), peak(nc.FC + nc.F_SUB))
    assert side > 5 * peak(nc.FC - 2.3e6)                 # sidebands stand well above an empty frequency
