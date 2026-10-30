"""Tests for dgs/cryo_readout.py: Boltzmann/Bose-Einstein limits, RLC circuits, hanger S21, attenuators."""
import math

import numpy as np
import pytest

from dgs import cryo_readout as R

F = 5e9


# --- PHYS 106 ---------------------------------------------------------------------------
def test_quantum_temperature_5GHz():
    assert abs(R.quantum_temperature(F) - 0.23995) < 1e-4


def test_excited_population_values():
    assert abs(R.excited_population(F, 4.0) - 1 / (1 + math.exp(0.23995 / 4))) < 1e-5
    assert R.excited_population(F, 4.0) > 0.48                 # nearly scrambled at 4 K
    assert abs(R.excited_population(F, 0.020) - math.exp(-0.23995 / 0.020)) < 1e-7
    assert R.excited_population(F, 0.020) < 1e-5
    assert abs(R.excited_population(F, 1e6) - 0.5) < 1e-6      # high-T limit


def test_effective_temperature_roundtrip():
    for T in (0.01, 0.05, 0.3, 4.0):
        assert abs(R.effective_temperature(F, float(R.excited_population(F, T))) / T - 1) < 1e-9


def test_level_populations():
    p = R.level_populations(F, 0.05, n_levels=2)
    assert abs(p[1] - R.excited_population(F, 0.05)) < 1e-12
    assert abs(R.level_populations(F, 0.1, 4).sum() - 1) < 1e-12
    harm = R.level_populations(F, 0.1, 3)
    assert abs(harm[2] / harm[1] - harm[1] / harm[0]) < 1e-12    # equal ratios on a ladder
    tr = R.level_populations(F, 0.1, 3, anharmonicity_hz=-250e6)
    assert tr[2] > harm[2]                                       # level 2 sits lower -> more populated


def test_photon_number_limits():
    assert abs(R.mean_photon_number(F, 300.0) - (300 / R.quantum_temperature(F) - 0.5)) < 1e-3
    assert R.mean_photon_number(F, 0.010) < 1e-10
    # Bose-Einstein vs Boltzmann: n/(n+1) = exp(-hf/kT) = P1/P0
    T = 0.2
    n = R.mean_photon_number(F, T)
    p1 = R.excited_population(F, T)
    assert abs(n / (n + 1) - p1 / (1 - p1)) < 1e-12


# --- EEE 117 ----------------------------------------------------------------------------
def test_rlc_resonance_and_q():
    f0 = R.rlc_resonance(1e-9, 1e-12)
    assert abs(f0 - 5.0329e9) < 1e5
    Rs, L, C = 2.0, 1e-9, 1e-12
    Qs = R.series_rlc_q(Rs, L, C)
    assert abs(Qs - 2 * math.pi * f0 * L / Rs) < 1e-9
    assert abs(R.parallel_rlc_q(1e4, L, C) - 2 * math.pi * f0 * 1e4 * C) < 1e-9


def test_series_bandpass_peak_and_bandwidth():
    Rs, L, C = 10.0, 1e-3, 1e-7          # bench values: ~15.9 kHz
    f0 = R.rlc_resonance(L, C)
    Q = R.series_rlc_q(Rs, L, C)
    assert abs(abs(R.series_rlc_bandpass(f0, Rs, L, C)) - 1) < 1e-12
    f = np.linspace(0.5 * f0, 1.5 * f0, 400001)
    mag = np.abs(R.series_rlc_bandpass(f, Rs, L, C))
    band = f[mag >= 1 / math.sqrt(2)]
    assert abs((band[-1] - band[0]) / (f0 / Q) - 1) < 1e-3
    assert abs((band[-1] - band[0]) - Rs / (2 * math.pi * L)) < 1.0  # bandwidth = R / (2 pi L)


def test_bandpass_depends_only_on_f_over_f0_and_Q():
    # bench (kHz) and chip (GHz) resonators with the same Q draw the same normalized curve
    bench = (10.0, 1e-3, 1e-7)
    L2 = 1e-9
    C2 = 1 / ((2 * math.pi * 7e9) ** 2 * L2)
    R2 = math.sqrt(L2 / C2) / R.series_rlc_q(*bench)
    u = np.linspace(0.8, 1.2, 101)
    h1 = R.series_rlc_bandpass(u * R.rlc_resonance(*bench[1:]), *bench)
    h2 = R.series_rlc_bandpass(u * R.rlc_resonance(L2, C2), R2, L2, C2)
    assert np.allclose(h1, h2, atol=1e-12)


def test_hanger_s21():
    f0, Qi, Qc = 7e9, 1e5, 2e4
    assert abs(abs(R.hanger_s21(f0, f0, Qi, Qc)) - Qc / (Qi + Qc)) < 1e-12
    assert abs(abs(R.hanger_s21(f0 * 1.01, f0, Qi, Qc)) - 1) < 1e-3   # far off resonance
    s0, s1, sep = R.dispersive_readout(f0, f0, 0.5e6, Qi, Qc)
    # same depth up to the (f - f0)/f0 normalization (~chi/f0), opposite phase
    assert sep > 0.5 and abs(abs(s0) - abs(s1)) < 1e-4
    assert abs(s0.imag + s1.imag) < 1e-3


# --- attenuator chain -------------------------------------------------------------------
def test_attenuator_chain():
    n0 = R.mean_photon_number(F, 300.0)
    assert R.attenuator_chain(F, 300.0, [(0, 4.0)])[-1] == pytest.approx(n0)
    assert R.attenuator_chain(F, 300.0, [(200, 4.0)])[-1] == pytest.approx(R.mean_photon_number(F, 4.0))
    chain = R.attenuator_chain(F, 300.0, [(20, 4.0), (20, 0.1), (20, 0.015)])
    A = 100.0
    n1 = n0 / A + (1 - 1 / A) * R.mean_photon_number(F, 4.0)
    assert chain[1] == pytest.approx(n1)
    n2 = n1 / A + (1 - 1 / A) * R.mean_photon_number(F, 0.1)
    n3 = n2 / A + (1 - 1 / A) * R.mean_photon_number(F, 0.015)
    assert chain[-1] == pytest.approx(n3)
    assert chain == sorted(chain, reverse=True)
    assert 3e-3 < chain[-1] < 5e-3      # 60 dB still leaves ~4e-3 photons, mostly the 100 mK stage's


@pytest.mark.parametrize("call", [
    lambda: R.excited_population(F, 0.0),
    lambda: R.excited_population(-1.0, 1.0),
    lambda: R.effective_temperature(F, 0.6),
    lambda: R.level_populations(F, 0.1, n_levels=1),
    lambda: R.level_populations(F, 0.1, 3, anharmonicity_hz=-6e9),
    lambda: R.rlc_resonance(0.0, 1e-12),
    lambda: R.hanger_s21(7e9, 7e9, -1, 1e4),
    lambda: R.attenuator_chain(F, 300.0, [(-3, 4.0)]),
])
def test_validation(call):
    with pytest.raises(ValueError):
        call()
