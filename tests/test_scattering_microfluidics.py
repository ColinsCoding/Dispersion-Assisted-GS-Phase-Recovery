"""Tests for dgs.scattering_microfluidics."""
import pathlib
import sys

import numpy as np
import pytest

from dgs import scattering_microfluidics as s

LAM = 532e-9


def test_gravity_flow_numbers():
    r = 0.25e-3
    R = s.tube_resistance(0.1, r)
    Q = s.gravity_flow(0.2, R)
    assert np.isclose(Q, 3.0e-8, rtol=0.01)                      # ~30 uL/s
    assert np.isclose(s.mean_velocity(Q, r), 0.153, rtol=0.01)
    assert 50 < s.channel_reynolds(Q, r) < 100                   # laminar
    tau = s.drain_time_constant(R, np.pi * 0.0075 ** 2)
    assert np.isclose(s.head_vs_time(0.2, tau, tau), 0.2 / np.e)


def test_poiseuille_profile_mean():
    r, vbar = 1.0, 2.0
    rr = np.linspace(0, r, 20001)
    v = s.poiseuille_velocity(rr, r, vbar)
    assert np.isclose(np.trapezoid(v * 2 * np.pi * rr, rr) / (np.pi * r ** 2), vbar, rtol=1e-6)
    assert v[0] == 2 * vbar and v[-1] == 0


def test_pulse_fwhm():
    w, v = 25e-6, 0.3
    t = np.linspace(-1e-3, 1e-3, 200001)
    p = s.pulse_shape(t, v, w)
    above = t[p >= 0.5]
    assert np.isclose(above[-1] - above[0], s.pulse_fwhm(w, v), rtol=1e-3)
    widths = s.transit_width_samples(2000, 0.25e-3, 0.15, w)
    assert widths.min() >= s.pulse_fwhm(w, 0.3) * 0.999            # centerline is fastest


def test_coincidence():
    Q, win = 3e-8, 3e-4
    c = s.max_concentration(Q, win, 0.01)
    assert np.isclose(s.coincidence_probability(c, Q, win), 0.01, rtol=1e-3)
    assert s.coincidence_probability(10 * c, Q, win) > 0.05


def test_mie_rayleigh_limit():
    m, x = 1.2, 0.05
    assert np.isclose(s.mie_efficiencies(m, x)[1], s.rayleigh_qsca(m, x), rtol=1e-3)


def test_mie_bohren_huffman_example():
    qext, qsca, qback = s.mie_efficiencies(1.55, 5.213)
    assert np.isclose(qsca, 3.105, atol=1e-3) and np.isclose(qext, qsca)   # lossless sphere
    assert np.isclose(qback, 2.92, atol=0.01)


def test_mie_soft_large_sphere_vs_anomalous_diffraction():
    # van de Hulst: Q_ext ~ 2 - (4/rho) sin rho + (4/rho^2)(1 - cos rho), rho = 2x(m-1);
    # approximate (no edge term), so a few-percent agreement is the expectation
    for m, x in [(1.02, 200.0), (1.05, 60.0)]:
        rho = 2 * x * (m - 1)
        ada = 2 - 4 / rho * np.sin(rho) + 4 / rho ** 2 * (1 - np.cos(rho))
        assert np.isclose(s.mie_efficiencies(m, x)[0], ada, rtol=0.05)


def test_angular_integral_recovers_csca():
    d = 3e-6
    x = s.size_parameter(d, LAM)
    csca = s.mie_efficiencies(s.N_POLYSTYRENE / s.N_WATER, x)[1] * np.pi * (d / 2) ** 2
    assert np.isclose(s.collected_cross_section(d, LAM, 0, np.pi, n_theta=4000), csca, rtol=1e-3)


def test_matches_scarlett_if_present():
    src = pathlib.Path(__file__).resolve().parents[1] / "scarlett-particle-tagging" / "src"
    if not src.exists():
        pytest.skip("scarlett sub-project not present")
    sys.path.insert(0, str(src))
    try:
        from scarlett import physics as P
    finally:
        sys.path.remove(str(src))
    for m, x in [(1.55, 5.213), (1.19, 17.7), (1.05, 0.8)]:
        ref = P.mie_efficiencies(m, x)
        q = s.mie_efficiencies(m, x)
        assert np.isclose(q[1], ref["qsca"], rtol=1e-10) and np.isclose(q[0], ref["qext"], rtol=1e-10)


def test_detector_chain():
    V = s.pulse_height_volts(1e-12, 1e-3, 25e-6, 0.3, 1e6)
    assert np.isclose(V, 2e-3 / (np.pi * 625e-12) * 1e-12 * 0.3 * 1e6)
    assert s.adc_counts(3.3) == 4095 and s.adc_counts(10.0) == 4095 and s.adc_counts(0.0) == 0
    assert np.isclose(s.tia_bandwidth(10e6, 1e6, 25e-12), 2.52e5, rtol=0.01)


def test_light_sheet_reduces_height_spread():
    hv = {1e-6: 0.1, 3e-6: 1.0}
    kw = dict(r=0.25e-3, vbar=0.15, w_flow=25e-6, rate_hz=50.0, seed=3)
    spot = s.simulate_events(4000, [3e-6], hv, w_cross=25e-6, **kw)
    sheet = s.simulate_events(4000, [3e-6], hv, w_cross=300e-6, **kw)
    cv = lambda a: a.std() / a.mean()
    assert cv(sheet["height"]) < 0.5 * cv(spot["height"])
    assert np.all(sheet["height"] <= 1.0 + 1e-12)
    assert np.isclose(s.peak_irradiance(1e-3, 25e-6, 300e-6) * 12, s.peak_irradiance(1e-3, 25e-6))


def test_adc_trace_peaks():
    ev = s.simulate_events(5, [3e-6], {3e-6: 1.0}, r=0.25e-3, vbar=0.15, w_flow=25e-6,
                           w_cross=1.0, rate_hz=100.0, seed=0)
    t, V, codes = s.adc_trace(ev, 500e3, 25e-6)
    assert codes.max() <= 4095
    assert V.max() >= 0.95 * ev["height"].max()            # coincident beads can only add (pile-up)
    gaps = np.diff(ev["t0"])
    lone = [i for i in range(len(gaps) + 1)
            if (i == 0 or gaps[i - 1] > 1e-3) and (i == len(gaps) or gaps[i] > 1e-3)]
    for i in lone:                                           # isolated pulses peak at their height
        sl = np.abs(t - ev["t0"][i]) < 2e-4
        assert np.isclose(V[sl].max(), ev["height"][i], rtol=0.02)


def test_budget_fits():
    b = s.budget_summary()
    assert b["within_cap"] and b["reserve"] > 50
    assert np.isclose(b["total"], b["subtotal"] + b["tax"] + b["shipping"])


@pytest.mark.parametrize("call", [
    lambda: s.tube_resistance(0.1, 0.0),
    lambda: s.gravity_flow(-1.0, 1e10),
    lambda: s.poiseuille_velocity(2.0, 1.0, 1.0),
    lambda: s.collected_cross_section(1e-6, LAM, 1.0, 0.5),
    lambda: s.collected_cross_section(1e-6, LAM, 0.1, 0.5, phi_fraction=0.0),
    lambda: s.adc_counts(1.0, bits=0),
    lambda: s.size_parameter(0.0, LAM),
])
def test_validation(call):
    with pytest.raises(ValueError):
        call()
