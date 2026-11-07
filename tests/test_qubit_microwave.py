import math

import numpy as np
import pytest

from dgs import qubit_microwave as qm


def test_units():
    assert qm.dbm_to_w(0) == pytest.approx(1e-3)
    assert qm.w_to_dbm(1.0) == pytest.approx(30.0)
    assert qm.vpeak_from_dbm(10) == pytest.approx(1.0, rel=1e-3)          # 10 mW into 50 ohm -> 1 V peak
    with pytest.raises(ValueError):
        qm.w_to_dbm(0)


def test_thermal_photons():
    f = 5e9
    assert qm.thermal_photons(f, 300) == pytest.approx(300 * qm.KB / (qm.H * f) - 0.5, rel=1e-4)   # kT >> hf
    assert qm.thermal_photons(f, 0.015) < 1e-6
    assert qm.effective_temperature(qm.thermal_photons(f, 0.05), f) == pytest.approx(0.05)


@pytest.mark.parametrize("g,phi", [(1.0, 0.0), (1.05, 0.0), (1.0, 0.05), (0.9, -0.1)])
def test_image_rejection_matches_waveform(g, phi):
    t = np.arange(0, 1e-6, 1 / 40e9)
    fi, flo = 100e6, 5e9
    rf = qm.iq_upconvert(t, np.cos(2 * np.pi * fi * t), np.sin(2 * np.pi * fi * t), flo, g, phi)
    usb, lsb = qm.tone_amplitude(rf, t, flo + fi), qm.tone_amplitude(rf, t, flo - fi)
    if g == 1 and phi == 0:
        assert lsb < 1e-9 and usb == pytest.approx(1.0)
    else:
        assert 20 * np.log10(usb / lsb) == pytest.approx(qm.image_rejection_db(g, phi), abs=1e-6)


def test_lo_leakage():
    t = np.arange(0, 1e-6, 1 / 40e9)
    rf = qm.iq_upconvert(t, 0 * t, 0 * t, 5e9, 1.1, 0.2, dc_i=0.01, dc_q=-0.02)
    assert qm.tone_amplitude(rf, t, 5e9) == pytest.approx(qm.lo_leakage_amplitude(0.01, -0.02, 1.1, 0.2), rel=1e-9)


def test_drive_line_cools_the_noise():
    rows = qm.drive_line(-10, 5e9)
    assert rows[0]["n_th"] > 1000 and rows[-1]["n_th"] < 0.01
    assert rows[-1]["p_dbm"] == pytest.approx(-10 - sum(a for _, a, _ in qm.DEFAULT_DRIVE_LINE))
    ns = [r["n_th"] for r in rows]
    assert all(b <= a + 1e-12 for a, b in zip(ns, ns[1:]))
    with pytest.raises(ValueError):
        qm.drive_line(0, 5e9, [("gain?", -3, 4.0)])


def test_attenuator_alone_limit():
    # a huge attenuation at T leaves exactly the thermal occupation of T
    rows = qm.drive_line(0, 6e9, [("big", 200.0, 0.05)])
    assert rows[-1]["n_th"] == pytest.approx(qm.thermal_photons(6e9, 0.05), rel=1e-9)


def test_friis():
    assert qm.friis_temperature([("a", 20, 0.3), ("b", 40, 3.0), ("c", 30, 100.0)]) == pytest.approx(0.3 + 0.03 + 1e-4)
    assert qm.friis_temperature([("b", 40, 3.0)]) == pytest.approx(3.0)


def test_readout_fidelity_matches_monte_carlo():
    T, tau, p = 0.33, 200e-9, -125.0
    r = qm.readout_fidelity(p, tau, T)
    rng = np.random.default_rng(0)
    n = 200000
    d = r["d_over_sigma"]
    x0 = rng.standard_normal(n); x1 = d + rng.standard_normal(n)
    mc_err = 0.5 * (np.mean(x0 > d / 2) + np.mean(x1 < d / 2))
    assert mc_err == pytest.approx(r["error"], abs=3e-3)
    tau99 = qm.tau_for_fidelity(p, T, 0.99)
    assert qm.readout_fidelity(p, tau99, T)["fidelity"] == pytest.approx(0.99, abs=1e-6)
    assert qm.tau_for_fidelity(p, 3.0, 0.99) == pytest.approx(tau99 * 3.0 / 0.33, rel=1e-3)   # E needed ~ T


def test_stack_mirrors_osi():
    assert [L["n"] for L in qm.QUANTUM_STACK] == list(range(7, 0, -1))
    assert all("L" in L["osi"] for L in qm.QUANTUM_STACK)


@pytest.mark.parametrize("theta,axis,expect", [
    (math.pi, "x", (0, 0, -1)), (math.pi / 2, "x", (0, -1, 0)), (math.pi / 2, "y", (1, 0, 0)), (2 * math.pi, "x", (0, 0, 1))])
def test_compiled_pulse_rotates_correctly(theta, axis, expect):
    p = qm.compile_gate(theta, axis)
    assert np.allclose(qm.simulate_iq_pulse(p), expect, atol=1e-9)


def test_pulse_area_and_validation():
    p = qm.compile_gate(math.pi, sigma=5e-9, kappa=2 * math.pi * 20e6)
    area = p["kappa"] * np.sum(np.hypot(p["I"], p["Q"])) * (p["t"][1] - p["t"][0])
    assert area == pytest.approx(math.pi)
    assert p["I"][0] < 0.02 * p["amplitude_V"] and p["I"][-1] < 0.02 * p["amplitude_V"]
    with pytest.raises(ValueError):
        qm.compile_gate(1.0, "z")
