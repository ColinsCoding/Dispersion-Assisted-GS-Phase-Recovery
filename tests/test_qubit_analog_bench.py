import math

import numpy as np
import pytest
import sympy as sp

from dgs import qubit_analog_bench as b
from dgs.circuit_rules import Circuit, impedance, s, _char_poly

L, C = 36e-6, 3.3e-9


def test_identical_modes_closed_form():
    for k in (0.02, 0.1, 0.3):
        wl, wh = b.normal_modes(L, L, C, C, k)
        w0 = 1 / math.sqrt(L * C)
        assert wl == pytest.approx(w0 / math.sqrt(1 + k)) and wh == pytest.approx(w0 / math.sqrt(1 - k))
        assert b.k_from_mode_split(wl, wh) == pytest.approx(k)


def test_modes_from_circuit_axioms():
    # KCL/KVL solver: the coupled tanks' characteristic polynomial has roots s^2 = -1/(LC(1 +/- k))
    Ls, Cs, k = sp.symbols("L C k", positive=True)
    c = Circuit().C("C1", "a", "0", Cs).L("L1", "a", "0", Ls).C("C2", "b", "0", Cs).L("L2", "b", "0", Ls).K("K", "L1", "L2", k)
    p = _char_poly(impedance(c, "a"))
    x = sp.Symbol("x")
    roots = sp.solve(sp.Poly(p.as_expr().subs(s ** 2, x).subs(s ** 4, x ** 2), x).as_expr(), x) if p.degree() == 4 else []
    expect = {sp.simplify(-1 / (Ls * Cs * (1 + k))), sp.simplify(-1 / (Ls * Cs * (1 - k)))}
    assert p.degree() == 4
    assert {sp.simplify(r) for r in roots} == expect


def test_simulation_conserves_energy_without_loss():
    t = np.linspace(0, 100e-6, 2001)
    sim = b.simulate(t, L, C, C, 0.0, 0.1)
    E = sim["E1"] + sim["E2"]
    assert np.allclose(E, E[0], rtol=1e-9)


def test_rabi_analogy_resonant():
    k = 0.03
    t = np.linspace(0, 150e-6, 30001)
    sim = b.simulate(t, L, C, C, 0.4, k)
    frac = sim["E2"] / (sim["E1"] + sim["E2"])
    w0 = 1 / math.sqrt(L * C)
    assert np.max(np.abs(frac - b.rabi_prediction(t, k, w0, w0))) < 0.05      # coupled-mode theory, error ~ k
    assert frac.max() > 0.97


def test_rabi_analogy_detuned_contrast_drops():
    k, t = 0.05, np.linspace(0, 200e-6, 40001)
    w1 = 1 / math.sqrt(L * C)
    maxima = []
    for d in (0.0, 0.05, 0.1, 0.2):
        C2 = C * (1 + d)
        sim = b.simulate(t, L, C, C2, 0.4, k)
        fr = sim["E2"] / (sim["E1"] + sim["E2"])
        pred = b.rabi_prediction(t, k, w1, 1 / math.sqrt(L * C2)).max()
        assert fr.max() == pytest.approx(pred, abs=0.06)
        maxima.append(fr.max())
    assert all(a > b_ for a, b_ in zip(maxima, maxima[1:]))


def test_k_recovered_from_noisy_trace():
    rng = np.random.default_rng(2)
    t = np.arange(0, 200e-6, 1 / 10e6)
    sim = b.simulate(t, L, C, C, 0.6, 0.08)
    v = sim["v1"] + 0.002 * rng.standard_normal(t.size)
    fl, fh = b.fft_peaks(t, v, 2, 3e5, 7e5)
    assert b.k_from_mode_split(fl, fh) == pytest.approx(0.08, abs=0.003)


def test_coil_design_numbers():
    cp = b.coil_pair_coupling(0.02, 30, 0.02, 0.255e-3, 0.02)
    assert 30e-6 < cp["L"] < 45e-6 and 0.05 < cp["k"] < 0.12
    assert b.coil_pair_coupling(0.02, 30, 0.02, 0.255e-3, 0.005)["k"] > cp["k"]
    assert b.wire_resistance(0.02, 30, 0.255e-3) == pytest.approx(0.31, rel=0.02)
    assert b.wire_resistance(0.02, 30, 0.255e-3, 5e5) > b.wire_resistance(0.02, 30, 0.255e-3)


def test_probe_and_kick():
    p = b.probe_loading(L, C)
    assert -0.003 < p["shift_frac"] < 0 and p["Q_probe"] > 1e4
    assert b.capacitive_kick(10, 10e-12, 3.3e-9) == pytest.approx(10 * 10e-12 / 3.31e-9)
    with pytest.raises(ValueError):
        b.k_from_mode_split(5e5, 4e5)
