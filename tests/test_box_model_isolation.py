"""Tests for dgs.box_model_isolation."""
import numpy as np
import pytest
import sympy as sp

from dgs import box_model_isolation as b

RB = 86.909 * b.AMU


def test_probability_symbolic_numeric_and_classical_limit():
    P = b.box_probability_sym()
    assert sp.limit(P, b.n_s, sp.oo) == (b.b_s - b.a_s) / b.L_s
    for n, a, bb in [(1, 0, 0.25), (2, 0.1, 0.7), (7, 0.3, 0.31)]:
        assert np.isclose(float(P.subs({b.n_s: n, b.a_s: a, b.b_s: bb, b.L_s: 1})), b.box_probability(n, a, bb))
    assert np.isclose(b.box_probability(2, 0, 0.25), 0.25)
    assert np.isclose(b.box_probability(1, 0, 0.25), 0.25 - 1 / (2 * np.pi))


def test_nodes_and_spread():
    assert np.allclose(b.box_nodes(4), [0.25, 0.5, 0.75])
    for n in (1, 3):
        psi = sp.lambdify(b.x_s, b.box_psi_sym(n, 1))
        assert np.allclose(psi(b.box_nodes(n)), 0, atol=1e-12)
    mx, mx2, dx = b.box_moments_sym()
    assert mx == b.L_s / 2
    assert np.isclose(float(dx.subs({b.n_s: 3, b.L_s: 1})), b.box_delta_x(3))
    assert np.isclose(b.box_delta_x(10 ** 6), 1 / np.sqrt(12))


def test_correspondence_spacing_goes_to_zero():
    assert b.relative_spacing(1) == 3 and b.relative_spacing(1000) < 0.003


def test_64_level_box_is_addressable_harmonic_is_not():
    assert b.min_gap_separation(b.transition_gaps("box", 64)) == 2.0
    assert b.min_gap_separation(b.transition_gaps("harmonic", 64)) == 0.0


def test_co_lines_harmonic_vs_morse():
    hv, an = b.vib_band_origins(b.CO["we"], b.CO["wexe"])
    assert np.isclose(an[0], 2143.24, atol=0.05)          # CO fundamental ~2143.27 cm^-1
    assert np.isclose(an[1], 4259.9, atol=0.5)            # first overtone ~4260 cm^-1
    dop = b.doppler_hwhm_cm(an[1], b.CO["mass_amu"])
    assert 0.003 < dop < 0.01
    assert (hv[1] - an[1]) / dop > 1e4


def test_harmonic_and_box_limits_of_the_solver():
    lv = b.tweezer_levels(1e-3 * b.K_B, 1e-6, RB, 4)
    om = b.tweezer_models(1e-3 * b.K_B, 1e-6, RB, 4)
    assert np.allclose(lv, om["harmonic_plus_quartic"], rtol=1e-3)


@pytest.mark.parametrize("order,expected", [(2, "harmonic_plus_quartic"), (16, "box"), (40, "box")])
def test_pick_model_follows_the_potential_shape(order, expected):
    best, scores, fitted = b.pick_model(1e-4 * b.K_B, 1e-6, RB, 6, order=order)
    assert best == expected
    if order == 40:
        assert scores["box"] < 0.02 and 1.5e-6 < fitted["box_L"] < 2.1e-6


def test_single_stage_transmissibility():
    assert np.isclose(b.transmissibility(np.sqrt(2), 0.0), 1.0)
    assert np.isclose(b.transmissibility(0.0, 0.3), 1.0)
    assert b.transmissibility(10, 0.0) < 0.011
    assert np.isclose(b.transmissibility(1.0, 0.05), np.sqrt(1 + 0.01) / 0.1)
    r = b.required_f0_single(0.01, 10.0)
    assert np.isclose(b.transmissibility(10.0 / r, 0.0), 0.01)


def test_two_stage_symbolic_matches_numeric_and_rolls_off_r4():
    d = b.two_stage_sym()
    S = d["symbols"]
    m1 = m2 = 1.0
    k = (2 * np.pi * 2) ** 2
    c = 2 * 0.05 * np.sqrt(k)
    f = 37.0
    Hs = complex(d["H"].subs({S["m1"]: m1, S["m2"]: m2, S["k1"]: k, S["k2"]: k, S["c1"]: c, S["c2"]: c,
                              S["s"]: 2j * np.pi * f}))
    assert np.isclose(abs(Hs), b.two_stage_transmissibility(f, 2, 2, 0.05, 0.05))
    hi = b.two_stage_transmissibility([100.0, 1000.0], 2, 2, 0.0, 0.0)
    assert 1e3 < hi[0] / hi[1] < 2e4                       # ~ (10)^4 per decade, -80 dB/decade


@pytest.mark.parametrize("call", [
    lambda: b.box_probability(0, 0, 0.5),
    lambda: b.box_probability(1, 0.6, 0.5),
    lambda: b.box_nodes(0),
    lambda: b.transition_gaps("morse"),
    lambda: b.tweezer_levels(-1, 1e-6, RB),
    lambda: b.pick_model(1e-30, 1e-6, RB, 1),
    lambda: b.transmissibility(1.0, -0.1),
    lambda: b.required_f0_single(1.5),
])
def test_validation(call):
    with pytest.raises(ValueError):
        call()
