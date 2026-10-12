"""Tests for dgs.nv_magnetometer."""
import numpy as np
import pytest
import sympy as sp

from dgs import nv_magnetometer as nv
from dgs import transimpedance_amplifier as tia


def test_spin1_algebra():
    comm, casimir = nv.spin1_algebra_check()
    assert comm == sp.zeros(3) and casimir == sp.zeros(3)


def test_odmr_lines_axial():
    lines = nv.odmr_frequencies_symbolic()
    D, g, Bz = sp.symbols("D gamma_e B_z", real=True)
    assert set(lines) == {D - g * Bz, D + g * Bz}
    f = nv.odmr_frequencies([0, 0, 1e-3])
    assert f == pytest.approx([nv.D_ZFS - nv.GAMMA_E * 1e-3, nv.D_ZFS + nv.GAMMA_E * 1e-3], rel=1e-9)


def test_transverse_field_only_shifts_second_order():
    f = nv.odmr_frequencies([1e-3, 0, 0])
    assert abs(f[1] - f[0]) < 1e6                       # no first-order splitting
    assert f.mean() > nv.D_ZFS                          # small upward shift ~ (gamma B)^2 / D


def test_lorentzian_slope_and_sensitivity():
    f_star, slope = nv.max_slope_symbolic()
    C, G, g, R = sp.symbols("C Gamma gamma_e R", positive=True)
    assert sp.simplify(f_star - G / (2 * sp.sqrt(3))) == 0
    assert sp.simplify(slope - 3 * sp.sqrt(3) * C / (4 * G)) == 0
    assert sp.simplify(nv.sensitivity_symbolic() - 4 * G / (3 * sp.sqrt(3) * g * C * sp.sqrt(R))) == 0
    # numeric slope check
    ff = np.linspace(-5e6, 5e6, 200001)
    num = np.max(np.abs(np.gradient(nv.lorentzian_dip(ff, 0, 0.1, 2e6), ff)))
    assert num == pytest.approx(3 * np.sqrt(3) * 0.1 / (4 * 2e6), rel=1e-3)
    assert nv.sensitivity(1e6, 0.3, 1e5) == pytest.approx(2.9e-7, rel=0.01)
    with pytest.raises(ValueError):
        nv.sensitivity(1e6, 1.5, 1e5)


def test_vector_magnetometry():
    f = np.linspace(2.70e9, 3.04e9, 6801)
    B = np.array([0.6e-3, -1.1e-3, 1.5e-3])
    pr = nv.projections_from_dips(nv.find_dips(f, nv.simulate_odmr(B, f, rng=0)))
    assert np.allclose(np.sort(pr), np.sort(np.abs(nv.NV_AXES @ B)), atol=3e-5)
    assert nv.symmetry_equivalent(nv.vector_from_projections(pr), B)
    small, bias = np.array([0.10e-3, -0.15e-3, 0.05e-3]), np.array([3.0e-3, 1.2e-3, 0.4e-3])
    est = nv.recover_vector_with_bias(small, bias, f, rng=1)
    assert np.allclose(est, small, atol=2e-5)


def test_mounts():
    m = nv.mount_symbolic()
    p, L, d, n = sp.symbols("p L d n", positive=True)
    assert sp.simplify(m["spot_shift"] - 2 * d * n * p / L) == 0
    assert nv.spot_shift(1, 100, 0.0254, 1.0) == pytest.approx(0.02, rel=1e-9)
    assert nv.thermal_spot_drift(1.0, 0.02, 0.025, 1.0) == pytest.approx(2 * 5.8e-6 * 0.02 / 0.025, rel=1e-6)
    assert nv.fluorescence_vs_offset(0, 1e-5) == 1.0
    assert nv.fluorescence_vs_offset(1e-5, 1e-5) == pytest.approx(np.exp(-2))
    with pytest.raises(ValueError):
        nv.tpi_to_pitch_m(0)


def test_tia_symbolic_and_flat_cf():
    H, w0, Q = nv.tia_closed_loop_symbolic()
    Rf, Cf, Cd, wt, s = sp.symbols("R_f C_f C_d omega_t s", positive=True)
    assert sp.limit(H, s, 0) == -Rf                    # DC transimpedance = -R_f
    Rv, Cdv, fg = 1e6, 50e-12, 10e6
    cf = nv.tia_flat_cf(Rv, Cdv, fg)
    f = np.logspace(2, 8, 20000)
    Hm = nv.tia_response(f, Rv, cf, Cdv, fg)
    assert Hm.max() / Rv == pytest.approx(1.0, abs=1e-3)           # no peaking
    assert tia.phase_margin(Rv, Cdv, cf, fg) == pytest.approx(65.5, abs=1.0)
    approx = tia.compensation_capacitor(Rv, Cdv, fg)
    assert approx < cf and nv.tia_response(f, Rv, approx, Cdv, fg).max() / Rv > 1.1   # peaks


def test_shot_noise_condition():
    R = nv.shot_noise_limited_rf(10e-6)
    assert 10e-6 * R == pytest.approx(2 * nv.K_B * 300 / nv.Q_E)
    b = nv.readout_noise_budget(10e-6, 700, 1e6, 50e-12, 1e4)
    assert b["shot"] > b["thermal"] > b["amplifier"]
