import numpy as np
import pytest
import sympy as sp
from dgs.antenna import (
    dipole_pattern, dipole_pattern_grid, dipole_directivity,
    half_wave_dipole, quarter_wave_transformer,
    array_factor, beam_width_3dB,
    fraunhofer_distance, link_budget, antenna_sympy_5,
)


# ── dipole pattern ────────────────────────────────────────────────────

def test_dipole_pattern_broadside_is_max():
    F_broadside = dipole_pattern(np.pi / 2)
    F_45 = dipole_pattern(np.pi / 4)
    assert F_broadside > F_45


def test_dipole_pattern_on_axis_near_zero():
    # theta = 0 and theta = pi are singularities; function returns 0 there
    assert dipole_pattern(0.0) == pytest.approx(0.0, abs=1e-6)
    assert dipole_pattern(np.pi) == pytest.approx(0.0, abs=1e-6)


def test_dipole_pattern_grid_normalized():
    r = dipole_pattern_grid()
    assert np.max(r["F_norm"]) == pytest.approx(1.0)


def test_dipole_directivity_known_value():
    r = dipole_directivity()
    assert r["directivity_linear"] == pytest.approx(1.64, abs=0.02)
    assert r["directivity_dBi"] == pytest.approx(2.15, abs=0.1)


# ── half_wave_dipole ──────────────────────────────────────────────────

def test_half_wave_dipole_length():
    d = half_wave_dipole(2.4e9)
    c = 2.99792458e8
    expected_length = c / (2 * 2.4e9)
    assert d["length_m"] == pytest.approx(expected_length, rel=1e-4)


def test_half_wave_dipole_radiation_resistance():
    d = half_wave_dipole(1e9)
    assert d["R_rad_ohm"] == pytest.approx(73.1, rel=0.01)


def test_half_wave_dipole_invalid():
    with pytest.raises(ValueError):
        half_wave_dipole(0)


def test_half_wave_dipole_far_field_positive():
    d = half_wave_dipole(1e9)
    assert d["far_field_boundary_m"] > 0


# ── quarter-wave transformer ──────────────────────────────────────────

def test_qwt_z0_geometric_mean():
    r = quarter_wave_transformer(50, 73, 2.4e9)
    assert r["Z_0_match_ohm"] == pytest.approx(np.sqrt(50 * 73), rel=1e-4)


def test_qwt_perfect_match_vswr():
    r = quarter_wave_transformer(50, 73, 2.4e9)
    assert r["VSWR_after"] == pytest.approx(1.0, abs=1e-6)


def test_qwt_self_match():
    r = quarter_wave_transformer(50, 50, 1e9)
    assert r["Z_0_match_ohm"] == pytest.approx(50.0)
    assert r["VSWR_before"] == pytest.approx(1.0, abs=1e-6)


def test_qwt_invalid():
    with pytest.raises(ValueError):
        quarter_wave_transformer(0, 73, 1e9)


# ── array factor ──────────────────────────────────────────────────────

def test_array_factor_single_element_flat():
    theta = np.linspace(-np.pi/2, np.pi/2, 100)
    af = array_factor(1, 0.5, theta)
    assert np.allclose(af["AF_power"], 1.0)


def test_array_factor_broadside_unsteered():
    theta = np.linspace(-np.pi/2, np.pi/2, 1000)
    af = array_factor(4, 0.5, theta, steering_angle_rad=0.0)
    peak_theta = theta[np.argmax(af["AF_power"])]
    assert abs(np.degrees(peak_theta)) < 5.0   # peak near 0 deg


def test_array_factor_steered():
    theta = np.linspace(-np.pi/2, np.pi/2, 1000)
    target_deg = 30.0
    af = array_factor(8, 0.5, theta, steering_angle_rad=np.radians(target_deg))
    peak_theta_deg = np.degrees(theta[np.argmax(af["AF_power"])])
    assert abs(peak_theta_deg - target_deg) < 5.0


def test_array_factor_normalized():
    theta = np.linspace(-np.pi/2, np.pi/2, 100)
    af = array_factor(4, 0.5, theta)
    assert np.max(af["AF_power"]) == pytest.approx(1.0, abs=0.01)


def test_array_factor_invalid_N():
    with pytest.raises(ValueError):
        array_factor(0, 0.5, np.array([0.0]))


def test_beam_width_decreases_with_N():
    bw4 = beam_width_3dB(4, 0.5)["BWFN_3dB_deg"]
    bw8 = beam_width_3dB(8, 0.5)["BWFN_3dB_deg"]
    assert bw4 > bw8


# ── fraunhofer distance ───────────────────────────────────────────────

def test_fraunhofer_distance_formula():
    r = fraunhofer_distance(0.1, 0.05)
    assert r["R_ff_m"] == pytest.approx(2 * 0.1**2 / 0.05, rel=1e-6)


def test_fraunhofer_invalid():
    with pytest.raises(ValueError):
        fraunhofer_distance(0, 0.1)


# ── link budget ───────────────────────────────────────────────────────

def test_link_budget_increases_with_power():
    lb1 = link_budget(0.01, 2.15, 2.15, 2.4e9, 10)
    lb10 = link_budget(0.10, 2.15, 2.15, 2.4e9, 10)
    assert lb10["P_rx_dBm"] > lb1["P_rx_dBm"]


def test_link_budget_decreases_with_distance():
    lb_near = link_budget(0.1, 2.15, 2.15, 2.4e9, 10)
    lb_far  = link_budget(0.1, 2.15, 2.15, 2.4e9, 100)
    assert lb_near["P_rx_dBm"] > lb_far["P_rx_dBm"]


def test_link_budget_invalid():
    with pytest.raises(ValueError):
        link_budget(0, 2.15, 2.15, 2.4e9, 10)


# ── sympy 5 ──────────────────────────────────────────────────────────

def test_antenna_sympy_5_count():
    eqs = antenna_sympy_5()
    assert len(eqs) == 5
    for eq in eqs.values():
        assert isinstance(eq, sp.Eq)
