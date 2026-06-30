"""Tests for dgs/suspension.py — spring-mass-damper, quarter-car, road PSD."""
import numpy as np
import sympy as sp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.suspension import (
    sdof_params, frequency_response, step_response_sdof,
    quarter_car_params, road_psd_response,
    transfer_function_sympy, suspension_sympy_5,
)


# ── SDOF params ───────────────────────────────────────────────────────────────

def test_sdof_underdamped():
    res = sdof_params(k=15000, m=300, c=1500)
    assert res["regime"] == "underdamped"
    assert 0 < res["zeta"] < 1


def test_sdof_critical():
    k, m = 10000.0, 100.0
    c_cr = 2 * np.sqrt(k * m)
    res = sdof_params(k=k, m=m, c=c_cr)
    assert res["regime"] == "critically_damped"
    assert abs(res["zeta"] - 1.0) < 1e-6


def test_sdof_overdamped():
    k, m = 10000.0, 100.0
    c_cr = 2 * np.sqrt(k * m)
    res = sdof_params(k=k, m=m, c=3*c_cr)
    assert res["regime"] == "overdamped"
    assert res["zeta"] > 1.0


def test_sdof_omega_n():
    res = sdof_params(k=1000.0, m=10.0, c=0.0)
    # omega_n = sqrt(k/m) = sqrt(100) = 10
    assert abs(res["omega_n"] - 10.0) < 1e-6


def test_sdof_q_factor():
    # Q = 1/(2*zeta). For zeta=0.5, Q=1.0
    res = sdof_params(k=1000.0, m=1.0, c=2*np.sqrt(1000))  # zeta=1 -> critical
    # zeta=1 -> Q=0.5
    assert abs(res["Q"] - 0.5) < 1e-6


def test_sdof_static_deflection():
    res = sdof_params(k=9810.0, m=1.0, c=1.0)
    # delta_st = mg/k = 9.81/9810 = 1e-3 m
    assert abs(res["delta_st"] - 1e-3) < 1e-6


# ── Frequency response ────────────────────────────────────────────────────────

def test_freq_response_DC():
    # At omega=0: |H|=1 (static unit displacement for unit force/k)
    res = frequency_response(10.0, 0.3, np.array([0.0]))
    assert abs(res["H_mag"][0] - 1.0) < 1e-6


def test_freq_response_resonance_peak():
    zeta = 0.1
    omega_n = 10.0
    omega_arr = np.linspace(0.1, 30, 1000)
    res = frequency_response(omega_n, zeta, omega_arr)
    # Peak H_mag should be near 1/(2*zeta*sqrt(1-zeta^2))
    H_peak_theory = 1 / (2 * zeta * np.sqrt(1 - zeta**2))
    assert abs(max(res["H_mag"]) - H_peak_theory) / H_peak_theory < 0.01


def test_freq_response_high_freq_rolloff():
    omega_arr = np.array([1.0, 10.0, 100.0])
    res = frequency_response(1.0, 0.5, omega_arr)
    # At high frequency (r >> 1): |H| ~ 1/r^2
    # H(100) << H(1)
    assert res["H_mag"][2] < res["H_mag"][0] * 0.01


def test_freq_response_phase_at_resonance():
    # At omega = omega_n (r=1): H_phase = -pi/2 for all zeta
    omega_n = 5.0
    res = frequency_response(omega_n, 0.3, np.array([omega_n]))
    # H_phase in radians: should be ~-pi/2
    assert abs(res["H_phase"][0] + np.pi/2) < 0.05


# ── Step response ─────────────────────────────────────────────────────────────

def test_step_response_underdamped_final_value():
    res = step_response_sdof(k=1000.0, m=10.0, c=50.0, F0=1000.0, t_end=5.0)
    # Final value: x_ss = F0/k = 1.0 (need enough time to settle)
    assert abs(res["x"][-1] - 1.0) < 0.05


def test_step_response_underdamped_overshoot():
    res = step_response_sdof(k=1000.0, m=10.0, c=50.0, F0=1000.0)
    # Underdamped should overshoot: max(x) > 1.0
    assert max(res["x"]) > 1.0


def test_step_response_critically_damped_no_overshoot():
    k, m = 1000.0, 10.0
    c_cr = 2 * np.sqrt(k * m)
    res = step_response_sdof(k=k, m=m, c=c_cr, F0=1000.0)
    # No overshoot for critically damped
    assert max(res["x"]) <= 1.001   # small tolerance for discrete t


def test_step_response_overdamped_no_overshoot():
    k, m = 1000.0, 10.0
    c_cr = 2 * np.sqrt(k * m)
    res = step_response_sdof(k=k, m=m, c=3*c_cr, F0=1000.0)
    assert max(res["x"]) <= 1.001


def test_step_response_returns_time():
    res = step_response_sdof(k=1000.0, m=10.0, c=50.0)
    assert "t" in res
    assert "x" in res
    assert len(res["t"]) == len(res["x"])


# ── Quarter-car ───────────────────────────────────────────────────────────────

def test_quarter_car_body_bounce():
    res = quarter_car_params()
    # Body bounce: 1–2 Hz
    assert 0.5 < res["f_body_Hz"] < 3.0


def test_quarter_car_wheel_hop():
    res = quarter_car_params()
    # Wheel hop: 8–20 Hz
    assert 5.0 < res["f_wheel_Hz"] < 25.0


def test_quarter_car_two_modes():
    res = quarter_car_params()
    modes = res["f_modes_Hz"]
    assert len(modes) == 2
    assert modes[0] < modes[1]   # body < wheel


def test_quarter_car_mode_separation():
    res = quarter_car_params()
    # Wheel hop should be significantly higher than body bounce
    assert res["f_wheel_Hz"] > 3 * res["f_body_Hz"]


# ── Road PSD ──────────────────────────────────────────────────────────────────

def test_road_psd_class_b_comfortable():
    omega_arr = np.linspace(0.5, 50, 500)
    res = road_psd_response(7.0, 0.3, omega_arr, road_roughness_class="B")
    # Class B road at ~1 Hz body: should be comfortable
    assert res["comfortable"] is True


def test_road_psd_class_d_rougher():
    omega_arr = np.linspace(0.5, 50, 500)
    res_b = road_psd_response(7.0, 0.3, omega_arr, road_roughness_class="B")
    res_d = road_psd_response(7.0, 0.3, omega_arr, road_roughness_class="D")
    # Class D is rougher (higher RMS accel)
    assert res_d["rms_acc_g"] > res_b["rms_acc_g"]


def test_road_psd_returns_spectrum():
    omega_arr = np.linspace(1, 30, 100)
    res = road_psd_response(7.0, 0.3, omega_arr)
    assert "G_road" in res
    assert "H_accel" in res
    assert len(res["G_road"]) == len(omega_arr)


# ── SymPy ─────────────────────────────────────────────────────────────────────

def test_transfer_function_sympy():
    res = transfer_function_sympy()
    assert "H_s" in res
    assert "poles" in res
    assert isinstance(res["H_s"], sp.Eq)


def test_suspension_sympy_5_count():
    eqs = suspension_sympy_5()
    assert len(eqs) == 5


def test_suspension_sympy_5_all_eq():
    eqs = suspension_sympy_5()
    for k, v in eqs.items():
        assert isinstance(v, sp.Basic), f"{k} should be sympy Basic"
