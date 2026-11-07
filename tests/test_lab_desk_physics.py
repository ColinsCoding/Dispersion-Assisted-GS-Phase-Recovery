"""Tests for projects/lab_desk/desk_physics.py (the lab-desk simulator's bench)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "projects", "lab_desk"))
import desk_physics as dp  # noqa: E402


def test_unit_conversion_smf28():
    # 1 km of SMF-28 at 1550 nm: D = 17 ps/nm  ->  beta2 = -21.7 ps^2/km
    assert dp.beta2L_from_psnm(17.0) == pytest.approx(-21.68e-27, rel=0.01)


def _pre_post_energy(bench):
    """Energy of the gas lines' impulse response before t = 0 divided by energy after it."""
    h = np.fft.ifft(bench.cell_H() - 1)              # t < 0 lives in the second half of the FFT array
    n = bench.N // 2
    return np.sum(np.abs(h[n:]) ** 2) / np.sum(np.abs(h[:n]) ** 2)


def test_causal_cell_has_no_precursor():
    # complex-Lorentzian lines (Kramers-Kronig pairs): almost nothing before the pulse arrives
    assert _pre_post_energy(dp.Bench()) < 2e-3


def test_noncausal_cell_rings_before_t0():
    # the real-only absorption model (2024 slides) is symmetric in time: half its response is a precursor
    b = dp.Bench()
    b.causal = False
    assert _pre_post_energy(b) > 0.5


def test_near_field_pair_recovers_field():
    b = dp.Bench()
    run = dp.TDGSARun(b).step(200)
    assert run.field_err[-1] < 0.15


def test_2024_far_field_pair_fails():
    b = dp.Bench()
    b.D1, b.D2 = 353.0, 872.0
    run = dp.TDGSARun(b).step(200)
    assert run.field_err[-1] > 0.8


def test_detector_bandwidth_hurts():
    b = dp.Bench()
    ideal = dp.TDGSARun(b).step(150).field_err[-1]
    b.bw = 8e9
    slow = dp.TDGSARun(b).step(150).field_err[-1]
    assert slow > 2 * ideal


def test_energy_conserved_by_dispersion():
    b = dp.Bench()
    e_cell = np.sum(np.abs(np.fft.ifft(b.field_after_cell())) ** 2)
    assert np.sum(np.abs(b.arm_field(40.0)) ** 2) == pytest.approx(e_cell, rel=1e-9)


def test_capture_files(tmp_path):
    dp.write_capture(dp.Bench(), str(tmp_path), n_iter=20)
    for name in ("desk_capture.npz", "desk_tdgsa.m", "desk_tdgsa.ipynb"):
        assert (tmp_path / name).exists()


# ---- THz probe (time-stretch front end)
def test_time_stretch_rescues_slow_scope():
    b = dp.Bench()
    b.bw = 8e9
    slow = dp.TDGSARun(b).step(200).field_err[-1]
    b.stretch = 30                                   # 8 GHz x 30 = 240 GHz effective
    stretched = dp.TDGSARun(b).step(200).field_err[-1]
    assert slow > 0.8 and stretched < 0.25


def test_effective_bandwidth():
    b = dp.Bench()
    b.bw, b.stretch = 33e9, 30
    assert b.effective_bw == pytest.approx(0.99e12)


# ---- topology probe (phase vortices of the Gaussian-window STFT)
def test_vortices_all_wind_the_same_way():
    b = dp.Bench()
    for D in (1, 10, 40):
        S, ts, fs = dp.stft(b.arm_field(D), b)
        qs = {q for *_, q in dp.vortices(S, ts, fs)}
        assert qs <= {-1}          # zeros of an analytic (Bargmann) function share one winding sign


def _match(D1, D2):
    b = dp.Bench()
    b.D1, b.D2 = D1, D2
    run = dp.TDGSARun(b).step(200)
    view = np.exp(1j * dp.beta2L_from_psnm(10) * b.omega ** 2 / 2)
    St, ts, fs = dp.stft(np.fft.ifft(b.field_after_cell() * view), b)
    Sr, _, _ = dp.stft(np.fft.ifft(run.spectrum() * view), b)
    return dp.vortex_match(dp.vortices(St, ts, fs), dp.vortices(Sr, ts, fs), ts, fs)


def test_good_recovery_puts_vortices_in_the_right_place():
    assert _match(1, 40) == 1.0


def test_failed_recovery_misses_the_vortices():
    assert _match(353, 872) < 0.5
