"""Tests for dgs.gps_denied_navigation."""
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest
import sympy as sp

from dgs import gps_denied_navigation as n

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_drift_symbolic_matches_numeric():
    t, ba, bg, g = sp.symbols("t b_a b_g g", positive=True)
    assert sp.simplify(n.drift_symbolic() - (ba * t**2 / 2 + g * bg * t**3 / 6)) == 0
    assert n.drift_error(10, 0.02) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        n.drift_error(-1, 0.01)


def test_better_sensors_drift_slower():
    times = [n.time_to_error(10, *n.SENSOR_GRADES[k]) for k in ("consumer MEMS", "tactical", "navigation")]
    assert times[0] < times[1] < times[2]
    assert n.drift_error(times[1], *n.SENSOR_GRADES["tactical"]) == pytest.approx(10, rel=1e-6)


def test_kalman_op_counts_and_cse_savings():
    b2, b3 = n.kalman_op_budget(2), n.kalman_op_budget(3)
    assert b2["update_cse"] < b2["update_naive"] and b3["update_cse"] < b3["update_naive"]
    assert b3["update_cse"] > b2["update_cse"]
    assert n.mcu_load(b3["update_cse"], 1000, mflops=10) < 0.05   # < 5% of a small FPU core at 1 kHz


def test_bias_estimating_filter_beats_dead_reckoning():
    r = n.simulate_navigation()
    err = {k: np.sqrt(np.mean((r[k] - r["truth"]) ** 2)) for k in ("dead_reckoning", "kf2", "kf3")}
    assert err["kf3"] < err["kf2"] < err["dead_reckoning"] / 50
    assert r["bias_estimate"] == pytest.approx(r["bias_true"], rel=0.02)


def test_atom_interferometer_phase_and_sensitivity():
    T, k, a = sp.symbols("T k_eff a", real=True)
    assert sp.simplify(n.ai_phase_symbolic() - k * a * T**2) == 0
    s = n.ai_shot_noise_sensitivity(0.01, 1e6)
    assert s == pytest.approx(1.24e-6, rel=0.01)
    assert n.ai_shot_noise_sensitivity(0.02, 1e6) == pytest.approx(s / 4)
    with pytest.raises(ValueError):
        n.ai_shot_noise_sensitivity(0.01, 1e6, contrast=0)


def test_ambiguity_ladder():
    a = 0.0123
    grid = np.linspace(-0.05, 0.05, 200001)

    def run(settings):
        ms = [n.ai_measure(a, T, 1e6, phase_offset=ph, n_shots=20, rng=i) for i, (T, ph) in enumerate(settings)]
        return n.ai_retrieve(ms, settings, grid, 1e6)[0]

    one = run([(0.005, 0)])
    two = run([(0.005, 0), (0.0071, 0)])
    quad = run([(0.005, 0), (0.005, np.pi / 2), (0.0071, 0), (0.0071, np.pi / 2)])
    assert abs(abs(one) - a) > 1e-3            # wrong 2 pi branch
    assert two == pytest.approx(-a, abs=2e-5)  # right magnitude, twin sign
    assert quad == pytest.approx(a, abs=2e-5)  # unique


def test_spinner_energy():
    E = n.disk_spinner_energy(0.15, 0.05, 10000)
    w = 10000 * 2 * np.pi / 60
    assert E == pytest.approx(0.5 * (0.5 * 0.15 * 0.05**2) * w**2)
    assert n.spinup_time(0.15, 0.05, 10000, 100, 0.5) == pytest.approx(E / 50)
    lo, hi = n.budget_range("antweight (1 lb / 454 g)")
    assert 0 < lo < hi
    with pytest.raises(ValueError):
        n.disk_spinner_energy(0, 0.05, 1000)


GCC = shutil.which("gcc") or (r"C:\msys64\mingw64\bin\gcc.exe" if os.path.exists(r"C:\msys64\mingw64\bin\gcc.exe") else None)


@pytest.mark.skipif(GCC is None, reason="gcc not available")
def test_c_twin_same_answer_at_every_optimization(tmp_path):
    src = ROOT / "learn" / "gps_denied_nav" / "kf3.c"
    outs = []
    env = dict(os.environ, PATH=str(pathlib.Path(GCC).parent) + os.pathsep + os.environ.get("PATH", ""))
    for opt in ("-O0", "-O2"):
        exe = tmp_path / f"kf3{opt}.exe"
        subprocess.run([GCC, "-std=c99", "-Wall", "-Wextra", "-Werror", opt, str(src), "-o", str(exe)],
                       check=True, env=env)
        line = subprocess.run([str(exe), "300000"], capture_output=True, text=True, check=True, env=env).stdout
        outs.append(line.split("bias_est=")[1].strip())
    assert outs[0] == outs[1]
    assert float(outs[0].split()[0]) == pytest.approx(9.80665e-3, rel=0.1)
