"""Tests for dgs.atomic_clock."""
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest
import sympy as sp

from dgs import atomic_clock as a

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_ramsey():
    D, T = sp.symbols("Delta T", real=True)
    assert sp.simplify(a.ramsey_symbolic() - (1 + sp.cos(D * T)) / 2) == 0
    Tp = sp.symbols("T", positive=True)
    assert sp.simplify(a.ramsey_fwhm_symbolic() - 1 / (2 * Tp)) == 0
    # finite pulses: on resonance still full transfer
    assert a.ramsey_fringe([0.0], 0.1, pulse_time=1e-3)[0] == pytest.approx(1.0, abs=1e-9)
    assert a.ramsey_fringe(0.0, 0.1) == pytest.approx(1.0)


def test_qpn_allan():
    nu0, T, N, Tc, tau = sp.symbols("nu_0 T N T_c tau", positive=True)
    expect = sp.sqrt(Tc / tau) / (2 * sp.pi * nu0 * T * sp.sqrt(N))
    assert sp.simplify(a.qpn_allan_symbolic() - expect) == 0
    s1 = a.qpn_allan(1, a.NU_CS, 0.5, 1e6, 1.0)
    assert a.qpn_allan(100, a.NU_CS, 0.5, 1e6, 1.0) == pytest.approx(s1 / 10)


def test_holdover_and_relativity():
    assert a.holdover_error(a.DAY, 1e-9) == pytest.approx(86.4e-6)
    g, v, tot = a.gps_relativity_us_per_day()
    assert g == pytest.approx(45.7, abs=0.1) and v == pytest.approx(-7.2, abs=0.1) and tot == pytest.approx(38.5, abs=0.1)
    assert a.height_shift(1.0) == pytest.approx(1.09e-16, rel=0.01)


def test_ieee754_and_accumulation():
    assert a.float_resolution(1e9) == pytest.approx(1.1920928955078125e-07)
    assert a.float_resolution(a.NU_SR) == 0.0625
    s32 = a.accumulate_time_fast(dtype=np.float32)
    s64 = a.accumulate_time_fast(dtype=np.float64)
    assert s32 - 360000 == pytest.approx(-12975.21875)
    assert abs(s64 - 360000) < 1e-4
    assert a.accumulate_time(0.1, 1000, np.float32) == a.accumulate_time_fast(0.1, 1000, np.float32)
    per, tot = a.patriot_chop_error()
    assert per == pytest.approx(9.5367431640625e-08) and tot == pytest.approx(0.3433, abs=1e-4)
    assert a.exact_sum_symbolic() == 360000


def test_adev_loop_equals_vectorized_and_slopes():
    y = a.powerlaw_frequency_noise(5000, 1.0, 0, rng=0)
    x = a.phase_from_frequency(y, 1.0)
    for m in (1, 5, 20):
        assert a.adev_loop(x, 1.0, m) == pytest.approx(a.adev(x, 1.0, m), rel=1e-12)
    assert a.adev_symbolic_second_difference() == 0
    expect_mdev = {"white PM": -1.5, "flicker PM": -1.0, "white FM": -0.5, "flicker FM": 0.0, "random walk FM": 0.5}
    for k, b in a.NOISE_BETA.items():
        xx = a.phase_from_frequency(a.powerlaw_frequency_noise(2**15, 1.0, b, rng=1), 1.0)
        f = a.slope_features(xx, 1.0)
        assert f[6:].mean() == pytest.approx(expect_mdev[k], abs=0.15)
    with pytest.raises(ValueError):
        a.mdev(np.zeros(10), 1.0, 5)


def test_classifiers():
    X, y = a.make_noise_dataset(n_per_class=40, n=2048)
    r = a.train_classifiers(X, y)
    assert r["forest_acc"] >= 0.95 and r["tree_acc"] >= 0.9
    imp = r["forest"].feature_importances_
    assert imp.shape == (12,) and imp.sum() == pytest.approx(1.0)


GCC = shutil.which("gcc") or (r"C:\msys64\mingw64\bin\gcc.exe" if os.path.exists(r"C:\msys64\mingw64\bin\gcc.exe") else None)


@pytest.mark.skipif(GCC is None, reason="gcc not available")
def test_c_twin(tmp_path):
    env = dict(os.environ, PATH=str(pathlib.Path(GCC).parent) + os.pathsep + os.environ.get("PATH", ""))
    exe = tmp_path / "time_accum.exe"
    subprocess.run([GCC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(ROOT / "learn" / "atomic_clock" / "time_accum.c"), "-o", str(exe)], check=True, env=env)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True, env=env).stdout
    lines = {l.split()[0]: l for l in out.splitlines()}
    assert "-12975.218750" in lines["float"]
    assert "+0.000e+00" in lines["uint64ns"]
    assert "-0.343323" in lines["fixed23"]
