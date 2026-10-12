"""Tests for dgs/spherical_bessel_scope.py."""
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest
import sympy as sp
from scipy.special import spherical_jn

from dgs.spherical_bessel_scope import (
    xs, rayleigh_jn, ode_residual, small_x_leading, taylor, asymptotic, recurrence_residual,
    waveform, to_dac_codes, codes_to_volts, scope_view, quantization_rms_error, lut_header,
    usb_budget, spherical_box_levels,
)


@pytest.mark.parametrize("n", range(5))
def test_rayleigh_solves_ode_and_matches_scipy(n):
    assert ode_residual(n) == 0
    f = sp.lambdify(xs, rayleigh_jn(n), "numpy")
    x = np.linspace(0.5, 25, 50)
    assert np.allclose(f(x), spherical_jn(n, x), atol=1e-12)


def test_limits():
    for n in range(4):
        lead = sp.limit(rayleigh_jn(n) / small_x_leading(n), xs, 0)
        assert lead == 1
    x = 200.0
    assert spherical_jn(2, x) == pytest.approx(float(asymptotic(2).subs(xs, x)), abs=5e-4)
    assert recurrence_residual(3) == 0
    assert sp.simplify(taylor(0, 6).removeO() - (1 - xs**2/6 + xs**4/120)) == 0


def test_dac_chain():
    t, x, y = waveform(0, x_max=30, period_s=1e-3, fs=1e6)
    assert len(t) == 1000 and y[0] == pytest.approx(1.0)
    codes = to_dac_codes(y, 12)
    assert codes.min() >= 0 and codes.max() == 4095
    assert codes_to_volts(codes).max() == pytest.approx(3.3)
    for b in (8, 10, 12):
        meas, theory = quantization_rms_error(waveform(2)[2], b)
        assert meas == pytest.approx(theory, rel=0.1)
    with pytest.raises(ValueError):
        to_dac_codes(y, 0)


def test_scope_bandwidth_smooths():
    t, x, y = waveform(3, fs=1e6)
    v = codes_to_volts(to_dac_codes(y))
    wide, narrow = scope_view(v, 1e6, 500e3), scope_view(v, 1e6, 5e3)
    assert np.abs(wide - v).max() < np.abs(narrow - v).max()


def test_usb_budget():
    assert not usb_budget(1e6, 12)["fits"]                 # 16 Mbit/s > ~7 Mbit/s usable
    assert usb_budget(100e3, 12)["fits"]
    assert usb_budget(1e6, 8)["needed Mbit/s"] == pytest.approx(8.0)


def test_spherical_box():
    lv = spherical_box_levels(l_max=1, n_max=2)
    assert lv[0][1:3] == (1, 0) and lv[0][3] == pytest.approx(np.pi)
    assert lv[1][2] == 1                                    # 1p lies below 2s


@pytest.mark.skipif(shutil.which("gcc") is None and not pathlib.Path(r"C:\msys64\mingw64\bin\gcc.exe").exists(),
                    reason="gcc not installed")
def test_lut_header_compiles(tmp_path):
    hdr, codes = lut_header(0, size=64)
    (tmp_path / "sph_j0_lut.h").write_text(hdr)
    (tmp_path / "main.c").write_text('#include <stdio.h>\n#include "sph_j0_lut.h"\n'
                                     'int main(void){printf("%d %d %d\\n", SPH_J0_LUT_SIZE, SPH_J0_LUT[0], SPH_J0_LUT[63]);return 0;}\n')
    gcc = shutil.which("gcc") or r"C:\msys64\mingw64\bin\gcc.exe"
    env = dict(os.environ, PATH=r"C:\msys64\mingw64\bin" + os.pathsep + os.environ.get("PATH", ""))
    exe = tmp_path / "t.exe"
    subprocess.run([gcc, "-std=c99", "-Wall", "-Werror", "-o", str(exe), str(tmp_path / "main.c")],
                   check=True, capture_output=True, env=env)
    out = subprocess.run([str(exe)], capture_output=True, text=True, env=env).stdout.split()
    assert out == ["64", str(codes[0]), str(codes[63])]
