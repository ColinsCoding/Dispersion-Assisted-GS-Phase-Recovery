"""Tests for dgs/thomson_vs_rutherford.py (+ opt-in MATLAB twin: RUN_MATLAB=1)."""
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest
import sympy as sp

from dgs.thomson_vs_rutherford import (
    thomson_field_symbolic, thomson_deflection_symbolic, theta_max_thomson, theta_max_electron,
    collisions_in_foil, theta_rms_multiple, log10_prob_beyond_multiple, rutherford_prob_beyond,
    mc_random_walk_rms, TIMELINE,
)


def test_gauss_law_inside_sphere():
    k, Q, R, r = sp.symbols("k Q R r", positive=True)
    assert sp.simplify(thomson_field_symbolic() - k * Q * r / R ** 3) == 0


def test_impulse_and_maximum():
    theta, b_star, theta_max = thomson_deflection_symbolic()
    k, q, Q, R, K = sp.symbols("k q Q R K", positive=True)
    assert sp.simplify(b_star - R / sp.sqrt(2)) == 0
    assert sp.simplify(theta_max - k * q * Q / (2 * K * R)) == 0


def test_numbers():
    assert theta_max_thomson() == pytest.approx(1.026e-4, rel=1e-3)
    assert theta_max_electron() == pytest.approx(1 / 7294.3, rel=1e-6)
    assert collisions_in_foil() == pytest.approx(1557, rel=1e-3)


def test_plum_pudding_forbids_backscatter_rutherford_allows():
    th = theta_rms_multiple(max(theta_max_thomson(), theta_max_electron()), collisions_in_foil())
    assert log10_prob_beyond_multiple(np.pi / 2, th) < -30000
    P = rutherford_prob_beyond(np.pi / 2)
    assert 1e-6 < P < 1e-3


def test_random_walk_monte_carlo():
    assert mc_random_walk_rms(1e-4, 400) == pytest.approx(theta_rms_multiple(1e-4, 400), rel=0.02)


def test_timeline_order():
    years = [y for y, *_ in TIMELINE]
    assert years == sorted(years)
    names = [w for _, w, _ in TIMELINE]
    assert names.index("Rutherford") < names.index("Bohr")      # Bohr came after


def test_validation():
    with pytest.raises(ValueError):
        theta_max_thomson(R=0)


@pytest.mark.skipif(os.environ.get("RUN_MATLAB") != "1", reason="set RUN_MATLAB=1 (MATLAB startup is slow)")
def test_matlab_twin_agrees():
    exe = shutil.which("matlab") or r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"
    root = pathlib.Path(__file__).resolve().parent.parent
    out = subprocess.run([exe, "-batch", "run('learn/matlab/thomson_vs_rutherford.m')"], cwd=root,
                         capture_output=True, text=True, timeout=600).stdout
    py = log10_prob_beyond_multiple(np.pi / 2, theta_rms_multiple(
        max(theta_max_thomson(), theta_max_electron()), collisions_in_foil()))
    assert f"10^({py:.0f})" in out and "MATLAB checks passed" in out
