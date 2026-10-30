"""Cross-check matlab/mp_toolkit.m (MATLAB R2025b, headless) against the Python toolkit.

Skips when MATLAB is not installed. Takes ~15 s (MATLAB start-up).
"""
import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dgs import gs_core as G
from dgs import one_d_quantum as Q

ROOT = Path(__file__).resolve().parents[1]


def find_matlab():
    return shutil.which("matlab") or (sorted(glob.glob(r"C:\Program Files\MATLAB\R20*\bin\matlab.exe")) or [None])[-1]


@pytest.fixture(scope="module")
def mat(tmp_path_factory):
    exe = find_matlab()
    if exe is None:
        pytest.skip("MATLAB not installed")
    out = tmp_path_factory.mktemp("mat") / "out.json"
    cp = subprocess.run([exe, "-batch", f"addpath('{(ROOT / 'matlab').as_posix()}'); mp_toolkit('{out.as_posix()}')"],
                        capture_output=True, text=True, timeout=600)
    if cp.returncode or not out.exists():
        pytest.fail("MATLAB failed:\n" + cp.stdout[-2000:] + cp.stderr[-2000:])
    return json.loads(out.read_text())


def test_box_and_tunneling_match(mat):
    assert np.allclose(mat["box_E"], [Q.box_energy(n, 0.5) for n in range(1, 5)], rtol=1e-6)
    assert mat["eta"] == pytest.approx(Q.penetration_depth(3.0, 5.0), rel=1e-6)
    assert mat["tunnel"] == pytest.approx(Q.tunneling_probability(3.0, 5.0, 0.4), rel=1e-6)


def test_fd_solver_matches(mat):
    x = np.linspace(-1.0, 1.5, 801)
    U = np.where((x > 0) & (x < 0.5), 0.0, 5.0)
    E, _ = Q.solve_schrodinger(x, U, n_states=3)
    assert np.allclose(mat["fd_E"], E, rtol=1e-8)
    assert mat["fd_norm"] == pytest.approx(1.0, abs=1e-9)


def test_dispersion_and_gs_match(mat):
    N, D1, D2 = 256, -5000.0, -5750.0
    k = np.arange(N)
    E_true = np.exp(1j * np.pi / 2 * np.mod(k // 8 * 7 + 3, 4))
    I1, I2 = np.abs(G.disperse(E_true, D1))**2, np.abs(G.disperse(E_true, D2))**2
    E = G.undisperse(np.sqrt(I1).astype(complex), D1)
    errs = []
    for _ in range(30):
        E = G.gs_iteration(E, I1, I2, D1, D2, unit_amplitude=True)
        errs.append(np.sqrt(np.mean((np.abs(G.disperse(E, D2))**2 - I2)**2)))
    assert mat["energy_ratio"] == pytest.approx(1.0, abs=1e-12)            # Parseval (Lean: H_norm)
    assert mat["roundtrip_err"] < 1e-12                                     # Lean: H_inverse
    assert np.allclose(mat["gs_errors"], errs, rtol=1e-6)
