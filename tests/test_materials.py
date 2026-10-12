"""Tests for dgs/materials.py and its C (gcc) and CUDA (nvcc) twins."""
import glob
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest
import sympy as sp

from dgs import materials as M

ROOT = pathlib.Path(__file__).resolve().parents[1]
HDR_DIR = ROOT / 'dgs' / 'c' / 'materials'


# --- published values the coefficients must reproduce --------------------------------------
@pytest.mark.parametrize('name, lam, expected, tol', [
    ('n_bk7', 0.5875618, 1.5168, 1e-4),          # SCHOTT n_d
    ('fused_silica', 0.5875618, 1.4585, 1e-4),   # Malitson n_d
    ('fused_silica', 1.55, 1.4440, 1e-4),
    ('sapphire_o', 0.5875618, 1.7682, 1e-4),
    ('caf2', 0.5875618, 1.4338, 1e-4),
    ('linbo3_e', 1.55, 2.138, 1e-3),
    ('linbo3_o', 1.55, 2.211, 1e-3),
])
def test_index_matches_published(name, lam, expected, tol):
    assert abs(float(M.n(name, lam)) - expected) < tol


@pytest.mark.parametrize('name, V, tol', [('n_bk7', 64.17, 0.05), ('fused_silica', 67.8, 0.1), ('caf2', 94.99, 0.1)])
def test_abbe_numbers(name, V, tol):
    assert abs(M.abbe(name) - V) < tol


def test_silica_dispersion_at_1550_and_zero_dispersion():
    assert -29 < float(M.beta2('silica', 1.55)) < -27        # about -28 ps^2/km
    assert 21 < float(M.D('silica', 1.55)) < 23              # about +22 ps/(nm km)
    assert abs(M.zero_dispersion_um('silica') - 1.27) < 0.01


def test_D_beta2_sign_convention():
    lam = np.linspace(0.5, 2.0, 7)
    lhs = M.D('bk7', lam)
    rhs = -2*np.pi*M.C0/((lam*1e-6)**2)*M.beta2('bk7', lam)*1e-27*1e6   # D = -2 pi c beta2 / lambda^2
    assert np.allclose(lhs, rhs, rtol=1e-12)


def test_closed_form_derivatives_match_sympy():
    lam = sp.symbols('lam', positive=True)
    m = M.get('n_bk7')
    nexpr = sp.sqrt(1 + sum(B*lam**2/(lam**2 - C) for B, C in zip(m.B, m.C)))
    n1, n2 = M.dn_dlam(m, 0.8)
    assert abs(float(sp.diff(nexpr, lam).subs(lam, 0.8)) - float(n1)) < 1e-12
    assert abs(float(sp.diff(nexpr, lam, 2).subs(lam, 0.8)) - float(n2)) < 1e-12


def test_lookup_aliases_and_errors():
    assert M.get('N-BK7').name == 'n_bk7' and M.get('Fused Silica').name == 'fused_silica'
    with pytest.raises(KeyError):
        M.get('unobtainium')
    with pytest.raises(ValueError):
        M.n('n_bk7', 3.0)                                     # outside the fit range


def test_fresnel_and_stretch():
    assert abs(float(M.fresnel_R('n_bk7', 0.5875618)) - 0.0422) < 1e-3
    assert abs(M.stretch_ps('silica', 1.0, 1.55, 10.0) - 10*float(M.D('silica', 1.55))) < 1e-9


# --- the C header must agree with Python ----------------------------------------------------
def _gcc():
    return shutil.which('gcc') or next((p for p in [r'C:\msys64\mingw64\bin\gcc.exe'] if os.path.exists(p)), None)


@pytest.mark.skipif(_gcc() is None or not (HDR_DIR / 'materials_cli.c').exists(), reason='gcc or the C sources not found')
def test_c_header_matches_python(tmp_path):
    gcc = _gcc()
    env = dict(os.environ, PATH=os.path.dirname(gcc) + os.pathsep + os.environ.get('PATH', ''))
    exe = tmp_path / 'materials_cli.exe'
    cp = subprocess.run([gcc, '-O2', '-Wall', '-Wextra', '-std=c99', str(HDR_DIR / 'materials_cli.c'), '-o', str(exe), '-lm'],
                        capture_output=True, text=True, env=env)
    assert cp.returncode == 0, cp.stderr
    out = subprocess.run([str(exe), '1.3'], capture_output=True, text=True, env=env).stdout.strip().splitlines()[1:]
    assert len(out) == len(M.LIBRARY)
    for line in out:
        name, *vals = line.split(',')
        py = [float(M.n(name, 1.3)), float(M.n_group(name, 1.3)), float(M.beta2(name, 1.3)), float(M.D(name, 1.3))]
        assert np.allclose([float(v) for v in vals], py, rtol=1e-9, atol=1e-9)
    assert subprocess.run([str(exe)], capture_output=True, env=env).returncode == 2


# --- the CUDA kernel must agree with Python -------------------------------------------------
def _cuda_tools():
    nvcc = shutil.which('nvcc') or (glob.glob(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe') or [None])[-1]
    vc = (glob.glob(r'C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat') or [None])[0]
    return nvcc, vc


@pytest.mark.skipif(_cuda_tools()[0] is None or (os.name == 'nt' and _cuda_tools()[1] is None)
                    or not (ROOT / 'dgs' / 'cuda' / 'materials.cu').exists(), reason='nvcc, MSVC or the CUDA source not found')
def test_cuda_matches_python(tmp_path):
    nvcc, vc = _cuda_tools()
    exe, out = tmp_path / 'materials_cuda.exe', tmp_path / 'grid.bin'
    src = ROOT / 'dgs' / 'cuda' / 'materials.cu'
    build = f'"{nvcc}" -O2 -arch=sm_89 -I"{HDR_DIR}" "{src}" -o "{exe}"'
    if os.name == 'nt':
        bat = tmp_path / 'b.bat'
        bat.write_text(f'@echo off\ncall "{vc}" >nul 2>nul\n{build}\nexit /b %ERRORLEVEL%\n')
        cp = subprocess.run(['cmd', '/c', str(bat)], capture_output=True, text=True, timeout=600)
    else:
        cp = subprocess.run(build, shell=True, capture_output=True, text=True, timeout=600)
    assert cp.returncode == 0, cp.stdout[-800:] + cp.stderr[-800:]
    npts, lo, hi = 1001, 0.4, 2.5
    rr = subprocess.run([str(exe), str(out), str(npts), str(lo), str(hi)], capture_output=True, text=True)
    if rr.returncode == 3:
        pytest.skip('no usable CUDA device: ' + rr.stderr.strip())
    assert rr.returncode == 0, rr.stderr
    grid = np.fromfile(out, dtype=np.float64).reshape(len(M.LIBRARY), 2, npts)
    lam = lo + (hi - lo)*np.arange(npts)/(npts - 1)
    for i, m in enumerate(M.LIBRARY.values()):
        ok = (lam >= m.lam_min_um) & (lam <= m.lam_max_um)
        assert np.all(np.isnan(grid[i, 0, ~ok]))
        assert np.allclose(grid[i, 0, ok], M.n(m, lam[ok]), rtol=1e-12)
        assert np.allclose(grid[i, 1, ok], M.beta2(m, lam[ok]), rtol=1e-9, atol=1e-9)
