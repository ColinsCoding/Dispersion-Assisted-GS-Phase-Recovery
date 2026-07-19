import os
import shutil

import numpy as np
import pytest

from dgs.circuits_polyglot import (
    rlc_rk4_python, compile_c_rlc, run_c_rlc, run_matlab_rlc,
    cross_validate_languages, GCC_DEFAULT, MATLAB_DEFAULT,
)

R, L, C, V, dt, n_steps = 100.0, 1e-3, 1e-6, 1.0, 1e-6, 5

GCC_AVAILABLE = shutil.which("gcc") is not None or os.path.exists(GCC_DEFAULT)
MATLAB_AVAILABLE = os.path.exists(MATLAB_DEFAULT)


def test_python_rk4_matches_dgs_spice():
    """The hand-unrolled RK4 reference must reproduce dgs.spice.rlc_step_response
    (same equations, same weights) so it's a legitimate baseline for the C/MATLAB
    cross-checks, not a from-scratch reimplementation that happens to agree."""
    from dgs.spice import rlc_step_response
    t = np.linspace(0, n_steps * dt, n_steps + 1)
    vc_spice, il_spice = rlc_step_response(R, L, C, t, V=V)
    vc_ref, il_ref = rlc_rk4_python(R, L, C, V, 0.0, 0.0, dt, n_steps)
    assert np.allclose(vc_spice, vc_ref, atol=1e-12)
    assert np.allclose(il_spice, il_ref, atol=1e-12)


def test_python_rk4_zero_input_stays_at_rest():
    vc, il = rlc_rk4_python(R, L, C, V=0.0, vc0=0.0, il0=0.0, dt=dt, n_steps=n_steps)
    assert np.allclose(vc, 0.0) and np.allclose(il, 0.0)


@pytest.mark.skipif(not GCC_AVAILABLE, reason="gcc not available on this machine")
def test_compile_and_run_c_rlc(tmp_path):
    gcc = shutil.which("gcc") or GCC_DEFAULT
    exe = compile_c_rlc(str(tmp_path), gcc_path=gcc)
    assert os.path.exists(exe)
    vc_c, il_c = run_c_rlc(exe, R, L, C, V, 0.0, 0.0, dt, n_steps)
    vc_py, il_py = rlc_rk4_python(R, L, C, V, 0.0, 0.0, dt, n_steps)
    assert len(vc_c) == n_steps + 1
    assert np.allclose(vc_c, vc_py, atol=1e-9)
    assert np.allclose(il_c, il_py, atol=1e-9)


@pytest.mark.skipif(not MATLAB_AVAILABLE, reason="MATLAB not available on this machine")
def test_run_matlab_rlc(tmp_path):
    vc_m, il_m = run_matlab_rlc(str(tmp_path), R, L, C, V, 0.0, 0.0, dt, n_steps,
                                 matlab_path=MATLAB_DEFAULT)
    vc_py, il_py = rlc_rk4_python(R, L, C, V, 0.0, 0.0, dt, n_steps)
    assert len(vc_m) == n_steps + 1
    assert np.allclose(vc_m, vc_py, atol=1e-9)
    assert np.allclose(il_m, il_py, atol=1e-9)


@pytest.mark.skipif(not (GCC_AVAILABLE and MATLAB_AVAILABLE),
                     reason="requires both gcc and MATLAB")
def test_cross_validate_all_three_languages(tmp_path):
    result = cross_validate_languages(str(tmp_path), R=R, L=L, C=C, V=V, dt=dt, n_steps=n_steps)
    assert result["max_abs_diff_python_vs_c"] < 1e-9
    assert result["max_abs_diff_python_vs_matlab"] < 1e-9
