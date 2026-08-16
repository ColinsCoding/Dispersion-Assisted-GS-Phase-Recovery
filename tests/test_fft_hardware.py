import os
import shutil

import numpy as np
import pytest
import sympy as sp

from dgs.fft_hardware import (
    geometric_series_closed_form, root_of_unity_orthogonality,
    bit_reverse_indices, fft_iterative_hardware,
    run_c, run_cuda, cross_validate_languages,
    GCC_DEFAULT, NVCC_DEFAULT,
)
from dgs.fourier_tools import fft_radix2

GCC_AVAILABLE = shutil.which("gcc") is not None or os.path.exists(GCC_DEFAULT)
NVCC_AVAILABLE = shutil.which("nvcc") is not None or os.path.exists(NVCC_DEFAULT)


def test_geometric_series_closed_form_holds_for_several_N():
    for N in range(2, 6):
        # raises AssertionError internally if the closed form fails to match
        geometric_series_closed_form(N)


def test_root_of_unity_orthogonality():
    """The identity that makes idft(dft(x)) == x: a geometric series of
    roots of unity telescopes to zero unless k == 0 (mod N)."""
    N = 8
    orth = root_of_unity_orthogonality(N)
    assert orth[0] == N
    assert all(orth[k] == 0 for k in range(1, N))


def test_bit_reverse_indices_known_values():
    # N=8 (3 bits): 0,1,2,...,7 -> 0,4,2,6,1,5,3,7
    assert bit_reverse_indices(8) == [0, 4, 2, 6, 1, 5, 3, 7]


def test_bit_reverse_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        bit_reverse_indices(6)


def test_iterative_matches_recursive_and_numpy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16) + 1j * rng.standard_normal(16)
    iterative = fft_iterative_hardware(x)
    recursive = fft_radix2(x)
    assert np.allclose(iterative, recursive)
    assert np.allclose(iterative, np.fft.fft(x))


def test_iterative_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        fft_iterative_hardware(np.arange(6, dtype=complex))


@pytest.mark.skipif(not GCC_AVAILABLE, reason="gcc not available on this machine")
def test_run_c_matches_numpy(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    gcc = shutil.which("gcc") or GCC_DEFAULT
    c_result = run_c(x, str(tmp_path), gcc_path=gcc)
    assert np.allclose(c_result, np.fft.fft(x), atol=1e-9)


@pytest.mark.skipif(not NVCC_AVAILABLE, reason="nvcc not available on this machine")
def test_run_cuda_matches_numpy(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.standard_normal(16) + 1j * rng.standard_normal(16)
    nvcc = shutil.which("nvcc") or NVCC_DEFAULT
    try:
        cuda_result = run_cuda(x, str(tmp_path), nvcc_path=nvcc)
    except RuntimeError as e:
        pytest.skip(f"nvcc present but compile/run failed (likely missing MSVC on PATH): {e}")
    assert np.allclose(cuda_result, np.fft.fft(x), atol=1e-9)


@pytest.mark.skipif(not GCC_AVAILABLE, reason="gcc not available on this machine")
def test_cross_validate_python_and_c(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    results, max_errs = cross_validate_languages(x, str(tmp_path), include_cuda=False)
    assert max_errs["python_iterative"] < 1e-9
    assert max_errs["c"] < 1e-9
