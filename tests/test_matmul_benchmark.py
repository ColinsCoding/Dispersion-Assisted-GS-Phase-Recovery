import os
import shutil
import numpy as np
import pytest
from dgs.matmul_benchmark import (
    pure_python_matmul, numpy_matmul, benchmark_pure_python, benchmark_numpy,
    compile_c_matmul, benchmark_c, language_comparison_table,
)


def test_pure_python_matmul_correctness():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = pure_python_matmul(A, B)
    expected = (np.array(A) @ np.array(B)).tolist()
    assert C == expected


def test_numpy_matmul_matches_pure_python():
    rng = np.random.default_rng(0)
    A = rng.random((5, 5))
    B = rng.random((5, 5))
    np_res = numpy_matmul(A, B)
    py_res = pure_python_matmul(A.tolist(), B.tolist())
    assert np.allclose(np_res, py_res)


def test_pure_python_matmul_identity():
    A = [[1.0, 0.0], [0.0, 1.0]]
    B = [[3.0, 4.0], [5.0, 6.0]]
    assert pure_python_matmul(A, B) == B


def test_benchmark_pure_python_returns_timing():
    res = benchmark_pure_python(10, n_trials=1)
    assert res["n"] == 10
    assert res["mean_s"] >= 0
    assert len(res["times"]) == 1


def test_benchmark_numpy_returns_timing():
    res = benchmark_numpy(20, n_trials=2)
    assert res["n"] == 20
    assert res["mean_s"] >= 0
    assert len(res["times"]) == 2


def test_numpy_faster_than_pure_python():
    py_res = benchmark_pure_python(60, n_trials=1)
    np_res = benchmark_numpy(60, n_trials=3)
    assert np_res["mean_s"] < py_res["mean_s"]


def test_language_comparison_table_structure():
    table = language_comparison_table()
    assert "Pure Python" in table
    assert "NumPy (BLAS)" in table
    assert "PyTorch (GPU/CUDA)" in table
    assert "common_lesson" in table
    for key in ["Pure Python", "NumPy (BLAS)", "C (raw triple loop)"]:
        assert "overhead_source" in table[key]
        assert "typical_speedup_vs_python" in table[key]


@pytest.mark.skipif(
    shutil.which("gcc") is None and not os.path.exists(r"C:\msys64\mingw64\bin\gcc.exe"),
    reason="gcc not available on this machine",
)
def test_compile_and_run_c_matmul(tmp_path):
    gcc = shutil.which("gcc") or r"C:\msys64\mingw64\bin\gcc.exe"
    exe = compile_c_matmul(str(tmp_path), gcc_path=gcc)
    assert os.path.exists(exe)
    res = benchmark_c(exe, 20, n_trials=2)
    assert res["mean_s"] >= 0
    assert len(res["times"]) == 2
