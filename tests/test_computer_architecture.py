import numpy as np
import pytest
import sympy as sp
from dgs.computer_architecture import (
    dtype_info, dtype_memory_for_array, float32_anatomy,
    memory_bandwidth_roofline, gs_algorithm_roofline,
    slurm_job_script, slurm_array_job,
    cuda_memory_plan, defense_research_influence,
    computer_arch_sympy_5,
)


# ── dtype_info ────────────────────────────────────────────────────────

def test_dtype_float32_bits():
    d = dtype_info("float32")
    assert d["bits"] == 32
    assert d["bytes"] == 4


def test_dtype_complex128_bits():
    d = dtype_info("complex128")
    assert d["bits"] == 128
    assert d["bytes"] == 16


def test_dtype_bfloat16_wider_range_than_float16():
    f16 = dtype_info("float16")
    bf16 = dtype_info("bfloat16")
    # bfloat16 has same exponent range as float32 (3.4e38), float16 maxes at 65504
    assert abs(bf16["max_val"]) > abs(f16["max_val"])


def test_dtype_invalid():
    with pytest.raises(ValueError):
        dtype_info("float128_does_not_exist")


def test_dtype_case_insensitive():
    d = dtype_info("Float32")
    assert d["bits"] == 32


# ── memory for array ─────────────────────────────────────────────────

def test_array_memory_complex64():
    m = dtype_memory_for_array("complex64", (65536,))
    assert m["bytes_per_element"] == 8   # complex64 = 2*float32 = 8 bytes
    assert m["total_bytes"] == 65536 * 8


def test_array_memory_float32_matrix():
    m = dtype_memory_for_array("float32", (1024, 1024))
    expected_bytes = 1024 * 1024 * 4
    assert m["total_bytes"] == expected_bytes
    assert m["total_MB"] == pytest.approx(expected_bytes / 1024**2, rel=1e-6)


def test_array_memory_scalar():
    m = dtype_memory_for_array("float64", ())
    assert m["n_elements"] == 1


# ── float32 anatomy ───────────────────────────────────────────────────

def test_float32_anatomy_positive_one():
    a = float32_anatomy(1.0)
    assert a["sign_bit"] == 0
    assert a["exponent_bits"] == 127   # 2^(127-127) = 1.0
    assert a["mantissa_bits"] == 0


def test_float32_anatomy_reconstructed():
    for v in [3.14, 1.0, -2.5, 0.125]:
        a = float32_anatomy(v)
        assert abs(a["reconstructed"] - v) < 1e-4 * abs(v) + 1e-7


def test_float32_anatomy_negative():
    a = float32_anatomy(-1.0)
    assert a["sign_bit"] == 1


# ── roofline model ────────────────────────────────────────────────────

def test_roofline_memory_bound():
    r = memory_bandwidth_roofline(1.0, 312.0, 2000.0)
    assert r["bound"] == "memory-bound"


def test_roofline_compute_bound():
    r = memory_bandwidth_roofline(1000.0, 10.0, 100.0)
    assert r["bound"] == "compute-bound"


def test_roofline_attainable_le_peak():
    r = memory_bandwidth_roofline(5.0, 100.0, 1000.0)
    assert r["attainable_Tflops"] <= 100.0


def test_gs_roofline_returns_valid():
    r = gs_algorithm_roofline(65536)
    assert r["operational_intensity"] > 0
    assert r["A100_analysis"]["bound"] in ("memory-bound", "compute-bound")


# ── SLURM ────────────────────────────────────────────────────────────

def test_slurm_script_contains_sbatch():
    script = slurm_job_script("test_job", n_gpus=1, time_hours=2)
    assert "#SBATCH" in script
    assert "--gres=gpu:1" in script


def test_slurm_script_job_name():
    script = slurm_job_script("my_experiment", n_gpus=4)
    assert "my_experiment" in script


def test_slurm_array_contains_array_directive():
    script = slurm_array_job(10)
    assert "--array=0-9" in script


def test_slurm_array_n_jobs():
    script = slurm_array_job(5)
    assert "SLURM_ARRAY_TASK_ID" in script


# ── cuda_memory_plan ──────────────────────────────────────────────────

def test_cuda_memory_large_model_doesnt_fit_4090():
    m = cuda_memory_plan(70000, batch_size=32, seq_len=2048, hidden_dim=8192)
    assert m["fits_on_RTX4090_24GB"] is False


def test_cuda_memory_small_model_fits_4090():
    m = cuda_memory_plan(100, batch_size=4, seq_len=128, hidden_dim=256)
    assert m["fits_on_RTX4090_24GB"] is True


def test_cuda_memory_total_positive():
    m = cuda_memory_plan(1000, batch_size=8, seq_len=512, hidden_dim=1024)
    assert m["total_GB"] > 0


# ── defense influence ─────────────────────────────────────────────────

def test_defense_influence_probabilities_positive():
    r = defense_research_influence(3, 7)
    assert 0 < r["P_phase1_win"] < 1
    assert 0 < r["P_phase2_win"] < r["P_phase1_win"]
    assert 0 < r["P_dod_transition"] < r["P_phase2_win"]


def test_defense_influence_more_submissions_higher_p():
    r5 = defense_research_influence(3, 7, n_submissions=5)
    r10 = defense_research_influence(3, 7, n_submissions=10)
    assert r10["P_phase1_win"] > r5["P_phase1_win"]


def test_defense_influence_invalid_trl():
    with pytest.raises(ValueError):
        defense_research_influence(0, 7)


# ── sympy 5 ──────────────────────────────────────────────────────────

def test_arch_sympy_5_count():
    eqs = computer_arch_sympy_5()
    assert len(eqs) == 5


def test_arch_sympy_5_are_sympy():
    for k, eq in computer_arch_sympy_5().items():
        assert isinstance(eq, sp.Basic), f"{k} not SymPy"
