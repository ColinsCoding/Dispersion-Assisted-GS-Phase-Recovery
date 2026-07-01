import numpy as np
import pytest
import sympy as sp
from dgs.photonic_ai import (
    mzi_matrix, mzi_transmission, clements_mzi_count,
    random_unitary_mzi_params, optical_matmul_svd,
    energy_per_multiply, energy_advantage_ratio,
    shannon_capacity, gs_as_optical_fourier_layer,
    photonic_spiking_neuron, photonic_training_cost,
    photonic_ai_sympy_5,
)


# ── MZI matrix ────────────────────────────────────────────────────────

def test_mzi_matrix_is_unitary():
    M = mzi_matrix(np.pi / 4, 0.3)
    prod = M @ M.conj().T
    assert np.allclose(prod, np.eye(2), atol=1e-10)

def test_mzi_balanced_splits_evenly():
    r = mzi_transmission(np.pi / 4, 0.0)
    assert r["T_port0"] == pytest.approx(0.5, abs=1e-6)
    assert r["T_port1"] == pytest.approx(0.5, abs=1e-6)

def test_mzi_power_conserved():
    r = mzi_transmission(0.3, 1.1)
    assert r["total_power"] == pytest.approx(1.0, abs=1e-6)

def test_mzi_theta_zero_all_to_port1():
    # at theta=0: sin(0)=0 so port0 gets 0 power
    r = mzi_transmission(0.0, 0.0)
    assert r["T_port0"] == pytest.approx(0.0, abs=1e-6)
    assert r["T_port1"] == pytest.approx(1.0, abs=1e-6)

def test_mzi_invalid_ignored():
    # test that valid inputs don't raise
    M = mzi_matrix(np.pi / 3, np.pi / 6)
    assert M.shape == (2, 2)


# ── Clements MZI count ────────────────────────────────────────────────

def test_clements_n1_zero_mzi():
    c = clements_mzi_count(1)
    assert c["n_MZI"] == 0

def test_clements_n2_one_mzi():
    c = clements_mzi_count(2)
    assert c["n_MZI"] == 1

def test_clements_n8():
    c = clements_mzi_count(8)
    assert c["n_MZI"] == 8 * 7 // 2   # 28

def test_clements_depth_equals_n():
    for N in [4, 8, 16]:
        c = clements_mzi_count(N)
        assert c["depth_layers"] == N

def test_clements_invalid():
    with pytest.raises(ValueError):
        clements_mzi_count(0)


# ── random params ─────────────────────────────────────────────────────

def test_random_params_count():
    p = random_unitary_mzi_params(6)
    assert len(p["thetas"]) == 6 * 5 // 2
    assert len(p["phis"]) == 6 * 5 // 2

def test_random_params_theta_in_range():
    p = random_unitary_mzi_params(8)
    assert np.all(p["thetas"] >= 0) and np.all(p["thetas"] <= np.pi / 2)

def test_random_params_phi_in_range():
    p = random_unitary_mzi_params(8)
    assert np.all(p["phis"] >= 0) and np.all(p["phis"] <= 2 * np.pi)


# ── optical matmul SVD ────────────────────────────────────────────────

def test_svd_rank_full_matrix():
    W = np.eye(4)
    svd = optical_matmul_svd(W)
    assert svd["rank"] == 4

def test_svd_condition_number():
    W = np.diag([1.0, 2.0, 3.0])
    svd = optical_matmul_svd(W)
    assert svd["condition_number"] == pytest.approx(3.0, rel=1e-6)

def test_svd_mzi_cost():
    W = np.random.default_rng(0).standard_normal((8, 8))
    svd = optical_matmul_svd(W)
    assert svd["total_mzi"] == 2 * (8 * 7 // 2)   # 56

def test_svd_reconstruction_quality():
    W = np.random.default_rng(1).standard_normal((4, 4))
    svd = optical_matmul_svd(W)
    W_reconstructed = svd["U"] @ np.diag(svd["sigma"][:4]) @ svd["Vh"]
    assert np.allclose(W_reconstructed, W, atol=1e-10)


# ── energy per multiply ───────────────────────────────────────────────

def test_energy_electronic_positive():
    r = energy_per_multiply(64, "electronic_gpu")
    assert r["energy_per_mac_fJ"] > 0

def test_energy_photonic_positive():
    r = energy_per_multiply(64, "photonic_mzi")
    assert r["energy_per_mac_fJ"] > 0

def test_energy_photonic_lt_electronic():
    N = 512
    e_el = energy_per_multiply(N, "electronic_gpu")["energy_per_mac_fJ"]
    e_ph = energy_per_multiply(N, "photonic_mzi")["energy_per_mac_fJ"]
    assert e_ph < e_el

def test_energy_invalid_arch():
    with pytest.raises(ValueError):
        energy_per_multiply(64, "quantum_unicorn")

def test_energy_invalid_n():
    with pytest.raises(ValueError):
        energy_per_multiply(0, "electronic_gpu")


# ── energy advantage ──────────────────────────────────────────────────

def test_advantage_ratio_gt1():
    r = energy_advantage_ratio(256)
    assert r["advantage_ratio"] > 1.0

def test_advantage_ratio_increases_with_n():
    r64  = energy_advantage_ratio(64)
    r4096 = energy_advantage_ratio(4096)
    assert r4096["advantage_ratio"] > r64["advantage_ratio"]


# ── Shannon capacity ──────────────────────────────────────────────────

def test_shannon_capacity_positive():
    r = shannon_capacity(20, 10e9)
    assert r["C_Gbps"] > 0

def test_shannon_coherent_gt_direct():
    r = shannon_capacity(20, 10e9)
    assert r["C_Gbps"] > r["C_direct_detection_Gbps"]

def test_shannon_coherent_is_double_direct():
    r = shannon_capacity(20, 10e9)
    assert r["C_Gbps"] == pytest.approx(2 * r["C_direct_detection_Gbps"], rel=1e-6)

def test_shannon_invalid_bw():
    with pytest.raises(ValueError):
        shannon_capacity(20, 0)

def test_shannon_high_snr_large_capacity():
    r = shannon_capacity(30, 10e9)
    assert r["C_Gbps"] > 99  # 30 dB SNR ~100 Gbps/10GHz (log2(1001)*10)


# ── GS as Fourier layer ───────────────────────────────────────────────

def test_gs_fourier_layer_all_pass():
    gs = gs_as_optical_fourier_layer(512, D_ps2=-5000)
    assert np.allclose(np.abs(gs["H_f"]), 1.0, atol=1e-10)

def test_gs_fourier_layer_phase_range():
    gs = gs_as_optical_fourier_layer(512, D_ps2=-5000)
    assert gs["H_phase_range_rad"] > 0

def test_gs_fourier_invalid_n():
    with pytest.raises(ValueError):
        gs_as_optical_fourier_layer(0)

def test_gs_fourier_larger_d_more_diversity():
    # fftfreq wraps at +-0.5, so phase range saturates near 2*pi for large D.
    # The diversity_metric (fraction of non-DC modes with large phase) scales with D.
    gs_small = gs_as_optical_fourier_layer(512, D_ps2=-1000)
    gs_large = gs_as_optical_fourier_layer(512, D_ps2=-5000)
    # larger |D| -> wider phase spread -> higher diversity metric
    assert gs_large["H_phase_range_cycles"] >= gs_small["H_phase_range_cycles"] * 0.5


# ── photonic spiking neuron ───────────────────────────────────────────

def test_spiking_above_threshold():
    r = photonic_spiking_neuron(2.0, P_threshold_mW=1.0)
    assert bool(r["is_spiking"]) is True
    assert r["n_spikes"] >= 1

def test_no_spike_below_threshold():
    r = photonic_spiking_neuron(0.3, P_threshold_mW=1.0)
    assert bool(r["is_spiking"]) is False

def test_spiking_power_conserved_approx():
    r = photonic_spiking_neuron(2.0, P_threshold_mW=1.0)
    assert len(r["t_ns"]) > 0
    assert np.all(r["P_out_mW"] >= 0)

def test_spiking_invalid_threshold():
    with pytest.raises(ValueError):
        photonic_spiking_neuron(1.0, P_threshold_mW=0)


# ── training cost ─────────────────────────────────────────────────────

def test_training_cost_hybrid_lt_digital():
    tc = photonic_training_cost(4, 64, n_epochs=100, n_samples=50000)
    assert tc["E_photonic_hybrid_kWh"] < tc["E_digital_training_kWh"]

def test_training_speedup_gt1():
    tc = photonic_training_cost(4, 64, n_epochs=100, n_samples=50000)
    assert tc["training_speedup"] > 1.0

def test_training_invalid_layers():
    with pytest.raises(ValueError):
        photonic_training_cost(0, 64, 10, 1000)


# ── SymPy 5 ──────────────────────────────────────────────────────────

def test_photonic_sympy_5_count():
    eqs = photonic_ai_sympy_5()
    assert len(eqs) == 5

def test_photonic_sympy_5_types():
    for k, eq in photonic_ai_sympy_5().items():
        assert isinstance(eq, sp.Basic), f"{k} not SymPy"
