"""Tests for dgs/quantum_modulo.py — Shor's algorithm, QFT, QPE, GHz spectroscopy."""
import numpy as np
import sympy as sp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.quantum_modulo import (
    mod_exp, mod_exp_sequence, multiplicative_order, shor_factor_classical,
    qft_matrix, qft_apply, qft_inverse, period_from_qft,
    qpe_simulate, shor_algorithm_steps,
    spectroscopy_quantum_limit, quantum_modulo_sympy_5,
)


# ── Modular exponentiation ────────────────────────────────────────────────────

def test_mod_exp_basic():
    # 2^10 mod 1000 = 24
    assert mod_exp(2, 10, 1000) == 24


def test_mod_exp_fermat_little():
    # Fermat: a^(p-1) ≡ 1 mod p for prime p, gcd(a,p)=1
    # 3^6 mod 7 = 1
    assert mod_exp(3, 6, 7) == 1


def test_mod_exp_sequence_period():
    res = mod_exp_sequence(7, 15, 32)
    assert res["period_r"] == 4   # 7^4=2401≡1 mod 15
    assert res["periodic"] is True


def test_mod_exp_sequence_coprime():
    res = mod_exp_sequence(7, 15, 16)
    assert bool(res["gcd_check"]) is True   # gcd(7,15)=1


def test_multiplicative_order_7mod15():
    r = multiplicative_order(7, 15)
    assert r == 4   # 7^4=2401=160*15+1 -> 1 mod 15


def test_multiplicative_order_not_coprime():
    r = multiplicative_order(3, 9)   # gcd(3,9)=3
    assert r is None


def test_shor_classical_15():
    res = shor_factor_classical(15)
    assert res["factor"] is not None
    f = res["factor"]
    co = res.get("cofactor", 15 // f)
    assert f * co == 15
    assert res.get("verify", True) is True


def test_shor_classical_21():
    res = shor_factor_classical(21, seed=0)
    assert res["factor"] is not None
    f = res["factor"]
    assert 21 % f == 0


# ── QFT ───────────────────────────────────────────────────────────────────────

def test_qft_matrix_unitary():
    U = qft_matrix(3)   # 8x8
    # U @ U^dagger = I
    np.testing.assert_allclose(U @ U.conj().T, np.eye(8), atol=1e-10)


def test_qft_apply_inverse():
    state = np.array([1, 0, 0, 0], dtype=complex)
    out = qft_apply(state)
    recovered = qft_inverse(out)
    np.testing.assert_allclose(recovered, state, atol=1e-12)


def test_qft_uniform_to_basis():
    # QFT of uniform superposition |+> = (1,1,1,1)/2 -> |0>
    N = 4
    state = np.ones(N) / np.sqrt(N)
    out = qft_apply(state)
    # Peak at k=0
    assert np.argmax(np.abs(out)) == 0


def test_qft_preserves_norm():
    state = np.random.randn(8) + 1j*np.random.randn(8)
    state /= np.linalg.norm(state)
    out = qft_apply(state)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-10


def test_period_from_qft_finds_period():
    # f(x) = 7^x mod 15 has period 4
    f_vals = np.array([pow(7, x, 15) for x in range(32)], dtype=float)
    res = period_from_qft(f_vals, N_qft=32)
    # QFT should give peak at k=8 -> r=32/8=4
    assert res["r_estimate"] == 4


# ── QPE ───────────────────────────────────────────────────────────────────────

def test_qpe_exact_phase():
    # phi = 0.25 = 1/4; with 4 ancilla (N=16), k_best = 4, phi_est = 0.25
    res = qpe_simulate(0.25, n_ancilla=4)
    assert abs(res["phi_est"] - 0.25) < res["precision"]


def test_qpe_precision_scales():
    r1 = qpe_simulate(0.3, n_ancilla=4)
    r2 = qpe_simulate(0.3, n_ancilla=8)
    assert r2["precision"] < r1["precision"]


def test_qpe_high_success_prob():
    # Exact phase -> high success probability
    res = qpe_simulate(0.5, n_ancilla=6)
    assert res["success_prob"] > 0.5


def test_qpe_error_bounded():
    n = 8
    res = qpe_simulate(0.3, n_ancilla=n)
    assert res["error"] <= res["precision"]


# ── Full Shor structure ───────────────────────────────────────────────────────

def test_shor_steps_15():
    res = shor_algorithm_steps(15, 7)
    assert res.get("factor") in [3, 5]
    assert res.get("verify") is True


def test_shor_steps_even():
    res = shor_algorithm_steps(14, 7)
    assert res["factor"] == 2   # 14 is even -> trivial


def test_shor_steps_has_steps():
    res = shor_algorithm_steps(15, 7)
    assert isinstance(res["steps"], list)
    assert len(res["steps"]) >= 1


# ── GHz spectroscopy quantum limit ────────────────────────────────────────────

def test_spectroscopy_shot_noise_limited():
    res = spectroscopy_quantum_limit(1.0, 1000)
    # At 1550nm/300K, n_BE << 1 -> shot-noise limited
    assert bool(res["shot_noise_limited"]) is True
    assert res["n_BE_at_300K"] < 1e-10


def test_spectroscopy_snr_scales_sqrt():
    r1 = spectroscopy_quantum_limit(1.0, 100)
    r2 = spectroscopy_quantum_limit(1.0, 400)
    # SNR_avg ~ sqrt(n_shots)
    ratio = r2["SNR_avg"] / r1["SNR_avg"]
    assert abs(ratio - 2.0) < 0.01


def test_spectroscopy_heisenberg_bandwidth():
    res = spectroscopy_quantum_limit(1.0, 1)
    # delta_nu_min = 1/(4*pi*T_rep) where T_rep=1ns for 1GHz
    T_rep = 1e-9
    expected_MHz = 1 / (4 * np.pi * T_rep) / 1e6
    assert abs(res["delta_nu_min_MHz"] - expected_MHz) < 1.0


def test_spectroscopy_rep_rate():
    res = spectroscopy_quantum_limit(5.0, 1)
    assert res["spectra_per_sec"] == 5e9


# ── SymPy equations ───────────────────────────────────────────────────────────

def test_sympy_5_count():
    eqs = quantum_modulo_sympy_5()
    assert len(eqs) == 5


def test_sympy_5_all_eq():
    eqs = quantum_modulo_sympy_5()
    for k, v in eqs.items():
        assert isinstance(v, sp.Basic), f"{k} is not sympy Basic"
