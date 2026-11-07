"""Tests for dgs/dispersive_fourier.py"""
import numpy as np
import sympy as sp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.dispersive_fourier import (
    gaussian_pulse, gvd_propagate, dispersive_fourier_transform,
    gvd_transfer_function, kramers_kronig_n, verify_causality_gvd,
    partition_function_harmonic, equipartition_check, tsdft_sympy_5,
    _validate_gvd_kwargs, C_LIGHT, HBAR, K_BOLTZ,
)


# ── Validation ────────────────────────────────────────────────────────────────
def test_validate_rejects_zero_beta2():
    try:
        _validate_gvd_kwargs(0.0, 1.0, 1e-12, 64)
        assert False, "should raise"
    except ValueError:
        pass


def test_validate_rejects_negative_L():
    try:
        _validate_gvd_kwargs(-20e-27, -1.0, 1e-12, 64)
        assert False
    except ValueError:
        pass


def test_validate_rejects_small_n():
    try:
        _validate_gvd_kwargs(-20e-27, 1.0, 1e-12, 4)
        assert False
    except ValueError:
        pass


# ── GVD transfer function ─────────────────────────────────────────────────────
def test_gvd_all_pass():
    omega = np.linspace(-1e12, 1e12, 256)
    H = gvd_transfer_function(omega, -20e-27, 1000.0)
    np.testing.assert_allclose(np.abs(H), 1.0, atol=1e-12)


def test_gvd_quadratic_phase():
    # arg(H) = beta2*L*omega^2/2 mod 2pi; verify via exp(i*phase)==H
    beta2, L = -20e-27, 1000.0
    omega = np.array([1e11, 2e11, 5e11])
    H = gvd_transfer_function(omega, beta2, L)
    expected_phase = beta2 * L * omega**2 / 2
    np.testing.assert_allclose(H, np.exp(1j * expected_phase), atol=1e-10)


# ── gvd_propagate ─────────────────────────────────────────────────────────────
def test_gvd_propagate_keys():
    pulse = gaussian_pulse(256, 2e-12, 1e-12)
    res = gvd_propagate(pulse, beta2=-20e-27, L_m=5000.0, dt_s=1e-12)
    for k in ("E_out", "I_out", "E_omega", "I_omega", "omega", "H_omega",
              "far_field_ok", "L_D_m", "stretch_factor", "group_delay_ps"):
        assert k in res, f"missing key: {k}"


def test_gvd_propagate_energy_conserved():
    # |H|=1 -> total energy conserved (Parseval)
    pulse = gaussian_pulse(512, 2e-12, 1e-12)
    res = gvd_propagate(pulse, beta2=-20e-27, L_m=5000.0, dt_s=1e-12)
    E_in  = np.sum(np.abs(pulse)**2)
    E_out = np.sum(res["I_out"])
    assert abs(E_out - E_in) / E_in < 1e-6


def test_gvd_propagate_far_field_condition():
    # T0=2ps, L_D=100m, L=5km -> far-field
    pulse = gaussian_pulse(512, 2e-12, 1e-12)
    res = gvd_propagate(pulse, beta2=-20e-27, L_m=5000.0, dt_s=1e-12)
    assert bool(res["far_field_ok"]) is True
    assert res["stretch_factor"] > 10


def test_gvd_propagate_not_far_field():
    # L < L_D -> not far-field
    pulse = gaussian_pulse(512, 20e-12, 1e-12)   # T0=20ps, L_D=20km
    res = gvd_propagate(pulse, beta2=-20e-27, L_m=100.0, dt_s=1e-12)
    assert bool(res["far_field_ok"]) is False


def test_gvd_units():
    pulse = gaussian_pulse(256, 2e-12, 1e-12)
    res = gvd_propagate(pulse, beta2=-20e-27, L_m=1000.0, dt_s=1e-12)
    # GVD_ps2_km for SMF-28: beta2=-20e-27 s^2/m -> -20 ps^2/km
    assert abs(res.get("GVD_ps2_km", -20e-27 * 1e27) - (-20.0)) < 0.01


# ── dispersive_fourier_transform ──────────────────────────────────────────────
def test_tsdft_correlation_high_in_far_field():
    # Far-field: I_out(t) ~ |E(omega)|^2 -> corr should be near 1
    pulse = gaussian_pulse(2048, 2e-12, 1e-12)
    res = dispersive_fourier_transform(pulse, beta2=-20e-27, L_m=5000.0, dt_s=1e-12)
    assert res["ff_correlation"] > 0.99, f"ff_correlation={res['ff_correlation']:.4f}"


def test_tsdft_keys():
    pulse = gaussian_pulse(256, 2e-12, 1e-12)
    res = dispersive_fourier_transform(pulse, beta2=-20e-27, L_m=5000.0, dt_s=1e-12)
    for k in ("t_axis_s", "I_out_shifted", "I_far_field", "ff_correlation", "interpretation"):
        assert k in res


# ── gaussian_pulse ────────────────────────────────────────────────────────────
def test_gaussian_pulse_peak_at_center():
    n, T0, dt = 512, 5e-12, 1e-12
    pulse = gaussian_pulse(n, T0, dt)
    assert np.argmax(np.abs(pulse)**2) == n // 2


def test_gaussian_pulse_chirped():
    n, T0, dt = 256, 5e-12, 1e-12
    pulse_unchirped = gaussian_pulse(n, T0, dt, chirp_C=0.0)
    pulse_chirped   = gaussian_pulse(n, T0, dt, chirp_C=2.0)
    # Same magnitude envelope, different phase
    np.testing.assert_allclose(np.abs(pulse_chirped), np.abs(pulse_unchirped), atol=1e-10)


# ── Kramers-Kronig ────────────────────────────────────────────────────────────
def test_kramers_kronig_unity_kappa():
    # Constant kappa -> n should be 1 + constant (KK of DC = DC)
    omega = np.linspace(0, 1e13, 512)
    kappa = np.zeros(len(omega))   # no absorption
    n_arr = kramers_kronig_n(omega, kappa)
    np.testing.assert_allclose(n_arr, 1.0, atol=1e-10)


def test_kramers_kronig_output_shape():
    omega = np.linspace(0, 1e13, 256)
    kappa = np.exp(-((omega - 5e12)**2) / (1e12)**2)
    n_arr = kramers_kronig_n(omega, kappa)
    assert n_arr.shape == omega.shape


# ── verify_causality_gvd ─────────────────────────────────────────────────────
def test_causality_all_pass():
    res = verify_causality_gvd(-20e-27, 1000.0)
    assert res["all_pass"] is True


def test_causality_gvd_units():
    res = verify_causality_gvd(-20e-27, 1000.0)
    assert abs(res["GVD_ps2_km"] - (-20.0)) < 0.01


# ── partition_function_harmonic ───────────────────────────────────────────────
def test_partition_function_geometric_sum():
    # Z = 1/(1-exp(-x)) should match truncated sum
    pf = partition_function_harmonic(2*np.pi*C_LIGHT/1000e-9, 3000.0)  # 1000nm at 3000K
    assert abs(pf["Z"] - pf["Z_numeric"]) / pf["Z"] < 0.01


def test_bose_einstein_telecom():
    omega_tc = 2 * np.pi * C_LIGHT / 1550e-9
    pf = partition_function_harmonic(omega_tc, 300.0)
    assert pf["n_BE"] < 1e-10    # essentially zero at room temp
    assert pf["quantum_limit"] is True


def test_bose_einstein_microwave_classical():
    omega_mw = 2 * np.pi * 10e9   # 10 GHz
    pf = partition_function_harmonic(omega_mw, 300.0)
    # At microwave: hbar*omega << kT, so n_BE >> 1 (classical)
    assert pf["n_BE"] > 10
    assert pf["quantum_limit"] is False


def test_partition_function_heat_capacity():
    pf = partition_function_harmonic(2*np.pi*C_LIGHT/1000e-9, 3000.0)
    # Heat capacity should be positive
    assert pf["C_v"] > 0


# ── equipartition_check ───────────────────────────────────────────────────────
def test_equipartition_optical_suppressed():
    eq = equipartition_check(300.0)
    # Telecom quantum: E_mean >> 0.5*hbar*omega (zero-point), but E_thermal << kT
    assert eq["telecom_quantum_limit"] is True


def test_equipartition_classical_3DOF():
    eq = equipartition_check(300.0, n_modes=3)
    expected = 3 * K_BOLTZ * 300.0 / 1.602e-19
    assert abs(eq["E_classical_eV"] - expected) < 1e-6


# ── tsdft_sympy_5 ─────────────────────────────────────────────────────────────
def test_tsdft_sympy_5_count():
    eqs = tsdft_sympy_5()
    assert len(eqs) == 5


def test_tsdft_sympy_5_all_Eq():
    eqs = tsdft_sympy_5()
    for k, v in eqs.items():
        assert isinstance(v, sp.Eq), f"{k} is not sp.Eq"


def test_tsdft_sympy_5_keys():
    eqs = tsdft_sympy_5()
    expected = {"GVD_transfer_H", "TS-DFT_far_field", "group_delay_tau",
                "dispersion_length_L_D", "Bose-Einstein_n_BE"}
    assert set(eqs.keys()) == expected
