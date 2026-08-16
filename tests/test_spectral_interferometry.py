import numpy as np
import pytest
from dgs.spectral_interferometry import (
    minimum_phase_from_log_magnitude, random_causal_profile,
    spectral_interferogram, valid_tau_range, hilbert_demodulate,
    quantize_enob, demodulation_rmse_vs_enob, spectral_regression_output_size,
)


def _rmse_report(magnitude, phase, result):
    mag_rmse = float(np.sqrt(np.mean((result["magnitude_est"] - magnitude) ** 2)))
    offset = np.angle(np.mean(np.exp(1j * (phase - result["phase_est"]))))
    aligned = np.angle(np.exp(1j * (result["phase_est"] + offset - phase)))
    phase_rmse_deg = float(np.degrees(np.sqrt(np.mean(aligned ** 2))))
    return mag_rmse, phase_rmse_deg


def test_minimum_phase_from_log_magnitude_zero_is_zero_phase():
    phase = minimum_phase_from_log_magnitude(np.zeros(64))
    np.testing.assert_allclose(phase, 0.0, atol=1e-10)


def test_random_causal_profile_shapes_and_positivity():
    rng = np.random.default_rng(0)
    magnitude, phase = random_causal_profile(128, rng)
    assert magnitude.shape == (128,)
    assert phase.shape == (128,)
    assert np.all(magnitude > 0)


def test_random_causal_profile_rejects_small_n():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        random_causal_profile(4, rng)


def test_spectral_interferogram_rejects_zero_tau():
    n = 32
    omega = np.linspace(-1, 1, n)
    E = np.ones(n, dtype=complex)
    with pytest.raises(ValueError):
        spectral_interferogram(E, E, omega, tau=0.0)


def test_spectral_interferogram_shape_mismatch_raises():
    omega = np.linspace(-1, 1, 32)
    E_short = np.ones(16, dtype=complex)
    E_full = np.ones(32, dtype=complex)
    with pytest.raises(ValueError):
        spectral_interferogram(E_short, E_full, omega, tau=10.0)


def test_valid_tau_range_is_ordered_and_positive():
    omega = np.linspace(-1, 1, 256)
    tau_min, tau_max = valid_tau_range(omega)
    assert 0 < tau_min < tau_max


def test_valid_tau_range_rejects_too_few_samples():
    omega = np.linspace(-1, 1, 4)
    with pytest.raises(ValueError):
        valid_tau_range(omega)


def test_hilbert_demodulate_recovers_profile_within_valid_tau_range():
    rng = np.random.default_rng(1)
    n = 256
    omega = np.linspace(-1.0, 1.0, n)
    tau_min, tau_max = valid_tau_range(omega)
    tau = (tau_min + tau_max) / 2.0
    magnitude, phase = random_causal_profile(n, rng)
    E_test = magnitude * np.exp(1j * phase)
    E_ref = np.ones(n, dtype=complex)
    S = spectral_interferogram(E_test, E_ref, omega, tau)
    result = hilbert_demodulate(S, omega, tau, E_ref=E_ref)
    mag_rmse, phase_rmse_deg = _rmse_report(magnitude, phase, result)
    assert mag_rmse < 0.05
    assert phase_rmse_deg < 5.0


def test_hilbert_demodulate_outside_valid_tau_range_degrades():
    # tau below the valid range should overlap the background band and be
    # noticeably worse than a tau safely inside the range -- this is the
    # exact failure mode valid_tau_range exists to warn against.
    rng = np.random.default_rng(1)
    n = 256
    omega = np.linspace(-1.0, 1.0, n)
    tau_min, tau_max = valid_tau_range(omega)
    good_tau = (tau_min + tau_max) / 2.0
    bad_tau = tau_min / 4.0
    magnitude, phase = random_causal_profile(n, rng)
    E_test = magnitude * np.exp(1j * phase)
    E_ref = np.ones(n, dtype=complex)

    S_good = spectral_interferogram(E_test, E_ref, omega, good_tau)
    result_good = hilbert_demodulate(S_good, omega, good_tau, E_ref=E_ref)
    _, phase_rmse_good = _rmse_report(magnitude, phase, result_good)

    S_bad = spectral_interferogram(E_test, E_ref, omega, bad_tau)
    result_bad = hilbert_demodulate(S_bad, omega, bad_tau, E_ref=E_ref)
    _, phase_rmse_bad = _rmse_report(magnitude, phase, result_bad)

    assert phase_rmse_bad > phase_rmse_good


def test_hilbert_demodulate_rejects_zero_tau():
    n = 32
    omega = np.linspace(-1, 1, n)
    S = np.ones(n)
    with pytest.raises(ValueError):
        hilbert_demodulate(S, omega, tau=0.0)


def test_quantize_enob_reduces_distinct_levels():
    signal = np.linspace(0, 1, 1000)
    q = quantize_enob(signal, enob=3)
    assert len(np.unique(q)) <= 2 ** 3 + 1


def test_quantize_enob_rejects_nonpositive():
    with pytest.raises(ValueError):
        quantize_enob(np.array([1.0, 2.0]), enob=0)


def test_quantize_enob_constant_signal_unchanged():
    signal = np.full(10, 3.0)
    q = quantize_enob(signal, enob=4)
    np.testing.assert_allclose(q, signal)


def test_demodulation_rmse_vs_enob_improves_with_more_bits():
    result = demodulation_rmse_vs_enob(n_trials=5, enob_values=[2, 10], n=128, rng_seed=0)
    assert result["mean_phase_rmse_rad"][1] < result["mean_phase_rmse_rad"][0]
    assert result["mean_magnitude_rmse"][1] < result["mean_magnitude_rmse"][0]


def test_demodulation_rmse_vs_enob_rejects_bad_input():
    with pytest.raises(ValueError):
        demodulation_rmse_vs_enob(n_trials=0, enob_values=[4])
    with pytest.raises(ValueError):
        demodulation_rmse_vs_enob(n_trials=5, enob_values=[])


def test_spectral_regression_output_size_matches_eq1():
    # Eq. (1): n_output = 2*|D|*delta_lambda*Fs
    D_ps_per_nm, delta_lambda_nm, Fs_hz = -1000.0, 20.0, 50e9
    n_out = spectral_regression_output_size(D_ps_per_nm, delta_lambda_nm, Fs_hz)
    delay_s = abs(D_ps_per_nm) * delta_lambda_nm * 1e-12
    expected = 2.0 * delay_s * Fs_hz
    assert n_out == pytest.approx(expected)


def test_spectral_regression_output_size_rejects_nonpositive_fs():
    with pytest.raises(ValueError):
        spectral_regression_output_size(-1000.0, 20.0, Fs_hz=0.0)
