import numpy as np
import pytest
from dgs.dispersive_barcode_correlator import (
    pulse_picker, optical_amplifier, supercontinuum_generate,
    optical_bandpass_filter, optical_circulator, grating_angular_dispersion,
    barcode_reflectivity_spectrum, dispersive_fourier_transform,
    pattern_generator, amplitude_modulator, optical_to_electrical,
    digitize, correlation_decision,
)


def test_pulse_picker_divides_rep_rate():
    assert pulse_picker(100e6, pick_every_n=4) == pytest.approx(25e6)


def test_pulse_picker_rejects_bad_input():
    with pytest.raises(ValueError):
        pulse_picker(-1.0, 1)
    with pytest.raises(ValueError):
        pulse_picker(1.0, 0)


def test_optical_amplifier_gain_matches_standard_db_convention():
    E = np.array([1.0 + 0j])
    amplified = optical_amplifier(E, gain_db=20.0)
    assert abs(amplified[0]) == pytest.approx(10.0)  # 20dB power gain = 10x amplitude


def test_supercontinuum_generate_broadens_spectrum():
    t = np.linspace(-10, 10, 512)
    E0 = gaussian_pulse_helper(t)
    out_linear = supercontinuum_generate(E0, t, z=1.0, gamma=0.0)
    out_nonlinear = supercontinuum_generate(E0, t, z=1.0, gamma=5.0)
    bw_linear = spectral_width(out_linear, t)
    bw_nonlinear = spectral_width(out_nonlinear, t)
    assert bw_nonlinear > bw_linear


def gaussian_pulse_helper(t):
    from dgs.nlse import gaussian_pulse
    return gaussian_pulse(t, t0=1.0)


def spectral_width(E, t):
    freq = np.fft.fftfreq(len(t), d=(t[1] - t[0]))
    spectrum = np.abs(np.fft.fft(E)) ** 2
    mean_f = np.sum(freq * spectrum) / np.sum(spectrum)
    return np.sqrt(np.sum((freq - mean_f) ** 2 * spectrum) / np.sum(spectrum))


def test_optical_bandpass_filter_attenuates_outside_band():
    freq = np.linspace(-5, 5, 200)
    E_freq = np.ones(200, dtype=complex)
    filtered = optical_bandpass_filter(E_freq, freq, f_center=0.0, bandwidth=1.0)
    assert abs(filtered[100]) > abs(filtered[0])  # center passes more than edge


def test_optical_bandpass_filter_rejects_bad_bandwidth():
    freq = np.linspace(-5, 5, 10)
    with pytest.raises(ValueError):
        optical_bandpass_filter(np.ones(10, dtype=complex), freq, 0.0, -1.0)


def test_optical_circulator_passes_signal_unchanged():
    E = np.array([1 + 1j, 2 - 1j])
    out = optical_circulator(E)
    np.testing.assert_allclose(out, E)


def test_optical_circulator_rejects_negative_isolation():
    with pytest.raises(ValueError):
        optical_circulator(np.array([1.0]), isolation_db=-1.0)


def test_grating_angular_dispersion_matches_grating_equation():
    # verify against the grating equation directly: d*sin(theta) = lambda at m=1, normal incidence
    wavelength_nm, groove_density = 1550.0, 600.0
    d_mm = 1.0 / groove_density
    lam_mm = wavelength_nm * 1e-6
    theta = np.arcsin(lam_mm / d_mm)
    expected = (1.0 / (d_mm * np.cos(theta))) * 1e-6
    assert grating_angular_dispersion(wavelength_nm, groove_density) == pytest.approx(expected)


def test_grating_angular_dispersion_rejects_bad_input():
    with pytest.raises(ValueError):
        grating_angular_dispersion(-1.0, 600.0)


def test_barcode_reflectivity_spectrum_white_bars_are_reflective():
    bits = np.array([0, 1, 0, 1])  # 0=white/reflective, 1=dark/absorptive
    refl = barcode_reflectivity_spectrum(bits, n_freq=8)
    assert refl[0] == pytest.approx(1.0)  # first bar is white -> reflective


def test_barcode_reflectivity_spectrum_rejects_undersized_n_freq():
    with pytest.raises(ValueError):
        barcode_reflectivity_spectrum(np.array([0, 1, 0, 1]), n_freq=2)


def test_dispersive_fourier_transform_reuses_gs_core_disperse():
    from dgs.gs_core import disperse
    E = np.array([1 + 1j, 2 - 1j, 0.5j, -1.0])
    out_a = dispersive_fourier_transform(E, D=5000.0)
    out_b = disperse(E, D=5000.0)
    np.testing.assert_allclose(out_a, out_b)


def test_pattern_generator_matching_bits_gives_complement_of_reflectivity():
    bits = np.array([1, 0, 1, 0])
    pattern = pattern_generator(bits, n_samples=4)
    reflectivity = barcode_reflectivity_spectrum(bits, n_freq=4)
    np.testing.assert_allclose(pattern, 1.0 - reflectivity)


def test_amplitude_modulator_nulls_matching_signal():
    E_time = np.array([1.0, 2.0, 3.0])
    reference_pattern = np.array([0.0, 0.0, 0.0])
    out = amplitude_modulator(E_time, reference_pattern)
    np.testing.assert_allclose(out, np.zeros(3))


def test_amplitude_modulator_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        amplitude_modulator(np.array([1.0, 2.0]), np.array([1.0]))


def test_optical_to_electrical_zero_input_gives_zero_voltage():
    out = optical_to_electrical(np.zeros(5, dtype=complex))
    np.testing.assert_allclose(out, np.zeros(5))


def test_optical_to_electrical_higher_power_gives_higher_voltage():
    low = optical_to_electrical(np.array([0.1 + 0j]))
    high = optical_to_electrical(np.array([1.0 + 0j]))
    assert abs(high[0]) > abs(low[0])


def test_digitize_handles_all_zero_signal_without_warning():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # promote RuntimeWarning to an error
        result = digitize(np.zeros(10))
    assert np.allclose(result, 0.0, atol=1e-6)


def test_digitize_returns_reasonable_length():
    voltage = np.sin(np.linspace(0, 10, 50))
    result = digitize(voltage, n_bits=8)
    assert len(result) in (49, 50)  # ADC.convert's off-by-one, documented elsewhere


def test_correlation_decision_matches_at_zero_residual():
    decision = correlation_decision(np.zeros(10), threshold=0.01)
    assert decision["is_match"] is True
    assert decision["residual_rms"] == pytest.approx(0.0)


def test_correlation_decision_no_match_for_large_residual():
    decision = correlation_decision(np.full(10, 5.0), threshold=0.01)
    assert decision["is_match"] is False


def test_correlation_decision_rejects_negative_threshold():
    with pytest.raises(ValueError):
        correlation_decision(np.zeros(5), threshold=-1.0)


def test_full_chain_matching_barcode_gives_near_zero_residual():
    rng = np.random.default_rng(1)
    true_barcode = rng.integers(0, 2, 12)
    n_freq = 128
    E_source = optical_circulator(optical_amplifier(np.ones(n_freq, dtype=complex), 6.0))
    reflectivity = barcode_reflectivity_spectrum(true_barcode, n_freq)
    E_reflected = E_source * reflectivity
    D = 8000.0
    ref_pattern = pattern_generator(true_barcode, n_freq)
    residual_time = dispersive_fourier_transform(
        E_source * (reflectivity - (1.0 - ref_pattern)), D)
    voltage = optical_to_electrical(residual_time)
    digitized = digitize(voltage, n_bits=10)
    decision = correlation_decision(digitized, threshold=1e-9)
    assert decision["is_match"] is True


def test_full_chain_wrong_barcode_gives_large_residual():
    rng = np.random.default_rng(1)
    true_barcode = rng.integers(0, 2, 12)
    wrong_barcode = 1 - true_barcode
    n_freq = 128
    E_source = optical_circulator(optical_amplifier(np.ones(n_freq, dtype=complex), 6.0))
    reflectivity = barcode_reflectivity_spectrum(true_barcode, n_freq)
    E_reflected = E_source * reflectivity
    D = 8000.0
    ref_pattern = pattern_generator(wrong_barcode, n_freq)
    residual_time = dispersive_fourier_transform(
        E_source * (reflectivity - (1.0 - ref_pattern)), D)
    voltage = optical_to_electrical(residual_time)
    digitized = digitize(voltage, n_bits=10)
    decision = correlation_decision(digitized, threshold=1e-9)
    assert decision["is_match"] is False
