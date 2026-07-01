import numpy as np
import pytest
import sympy as sp
from dgs.analog_design import (
    inverting_amplifier, noninverting_amplifier, summing_amplifier,
    op_amp_integrator, op_amp_differentiator,
    first_order_lpf, butterworth_lpf_2nd_order, butterworth_order_for_spec,
    johnson_nyquist_noise, noise_figure_cascaded,
    adc_specs, enob_from_snr,
    coax_impedance, transmission_line_reflection,
    gs_receiver_analog_frontend,
    analog_design_sympy_5,
)


# ── inverting amplifier ───────────────────────────────────────────────

def test_inverting_gain():
    r = inverting_amplifier(1e3, 10e3, 1.0)
    assert r["Av"] == pytest.approx(-10.0)
    assert r["V_out_V"] == pytest.approx(-10.0)


def test_inverting_unity_gain():
    r = inverting_amplifier(1e3, 1e3, 0.5)
    assert r["Av"] == pytest.approx(-1.0)


def test_inverting_invalid():
    with pytest.raises(ValueError):
        inverting_amplifier(0, 10e3, 1.0)


# ── non-inverting amplifier ───────────────────────────────────────────

def test_noninverting_gain():
    r = noninverting_amplifier(1e3, 9e3, 1.0)
    assert r["Av"] == pytest.approx(10.0)


def test_noninverting_unity_buffer():
    r = noninverting_amplifier(1e3, 0, 0.7)
    assert r["Av"] == pytest.approx(1.0)
    assert r["V_out_V"] == pytest.approx(0.7)


# ── summing amplifier ─────────────────────────────────────────────────

def test_summing_two_inputs():
    r = summing_amplifier(10e3, [(1.0, 10e3), (2.0, 10e3)])
    assert r["V_out_V"] == pytest.approx(-3.0)


def test_summing_invalid_empty():
    with pytest.raises(ValueError):
        summing_amplifier(10e3, [])


# ── integrator ────────────────────────────────────────────────────────

def test_integrator_output():
    r = op_amp_integrator(10e3, 1e-9, 1.0, 1e-6)  # R=10k, C=1nF, Vin=1V, t=1us
    tau = 10e3 * 1e-9
    expected = -1.0 * 1e-6 / tau
    assert r["V_out_V"] == pytest.approx(expected, rel=1e-6)


def test_integrator_zero_time():
    r = op_amp_integrator(10e3, 1e-9, 1.0, 0.0)
    assert r["V_out_V"] == pytest.approx(0.0)


def test_integrator_invalid():
    with pytest.raises(ValueError):
        op_amp_integrator(0, 1e-9, 1.0, 1e-6)


# ── differentiator ────────────────────────────────────────────────────

def test_differentiator_output():
    r = op_amp_differentiator(10e3, 1e-9, 1e6)  # dV/dt=1 MV/s
    expected = -10e3 * 1e-9 * 1e6
    assert r["V_out_V"] == pytest.approx(expected, rel=1e-6)


# ── first-order LPF ───────────────────────────────────────────────────

def test_lpf_cutoff_3dB():
    R = 10e3
    C = 1e-9
    f_c = 1 / (2 * np.pi * R * C)
    r = first_order_lpf(R, C, f_c)
    assert r["H_magnitude"] == pytest.approx(1/np.sqrt(2), rel=1e-4)
    assert bool(r["at_3dB"]) is True


def test_lpf_passband_near_unity():
    r = first_order_lpf(10e3, 1e-9, 1.0)   # very low frequency
    assert r["H_magnitude"] == pytest.approx(1.0, abs=1e-3)


def test_lpf_stopband_attenuated():
    R = 10e3
    C = 1e-9
    f_c = 1 / (2 * np.pi * R * C)
    r = first_order_lpf(R, C, f_c * 100)   # 100x cutoff
    assert r["H_magnitude"] < 0.02


def test_lpf_invalid():
    with pytest.raises(ValueError):
        first_order_lpf(0, 1e-9, 1e3)


# ── Butterworth 2nd order ─────────────────────────────────────────────

def test_butterworth_cutoff_3dB():
    r = butterworth_lpf_2nd_order(1e6, 1e6)
    assert r["H_magnitude"] == pytest.approx(1/np.sqrt(2), rel=0.01)


def test_butterworth_rolloff_40dB_decade():
    r10 = butterworth_lpf_2nd_order(1e6, 10e6)
    r100 = butterworth_lpf_2nd_order(1e6, 100e6)
    # 10x frequency -> 40 dB more attenuation (2nd order)
    rolloff = r10["H_dB"] - r100["H_dB"]
    assert rolloff == pytest.approx(40.0, abs=2.0)


def test_butterworth_passband():
    r = butterworth_lpf_2nd_order(1e6, 100e3)
    assert r["H_magnitude"] > 0.95


# ── Butterworth order spec ────────────────────────────────────────────

def test_butterworth_order_positive():
    r = butterworth_order_for_spec(1e6, 10e6, 3, 40)
    assert r["n_order"] >= 1


def test_butterworth_order_stricter_spec_needs_higher_order():
    r_loose = butterworth_order_for_spec(1e6, 10e6, 3, 40)
    r_tight = butterworth_order_for_spec(1e6, 10e6, 3, 60)
    assert r_tight["n_order"] >= r_loose["n_order"]


def test_butterworth_order_invalid():
    with pytest.raises(ValueError):
        butterworth_order_for_spec(0, 10e6, 3, 40)


# ── Johnson-Nyquist noise ─────────────────────────────────────────────

def test_johnson_noise_increases_with_R():
    n1 = johnson_nyquist_noise(50, 1e9)
    n100 = johnson_nyquist_noise(5000, 1e9)
    assert n100["v_n_rms_V"] > n1["v_n_rms_V"]


def test_johnson_noise_increases_with_BW():
    n1 = johnson_nyquist_noise(50, 1e6)
    n2 = johnson_nyquist_noise(50, 1e9)
    assert n2["v_n_rms_V"] > n1["v_n_rms_V"]


def test_johnson_noise_zero_BW():
    n = johnson_nyquist_noise(50, 0)
    assert n["v_n_rms_V"] == pytest.approx(0.0)


# ── cascaded noise figure ─────────────────────────────────────────────

def test_friis_dominated_by_first_stage():
    # Stage 1: NF=2 dB, G=20 dB; Stage 2: NF=10 dB, G=0 dB
    r = noise_figure_cascaded([2.0, 10.0], [20.0, 0.0])
    # Total NF should be close to first stage NF=2 dB (high gain suppresses stage 2)
    assert r["NF_total_dB"] < 3.0


def test_friis_single_stage():
    r = noise_figure_cascaded([5.0], [10.0])
    assert r["NF_total_dB"] == pytest.approx(5.0, abs=0.01)


def test_friis_mismatched_lengths():
    with pytest.raises(ValueError):
        noise_figure_cascaded([2.0], [10.0, 5.0])


# ── ADC specs ─────────────────────────────────────────────────────────

def test_adc_snr_8bit():
    a = adc_specs(8, 2.0, 1e9)
    assert a["SNR_ideal_dB"] == pytest.approx(6.02 * 8 + 1.76, rel=1e-4)


def test_adc_lsb_12bit():
    a = adc_specs(12, 4.096, 1e9)
    expected_LSB = 4.096 / 2**12
    assert a["LSB_V"] == pytest.approx(expected_LSB, rel=1e-6)


def test_adc_nyquist_half_sample():
    a = adc_specs(8, 2.0, 20e9)
    assert a["f_nyquist_Hz"] == pytest.approx(10e9)


def test_adc_invalid():
    with pytest.raises(ValueError):
        adc_specs(0, 2.0, 1e9)


def test_enob_from_snr():
    # ENOB(SNR=49.92 dB) should be ~8 bits
    r = enob_from_snr(6.02 * 8 + 1.76)
    assert r["ENOB"] == pytest.approx(8.0, abs=0.01)


# ── coax impedance ────────────────────────────────────────────────────

def test_coax_50_ohm():
    # RG-58: inner d~0.9mm, outer D~2.95mm, eps_r~2.25 -> ~50 Ohm
    r = coax_impedance(2.95e-3, 0.9e-3, 2.25)
    assert 40 < r["Z0_ohm"] < 60   # roughly 50 Ohm


def test_coax_invalid_geometry():
    with pytest.raises(ValueError):
        coax_impedance(1e-3, 5e-3, 2.25)


# ── transmission line reflection ──────────────────────────────────────

def test_matched_load_no_reflection():
    r = transmission_line_reflection(50.0, 50.0)
    assert r["Gamma_mag"] == pytest.approx(0.0, abs=1e-10)
    assert r["VSWR"] == pytest.approx(1.0, abs=1e-6)


def test_open_circuit_full_reflection():
    r = transmission_line_reflection(1e12, 50.0)
    assert r["Gamma_mag"] == pytest.approx(1.0, abs=1e-3)


def test_short_circuit_full_reflection():
    r = transmission_line_reflection(1e-6, 50.0)
    assert r["Gamma_mag"] == pytest.approx(1.0, abs=1e-3)


# ── GS receiver front-end ─────────────────────────────────────────────

def test_gs_frontend_signal_positive():
    fe = gs_receiver_analog_frontend()
    assert fe["V_signal_mV"] > 0


def test_gs_frontend_snr_positive():
    fe = gs_receiver_analog_frontend()
    assert fe["SNR_analog_dB"] > 0


def test_gs_frontend_oversampled():
    fe = gs_receiver_analog_frontend(BW_GHz=5.0, f_sample_GHz=20.0)
    assert fe["oversampled_by"] == pytest.approx(2.0)


# ── sympy 5 ──────────────────────────────────────────────────────────

def test_analog_sympy_5_count():
    eqs = analog_design_sympy_5()
    assert len(eqs) == 5


def test_analog_sympy_5_types():
    for k, eq in analog_design_sympy_5().items():
        assert isinstance(eq, sp.Basic), f"{k} not SymPy"
