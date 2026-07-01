"""Tests for dgs/coppinger1999.py — Coppinger, Bhushan, Jalali 1999 IEEE MTT"""
import numpy as np
import sympy as sp
import pytest
from dgs.coppinger1999 import (
    eq1_chirped_pulse_time, eq2_chirped_pulse_freq,
    eq8_stretch_factor, eq9_dispersion_penalty,
    verify_fig2_stretch_factors, freq_to_time_mapping,
    simulate_fig2, dispersive_transfer_function,
    appendix_gamma_definition, mzm_cmos_specs,
    mzm_transfer_function, eq7_detected_intensity,
    derive_eq1_from_convolution,
    t, f, tau, beta2, L1, L2,
)


def test_derive_eq1_convolution_keys():
    result = derive_eq1_from_convolution(verbose=False)
    for key in ['complex_width_W', 'E_exact_time', 'E_approx_eq1',
                'output_pulse_width_tau', 'instantaneous_frequency']:
        assert key in result


def test_derive_eq1_complex_width():
    import sympy as sp
    result = derive_eq1_from_convolution(verbose=False)
    tau0 = sp.Symbol('tau_0', positive=True)
    W = result['complex_width_W']
    # W = tau0^2 - j*2*beta2*L1 -- imaginary part should be negative (for real beta2>0)
    # Real part should be tau0^2
    assert sp.re(W) == tau0**2 or sp.simplify(sp.re(W) - tau0**2) == 0


def test_derive_eq1_instantaneous_freq_linear():
    import sympy as sp
    result = derive_eq1_from_convolution(verbose=False)
    f_inst = result['instantaneous_frequency']
    # f_inst = t / (2*pi*L1*beta2) -- linear in t, derivative is constant
    df_dt = sp.diff(f_inst, t)
    assert sp.simplify(df_dt + 1/(2*sp.pi*L1*beta2)) == 0  # df/dt = -1/(2*pi*L1*b2)


def test_eq1_is_gaussian():
    E = eq1_chirped_pulse_time()
    # At t=0: exp(0) = 1
    assert E.subs(t, 0) == 1


def test_eq2_at_f0():
    E = eq2_chirped_pulse_freq()
    # At f=0 should be sqrt(pi)/sqrt(tau^-2 + j/2L1b2)
    val = E.subs(f, 0)
    expected = sp.sqrt(sp.pi) / sp.sqrt(tau**(-2) + sp.I/(2*L1*beta2))
    assert sp.simplify(val - expected) == 0


def test_stretch_factors_fig2():
    results = verify_fig2_stretch_factors()
    for r in results:
        assert r['match'], f"Stretch factor mismatch: {r}"


def test_stretch_factor_formula():
    assert eq8_stretch_factor(1.0, 0.0) == 1.0
    assert eq8_stretch_factor(1.1, 2.2) == pytest.approx(3.0, rel=0.01)
    assert eq8_stretch_factor(1.1, 5.5) == pytest.approx(6.0, rel=0.01)
    assert eq8_stretch_factor(1.1, 7.6) == pytest.approx(7.909, rel=0.01)


def test_dispersion_penalty_at_zero_freq():
    # At fm=0: penalty = cos(0)^2 = 1 (no penalty)
    pen = eq9_dispersion_penalty(L2_km=5.5, beta2_ps2km=-20, fm_ghz=0, M=6)
    assert abs(pen - 1.0) < 1e-10


def test_dispersion_penalty_range():
    pen = eq9_dispersion_penalty(5.5, -20, 30, 6)
    assert 0.0 <= pen <= 1.0


def test_transfer_function_at_f0():
    H = dispersive_transfer_function(L2)
    # At f=0: exp(0) = 1
    assert H.subs(f, 0) == 1


def test_transfer_function_is_unitary():
    # |exp(j*phase)| = 1 always
    H = dispersive_transfer_function(L2)
    # Verify symbolically: H * H.conjugate() = 1
    product = sp.simplify(H * sp.conjugate(H))
    # Should be 1 (assuming real L2, beta2, f)
    assert product == 1


def test_gamma_definition():
    g = appendix_gamma_definition()
    # At L1->0: gamma -> 1
    g0 = g.subs(L1, 0)
    assert g0 == 1


def test_freq_to_time_sign():
    # Negative beta2: positive frequency -> negative time delay
    t_ps = freq_to_time_mapping(beta2_ps2km=-20, L_km=5.5, f_hz=30e9)
    assert t_ps < 0, "Negative GVD should give negative group delay for positive freq"


def test_freq_to_time_magnitude():
    # t = 2*pi*beta2*L*f in SI, convert to ps
    # beta2=-20ps^2/km=-20e-27s^2/m, L=5.5km=5500m, f=30GHz=30e9Hz
    expected = 2*np.pi * (-20e-27) * 5500 * 30e9 * 1e12
    t_ps = freq_to_time_mapping(-20, 5.5, 30e9)
    assert abs(t_ps - expected) < 0.01


def test_simulate_fig2_shapes():
    sims = simulate_fig2(L1_km=1.1, L2_values_km=[0.0, 2.2], tau_ps=100)
    for key in [0.0, 2.2]:
        t_arr, I_arr, M = sims[key]
        assert len(t_arr) == 4000
        assert I_arr.max() > 0
        assert np.all(I_arr >= 0)


def test_simulate_fig2_stretch_factor():
    sims = simulate_fig2(L1_km=1.1, L2_values_km=[5.5], tau_ps=50)
    _, _, M = sims[5.5]
    assert abs(M - 6.0) < 0.01


def test_mzm_quadrature_bias():
    t_arr = np.linspace(0, 1e-9, 100)
    V_pi = 3.5
    I, a_eff = mzm_transfer_function(V_pi/2, V_pi, 0.1, 2*np.pi*1e9, t_arr)
    # At quadrature: a_eff ~ pi*V_rf/(2*V_pi) -- maximum linear response
    expected_a = np.pi * 0.1 / V_pi
    assert abs(a_eff - expected_a) < 0.01


def test_mzm_pi_bias_zero_response():
    # At V_bias = V_pi (null point): output = 0, small-signal gain = 0
    t_arr = np.linspace(0, 1, 100)
    V_pi = 3.5
    _, a_eff = mzm_transfer_function(V_pi, V_pi, 0.01, 1.0, t_arr)
    # sin(pi*1) = 0 so a_eff should be ~0
    assert abs(a_eff) < 1e-10


def test_mzm_cmos_specs_keys():
    specs = mzm_cmos_specs()
    required = ['V_pi_V', 'bandwidth_3dB_GHz', 'operating_wavelength_nm', 'cmos_node']
    for k in required:
        assert k in specs


def test_detected_intensity_numeric():
    t_arr, I, meta = eq7_detected_intensity(numeric=True, L1_km=1.1, L2_km=5.5)
    assert I.max() > 0
    assert meta['M'] == pytest.approx(6.0, rel=0.01)
    assert meta['fm_eff_GHz'] == pytest.approx(30/6, rel=0.01)
