"""Tests for dgs/lnbo3.py -- LiNbO3 material handler"""
import numpy as np
import pytest
from dgs.lnbo3 import (
    sellmeier, gvd_lnbo3, eo_phase_shift, vpi, mzm_lnbo3,
    hermitian_observable_demo, sympy_hermitian_symbolic,
    PYTHON_VS_JAVA, EO_TENSOR, SELLMEIER,
)


def test_sellmeier_1550nm_extraordinary():
    n = sellmeier(1.55, 'extraordinary')
    # LiNbO3 extraordinary at 1550nm: ~2.14-2.15
    assert 2.10 < n < 2.20


def test_sellmeier_1550nm_ordinary():
    n = sellmeier(1.55, 'ordinary')
    # LiNbO3 ordinary at 1550nm: ~2.21-2.23
    assert 2.18 < n < 2.25


def test_birefringence():
    # Extraordinary < ordinary (negative uniaxial crystal)
    n_e = sellmeier(1.55, 'extraordinary')
    n_o = sellmeier(1.55, 'ordinary')
    assert n_e < n_o, "LiNbO3 is negative uniaxial: n_e < n_o"


def test_sellmeier_visible():
    # At 0.633 um (HeNe), n should be higher than at 1550nm (normal dispersion)
    n_vis = sellmeier(0.633, 'extraordinary')
    n_ir = sellmeier(1.55, 'extraordinary')
    assert n_vis > n_ir, "Normal dispersion: n increases at shorter wavelength"


def test_gvd_positive_normal_dispersion():
    b2 = gvd_lnbo3(1.55, 'extraordinary')
    # Bulk LiNbO3 at 1550nm: normal dispersion -> beta2 > 0
    assert b2 > 0


def test_vpi_scales_with_length():
    Vpi_50 = vpi(L_mm=50.0)
    Vpi_100 = vpi(L_mm=100.0)
    # V_pi * L = const -> longer device has lower V_pi
    assert abs(Vpi_50 * 50 - Vpi_100 * 100) < 0.01  # V_pi*L = const


def test_vpi_reasonable_range():
    Vpi = vpi(L_mm=50.0)
    # Typical LiNbO3 waveguide MZM: V_pi in range 0.5-5 V for 50mm
    assert 0.1 < Vpi < 10.0


def test_eo_phase_shift_at_vpi():
    Vpi_val = vpi(L_mm=50.0)
    phi = eo_phase_shift(Vpi_val, L_mm=50.0)
    # At V=V_pi, phase shift should be pi
    assert abs(phi - np.pi) < 0.01


def test_eo_phase_shift_at_half_vpi():
    Vpi_val = vpi(L_mm=50.0)
    phi = eo_phase_shift(Vpi_val/2, L_mm=50.0)
    assert abs(phi - np.pi/2) < 0.01


def test_eo_linearity():
    # Phase shift is linear in V (Pockels effect, not Kerr)
    phi1 = eo_phase_shift(1.0, L_mm=50.0)
    phi2 = eo_phase_shift(2.0, L_mm=50.0)
    assert abs(phi2 - 2*phi1) < 1e-10


def test_mzm_output_shape():
    t_ps = np.linspace(-100, 100, 256)
    E_out, I_out, a_eff, Vpi_val = mzm_lnbo3(V_bias=0.5, V_rf=0.1,
                                               fm_ghz=30, t_ps=t_ps)
    assert E_out.shape == (256,)
    assert I_out.shape == (256,)
    assert 0 <= I_out.max() <= 1.01


def test_mzm_quadrature_max_modulation():
    t_ps = np.linspace(-10, 10, 100)
    Vpi_val = vpi(L_mm=50.0)
    # Quadrature bias = V_pi/2 -> maximum a_eff
    _, _, a_q, _ = mzm_lnbo3(V_bias=Vpi_val/2, V_rf=0.01,
                               fm_ghz=1, t_ps=t_ps)
    # Null bias = V_pi -> zero a_eff
    _, _, a_null, _ = mzm_lnbo3(V_bias=Vpi_val, V_rf=0.01,
                                  fm_ghz=1, t_ps=t_ps)
    assert a_q > abs(a_null)


def test_hermitian_real_eigenvalues():
    result = hermitian_observable_demo()
    eigs = result['eigenvalues_real']
    assert np.all(np.isreal(eigs)), "Hermitian matrix must have real eigenvalues"


def test_hermitian_is_hermitian():
    result = hermitian_observable_demo()
    H = result['H_matrix']
    assert result['is_hermitian']
    assert np.allclose(H, H.conj().T)


def test_energy_ratio_negative():
    # One eigenvalue is negative (energy levels can straddle zero for this matrix)
    result = hermitian_observable_demo()
    eigs = result['eigenvalues_real']
    assert eigs[0] < 0 < eigs[1]


def test_python_vs_java_has_required_keys():
    for key in ['complex_number', 'matrix_multiply', 'hermitian_check',
                'eigenvalues', 'symbolic_math', 'fft', 'torch_autograd']:
        assert key in PYTHON_VS_JAVA
        assert 'python' in PYTHON_VS_JAVA[key]
        assert 'java' in PYTHON_VS_JAVA[key]


def test_r33_largest_eo_coefficient():
    # r33 is the largest EO coefficient in LiNbO3
    assert EO_TENSOR['r33'] > EO_TENSOR['r13']
    assert EO_TENSOR['r33'] > EO_TENSOR['r22']


def test_sympy_hermitian_symbolic():
    result = sympy_hermitian_symbolic()
    import sympy as sp
    H = result['H_symbolic']
    # Should be 2x2
    assert H.shape == (2, 2)
    # Should have two eigenvalues
    assert len(result['eigenvalues']) == 2
