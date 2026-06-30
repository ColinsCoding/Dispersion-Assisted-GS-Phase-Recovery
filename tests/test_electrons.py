"""Tests for dgs/electrons.py"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.electrons import (
    cyclotron_frequency, cyclotron_radius, cyclotron_period,
    exb_drift, lorentz_factor, relativistic_kinetic_energy,
    debroglie_wavelength, compton_scatter, larmor_power, hall_voltage,
    Q_E, M_E, C_LIGHT, H_PLANCK,
)

def test_cyclotron_frequency():
    # omega_c = qB/m; for B=1T electron: 1.76e11 rad/s
    wc = cyclotron_frequency(B=1.0)
    assert abs(wc - Q_E / M_E) < 1e6, f"wc={wc}"

def test_cyclotron_period_orbit():
    # Period T = 2*pi/omega_c; radius r = mv/(qB)
    B, v = 0.1, 1e6
    T  = cyclotron_period(B)
    wc = cyclotron_frequency(B)
    assert abs(T - 2*np.pi/wc) < 1e-12
    r  = cyclotron_radius(v, B)
    assert abs(r - M_E*v/(Q_E*B)) < 1e-15

def test_exb_drift_magnitude():
    E = np.array([1e3, 0, 0])
    B = np.array([0, 0, 0.1])
    v = exb_drift(E, B)
    # |v| = E/B = 1e4 m/s
    assert abs(np.linalg.norm(v) - 1e4) < 1, f"|v_drift|={np.linalg.norm(v)}"
    # Perpendicular to both E and B
    assert abs(np.dot(v, E)) < 1e-6
    assert abs(np.dot(v, B)) < 1e-6

def test_lorentz_factor():
    # gamma(0) = 1; gamma(0.6c) = 1.25
    assert abs(lorentz_factor(0) - 1.0) < 1e-10
    assert abs(lorentz_factor(0.6*C_LIGHT) - 1.25) < 1e-4

def test_nonrelativistic_ke_limit():
    # (gamma-1)*mc^2 -> (1/2)mv^2 for v << c
    v = 1e5   # v/c = 3e-4
    KE_rel = relativistic_kinetic_energy(v)
    KE_nr  = 0.5 * M_E * v**2
    assert abs(KE_rel - KE_nr) / KE_nr < 1e-7

def test_debroglie_100eV():
    # 100 eV electron: lambda = h/sqrt(2*m*KE) = 1.226 Angstrom
    lam = debroglie_wavelength(KE_eV=100)
    lam_theory = H_PLANCK / np.sqrt(2 * M_E * 100 * Q_E)
    assert abs(lam - lam_theory) / lam_theory < 1e-8

def test_compton_no_shift_theta0():
    # theta=0: forward scatter, no wavelength change
    _, dl, KE = compton_scatter(0.1e-9, theta=0.0)
    assert abs(dl) < 1e-20
    assert abs(KE) < 1e-10

def test_compton_theta180():
    # theta=pi: backscatter, max shift = 2 * lambda_C
    lambda_C = H_PLANCK / (M_E * C_LIGHT)
    _, dl, _ = compton_scatter(0.1e-9, theta=np.pi)
    assert abs(dl - 2*lambda_C) / lambda_C < 1e-8

def test_larmor_power_positive():
    # Any acceleration -> positive power
    P = larmor_power(a=1e15)
    assert P > 0

def test_hall_voltage():
    # V_H = IB/(nqt)
    I, B, n, t = 1e-3, 1.0, 8.5e28, 1e-3
    V_H = hall_voltage(I, B, n, t)
    expected = I*B/(n*Q_E*t)
    assert abs(V_H - expected) / expected < 1e-10
