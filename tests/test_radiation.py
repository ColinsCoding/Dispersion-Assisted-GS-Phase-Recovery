"""Tests for dgs/radiation.py — dipole radiation, unit sphere, equipotentials."""
import numpy as np
import sympy as sp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.radiation import (
    dipole_power_pattern, dipole_gain, dipole_poynting,
    larmor_radiated_power, unit_sphere_integral, sphere_integral_dipole,
    dipole_equipotentials, electric_field_dipole,
    double_well_stability, noether_charge_EM, radiation_sympy_5,
)


def test_dipole_pattern_equatorial():
    # At theta=pi/2: sin^2=1 (maximum)
    assert abs(dipole_power_pattern(np.pi/2) - 1.0) < 1e-12


def test_dipole_pattern_axial():
    # At theta=0 and pi: sin^2=0 (null)
    assert abs(dipole_power_pattern(0.0)) < 1e-12
    assert abs(dipole_power_pattern(np.pi)) < 1e-12


def test_dipole_gain_max():
    # G(pi/2) = 3/2 = 1.5
    assert abs(dipole_gain(np.pi/2) - 1.5) < 1e-12


def test_dipole_gain_dBi():
    # 10*log10(1.5) = 1.76 dBi
    assert abs(10*np.log10(dipole_gain(np.pi/2)) - 1.7609) < 0.001


def test_sphere_integral_dipole():
    res = sphere_integral_dipole()
    # Numerical should match analytical 8*pi/3 within 1%
    assert res["error_pct"] < 1.0
    # D gain should be close to 1.5
    assert abs(res["D_gain"] - 1.5) < 0.05


def test_sphere_integral_isotropic():
    # Isotropic: f=1 everywhere -> integral = 4*pi, D=1
    res = unit_sphere_integral(lambda th, ph: np.ones_like(th))
    assert abs(res["integral"] - 4*np.pi) / (4*np.pi) < 0.01
    assert abs(res["D_gain"] - 1.0) < 0.05


def test_larmor_formula():
    q = 1.602e-19; a = 1e15
    P = larmor_radiated_power(q, a)
    assert P > 0
    # Classical result: ~5.7e-24 W for electron at 1e15 m/s^2
    assert abs(P - 5.71e-24) / 5.71e-24 < 0.01


def test_dipole_poynting_theta_dependence():
    # Poynting: S(theta=pi/2) = max, S(theta=0) = 0
    r, omega, p0 = 1.0, 2*np.pi*1e9, 1e-9
    s90 = dipole_poynting(np.pi/2, r, omega, p0)
    s0  = dipole_poynting(0.0,    r, omega, p0)
    assert s90["S_r"] > 0
    assert abs(s0["S_r"]) < 1e-30


def test_dipole_poynting_1_over_r2():
    # Poynting falls as 1/r^2
    omega, p0 = 2*np.pi*1e9, 1e-9
    s1 = dipole_poynting(np.pi/2, 1.0, omega, p0)
    s2 = dipole_poynting(np.pi/2, 2.0, omega, p0)
    ratio = s1["S_r"] / s2["S_r"]
    assert abs(ratio - 4.0) < 1e-6   # 2^2=4


def test_electric_field_dipole_symmetry():
    # On z-axis (x=0): E_x=0, E_z dominates
    x = np.array([0.0]); z = np.array([1.0])
    E = electric_field_dipole(x, z, p=1.0)
    assert abs(E["Ex"][0]) < 1e-20
    assert E["Ez"][0] > 0


def test_electric_field_equatorial():
    # On equatorial plane (z=0, x>0): E_x (radial) = 0; E_z = -p/(4*pi*eps0*r^3)
    # Formula: Ex = p/(4pi*eps0) * 3*x*z/r^5 = 0 when z=0
    x = np.array([1.0]); z = np.array([0.0])
    E = electric_field_dipole(x, z, p=1.0)
    assert abs(E["Ex"][0]) < 1e-20       # radial (x) component = 0
    assert abs(E["Ez"][0]) > 1.0          # theta component non-zero (field points -z)


def test_dipole_equipotentials_shape():
    surfaces = dipole_equipotentials([1.0, 4.0, 9.0])
    assert len(surfaces["surfaces"]) == 3
    for K, surf in surfaces["surfaces"].items():
        assert len(surf["r"]) > 0
        assert np.all(surf["r"] >= 0)


def test_double_well_minima():
    dw = double_well_stability(a=1.0, b=0.1)
    # Minima should be at x = ±1
    assert abs(dw["minima_x"][0] - 1.0) < 1e-10
    assert abs(dw["minima_x"][1] + 1.0) < 1e-10


def test_double_well_barrier():
    # V_barrier = a^2 for (x^2-a)^2 -> barrier at x=0 is a^2
    dw = double_well_stability(a=2.0, b=0.1)
    assert abs(dw["V_barrier"] - 4.0) < 1e-10


def test_noether_charge_em_keys():
    nc = noether_charge_EM()
    for k in ("P_Larmor", "P_dipole", "Poynting_S", "charge_consv"):
        assert k in nc
        assert isinstance(nc[k], sp.Eq)


def test_noether_sphere_integral():
    nc = noether_charge_EM()
    # sphere_integral = 8*pi/3
    assert nc["sphere_integral"] == sp.Rational(8, 1) * sp.pi / 3


def test_radiation_sympy_5_count():
    eqs = radiation_sympy_5()
    assert len(eqs) == 5


def test_radiation_sympy_5_all_Eq():
    eqs = radiation_sympy_5()
    for k, v in eqs.items():
        assert isinstance(v, sp.Eq), f"{k} not sp.Eq"
