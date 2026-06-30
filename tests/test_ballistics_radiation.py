import numpy as np
import pytest
import sympy as sp
from dgs.ballistics_radiation import (
    projectile_range, projectile_trajectory, symmetry_of_trajectory,
    impulse_delta, larmor_power, dipole_radiation_resistance,
    impedance_matching_vswr, skin_depth_copper,
    quadrupole_moment_sympy, ballistics_radiation_sympy_5,
)


def test_range_45_degrees_is_max():
    R_45 = projectile_range(100, 45)["range_m"]
    R_30 = projectile_range(100, 30)["range_m"]
    R_60 = projectile_range(100, 60)["range_m"]
    assert R_45 > R_30
    assert R_45 > R_60


def test_range_symmetry_30_60():
    R_30 = projectile_range(100, 30)["range_m"]
    R_60 = projectile_range(100, 60)["range_m"]
    assert R_30 == pytest.approx(R_60, rel=1e-6)


def test_range_scales_as_v0_squared():
    R1 = projectile_range(10, 45)["range_m"]
    R2 = projectile_range(20, 45)["range_m"]
    assert R2 / R1 == pytest.approx(4.0, rel=1e-4)


def test_range_invalid_angle():
    with pytest.raises(ValueError):
        projectile_range(100, 0)
    with pytest.raises(ValueError):
        projectile_range(100, 90)


def test_trajectory_starts_at_origin():
    tr = projectile_trajectory(50, 45)
    assert tr["x"][0] == pytest.approx(0.0)
    assert tr["y"][0] == pytest.approx(0.0)


def test_trajectory_ends_near_ground():
    tr = projectile_trajectory(50, 45)
    assert abs(tr["y"][-1]) < 0.5   # last point near y=0


def test_trajectory_symmetry():
    sym = symmetry_of_trajectory(50, 45)
    assert bool(sym["is_symmetric"]) is True


def test_impulse_velocity_change():
    r = impulse_delta(J_N_s=5.0, m_kg=2.0)
    assert r["delta_v"] == pytest.approx(2.5)


def test_larmor_power_positive():
    r = larmor_power(1e10)
    assert r["power_W"] > 0


def test_larmor_power_scales_as_a_squared():
    p1 = larmor_power(1e10)["power_W"]
    p2 = larmor_power(2e10)["power_W"]
    assert p2 / p1 == pytest.approx(4.0, rel=1e-4)


def test_short_dipole_radiation_resistance():
    # L/lambda = 0.1 -> R_rad = 80*pi^2*(0.1)^2 ~ 7.9 Ohm
    r = dipole_radiation_resistance(0.1, 1.0)
    assert r["R_rad_ohm"] == pytest.approx(80 * np.pi**2 * 0.01, rel=0.01)
    assert r["regime"] == "short dipole"


def test_half_wave_dipole_regime():
    r = dipole_radiation_resistance(0.5, 1.0)
    assert r["regime"] == "half-wave"


def test_vswr_perfect_match():
    r = impedance_matching_vswr(50, 50)
    assert r["VSWR"] == pytest.approx(1.0)
    assert r["power_fraction_delivered"] == pytest.approx(1.0)


def test_vswr_50_to_73():
    r = impedance_matching_vswr(50, 73)
    assert 1.0 < r["VSWR"] < 2.0
    assert r["power_fraction_delivered"] > 0.9


def test_skin_depth_copper_1ghz():
    r = skin_depth_copper(1e9)
    assert 1.0 < r["skin_depth_um"] < 5.0   # expect ~2.1 um


def test_skin_depth_scales_as_inv_sqrt_freq():
    r1 = skin_depth_copper(1e9)
    r4 = skin_depth_copper(4e9)
    assert r1["skin_depth_m"] / r4["skin_depth_m"] == pytest.approx(2.0, rel=0.01)


def test_quadrupole_sympy_returns_equations():
    eqs = quadrupole_moment_sympy()
    assert isinstance(eqs["Q_xx"], sp.Eq)
    assert isinstance(eqs["Dipole_power"], sp.Eq)


def test_ballistics_sympy_5_count():
    eqs = ballistics_radiation_sympy_5()
    assert len(eqs) == 5
    for eq in eqs.values():
        assert isinstance(eq, sp.Eq)
