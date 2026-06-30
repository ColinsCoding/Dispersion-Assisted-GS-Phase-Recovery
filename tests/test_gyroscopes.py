import numpy as np
import pytest
from dgs.gyroscopes import (
    center_of_mass, moment_of_inertia_point_masses, moment_of_inertia_disk,
    moment_of_inertia_hoop, moment_of_inertia_rod_center, parallel_axis_theorem,
    small_angle_error, pendulum_period_small_angle, pendulum_rk4,
    precession_rate, nutation_frequency, gyroscope_ee_analogy,
    gyroscope_sympy_5,
)


def test_center_of_mass_symmetric():
    com = center_of_mass([1.0, 1.0], [[-1.0, 0.0], [1.0, 0.0]])
    assert np.allclose(com["R_cm"], [0.0, 0.0])
    assert com["total_mass"] == 2.0


def test_center_of_mass_weighted():
    com = center_of_mass([1.0, 3.0], [0.0, 4.0])
    assert com["R_cm"][0] == pytest.approx(3.0)


def test_center_of_mass_zero_mass_raises():
    with pytest.raises(ValueError):
        center_of_mass([0.0, 0.0], [0.0, 1.0])


def test_moment_of_inertia_point_masses():
    I = moment_of_inertia_point_masses([2.0, 3.0], [1.0, 2.0])
    assert I == pytest.approx(2.0 * 1 + 3.0 * 4)


def test_moment_of_inertia_disk():
    assert moment_of_inertia_disk(2.0, 0.5) == pytest.approx(0.25)


def test_moment_of_inertia_hoop():
    assert moment_of_inertia_hoop(2.0, 0.5) == pytest.approx(0.5)


def test_moment_of_inertia_rod_center():
    assert moment_of_inertia_rod_center(6.0, 2.0) == pytest.approx(2.0)


def test_hoop_greater_than_disk_same_mr():
    # hoop concentrates mass at the rim -> larger I than a uniform disk
    assert moment_of_inertia_hoop(1.0, 1.0) > moment_of_inertia_disk(1.0, 1.0)


def test_parallel_axis_theorem():
    I_cm = moment_of_inertia_disk(1.0, 1.0)
    I_shifted = parallel_axis_theorem(I_cm, 1.0, 2.0)
    assert I_shifted == pytest.approx(I_cm + 4.0)


def test_small_angle_error_grows_with_theta():
    res = small_angle_error(np.array([0.01, 0.5, 1.0]))
    err = res["relative_error"]
    assert err[0] < err[1] < err[2]


def test_small_angle_error_zero_at_zero():
    res = small_angle_error(np.array([0.0]))
    assert res["relative_error"][0] == 0.0


def test_pendulum_period_small_angle():
    T = pendulum_period_small_angle(1.0, g=9.80665)
    assert T == pytest.approx(2 * np.pi * np.sqrt(1.0 / 9.80665))


def test_pendulum_rk4_small_amplitude_matches_shm_period():
    # small theta0 -> nonlinear pendulum period should match SHM period closely
    L = 1.0
    res = pendulum_rk4(theta0_rad=0.05, omega0=0.0, L=L, t_max=4.0, dt=0.0005, small_angle=False)
    theta = res["theta"]
    # find first zero crossing after start (quarter period) via sign change
    crossing_idx = np.where(np.diff(np.sign(theta)))[0][0]
    t = res["t"]
    quarter_period_numeric = t[crossing_idx]
    T_theory = pendulum_period_small_angle(L)
    assert quarter_period_numeric == pytest.approx(T_theory / 4, rel=0.05)


def test_pendulum_rk4_large_amplitude_period_longer_than_small_angle():
    L = 1.0
    res_large = pendulum_rk4(theta0_rad=2.5, omega0=0.0, L=L, t_max=6.0, dt=0.0005, small_angle=False)
    theta = res_large["theta"]
    crossing_idx = np.where(np.diff(np.sign(theta)))[0][0]
    t = res_large["t"]
    quarter_period_large = t[crossing_idx]
    T_small = pendulum_period_small_angle(L)
    assert quarter_period_large > T_small / 4


def test_precession_rate_basic():
    res = precession_rate(mass=1.0, g=9.80665, r=0.1, I_spin=0.01, omega_spin=50.0)
    expected = 1.0 * 9.80665 * 0.1 / (0.01 * 50.0)
    assert res["Omega_p_rad_s"] == pytest.approx(expected)


def test_precession_rate_zero_spin_raises():
    with pytest.raises(ValueError):
        precession_rate(mass=1.0, g=9.8, r=0.1, I_spin=0.01, omega_spin=0)


def test_nutation_frequency():
    omega_n = nutation_frequency(I_spin=0.01, I_transverse=0.02, omega_spin=100.0)
    assert omega_n == pytest.approx(0.01 * 100.0 / 0.02)


def test_gyroscope_ee_analogy_structure():
    a = gyroscope_ee_analogy()
    assert "Steady precession" in a
    assert "Nutation (wobble)" in a
    assert "common_math" in a
    for key in ["Steady precession", "Nutation (wobble)"]:
        assert "mechanical" in a[key] and "electrical" in a[key]


def test_gyroscope_sympy_5_count_and_type():
    import sympy as sp
    eqs = gyroscope_sympy_5()
    assert len(eqs) == 5
    for eq in eqs.values():
        assert isinstance(eq, sp.Eq)
