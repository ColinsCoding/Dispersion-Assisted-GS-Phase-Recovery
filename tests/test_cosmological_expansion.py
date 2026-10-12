"""Tests for dgs/cosmological_expansion.py"""
import numpy as np
import pytest

from dgs.cosmological_expansion import (
    hubble_constant_per_gyr, friedmann_H, scale_factor_matter_dominated,
    scale_factor_radiation_dominated, integrate_scale_factor,
    age_of_universe_gyr, redshift_from_scale_factor, hubble_law_velocity_km_s,
    hubble_radius_mpc, C_SI, H0_KM_S_MPC,
)


class TestFriedmannEquation:
    def test_H_today_equals_H0(self):
        assert abs(friedmann_H(1.0) - H0_KM_S_MPC) < 1e-8

    def test_H_grows_looking_backward_in_time(self):
        # smaller a (earlier universe) -> larger H
        assert friedmann_H(0.1) > friedmann_H(0.5) > friedmann_H(1.0)

    def test_radiation_term_dominates_at_tiny_a(self):
        # Omega_r/a^4 >> Omega_m/a^3 for small enough a
        H_tiny = friedmann_H(1e-6)
        H_matter_only_approx = H0_KM_S_MPC * np.sqrt(0.315 / (1e-6)**3)
        assert H_tiny > H_matter_only_approx   # radiation term adds on top


class TestScaleFactorClosedForms:
    def test_matter_dominated_satisfies_its_own_ode(self):
        # d/dt[ln a] = (2/3)/t exactly for a=(t/t0)^(2/3). np.gradient's
        # one-sided differences at the array boundary are much less accurate
        # than its interior central differences, so (as elsewhere in this
        # repo, e.g. test_ehrenfest_theorem.py) the boundary is trimmed --
        # this is a property of finite differencing, not the closed form.
        t = np.linspace(0.5, 5.0, 100)
        a = scale_factor_matter_dominated(t)
        dlna_dt = np.gradient(np.log(a), t)
        interior = slice(10, -10)
        assert np.max(np.abs(dlna_dt[interior] - (2.0/3.0)/t[interior])) < 1e-3

    def test_radiation_dominated_satisfies_its_own_ode(self):
        t = np.linspace(0.5, 5.0, 100)
        a = scale_factor_radiation_dominated(t)
        dlna_dt = np.gradient(np.log(a), t)
        interior = slice(10, -10)
        assert np.max(np.abs(dlna_dt[interior] - 0.5/t[interior])) < 1e-3


class TestIntegrateScaleFactor:
    def test_matches_matter_only_closed_form(self):
        t = np.linspace(0.01, 10.0, 50)
        from scipy.integrate import solve_ivp
        def rhs(tt, y):
            a = max(y[0], 1e-12)
            return [a * (2.0/3.0) / tt]
        sol = solve_ivp(rhs, (t[0], t[-1]), [scale_factor_matter_dominated(t[0])],
                         t_eval=t, rtol=1e-10, atol=1e-14)
        a_closed = scale_factor_matter_dominated(t)
        assert np.max(np.abs(sol.y[0] - a_closed)) < 1e-6

    def test_scale_factor_is_monotonically_increasing(self):
        t, a = integrate_scale_factor((0.0, 20.0), a0=1e-8, n_eval=1000)
        assert np.all(np.diff(a) > 0)

    def test_reaches_a_equals_one_near_current_age(self):
        t, a = integrate_scale_factor((0.0, 20.0), a0=1e-8, n_eval=200000)
        idx = np.argmin(np.abs(a - 1.0))
        assert abs(t[idx] - 13.787) < 0.2


class TestAgeOfUniverse:
    def test_matches_known_planck_value(self):
        age = age_of_universe_gyr()
        assert abs(age - 13.787) < 0.2

    def test_two_independent_methods_agree(self):
        age_quad = age_of_universe_gyr()
        t, a = integrate_scale_factor((0.0, 20.0), a0=1e-8, n_eval=200000)
        idx = np.argmin(np.abs(a - 1.0))
        age_forward = t[idx]
        assert abs(age_quad - age_forward) < 0.1


class TestRedshift:
    def test_zero_redshift_today(self):
        assert redshift_from_scale_factor(1.0) == 0.0

    def test_half_scale_factor_gives_z_one(self):
        assert abs(redshift_from_scale_factor(0.5) - 1.0) < 1e-12

    def test_redshift_increases_as_scale_factor_shrinks(self):
        z1 = redshift_from_scale_factor(0.5)
        z2 = redshift_from_scale_factor(0.1)
        assert z2 > z1


class TestHubbleLawAndRadius:
    def test_hubble_radius_is_physically_reasonable(self):
        # should be ~4000-5000 Mpc (roughly c/H0), NOT off by ~19 orders of
        # magnitude the way an earlier unit-conversion bug produced
        d = hubble_radius_mpc()
        assert 4000 < d < 5000

    def test_velocity_at_hubble_radius_equals_c(self):
        d = hubble_radius_mpc()
        v_km_s = hubble_law_velocity_km_s(d)
        assert abs(v_km_s - C_SI / 1000.0) / (C_SI / 1000.0) < 1e-8

    def test_velocity_scales_linearly_with_distance(self):
        v1 = hubble_law_velocity_km_s(100.0)
        v2 = hubble_law_velocity_km_s(200.0)
        assert abs(v2 - 2 * v1) < 1e-9

    def test_hubble_radius_roughly_matches_age_times_c(self):
        # not exact (universe isn't matter-only), but same order of magnitude
        d_mpc = hubble_radius_mpc()
        d_gly = d_mpc * 3.2616e-3   # Mpc -> billion light-years
        age = age_of_universe_gyr()
        assert 0.5 < d_gly / age < 1.5
