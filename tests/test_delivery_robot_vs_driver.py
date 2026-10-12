"""Tests for dgs/delivery_robot_vs_driver.py."""
import numpy as np
import pytest

from dgs.delivery_robot_vs_driver import (
    Courier, ROBOT, DRIVER, MODES, GPS_OPEN_SKY, GPS_URBAN_CANYON,
    trip_time_min, trip_cost, cost_line, crossover_distance,
    monte_carlo_trip_times, gps_error, simulate_path_following, with_rain,
)


def test_trip_time_closed_form():
    c = Courier("x", speed_mph=30.0, overhead_min=5.0, cost_fixed=0,
                cost_per_mile=0, cost_per_hour=0)
    assert trip_time_min(c, 15.0) == pytest.approx(5.0 + 30.0)


def test_out_of_range_is_inf():
    assert np.isinf(trip_time_min(ROBOT, ROBOT.max_range_mi + 0.1))
    assert np.isinf(trip_cost(ROBOT, ROBOT.max_range_mi + 0.1))


def test_cost_line_matches_trip_cost():
    for rain in (False, True):
        for vot in (0.0, 0.3):
            for c in (ROBOT, DRIVER):
                a, b = cost_line(c, rain, vot)
                d = np.array([0.0, 0.7, 2.2])
                assert np.allclose(trip_cost(c, d, rain, vot), a + b * d)


def test_crossover_costs_equal():
    d = crossover_distance(value_of_time=0.25)
    assert 0 < d < ROBOT.max_range_mi
    assert trip_cost(ROBOT, d, value_of_time=0.25) == pytest.approx(
        trip_cost(DRIVER, d, value_of_time=0.25))
    # robot cheaper below, driver cheaper above
    assert trip_cost(ROBOT, d / 2, value_of_time=0.25) < trip_cost(
        DRIVER, d / 2, value_of_time=0.25)


def test_no_crossover_when_robot_dominates():
    assert np.isnan(crossover_distance(value_of_time=0.0))


def test_rain_slows_both():
    for c in (ROBOT, DRIVER):
        assert trip_time_min(c, 1.0, rain=True) > trip_time_min(c, 1.0)


def test_with_rain_equivalent():
    for c in (ROBOT, DRIVER):
        assert trip_cost(with_rain(c), 1.3) == pytest.approx(
            trip_cost(c, 1.3, rain=True))


def test_monte_carlo_mean_speed():
    t = monte_carlo_trip_times(DRIVER, 3.0, n=200_000, speed_cv=0.3, seed=1)
    v = 3.0 * 60.0 / (t - DRIVER.overhead_min)
    assert v.mean() == pytest.approx(DRIVER.speed_mph, rel=0.01)


def test_gps_bias_stationary_std():
    _, b = gps_error(200_000, 0.1, sigma_white=0.0, sigma_bias=3.0,
                     tau_bias=5.0, seed=2)
    assert b.std() == pytest.approx(3.0, rel=0.05)


def test_fusion_beats_raw_gps_and_vision_beats_bias():
    # average over seeds: a single bias realisation can be unlucky for 'fused'
    seeds = range(10)
    res = {m: [simulate_path_following(m, gps=GPS_URBAN_CANYON, seed=s)
               for s in seeds] for m in MODES}
    mean = lambda m, k: np.mean([r[k] for r in res[m]])
    assert mean("fused", "rms_y") < mean("gps", "rms_y") / 2
    # fusion cannot remove the bias: its error stays at the bias scale
    assert mean("fused", "rms_y") > 0.25 * GPS_URBAN_CANYON["sigma_bias"]
    assert mean("fused+vision", "rms_y") < 0.05
    assert mean("fused+vision", "frac_off") == 0.0
    # derivative kick on stepping GPS fixes saturates the steering
    assert mean("gps", "frac_saturated") > mean("fused", "frac_saturated")


def test_noise_free_loop_converges():
    gps = dict(sigma_white=0.0, sigma_bias=0.0, tau_bias=60.0)
    s = simulate_path_following("fused+vision", gps=gps, sigma_heading=0.0,
                                sigma_vision=1e-3, y0=0.8, T=60.0)
    assert abs(s["y"][-1]) < 0.02


@pytest.mark.parametrize("kw", [
    dict(speed_mph=0), dict(overhead_min=-1), dict(cost_fixed=-1),
    dict(rain_speed_factor=0), dict(rain_cost_factor=0.5),
    dict(rain_availability=1.5),
])
def test_courier_validation(kw):
    base = dict(name="x", speed_mph=5, overhead_min=1, cost_fixed=1,
                cost_per_mile=1, cost_per_hour=1)
    base.update(kw)
    with pytest.raises(ValueError):
        Courier(**base)


def test_input_validation():
    with pytest.raises(ValueError):
        trip_time_min(ROBOT, -1)
    with pytest.raises(ValueError):
        trip_cost(ROBOT, 1, value_of_time=-1)
    with pytest.raises(ValueError):
        simulate_path_following("lidar")
    with pytest.raises(ValueError):
        gps_error(0, 0.1, 1, 1, 1)


# ── 1-year economics, hills, collisions ──────────────────────────────
from dgs.delivery_robot_vs_driver import (
    annual_breakeven_trips, max_trips_per_year, hill_speed, climb_energy_wh,
    step_force_ratio, max_crack_height, kinetic_energy, collision_energy,
    velocity_vector, MOVERS,
)
from dgs.engineering_economics import capital_recovery_factor


def test_breakeven_free_robot_is_zero_and_linear_in_price():
    assert annual_breakeven_trips(0.0, 1.0) == 0.0
    n1 = annual_breakeven_trips(5_000.0, 1.0)
    n2 = annual_breakeven_trips(10_000.0, 1.0)
    assert n2 == pytest.approx(2 * n1)


def test_breakeven_matches_hand_formula():
    from dataclasses import replace
    P, d, i, n = 10_000.0, 1.2, 0.08, 3
    A = capital_recovery_factor(P, i, n)
    save = trip_cost(DRIVER, d) - trip_cost(replace(ROBOT, cost_fixed=0.0), d)
    assert annual_breakeven_trips(P, d, life_years=n, interest=i) == pytest.approx(A / save)


def test_breakeven_inf_out_of_range():
    assert np.isinf(annual_breakeven_trips(10_000.0, ROBOT.max_range_mi + 1))


def test_max_trips_per_year():
    d = 1.0
    t_round = ROBOT.overhead_min + 120.0 * d / ROBOT.speed_mph
    assert max_trips_per_year(d, hours_per_day=10, days=100) == pytest.approx(
        10 * 60 * 100 / t_round)


def test_hill_power_balance():
    m, P, crr, g = 30.0, 80.0, 0.03, 0.3
    v = float(hill_speed(g, power_w=P, mass_kg=m, crr=crr, v_cap=99.0))
    a = np.arctan(g)
    assert P == pytest.approx(m * 9.81 * (np.sin(a) + crr * np.cos(a)) * v)
    assert float(hill_speed(-0.2)) == pytest.approx(3.5 * 0.44704)  # downhill = cap
    assert np.all(np.diff(hill_speed(np.linspace(0, 0.4, 20))) <= 1e-12)


def test_climb_energy():
    assert float(climb_energy_wh(10.0, mass_kg=36.0, drivetrain_eff=1.0)) == pytest.approx(
        36.0 * 9.81 * 10.0 / 3600.0)


def test_step_force_geometry():
    D = 0.2
    assert float(step_force_ratio(0.0, D)) == 0.0
    # h = r(1 - cos 45deg) puts the contact at 45 deg -> F = W
    r = D / 2
    assert float(step_force_ratio(r * (1 - np.cos(np.pi / 4)), D)) == pytest.approx(1.0)
    assert np.isinf(step_force_ratio(max_crack_height(D), D))


def test_kinetic_energy_scalar_and_vector():
    assert kinetic_energy(2.0, 3.0) == pytest.approx(9.0)
    assert kinetic_energy(2.0, [3.0, 4.0]) == pytest.approx(25.0)


def test_collision_energy_limits():
    m, v = 70.0, 5.0
    # equal masses, head-on at equal speed: all KE is destroyed
    e = collision_energy(m, [v, 0], m, [-v, 0])
    assert e == pytest.approx(2 * kinetic_energy(m, v))
    # same velocity vectors: nothing to destroy
    assert collision_energy(m, [v, 0], m, [v, 0]) == 0.0
    # huge m1 hitting stationary m2: damage -> 1/2 m2 v^2, not 1/2 m1 v^2
    assert collision_energy(1e9, [v, 0], m, [0, 0]) == pytest.approx(
        kinetic_energy(m, v), rel=1e-6)


def test_collision_depends_on_angle():
    m, v = 80.0, 6.0
    same = collision_energy(m, velocity_vector(v, 0), m, velocity_vector(v, 0))
    cross = collision_energy(m, velocity_vector(v, 0), m, velocity_vector(v, 90))
    head = collision_energy(m, velocity_vector(v, 0), m, velocity_vector(v, 180))
    assert same == pytest.approx(0.0, abs=1e-9) and same < cross < head
    assert head == pytest.approx(2 * cross)


def test_robot_is_lowest_energy_mover():
    ke = {k: kinetic_energy(m, v) for k, (m, v) in MOVERS.items()}
    assert min(ke, key=ke.get) == "sidewalk robot"
