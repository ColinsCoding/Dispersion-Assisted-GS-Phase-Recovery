"""Sidewalk delivery robot vs. human gig driver -- trip model + robot navigation.

Two halves, one question ("when does a 4 mph robot beat a 25 mph car?"):

1. Trip economics (closed form)
   time(d)  = overhead + 60 d / (v * f_rain)                [min]
   G(d)     = c_fixed + c_mile d + (c_hour/60 + VOT) time   [$, generalized cost]
   Both couriers give G linear in d, so the cost crossover is one division:
       d* = (a_robot - a_driver) / (b_driver - b_robot)

2. Navigation (simulation)
   A robot holding the centerline of a sidewalk: y = cross-track error.
       theta_dot = u                (steering-rate command)
       y_dot     = v sin(theta)
   u comes from dgs.pid_controller.PIDController acting on an ESTIMATE of y.
   Three estimators, in increasing order of honesty about real robots:
       'gps'          -- 1 Hz GPS fix, zero-order-held (what a phone gives you)
       'fused'        -- scalar Kalman filter: odometry+IMU predict, GPS update
       'fused+vision' -- same, plus a 10 Hz camera curb measurement
   GPS error = white noise + Gauss-Markov bias. The Kalman filter averages away
   the white part; it CANNOT see the bias (no other sensor observes it). Only an
   independent absolute reference (vision) removes it -- that is the lesson.

Every default number is an assumption, labelled as such. Change them.
"""
from dataclasses import dataclass, replace

import numpy as np

from dgs.pid_controller import PIDController

MPH_TO_MS = 0.44704
MI_TO_M = 1609.344


# ── 1. trip economics ────────────────────────────────────────────────

@dataclass(frozen=True)
class Courier:
    """Parameters of one delivery mode. All defaults below are assumptions.

    speed_mph         -- effective door-to-door speed (not top speed)
    overhead_min      -- fixed time per trip: loading, parking, walking to door
    cost_fixed        -- $ per trip independent of distance (amortised hardware,
                         platform fee, base pay)
    cost_per_mile     -- $ per mile (fuel/energy + wear)
    cost_per_hour     -- $ per hour of trip time (driver pay, remote supervision)
    max_range_mi      -- one-way service radius (battery / dispatch policy)
    rain_speed_factor -- multiply speed by this in rain (0 < f <= 1)
    rain_cost_factor  -- multiply $ costs by this in rain (surge pay etc.)
    rain_availability -- probability the mode is operating at all in rain
    """
    name: str
    speed_mph: float
    overhead_min: float
    cost_fixed: float
    cost_per_mile: float
    cost_per_hour: float
    max_range_mi: float = np.inf
    rain_speed_factor: float = 1.0
    rain_cost_factor: float = 1.0
    rain_availability: float = 1.0

    def __post_init__(self):
        if self.speed_mph <= 0:
            raise ValueError("speed_mph must be positive")
        if self.overhead_min < 0:
            raise ValueError("overhead_min must be non-negative")
        if min(self.cost_fixed, self.cost_per_mile, self.cost_per_hour) < 0:
            raise ValueError("costs must be non-negative")
        if self.max_range_mi <= 0:
            raise ValueError("max_range_mi must be positive")
        if not 0 < self.rain_speed_factor <= 1:
            raise ValueError("rain_speed_factor must be in (0, 1]")
        if self.rain_cost_factor < 1:
            raise ValueError("rain_cost_factor must be >= 1")
        if not 0 <= self.rain_availability <= 1:
            raise ValueError("rain_availability must be in [0, 1]")


# Assumed defaults -- order-of-magnitude, not quotes from any company.
ROBOT = Courier(
    name="sidewalk robot", speed_mph=3.5, overhead_min=2.0,
    cost_fixed=1.50, cost_per_mile=0.05, cost_per_hour=3.00,
    max_range_mi=3.0, rain_speed_factor=0.75, rain_cost_factor=1.0,
    rain_availability=0.6,
)
DRIVER = Courier(
    name="human driver", speed_mph=15.0, overhead_min=6.0,
    cost_fixed=2.50, cost_per_mile=0.67, cost_per_hour=20.0,
    rain_speed_factor=0.85, rain_cost_factor=1.3, rain_availability=1.0,
)


def _check_distance(d_mi):
    d = np.asarray(d_mi, dtype=float)
    if np.any(d < 0):
        raise ValueError("distance must be non-negative")
    return d


def trip_time_min(courier, d_mi, rain=False):
    """Door-to-door trip time in minutes; np.inf beyond the service radius."""
    d = _check_distance(d_mi)
    f = courier.rain_speed_factor if rain else 1.0
    t = courier.overhead_min + 60.0 * d / (courier.speed_mph * f)
    return np.where(d <= courier.max_range_mi, t, np.inf)


def trip_cost(courier, d_mi, rain=False, value_of_time=0.0):
    """Generalized trip cost in $: operator cost + customer wait * value_of_time.

    value_of_time is $ per minute the customer is waiting (0 = operator cost only).
    """
    if value_of_time < 0:
        raise ValueError("value_of_time must be non-negative")
    d = _check_distance(d_mi)
    t = trip_time_min(courier, d, rain)
    k = courier.rain_cost_factor if rain else 1.0
    in_range = np.isfinite(t)
    t_ok = np.where(in_range, t, 0.0)          # avoid 0 * inf = nan
    op = k * (courier.cost_fixed + courier.cost_per_mile * d
              + courier.cost_per_hour * t_ok / 60.0)
    return np.where(in_range, op + value_of_time * t_ok, np.inf)


def cost_line(courier, rain=False, value_of_time=0.0):
    """Return (a, b) with trip_cost = a + b*d inside the service radius."""
    if value_of_time < 0:
        raise ValueError("value_of_time must be non-negative")
    f = courier.rain_speed_factor if rain else 1.0
    k = courier.rain_cost_factor if rain else 1.0
    per_min = k * courier.cost_per_hour / 60.0 + value_of_time
    a = k * courier.cost_fixed + per_min * courier.overhead_min
    b = k * courier.cost_per_mile + per_min * 60.0 / (courier.speed_mph * f)
    return a, b


def crossover_distance(robot=ROBOT, driver=DRIVER, rain=False, value_of_time=0.0):
    """Distance (mi) where the two generalized costs are equal.

    Returns np.nan if the lines never cross at d >= 0 (one mode dominates).
    Ignores the service radius -- compare the result to robot.max_range_mi.
    """
    a_r, b_r = cost_line(robot, rain, value_of_time)
    a_d, b_d = cost_line(driver, rain, value_of_time)
    if np.isclose(b_r, b_d):
        return np.nan
    d = (a_d - a_r) / (b_r - b_d)
    return d if d >= 0 else np.nan


def monte_carlo_trip_times(courier, d_mi, rain=False, n=10_000, speed_cv=0.25,
                           seed=0):
    """Sample trip times with lognormal speed (coefficient of variation speed_cv).

    The lognormal is parameterised so E[speed] equals the courier's mean speed.
    Traffic, lights, crowded sidewalks -- all lumped into one spread.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if speed_cv < 0:
        raise ValueError("speed_cv must be non-negative")
    d = float(_check_distance(d_mi))
    rng = np.random.default_rng(seed)
    f = courier.rain_speed_factor if rain else 1.0
    mean_v = courier.speed_mph * f
    s2 = np.log1p(speed_cv ** 2)
    v = rng.lognormal(np.log(mean_v) - s2 / 2, np.sqrt(s2), size=n)
    t = courier.overhead_min + 60.0 * d / v
    return t if d <= courier.max_range_mi else np.full(n, np.inf)


# ── 2. GPS error model ───────────────────────────────────────────────

GPS_OPEN_SKY = dict(sigma_white=1.5, sigma_bias=1.0, tau_bias=60.0)
GPS_URBAN_CANYON = dict(sigma_white=5.0, sigma_bias=4.0, tau_bias=60.0)


def gps_error(n, dt, sigma_white, sigma_bias, tau_bias, seed=0):
    """GPS position error (m) = white noise + first-order Gauss-Markov bias.

    The bias b obeys b[k+1] = phi b[k] + w, phi = exp(-dt/tau), with w scaled so
    the stationary std of b is sigma_bias (multipath that drifts slowly).
    Returns (total_error, bias).
    """
    if n <= 0 or dt <= 0 or tau_bias <= 0:
        raise ValueError("n, dt and tau_bias must be positive")
    if sigma_white < 0 or sigma_bias < 0:
        raise ValueError("sigmas must be non-negative")
    rng = np.random.default_rng(seed)
    phi = np.exp(-dt / tau_bias)
    q = sigma_bias * np.sqrt(1 - phi ** 2)
    b = np.empty(n)
    b[0] = rng.normal(0, sigma_bias)
    w = rng.normal(0, q, size=n)
    for k in range(1, n):
        b[k] = phi * b[k - 1] + w[k]
    return b + rng.normal(0, sigma_white, size=n), b


# ── 3. path following: PID on a (possibly fused) cross-track estimate ──

MODES = ("gps", "fused", "fused+vision")


def simulate_path_following(mode="fused", v_ms=3.5 * MPH_TO_MS, T=300.0, dt=0.1,
                            gps_rate_hz=1.0, gps=GPS_OPEN_SKY, Kp=0.14, Ki=0.0,
                            Kd=0.45, u_max=1.0, sigma_heading=0.01,
                            sigma_vision=0.10, y0=0.5, seed=0):
    """Simulate a robot holding y = 0 on a sidewalk.

    Default gains place the linearised loop  y'' + v Kd y' + v Kp y = 0
    at omega_n ~ 0.5 rad/s, zeta ~ 0.8 for v = 3.5 mph.
    Returns a dict of time series plus summary metrics.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if v_ms <= 0 or T <= 0 or dt <= 0 or gps_rate_hz <= 0:
        raise ValueError("v_ms, T, dt and gps_rate_hz must be positive")
    if u_max <= 0:
        raise ValueError("u_max must be positive")

    n = int(round(T / dt))
    rng = np.random.default_rng(seed + 1)
    gps_err, gps_bias = gps_error(n, dt, seed=seed, **gps)
    gps_every = max(1, int(round(1.0 / (gps_rate_hz * dt))))
    R_gps = gps["sigma_white"] ** 2 + gps["sigma_bias"] ** 2   # filter's belief
    R_vis = sigma_vision ** 2
    Q = (v_ms * dt * sigma_heading) ** 2 + (0.01 * v_ms * dt) ** 2  # IMU + slip

    pid = PIDController(Kp, Ki, Kd, dt=dt, u_min=-u_max, u_max=u_max)
    y, th = y0, 0.0
    y_est, P = 0.0, R_gps          # start ignorant: estimate 0, variance ~ GPS
    y_hold = 0.0

    out = {k: np.empty(n) for k in ("t", "y", "y_est", "y_gps", "theta", "u")}
    for k in range(n):
        # sensors
        new_fix = k % gps_every == 0
        if new_fix:
            y_hold = y + gps_err[k]
        if mode == "gps":
            y_est = y_hold
        else:
            # predict with odometry + noisy IMU heading
            th_imu = th + rng.normal(0, sigma_heading)
            y_est += v_ms * np.sin(th_imu) * dt
            P += Q
            if new_fix:
                K = P / (P + R_gps)
                y_est += K * (y_hold - y_est)
                P *= 1 - K
            if mode == "fused+vision":
                z = y + rng.normal(0, sigma_vision)
                K = P / (P + R_vis)
                y_est += K * (z - y_est)
                P *= 1 - K

        u = pid.step(0.0 - y_est)

        out["t"][k], out["y"][k], out["y_est"][k] = k * dt, y, y_est
        out["y_gps"][k], out["theta"][k], out["u"][k] = y_hold, th, u

        # plant (Euler; dt is small relative to 1/omega_n)
        th += u * dt
        y += v_ms * np.sin(th) * dt

    out["gps_bias"] = gps_bias
    out["u_max"] = u_max
    out.update(path_metrics(out))
    return out


def path_metrics(sim, half_width=1.0, settle_s=20.0):
    """RMS cross-track error, fraction of time off the sidewalk, steering effort.

    The first settle_s seconds are dropped so the initial offset y0 doesn't
    dominate. half_width is half the usable sidewalk width (m).
    """
    m = sim["t"] >= settle_s
    y, u = sim["y"][m], sim["u"][m]
    return {
        "rms_y": float(np.sqrt(np.mean(y ** 2))),
        "frac_off": float(np.mean(np.abs(y) > half_width)),
        "rms_u": float(np.sqrt(np.mean(u ** 2))),
        "frac_saturated": float(np.mean(np.abs(u) >= 0.999 * sim["u_max"])),
    }


def with_rain(courier):
    """Convenience: a copy of courier whose dry numbers are its rain numbers."""
    return replace(
        courier,
        name=courier.name + " (rain)",
        speed_mph=courier.speed_mph * courier.rain_speed_factor,
        cost_fixed=courier.cost_fixed * courier.rain_cost_factor,
        cost_per_mile=courier.cost_per_mile * courier.rain_cost_factor,
        cost_per_hour=courier.cost_per_hour * courier.rain_cost_factor,
        rain_speed_factor=1.0, rain_cost_factor=1.0,
    )


# ── 4. one-year fleet economics: buy a robot or keep paying drivers? ──

def annual_breakeven_trips(robot_price, d_mi, life_years=1.0, interest=0.0,
                           robot=ROBOT, driver=DRIVER, rain=False,
                           value_of_time=0.0):
    """Trips per year one robot must make for its hardware to pay off.

    The hardware is turned into an equal annual payment with the capital
    recovery factor A/P (dgs.engineering_economics). Each robot trip saves
    (driver cost - robot MARGINAL cost), where marginal = robot with
    cost_fixed = 0 (cost_fixed in the default ROBOT already includes an
    amortised hardware guess -- counting it again would double-charge).

        N* = A / (G_driver(d) - G_robot_marginal(d))

    Returns np.inf if the robot saves nothing per trip at this distance.
    """
    from dgs.engineering_economics import capital_recovery_factor
    if robot_price < 0:
        raise ValueError("robot_price must be non-negative")
    if life_years <= 0:
        raise ValueError("life_years must be positive")
    A = capital_recovery_factor(robot_price, interest, life_years) if robot_price else 0.0
    marginal = replace(robot, cost_fixed=0.0)
    saving = (trip_cost(driver, d_mi, rain, value_of_time)
              - trip_cost(marginal, d_mi, rain, value_of_time))
    saving = float(saving)
    if not np.isfinite(saving) or saving <= 0:
        return np.inf
    return A / saving


def max_trips_per_year(d_mi, robot=ROBOT, hours_per_day=12.0, days=365):
    """Physical ceiling: how many round trips one robot can fit in a year.

    A round trip is out-and-back: 2*d at robot speed, plus the overhead once.
    """
    if hours_per_day <= 0 or hours_per_day > 24 or days <= 0:
        raise ValueError("hours_per_day must be in (0, 24], days positive")
    d = float(_check_distance(d_mi))
    if d > robot.max_range_mi:
        return 0.0
    t_round_min = robot.overhead_min + 2 * 60.0 * d / robot.speed_mph
    return hours_per_day * 60.0 * days / t_round_min


# ── 5. hills and cracks ──────────────────────────────────────────────

G = 9.81


def hill_speed(grade, power_w=80.0, mass_kg=30.0, crr=0.03,
               v_cap=3.5 * MPH_TO_MS):
    """Steady uphill speed (m/s) for a power-limited robot.

    Power balance at constant speed:  P = m g (sin a + Crr cos a) v,
    a = atan(grade). Capped at the robot's legal / design speed v_cap.
    Downhill (grade < 0) returns v_cap: the motors brake, they don't limit.
    """
    if power_w <= 0 or mass_kg <= 0 or crr < 0 or v_cap <= 0:
        raise ValueError("power_w, mass_kg, v_cap positive; crr non-negative")
    g = np.asarray(grade, dtype=float)
    a = np.arctan(g)
    resist = mass_kg * G * (np.sin(a) + crr * np.cos(a))
    with np.errstate(divide="ignore"):
        v = np.where(resist > 0, power_w / resist, np.inf)
    return np.minimum(v, v_cap)


def climb_energy_wh(rise_m, mass_kg=30.0, drivetrain_eff=0.7):
    """Battery energy (Wh) to lift the robot rise_m: m g h / efficiency."""
    if mass_kg <= 0 or not 0 < drivetrain_eff <= 1:
        raise ValueError("mass_kg positive, drivetrain_eff in (0, 1]")
    return mass_kg * G * np.asarray(rise_m, dtype=float) / drivetrain_eff / 3600.0


def max_crack_height(wheel_diameter_m):
    """Tallest step a rigid wheel can roll over: the wheel radius.

    Pushing horizontally at the axle, a step of height h needs a force
    F = W * sqrt(h(2r - h)) / (r - h), which diverges as h -> r. Real
    robots manage roughly a third of the wheel diameter (h ~ 0.67 r).
    """
    if wheel_diameter_m <= 0:
        raise ValueError("wheel_diameter_m must be positive")
    return wheel_diameter_m / 2.0


def step_force_ratio(h_m, wheel_diameter_m):
    """F / W needed to push a wheel up a step of height h (horizontal push)."""
    r = wheel_diameter_m / 2.0
    h = np.asarray(h_m, dtype=float)
    if np.any(h < 0):
        raise ValueError("h_m must be non-negative")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.sqrt(h * (2 * r - h)) / (r - h)
    return np.where(h < r, out, np.inf)


# ── 6. kinetic energy and collision vectors ──────────────────────────

# (mass incl. rider/load in kg, typical speed in m/s). Assumed round numbers.
MOVERS = {
    "walker (shoes)": (70.0, 1.4),
    "sidewalk robot": (30.0, 3.5 * MPH_TO_MS),
    "bike":           (85.0, 6.0),
    "skier":          (80.0, 12.0),
    "car @ 25 mph":   (1500.0, 25 * MPH_TO_MS),
}


def kinetic_energy(m, v):
    """KE = 1/2 m |v|^2 (J). v may be a scalar speed or a vector."""
    if m < 0:
        raise ValueError("mass must be non-negative")
    v = np.atleast_1d(np.asarray(v, dtype=float))
    return 0.5 * m * float(v @ v)


def collision_energy(m1, v1, m2, v2):
    """Energy (J) turned into damage in a perfectly inelastic 2-D collision.

    In the centre-of-mass frame only RELATIVE motion can be destroyed:
        dKE = 1/2 mu |v1 - v2|^2,   mu = m1 m2 / (m1 + m2)  (reduced mass)
    v1, v2 are velocity VECTORS -- the crossing angle matters, not just speed.
    """
    if m1 <= 0 or m2 <= 0:
        raise ValueError("masses must be positive")
    dv = np.asarray(v1, dtype=float) - np.asarray(v2, dtype=float)
    mu = m1 * m2 / (m1 + m2)
    return 0.5 * mu * float(dv @ dv)


def velocity_vector(speed, heading_deg):
    """2-D velocity vector from speed and heading (degrees from +x)."""
    th = np.radians(heading_deg)
    return speed * np.array([np.cos(th), np.sin(th)])
