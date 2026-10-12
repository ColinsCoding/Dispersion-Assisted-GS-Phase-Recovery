"""Navigation without GPS: inertial drift, a Kalman filter that fits a cheap
microcontroller, and the quantum (atom-interferometer) accelerometer that
would slow the drift -- read out by phase retrieval.

  * DEAD RECKONING. Integrate an accelerometer twice and any bias b becomes a
    position error 1/2 b t^2. A gyro bias tilts the platform, leaking gravity
    into the horizontal channel: error g b_g t^3 / 6. Derived in SymPy.
  * KALMAN FILTER. A 2-state (position, velocity) filter fuses the
    accelerometer with occasional position fixes (landmark, map match). SymPy
    counts the * / + - operations per update -- the number that decides
    whether it runs on a small microcontroller.
  * QUANTUM ACCELEROMETER. A light-pulse atom interferometer accumulates phase
    phi = k_eff a T^2. The detector sees only a fringe P = (1 - C cos phi)/2,
    an intensity -- so recovering a is phase retrieval, with the same 2 pi and
    sign ambiguities as dispersion-assisted GS. Two interrogation times
    (T1, T2) play the role of two dispersions (D1, D2).

Sensor numbers are ORDER-OF-MAGNITUDE grade classes (consumer / tactical /
navigation), typical of textbook tables, not specs of any product.
Civilian education; the same physics serves drones, submarines, warehouse
robots, and phones in tunnels.
"""
import numpy as np
import sympy as sp

G0 = 9.80665
MG = 1e-3 * G0                     # one milli-g in m/s^2
DEG_PER_H = np.pi / 180 / 3600     # rad/s
RB87_WAVELENGTH = 780.241e-9       # D2 line [m]

SENSOR_GRADES = {                   # (accelerometer bias [m/s^2], gyro bias [rad/s])
    "consumer MEMS": (1.0 * MG, 10.0 * DEG_PER_H),
    "tactical": (0.1 * MG, 1.0 * DEG_PER_H),
    "navigation": (0.025 * MG, 0.01 * DEG_PER_H),
}


# ── 1. dead-reckoning drift ──────────────────────────────────────────

def drift_symbolic():
    """Integrate x'' = b_a + g b_g t (tilt grows linearly) from rest.
    Returns position error x(t)."""
    t, ba, bg, g = sp.symbols("t b_a b_g g", positive=True)
    x = sp.Function("x")
    sol = sp.dsolve(sp.Eq(x(t).diff(t, 2), ba + g * bg * t),
                    ics={x(0): 0, x(t).diff(t).subs(t, 0): 0})
    return sp.expand(sol.rhs)


def drift_error(t_s, accel_bias, gyro_bias=0.0, g=G0):
    """|position error| [m] = 1/2 b_a t^2 + g b_g t^3 / 6."""
    t = np.asarray(t_s, float)
    if np.any(t < 0):
        raise ValueError("time must be non-negative")
    return 0.5 * accel_bias * t**2 + g * gyro_bias * t**3 / 6


def time_to_error(limit_m, accel_bias, gyro_bias=0.0, g=G0):
    """Seconds until the drift reaches limit_m (bisection on the monotone cubic)."""
    if limit_m <= 0:
        raise ValueError("limit_m must be positive")
    lo, hi = 0.0, 1.0
    while drift_error(hi, accel_bias, gyro_bias, g) < limit_m:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if drift_error(mid, accel_bias, gyro_bias, g) < limit_m else (lo, mid)
    return hi


# ── 2. Kalman filter and its operation count ─────────────────────────

def _kf_matrices(n_states, dt, sym=True):
    """F, B for the (x, v) or (x, v, accel-bias) model. The bias state is
    subtracted from the measured acceleration: x'' = a_meas - b."""
    half = dt**2 / 2
    if n_states == 2:
        F, B = [[1, dt], [0, 1]], [half, dt]
    elif n_states == 3:
        F, B = [[1, dt, -half], [0, 1, -dt], [0, 0, 1]], [half, dt, 0]
    else:
        raise ValueError("n_states must be 2 or 3")
    if sym:
        return sp.Matrix(F), sp.Matrix(B)
    return np.array(F, float), np.array(B, float)


def kalman_symbolic(n_states=2):
    """Symbolic KF: predict with accel input u, update with a position fix z.
    Returns dict of SymPy expressions for the new state and the upper triangle
    of the covariance."""
    dt, u, z, q, qb, rr = sp.symbols("dt u z q q_b r", positive=True)
    F, B = _kf_matrices(n_states, dt)
    X = sp.Matrix(sp.symbols(["x", "v", "b"][:n_states], real=True))
    P = sp.Matrix(n_states, n_states, lambda i, j: sp.Symbol(f"p{min(i, j) + 1}{max(i, j) + 1}"))
    Q = q * B * B.T
    if n_states == 3:
        Q[2, 2] += qb * dt
    xp = F * X + B * u
    Pp = F * P * F.T + Q
    H = sp.zeros(1, n_states)
    H[0, 0] = 1
    S = (H * Pp * H.T)[0] + rr
    K = Pp * H.T / S
    xn = xp + K * (z - (H * xp)[0])
    Pn = (sp.eye(n_states) - K * H) * Pp
    tri = lambda M: [M[i, j] for i in range(n_states) for j in range(i, n_states)]
    return {"predict_x": list(xp), "predict_P": tri(Pp), "update_x": list(xn), "update_P": tri(Pn)}


def count_ops(exprs, cse=True):
    """Arithmetic operation count (+ - * / and powers) for a list of
    expressions. With cse=True, shared subexpressions are computed once, which
    is what an optimizing compiler (or a careful human) does."""
    exprs = list(exprs)
    if cse:
        reps, reduced = sp.cse(exprs)
        exprs = [e for _, e in reps] + reduced
    return int(sum(sp.count_ops(e) for e in exprs))


def kalman_op_budget(n_states=2):
    """Ops per predict-only step and per predict+update step, naive vs CSE."""
    e = kalman_symbolic(n_states)
    pred = e["predict_x"] + e["predict_P"]
    full = e["update_x"] + e["update_P"]
    return {"predict_naive": count_ops(pred, False), "predict_cse": count_ops(pred, True),
            "update_naive": count_ops(full, False), "update_cse": count_ops(full, True)}


def mcu_load(ops_per_update, rate_hz, mflops=10.0):
    """Fraction of a microcontroller's arithmetic throughput used.
    mflops is an ASSUMED sustained rate (a small FPU core does a few to tens)."""
    if rate_hz <= 0 or mflops <= 0:
        raise ValueError("rate_hz and mflops must be positive")
    return ops_per_update * rate_hz / (mflops * 1e6)


class KalmanNav:
    """Numeric twin of kalman_symbolic. n_states=2: (x, v). n_states=3: also
    estimates the accelerometer bias, which is what stops the t^2 drift."""

    def __init__(self, dt, q, r, n_states=2, q_bias=1e-10, bias_sigma0=5 * MG):
        self.F, self.B = _kf_matrices(n_states, dt, sym=False)
        self.Q = q * np.outer(self.B, self.B)
        self.P = np.zeros((n_states, n_states))
        if n_states == 3:
            self.Q[2, 2] += q_bias * dt
            self.P[2, 2] = bias_sigma0**2
        self.r = r
        self.x = np.zeros(n_states)

    def predict(self, u):
        self.x = self.F @ self.x + self.B * u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        S = self.P[0, 0] + self.r
        K = self.P[:, 0] / S
        self.x = self.x + K * (z - self.x[0])
        self.P = self.P - np.outer(K, self.P[0, :])


def simulate_navigation(t_end=600.0, dt=0.01, accel_bias=1.0 * MG, accel_noise=0.02,
                        fix_every_s=30.0, fix_sigma=2.0, rng=0):
    """1-D vehicle doing a gentle sine maneuver, position fixes every
    fix_every_s. Returns dict: time, truth, pure dead reckoning, a 2-state KF
    (q tuned so it trusts fixes), a 3-state bias-estimating KF, and the
    3-state filter's final bias estimate."""
    rng = np.random.default_rng(rng)
    n = int(round(t_end / dt))
    t = np.arange(n) * dt
    a_true = 0.2 * np.sin(2 * np.pi * t / 120)
    x_true = np.cumsum(np.cumsum(a_true) * dt) * dt
    a_meas = a_true + accel_bias + accel_noise * rng.normal(size=n)
    x_dr = np.cumsum(np.cumsum(a_meas) * dt) * dt
    kf2 = KalmanNav(dt, q=0.1, r=fix_sigma**2, n_states=2)
    kf3 = KalmanNav(dt, q=accel_noise**2, r=fix_sigma**2, n_states=3)
    x2, x3 = np.empty(n), np.empty(n)
    every = int(round(fix_every_s / dt))
    for i in range(n):
        kf2.predict(a_meas[i])
        kf3.predict(a_meas[i])
        if i > 0 and i % every == 0:
            z = x_true[i] + fix_sigma * rng.normal()
            kf2.update(z)
            kf3.update(z)
        x2[i], x3[i] = kf2.x[0], kf3.x[0]
    return {"t": t, "truth": x_true, "dead_reckoning": x_dr, "kf2": x2, "kf3": x3,
            "bias_estimate": float(kf3.x[2]), "bias_true": accel_bias}




# ── 3. quantum accelerometer: atom-interferometer phase retrieval ────

def k_eff(wavelength=RB87_WAVELENGTH):
    """Two-photon Raman wavevector, counter-propagating beams: 2 * 2 pi / lambda."""
    return 4 * np.pi / wavelength


def ai_phase_symbolic():
    """Phase of a Mach-Zehnder atom interferometer (pulses at 0, T, 2T):
    phi = k_eff [x(0) - 2 x(T) + x(2T)] with x(t) = x0 + v0 t + a t^2/2."""
    t, T, k, x0, v0, a = sp.symbols("t T k_eff x_0 v_0 a", real=True)
    x = x0 + v0 * t + a * t**2 / 2
    return sp.simplify(k * (x.subs(t, 0) - 2 * x.subs(t, T) + x.subs(t, 2 * T)))


def ai_fringe(a, T, contrast=0.5, phase_offset=0.0, wavelength=RB87_WAVELENGTH):
    """Excited-state probability P = (1 - C cos(k_eff a T^2 + phi_L)) / 2: the
    INTENSITY. phi_L is a laser phase step the experimenter controls."""
    return 0.5 * (1 - contrast * np.cos(k_eff(wavelength) * np.asarray(a, float) * T**2 + phase_offset))


def ai_shot_noise_sensitivity(T, n_atoms, contrast=0.5, wavelength=RB87_WAVELENGTH):
    """Per-shot acceleration resolution at mid-fringe: 1 / (C k_eff T^2 sqrt(N))."""
    if T <= 0 or n_atoms <= 0 or not 0 < contrast <= 1:
        raise ValueError("need T > 0, n_atoms > 0, 0 < contrast <= 1")
    return 1.0 / (contrast * k_eff(wavelength) * T**2 * np.sqrt(n_atoms))


def ai_measure(a_true, T, n_atoms, contrast=0.5, phase_offset=0.0, n_shots=1, rng=None):
    """Simulated atom-number readout: fraction excited out of N, n_shots repeats."""
    rng = np.random.default_rng(rng)
    p = ai_fringe(a_true, T, contrast, phase_offset)
    return rng.binomial(int(n_atoms), p, size=n_shots) / n_atoms


def ai_retrieve(measurements, settings, a_grid, n_atoms, contrast=0.5):
    """Least-squares phase retrieval over a candidate grid of accelerations.
    settings: one (T, phase_offset) pair per measurement set.
      * one T: a comb of equally good answers (cos is 2 pi periodic) AND a
        +a / -a twin (cos is even) -- the twin-image ambiguity of GS;
      * several T's: the comb collapses, as two dispersions D1, D2 do in GS;
      * a pi/2 laser phase step: the +/- twin is broken (quadrature readout).
    Returns (best a, cost curve)."""
    a_grid = np.asarray(a_grid, float)
    cost = np.zeros_like(a_grid)
    sigma2 = 0.25 / n_atoms
    for P_meas, (T, ph) in zip(measurements, settings):
        model = ai_fringe(a_grid, T, contrast, ph)
        cost += np.sum((np.atleast_1d(P_meas)[:, None] - model[None, :]) ** 2, axis=0) / sigma2
    return float(a_grid[np.argmin(cost)]), cost


def drift_with_quantum_accel(t_s, sensitivity_per_shot, shot_rate_hz=1.0, gyro_bias=0.0):
    """If the accelerometer bias is calibrated down to the averaged quantum
    resolution after time t (white noise averages as 1/sqrt(shots)), the
    residual position error is dominated by random-walk growth:
    sigma_x(t) ~ sigma_a / sqrt(rate) * t^(3/2) / sqrt(3)."""
    t = np.asarray(t_s, float)
    return sensitivity_per_shot / np.sqrt(shot_rate_hz) * t**1.5 / np.sqrt(3) + G0 * gyro_bias * t**3 / 6


# ── 4. hobby combat robotics: spinner energy and a cost sketch ───────

def disk_spinner_energy(mass_kg, radius_m, rpm):
    """Kinetic energy 1/2 I omega^2 of a solid-disk weapon, I = m r^2 / 2."""
    for n, v in (("mass_kg", mass_kg), ("radius_m", radius_m), ("rpm", rpm)):
        if v <= 0:
            raise ValueError(f"{n} must be positive")
    w = rpm * 2 * np.pi / 60
    return 0.25 * mass_kg * radius_m**2 * w**2


def spinup_time(mass_kg, radius_m, rpm, motor_power_w, efficiency=0.6):
    """Seconds to reach rpm if a fraction of motor power goes into the disk."""
    if motor_power_w <= 0 or not 0 < efficiency <= 1:
        raise ValueError("motor_power_w > 0 and 0 < efficiency <= 1")
    return disk_spinner_energy(mass_kg, radius_m, rpm) / (efficiency * motor_power_w)


# Rough hobby cost ranges (USD) for a first build in each class. These are
# ROUGH PLANNING NUMBERS that vary widely with parts choices and prices --
# replace them with quotes from actual suppliers.
COMBAT_ROBOT_BUDGET_USD = {
    "antweight (1 lb / 454 g)": {"drive motors + ESCs": (40, 120), "weapon motor + ESC": (20, 60),
                                 "battery + charger": (30, 80), "radio tx/rx": (30, 100),
                                 "chassis material / 3D print": (10, 50)},
    "beetleweight (3 lb / 1.36 kg)": {"drive motors + ESCs": (60, 200), "weapon motor + ESC": (40, 150),
                                      "battery + charger": (40, 120), "radio tx/rx": (30, 100),
                                      "chassis (UHMW / aluminum / TPU)": (40, 200)},
}


def budget_range(weight_class):
    items = COMBAT_ROBOT_BUDGET_USD[weight_class]
    return sum(lo for lo, _ in items.values()), sum(hi for _, hi in items.values())
