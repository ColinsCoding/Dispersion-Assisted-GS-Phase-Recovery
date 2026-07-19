"""Speed wobble (skateboard/bicycle/shopping-cart shimmy): a real
self-excited instability, and the numerical-integrator artifact that can
fake one.

The steering/truck angle phi(t) of a skateboard, bicycle fork, or castor
wheel obeys (to leading order) a damped oscillator whose EFFECTIVE damping
decreases with forward speed v:

    m phi'' + b(v) phi' + k phi = 0,     b(v) = b0 - b1*v

Below a critical speed v_crit = b0/b1, damping is positive and any wobble
decays. Above it, damping goes negative -- the system actively pumps energy
INTO the oscillation, and a tiny perturbation grows exponentially: this is
the real physics of riding "too fast" and the board/bike starting to shake
uncontrollably (true shimmy).

Separately: a numerical integrator with too large a time step can produce
exactly the same symptom (growing oscillation) on a system that is
PHYSICALLY stable (v < v_crit) -- numerical instability mimicking real
instability. Comparing explicit Euler against RK4 at the same physically-
stable speed is how you tell the two apart.
"""

import numpy as np


def critical_speed(b0, b1):
    """v_crit = b0 / b1 -- the forward speed above which effective damping
    b(v) = b0 - b1*v goes negative and any wobble grows instead of decaying."""
    return b0 / b1


def wobble_rhs(state, v, m, b0, b1, k):
    """state = [phi, phi_dot]. Returns d(state)/dt for
    m phi'' + (b0 - b1*v) phi' + k phi = 0."""
    phi, phi_dot = state
    b_eff = b0 - b1 * v
    phi_ddot = -(b_eff * phi_dot + k * phi) / m
    return np.array([phi_dot, phi_ddot])


def euler_step(f, state, h, *args):
    """One explicit (forward) Euler step -- cheap, but only conditionally
    stable: too large an h can make even a physically decaying system blow
    up numerically."""
    return state + h * f(state, *args)


def rk4_step(f, state, h, *args):
    """One classical 4th-order Runge-Kutta step -- much better stability
    margin than Euler at the same step size."""
    k1 = f(state, *args)
    k2 = f(state + h / 2 * k1, *args)
    k3 = f(state + h / 2 * k2, *args)
    k4 = f(state + h * k3, *args)
    return state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_wobble(v, m, b0, b1, k, phi0, phi_dot0, t, method="rk4"):
    """Integrate the wobble ODE over time grid `t` with the given fixed-step
    method ("euler" or "rk4"). Returns the (T, 2) [phi, phi_dot] trajectory."""
    step = euler_step if method == "euler" else rk4_step
    state = np.array([phi0, phi_dot0], dtype=float)
    out = np.zeros((len(t), 2))
    out[0] = state
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        state = step(wobble_rhs, state, h, v, m, b0, b1, k)
        out[i + 1] = state
    return out


def envelope_growth_rate(phi_trajectory, t):
    """Fit an exponential envelope |phi(t)| ~ exp(lambda t) to the trajectory
    (via a linear fit on log|phi|), returning lambda: negative = decaying
    (stable), positive = growing (unstable wobble, real or numerical)."""
    amplitude = np.abs(phi_trajectory)
    amplitude = np.clip(amplitude, 1e-12, None)   # avoid log(0)
    log_amp = np.log(amplitude)
    A = np.vstack([t, np.ones_like(t)]).T
    slope, intercept = np.linalg.lstsq(A, log_amp, rcond=None)[0]
    return float(slope)
