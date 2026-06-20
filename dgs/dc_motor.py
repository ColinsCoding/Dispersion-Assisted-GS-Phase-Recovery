"""A permanent-magnet DC motor: the electromechanical model.

Two coupled equations -- an electrical loop (copper resistance + winding
inductance + back-EMF) and a mechanical one (torque vs inertia, friction, and
load):

    electrical:  V = R i + L di/dt + Ke w      (Ke w = back-EMF)
    mechanical:  J dw/dt = Kt i - b w - tau_L

In SI the torque and back-EMF constants are equal, Kt = Ke (one motor constant).
The torque-speed curve is a straight line from the stall torque to the no-load
speed; the "copper" loss is the i^2 R heating in the windings. Standalone (EE);
the magnetostatics behind the torque is griffiths.magnetostatics, the circuit
loop is spice_sym, and closing a speed loop around it is PID control.
"""

import numpy as np


def steady_state(V, R, Ke, Kt, b=0.0, tau_load=0.0):
    """Steady-state speed and current (d/dt = 0).

    w_ss = (Kt V / R - tau_L) / (b + Kt Ke / R),  i_ss = (V - Ke w_ss) / R.
    """
    if R <= 0:
        raise ValueError("R must be > 0")
    w = (Kt * V / R - tau_load) / (b + Kt * Ke / R)
    i = (V - Ke * w) / R
    return {"speed": w, "current": i, "torque": Kt * i}


def no_load_speed(V, R, Ke, Kt, b=0.0):
    """Speed with no external load (tau_L = 0)."""
    return steady_state(V, R, Ke, Kt, b, 0.0)["speed"]


def stall_torque(V, R, Kt):
    """Torque at zero speed: i = V/R (back-EMF zero), tau = Kt V / R."""
    return Kt * V / R


def torque_speed_curve(V, R, Ke, Kt, n=100):
    """The (omega, torque) operating line: tau = Kt V / R - (Kt Ke / R) omega,
    from stall (omega=0) to no-load (tau=0). Returns (omega, tau)."""
    w0 = V / Ke
    w = np.linspace(0, w0, n)
    tau = Kt * V / R - (Kt * Ke / R) * w
    return w, tau


def simulate(V, R, L, Ke, Kt, J, b, tau_load=0.0, dt=1e-4, t_end=0.5, V_of_t=None):
    """Transient response by RK4 on the state [current i, speed w], starting at rest.

    V_of_t(t) optionally overrides the constant supply V (e.g. a step or PWM).
    Returns dict: t, i, w, torque.
    """
    if L <= 0 or J <= 0:
        raise ValueError("L and J must be > 0")
    n = int(t_end / dt)
    t = np.arange(n + 1) * dt
    i = np.zeros(n + 1)
    w = np.zeros(n + 1)

    def deriv(state, tt):
        ii, ww = state
        v = V_of_t(tt) if V_of_t is not None else V
        di = (v - R * ii - Ke * ww) / L
        dw = (Kt * ii - b * ww - tau_load) / J
        return np.array([di, dw])

    s = np.array([0.0, 0.0])
    for k in range(n):
        k1 = deriv(s, t[k])
        k2 = deriv(s + 0.5 * dt * k1, t[k] + 0.5 * dt)
        k3 = deriv(s + 0.5 * dt * k2, t[k] + 0.5 * dt)
        k4 = deriv(s + dt * k3, t[k] + dt)
        s = s + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        i[k + 1], w[k + 1] = s
    return {"t": t, "i": i, "w": w, "torque": Kt * i}


def efficiency(V, i, w, torque):
    """Power split: input V i, mechanical out tau w, copper loss i^2 R folded in
    via P_in - P_out. Returns (eta, P_in, P_out)."""
    P_in = V * i
    P_out = torque * w
    eta = np.where(P_in > 1e-12, P_out / P_in, 0.0)
    return eta, P_in, P_out
