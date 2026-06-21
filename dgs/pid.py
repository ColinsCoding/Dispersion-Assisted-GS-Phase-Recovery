"""PID control -- the feedback loop behind every instrument.

Measure, subtract from the setpoint to get the error, and drive an output that is
the sum of three terms:

    u(t) = Kp e(t) + Ki integral e dt + Kd de/dt
           |          |                  |
        proportional  integral           derivative
        (push now)   (kill steady-state  (damp the
                      offset)             overshoot)

This locks a laser to a wavelength, a current source to a setpoint, a cavity to a
fringe -- the control half of the measure-and-correct loop this repo is about.
Discrete-time, with output clamping + anti-windup. NumPy only. Education.
"""

import numpy as np


class PID:
    """Discrete PID controller. Call update(measurement) each time step dt."""

    def __init__(self, kp, ki, kd, setpoint=0.0, dt=1.0, out_min=None, out_max=None):
        if dt <= 0:
            raise ValueError("dt must be > 0")
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint, self.dt = setpoint, dt
        self.out_min, self.out_max = out_min, out_max
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_error = None

    def update(self, measurement):
        """One control step: returns the actuator command for this measurement."""
        error = self.setpoint - measurement
        deriv = 0.0 if self._prev_error is None else (error - self._prev_error) / self.dt
        # tentative integral; commit only if the output isn't saturating (anti-windup)
        integral = self._integral + error * self.dt
        u = self.kp * error + self.ki * integral + self.kd * deriv
        saturated = ((self.out_max is not None and u > self.out_max) or
                     (self.out_min is not None and u < self.out_min))
        if not saturated:
            self._integral = integral          # only wind up the integral when not clamped
        u = self._clamp(self.kp * error + self.ki * self._integral + self.kd * deriv)
        self._prev_error = error
        return u

    def _clamp(self, u):
        if self.out_max is not None:
            u = min(u, self.out_max)
        if self.out_min is not None:
            u = max(u, self.out_min)
        return u


def simulate(pid, plant, n_steps):
    """Closed-loop sim. `plant(u) -> new output` (a stateful closure). Returns
    (t, y, u) arrays. The controller sees the output, commands u, the plant moves."""
    y = 0.0
    ts, ys, us = [], [], []
    for i in range(n_steps):
        u = pid.update(y)
        ys.append(y); us.append(u); ts.append(i * pid.dt)
        y = float(plant(u))
    return np.array(ts), np.array(ys), np.array(us)


def first_order_plant(tau, gain=1.0, dt=1.0, y0=0.0):
    """Lag plant y' = -y/tau + gain*u (stateful closure plant(u) -> y)."""
    s = {"y": float(y0)}
    def step(u):
        s["y"] += dt * (-s["y"] / tau + gain * u)
        return s["y"]
    return step


def second_order_plant(wn, zeta, gain=1.0, dt=0.05, y0=0.0):
    """Second-order plant y'' = wn^2(gain*u - y) - 2*zeta*wn*y' (mass-spring-damper).
    Underdamped (zeta<1) plants OVERSHOOT under PI control -- which is where the
    derivative term earns its keep. Stateful closure plant(u) -> y."""
    s = {"y": float(y0), "v": 0.0}
    def step(u):
        a = wn**2 * (gain * u - s["y"]) - 2 * zeta * wn * s["v"]
        s["v"] += a * dt
        s["y"] += s["v"] * dt
        return s["y"]
    return step


if __name__ == "__main__":
    def mk():
        return second_order_plant(wn=1.0, zeta=0.25, dt=0.05)
    for name, pid in [("P only ", PID(3.0, 0.0, 0.0, setpoint=1.0, dt=0.05)),
                      ("PI     ", PID(3.0, 2.0, 0.0, setpoint=1.0, dt=0.05)),
                      ("PID    ", PID(3.0, 2.0, 2.0, setpoint=1.0, dt=0.05))]:
        _, y, _ = simulate(pid, mk(), 1200)
        print(f"{name}: final={y[-1]:.3f}  peak={y.max():.3f}  overshoot={max(0,y.max()-1)*100:.0f}%")
