"""PID controller -- object-oriented, RK4 plant simulation, science solver.

P = Proportional: react to CURRENT error.
I = Integral:     react to ACCUMULATED past error (eliminates steady-state offset).
D = Derivative:   react to RATE of change (damps oscillation, looks ahead).

u(t) = Kp*e(t) + Ki*integral(e(t)) + Kd*d/dt(e(t))

Used everywhere in engineering:
  Aerospace: altitude/attitude hold on a drone
  Photonics: laser cavity length lock (PDH locking uses derivative term)
  This repo: could stabilise the ADC sampling clock in the RogueGuard receiver

OOP structure:
  PIDController -- stateful: holds gains, integrator, last error
  StepResponseSimulator -- wraps any plant ODE + PIDController
  ScienceProblemSolver -- reads problem description, dispatches to module functions

File I/O: ScienceProblemSolver logs results to a JSON file so sessions persist
(intro to OOP + file I/O in one class).
"""
import json
import os
import numpy as np
import sympy as sp


# ── PID controller (OOP) ─────────────────────────────────────────────

class PIDController:
    """Discrete-time PID controller with anti-windup clamp.

    Parameters
    ----------
    Kp, Ki, Kd : float  -- proportional, integral, derivative gains
    dt : float          -- sample period (s)
    u_min, u_max : float -- output clamp (anti-windup)
    """

    def __init__(self, Kp, Ki, Kd, dt=0.01, u_min=-1e9, u_max=1e9):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.dt = float(dt)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, error):
        """Compute control output for one timestep given current error."""
        self._integral += error * self.dt
        derivative = (error - self._prev_error) / self.dt
        u = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        u = float(np.clip(u, self.u_min, self.u_max))
        self._prev_error = error
        return u

    @property
    def integral(self):
        return self._integral

    def __repr__(self):
        return (f"PIDController(Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}, "
                f"dt={self.dt})")


# ── first-order plant: tau * dy/dt + y = K_plant * u ─────────────────

def simulate_first_order(pid, setpoint, tau_plant=1.0, K_plant=1.0,
                          t_end=10.0, disturbance_t=None, disturbance_amp=0.0):
    """Simulate a first-order plant under PID control via Euler integration.

    Plant: tau * dy/dt + y = K_plant * u(t)
    dy/dt = (K_plant * u - y) / tau

    Parameters
    ----------
    pid : PIDController
    setpoint : float or array -- reference trajectory; scalar -> step at t=0
    tau_plant : float         -- plant time constant (s)
    K_plant : float           -- plant DC gain
    t_end : float             -- simulation duration (s)
    disturbance_t : float     -- time to inject a step disturbance (None = none)
    disturbance_amp : float   -- disturbance magnitude

    Returns dict with arrays: t, y, u, error
    """
    pid.reset()
    dt = pid.dt
    t_arr = np.arange(0, t_end, dt)
    n = len(t_arr)
    y_arr = np.zeros(n)
    u_arr = np.zeros(n)
    e_arr = np.zeros(n)

    sp_arr = (np.full(n, setpoint) if np.isscalar(setpoint)
              else np.asarray(setpoint)[:n])

    y = 0.0
    for i, t in enumerate(t_arr):
        # disturbance injection
        d = disturbance_amp if (disturbance_t is not None and t >= disturbance_t) else 0.0
        ref = sp_arr[i]
        err = ref - y
        u = pid.step(err)
        # Euler: y_{i+1} = y_i + dt * (K*u - y) / tau + disturbance
        y += dt * ((K_plant * u - y) / tau_plant + d)
        y_arr[i] = y
        u_arr[i] = u
        e_arr[i] = err

    return {"t": t_arr, "y": y_arr, "u": u_arr, "error": e_arr,
            "setpoint": sp_arr, "pid": pid}


def step_response_metrics(result):
    """Rise time, overshoot, settling time from a step response dict."""
    y = result["y"]
    sp_val = float(result["setpoint"][-1])
    if sp_val == 0:
        return {"rise_time_s": None, "overshoot_pct": 0.0, "settling_time_s": None}
    t = result["t"]
    dt = t[1] - t[0]

    # Rise time: 10% -> 90% of setpoint
    idx_10 = next((i for i, v in enumerate(y) if v >= 0.1 * sp_val), None)
    idx_90 = next((i for i, v in enumerate(y) if v >= 0.9 * sp_val), None)
    rise_time = (idx_90 - idx_10) * dt if (idx_10 is not None and idx_90 is not None) else None

    overshoot = max(0.0, (y.max() - sp_val) / sp_val * 100)

    # Settling time: last time |y - sp| > 2% * sp
    band = 0.02 * abs(sp_val)
    outside = np.where(np.abs(y - sp_val) > band)[0]
    settling_time = float(t[outside[-1]]) if len(outside) > 0 else 0.0

    return {"rise_time_s": rise_time, "overshoot_pct": overshoot,
            "settling_time_s": settling_time}


# ── PID SymPy formalism ───────────────────────────────────────────────

def pid_sympy_5():
    """Five key PID equations in SymPy."""
    t_s = sp.Symbol('t', real=True, positive=True)
    Kp, Ki, Kd = sp.symbols('K_p K_i K_d', positive=True)
    e = sp.Function('e')(t_s)
    u = sp.Function('u')(t_s)
    tau = sp.Symbol('tau', positive=True)
    K_pl = sp.Symbol('K_plant', positive=True)
    y_s = sp.Function('y')(t_s)
    s = sp.Symbol('s')    # Laplace variable

    return {
        "PID_output":
            sp.Eq(u, Kp*e + Ki*sp.Integral(e, t_s) + Kd*sp.Derivative(e, t_s)),
        "First_order_plant":
            sp.Eq(tau*sp.Derivative(y_s, t_s) + y_s, K_pl*u),
        "PID_transfer_function":
            sp.Eq(sp.Symbol('C(s)'), Kp + Ki/s + Kd*s),
        "Closed_loop_tf":
            sp.Eq(sp.Symbol('T(s)'),
                  sp.Symbol('C(s)')*sp.Symbol('P(s)') /
                  (1 + sp.Symbol('C(s)')*sp.Symbol('P(s)'))),
        "Steady_state_error_PI":
            sp.Eq(sp.Symbol('e_ss'), sp.Integer(0)),
    }


# ── OOP science problem solver with JSON file I/O ────────────────────

class ScienceProblemSolver:
    """OOP wrapper: logs solved problems to a JSON file for session persistence.

    Usage
    -----
    solver = ScienceProblemSolver("my_session.json")
    solver.solve("beat_freq", f1=440, f2=443)
    solver.solve("pid_step", Kp=2.0, Ki=0.5, Kd=0.1)
    solver.history()   # returns list of past solved problems
    """

    SUPPORTED = ("beat_freq", "pid_step", "standing_wave", "collatz")

    def __init__(self, filepath="science_session.json"):
        self.filepath = filepath
        self._log = self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                content = f.read().strip()
            if content:
                return json.loads(content)
        return []

    def _save(self):
        with open(self.filepath, "w") as f:
            json.dump(self._log, f, indent=2)

    def solve(self, problem_type, **kwargs):
        """Dispatch to appropriate solver, log result, return result dict."""
        if problem_type not in self.SUPPORTED:
            raise ValueError(f"Unknown problem type '{problem_type}'. "
                             f"Supported: {self.SUPPORTED}")

        if problem_type == "beat_freq":
            from dgs.beat_frequency import beat_frequency
            result = beat_frequency(kwargs["f1"], kwargs["f2"])

        elif problem_type == "pid_step":
            pid = PIDController(
                Kp=kwargs.get("Kp", 1.0),
                Ki=kwargs.get("Ki", 0.1),
                Kd=kwargs.get("Kd", 0.05),
                dt=kwargs.get("dt", 0.01),
            )
            sim = simulate_first_order(
                pid,
                setpoint=kwargs.get("setpoint", 1.0),
                tau_plant=kwargs.get("tau_plant", 1.0),
                t_end=kwargs.get("t_end", 10.0),
            )
            m = step_response_metrics(sim)
            result = {"rise_time_s": m["rise_time_s"],
                      "overshoot_pct": m["overshoot_pct"],
                      "settling_time_s": m["settling_time_s"]}

        elif problem_type == "standing_wave":
            from dgs.beat_frequency import standing_wave_modes
            modes = standing_wave_modes(
                kwargs["L_m"], kwargs["v_ms"],
                n_max=kwargs.get("n_max", 5),
            )
            result = {"modes_hz": modes}

        elif problem_type == "collatz":
            from dgs.algorithms import collatz_sequence
            result = {"sequence": collatz_sequence(kwargs["n"])}

        entry = {"problem": problem_type, "kwargs": kwargs, "result": result}
        self._log.append(entry)
        self._save()
        return result

    def history(self):
        """Return list of all logged problem solutions."""
        return list(self._log)

    def clear(self):
        """Wipe the session log file."""
        self._log = []
        self._save()

    def __repr__(self):
        return f"ScienceProblemSolver('{self.filepath}', {len(self._log)} entries)"


if __name__ == "__main__":
    print("=== PID step response (Kp=2, Ki=0.5, Kd=0.1) ===")
    pid = PIDController(Kp=2.0, Ki=0.5, Kd=0.1, dt=0.01)
    result = simulate_first_order(pid, setpoint=1.0, tau_plant=1.0, t_end=10.0)
    m = step_response_metrics(result)
    print(f"  rise time:    {m['rise_time_s']:.3f} s")
    print(f"  overshoot:    {m['overshoot_pct']:.1f}%")
    print(f"  settling:     {m['settling_time_s']:.2f} s")
    print(f"  final value:  {result['y'][-1]:.4f} (setpoint=1.0)")

    print("\n=== ScienceProblemSolver OOP demo ===")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        solver = ScienceProblemSolver(tmp.name)
    r = solver.solve("beat_freq", f1=440, f2=443)
    print(f"  beat_freq: {r}")
    print(f"  history length: {len(solver.history())}")

    print("\n=== SymPy 5 ===")
    for k, eq in pid_sympy_5().items():
        print(f"  {k}: {eq}")
