"""The spoon: F = dp/dt = m*a as a CONTINUOUS force applied over a
scooping motion, in contrast to the tongs' single instantaneous impulse
(dgs.chicken_bbq_simulator.apply_flip_impulse). "Digital mechanics" here
means literally that: Newton's second law is a differential equation, and
a digital computer can only approximate its solution by discretizing time
-- this module makes that approximation error visible and measured, not
hidden.

Three levels of rigor, all solving the SAME constant-force problem:
  1. exact_constant_force_solution -- the closed-form answer (F/m is
     constant, so v(t) and x(t) are exactly integrable by hand).
  2. integrate_force_euler -- a hand-rolled semi-implicit Euler
     integrator: the simplest possible digital approximation. Velocity
     comes out exact (dv/dt=a=const sums exactly regardless of step
     count), but POSITION has real, measured O(dt) error -- a genuine,
     textbook first-order truncation error, not a bug.
  3. simulate_force_mujoco -- the same constant force applied to a real
     MuJoCo free body via xfrc_applied, integrated by MuJoCo's own RK4
     scheme. Matches the exact solution to machine precision (RK4 is
     4th-order; the error shrinks so fast that finite dt is
     indistinguishable from dt->0 at these scales) -- a real demonstration
     of why choice of integrator matters, not just step size.
"""

import numpy as np
import mujoco


def exact_constant_force_solution(F, m, v0, x0, t):
    """Closed-form v(t), x(t) for constant force F on mass m: a=F/m,
    v(t)=v0+a*t, x(t)=x0+v0*t+0.5*a*t^2."""
    a = F / m
    t = np.asarray(t, dtype=float)
    v = v0 + a * t
    x = x0 + v0 * t + 0.5 * a * t ** 2
    return v, x


def integrate_force_euler(F, m, v0, x0, duration, dt):
    """Semi-implicit ("symplectic") Euler: v_{n+1}=v_n+a*dt,
    x_{n+1}=x_n+v_{n+1}*dt. The simplest possible digital approximation
    of F=ma -- first-order accurate in position, exact in velocity for a
    CONSTANT force (a special case; for a time-varying force velocity
    would also pick up O(dt) error)."""
    n = int(round(duration / dt))
    a = F / m
    t = np.zeros(n + 1)
    v = np.zeros(n + 1)
    x = np.zeros(n + 1)
    v[0], x[0] = v0, x0
    for i in range(n):
        v[i + 1] = v[i] + a * dt
        x[i + 1] = x[i] + v[i + 1] * dt
        t[i + 1] = t[i] + dt
    return t, v, x


def simulate_force_mujoco(F, m, v0, x0, duration, dt=0.001):
    """Apply the SAME constant force to a real MuJoCo free body
    (xfrc_applied) and let MuJoCo's own RK4 integrator solve it --
    reusing an actual physics engine's numerics, not a second hand-rolled
    approximation."""
    xml = f"""
    <mujoco>
      <option gravity="0 0 0" timestep="{dt}" integrator="RK4"/>
      <worldbody>
        <body name="item" pos="{x0} 0 0">
          <freejoint/>
          <inertial pos="0 0 0" mass="{m}" diaginertia="0.0001 0.0001 0.0001"/>
          <geom type="sphere" size="0.02" mass="{m}" contype="0" conaffinity="0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "item")
    data.qvel[0] = v0
    data.xfrc_applied[body_id, 0] = F
    mujoco.mj_forward(model, data)

    n_steps = int(round(duration / dt))
    t = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    x = np.zeros(n_steps + 1)
    v[0], x[0] = data.qvel[0], data.qpos[0]
    for i in range(n_steps):
        mujoco.mj_step(model, data)
        t[i + 1] = data.time
        v[i + 1] = data.qvel[0]
        x[i + 1] = data.qpos[0]
    return t, v, x


class SpoonScoop:
    """A spoon applying a continuous force to a food item over a scooping
    motion -- the F=ma counterpart to the tongs' apply_flip_impulse. The
    force profile is a smooth ease-in/ease-out (not a step function): real
    scooping accelerates and decelerates, it doesn't snap on and off."""

    def __init__(self, peak_force, mass, duration):
        self.peak_force = float(peak_force)
        self.mass = float(mass)
        self.duration = float(duration)

    def force_at(self, t):
        """Smooth force profile: zero at t=0 and t=duration, peaking at
        the midpoint -- sin(pi*t/duration), scaled to peak_force."""
        t = np.asarray(t, dtype=float)
        in_range = (t >= 0) & (t <= self.duration)
        force = self.peak_force * np.sin(np.pi * np.clip(t, 0, self.duration) / self.duration)
        return np.where(in_range, force, 0.0)

    def integrate_euler(self, v0=0.0, x0=0.0, dt=1e-4):
        """Numerically integrate the FULL time-varying force profile
        (not just the constant-force special case) via semi-implicit
        Euler."""
        n = int(round(self.duration / dt))
        t = np.zeros(n + 1)
        v = np.zeros(n + 1)
        x = np.zeros(n + 1)
        v[0], x[0] = v0, x0
        for i in range(n):
            a_i = self.force_at(t[i]) / self.mass
            v[i + 1] = v[i] + a_i * dt
            x[i + 1] = x[i] + v[i + 1] * dt
            t[i + 1] = t[i] + dt
        return t, v, x


if __name__ == "__main__":
    F, m, v0, x0, duration = 5.0, 0.15, 0.0, 0.0, 0.5

    print("=== Constant force F=ma: exact vs. hand-rolled Euler vs. MuJoCo RK4 ===\n")
    v_exact, x_exact = exact_constant_force_solution(F, m, v0, x0, duration)
    print(f"exact:  v={v_exact:.6f}  x={x_exact:.6f}")

    for dt in (0.1, 0.01, 0.001, 0.0001):
        t, v, x = integrate_force_euler(F, m, v0, x0, duration, dt)
        print(f"Euler dt={dt:7.4f}:  v={v[-1]:.6f} (err {abs(v[-1]-v_exact):.2e})  "
              f"x={x[-1]:.6f} (err {abs(x[-1]-x_exact):.2e})")

    t_mj, v_mj, x_mj = simulate_force_mujoco(F, m, v0, x0, duration)
    print(f"MuJoCo RK4 dt=0.001:  v={v_mj[-1]:.6f} (err {abs(v_mj[-1]-v_exact):.2e})  "
          f"x={x_mj[-1]:.6f} (err {abs(x_mj[-1]-x_exact):.2e})")

    print("\n=== SpoonScoop: a smooth, physically realistic force profile ===\n")
    # 50g of food, a gentle 0.1N push over 0.6s -- realistic scoop scale,
    # not a launch: peaks around 0.76 m/s and travels ~23cm
    scoop = SpoonScoop(peak_force=0.1, mass=0.05, duration=0.6)
    t, v, x = scoop.integrate_euler(dt=1e-4)
    print(f"peak velocity reached: {v.max():.3f} m/s")
    print(f"final position: {x[-1]:.4f} m")
