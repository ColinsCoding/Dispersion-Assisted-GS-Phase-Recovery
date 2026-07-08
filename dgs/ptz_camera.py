"""A PTZ camera as a 2-DOF gimbal: pointing, statics, and Lagrangian dynamics.

A pan-tilt (PTZ) camera is a two-degree-of-freedom mechanism -- a pan angle theta
(azimuth, about the vertical axis) and a tilt angle phi (elevation, about a horizontal
axis). Those two "variables" are what a controller commands; everything about aiming
and driving the camera follows from them:

  POINTING (kinematics). The line of sight is the unit vector
        (cos phi cos theta, cos phi sin theta, sin phi),
  so aiming at a target is the inverse: theta = atan2(y, x), phi = asin(z). Point the
  camera by solving for (theta, phi).

  STATICS. The camera's center of mass sits a distance d out along the optical axis,
  so gravity exerts a tilt torque. To HOLD a tilt without moving, the motor must
  supply
        tau_hold = m g d cos(phi)
  -- maximum when the camera looks level (phi = 0, CG cantilevered out) and ZERO when
  it looks straight up or down (CG over the axis). This is the static sizing of the
  tilt motor.

  DYNAMICS (Lagrangian). With L = T - V and
        T = 1/2 I_tilt phi_dot^2 + 1/2 (I_p0 + m d^2 cos^2 phi) theta_dot^2,
        V = m g d sin phi,
  the Euler-Lagrange equations give the accelerations under motor torques -- INCLUDING
  the coupling: tilting changes the pan-axis inertia (m d^2 cos^2 phi), so a fast pan
  pushes on the tilt (a centrifugal term) and vice versa. With no torque or friction,
  total energy is conserved -- the check used here.

Builds on the same Lagrangian machinery as dgs.lagrangian and the same
symmetry/energy-conservation idea as dgs.symmetry_physics. NumPy only; py-3.13.
"""

import numpy as np

G = 9.80665      # m/s^2


# ----------------------------------------------------------------------
# Pointing kinematics: (pan, tilt) <-> line of sight
# ----------------------------------------------------------------------

def pointing_direction(pan, tilt):
    """Camera line-of-sight unit vector for pan (azimuth) and tilt (elevation),
    both in radians: (cos phi cos theta, cos phi sin theta, sin phi)."""
    return np.array([np.cos(tilt) * np.cos(pan),
                     np.cos(tilt) * np.sin(pan),
                     np.sin(tilt)])


def aim_at(target_direction):
    """Inverse pointing: the (pan, tilt) that aims the camera along a target
    direction. tilt = asin(z), pan = atan2(y, x). Returns radians."""
    v = np.asarray(target_direction, float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("target direction must be nonzero")
    v = v / n
    return float(np.arctan2(v[1], v[0])), float(np.arcsin(np.clip(v[2], -1, 1)))


# ----------------------------------------------------------------------
# Statics: the torque to hold a tilt against gravity
# ----------------------------------------------------------------------

def holding_torque(mass, cg_distance, tilt, g=G):
    """Static tilt-motor torque needed to hold the camera at `tilt` against
    gravity: tau = m g d cos(phi). Max at phi=0 (level), zero looking up/down."""
    if mass < 0 or cg_distance < 0:
        raise ValueError("mass and cg_distance must be non-negative")
    return mass * g * cg_distance * np.cos(tilt)


# ----------------------------------------------------------------------
# Lagrangian dynamics
# ----------------------------------------------------------------------

def pan_inertia(tilt, I_p0, mass, cg_distance):
    """Moment of inertia about the vertical PAN axis, which depends on tilt: the
    camera's horizontal lever arm is d cos(phi), so I_pan = I_p0 + m d^2 cos^2 phi."""
    return I_p0 + mass * cg_distance ** 2 * np.cos(tilt) ** 2


def forward_dynamics(state, tau_pan, tau_tilt, params):
    """Angular accelerations (theta_ddot, phi_ddot) from the Euler-Lagrange
    equations. state = (theta, phi, theta_dot, phi_dot); params has keys I_p0,
    I_tilt, mass, cg_distance, g. Includes the pan-tilt inertial coupling."""
    theta, phi, td, pd = state
    I_p0, I_t = params["I_p0"], params["I_tilt"]
    m, d, g = params["mass"], params["cg_distance"], params.get("g", G)
    I_pan = pan_inertia(phi, I_p0, m, d)
    dIpan_dphi = -m * d ** 2 * np.sin(2 * phi)          # d/dphi (m d^2 cos^2 phi)
    # pan: I_pan*theta_ddot + dIpan_dphi*phi_dot*theta_dot = tau_pan
    theta_ddot = (tau_pan - dIpan_dphi * pd * td) / I_pan
    # tilt: I_tilt*phi_ddot + 1/2 m d^2 sin(2phi) theta_dot^2 + m g d cos(phi) = tau_tilt
    phi_ddot = (tau_tilt + 0.5 * dIpan_dphi * td ** 2 - m * g * d * np.cos(phi)) / I_t
    return theta_ddot, phi_ddot


def energy(state, params):
    """Total mechanical energy T + V -- conserved when no torque or friction acts."""
    theta, phi, td, pd = state
    I_p0, I_t = params["I_p0"], params["I_tilt"]
    m, d, g = params["mass"], params["cg_distance"], params.get("g", G)
    T = 0.5 * I_t * pd ** 2 + 0.5 * pan_inertia(phi, I_p0, m, d) * td ** 2
    V = m * g * d * np.sin(phi)
    return float(T + V)


def simulate(state0, params, tau_pan=0.0, tau_tilt=0.0, t_end=2.0, dt=1e-3):
    """Integrate the gimbal with RK4 under constant motor torques. Returns dict
    with the trajectory, the energy time series, and its fractional drift (near
    zero with no applied torque -- the conservation check)."""
    if t_end <= 0 or dt <= 0:
        raise ValueError("t_end and dt must be positive")

    def deriv(s):
        td, pd = s[2], s[3]
        ta, pa = forward_dynamics(s, tau_pan, tau_tilt, params)
        return np.array([td, pd, ta, pa])

    n = int(t_end / dt)
    s = np.asarray(state0, float)
    traj = np.empty((n + 1, 4)); traj[0] = s
    for i in range(n):
        k1 = deriv(s); k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2); k4 = deriv(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[i + 1] = s
    E = np.array([energy(p, params) for p in traj])
    drift = float(np.ptp(E) / abs(E[0])) if E[0] != 0 else float(np.ptp(E))
    return {"trajectory": traj, "energy": E, "energy_drift": drift}


if __name__ == "__main__":
    print("POINTING: (pan, tilt) -> line of sight")
    for pan, tilt in [(0, 0), (np.pi/2, 0), (0, np.pi/2)]:
        los = pointing_direction(pan, tilt)
        print(f"  pan={np.degrees(pan):5.0f} tilt={np.degrees(tilt):5.0f} -> "
              f"{np.round(los, 3)}")
    tgt = [1, 1, 1]
    pan, tilt = aim_at(tgt)
    print(f"  aim at {tgt}: pan={np.degrees(pan):.1f} deg, tilt={np.degrees(tilt):.1f} deg, "
          f"round-trip {np.round(pointing_direction(pan, tilt)*np.sqrt(3), 3)}")

    print("\nSTATICS: tilt holding torque tau = m g d cos(phi)")
    m, d = 0.5, 0.05
    for deg in (0, 45, 90):
        print(f"  tilt={deg:3d} deg -> {holding_torque(m, d, np.radians(deg)):.3f} N.m")

    print("\nDYNAMICS: released from level, no motor torque (a pendulum in tilt)")
    params = {"I_p0": 0.01, "I_tilt": 0.005, "mass": m, "cg_distance": d}
    sim = simulate([0.0, 0.0, 0.0, 0.0], params, t_end=3.0)
    print(f"  energy drift over 3 s: {sim['energy_drift']:.2e} (conserved)")
    print(f"  tilt swings to min {np.degrees(sim['trajectory'][:,1].min()):.1f} deg "
          f"(toward straight-down, CG lowest)")

    # static check: hold at 30 deg with the computed holding torque -> no tilt accel
    phi0 = np.radians(30)
    tau = holding_torque(m, d, phi0)
    _, phi_ddot = forward_dynamics([0, phi0, 0, 0], 0.0, tau, params)
    print(f"\n  holding torque at 30 deg gives tilt acceleration {phi_ddot:.2e} (static)")
