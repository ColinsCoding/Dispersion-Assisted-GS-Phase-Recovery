"""Cross-validate dgs.gyroscopes against a real, independent physics
engine (MuJoCo) instead of just our own RK4 code.

Two scenarios, both built with an EXPLICIT <inertial> tag (mass, pos,
diaginertia, identity quat) rather than letting MuJoCo auto-derive inertia
from geom shape -- auto-derived inertia picks its OWN principal-axis frame,
which need not align with the body's geometric local axes (verified
directly: for a simple cylinder, MuJoCo assigned the spin-axis moment of
inertia to local X, not Z, silently invalidating any "qvel = spin about
local Z" assumption). Pinning diaginertia to an identity-quat inertial
frame removes that ambiguity entirely.

CONVENTION NOTE, also verified directly rather than assumed: for MuJoCo
ball/free joints, the angular qvel components should be set in the BODY-
LOCAL frame regardless of the body's current orientation (qpos) -- setting
qvel=[0,0,omega] means "spin about the body's own local z-axis" even when
that axis has been tilted away from world-z, not "spin about world-z".
Rotating the desired body-frame vector by the initial orientation quaternion
first (the natural-seeming thing to try) is WRONG and produces a top that
visibly tumbles instead of precessing -- confirmed by direct comparison.

  1. build_precessing_top_model / simulate_precessing_top: the classic
     gravity-driven top from dgs.gyroscopes.precession_rate.
  2. build_asymmetric_free_body_model / simulate_free_rigid_body: the
     torque-free asymmetric top from dgs.gyroscopes.euler_rigid_body_rhs /
     the tennis racket theorem.
"""

import numpy as np
import mujoco


# -- Scenario 1: gravity-driven precessing top --------------------------------

def build_precessing_top_model(m_disk, R_disk, r, g=9.80665, dt=0.0002):
    """A disk of mass m_disk, radius R_disk, mounted a distance r from a
    fixed ball-joint pivot along the shaft. I_spin (disk about its own
    symmetry axis) and I_transverse (about a diameter) are set explicitly
    via <inertial>, pinned to the body's local z-axis by the (default,
    identity) inertial-frame quat."""
    I_spin = 0.5 * m_disk * R_disk ** 2
    I_transverse = 0.25 * m_disk * R_disk ** 2
    xml = f"""
    <mujoco>
      <option gravity="0 0 -{g}" timestep="{dt}" integrator="RK4"/>
      <worldbody>
        <body name="top" pos="0 0 0">
          <joint type="ball" pos="0 0 0"/>
          <inertial pos="0 0 {r}" mass="{m_disk}" diaginertia="{I_transverse} {I_transverse} {I_spin}"/>
          <geom type="cylinder" size="{R_disk} 0.01" pos="0 0 {r}" mass="{m_disk}"
                rgba="0.8 0.2 0.2 1" contype="0" conaffinity="0"/>
          <geom type="cylinder" size="0.005 {r / 2}" pos="0 0 {r / 2}" mass="0.0"
                rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml), I_spin, I_transverse


def simulate_precessing_top(m_disk, R_disk, r, omega_spin, theta0=0.3, t_max=2.0,
                             g=9.80665, dt=0.0002):
    """Release the top from rest (pure spin, zero precession/nutation
    rate) at tilt theta0 and track the shaft direction over time. Returns
    t, phi (azimuthal/precession angle), theta (polar/tilt angle),
    and the measured MEAN precession rate (total phi change / total time)."""
    model, I_spin, I_transverse = build_precessing_top_model(m_disk, R_disk, r, g=g, dt=dt)
    data = mujoco.MjData(model)

    axis = np.array([1.0, 0.0, 0.0])
    quat0 = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat0, axis, theta0)
    data.qpos[:] = quat0
    data.qvel[:] = [0.0, 0.0, omega_spin]   # body-local frame -- spin about the shaft's own axis
    mujoco.mj_forward(model, data)

    n_steps = int(t_max / dt)
    t = np.zeros(n_steps)
    phi = np.zeros(n_steps)
    theta = np.zeros(n_steps)
    for i in range(n_steps):
        zdir = np.zeros(3)
        mujoco.mju_rotVecQuat(zdir, np.array([0.0, 0.0, 1.0]), data.qpos[:4])
        phi[i] = np.arctan2(zdir[1], zdir[0])
        theta[i] = np.arccos(np.clip(zdir[2], -1.0, 1.0))
        t[i] = data.time
        mujoco.mj_step(model, data)

    phi_unwrapped = np.unwrap(phi)
    mean_precession_rate = (phi_unwrapped[-1] - phi_unwrapped[0]) / (t[-1] - t[0])
    # parallel-axis theorem: the pivot is a distance r from the disk's
    # center of mass, so the transverse inertia ABOUT THE PIVOT (what
    # dgs.gyroscopes.nutation_frequency actually needs) is I_transverse
    # (about the center) plus m*r**2 -- see nutation_frequency's docstring
    # for why conflating the two silently breaks the comparison.
    I_transverse_pivot = I_transverse + m_disk * r ** 2
    return {"t": t, "phi": phi_unwrapped, "theta": theta,
            "mean_precession_rate": mean_precession_rate, "I_spin": I_spin,
            "I_transverse": I_transverse, "I_transverse_pivot": I_transverse_pivot}


def measured_nutation_frequency(t, theta):
    """Extract the dominant oscillation frequency of theta(t) (the tilt
    angle's wobble about its slow precession-averaged drift) via FFT, and
    return it as an angular frequency (rad/s) -- directly comparable to
    dgs.gyroscopes.nutation_frequency's fast-top prediction
    omega_n = I_spin*omega_spin/I_transverse.

    Detrending matters: theta(t) is a nutation oscillation RIDING ON TOP
    of a slow settling drift (the top released from rest at theta0 sags a
    bit before nutating around its new mean), not a pure oscillation about
    a fixed mean -- feeding the raw signal to an FFT would leak power into
    spuriously low frequencies from that drift. Subtracting a linear fit
    removes it without touching the oscillation itself."""
    dt = float(t[1] - t[0])
    linear_fit = np.polyval(np.polyfit(t, theta, 1), t)
    detrended = theta - linear_fit
    spectrum = np.abs(np.fft.rfft(detrended))
    freqs = np.fft.rfftfreq(len(theta), d=dt)
    peak_idx = 1 + np.argmax(spectrum[1:])   # skip the DC bin (index 0)
    f_peak = freqs[peak_idx]
    return 2 * np.pi * f_peak


# -- Scenario 2: torque-free asymmetric top (tennis racket theorem) -----------

def build_asymmetric_free_body_model(I1, I2, I3, mass=1.0, dt=0.001):
    """A free body with EXPLICIT diagonal inertia (I1, I2, I3) about its
    own local x, y, z -- no gravity, no contact, pure torque-free rotation,
    exactly the setup dgs.gyroscopes.integrate_euler_rigid_body solves
    analytically/numerically without a physics engine at all."""
    xml = f"""
    <mujoco>
      <option gravity="0 0 0" timestep="{dt}" integrator="RK4"/>
      <worldbody>
        <body name="box">
          <freejoint/>
          <inertial pos="0 0 0" mass="{mass}" diaginertia="{I1} {I2} {I3}"/>
          <geom type="box" size="0.3 0.4 0.5" mass="{mass}" contype="0" conaffinity="0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def simulate_free_rigid_body(omega0, I1, I2, I3, t_max=20.0, dt=0.001):
    """Integrate the free body forward with MuJoCo's own physics engine
    (not our RK4 code) and return the BODY-FRAME angular velocity history,
    obtained by rotating the (freejoint, body-local) qvel back through the
    current orientation quaternion at every step."""
    model = build_asymmetric_free_body_model(I1, I2, I3, dt=dt)
    data = mujoco.MjData(model)
    data.qvel[3:6] = omega0
    mujoco.mj_forward(model, data)

    n_steps = int(t_max / dt)
    t = np.arange(n_steps) * dt
    omega_body = np.zeros((n_steps, 3))
    for i in range(n_steps):
        w_local = data.qvel[3:6].copy()   # already body-local for a freejoint (same convention as ball joint)
        omega_body[i] = w_local
        mujoco.mj_step(model, data)
    return {"t": t, "omega": omega_body}


if __name__ == "__main__":
    print("=== Scenario 1: gravity-driven precessing top, vs. dgs.gyroscopes.precession_rate ===\n")
    from dgs.gyroscopes import precession_rate, nutation_frequency

    m_disk, R_disk, r = 0.2, 0.1, 0.3
    for omega_spin in (300.0, 1000.0, 3000.0):
        run = simulate_precessing_top(m_disk, R_disk, r, omega_spin)
        analytic = precession_rate(mass=m_disk, g=9.80665, r=r, I_spin=run["I_spin"], omega_spin=omega_spin)
        ratio = run["mean_precession_rate"] / analytic["Omega_p_rad_s"]
        print(f"omega_spin={omega_spin:6.0f}: theta range [{run['theta'].min():.4f},{run['theta'].max():.4f}]  "
              f"measured Omega_p={run['mean_precession_rate']:.4f}  analytic={analytic['Omega_p_rad_s']:.4f}  "
              f"ratio={ratio:.4f}")

    print("\n=== Nutation frequency: FFT of MuJoCo's theta(t), vs. dgs.gyroscopes.nutation_frequency ===\n")
    print("(I_transverse must be about the PIVOT, not the disk's own center of mass --")
    print(" using the center-of-mass value first gave ratios of ~0.02-0.03; the parallel-axis")
    print(" term m*r**2 dominates I_transverse here (0.0005 about center vs 0.018 from m*r**2))\n")
    for omega_spin in (300.0, 1000.0, 3000.0):
        run = simulate_precessing_top(m_disk, R_disk, r, omega_spin)
        omega_n_measured = measured_nutation_frequency(run["t"], run["theta"])
        omega_n_analytic = nutation_frequency(run["I_spin"], run["I_transverse_pivot"], omega_spin)
        ratio = omega_n_measured / omega_n_analytic
        print(f"omega_spin={omega_spin:6.0f}: measured omega_n={omega_n_measured:8.2f} rad/s  "
              f"analytic omega_n={omega_n_analytic:8.2f} rad/s  ratio={ratio:.4f}")

    print("\n=== Scenario 2: torque-free asymmetric top, vs. dgs.gyroscopes tennis racket theorem ===\n")
    I1, I2, I3 = 1.0, 2.0, 3.0
    for name, idx, omega0 in [("axis1 (smallest I)", 0, [5.0, 1e-3, 1e-3]),
                               ("axis2 (intermediate I)", 1, [1e-3, 5.0, 1e-3]),
                               ("axis3 (largest I)", 2, [1e-3, 1e-3, 5.0])]:
        run = simulate_free_rigid_body(omega0, I1, I2, I3, t_max=20.0, dt=0.001)
        transverse = np.delete(run["omega"], idx, axis=1)
        max_transverse = float(np.max(np.abs(transverse)))
        print(f"{name}: max transverse omega = {max_transverse:.4f}  "
              f"({'FLIPPED' if max_transverse > 2.5 else 'stayed bounded'})")
