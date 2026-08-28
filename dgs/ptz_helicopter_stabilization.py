"""Mounting dgs.ptz_camera's 2-DOF gimbal on a helicopter: platform-vibration
disturbance rejection.

dgs.ptz_camera derives the gimbal's own pointing/statics/dynamics assuming a
FIXED base. Bolt that same gimbal to a helicopter and the base itself rotates
-- rotor imbalance and blade-pass forces shake the airframe at well-known,
narrow frequencies, and that shaking couples straight into where the camera
points, whether or not the gimbal motors move at all.

THE KEY RELATION (kept to one axis -- tilt -- for clarity; pan couples the
same way): what actually matters for keeping a target in frame is the
INERTIAL line-of-sight angle, not the gimbal's angle relative to its own
mounting bracket:

    inertial_tilt(t) = base_disturbance_tilt(t) + gimbal_relative_tilt(t)

A gimbal with no active control just sits at gimbal_relative_tilt ~ 0, so
the full platform vibration passes straight through to inertial_tilt --
this is why an un-stabilized camera bolted to a helicopter is useless past a
few hundred feet. An INERTIALLY stabilized gimbal instead drives
gimbal_relative_tilt to actively COUNTER-ROTATE against the disturbance
(dgs.pid.PID closing the loop on the inertial angle, not the relative one),
using dgs.ptz_camera.forward_dynamics unmodified as the actual tilt-axis
plant the controller commands torque into.

REALISTIC DISTURBANCE FREQUENCIES: a light helicopter's main rotor turns at
roughly 300-400 RPM (~5-7 Hz, "1/rev"); a 2-bladed teetering rotor (Robinson
R22/R44-class) produces its dominant vibration at BLADE-PASS frequency,
2x that, ~11-14 Hz. MAIN_ROTOR_HZ/BLADE_PASS_HZ below use representative
values in that range, not a specific aircraft's measured spectrum.
"""

import numpy as np

from dgs.ptz_camera import forward_dynamics
from dgs.pid import PID

MAIN_ROTOR_HZ = 5.5      # representative light-helicopter main-rotor 1/rev (~330 RPM)
N_BLADES = 2
BLADE_PASS_HZ = MAIN_ROTOR_HZ * N_BLADES   # dominant vibration harmonic, 2-bladed rotor


def _check_positive(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def base_disturbance_tilt(t, amplitude_deg=1.5, freq_hz=BLADE_PASS_HZ):
    """Platform (helicopter body) attitude disturbance at time t: a sinusoid
    at the dominant vibration frequency. Real airframe vibration is a
    narrowband spectrum peaked here, not a pure tone, but a single sinusoid
    is the right first-order model for whether a controller can reject it
    at all."""
    if amplitude_deg < 0:
        raise ValueError(f"amplitude_deg must be non-negative, got {amplitude_deg}")
    _check_positive(freq_hz, "freq_hz")
    return np.radians(amplitude_deg) * np.sin(2 * np.pi * freq_hz * t)


def default_gimbal_params():
    """Same order-of-magnitude camera as dgs.ptz_camera's own demo: a small
    PTZ unit, ~0.5 kg, CG 5 cm out along the optical axis."""
    return {"I_p0": 0.01, "I_tilt": 0.005, "mass": 0.5, "cg_distance": 0.05}


def simulate_stabilization(t_end=2.0, dt=1e-3, disturbance_amplitude_deg=1.5,
                            disturbance_freq_hz=BLADE_PASS_HZ,
                            pid_gains=(150.0, 20.0, 5.0), reference="inertial",
                            gimbal_params=None):
    """Simulate the tilt axis of a gimbal bolted to a vibrating helicopter
    body, controlled by the SAME PID structure and gains in both cases --
    only what it's told to measure differs:

      reference="relative": the controller holds the gimbal's tilt at 0
      RELATIVE to its own mounting bracket (a basic position-hold servo
      with no idea the base is rotating -- no gyro feedback). Since the
      controller does its job and keeps gimbal_relative_tilt near 0, the
      inertial tilt ends up being just about the base disturbance,
      UNFILTERED. This is the realistic non-stabilized baseline -- not
      "zero torque" (which just free-falls under gravity, as
      dgs.ptz_camera's own pendulum demo shows), but "holds its own
      angle and nothing more."

      reference="inertial": the controller measures the INERTIAL tilt
      (base disturbance + gimbal's own relative tilt) -- the number a
      gyroscope on the camera itself would report -- and actively drives
      the gimbal to counter-rotate, canceling the disturbance at the
      inertial level. This is what a real 3-axis stabilized gimbal does.

    Both cases use dgs.ptz_camera.forward_dynamics unmodified as the
    tilt-axis plant. Returns dict with time, inertial_tilt,
    gimbal_relative_tilt, base_tilt (all radians), torque history, and
    the RMS inertial-tilt error (radians) -- the actual "how well is the
    camera actually pointed" number in both cases."""
    if t_end <= 0 or dt <= 0:
        raise ValueError("t_end and dt must be positive")
    if reference not in ("relative", "inertial"):
        raise ValueError(f"reference must be 'relative' or 'inertial', got {reference!r}")
    if gimbal_params is None:
        gimbal_params = default_gimbal_params()

    n = int(t_end / dt)
    state = np.zeros(4)  # [pan, tilt_relative, pan_dot, tilt_dot] -- pan unused (tau_pan=0 throughout)
    pid = PID(*pid_gains, setpoint=0.0, dt=dt, out_min=-3.0, out_max=3.0)

    t_hist = np.empty(n + 1)
    inertial_hist = np.empty(n + 1)
    relative_hist = np.empty(n + 1)
    base_hist = np.empty(n + 1)
    tau_hist = np.empty(n + 1)

    for i in range(n + 1):
        t = i * dt
        base = base_disturbance_tilt(t, disturbance_amplitude_deg, disturbance_freq_hz)
        inertial = base + state[1]
        measurement = inertial if reference == "inertial" else state[1]
        tau_tilt = pid.update(measurement)

        t_hist[i] = t
        inertial_hist[i] = inertial
        relative_hist[i] = state[1]
        base_hist[i] = base
        tau_hist[i] = tau_tilt

        if i < n:
            _, phi_ddot = forward_dynamics(state, 0.0, tau_tilt, gimbal_params)
            state[3] += phi_ddot * dt
            state[1] += state[3] * dt

    rms_inertial_error = float(np.sqrt(np.mean(inertial_hist ** 2)))
    return {
        "t": t_hist, "inertial_tilt": inertial_hist, "gimbal_relative_tilt": relative_hist,
        "base_tilt": base_hist, "torque": tau_hist, "rms_inertial_error_rad": rms_inertial_error,
    }


def verify_stabilization_reduces_error(disturbance_amplitude_deg=1.5,
                                        disturbance_freq_hz=BLADE_PASS_HZ,
                                        pid_gains=(150.0, 20.0, 5.0)):
    """The actual claim this module makes: measuring the INERTIAL angle
    (gyro feedback) instead of the RELATIVE angle -- same controller, same
    gains, same plant -- reduces the RMS inertial line-of-sight error.
    Runs both from identical conditions and checks the inertial-reference
    case is meaningfully (>5x) smaller, not just numerically different."""
    relative_ref = simulate_stabilization(
        disturbance_amplitude_deg=disturbance_amplitude_deg,
        disturbance_freq_hz=disturbance_freq_hz, pid_gains=pid_gains, reference="relative")
    inertial_ref = simulate_stabilization(
        disturbance_amplitude_deg=disturbance_amplitude_deg,
        disturbance_freq_hz=disturbance_freq_hz, pid_gains=pid_gains, reference="inertial")
    ratio = relative_ref["rms_inertial_error_rad"] / max(inertial_ref["rms_inertial_error_rad"], 1e-15)
    return {
        "relative_reference_rms_rad": relative_ref["rms_inertial_error_rad"],
        "inertial_reference_rms_rad": inertial_ref["rms_inertial_error_rad"],
        "improvement_ratio": ratio,
        "meaningfully_improved": bool(ratio > 5.0),
    }


if __name__ == "__main__":
    print(f"Disturbance model: main rotor {MAIN_ROTOR_HZ:.1f} Hz, "
          f"blade-pass {BLADE_PASS_HZ:.1f} Hz ({N_BLADES}-bladed rotor)\n")

    check = verify_stabilization_reduces_error()
    print("=== Relative-hold (unstabilized) vs. inertial-hold (stabilized) ===")
    print(f"relative-reference RMS error: {np.degrees(check['relative_reference_rms_rad']):.3f} deg")
    print(f"inertial-reference RMS error: {np.degrees(check['inertial_reference_rms_rad']):.4f} deg")
    print(f"improvement: {check['improvement_ratio']:.1f}x  "
          f"(meaningfully improved: {check['meaningfully_improved']})")

    # a representative illustrative threshold: a modest PTZ zoom lens with a
    # ~6 deg field of view needs angular jitter well under ~0.1 deg to avoid
    # visible frame-to-frame wander at the pixel level -- NOT a spec for any
    # specific lens/sensor, just a sanity scale to compare against
    threshold_deg = 0.1
    stabilized_deg = np.degrees(check["inertial_reference_rms_rad"])
    print(f"\nillustrative usability threshold: {threshold_deg} deg "
          f"-> inertial-reference case {'PASSES' if stabilized_deg < threshold_deg else 'still exceeds it'}")
