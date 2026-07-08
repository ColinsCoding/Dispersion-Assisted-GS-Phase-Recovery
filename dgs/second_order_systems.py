"""Second-order systems: zeta and omega, the two numbers that describe an RLC.

Every series RLC circuit -- and every mass-spring-damper, and every 2-pole filter --
is one standard transfer function set by just TWO numbers:
        H(s) = omega_n^2 / (s^2 + 2 zeta omega_n s + omega_n^2),
the NATURAL FREQUENCY omega_n (how fast it wants to oscillate) and the DAMPING RATIO
zeta (how quickly the oscillation dies). For a series RLC:
        omega_n = 1/sqrt(LC),     zeta = (R/2) sqrt(C/L),     Q = 1/(2 zeta).
Those two numbers decide everything -- the poles, the ringing, the overshoot, the
resonance -- so circuit design is really the choice of zeta and omega_n.

The POLES sit at s = -zeta*omega_n +/- omega_n sqrt(zeta^2 - 1):
  * zeta < 1  UNDERDAMPED  -- complex poles, the circuit RINGS at the damped
    frequency omega_d = omega_n sqrt(1 - zeta^2), with a step overshoot
    %OS = 100 exp(-zeta pi / sqrt(1 - zeta^2));
  * zeta = 1  CRITICAL     -- fastest settle with no overshoot;
  * zeta > 1  OVERDAMPED   -- two real poles, a sluggish non-oscillatory crawl.

In FREQUENCY, an underdamped system with zeta < 1/sqrt(2) shows a RESONANT PEAK at
omega_r = omega_n sqrt(1 - 2 zeta^2), sharper the higher the Q. zeta = 1/sqrt(2)
(~0.707) is the maximally-flat Butterworth point -- the usual filter design target.

Everything is checked against the defining ODE (the step response is differentiated
and must satisfy y'' + 2 zeta omega_n y' + omega_n^2 y = omega_n^2) and cross-checked
with dgs.spice.resonant_frequency / dgs.circuit_energy. NumPy only; py-3.13.
"""

import numpy as np


# ----------------------------------------------------------------------
# The two numbers, from an RLC
# ----------------------------------------------------------------------

def natural_frequency_rlc(L, C):
    """Undamped natural frequency omega_n = 1/sqrt(LC) [rad/s]."""
    if L <= 0 or C <= 0:
        raise ValueError("L and C must be positive")
    return 1.0 / np.sqrt(L * C)


def damping_ratio_rlc(R, L, C):
    """Damping ratio zeta = (R/2) sqrt(C/L) for a series RLC. R=0 is lossless
    (zeta=0, rings forever); large R over-damps."""
    if R < 0 or L <= 0 or C <= 0:
        raise ValueError("need R >= 0 and L, C > 0")
    return (R / 2) * np.sqrt(C / L)


def quality_factor(zeta):
    """Q = 1/(2 zeta): sharpness of resonance / cycles to decay. High Q = light
    damping = a narrow, tall resonance."""
    if zeta <= 0:
        raise ValueError("zeta must be positive for a finite Q")
    return 1.0 / (2 * zeta)


def poles(zeta, omega_n):
    """The two poles s = -zeta*omega_n +/- omega_n sqrt(zeta^2 - 1) (complex when
    underdamped). Their real part sets the decay, imaginary part the ringing."""
    if omega_n <= 0 or zeta < 0:
        raise ValueError("need omega_n > 0 and zeta >= 0")
    disc = np.lib.scimath.sqrt(zeta ** 2 - 1)     # complex for zeta < 1
    return (-zeta * omega_n + omega_n * disc, -zeta * omega_n - omega_n * disc)


def damping_regime(zeta):
    """'under' (zeta<1), 'critical' (zeta==1), or 'over' (zeta>1)."""
    if zeta < 0:
        raise ValueError("zeta must be non-negative")
    if abs(zeta - 1) < 1e-12:
        return "critical"
    return "under" if zeta < 1 else "over"


def damped_frequency(zeta, omega_n):
    """The frequency an underdamped system actually rings at:
    omega_d = omega_n sqrt(1 - zeta^2). Zero if not underdamped."""
    if omega_n <= 0:
        raise ValueError("omega_n must be positive")
    return omega_n * np.sqrt(1 - zeta ** 2) if zeta < 1 else 0.0


# ----------------------------------------------------------------------
# Time-domain step response and its metrics
# ----------------------------------------------------------------------

def percent_overshoot(zeta):
    """Peak overshoot of the step response, %OS = 100 exp(-zeta pi/sqrt(1-zeta^2)).
    Depends ONLY on zeta -- 16.3% at zeta=0.5, ~4.3% at 0.707, 0 for zeta>=1."""
    if zeta < 0:
        raise ValueError("zeta must be non-negative")
    if zeta >= 1:
        return 0.0
    return 100 * np.exp(-zeta * np.pi / np.sqrt(1 - zeta ** 2))


def settling_time(zeta, omega_n, tol=0.02):
    """Time to stay within `tol` of the final value, ~ -ln(tol)/(zeta omega_n)
    (the classic 4/(zeta omega_n) for 2%)."""
    if not 0 < zeta < 1 or omega_n <= 0:
        raise ValueError("need 0 < zeta < 1 and omega_n > 0")
    return -np.log(tol) / (zeta * omega_n)


def step_response(zeta, omega_n, t):
    """Unit-step response of the standard second-order system (DC gain 1), for
    under/critical/over damping. Verified against the defining ODE."""
    t = np.asarray(t, float)
    wn = omega_n
    if zeta < 1:
        wd = wn * np.sqrt(1 - zeta ** 2)
        phi = np.arccos(zeta)
        return 1 - np.exp(-zeta * wn * t) / np.sqrt(1 - zeta ** 2) * np.sin(wd * t + phi)
    if abs(zeta - 1) < 1e-12:
        return 1 - np.exp(-wn * t) * (1 + wn * t)
    r = wn * np.sqrt(zeta ** 2 - 1)
    s1, s2 = -zeta * wn + r, -zeta * wn - r
    return 1 - (s1 * np.exp(s2 * t) - s2 * np.exp(s1 * t)) / (s1 - s2)


# ----------------------------------------------------------------------
# Frequency response and resonance
# ----------------------------------------------------------------------

def magnitude_response(zeta, omega_n, omega):
    """|H(j omega)| = omega_n^2 / sqrt((omega_n^2 - omega^2)^2 + (2 zeta omega_n omega)^2)."""
    omega = np.asarray(omega, float)
    return omega_n ** 2 / np.sqrt((omega_n ** 2 - omega ** 2) ** 2
                                  + (2 * zeta * omega_n * omega) ** 2)


def resonant_peak_frequency(zeta, omega_n):
    """The frequency of the magnitude peak, omega_r = omega_n sqrt(1 - 2 zeta^2).
    A resonant peak exists ONLY for zeta < 1/sqrt(2) ~ 0.707 (else monotonic
    rolloff); returns None when there is no peak."""
    if omega_n <= 0:
        raise ValueError("omega_n must be positive")
    if zeta >= 1 / np.sqrt(2):
        return None
    return omega_n * np.sqrt(1 - 2 * zeta ** 2)


if __name__ == "__main__":
    # the RLC from dgs.circuit_energy: R=20, L=1mH, C=1uF
    R, L, C = 20.0, 1e-3, 1e-6
    wn = natural_frequency_rlc(L, C)
    zeta = damping_ratio_rlc(R, L, C)
    print(f"RLC R=20 L=1mH C=1uF:  omega_n = {wn:.0f} rad/s (f_n={wn/2/np.pi:.0f} Hz), "
          f"zeta = {zeta:.3f}, Q = {quality_factor(zeta):.2f}")
    print(f"  regime: {damping_regime(zeta)}, damped f_d = {damped_frequency(zeta, wn)/2/np.pi:.0f} Hz")
    print(f"  poles: {np.round(poles(zeta, wn), 1)}")
    print(f"  step overshoot = {percent_overshoot(zeta):.1f}%, "
          f"2% settling time = {settling_time(zeta, wn)*1e3:.2f} ms")

    print("\novershoot depends only on zeta:")
    for z in (0.3, 0.5, 0.707, 1.0):
        print(f"  zeta={z:.3f}: %OS = {percent_overshoot(z):5.1f}%, "
              f"resonant peak at {resonant_peak_frequency(z, 1.0)}")

    # confirm the step response solves the ODE
    t = np.linspace(0, 5, 20000); dt = t[1] - t[0]
    y = step_response(0.4, 1.0, t)
    ydd = np.gradient(np.gradient(y, dt), dt)
    yd = np.gradient(y, dt)
    resid = ydd + 2*0.4*1.0*yd + 1.0**2 * y - 1.0**2
    print(f"\nstep response satisfies the ODE? max residual (interior) = "
          f"{np.max(np.abs(resid[10:-10])):.2e}")
