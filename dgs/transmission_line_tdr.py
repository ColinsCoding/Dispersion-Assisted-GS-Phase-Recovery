"""Classical electrodynamics, applied to computer engineering, as an actual
experimental technique: Time-Domain Reflectometry (TDR) -- send a step
pulse down a transmission line, measure the reflected pulse, and INFER an
unknown load's impedance from it. This is exactly how real cable/PCB fault
locators work, and it is also a genuine "modern physics experiment"
technique (measuring propagation velocity and dielectric properties of a
line by timing reflections, the same principle as radar/lidar ranging).

Two classical-E&M results underlie the whole thing:
  1. SKIN EFFECT: high-frequency AC resistance R(f) ~ sqrt(f), because the
     skin depth delta(f) ~ 1/sqrt(f) shrinks the effective conducting
     area. Differentiating R(f) is a genuine logarithmic-differentiation
     exercise: ln(R) = 0.5*ln(f) + const, so dR/df = 0.5*R/f -- computed
     WITHOUT ever expanding the square root, verified against direct
     differentiation.
  2. TRANSMISSION LINE THEORY (the telegrapher's equations): a line's
     characteristic impedance Z0=sqrt(L'/C') and propagation velocity
     v=1/sqrt(L'C') determine how a pulse reflects off any impedance
     mismatch, Gamma=(Z_L-Z0)/(Z_L+Z0) -- the reflection coefficient a
     TDR instrument actually measures.
"""

import numpy as np

MU0 = 4 * np.pi * 1e-7   # vacuum permeability, H/m


def skin_depth(f, rho, mu_r=1.0):
    """Skin depth delta = sqrt(2*rho/(omega*mu)), omega=2*pi*f -- the
    depth over which current density falls to 1/e of its surface value."""
    if f <= 0 or rho <= 0 or mu_r <= 0:
        raise ValueError("f, rho, mu_r must be positive")
    omega = 2 * np.pi * f
    mu = mu_r * MU0
    return np.sqrt(2 * rho / (omega * mu))


def ac_resistance_per_length(f, rho, mu_r, wire_radius):
    """High-frequency AC resistance per unit length, thin-shell
    approximation (valid once skin_depth << wire_radius): current only
    flows in an annulus of thickness delta near the surface, so
    R'(f) = rho / (2*pi*wire_radius*delta(f)) -- and since
    delta ~ f^(-1/2), R'(f) ~ f^(+1/2)."""
    if wire_radius <= 0:
        raise ValueError("wire_radius must be positive")
    delta = skin_depth(f, rho, mu_r)
    return rho / (2 * np.pi * wire_radius * delta)


def dR_df_via_log_differentiation(f, rho, mu_r, wire_radius):
    """dR/df, computed by LOGARITHMIC DIFFERENTIATION rather than
    expanding the square root: since R(f) = A*f^(1/2) for some constant A
    (all the f-independent physics folded into A), ln(R) = ln(A) +
    0.5*ln(f), so d(ln R)/df = 0.5/f, i.e. dR/df = 0.5*R/f -- this
    identity holds for ANY power-law R(f)=A*f^n: d(ln R)/df = n/f."""
    if f <= 0:
        raise ValueError("f must be positive")
    R = ac_resistance_per_length(f, rho, mu_r, wire_radius)
    return 0.5 * R / f


def dR_df_direct(f, rho, mu_r, wire_radius, h=None):
    """The SAME derivative, computed by ordinary central-difference
    numerical differentiation (no log-rule shortcut at all) -- an
    independent cross-check of the log-differentiation result."""
    if h is None:
        h = f * 1e-6
    R_plus = ac_resistance_per_length(f + h, rho, mu_r, wire_radius)
    R_minus = ac_resistance_per_length(f - h, rho, mu_r, wire_radius)
    return (R_plus - R_minus) / (2 * h)


def characteristic_impedance(L_per_len, C_per_len):
    """Z0 = sqrt(L'/C'), the telegrapher's-equations impedance a matched
    load must present to avoid any reflection at all."""
    if L_per_len <= 0 or C_per_len <= 0:
        raise ValueError("L_per_len and C_per_len must be positive")
    return np.sqrt(L_per_len / C_per_len)


def propagation_velocity(L_per_len, C_per_len):
    """v = 1/sqrt(L'C'), the speed a pulse actually travels down the line
    (the SAME 1/sqrt(LC) structure as an LC-tank's resonant frequency
    formula, appearing here as a propagation speed instead)."""
    if L_per_len <= 0 or C_per_len <= 0:
        raise ValueError("L_per_len and C_per_len must be positive")
    return 1.0 / np.sqrt(L_per_len * C_per_len)


def reflection_coefficient(Z0, Z_load):
    """Gamma = (Z_load - Z0) / (Z_load + Z0): the fraction of an incident
    pulse's amplitude reflected at an impedance discontinuity. Gamma=0
    (matched, no reflection), Gamma=+1 (open circuit), Gamma=-1 (short)."""
    if Z0 <= 0:
        raise ValueError("Z0 must be positive")
    if Z_load + Z0 == 0:
        raise ValueError("Z_load = -Z0 makes the reflection coefficient singular")
    return (Z_load - Z0) / (Z_load + Z0)


def load_impedance_from_reflection(gamma, Z0):
    """The TDR MEASUREMENT step, inverted: given an observed reflection
    coefficient, solve for the unknown load impedance,
    Z_load = Z0*(1+gamma)/(1-gamma) -- the exact algebraic inverse of
    reflection_coefficient, verified as a round trip below."""
    if Z0 <= 0:
        raise ValueError("Z0 must be positive")
    if abs(gamma - 1.0) < 1e-15:
        raise ValueError("gamma=1 (open circuit) makes Z_load infinite")
    return Z0 * (1 + gamma) / (1 - gamma)


def tdr_step_response(Z0, Z_load, Z_source, line_length, velocity, t, pulse_amplitude=1.0):
    """Simulate a real TDR trace at the SOURCE end of the line: a step
    pulse launched at t=0 (amplitude set by the source/Z0 voltage divider)
    travels to the load, reflects with coefficient Gamma, and returns at
    t = 2*line_length/velocity -- the classic two-level TDR step waveform
    used to locate faults and identify impedance mismatches."""
    if line_length <= 0 or velocity <= 0:
        raise ValueError("line_length and velocity must be positive")
    t = np.asarray(t, dtype=float)
    # voltage actually launched onto the line (source/Z0 divider)
    v_incident = pulse_amplitude * Z0 / (Z_source + Z0)
    gamma = reflection_coefficient(Z0, Z_load)
    t_round_trip = 2 * line_length / velocity
    v = np.where(t < 0, 0.0, v_incident)
    v = np.where(t >= t_round_trip, v_incident * (1 + gamma), v)
    return v, gamma, t_round_trip


if __name__ == "__main__":
    print("=== Skin effect: dR/df via logarithmic differentiation ===")
    rho, mu_r, a = 1.68e-8, 1.0, 0.5e-3   # copper wire, 0.5mm radius
    f_test = 1e8   # 100 MHz
    R_at_f = ac_resistance_per_length(f_test, rho, mu_r, a)
    dR_logdiff = dR_df_via_log_differentiation(f_test, rho, mu_r, a)
    dR_numeric = dR_df_direct(f_test, rho, mu_r, a)
    print(f"R'(f={f_test:.0e} Hz) = {R_at_f:.4f} Ohm/m")
    print(f"dR/df via log-differentiation (0.5*R/f): {dR_logdiff:.6e} Ohm/m/Hz")
    print(f"dR/df via direct numerical derivative:    {dR_numeric:.6e} Ohm/m/Hz")
    print(f"relative difference: {abs(dR_logdiff-dR_numeric)/dR_numeric:.2e}")

    print("\n=== Transmission line theory: a real PCB trace ===")
    L_per_len, C_per_len = 250e-9, 100e-12   # typical PCB microstrip, H/m and F/m
    Z0 = characteristic_impedance(L_per_len, C_per_len)
    v = propagation_velocity(L_per_len, C_per_len)
    print(f"Z0 = {Z0:.1f} Ohm (typical PCB target is ~50 Ohm)")
    print(f"propagation velocity = {v:.3e} m/s ({v/3e8:.2f}c)")

    print("\n=== TDR experiment: measuring an unknown load ===")
    Z_load_true = 75.0   # the "unknown" impedance being measured
    Z_source = 50.0
    line_length = 0.3   # meters
    t = np.linspace(-1e-9, 6e-9, 500)
    v_trace, gamma, t_rt = tdr_step_response(Z0, Z_load_true, Z_source, line_length, v, t)
    print(f"true load impedance: {Z_load_true} Ohm, reflection coefficient Gamma={gamma:.4f}")
    print(f"round-trip time (fault/load location signature): {t_rt*1e9:.3f} ns")

    # the actual "measurement": infer Z_load from the observed Gamma, blind to the true value
    Z_load_inferred = load_impedance_from_reflection(gamma, Z0)
    print(f"inferred load impedance from Gamma alone: {Z_load_inferred:.4f} Ohm "
          f"(true: {Z_load_true} Ohm, error: {abs(Z_load_inferred-Z_load_true):.2e})")

    inferred_length = t_rt * v / 2
    print(f"inferred distance to fault/load from timing: {inferred_length:.4f} m "
          f"(true: {line_length} m)")
