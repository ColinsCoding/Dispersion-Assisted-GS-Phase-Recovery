"""welding_circuits.py -- arc-welding power-supply circuits, algebra through calculus.

A transformer-based arc welder is a real, teachable RL circuit: step-down
transformer secondary (resistance R, leakage inductance L) driving an arc
that behaves, to first order, like a load resistance. Three levels, same
circuit, matching this repo's usual algebra -> calculus -> engineering-number
pipeline:

  ALGEBRA   (elementary_algebra.solve_for_missing_factor): Ohm's law V=IR
            and power P=IV for the arc itself.
  CALCULUS  (high-school level, shown symbolically via SymPy): when the
            electrode strikes the workpiece, current doesn't jump
            instantly to its steady value -- it rises along the classic
            first-order RL transient i(t) = (V/R)(1-e^{-tR/L}), the
            solution of L*di/dt + R*i = V. answers the concrete question
            "does the arc current settle out in under 1 second?"
  CIRCUIT ENGINEERING (dgs/ac_circuits.py's impedance_L/series_impedance,
            reused directly, not reimplemented): the transformer secondary's
            AC impedance at line frequency, and a practical heat-input
            formula welders actually use to judge weld penetration.

Typical real numbers used in the demo below (stick/SMAW welding): open-
circuit voltage ~60-80 V, arc voltage ~20-25 V once striking, welding
current ~100-200 A -- representative textbook figures, not a specific
manufacturer's datasheet.
"""
from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Dict

from dgs.elementary_algebra import solve_for_missing_factor
from dgs.ac_circuits import impedance_L, series_impedance, average_power


# ── 1. Algebra: Ohm's law and power for the arc ─────────────────────────────

def arc_resistance(V_arc: float, I_weld: float) -> float:
    """Ohm's law R=V/I for the arc itself, reusing elementary_algebra's
    solve_for_missing_factor (ab=c -> b=c/a) rather than reimplementing
    division -- here a=I, c=V, solving for b=R."""
    if V_arc <= 0 or I_weld <= 0:
        raise ValueError("V_arc and I_weld must be positive")
    return solve_for_missing_factor(I_weld, V_arc)


def arc_power(V_arc: float, I_weld: float) -> float:
    """P=VI: electrical power delivered into the arc (heat + light + metal
    transfer), the algebra-level quantity everything else in this module
    builds on."""
    if V_arc <= 0 or I_weld <= 0:
        raise ValueError("V_arc and I_weld must be positive")
    return V_arc * I_weld


# ── 2. Calculus: the RL current-rise transient at arc strike ───────────────

def rl_transient_symbolic():
    """Derive i(t) = (V/R)(1-e^{-tR/L}) from L*di/dt + R*i = V via SymPy
    dsolve -- the high-school-calculus content this module is built around,
    shown symbolically rather than just stated."""
    t, R, L, V = sp.symbols('t R L V', positive=True)
    i = sp.Function('i')
    ode = sp.Eq(L * sp.Derivative(i(t), t) + R * i(t), V)
    solution = sp.dsolve(ode, i(t), ics={i(0): 0})
    return ode, solution


def rl_transient_current(t, V: float, R: float, L: float) -> np.ndarray:
    """Numeric i(t) = (V/R)(1-e^{-tR/L}) -- the RL current-rise transient at
    arc strike. t may be a scalar or array (seconds)."""
    if V <= 0 or R <= 0 or L <= 0:
        raise ValueError("V, R, and L must all be positive")
    t = np.asarray(t, dtype=float)
    if np.any(t < 0):
        raise ValueError("t must be non-negative")
    tau = L / R
    return (V / R) * (1.0 - np.exp(-t / tau))


def time_to_steady_state(R: float, L: float, tolerance: float = 0.02) -> float:
    """Time for the RL transient to settle within `tolerance` (fraction, e.g.
    0.02 = within 2%) of its final steady-state current: t = -tau*ln(tolerance),
    tau=L/R. Answers "does the arc current settle out in under 1 second?" for
    a given transformer's R, L directly."""
    if R <= 0 or L <= 0:
        raise ValueError("R and L must be positive")
    if not (0.0 < tolerance < 1.0):
        raise ValueError(f"tolerance={tolerance}: must be in (0, 1)")
    tau = L / R
    return -tau * np.log(tolerance)


# ── 3. Circuit engineering: transformer secondary impedance ────────────────

def transformer_secondary_impedance(R_secondary: float, L_leakage: float, omega: float) -> complex:
    """AC impedance of the transformer secondary at angular frequency omega,
    reusing dgs/ac_circuits.py's impedance_L and series_impedance directly
    (not reimplemented) -- Z = R + jωL."""
    if R_secondary <= 0 or L_leakage <= 0 or omega <= 0:
        raise ValueError("R_secondary, L_leakage, and omega must be positive")
    Z_L = impedance_L(L_leakage, omega)
    return series_impedance(R_secondary, Z_L)


def open_circuit_to_arc_voltage_drop(V_open_circuit: float, I_weld: float,
                                      R_secondary: float) -> float:
    """The secondary's own resistance drops some of the open-circuit voltage
    before the arc ever sees it: V_arc = V_open_circuit - I*R_secondary.
    A real welder's arc voltage is always less than its rated open-circuit
    voltage for exactly this reason."""
    if V_open_circuit <= 0 or I_weld <= 0 or R_secondary < 0:
        raise ValueError("V_open_circuit and I_weld must be positive; R_secondary must be non-negative")
    V_arc = V_open_circuit - I_weld * R_secondary
    if V_arc <= 0:
        raise ValueError("computed V_arc <= 0: R_secondary too large for this I_weld/V_open_circuit pair")
    return V_arc


# ── 4. A practical number: heat input (weld-shop formula, not derived here) ─

def heat_input(V_arc: float, I_weld: float, travel_speed_mm_s: float,
                efficiency: float = 0.8) -> float:
    """Heat input HI = eta*V*I/travel_speed (J/mm), the formula welding
    procedure specifications actually use to bound weld penetration/HAZ
    size -- a standard empirical process formula (not derived from the RL
    circuit above), efficiency ~0.8 typical for SMAW/GMAW arcs."""
    if V_arc <= 0 or I_weld <= 0 or travel_speed_mm_s <= 0:
        raise ValueError("V_arc, I_weld, and travel_speed_mm_s must be positive")
    if not (0.0 < efficiency <= 1.0):
        raise ValueError(f"efficiency={efficiency}: must be in (0, 1]")
    return efficiency * V_arc * I_weld / travel_speed_mm_s


if __name__ == "__main__":
    print("=== 1. Algebra: Ohm's law and power for the arc ===")
    V_arc, I_weld = 22.0, 150.0   # typical SMAW stick-welding numbers
    R_arc = arc_resistance(V_arc, I_weld)
    P_arc = arc_power(V_arc, I_weld)
    print(f"  V_arc={V_arc} V, I_weld={I_weld} A  ->  R_arc={R_arc:.4f} ohm, P_arc={P_arc:.0f} W")

    print("\n=== 2. Calculus: RL transient at arc strike ===")
    ode, solution = rl_transient_symbolic()
    print(f"  ODE: {ode}")
    print(f"  solution: {solution}")

    R_sec, L_leak = 0.05, 0.0008   # ohm, henry -- representative transformer secondary
    t_settle = time_to_steady_state(R_sec, L_leak, tolerance=0.02)
    print(f"  R_secondary={R_sec} ohm, L_leakage={L_leak} H  ->  "
          f"settles to within 2% in {t_settle*1000:.2f} ms "
          f"({'UNDER 1 second' if t_settle < 1.0 else 'over 1 second'})")

    print("\n=== 3. Circuit engineering: transformer secondary impedance ===")
    omega_60hz = 2 * np.pi * 60.0
    Z_sec = transformer_secondary_impedance(R_sec, L_leak, omega_60hz)
    print(f"  Z_secondary(60 Hz) = {Z_sec:.4f} ohm  |Z|={abs(Z_sec):.4f} ohm")

    V_open_circuit = 70.0
    V_arc_actual = open_circuit_to_arc_voltage_drop(V_open_circuit, I_weld, R_sec)
    print(f"  V_open_circuit={V_open_circuit} V, I_weld={I_weld} A, R_secondary={R_sec} ohm  ->  "
          f"V_arc={V_arc_actual:.2f} V (rated {V_open_circuit} V never actually reaches the arc)")

    print("\n=== 4. Practical number: heat input ===")
    travel_speed = 3.0   # mm/s, typical manual SMAW travel speed
    HI = heat_input(V_arc, I_weld, travel_speed)
    print(f"  travel_speed={travel_speed} mm/s  ->  heat input = {HI:.1f} J/mm")
