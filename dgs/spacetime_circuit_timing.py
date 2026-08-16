"""spacetime_circuit_timing.py -- signal propagation is bounded by the
speed of light (more precisely, by v=c/sqrt(eps_r_eff) in a real
dielectric), and at high clock frequencies / long traces this WIRE delay
becomes a real, dominant timing constraint alongside gate delay --
extending dgs/logic_timing.py's Circuit critical-path model (which only
counts gate delay) with real electromagnetic propagation physics.

GROUNDING FACTS, VERIFIED NUMERICALLY (not asserted):
  - Rear Admiral Grace Hopper's famous teaching prop: an 11.8-inch wire is
    exactly how far light travels in vacuum in 1 nanosecond
    (c * 1ns = 0.2998 m = 11.80 in) -- verified in
    verify_nanosecond_wire_length below.
  - The well-known PCB rule of thumb "~170 ps/inch on FR4" -- verified
    from propagation_delay_per_length(eps_r=4.3) below, giving ~176 ps/in,
    matching the commonly cited figure (which varies ~150-180 ps/in
    across sources depending on the exact effective dielectric constant
    assumed for a specific trace geometry -- not a single universal
    constant).

WHY THIS MATTERS FOR REAL CHIP/BOARD DESIGN: gate delay (what
dgs/logic_timing.py already models) shrinks with each process node, but
wire propagation delay does NOT shrink the same way -- a signal still has
to physically cross a real distance at a bounded speed. At high enough
clock frequency or long enough interconnect, wire delay can dominate the
critical path even when every individual gate is fast -- literally a
speed-of-light-in-a-dielectric constraint on how fast a real circuit's
signals can cross real distance, not a gate-technology limit.
"""

from __future__ import annotations
import numpy as np
from typing import Dict

from dgs.logic_timing import ripple_carry_delay

C_LIGHT = 2.99792458e8  # m/s, exact (SI defined)
INCH_M = 0.0254


# ── 1. Propagation velocity/delay in a real dielectric ───────────────────────

def propagation_velocity(eps_r_eff: float) -> float:
    """v = c/sqrt(eps_r_eff) -- the phase velocity of a signal in a
    dielectric medium (a PCB substrate, an on-chip interconnect
    dielectric), the SAME sqrt(eps_r) slowdown as any electromagnetic wave
    in a linear dielectric (Maxwell's equations, c/n with n=sqrt(eps_r)
    for a non-magnetic material)."""
    if eps_r_eff < 1.0:
        raise ValueError(f"eps_r_eff={eps_r_eff}: a physical dielectric has eps_r >= 1")
    return C_LIGHT / np.sqrt(eps_r_eff)


def propagation_delay_per_length(eps_r_eff: float) -> float:
    """Inverse of propagation_velocity: seconds of delay per meter of
    trace/interconnect length."""
    return 1.0 / propagation_velocity(eps_r_eff)


def verify_nanosecond_wire_length(tol_in: float = 0.01) -> Dict:
    """Verify Grace Hopper's famous 'nanosecond' teaching wire: how far
    light travels in vacuum (eps_r=1) in exactly 1 ns, in inches, compared
    to the historically cited 11.8 inches."""
    length_m = C_LIGHT * 1e-9
    length_in = length_m / INCH_M
    famous_value_in = 11.8
    return {"computed_inches": length_in, "famous_value_inches": famous_value_in,
            "matches": bool(abs(length_in - famous_value_in) < tol_in)}


# ── 2. Lumped vs. distributed: when a trace must be treated as a transmission line

def electrically_long_threshold_m(rise_time_s: float, eps_r_eff: float,
                                   fraction: float = 0.5) -> float:
    """A trace is 'electrically long' (must be treated as a transmission
    line, not a lumped wire) once its one-way propagation delay exceeds
    `fraction` of the signal's rise time -- a standard high-speed-digital-
    design rule of thumb. The exact fraction VARIES across textbooks/
    sources (common choices range roughly 1/6 to 1/2 of the rise time);
    `fraction` defaults to the more conservative 1/2 and should be set
    explicitly if a specific design guideline's convention is being
    followed, not assumed universal.
    """
    if rise_time_s <= 0:
        raise ValueError(f"rise_time_s={rise_time_s}: must be positive")
    if not (0 < fraction <= 1):
        raise ValueError(f"fraction={fraction}: must be in (0,1]")
    v = propagation_velocity(eps_r_eff)
    return fraction * rise_time_s * v


# ── 3. Extending dgs/logic_timing.py's ripple-adder critical path with wire delay

def ripple_carry_total_delay_with_wire(n_bits: int, gate_delay_ps: float,
                                        trace_length_per_stage_m: float,
                                        eps_r_eff: float = 4.3) -> Dict:
    """Total critical-path delay of an n-bit ripple-carry adder, adding
    REAL wire propagation delay (one trace length per carry stage, the
    physical distance the carry signal must ripple across on a real
    board/die) on top of dgs.logic_timing.ripple_carry_delay's existing
    gate-only delay model (reused directly, not reimplemented).

    Returns both delays separately plus the resulting fmax with and
    without wire delay, so the wire-delay CONTRIBUTION is explicit.
    """
    if n_bits < 1:
        raise ValueError(f"n_bits={n_bits}: must be >= 1")
    if gate_delay_ps <= 0:
        raise ValueError(f"gate_delay_ps={gate_delay_ps}: must be positive")
    if trace_length_per_stage_m < 0:
        raise ValueError("trace_length_per_stage_m must be non-negative")

    gate_only_ps = ripple_carry_delay(n_bits, gate_delay=gate_delay_ps)
    wire_delay_ps = n_bits * trace_length_per_stage_m * propagation_delay_per_length(eps_r_eff) * 1e12
    total_ps = gate_only_ps + wire_delay_ps

    return {"gate_only_delay_ps": gate_only_ps, "wire_delay_ps": wire_delay_ps,
            "total_delay_ps": total_ps,
            "fmax_gate_only_GHz": 1000.0 / gate_only_ps,
            "fmax_with_wire_GHz": 1000.0 / total_ps,
            "wire_delay_fraction": wire_delay_ps / total_ps}


def wire_delay_dominance_sweep(gate_delay_ps: float, trace_length_per_stage_m: float,
                                eps_r_eff: float = 4.3, n_bits_range=None) -> Dict:
    """Sweep adder width n_bits and return, for each width, the wire-delay
    fraction of the total critical path -- showing that wire delay's
    SHARE grows with n_bits (more carry stages -> more accumulated trace
    length), even though gate_delay_ps and trace_length_per_stage_m are
    both held fixed."""
    n_bits_range = n_bits_range if n_bits_range is not None else np.arange(4, 129, 4)
    fractions = []
    for n in n_bits_range:
        result = ripple_carry_total_delay_with_wire(
            int(n), gate_delay_ps, trace_length_per_stage_m, eps_r_eff)
        fractions.append(result["wire_delay_fraction"])
    return {"n_bits": np.asarray(n_bits_range), "wire_delay_fraction": np.asarray(fractions)}


if __name__ == "__main__":
    print("=== 1. Grace Hopper's nanosecond wire, verified ===")
    ns_check = verify_nanosecond_wire_length()
    print(f"  Light travels {ns_check['computed_inches']:.2f} in vacuum in 1 ns "
          f"(famous value: {ns_check['famous_value_inches']} in)  match: {ns_check['matches']}")

    print("\n=== 2. FR4 PCB propagation delay, verified against the ~170 ps/in rule of thumb ===")
    t_pd_fr4 = propagation_delay_per_length(eps_r_eff=4.3)
    print(f"  {t_pd_fr4 * INCH_M * 1e12:.1f} ps/inch on FR4 (eps_r=4.3)")

    print("\n=== 3. Electrically-long threshold ===")
    rise_time = 100e-12  # 100 ps rise time -- representative of a fast modern digital edge
    L_crit = electrically_long_threshold_m(rise_time, eps_r_eff=4.3)
    print(f"  At a {rise_time*1e12:.0f} ps rise time on FR4, traces longer than "
          f"{L_crit*1000:.1f} mm must be treated as transmission lines.")

    print("\n=== 4. Wire delay in a 64-bit ripple-carry adder ===")
    result = ripple_carry_total_delay_with_wire(
        n_bits=64, gate_delay_ps=20.0, trace_length_per_stage_m=0.002, eps_r_eff=4.3)
    print(f"  gate-only delay:  {result['gate_only_delay_ps']:.1f} ps  "
          f"(fmax={result['fmax_gate_only_GHz']:.3f} GHz)")
    print(f"  + wire delay:     {result['wire_delay_ps']:.1f} ps")
    print(f"  = total delay:    {result['total_delay_ps']:.1f} ps  "
          f"(fmax={result['fmax_with_wire_GHz']:.3f} GHz)")
    print(f"  wire delay is {result['wire_delay_fraction']*100:.1f}% of the total critical path")
