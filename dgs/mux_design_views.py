"""One 2:1 multiplexer, three design views -- the Gajski-Kuhn Y-chart in code.

A digital block can be described at rising levels of abstraction, and the same
2:1 mux looks different in each "view":

  BEHAVIORAL view  -- WHAT it does, as a Boolean equation:
        Y = S'.A + S.B         (pick A when S=0, B when S=1)
     No structure, no gates -- just the function. Verified symbolically here
     with SymPy (the disjunctive normal form of the truth table).

  STRUCTURAL view  -- HOW it's built, as a netlist of gates:
        Y = OR( AND(NOT(S), A), AND(S, B) )
     Interconnected primitives (reuses dgs.digital_logic gates). Simulating the
     netlist over all 8 inputs must reproduce the behavioral truth table.
     Transistor cost from dgs.digital_logic.cmos_cost -- and the compact
     "solid-state" realization is the TRANSMISSION-GATE mux (pass transistors),
     just 2 TGs + 1 inverter = 6 transistors vs ~20 for AND-OR-INV.

  SIGNAL & TIMING view -- WHEN the answer arrives, as transient analysis:
     A real gate output does not switch instantly; each stage is a first-order
     RC, so the output is the input CONVOLVED with the RC impulse response
        h(t) = (1/tau) e^(-t/tau),   step response 1 - e^(-t/tau).
     The 50% crossing is the propagation delay t_pd = tau*ln2, and the critical
     path (select-inverter -> AND -> OR) sums stage delays. Cascading stages =
     convolving their impulse responses -- so timing IS convolution.

Behavioral -> structural -> physical/timing is exactly the arc an EE/CE student
walks over a multi-year sequence (Boolean algebra, then logic design, then the
transient analysis that says whether it meets clock). SymPy + NumPy; py-3.13.
"""

import numpy as np
import sympy as sp
from dgs import digital_logic as dl


# ----------------------------------------------------------------------
# BEHAVIORAL view: the Boolean equation Y = S'.A + S.B
# ----------------------------------------------------------------------

def mux2_behavioral(a, b, s):
    """The function itself: return A when S=0, B when S=1. This is the whole
    behavioral spec -- no gates, no timing."""
    for v in (a, b, s):
        if v not in (0, 1):
            raise ValueError("inputs must be 0/1")
    return b if s else a


def mux2_truth_table():
    """All 8 rows (A, B, S, Y) of the behavioral mux."""
    from itertools import product
    return [(a, b, s, mux2_behavioral(a, b, s))
            for a, b, s in product((0, 1), repeat=3)]


def mux2_boolean_equation():
    """The behavioral view as a symbolic equation. Builds Y = S'.A + S.B in
    SymPy, confirms its truth table matches mux2_behavioral, and returns the
    simplified disjunctive normal form (sum of products). Proof that the
    equation and the function are the same object."""
    A, B, S = sp.symbols("A B S")
    Y = (~S & A) | (S & B)
    # check against the behavioral function over every assignment
    ok = all(
        bool(Y.subs({A: bool(a), B: bool(b), S: bool(s)})) == bool(mux2_behavioral(a, b, s))
        for a, b, s in __import__("itertools").product((0, 1), repeat=3)
    )
    return {"expr": Y, "dnf": sp.to_dnf(Y, simplify=True), "matches_table": ok}


# ----------------------------------------------------------------------
# STRUCTURAL view: the gate netlist, and the transistor cost
# ----------------------------------------------------------------------

def mux2_structural(a, b, s):
    """The mux built from primitives: OR(AND(NOT(S), A), AND(S, B)). Uses the
    dgs.digital_logic gates so the structure is real, not a restatement of the
    behavioral function."""
    not_s = dl.NOT(s)
    return dl.OR(dl.AND(not_s, a), dl.AND(s, b))


def structural_matches_behavioral():
    """Simulate the gate netlist over all 8 inputs and confirm it reproduces
    the behavioral truth table -- the equivalence that lets synthesis replace
    an equation with gates."""
    from itertools import product
    return all(mux2_structural(a, b, s) == mux2_behavioral(a, b, s)
               for a, b, s in product((0, 1), repeat=3))


def mux2_transistor_cost():
    """Transistor counts for two realizations:
      * 'and_or_invert' : NOT + 2*AND2 + OR2 from static-CMOS primitives,
      * 'transmission_gate' : 2 transmission gates (4T) + 1 select inverter (2T)
        -- the compact 'solid-state' pass-transistor mux, 6 transistors.
    The transmission-gate version is why real 2:1 muxes are cheap."""
    aoi = dl.cmos_cost("NOT") + 2 * dl.cmos_cost("AND2") + dl.cmos_cost("OR2")
    tg = 2 * 2 + dl.cmos_cost("NOT")     # each transmission gate = 1 NMOS + 1 PMOS
    return {"and_or_invert": aoi, "transmission_gate": tg,
            "path_gates_aoi": 3}          # NOT -> AND -> OR on the critical path


# ----------------------------------------------------------------------
# SIGNAL & TIMING view: transient analysis = convolution with an RC
# ----------------------------------------------------------------------

def rc_impulse_response(t, tau):
    """First-order RC gate impulse response h(t) = (1/tau) e^(-t/tau) for
    t >= 0 (causal), zero before. Convolving an input with this is what turns
    an ideal logic edge into a real, finite-slope transition."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    t = np.asarray(t, float)
    return np.where(t >= 0, np.exp(-np.clip(t, 0, None) / tau) / tau, 0.0)


def rc_step_response(t, tau):
    """Analytic step response 1 - e^(-t/tau): the ideal 0->1 edge after the RC
    smears it. The reference the convolution must reproduce."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    t = np.asarray(t, float)
    return np.where(t >= 0, 1 - np.exp(-np.clip(t, 0, None) / tau), 0.0)


def gate_output_via_convolution(x, t, tau):
    """The timing view made literal: numerically CONVOLVE input x(t) with the
    RC impulse response to get the gate output. For a unit step this returns
    1 - e^(-t/tau), matching rc_step_response -- timing is convolution."""
    t = np.asarray(t, float)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("t must be a 1-D time grid")
    dt = t[1] - t[0]
    if not np.allclose(np.diff(t), dt):
        raise ValueError("t must be uniformly spaced")
    h = rc_impulse_response(t, tau)
    y = np.convolve(np.asarray(x, float), h)[:len(t)] * dt
    return y


def propagation_delay(tau):
    """50%-crossing propagation delay of one RC stage: t_pd = tau*ln2."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    return tau * np.log(2)


def critical_path_delay(stage_taus):
    """Total propagation delay along the mux critical path: the per-stage RC
    delays summed (select-inverter -> AND -> OR). A first-order estimate of
    when Y is guaranteed valid -- the number that must fit inside the clock."""
    stage_taus = list(stage_taus)
    if not stage_taus or any(tp <= 0 for tp in stage_taus):
        raise ValueError("need one or more positive stage taus")
    return float(sum(propagation_delay(tp) for tp in stage_taus))


def cascade_step_response(t, stage_taus):
    """Push a step through a CHAIN of RC stages by successively convolving
    each stage's impulse response -- the physical meaning of a multi-gate
    path. Returns the output waveform; its 50% delay exceeds any single
    stage's, illustrating that cascading = convolving."""
    t = np.asarray(t, float)
    dt = t[1] - t[0]
    y = rc_step_response(t, stage_taus[0])
    for tau in stage_taus[1:]:
        h = rc_impulse_response(t, tau)
        y = np.convolve(y, h)[:len(t)] * dt
    return y


if __name__ == "__main__":
    print("BEHAVIORAL: truth table (A,B,S,Y)")
    for row in mux2_truth_table():
        print("  ", row)
    eq = mux2_boolean_equation()
    print("  equation Y =", eq["dnf"], " matches table?", eq["matches_table"])

    print("\nSTRUCTURAL: netlist == behavioral?", structural_matches_behavioral())
    cost = mux2_transistor_cost()
    print(f"  transistors: AND-OR-INV = {cost['and_or_invert']}T, "
          f"transmission-gate = {cost['transmission_gate']}T (the solid-state mux)")

    print("\nTIMING: gate output = input convolved with RC impulse response")
    tau = 20e-12                                   # 20 ps stage
    t = np.linspace(0, 200e-12, 4000)
    step = np.ones_like(t)
    y = gate_output_via_convolution(step, t, tau)
    err = np.max(np.abs(y - rc_step_response(t, tau)))
    print(f"  convolution vs analytic step response: max error {err:.2e}")
    print(f"  one-stage t_pd = tau*ln2 = {propagation_delay(tau)*1e12:.1f} ps")
    taus = [tau, tau, tau]                          # INV -> AND -> OR
    print(f"  critical-path delay (3 stages) = {critical_path_delay(taus)*1e12:.1f} ps")
    yc = cascade_step_response(t, taus)
    t50 = t[np.argmax(yc >= 0.5)]
    print(f"  cascaded 50% delay = {t50*1e12:.1f} ps (> one stage: cascading = convolving)")
