"""AP-Physics DC circuits -- resistor networks, dividers, and Kirchhoff as
linear algebra.

The whole of AP-1/AP-2 DC circuits is Ohm's law V=IR plus two bookkeeping
rules (Kirchhoff): currents into a node sum to zero, voltages around a loop
sum to zero. Everything below is those three facts:

  * series/parallel reduction collapses a network to one number
  * voltage/current dividers are the two-resistor special cases
  * a real battery is an emf in series with internal resistance r, so its
    TERMINAL voltage sags under load: V = emf - I r
  * power dissipated: P = IV = I^2 R = V^2/R (one Joule-heating fact, three faces)

The finale, `nodal_solve`, is the grown-up version: node-voltage analysis is
just solving a linear system G v = i, where G is the conductance (Laplacian)
matrix of the resistor graph. That is the same "if you know linear algebra"
move as the parity operator in dgs.even_odd -- a physics problem becomes a
matrix equation. G is symmetric positive-definite, exactly the graph
Laplacian weighted by conductances 1/R.

NumPy only, py -3.13 safe. Education.
"""

import numpy as np


# ----------------------------------------------------------------------
# Series / parallel reduction
# ----------------------------------------------------------------------

def _check_resistors(Rs):
    Rs = np.asarray(Rs, float)
    if Rs.size == 0:
        raise ValueError("need at least one resistor")
    if np.any(Rs <= 0):
        raise ValueError("resistances must be positive (ohms)")
    return Rs


def series_resistance(Rs):
    """Resistors in series add: R_eq = sum R_k (always >= the largest)."""
    return float(np.sum(_check_resistors(Rs)))


def parallel_resistance(Rs):
    """Resistors in parallel add as conductances: 1/R_eq = sum 1/R_k
    (R_eq is always <= the smallest). Two resistors: R1 R2/(R1+R2)."""
    return float(1.0 / np.sum(1.0 / _check_resistors(Rs)))


def equivalent_resistance(spec):
    """Collapse a nested network to one resistance. `spec` is either a number
    (a leaf resistor) or a tuple ('series'|'parallel', [sub-specs...]):

        ('series', [100, ('parallel', [200, 200]), 50])  ->  100+100+50 = 250
    """
    if isinstance(spec, (int, float)):
        if spec <= 0:
            raise ValueError("resistances must be positive (ohms)")
        return float(spec)
    kind, subs = spec
    vals = [equivalent_resistance(s) for s in subs]
    if kind == "series":
        return series_resistance(vals)
    if kind == "parallel":
        return parallel_resistance(vals)
    raise ValueError(f"unknown combinator {kind!r} (use 'series' or 'parallel')")


# ----------------------------------------------------------------------
# Dividers and real batteries
# ----------------------------------------------------------------------

def voltage_divider(v_in, r_across, r_other):
    """Voltage across r_across in a two-resistor series string:
    V = v_in * r_across / (r_across + r_other). The output is always a
    FRACTION of the input -- a divider cannot amplify."""
    if r_across <= 0 or r_other <= 0:
        raise ValueError("resistances must be positive (ohms)")
    return v_in * r_across / (r_across + r_other)


def current_divider(i_in, r_this, r_other):
    """Current through r_this when two resistors share a node pair. Current
    prefers the LOW-resistance path, so it scales with the OTHER resistor:
    I = i_in * r_other / (r_this + r_other)."""
    if r_this <= 0 or r_other <= 0:
        raise ValueError("resistances must be positive (ohms)")
    return i_in * r_other / (r_this + r_other)


def terminal_voltage(emf, r_internal, r_load):
    """A real battery: emf in series with internal resistance r_internal,
    driving r_load. Returns (I, V_terminal). The terminal voltage sags below
    the emf under load -- V = emf - I r_internal -- which is why your car's
    headlights dim when the starter cranks."""
    if r_internal < 0 or r_load <= 0:
        raise ValueError("r_internal must be >= 0 and r_load > 0 (ohms)")
    current = emf / (r_internal + r_load)
    return current, emf - current * r_internal


def power(*, v=None, i=None, r=None):
    """Joule heating from any two of (v, i, r): P = IV = I^2 R = V^2/R."""
    if v is not None and i is not None:
        return v * i
    if i is not None and r is not None:
        if r < 0:
            raise ValueError("r must be >= 0")
        return i * i * r
    if v is not None and r is not None:
        if r <= 0:
            raise ValueError("r must be > 0 to compute V^2/R")
        return v * v / r
    raise ValueError("supply exactly two of v, i, r")


# ----------------------------------------------------------------------
# Kirchhoff as linear algebra: node-voltage analysis  G v = i
# ----------------------------------------------------------------------

def conductance_matrix(num_nodes, resistors):
    """Build the (num_nodes-1) x (num_nodes-1) conductance matrix G for node
    analysis, with node 0 as ground/reference. `resistors` is a list of
    (a, b, R) edges. G is the graph Laplacian weighted by 1/R with the
    ground row/column removed: diagonal = sum of conductances at the node,
    off-diagonal = -(shared conductance). Symmetric positive-definite."""
    if num_nodes < 2:
        raise ValueError("need at least 2 nodes (one is ground)")
    n = num_nodes - 1
    G = np.zeros((n, n))
    for a, b, R in resistors:
        if R <= 0:
            raise ValueError("resistances must be positive (ohms)")
        if not (0 <= a < num_nodes and 0 <= b < num_nodes) or a == b:
            raise ValueError(f"bad edge ({a},{b}) for {num_nodes} nodes")
        g = 1.0 / R
        if a > 0:
            G[a - 1, a - 1] += g
        if b > 0:
            G[b - 1, b - 1] += g
        if a > 0 and b > 0:
            G[a - 1, b - 1] -= g
            G[b - 1, a - 1] -= g
    return G


def nodal_solve(num_nodes, resistors, current_sources=None):
    """Solve a resistor network by node-voltage analysis: G v = i.
    Node 0 is ground (0 V). `current_sources` maps node -> amps injected INTO
    that node (a real battery emf/r in series becomes a Norton source
    I=emf/r into its node, with r included as a resistor to ground). Returns
    the full node-voltage array (length num_nodes, node 0 = 0).

    This is Kirchhoff's current law written once per node and stacked into a
    matrix -- the linear-algebra view of a circuit."""
    G = conductance_matrix(num_nodes, resistors)
    i = np.zeros(num_nodes - 1)
    if current_sources:
        for node, amps in current_sources.items():
            if node == 0:
                continue  # current into ground is the return path, not an unknown
            if not (0 < node < num_nodes):
                raise ValueError(f"source node {node} out of range")
            i[node - 1] += amps
    v = np.linalg.solve(G, i)
    return np.concatenate([[0.0], v])


def norton_from_battery(emf, r_internal):
    """Thevenin (emf, r) -> Norton (I_short, r): a battery with internal
    resistance r is equivalent to a current source emf/r in parallel with r.
    Returns (I_norton, r) so you can drop it into nodal_solve."""
    if r_internal <= 0:
        raise ValueError("r_internal must be > 0 for a Norton equivalent")
    return emf / r_internal, r_internal


if __name__ == "__main__":
    print("=== series / parallel ===")
    print(f"  100 + 100 series      = {series_resistance([100, 100]):.1f} ohm")
    print(f"  200 || 200 parallel   = {parallel_resistance([200, 200]):.1f} ohm")
    net = ("series", [100, ("parallel", [200, 200]), 50])
    print(f"  nested {net} = {equivalent_resistance(net):.1f} ohm")

    print("\n=== dividers ===")
    print(f"  12 V across 3k of (1k+3k) = {voltage_divider(12, 3000, 1000):.2f} V")
    print(f"  3 A through 100 || 200 (the 100) = "
          f"{current_divider(3.0, 100, 200):.3f} A")

    print("\n=== real battery (emf 12 V, r 0.5 ohm) ===")
    for RL in (100.0, 10.0, 1.0):
        I, Vt = terminal_voltage(12.0, 0.5, RL)
        print(f"  load {RL:6.1f} ohm -> I={I:.3f} A, terminal V={Vt:.3f} V, "
              f"P_load={power(v=Vt, i=I):.3f} W")

    print("\n=== Kirchhoff as linear algebra: G v = i ===")
    # 9V (via 1-ohm Norton) into node 1; ladder 1->2 (2 ohm), 2->gnd (3 ohm)
    I_n, r = norton_from_battery(9.0, 1.0)
    volts = nodal_solve(3, [(0, 1, r), (1, 2, 2.0), (2, 0, 3.0)],
                        current_sources={1: I_n})
    print(f"  node voltages: {np.round(volts, 3)}  (node 0 = ground)")
