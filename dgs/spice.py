"""A minimal SPICE -- circuit simulation as linear algebra + differential equations.

This is the engine under Cadence/Synopsys, in miniature, and it shows why the
foundational math IS the circuit work -- you do not choose between them:

  * DC operating point  -> LINEAR ALGEBRA. Modified Nodal Analysis (MNA) stamps the
    circuit into a matrix and solves G v = i for every node voltage at once.
  * Transient response  -> DIFFERENTIAL EQUATIONS. A capacitor obeys i = C dv/dt and
    an inductor v = L di/dt, so an RLC loop is a 2nd-order ODE; integrate it in time.
  * Damping / resonance -> the CHARACTERISTIC POLYNOMIAL s^2 + (R/L)s + 1/(LC),
    the same root-location idea as dgs.pid stability.

A netlist is a list of tuples, node 0 = ground:
    ('R', n1, n2, ohms) | ('V', n1, n2, volts) | ('I', n1, n2, amps)

NumPy only. Education -- not a replacement for ngspice, a window into it.
"""

import numpy as np


# ── DC: Modified Nodal Analysis (the linear-algebra core) ───────────
def dc_nodal_analysis(elements, n_nodes):
    """Solve a resistive circuit with independent sources. Returns the node-voltage
    vector [v1..vN] (ground = node 0 = 0 V) by building and solving the MNA system.

    MNA augments the node-voltage unknowns with one current unknown per voltage
    source, so ideal voltage sources are handled exactly (no source resistance)."""
    vsrc = [e for e in elements if e[0] == "V"]
    M = len(vsrc)
    size = n_nodes + M
    A = np.zeros((size, size))
    z = np.zeros(size)
    idx = lambda n: n - 1                      # node n>=1 -> matrix row; ground (0) skipped

    for e in elements:
        if e[0] == "R":
            _, a, b, val = e
            g = 1.0 / val
            for p, q, s in [(a, a, g), (b, b, g), (a, b, -g), (b, a, -g)]:
                if p != 0 and q != 0:
                    A[idx(p), idx(q)] += s
        elif e[0] == "I":
            _, a, b, val = e                   # current out of a, into b
            if a != 0:
                z[idx(a)] -= val
            if b != 0:
                z[idx(b)] += val

    for k, (_, a, b, val) in enumerate(vsrc):
        m = n_nodes + k                        # the extra current unknown for this source
        if a != 0:
            A[idx(a), m] += 1; A[m, idx(a)] += 1
        if b != 0:
            A[idx(b), m] -= 1; A[m, idx(b)] -= 1
        z[m] = val                             # V(a) - V(b) = val

    x = np.linalg.solve(A, z)
    return x[:n_nodes]                          # node voltages (drop the source currents)


# ── transient: the RLC loop as a 2nd-order ODE (differential eqs) ────
def rlc_step_response(R, L, C, t, V=1.0, vc0=0.0, il0=0.0):
    """Series RLC driven by a DC step V. State (vc, iL) obeys
        L diL/dt = V - R iL - vc ,   C dvc/dt = iL
    Integrated with RK4. Returns (vc(t), iL(t)) -- the capacitor voltage and loop
    current. This is the ODE behind every ring, overshoot, and settling time."""
    t = np.asarray(t, float)
    dt = t[1] - t[0]
    vc = np.zeros_like(t); il = np.zeros_like(t)
    vc[0], il[0] = vc0, il0

    def deriv(state):
        v, i = state
        return np.array([i / C, (V - R * i - v) / L])   # [dvc/dt, diL/dt]

    s = np.array([vc0, il0])
    for n in range(1, len(t)):
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        vc[n], il[n] = s
    return vc, il


def rlc_damping(R, L, C):
    """Classify a series RLC from its characteristic roots s^2 + (R/L)s + 1/(LC) = 0.
    Returns (regime, roots): 'under' (complex roots -> rings), 'critical' (repeated),
    or 'over' (real roots -> no oscillation). Same root test as dgs.pid.is_stable."""
    roots = np.roots([1.0, R / L, 1.0 / (L * C)])
    damp_sq, nat_sq = (R / L) ** 2, 4.0 / (L * C)
    disc = damp_sq - nat_sq
    # compare the two terms RELATIVELY: at critical damping they are equal, but each
    # is ~ (R/L)^2 (huge), so an absolute tolerance would never trigger
    if abs(disc) < 1e-9 * nat_sq:
        regime = "critical"
    else:
        regime = "over" if disc > 0 else "under"
    return regime, roots


def resonant_frequency(L, C):
    """Undamped natural frequency f0 = 1/(2 pi sqrt(LC)) in Hz."""
    return 1.0 / (2 * np.pi * np.sqrt(L * C))


def critical_resistance(L, C):
    """The R that makes a series RLC critically damped: R = 2 sqrt(L/C)."""
    return 2.0 * np.sqrt(L / C)


if __name__ == "__main__":
    # DC: a voltage divider 10V across two equal 1k resistors -> 5V at the middle
    net = [("V", 1, 0, 10.0), ("R", 1, 2, 1000.0), ("R", 2, 0, 1000.0)]
    v = dc_nodal_analysis(net, n_nodes=2)
    print(f"divider node voltages: v1={v[0]:.3f} V, v2={v[1]:.3f} V  (expect 10, 5)")

    # transient: underdamped RLC rings at ~ its resonant frequency
    L, C = 1e-3, 1e-6
    f0 = resonant_frequency(L, C)
    for R in (10.0, critical_resistance(L, C), 500.0):
        regime, roots = rlc_damping(R, L, C)
        print(f"R={R:7.1f}  {regime:8s} damped  roots={np.round(roots,1)}")
    print(f"resonant f0 = {f0:.1f} Hz, R_crit = {critical_resistance(L,C):.1f} ohm")
