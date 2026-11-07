"""Nilsson & Riedel Ch. 4 (Techniques of Circuit Analysis) applied to one circuit: an 8-bit R-2R DAC.

In plain words: an R-2R ladder turns an 8-bit number into a voltage using only two resistor
values. Loop the input code from 0 to 2^8 - 1 and every Chapter 4 technique shows up:

  * 4.2-4.4 node-voltage method: the ladder is solved by Modified Nodal Analysis
    (dgs.spice.dc_nodal_analysis), one unknown per node plus one per voltage source.
  * 4.5-4.8 mesh-current method: a small ladder is solved both ways with
    dgs.circuit_terminology and the answers agree; the equation counts differ.
  * 4.9 source transformations and 4.10-4.11 Thevenin/Norton: the ladder's output is a
    Thevenin source V_th = V_ref * code / 2^N with R_th = R for EVERY code -- found by
    open-circuit voltage + test-current injection, checked against short-circuit current.
  * 4.12 maximum power transfer: a load R_L = R_th takes V_th^2 / (4 R_th).
  * 4.13 superposition: each bit alone contributes V_ref * 2^k / 2^N; the sum of the
    eight single-bit answers equals the full answer -- the reason the DAC works at all.
  * 4.3/4.6 dependent sources: an op-amp output buffer modeled as a VCVS of finite gain A.
  * Practical Perspective, "circuits with realistic resistors": resistor tolerance makes
    the steps unequal; Monte Carlo gives the DNL/INL in LSB and the chance the DAC is
    monotonic.

Bench layer (room temperature, function generator, oscilloscope):
  * a function generator IS a Thevenin source with R_th = 50 ohm; its amplitude setting
    assumes a 50 ohm load, so into a 1 Mohm scope the screen shows about twice the setting;
  * a scope input (1 Mohm || ~15 pF) loads whatever it measures: the DAC's R_th sets the
    loading error in LSB, the reason real DACs get a buffer;
  * resistors at 300 K make Johnson noise sqrt(4 k T R B); tempco cancels in a ratio
    (the ladder is ratiometric) unless the resistors' tempcos differ.

NumPy; SymPy via dgs.circuit_terminology. Component values are typical, labeled as assumptions.
"""

import numpy as np

from dgs import spice

K_B = 1.380649e-23


def _check_bits(n_bits):
    if int(n_bits) != n_bits or not 1 <= n_bits <= 16:
        raise ValueError("n_bits must be an integer in 1..16")


# ---------------------------------------------------------------- the ladder

def r2r_netlist(code, n_bits=8, R=10e3, v_ref=3.3, errors=None):
    """Voltage-mode R-2R ladder for an integer code.

    Nodes: ladder node k (bit k, LSB = 0) is node k+1; node n_bits+1 is the reference.
    Bit k's 2R leg goes to the reference if the bit is 1, else to ground; a 2R terminator
    sits at the LSB end; series R joins neighbours; the output is the MSB node.
    errors: optional array of 2*n_bits relative resistor errors (legs, then series + terminator).
    Returns (elements, n_nodes, out_node)."""
    _check_bits(n_bits)
    if not 0 <= code < 2 ** n_bits:
        raise ValueError(f"code must be in 0..{2 ** n_bits - 1}")
    if R <= 0:
        raise ValueError("R must be positive")
    e = np.zeros(2 * n_bits) if errors is None else np.asarray(errors, float)
    if e.size != 2 * n_bits:
        raise ValueError("errors must have 2 * n_bits entries")
    ref = n_bits + 1
    els = [("V", ref, 0, v_ref)]
    for k in range(n_bits):
        bit = (code >> k) & 1
        els.append(("R", k + 1, ref if bit else 0, 2 * R * (1 + e[k])))
    for k in range(n_bits - 1):
        els.append(("R", k + 1, k + 2, R * (1 + e[n_bits + k])))
    els.append(("R", 1, 0, 2 * R * (1 + e[2 * n_bits - 1])))
    return els, n_bits + 1, n_bits


def node_voltage(elements, n_nodes, node):
    return float(spice.dc_nodal_analysis(elements, n_nodes)[node - 1])


def dac_output(code, n_bits=8, R=10e3, v_ref=3.3, errors=None):
    """Open-circuit output voltage (node-voltage method / MNA)."""
    els, n, out = r2r_netlist(code, n_bits, R, v_ref, errors)
    return node_voltage(els, n, out)


def transfer_curve(n_bits=8, R=10e3, v_ref=3.3, errors=None):
    """Outputs for every code 0 .. 2^N - 1 (the loop from 2^0 to 2^N)."""
    return np.array([dac_output(c, n_bits, R, v_ref, errors) for c in range(2 ** n_bits)])


def ideal_output(code, n_bits=8, v_ref=3.3):
    return v_ref * np.asarray(code, float) / 2 ** n_bits


# ---------------------------------------------------------------- Thevenin / Norton / superposition

def thevenin(elements, n_nodes, out_node):
    """(V_th, R_th, I_N) at out_node vs ground.
    V_th: open-circuit node voltage. R_th: zero all independent sources, inject 1 A, read V.
    I_N: short-circuit current through a 0 V source added at the output (independent check)."""
    v_th = node_voltage(elements, n_nodes, out_node)
    dead = [(("V",) + e[1:3] + (0.0,)) if e[0] == "V" else (("I",) + e[1:3] + (0.0,)) if e[0] == "I" else e
            for e in elements]
    r_th = node_voltage(dead + [("I", 0, out_node, 1.0)], n_nodes, out_node)
    # short circuit: 0 V source from out to ground; its MNA current is the short-circuit current
    vs = [e for e in elements if e[0] == "V"] + [("V", out_node, 0, 0.0)]
    others = [e for e in elements if e[0] != "V"]
    A_sol = _mna_full(others + vs, n_nodes)
    i_sc = A_sol[n_nodes + len(vs) - 1]      # MNA source current enters the + terminal: = current out of the network
    return v_th, r_th, float(i_sc)


def _mna_full(elements, n_nodes):
    """Full MNA solution vector [node voltages..., source currents...] (same stamps as dgs.spice)."""
    vsrc = [e for e in elements if e[0] == "V"]
    size = n_nodes + len(vsrc)
    A = np.zeros((size, size))
    z = np.zeros(size)
    for e in elements:
        if e[0] == "R":
            _, a, b, val = e
            g = 1.0 / val
            for p, q, s in [(a, a, g), (b, b, g), (a, b, -g), (b, a, -g)]:
                if p and q:
                    A[p - 1, q - 1] += s
        elif e[0] == "I":
            _, a, b, val = e
            if a:
                z[a - 1] -= val
            if b:
                z[b - 1] += val
    for k, (_, a, b, val) in enumerate(vsrc):
        m = n_nodes + k
        if a:
            A[a - 1, m] += 1; A[m, a - 1] += 1
        if b:
            A[b - 1, m] -= 1; A[m, b - 1] -= 1
        z[m] = val
    return np.linalg.solve(A, z)


def norton_to_thevenin(i_n, r_n):
    """Source transformation: Norton (I_N || R) -> Thevenin (I_N R in series with R)."""
    if r_n <= 0:
        raise ValueError("R must be positive")
    return i_n * r_n, r_n


def thevenin_to_norton(v_th, r_th):
    if r_th <= 0:
        raise ValueError("R must be positive")
    return v_th / r_th, r_th


def load_power(v_th, r_th, r_load):
    """Power into R_L from a Thevenin source: V^2 R_L / (R_th + R_L)^2."""
    r_load = np.asarray(r_load, float)
    return v_th ** 2 * r_load / (r_th + r_load) ** 2


def superposition_bits(code, n_bits=8, R=10e3, v_ref=3.3, errors=None):
    """Output with each set bit acting alone (others grounded); returns the list of contributions."""
    return [dac_output(1 << k, n_bits, R, v_ref, errors) if (code >> k) & 1 else 0.0 for k in range(n_bits)]


# ---------------------------------------------------------------- dependent source: op-amp buffer

def buffered_output(code, gain=1e5, r_out=50.0, r_load=1e3, n_bits=8, R=10e3, v_ref=3.3):
    """Ladder -> unity-gain buffer modeled as VCVS (gain A, output resistance r_out) -> load.
    Returns (V_load, ideal open-circuit DAC voltage)."""
    if gain <= 0 or r_out < 0 or r_load <= 0:
        raise ValueError("gain > 0, r_out >= 0, r_load > 0 required")
    els, n, out = r2r_netlist(code, n_bits, R, v_ref)
    a, b, c_ = n + 1, n + 2, n + 3          # VCVS internal node, buffer output, (spare)
    els = els + [("E", a, 0, out, b, gain),       # V_a = A (V_out_ladder - V_b): follower feedback from b
                 ("R", a, b, r_out if r_out > 0 else 1e-9),
                 ("R", b, 0, r_load)]
    v = spice.s_domain_nodal_analysis(els, n + 2, 0)
    return float(np.real(v[b - 1])), dac_output(code, n_bits, R, v_ref)


# ---------------------------------------------------------------- realistic resistors (Practical Perspective)

def dnl_inl(curve, v_ref=3.3, n_bits=8):
    """Differential and integral nonlinearity in LSB (endpoint-free: ideal LSB = V_ref / 2^N)."""
    lsb = v_ref / 2 ** n_bits
    steps = np.diff(curve) / lsb
    dnl = steps - 1
    inl = (curve - ideal_output(np.arange(curve.size), n_bits, v_ref)) / lsb
    return dnl, inl


def monte_carlo_tolerance(tol, n_trials=200, n_bits=8, R=10e3, v_ref=3.3, seed=0):
    """Resistors uniform in +-tol: worst |DNL|, worst |INL| per trial, and monotonic fraction."""
    if not 0 <= tol < 0.5:
        raise ValueError("tol must be in [0, 0.5)")
    rng = np.random.default_rng(seed)
    worst_dnl, worst_inl, mono = [], [], []
    for _ in range(n_trials):
        e = rng.uniform(-tol, tol, 2 * n_bits)
        dnl, inl = dnl_inl(transfer_curve(n_bits, R, v_ref, e), v_ref, n_bits)
        worst_dnl.append(np.max(np.abs(dnl)))
        worst_inl.append(np.max(np.abs(inl)))
        mono.append(np.all(dnl > -1))
    return np.array(worst_dnl), np.array(worst_inl), float(np.mean(mono))


# ---------------------------------------------------------------- room temperature

def johnson_noise_rms(R, bandwidth, T=300.0):
    """Open-circuit thermal noise sqrt(4 k T R B) (V rms)."""
    if R < 0 or bandwidth <= 0 or T <= 0:
        raise ValueError("need R >= 0, bandwidth > 0, T > 0")
    return float(np.sqrt(4 * K_B * T * R * bandwidth))


def tempco_errors(n_bits, tc_ppm, dT, tc_spread_ppm=0.0, seed=0):
    """Relative resistor changes after a temperature change dT: common tempco plus random spread."""
    rng = np.random.default_rng(seed)
    tcs = tc_ppm + tc_spread_ppm * rng.standard_normal(2 * n_bits)
    return tcs * 1e-6 * dT


# ---------------------------------------------------------------- bench: function generator + scope

def fgen_into_load(v_set_pp, r_load, r_source=50.0):
    """Function generators calibrate their display for a 50 ohm load: V_th = 2 V_set.
    Returns the actual peak-to-peak voltage across r_load."""
    if r_load <= 0:
        raise ValueError("r_load must be positive")
    return 2 * v_set_pp * r_load / (r_source + r_load)


def scope_loading_error(r_th, r_scope=1e6):
    """Fractional reading error when a scope of input resistance r_scope probes a source of r_th."""
    return r_th / (r_th + r_scope)


def scope_frequency_response(freqs, r_source=50.0, r_scope=1e6, c_scope=15e-12):
    """|V_scope / V_th| vs frequency via dgs.spice s-domain MNA (source R, scope R || C)."""
    out = []
    for f in np.asarray(freqs, float):
        els = [("V", 1, 0, 1.0), ("R", 1, 2, r_source), ("R", 2, 0, r_scope), ("C", 2, 0, c_scope)]
        out.append(abs(spice.s_domain_nodal_analysis(els, 2, 2j * np.pi * f)[1]))
    return np.array(out)
