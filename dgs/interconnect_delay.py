"""Interconnect delay: why in modern chips the WIRE, not the gate, sets the clock.

Last level up from a single gate (dgs.mux_design_views) is the whole design flow,
and its hardest reality is physical: a signal's delay depends on WHERE its wire
goes, because a wire is a distributed RC line whose delay grows with the SQUARE of
its length.

  DESIGN ABSTRACTION (transistor -> gate -> RT -> processor) and the FLOW that
  lowers a design through them:
     synthesis   : RTL -> technology-INDEPENDENT netlist -> technology mapping to
                   a standard-cell library -> technology-DEPENDENT gate netlist.
     physical    : placement (where each cell sits) + routing (the wires) -- this
                   is where interconnect LENGTH, hence delay, is decided.
     verification: DRC/LVS + static timing analysis (does every path meet the
                   clock constraint?).
     testing     : manufacturing test / scan / ATPG on real silicon.

  THE WIRE DELAY. A wire of length L has resistance R = r*L and capacitance
  C = c*L (r, c per unit length). Its delay is set by the product:
     lumped 1-pole 50%:      0.69 * R * C           = 0.69 * r c * L^2
     distributed RC 50%:     ~0.38 * R * C          (Bakoglu; the real wire)
     Elmore (far end):       0.5 * R * C            (first moment of the response)
  All scale as L^2 -- double the wire, QUADRUPLE the delay. That is the whole
  reason placement matters and long global wires must be BUFFERED (breaking L
  into k pieces turns L^2 into L^2/k).

  WAVE vs DIFFUSION (the group-velocity tie-in). Short on-chip wires are RC:
  delay is diffusive, ~L^2, no meaningful velocity. Long/global and package lines
  become LC transmission lines where a signal travels as a WAVE at the group
  velocity v = 1/sqrt(l*c), so time-of-flight is LINEAR in L. Same
  dω/dk group-velocity idea from the dispersion work, now setting chip latency.

Everything is a simple, checkable formula. NumPy only; py-3.13.
"""

import numpy as np


# ----------------------------------------------------------------------
# Reference: the abstraction levels and the design flow
# ----------------------------------------------------------------------

def abstraction_levels():
    """The design hierarchy from lowest to highest, each with its native
    model. Higher levels hide detail; lower levels are where delay is real."""
    return [
        ("transistor", "MOSFETs, RC, analog behavior", "SPICE"),
        ("gate", "Boolean logic gates, static timing", "gate netlist"),
        ("RT (register-transfer)", "registers + combinational logic, clocked", "RTL / HDL"),
        ("processor / architecture", "datapath, ISA, pipeline", "behavioral / C"),
    ]


def design_flow():
    """The four flow stages that lower RTL to tested silicon, and the netlist
    each produces. Physical design is where interconnect length is fixed."""
    return [
        ("synthesis", "RTL -> logic optimization -> technology mapping",
         "technology-dependent gate netlist"),
        ("physical design", "placement + routing set cell and wire LOCATIONS",
         "placed & routed netlist (with parasitics)"),
        ("verification", "DRC / LVS + static timing analysis vs the clock",
         "signed-off netlist"),
        ("testing", "scan / ATPG manufacturing test on silicon", "test vectors"),
    ]


# ----------------------------------------------------------------------
# The wire: R, C, and the L^2 delay
# ----------------------------------------------------------------------

def wire_rc(length_um, r_per_um=0.1, c_fF_per_um=0.2):
    """Total resistance [ohm] and capacitance [F] of a wire: R = r*L,
    C = c*L. Defaults are representative global-interconnect values
    (0.1 ohm/um, 0.2 fF/um)."""
    if length_um <= 0:
        raise ValueError("length_um must be positive")
    if r_per_um <= 0 or c_fF_per_um <= 0:
        raise ValueError("r and c per length must be positive")
    R = r_per_um * length_um
    C = c_fF_per_um * 1e-15 * length_um
    return R, C


def elmore_delay(length_um, r_per_um=0.1, c_fF_per_um=0.2,
                 R_driver=0.0, C_load=0.0):
    """Elmore delay of a distributed RC wire driven by R_driver into a load
    C_load:
        t = R_driver*(C_wire + C_load) + R_wire*(C_wire/2 + C_load).
    The R_wire*C_wire/2 term is the distributed-wire self-delay, quadratic in
    length. Returns seconds."""
    if R_driver < 0 or C_load < 0:
        raise ValueError("R_driver and C_load must be non-negative")
    R_w, C_w = wire_rc(length_um, r_per_um, c_fF_per_um)
    return R_driver * (C_w + C_load) + R_w * (C_w / 2 + C_load)


def distributed_delay_50(length_um, r_per_um=0.1, c_fF_per_um=0.2):
    """Bakoglu 50% delay of an unbuffered distributed RC wire: ~0.38*R*C.
    This is the number a timing tool reports for the wire itself."""
    R, C = wire_rc(length_um, r_per_um, c_fF_per_um)
    return 0.38 * R * C


def delay_length_exponent(r_per_um=0.1, c_fF_per_um=0.2):
    """Empirically fit the exponent p in (wire delay) ~ L^p over a length
    sweep. Returns p, which comes out to 2.0 -- the quadratic wall that makes
    long wires so costly and forces buffering."""
    L = np.array([50, 100, 200, 400, 800, 1600.0])
    d = np.array([distributed_delay_50(x, r_per_um, c_fF_per_um) for x in L])
    p = np.polyfit(np.log(L), np.log(d), 1)[0]
    return float(p)


# ----------------------------------------------------------------------
# Buffer insertion: break L^2 into k*(L/k)^2 = L^2/k
# ----------------------------------------------------------------------

def buffered_delay(length_um, k_buffers, r_per_um=0.1, c_fF_per_um=0.2,
                   R_buf=150.0, C_buf=1e-15, t_buf=5e-12):
    """Delay of a long wire split into k equal segments, each driven by a
    buffer (output resistance R_buf, input cap C_buf, intrinsic delay t_buf).
    The wire's quadratic self-delay per segment shrinks as (L/k)^2, but each
    buffer adds fixed delay -- so there is an OPTIMUM k. Returns seconds."""
    if k_buffers < 1:
        raise ValueError("k_buffers must be >= 1")
    seg = length_um / k_buffers
    R_w, C_w = wire_rc(seg, r_per_um, c_fF_per_um)
    per_stage = t_buf + R_buf * (C_w + C_buf) + R_w * (C_w / 2 + C_buf)
    return k_buffers * per_stage


def optimal_buffers(length_um, r_per_um=0.1, c_fF_per_um=0.2,
                    R_buf=150.0, C_buf=1e-15, t_buf=5e-12, k_max=200):
    """Search the buffer count k that minimizes buffered_delay. Returns
    (k_opt, delay_opt). Above modest lengths k_opt > 1 and the buffered delay
    beats the unbuffered wire -- the standard repeater-insertion result."""
    ks = np.arange(1, k_max + 1)
    delays = np.array([buffered_delay(length_um, int(k), r_per_um, c_fF_per_um,
                                      R_buf, C_buf, t_buf) for k in ks])
    i = int(np.argmin(delays))
    return int(ks[i]), float(delays[i])


# ----------------------------------------------------------------------
# Gate vs wire, and the wave (LC) limit
# ----------------------------------------------------------------------

def gate_vs_wire(node_nm, wire_len_um=2000.0, r_per_um=0.1, c_fF_per_um=0.2):
    """Compare a gate's delay (shrinks with the technology node) against a
    fixed-length global wire's delay (does NOT shrink). Rough model: gate
    delay ~ 0.5 ps per nm of node. Returns (gate_delay, wire_delay,
    wire_dominates). As nodes shrink, interconnect overtakes the gate -- the
    reason wire delay, not gate delay, limits modern chips."""
    if node_nm <= 0:
        raise ValueError("node_nm must be positive")
    gate = 0.5e-12 * (node_nm / 65.0)     # gate delay scales down with the node
    wire = distributed_delay_50(wire_len_um, r_per_um, c_fF_per_um)
    return gate, wire, bool(wire > gate)


def lc_line_delay(length_um, l_pH_per_um=0.4, c_fF_per_um=0.2):
    """Time-of-flight of an LC transmission line: t = L*sqrt(l*c), the wave
    limit. Signals travel at the group velocity v = 1/sqrt(l*c), so this delay
    is LINEAR in length -- unlike the RC wire's L^2. Long package and global
    lines approach this floor. Returns (delay_s, velocity_m_per_s)."""
    if length_um <= 0:
        raise ValueError("length_um must be positive")
    l = l_pH_per_um * 1e-12 / 1e-6      # H per meter
    c = c_fF_per_um * 1e-15 / 1e-6      # F per meter
    v = 1.0 / np.sqrt(l * c)            # m/s
    return length_um * 1e-6 / v, v


if __name__ == "__main__":
    print("abstraction levels:")
    for name, what, model in abstraction_levels():
        print(f"  {name:26s} {what:42s} [{model}]")
    print("design flow:")
    for stage, what, netlist in design_flow():
        print(f"  {stage:16s} -> {netlist}")

    print("\nwire delay grows as L^2:")
    for L in (100, 200, 400, 800):
        print(f"  L={L:5d} um: distributed 50% delay = "
              f"{distributed_delay_50(L)*1e12:6.1f} ps")
    print(f"  fitted exponent p in delay ~ L^p: {delay_length_exponent():.3f}")

    L = 4000.0
    one_driver = buffered_delay(L, 1) * 1e12      # single driver, no repeaters
    kopt, dopt = optimal_buffers(L)
    print(f"\n{L:.0f} um wire: single driver {one_driver:.0f} ps -> "
          f"{kopt} repeaters give {dopt*1e12:.0f} ps "
          f"({(1-dopt*1e12/one_driver)*100:.0f}% faster)")

    print("\ngate vs a 2 mm global wire across nodes:")
    for node in (130, 65, 28, 7):
        g, w, dom = gate_vs_wire(node)
        print(f"  {node:3d} nm: gate {g*1e12:5.2f} ps, wire {w*1e12:5.1f} ps, "
              f"wire dominates? {dom}")
    d, v = lc_line_delay(4000.0)
    print(f"\nLC wave limit for 4 mm: {d*1e12:.1f} ps at v = {v/1e8:.2f}e8 m/s "
          f"(linear in L, the group-velocity floor)")
