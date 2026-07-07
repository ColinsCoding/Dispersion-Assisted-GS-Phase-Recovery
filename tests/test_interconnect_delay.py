"""Test dgs.interconnect_delay: the abstraction levels and design flow, the
L^2 wire-delay law (fitted exponent 2.0), Elmore quadratic scaling, repeater
insertion finding an interior optimum that beats a single driver, the gate-vs-
wire crossover, and the linear LC (group-velocity) wave limit. NumPy only."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import interconnect_delay as ic

# 1. abstraction hierarchy: 4 levels, transistor lowest -> processor highest
levels = ic.abstraction_levels()
assert len(levels) == 4
assert levels[0][0] == "transistor" and "processor" in levels[-1][0]

# 2. design flow: synthesis -> physical -> verification -> testing, and
#    synthesis emits a technology-DEPENDENT netlist
flow = ic.design_flow()
names = [s[0] for s in flow]
assert names == ["synthesis", "physical design", "verification", "testing"]
assert "technology-dependent" in flow[0][2]
assert "LOCATION" in flow[1][1] or "location" in flow[1][1].lower()

# 3. wire R and C are linear in length
R1, C1 = ic.wire_rc(100)
R2, C2 = ic.wire_rc(200)
assert np.isclose(R2, 2 * R1) and np.isclose(C2, 2 * C1)
assert np.isclose(R1, 0.1 * 100) and np.isclose(C1, 0.2e-15 * 100)

# 4. Elmore self-delay is quadratic: 2x length -> 4x delay (no driver/load)
e1 = ic.elmore_delay(100)
e2 = ic.elmore_delay(200)
assert np.isclose(e2 / e1, 4.0, rtol=1e-9)
# closed form: R_w * C_w / 2 = 0.5 * r * c * L^2
assert np.isclose(ic.elmore_delay(300), 0.5 * (0.1 * 300) * (0.2e-15 * 300))

# 5. distributed 50% delay is 0.38*R*C and also scales as L^2 (exponent -> 2.0)
R, C = ic.wire_rc(500)
assert np.isclose(ic.distributed_delay_50(500), 0.38 * R * C)
assert np.isclose(ic.delay_length_exponent(), 2.0, atol=1e-6)

# 6. repeater insertion: an interior optimum that beats the single driver
L = 4000.0
one_driver = ic.buffered_delay(L, 1)
k_opt, d_opt = ic.optimal_buffers(L)
assert k_opt > 1                       # long wire wants repeaters
assert d_opt < one_driver              # and they help
assert d_opt < ic.buffered_delay(L, 20)   # optimum is interior (not "more is better")
assert d_opt == min(ic.buffered_delay(L, k) for k in range(1, 30))
# a short wire wants few/no repeaters
k_short, _ = ic.optimal_buffers(100.0)
assert k_short <= k_opt

# 7. gate vs wire: a fixed global wire's delay is constant across nodes while
#    the gate shrinks, so interconnect dominates at advanced nodes
g130, w130, dom130 = ic.gate_vs_wire(130)
g7, w7, dom7 = ic.gate_vs_wire(7)
assert np.isclose(w130, w7)            # wire delay independent of node
assert g7 < g130                       # gate shrinks with the node
assert dom7 and dom130                 # wire dominates a 2 mm line at both

# 8. LC wave limit is LINEAR in length (group-velocity floor), unlike RC's L^2
d1, v1 = ic.lc_line_delay(1000)
d2, v2 = ic.lc_line_delay(2000)
assert np.isclose(d2 / d1, 2.0, rtol=1e-9)      # linear, not quadratic
assert np.isclose(v1, v2)                        # velocity independent of length
assert 0.5e8 < v1 < 3e8                          # a sane fraction of c

# 9. kwarg bounds
for bad in (lambda: ic.wire_rc(0),
            lambda: ic.wire_rc(100, r_per_um=0),
            lambda: ic.elmore_delay(100, C_load=-1),
            lambda: ic.buffered_delay(100, 0),
            lambda: ic.gate_vs_wire(0),
            lambda: ic.lc_line_delay(-5)):
    try:
        bad()
        assert False, "expected ValueError"
    except ValueError:
        pass

print("test_interconnect_delay: all checks passed")
