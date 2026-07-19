"""Test AP DC circuits: series/parallel, dividers, batteries, nodal analysis."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import dc_circuits as dc

# 1. series adds, parallel adds conductances; bounds respected
assert dc.series_resistance([100, 100, 50]) == 250.0
assert abs(dc.parallel_resistance([200, 200]) - 100.0) < 1e-9
assert abs(dc.parallel_resistance([100, 100, 100]) - 100 / 3) < 1e-9
# equal N resistors in parallel -> R/N; in series -> N*R (the AP shortcuts)
assert abs(dc.parallel_resistance([60] * 3) - 20.0) < 1e-9
# parallel R_eq <= smallest <= largest <= series R_eq
Rs = [47, 220, 1000]
assert dc.parallel_resistance(Rs) < min(Rs) <= max(Rs) < dc.series_resistance(Rs)

# 2. nested network reduction
net = ("series", [100, ("parallel", [200, 200]), 50])
assert dc.equivalent_resistance(net) == 250.0
assert dc.equivalent_resistance(("parallel", [("series", [10, 10]), 20])) == 10.0
assert dc.equivalent_resistance(330) == 330.0

# 3. voltage divider: 12 V across 3k of (1k+3k) = 9 V; a divider never amplifies
assert abs(dc.voltage_divider(12, 3000, 1000) - 9.0) < 1e-9
assert dc.voltage_divider(5, 100, 100) == 2.5           # equal split
assert dc.voltage_divider(5, 100, 900) < 5              # always a fraction

# 4. current divider: current favors the low-R path (scales with the OTHER R)
assert abs(dc.current_divider(3.0, 100, 200) - 2.0) < 1e-9   # 100-ohm gets more
assert abs(dc.current_divider(3.0, 200, 100) - 1.0) < 1e-9
assert (dc.current_divider(3.0, 100, 200) + dc.current_divider(3.0, 200, 100)
        - 3.0) < 1e-9                                         # currents sum to i_in

# 5. real battery: terminal voltage sags under load; -> emf as R_load -> inf
I, Vt = dc.terminal_voltage(12.0, 0.5, 10.0)
assert abs(I - 12 / 10.5) < 1e-9 and abs(Vt - (12 - I * 0.5)) < 1e-9
assert Vt < 12.0
_, Vt_open = dc.terminal_voltage(12.0, 0.5, 1e9)
assert abs(Vt_open - 12.0) < 1e-3                            # open circuit -> emf
# heavier load -> more sag
_, Vt_heavy = dc.terminal_voltage(12.0, 0.5, 1.0)
assert Vt_heavy < Vt

# 6. power: three faces of Joule heating agree
assert dc.power(v=10, i=2) == 20.0
assert dc.power(i=2, r=5) == 20.0
assert dc.power(v=10, r=5) == 20.0
for bad in (lambda: dc.power(v=10),
            lambda: dc.power(v=1, r=0),
            lambda: dc.power()):
    try:
        bad(); raise AssertionError("expected ValueError")
    except ValueError:
        pass

# 7. conductance matrix is the weighted graph Laplacian: symmetric SPD
G = dc.conductance_matrix(3, [(0, 1, 1.0), (1, 2, 2.0), (2, 0, 3.0)])
assert np.allclose(G, G.T)
assert np.all(np.linalg.eigvalsh(G) > 0)                    # positive-definite
assert np.allclose(G, [[1.5, -0.5], [-0.5, 0.5 + 1 / 3]])

# 8. Ohm's law from nodal solve: 1 A into a 5-ohm resistor to ground -> 5 V
v = dc.nodal_solve(2, [(0, 1, 5.0)], {1: 1.0})
assert np.allclose(v, [0.0, 5.0])

# 9. the ladder: 9 V (Norton, r=1) into node1, 2-ohm to node2, 3-ohm to ground.
#    hand solution [0, 7.5, 4.5] -- verified by KCL at both nodes.
I_n, r = dc.norton_from_battery(9.0, 1.0)
assert (I_n, r) == (9.0, 1.0)
v = dc.nodal_solve(3, [(0, 1, r), (1, 2, 2.0), (2, 0, 3.0)], {1: I_n})
assert np.allclose(v, [0.0, 7.5, 4.5])
# KCL check: current in at node1 == current out
assert abs((9.0 - v[1] / 1.0 - (v[1] - v[2]) / 2.0)) < 1e-9
assert abs(((v[1] - v[2]) / 2.0 - v[2] / 3.0)) < 1e-9       # node2 balances

# 10. nodal reproduces the series voltage divider (cross-check vs formula)
#    12 V behind 1 kohm feeding 3 kohm to ground: V_3k = 12*3/(1+3) = 9 V
I_n, r = dc.norton_from_battery(12.0, 1000.0)
v = dc.nodal_solve(2, [(0, 1, r), (1, 0, 3000.0)], {1: I_n})
assert abs(v[1] - 9.0) < 1e-9

# 11. bounds
for bad in (lambda: dc.series_resistance([]),
            lambda: dc.parallel_resistance([100, -5]),
            lambda: dc.equivalent_resistance(("bogus", [1, 2])),
            lambda: dc.conductance_matrix(1, []),
            lambda: dc.nodal_solve(3, [(0, 1, 1), (1, 1, 2)], {1: 1}),
            lambda: dc.norton_from_battery(9, 0)):
    try:
        bad(); raise AssertionError("expected ValueError")
    except ValueError:
        pass

print(f"TEST PASS  (series 250, 200||200=100; divider 9 V; current divider 2/1 A; "
      f"battery sag 12->{Vt:.2f} V; P=20 W three ways; G=Laplacian SPD; "
      f"ladder nodes [0,7.5,4.5]; nodal==divider 9 V)")
