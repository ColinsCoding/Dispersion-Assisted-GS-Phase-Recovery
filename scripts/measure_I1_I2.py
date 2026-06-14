"""Measure I1 & I2 in a two-loop circuit; cross-check against hand mesh analysis."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
import spice_sym as sx

# Two-loop circuit (DC):
#   V1=10V at node1 -> R1(2 ohm) -> node2 -> R2(3 ohm) -> node3 = V2=5V
#   shared branch R3(4 ohm) from node2 to ground.
#   I1 = current through R1, I2 = current through R2.
netlist = """
V1 1 0 10
R1 1 2 2
R3 2 0 4
R2 2 3 3
V2 3 0 5
"""

node_v, src_i = sx.solve_circuit(netlist, n_nodes=3, s=0)
print("node voltages:", {k: sp.nsimplify(v) for k, v in node_v.items()})

I = sx.branch_currents(netlist, n_nodes=3, s=0)
I1, I2, I3 = I["R1"], I["R2"], I["R3"]
print(f"\nI1 (through R1) = {I1} = {float(I1):.4f} A")
print(f"I2 (through R2) = {I2} = {float(I2):.4f} A")
print(f"I3 (through R3) = {I3} = {float(I3):.4f} A")

# hand mesh / nodal check: KCL at node 2 gives V2 = 80/13
V2 = sp.Rational(80, 13)
I1_hand = (10 - V2) / 2
I2_hand = (V2 - 5) / 3
I3_hand = V2 / 4
print("\nhand check:  I1 =", I1_hand, " I2 =", I2_hand, " I3 =", I3_hand)
print("KCL at node 2 (I1 = I2 + I3):", sp.simplify(I1 - (I2 + I3)) == 0)
print("engine == hand?", sp.simplify(I1 - I1_hand) == 0, sp.simplify(I2 - I2_hand) == 0)

# fully symbolic version: keep R1,R2,R3,V1,V2 as symbols
sym_net = """
V1 1 0 V1
R1 1 2 R1
R3 2 0 R3
R2 2 3 R2
V2 3 0 V2
"""
Isym = sx.branch_currents(sym_net, n_nodes=3, s=0)
print("\nsymbolic I1 =", sp.simplify(Isym["R1"]))
print("symbolic I2 =", sp.simplify(Isym["R2"]))
