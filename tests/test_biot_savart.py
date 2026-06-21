"""Prove Biot-Savart by measurement: the numerical cross-product integral must
reproduce the textbook closed forms (infinite wire, loop on axis, loop center)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from griffiths import magnetostatics as ms

MU0 = ms.MU0_SI
I = 3.0

# 1. INFINITE STRAIGHT WIRE: |B| = mu0 I / (2 pi s), and B circles the wire.
#    Approximate "infinite" with a long wire along z; measure at (s,0,0).
s = 0.05
wire = ms.straight_wire_path(length=400.0, n=40001, axis=2)   # L >> s
B = ms.biot_savart(wire, I, [s, 0.0, 0.0])
B_expected = MU0 * I / (2 * np.pi * s)
assert abs(np.linalg.norm(B) - B_expected) / B_expected < 1e-3      # magnitude
# direction: current +z, position +x  ->  B along +y (right-hand rule)
assert abs(B[1]) > 0 and abs(B[0]) < 1e-9 * B_expected and abs(B[2]) < 1e-9 * B_expected
assert B[1] > 0

# 2. CIRCULAR LOOP ON AXIS: B_z = mu0 I R^2 / (2 (R^2+z^2)^{3/2}), along +z.
R = 0.1
loop = ms.circular_loop_path(R, n=4001)
for z in (0.0, 0.05, 0.2):
    B = ms.biot_savart(loop, I, [0.0, 0.0, z])
    Bz_expected = MU0 * I * R**2 / (2 * (R**2 + z**2) ** 1.5)
    assert abs(B[2] - Bz_expected) / Bz_expected < 2e-3, (z, B[2], Bz_expected)
    assert abs(B[0]) < 1e-6 * Bz_expected and abs(B[1]) < 1e-6 * Bz_expected   # on axis: pure z

# 3. LOOP CENTER (z=0) special case: B = mu0 I / (2 R)
Bc = ms.biot_savart(loop, I, [0.0, 0.0, 0.0])
assert abs(Bc[2] - MU0 * I / (2 * R)) / (MU0 * I / (2 * R)) < 2e-3

# 4. the numerical integral agrees with the module's SYMBOLIC closed forms
import sympy as sp
wire_sym = float(ms.wire_field(I, s).subs(ms.mu0, MU0))
assert abs(np.linalg.norm(ms.biot_savart(wire, I, [s, 0, 0])) - wire_sym) / wire_sym < 1e-3
axis_sym = float(ms.loop_field_axis(I, R, 0.05).subs(ms.mu0, MU0))
assert abs(ms.biot_savart(loop, I, [0, 0, 0.05])[2] - axis_sym) / axis_sym < 2e-3

print(f"TEST PASS  (numerical Biot-Savart integral == closed forms: infinite wire "
      f"mu0 I/2pi s within 0.1%, loop axis & center mu0 I/2R within 0.2%; "
      f"directions right-handed). Proved, not taken at face value.")
