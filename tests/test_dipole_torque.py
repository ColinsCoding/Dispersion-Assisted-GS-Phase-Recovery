"""Test magnetic-dipole torque tau = m x B and energy U = -m.B (the motor torque)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
from griffiths import magnetostatics as ms

m0, B0, th = sp.symbols("m0 B0 theta", positive=True)

# dipole at angle theta to a field B along z; m in the y-z plane
m = [0, m0 * sp.sin(th), m0 * sp.cos(th)]
B = [0, 0, B0]

# 1. torque magnitude = m B |sin(theta)|  (reverses sign past theta = pi)
tau = ms.dipole_torque(m, B)
tau_mag = sp.simplify(sp.sqrt((tau.T * tau)[0]))
assert sp.simplify(tau_mag - m0 * B0 * sp.Abs(sp.sin(th))) == 0, tau_mag
# torque points along x (perpendicular to both m and B), driving rotation toward B
assert sp.simplify(tau[0] - m0 * B0 * sp.sin(th)) == 0
assert tau[1] == 0 and tau[2] == 0

# 2. aligned -> zero torque (equilibrium); perpendicular -> maximum
assert sp.simplify(ms.dipole_torque([0, 0, m0], B)) == sp.zeros(3, 1)      # m || B
assert sp.simplify(sp.sqrt((ms.dipole_torque([0, m0, 0], B).T *
                            ms.dipole_torque([0, m0, 0], B))[0]) - m0 * B0) == 0  # m perp B

# 3. energy U = -m.B = -m0 B0 cos(theta): minimum aligned, maximum anti-aligned
U = ms.dipole_energy(m, B)
assert sp.simplify(U - (-m0 * B0 * sp.cos(th))) == 0, U
assert sp.simplify(U.subs(th, 0) - (-m0 * B0)) == 0          # aligned -> lowest
assert sp.simplify(U.subs(th, sp.pi) - (m0 * B0)) == 0       # anti-aligned -> highest

# 4. the torque magnitude is the angular derivative of the energy: |tau| = |dU/dtheta|
#    (the restoring torque drives the dipole downhill in U toward alignment)
assert sp.simplify(sp.Abs(tau[0]) - sp.Abs(sp.diff(U, th))) == 0

# 5. motor torque: a current loop has m = N I A, so peak torque = N I A B
N, I, A = sp.symbols("N I A", positive=True)
loop_tau = ms.dipole_torque([N*I*A, 0, 0], [0, B0, 0])      # m perp B -> max
assert sp.simplify(sp.sqrt((loop_tau.T*loop_tau)[0]) - N*I*A*B0) == 0

print("TEST PASS  (tau = m x B, |tau| = m B sin(theta); zero aligned, max perp; "
      "U = -m.B = -m B cos(theta); tau = -dU/dtheta; motor peak torque = N I A B)")
